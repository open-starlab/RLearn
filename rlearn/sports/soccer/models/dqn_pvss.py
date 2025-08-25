from __future__ import annotations
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DQNConfig:
    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 10.0

    gamma: float = 0.99
    target_update_freq: int = 1000
    tau: float = 1.0

    hidden_dim: int = 128

    use_double: bool = True
    use_cql: bool = True
    cql_alpha: float = 1.0

    loss: str = "huber"
    huber_delta: float = 1.0

    amp: bool = False
    seed: int = 42


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.fc3 = nn.Linear(hidden_dim, action_dim)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def compute_td_loss(
    q_values: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_obs: Tensor,
    dones: Tensor,
    q_net: nn.Module,
    target_net: nn.Module,
    gamma: float,
    use_double: bool,
    loss: str = "huber",
    huber_delta: float = 1.0,
) -> Tensor:
    device = q_values.device
    batch_idx = torch.arange(q_values.size(0), device=device)
    q_pred = q_values[batch_idx, actions]

    with torch.no_grad():
        if use_double:
            next_q_online = q_net(next_obs)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = target_net(next_obs)
            q_next = next_q_target[batch_idx, next_actions]
        else:
            q_next = target_net(next_obs).max(dim=1).values

    q_target = rewards + gamma * q_next * (1.0 - dones)

    if loss == "huber":
        return F.smooth_l1_loss(q_pred, q_target, beta=huber_delta)
    elif loss == "mse":
        return F.mse_loss(q_pred, q_target)
    else:
        raise ValueError(f"Unsupported loss: {loss}")


def compute_cql_loss(q_values: Tensor, actions: Tensor, alpha: float) -> Tensor:
    logsumexp_q = torch.logsumexp(q_values, dim=1)
    batch_idx = torch.arange(q_values.size(0), device=q_values.device)
    q_taken = q_values[batch_idx, actions]
    return alpha * (logsumexp_q - q_taken).mean()


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: Optional[Dict[str, Any] | DQNConfig] = None) -> None:
        if config is None:
            cfg = DQNConfig()
        elif isinstance(config, DQNConfig):
            cfg = config
        else:
            cfg = DQNConfig(**config)

        self.cfg = cfg
        self._set_seed(cfg.seed)

        self.gamma = cfg.gamma
        self.use_double = cfg.use_double
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQNNetwork(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net = DQNNetwork(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        # New-style torch.amp API
        self._scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp))
        self.step_count = 0

    # Convenience so training scripts can do: agent = DQNAgent(...).to(device)
    def to(self, device: str | torch.device) -> "DQNAgent":
        dev = torch.device(device)
        self.device = dev
        self.q_net.to(dev)
        self.target_net.to(dev)
        return self

    @torch.no_grad()
    def act(self, obs: Tensor) -> Tensor:
        self.q_net.eval()
        q = self.q_net(obs.to(self.device).float())
        return q.argmax(dim=1)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        obs = batch['obs'].to(self.device).float()
        actions = batch['actions'].to(self.device).long()
        rewards = batch['rewards'].to(self.device).float()
        next_obs = batch['next_obs'].to(self.device).float()
        dones = batch['dones'].to(self.device).float()

        with torch.amp.autocast("cuda", enabled=bool(cfg.amp)):
            q_values = self.q_net(obs)
            td_loss = compute_td_loss(
                q_values, actions, rewards, next_obs, dones,
                self.q_net, self.target_net, cfg.gamma, cfg.use_double,
                loss=cfg.loss, huber_delta=cfg.huber_delta
            )
            loss = td_loss
            logs = {"td_loss": float(td_loss.item())}

            if cfg.use_cql:
                cql_loss = compute_cql_loss(q_values, actions, cfg.cql_alpha)
                loss = loss + cql_loss
                logs["cql_loss"] = float(cql_loss.item())

        self.optimizer.zero_grad(set_to_none=True)
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self.optimizer)
        total_norm = clip_grad_norm_(self.q_net.parameters(), cfg.grad_clip)
        logs["grad_norm"] = float(total_norm.item() if torch.is_tensor(total_norm) else total_norm)
        self._scaler.step(self.optimizer)
        self._scaler.update()

        self.step_count += 1
        if cfg.tau < 1.0:
            self._polyak_update(self.target_net, self.q_net, cfg.tau)
        elif self.step_count % cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return logs

    def save(self, path: str) -> None:
        torch.save({
            "cfg": asdict(self.cfg),
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
        }, path)

    def load(self, path: str, map_location: Optional[str | torch.device] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = int(ckpt.get("step_count", 0))

    @staticmethod
    def _polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.lerp_(sp.data, tau)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
