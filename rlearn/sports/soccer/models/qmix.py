# coding: utf-8
"""
Offline QMIX implementation for the RLearn openSTARLab framework.
Integrates:
  - Double-Q learning (decoupled action selection/evaluation)
  - Conservative Q-Learning (CQL) regularization (per-agent, non-negative)
  - Periodic hard target synchronization
  - Global seeding for reproducibility
  - DataLoader for RoboCup2D tracking logs
"""
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader, Dataset  # type: ignore

from rlearn.sports.soccer.models.q_model_base import QModelBase


class AgentNetwork(nn.Module):
    """Per-agent DRQN network (shared weights)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        rnn_dim: int = 64,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(hidden_dim, rnn_dim, batch_first=True)
        self.q_out = nn.Linear(rnn_dim, action_dim)

    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(1, batch_size, self.rnn.hidden_size)

    def forward(
        self,
        obs: Tensor,  # (batch*agents, time, obs_dim)
        hidden: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        x = self.fc(obs)
        out, h = self.rnn(x, hidden)
        q = self.q_out(out)
        return q, h


class MixingNetwork(nn.Module):
    """State-conditioned mixer with nonnegative hypernetworks."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hyper_hidden: int = 64,
        mixing_hidden: int = 32,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden = mixing_hidden

        # Hypernets produce layer weights/biases conditioned on state
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def forward(self, agent_qs: Tensor, states: Tensor) -> Tensor:
        """
        agent_qs: (B*, n_agents)
        states:   (B*, state_dim)
        returns:  (B*,) scalar Q_tot per (state, agent_qs) pair
        """
        bs = states.size(0)

        # First layer: (B*, 1, n_agents) @ (B*, n_agents, mixing_hidden) -> (B*, 1, mixing_hidden)
        w1 = torch.abs(self.hyper_w1(states)).view(bs, agent_qs.size(1), self.mixing_hidden)
        b1 = self.hyper_b1(states).view(bs, 1, self.mixing_hidden)
        h1 = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        h1 = F.relu(h1)  # (B*, 1, mixing_hidden)

        # Second layer: (B*, 1, mixing_hidden) @ (B*, mixing_hidden, 1) -> (B*, 1, 1)
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.mixing_hidden, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        q_tot = torch.bmm(h1, w2) + b2  # (B*, 1, 1)

        return q_tot.view(bs)


@QModelBase.register("qmix_offline")
class QMIXOfflineModel(QModelBase):
    """
    Multi-agent offline RL with QMIX factorization + CQL.
    Pure PyTorch (no Lightning Trainer calls).
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        cql_alpha: float = 1.0,
        target_update_interval: int = 100,
        seed: int = 42,
        agent_hidden_dim: int = 128,
        agent_rnn_dim: int = 64,
        hyper_hidden_dim: int = 64,
        mixing_hidden_dim: int = 32,
        grad_norm_clip: float = 10.0,
        **kwargs: Any,
    ):
        super().__init__()
        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.target_update_interval = target_update_interval
        self.grad_norm_clip = grad_norm_clip
        self.step_count = 0
        self.n_agents = n_agents
        self.action_dim = action_dim

        # Networks
        self.agent = AgentNetwork(obs_dim, action_dim, agent_hidden_dim, agent_rnn_dim)
        self.mixer = MixingNetwork(n_agents, state_dim, hyper_hidden_dim, mixing_hidden_dim)
        self.target_agent = AgentNetwork(obs_dim, action_dim, agent_hidden_dim, agent_rnn_dim)
        self.target_mixer = MixingNetwork(n_agents, state_dim, hyper_hidden_dim, mixing_hidden_dim)
        self._hard_update()

    def _hard_update(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Compute TD loss + CQL penalty for a batch (pure PyTorch).
        Expects keys:
          obs [B,T,N,O], actions [B,T,N], states [B,T,S],
          next_obs [B,T,N,O], next_states [B,T,S],
          rewards [B,T] or [B,T,1], dones [B,T] or [B,T,1]
        """
        B, T, N, O = batch['obs'].shape
        device = batch['obs'].device

        # ---- Current Q-values ----
        obs_flat = batch['obs'].reshape(B * N, T, O)  # (B*N, T, O)
        hid = self.agent.init_hidden(B * N).to(device)
        q_flat, _ = self.agent(obs_flat, hid)  # (B*N, T, A)
        q = q_flat.reshape(B, N, T, self.action_dim).permute(0, 2, 1, 3)  # [B, T, N, A]
        q_taken = q.gather(-1, batch['actions'].unsqueeze(-1)).squeeze(-1)  # [B, T, N]

        flat_q = q_taken.reshape(B * T, N)  # (B*T, N)
        states = batch['states'].reshape(B * T, -1)  # (B*T, S)
        joint_q = self.mixer(flat_q, states).reshape(B, T)  # (B, T)

        # ---- Target Q-values (Double-Q) ----
        next_obs_flat = batch['next_obs'].reshape(B * N, T, O)  # (B*N, T, O)
        hid_main = self.agent.init_hidden(B * N).to(device)
        q_main_next, _ = self.agent(next_obs_flat, hid_main)  # (B*N, T, A)
        q_main_next = q_main_next.reshape(B, N, T, self.action_dim).permute(0, 2, 1, 3)
        greedy_actions = q_main_next.argmax(-1)  # (B, T, N)

        hid_tgt = self.target_agent.init_hidden(B * N).to(device)
        q_tgt_flat, _ = self.target_agent(next_obs_flat, hid_tgt)
        q_tgt_next = q_tgt_flat.reshape(B, N, T, self.action_dim).permute(0, 2, 1, 3)
        q_next_taken = q_tgt_next.gather(-1, greedy_actions.unsqueeze(-1)).squeeze(-1)  # (B,T,N)

        flat_next_q = q_next_taken.reshape(B * T, N)  # (B*T, N)
        next_states = batch['next_states'].reshape(B * T, -1)  # (B*T, S)
        joint_next_q = self.target_mixer(flat_next_q, next_states).reshape(B, T)  # (B, T)

        rewards = batch['rewards']
        if rewards.dim() == 3:
            rewards = rewards.squeeze(-1)
        dones = batch['dones']
        if dones.dim() == 3:
            dones = dones.squeeze(-1)

        target_q = rewards + self.gamma * (1 - dones) * joint_next_q

        # ---- TD Loss ----
        td_loss = F.mse_loss(joint_q, target_q.detach())

        # ---- CQL per-agent (non-negative by construction) ----
        # Penalize large Q-values on unseen actions relative to dataset actions.
        # q: [B, T, N, A], actions: [B, T, N]
        logsumexp_all = torch.logsumexp(q, dim=-1)  # [B, T, N]
        data_q_agents = q.gather(-1, batch['actions'].unsqueeze(-1)).squeeze(-1)  # [B, T, N]
        cql_per_agent = logsumexp_all - data_q_agents  # >= 0 elementwise
        cql_loss = self.cql_alpha * cql_per_agent.mean()

        loss = td_loss + cql_loss

        # ---- Target update counter ----
        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self._hard_update()

        return loss

    def get_actions(
        self,
        obs: Tensor,  # [B, N, O] (single timestep)
        hidden: Optional[Tensor] = None,
        epsilon: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Greedy (or epsilon-greedy) per-agent action selection for a single timestep.
        """
        B, N, O = obs.shape
        device = obs.device
        if hidden is None:
            hidden = self.agent.init_hidden(B * N).to(device)

        obs_seq = obs.reshape(B * N, 1, O)  # add time dim
        with torch.no_grad():
            qv, new_h = self.agent(obs_seq, hidden)  # [B*N, 1, A]
            qv = qv.squeeze(1).reshape(B, N, -1)  # [B, N, A]
            greedy = qv.argmax(-1)  # [B, N]
            rnd = torch.randint_like(greedy, high=qv.size(-1))
            mask = torch.rand_like(greedy, dtype=torch.float32) < epsilon
            actions = torch.where(mask, rnd, greedy)
        return actions, new_h


class TrackingRLDataset(Dataset):
    """
    Dataset for RoboCup2D offline RL:
    parses tracking logs into obs, next_obs, states, next_states, actions, rewards, dones.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        n_agents: int,
    ):
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.paths = list(Path(data_dir).glob('*.tracking.csv'))
        self.windows: List[np.ndarray] = []

        for p in self.paths:
            df = pd.read_csv(p)
            # ball + left agents l1..l{n_agents}
            ball = ['b_x', 'b_y', 'b_vx', 'b_vy']
            agent_cols = []
            for i in range(1, n_agents + 1):
                agent_cols += [f'l{i}_x', f'l{i}_y', f'l{i}_vx', f'l{i}_vy', f'l{i}_stamina']
            cols = ball + agent_cols
            if not set(cols).issubset(df.columns):
                # Skip files missing required columns
                continue
            arr = df[cols].values.astype(np.float32)
            # sliding windows of length seq_len+1
            for i in range(max(0, len(arr) - seq_len)):
                self.windows.append(arr[i : i + seq_len + 1])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        data = self.windows[idx]  # (L+1, F)
        seq = torch.from_numpy(data)
        # Global state is first 4 dims (ball)
        states = seq[:-1, :4]
        next_states = seq[1:, :4]
        # Agent obs dims: remaining dims reshape to (L, n_agents, obs_dim)
        agent_feat = seq[:, 4:].reshape(self.seq_len + 1, self.n_agents, -1)
        obs = agent_feat[:-1]
        next_obs = agent_feat[1:]
        # Placeholder zeros for actions, rewards, dones (replace with real labels when available)
        actions = torch.zeros(self.seq_len, self.n_agents, dtype=torch.long)
        rewards = torch.zeros(self.seq_len, dtype=torch.float32)
        dones = torch.zeros(self.seq_len, dtype=torch.float32)
        return {
            'obs': obs,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'next_states': next_states,
            'dones': dones,
        }


# Utility to create DataLoader
def make_qmix_dataloader(
    data_dir: str,
    seq_len: int,
    n_agents: int,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    ds = TrackingRLDataset(data_dir, seq_len, n_agents)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
