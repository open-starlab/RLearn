# coding: utf-8
"""
Offline QMIX implementation for the RLearn openSTARLab framework.
Integrates:
  - Double-Q learning (decoupled action selection/evaluation)
  - Conservative Q-Learning (CQL) regularization (per-agent, non-negative)
  - Periodic hard target synchronization
  - Global seeding for reproducibility
  - DataLoader for RoboCup2D tracking logs
  - Team-based data splitting for evaluation
  - Training/evaluation loops with CLI interface
  - Action shaping and reward shaping capabilities
  - RNN-aware training with burn-in and n-step returns
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

try:
    from rlearn.sports.soccer.models.q_model_base import QModelBase
except Exception:
    class QModelBase:
        @staticmethod
        def register(_name: str):
            def deco(cls): return cls
            return deco


# ----------------------------
# Dataset and Shaping Configuration
# ----------------------------
REQ_BALL = ["b_x", "b_y", "b_vx", "b_vy"]

def agent_cols(n_agents: int) -> List[str]:
    cols: List[str] = []
    left = min(11, n_agents)
    for i in range(1, left + 1):
        cols += [f"l{i}_x", f"l{i}_y", f"l{i}_vx", f"l{i}_vy", f"l{i}_stamina"]
    right = max(0, n_agents - left)
    right = min(11, right)
    for i in range(1, right + 1):
        cols += [f"r{i}_x", f"r{i}_y", f"r{i}_vx", f"r{i}_vy", f"r{i}_stamina"]
    return cols

class ShapeCfg:
    """Configuration for action and reward shaping."""
    def __init__(
        self,
        speed_still: float = 0.05,
        pass_ball_speed: float = 1.5,
        pass_sep: float = 0.8,
    ):
        self.speed_still = speed_still
        self.pass_ball_speed = pass_ball_speed
        self.pass_sep = pass_sep

class WindowsDatasetShaped(Dataset):
    """Dataset with action and reward shaping capabilities."""
    def __init__(self, files: List[Path], seq_len: int, n_agents: int, shape_cfg: ShapeCfg):
        self.seq_len = seq_len
        self.n_agents = n_agents
        self.shape_cfg = shape_cfg
        self.windows: List[Dict[str, np.ndarray]] = []
        
        needed = REQ_BALL + agent_cols(n_agents)
        for p in files:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
                
            if not set(needed).issubset(df.columns):
                print(f"[WARN] skip {p.name}: missing required columns")
                continue
                
            arr = df[needed].astype("float32").values
            if len(arr) <= seq_len:
                continue
                
            for i in range(len(arr) - seq_len):
                window_data = arr[i:i + seq_len + 1]
                shaped_actions, shaped_rewards = self._shape_window(window_data)
                
                self.windows.append({
                    'data': window_data,
                    'actions': shaped_actions,
                    'rewards': shaped_rewards
                })

    def _shape_window(self, window_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply action and reward shaping to a window of data."""
        # Placeholder implementation - replace with your actual shaping logic
        seq_len = window_data.shape[0] - 1
        actions = np.zeros((seq_len, self.n_agents), dtype=np.int64)
        rewards = np.zeros(seq_len, dtype=np.float32)
        
        # Simple shaping logic based on ball movement
        for t in range(seq_len):
            # Ball velocity magnitude
            ball_vx, ball_vy = window_data[t, 2], window_data[t, 3]
            ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
            
            # Reward based on ball speed (simplified)
            rewards[t] = ball_speed * 0.1
            
            # Simple action assignment based on proximity to ball
            ball_x, ball_y = window_data[t, 0], window_data[t, 1]
            for agent_idx in range(self.n_agents):
                agent_offset = 4 + agent_idx * 5
                agent_x, agent_y = window_data[t, agent_offset], window_data[t, agent_offset + 1]
                dist = np.sqrt((agent_x - ball_x)**2 + (agent_y - ball_y)**2)
                
                # Simple action: move toward ball if far away
                if dist > 5.0:
                    actions[t, agent_idx] = 1  # Move toward ball
                else:
                    actions[t, agent_idx] = 0  # Stay still
        
        return actions, rewards

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]
        seq = torch.from_numpy(window['data'])
        
        states = seq[:-1, :4]
        next_states = seq[1:, :4]
        feat = seq[:, 4:]
        
        obs = feat[:-1].reshape(self.seq_len, self.n_agents, -1)
        next_obs = feat[1:].reshape(self.seq_len, self.n_agents, -1)
        
        actions = torch.from_numpy(window['actions'])
        rewards = torch.from_numpy(window['rewards'])
        dones = torch.zeros(self.seq_len, dtype=torch.float32)
        
        return {
            "obs": obs, 
            "next_obs": next_obs, 
            "states": states,
            "next_states": next_states, 
            "actions": actions,
            "rewards": rewards, 
            "dones": dones
        }


# ----------------------------
# Split helpers (team hold-out)
# ----------------------------
def _ratio_from_tag(tag: str) -> float:
    return {"1_9": 0.10, "2_8": 0.20, "3_7": 0.30}[tag]

def _list_csvs(data_dir: str) -> List[Path]:
    return sorted(Path(data_dir).glob("*.csv"))

def _teams_from_head(p: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        df = pd.read_csv(p, nrows=1)
    except Exception:
        return None, None
    l = str(df.iloc[0]["l_name"]) if "l_name" in df.columns else None
    r = str(df.iloc[0]["r_name"]) if "r_name" in df.columns else None
    l = l.upper() if l and l != "None" else None
    r = r.upper() if r and r != "None" else None
    return l, r

def _build_team_index(files: List[Path]) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in files:
        l, r = _teams_from_head(p)
        for t in (l, r):
            if t:
                idx.setdefault(t, []).append(p)
    return idx

def _choose_heldout_teams(team_idx: Dict[str, List[Path]], tag: str, seed: int) -> List[str]:
    teams = sorted(team_idx.keys())
    if not teams:
        return []
    test_ratio = _ratio_from_tag(tag)
    n_hold = max(1, int(round(len(teams) * test_ratio)))
    rng = random.Random(seed)
    rng.shuffle(teams)
    return sorted(teams[:n_hold])

def split_by_team(files: List[Path], split_tag: str, seed: int) -> Tuple[List[Path], List[Path], List[str]]:
    t_idx = _build_team_index(files)
    heldout = _choose_heldout_teams(t_idx, split_tag, seed)
    test = set()
    for t in heldout:
        for p in t_idx.get(t, []):
            test.add(p)
    test_files = sorted(test)
    train_files = sorted([p for p in files if p not in test])
    return train_files, test_files, heldout


# ----------------------------
# Networks
# ----------------------------
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
        obs: Tensor,
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
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w1(states)).view(bs, agent_qs.size(1), self.mixing_hidden)
        b1 = self.hyper_b1(states).view(bs, 1, self.mixing_hidden)
        h1 = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        h1 = F.relu(h1)

        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.mixing_hidden, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        q_tot = torch.bmm(h1, w2) + b2
        return q_tot.view(bs)


@QModelBase.register("qmix_offline")
class QMIXOfflineModel(nn.Module):
    """
    Multi-agent offline RL with QMIX factorization + CQL.
    RNN training with burn-in and n-step TD targets.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        *,
        gamma: float = 0.99,
        lr: float = 2e-4,
        cql_alpha: float = 0.1,
        target_update_interval: int = 200,
        seed: int = 42,
        agent_hidden_dim: int = 128,
        agent_rnn_dim: int = 64,
        hyper_hidden_dim: int = 64,
        mixing_hidden_dim: int = 32,
        grad_norm_clip: float = 5.0,
        td_loss: str = "huber",
        huber_delta: float = 1.0,
        burn_in: int = 5,
        n_step: int = 1,
        q_clip: float = 50.0,
        lambda_as: float = 0.0,
        lambda_l1: float = 0.0,
    ):
        super().__init__()
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
        self.td_loss = td_loss
        self.huber_delta = huber_delta
        self.burn_in = max(0, int(burn_in))
        self.n_step = max(1, int(n_step))
        self.q_clip = float(q_clip)
        self.lambda_as = float(lambda_as)
        self.lambda_l1 = float(lambda_l1)

        self.agent = AgentNetwork(obs_dim, action_dim, agent_hidden_dim, agent_rnn_dim)
        self.mixer = MixingNetwork(n_agents, state_dim, hyper_hidden_dim, mixing_hidden_dim)
        self.target_agent = AgentNetwork(obs_dim, action_dim, agent_hidden_dim, agent_rnn_dim)
        self.target_mixer = MixingNetwork(n_agents, state_dim, hyper_hidden_dim, mixing_hidden_dim)
        self._hard_update()

        self._opt = AdamW(self.parameters(), lr=lr, weight_decay=0.0)

    def _hard_update(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _td(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.td_loss == "huber":
            return F.smooth_l1_loss(pred, target, beta=self.huber_delta)
        elif self.td_loss == "mse":
            return F.mse_loss(pred, target)
        raise ValueError(f"Unsupported td_loss={self.td_loss}")

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        for k in ("obs", "next_obs", "states", "next_states", "rewards", "dones", "actions"):
            batch[k] = torch.nan_to_num(batch[k], nan=0.0, posinf=0.0, neginf=0.0)

        B, T, N, O = batch['obs'].shape
        device = batch['obs'].device
        burn = min(self.burn_in, T - 1)
        nstep = self.n_step
        A = self.action_dim

        obs_flat = batch['obs'].reshape(B * N, T, O)
        hid = self.agent.init_hidden(B * N).to(device)
        q_flat, _ = self.agent(obs_flat, hid)
        q = q_flat.view(B, N, T, A).permute(0, 2, 1, 3).contiguous()

        if self.q_clip > 0:
            q = q.clamp(-self.q_clip, self.q_clip)

        as_loss = torch.zeros((), device=device)
        if self.lambda_as > 0.0 and 'actions' in batch:
            acts = batch['actions'].long().clamp_min(0).clamp_max(A - 1)
            ce_input = q.view(B * T * N, A)
            ce_target = acts.view(B * T * N)
            as_loss = F.cross_entropy(ce_input, ce_target)

        acts = batch['actions'].long().clamp_(0, A - 1)
        q_taken = q.gather(-1, acts.unsqueeze(-1)).squeeze(-1)

        states = batch['states'].reshape(B * T, -1)
        joint_q = self.mixer(q_taken.reshape(B * T, N), states).view(B, T)

        next_obs_flat = batch['next_obs'].reshape(B * N, T, O)
        hid_sel = self.agent.init_hidden(B * N).to(device)
        q_main_next, _ = self.agent(next_obs_flat, hid_sel)
        q_main_next = q_main_next.view(B, N, T, A).permute(0, 2, 1, 3).contiguous()
        if self.q_clip > 0:
            q_main_next = q_main_next.clamp(-self.q_clip, self.q_clip)
        greedy_next = q_main_next.argmax(-1)

        hid_tgt = self.target_agent.init_hidden(B * N).to(device)
        q_tgt_flat, _ = self.target_agent(next_obs_flat, hid_tgt)
        q_tgt_next = q_tgt_flat.view(B, N, T, A).permute(0, 2, 1, 3).contiguous()
        if self.q_clip > 0:
            q_tgt_next = q_tgt_next.clamp(-self.q_clip, self.q_clip)
        q_next_taken = q_tgt_next.gather(-1, greedy_next.unsqueeze(-1)).squeeze(-1)

        joint_next = self.target_mixer(q_next_taken.reshape(B * T, N),
                                       batch['next_states'].reshape(B * T, -1)).view(B, T)

        rewards = batch['rewards']
        if rewards.dim() == 3:
            rewards = rewards.squeeze(-1)
        dones = batch['dones']
        if dones.dim() == 3:
            dones = dones.squeeze(-1)
        not_done = (1.0 - dones).clamp(0, 1)

        target = torch.zeros_like(joint_q)
        for t in range(T):
            t_end = min(T - 1, t + nstep - 1)
            g = torch.zeros(B, device=device)
            gamma = 1.0
            alive = torch.ones(B, device=device)
            for k in range(t, t_end + 1):
                g = g + gamma * rewards[:, k]
                alive = alive * not_done[:, k]
                gamma *= self.gamma
            bootstrap = torch.zeros(B, device=device)
            if t + nstep < T:
                bootstrap = gamma * alive * joint_next[:, t + nstep]
            target[:, t] = g + bootstrap

        mask = torch.ones_like(target, dtype=torch.bool)
        if burn > 0:
            mask[:, :burn] = False

        td = self._td(joint_q[mask], target.detach()[mask])

        logsumexp_all = torch.logsumexp(q, dim=-1)
        data_q_agents = q_taken
        cql = self.cql_alpha * (logsumexp_all - data_q_agents).clamp_min(0).mean()

        l1 = torch.zeros((), device=device)
        if self.lambda_l1 > 0.0:
            l1 = sum(p.abs().sum() for p in self.parameters()) * self.lambda_l1

        loss = td + cql + self.lambda_as * as_loss + l1

        if not torch.isfinite(loss):
            return torch.zeros((), device=device, requires_grad=True)

        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self._hard_update()

        return loss

    @torch.no_grad()
    def evaluate_batch(self, batch: Dict[str, Tensor]) -> float:
        loss = self.compute_loss(batch)
        val = float(loss.detach().cpu()) if torch.isfinite(loss) else float('inf')
        return val

    def get_actions(
        self,
        obs: Tensor,
        hidden: Optional[Tensor] = None,
        epsilon: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        B, N, O = obs.shape
        device = obs.device
        if hidden is None:
            hidden = self.agent.init_hidden(B * N).to(device)

        obs_seq = obs.reshape(B * N, 1, O)
        with torch.no_grad():
            qv, new_h = self.agent(obs_seq, hidden)
            qv = qv.squeeze(1).reshape(B, N, -1)
            greedy = qv.argmax(-1)
            rnd = torch.randint_like(greedy, high=qv.size(-1))
            mask = torch.rand_like(greedy, dtype=torch.float32) < epsilon
            actions = torch.where(mask, rnd, greedy)
        return actions, new_h


# ----------------------------
# Train / Eval helpers
# ----------------------------
def train_epoch(model, loader, device, *, amp=True, scaler=None, grad_clip=5.0):
    model.train()
    total, steps = 0.0, 0
    for b in loader:
        for k in b:
            b[k] = b[k].to(device, non_blocking=True)
            b[k] = torch.nan_to_num(b[k], nan=0.0, posinf=0.0, neginf=0.0)

        model._opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=bool(amp)):
            loss = model.compute_loss(b)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(model._opt)
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(model._opt)
        scaler.update()

        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)

@torch.no_grad()
def eval_epoch(model, loader, device, *, amp=True):
    model.eval()
    total, steps = 0.0, 0
    for b in loader:
        for k in b:
            b[k] = b[k].to(device, non_blocking=True)
            b[k] = torch.nan_to_num(b[k], nan=0.0, posinf=0.0, neginf=0.0)
        with torch.amp.autocast("cuda", enabled=bool(amp)):
            loss = model.compute_loss(b)
        if not torch.isfinite(loss):
            continue
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


# ----------------------------
# CLI / Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./dataset")
    p.add_argument("--save_root", type=str, default="./results/qmix_as")
    p.add_argument("--split_tag", choices=["1_9", "2_8", "3_7"], default="1_9")

    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--n_agents", type=int, default=22)
    p.add_argument("--obs_dim", type=int, default=5)
    p.add_argument("--state_dim", type=int, default=4)
    p.add_argument("--action_dim", type=int, default=10)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--cql_alpha", type=float, default=0.1)
    p.add_argument("--td_loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--burn_in", type=int, default=5)
    p.add_argument("--n_step", type=int, default=1)
    p.add_argument("--q_clip", type=float, default=50.0)
    p.add_argument("--lambda_as", type=float, default=0.0)
    p.add_argument("--lambda_l1", type=float, default=0.0)

    p.add_argument("--pass_ball_speed", type=float, default=1.5)
    p.add_argument("--pass_sep", type=float, default=0.8)
    p.add_argument("--speed_still", type=float, default=0.05)

    return p.parse_args()

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    files = _list_csvs(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files in {args.data_dir}")

    train_files, test_files, heldout = split_by_team(files, args.split_tag, args.seed)
    print(f"[Split] by_team {args.split_tag}: held-out teams = {heldout}")
    print(f"[Split] train files = {len(train_files)} | test files = {len(test_files)}")

    shape_cfg = ShapeCfg(
        speed_still=args.speed_still,
        pass_ball_speed=args.pass_ball_speed,
        pass_sep=args.pass_sep,
    )
    train_ds = WindowsDatasetShaped(train_files, args.seq_len, args.n_agents, shape_cfg)
    test_ds = WindowsDatasetShaped(test_files, args.seq_len, args.n_agents, shape_cfg)
    print(f"[Windows] train={len(train_ds)} | test={len(test_ds)}")
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Empty train or test windows after team split. Try a different seed or tag.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = QMIXOfflineModel(
        obs_dim=args.obs_dim, state_dim=args.state_dim, action_dim=args.action_dim,
        n_agents=args.n_agents, lr=args.lr, cql_alpha=args.cql_alpha,
        td_loss=args.td_loss, huber_delta=args.huber_delta,
        burn_in=args.burn_in, n_step=args.n_step, q_clip=args.q_clip,
        lambda_as=args.lambda_as, lambda_l1=args.lambda_l1,
    ).to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))
    sched = ReduceLROnPlateau(model._opt, mode="min", patience=args.patience, factor=0.5)

    hist = {"train_loss": [], "test_loss": []}
    for ep in range(args.epochs):
        tr = train_epoch(model, train_loader, device, amp=args.amp, scaler=scaler, grad_clip=args.grad_clip)
        te = eval_epoch(model, test_loader, device, amp=args.amp)
        sched.step(te)
        hist["train_loss"].append(tr)
        hist["test_loss"].append(te)
        print(f"Epoch {ep:02d} | train_loss={tr:.6f} | test_loss={te:.6f} | lr={model._opt.param_groups[0]['lr']:.2e}")

        if (ep + 1) % 5 == 0:
            model._hard_update()

    outdir = Path(args.save_root) / f"{args.split_tag}_team" / "by_team" / args.split_tag
    (outdir / "splits").mkdir(parents=True, exist_ok=True)
    with open(outdir / "splits" / "train_files.txt", "w") as f:
        for p in train_files:
            f.write(str(p) + "\n")
    with open(outdir / "splits" / "test_files.txt", "w") as f:
        for p in test_files:
            f.write(str(p) + "\n")
    with open(outdir / "metrics.json", "w") as f:
        json.dump(hist, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(hist["train_loss"], label="Train (TD+CQL[+AS+L1])")
    plt.plot(hist["test_loss"], label="Test (TD+CQL[+AS+L1])")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"QMIX Offline — by_team {args.split_tag}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "learning_curves.png", dpi=150)
    plt.close()

    torch.save(model.agent.state_dict(), outdir / "agent.pt")
    torch.save(model.mixer.state_dict(), outdir / "mixer.pt")
    print(f"[Saved] {outdir/'agent.pt'}")
    print(f"[Saved] {outdir/'mixer.pt'}")
    print(f"[Saved] {outdir/'metrics.json'}")
    print(f"[Saved] {outdir/'learning_curves.png'}")


if __name__ == "__main__":
    main()
