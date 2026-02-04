from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlearn.sports.soccer.models.q_model_base import QModelBase
from rlearn.sports.soccer.modules.optimizer import LRScheduler, Optimizer
from rlearn.sports.soccer.modules.seq2seq_encoder import Seq2SeqEncoder


def _nan_to_num_(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class MixingNetwork(nn.Module):
    """
    State-conditioned mixer (QMIX) with non-negative hypernet weights to preserve monotonicity.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hyper_hidden: int = 64,
        mixing_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.mixing_hidden = int(mixing_hidden)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, self.n_agents * self.mixing_hidden),
        )
        self.hyper_b1 = nn.Linear(state_dim, self.mixing_hidden)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, self.mixing_hidden),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: (B, N)    per-agent scalar Q-values
        states:   (B, S)    global state features
        returns:  (B,)      joint Q_tot
        """
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(states).view(bs, 1, self.mixing_hidden)

        h1 = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        h1 = F.relu(h1)

        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.mixing_hidden, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)

        q_tot = torch.bmm(h1, w2) + b2
        return q_tot.view(bs)


@QModelBase.register("qmix")
class QMixModel(QModelBase):
    """
    QMIX LightningModule (CTDE):
    - Shared per-agent sequence encoder + linear head -> Q_i(t, a)
    - Mixer network -> Q_tot(t)
    - TD loss on Q_tot with target networks (hard or soft update)
    - Optional Double-Q selection (default True)
    - Optional Conservative Q-Learning regularizer (CQL)
    """

    def __init__(
        self,
        observation_dim: int,
        sequence_encoder: Dict[str, Any],
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        # SARSA-shaped params (accepted for compatibility with current training entrypoint)
        vocab_size: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        gamma: float = 0.99,
        lambda_: float = 0.0,
        lambda2_: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        offball_action_idx: Optional[list[int]] = None,
        onball_action_idx: Optional[list[int]] = None,
        defensive_action_idx: Optional[list[int]] = None,
        # QMIX-specific params (preferred)
        n_agents: Optional[int] = None,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        mixer_network: Optional[Dict[str, Any]] = None,
        *,
        target_update_interval: int = 200,
        tau: float = 1.0,  # tau==1.0 => hard update; 0<tau<1 => soft update
        use_double_q: bool = True,
        td_loss: str = "huber",  # "huber" | "mse"
        huber_delta: float = 1.0,
        q_clip: float = 0.0,
        burn_in: int = 0,
        cql_alpha: float = 0.0,
        lambda_as: float = 0.0,
        lambda_l1: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.n_agents = int(n_agents) if n_agents is not None else 1
        self.state_dim = int(state_dim) if state_dim is not None else self.observation_dim * self.n_agents

        inferred_action_dim = action_dim if action_dim is not None else vocab_size
        if inferred_action_dim is None:
            raise ValueError("action_dim (or vocab_size) must be provided")
        self.action_dim = int(inferred_action_dim)

        self.gamma = float(gamma)
        self.use_double_q = bool(use_double_q)

        self.td_loss = str(td_loss)
        self.huber_delta = float(huber_delta)
        self.q_clip = float(q_clip)
        self.burn_in = max(0, int(burn_in))

        self.cql_alpha = float(cql_alpha)
        self.lambda_as = float(lambda_as)
        self.lambda_l1 = float(lambda_l1)

        self._optimizer_config = optimizer
        self._scheduler_config = scheduler

        self.target_update_interval = max(1, int(target_update_interval))
        self.tau = float(tau)
        self._target_update_counter = 0

        self.pad_token_id = pad_token_id
        self.lambda_ = lambda_
        self.lambda2_ = lambda2_
        self.class_weights = class_weights
        self.offball_action_idx = offball_action_idx
        self.onball_action_idx = onball_action_idx
        self.defensive_action_idx = defensive_action_idx

        # Per-agent shared encoder/head.
        self.encoder = Seq2SeqEncoder.from_params(sequence_encoder)
        self.obs_proj = nn.Linear(self.observation_dim, self.encoder.get_input_dim())
        self.q_head = nn.Linear(self.encoder.get_output_dim(), self.action_dim)

        mixer_network = mixer_network or {}
        self.mixer = MixingNetwork(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            hyper_hidden=int(mixer_network.get("hyper_hidden", 64)),
            mixing_hidden=int(mixer_network.get("mixing_hidden", 32)),
        )

        # Target copies.
        self.target_encoder = Seq2SeqEncoder.from_params(sequence_encoder)
        self.target_obs_proj = nn.Linear(self.observation_dim, self.target_encoder.get_input_dim())
        self.target_q_head = nn.Linear(self.target_encoder.get_output_dim(), self.action_dim)
        self.target_mixer = MixingNetwork(
            n_agents=self.n_agents,
            state_dim=self.state_dim,
            hyper_hidden=int(mixer_network.get("hyper_hidden", 64)),
            mixing_hidden=int(mixer_network.get("mixing_hidden", 32)),
        )
        self._hard_update_targets()

    def _hard_update_targets(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_obs_proj.load_state_dict(self.obs_proj.state_dict())
        self.target_q_head.load_state_dict(self.q_head.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    @torch.no_grad()
    def _soft_update_targets(self, tau: float) -> None:
        def _polyak(target: nn.Module, source: nn.Module) -> None:
            for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

        _polyak(self.target_encoder, self.encoder)
        _polyak(self.target_obs_proj, self.obs_proj)
        _polyak(self.target_q_head, self.q_head)
        _polyak(self.target_mixer, self.mixer)

    def _get(self, batch: Dict[str, torch.Tensor], *keys: str) -> Optional[torch.Tensor]:
        for k in keys:
            if k in batch:
                return batch[k]
        return None

    def _derive_state(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(2)
        if obs.dim() != 4:
            raise ValueError(f"Unsupported obs rank {obs.dim()} (expected 3 or 4)")
        bsz, t, n_agents, obs_dim = obs.shape
        if obs_dim != self.observation_dim or n_agents != self.n_agents:
            raise ValueError(
                f"obs mismatch: expected (*,{self.n_agents},{self.observation_dim}), got {tuple(obs.shape)}"
            )
        return obs.reshape(bsz, t, n_agents * obs_dim)

    def _parse_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self._get(batch, "observation", "obs")
        state = self._get(batch, "state", "states")
        action = self._get(batch, "action", "actions")
        reward = self._get(batch, "reward", "rewards")
        done = self._get(batch, "done", "dones", "terminated")

        if obs is None or action is None or reward is None or done is None:
            missing = [
                k
                for k, v in {
                    "observation/obs": obs,
                    "action/actions": action,
                    "reward/rewards": reward,
                    "done/dones/terminated": done,
                }.items()
                if v is None
            ]
            raise KeyError(f"QMixModel batch is missing keys: {missing}")

        obs = _nan_to_num_(obs.float())
        reward = _nan_to_num_(reward.float())
        done = _nan_to_num_(done.float())
        action = action.long()

        if reward.dim() == 3:
            reward = reward.squeeze(-1)
        if done.dim() == 3:
            done = done.squeeze(-1)

        if obs.dim() == 3:
            if self.n_agents == 1:
                obs = obs.unsqueeze(2)
            else:
                if obs.size(-1) != self.n_agents * self.observation_dim:
                    raise ValueError(
                        f"Expected obs last dim == n_agents*observation_dim "
                        f"({self.n_agents*self.observation_dim}), got {obs.size(-1)}"
                    )
                obs = obs.view(obs.size(0), obs.size(1), self.n_agents, self.observation_dim)
        elif obs.dim() != 4:
            raise ValueError(f"Unsupported obs rank {obs.dim()} (expected 3 or 4)")

        if state is None:
            state = self._derive_state(obs)
        else:
            state = _nan_to_num_(state.float())

        if action.dim() == 2:
            action = action.unsqueeze(-1).expand(-1, -1, self.n_agents)
        elif action.dim() != 3:
            raise ValueError(f"Unsupported action rank {action.dim()} (expected 2 or 3)")

        return obs, state, action, reward, done

    def _time_mask(self, batch: Dict[str, torch.Tensor], t_steps: int, device: torch.device) -> torch.Tensor:
        mask = self._get(batch, "mask")
        if mask is None:
            return torch.ones((batch["action"].size(0), t_steps), device=device, dtype=torch.float32)

        mask = mask.float()
        if mask.dim() == 3:
            mask = mask.all(dim=-1).float()
        elif mask.dim() != 2:
            raise ValueError(f"Unsupported mask rank {mask.dim()} (expected 2 or 3)")

        if mask.size(1) != t_steps:
            if mask.size(1) == t_steps - 1:
                pad = torch.ones((mask.size(0), 1), device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, pad], dim=1)
            else:
                raise ValueError(f"mask length {mask.size(1)} does not match T={t_steps}")

        return mask.to(device=device)

    def _agent_qs(self, obs: torch.Tensor) -> torch.Tensor:
        bsz, t_steps, n_agents, obs_dim = obs.shape
        x = F.relu(self.obs_proj(obs.reshape(bsz * n_agents, t_steps, obs_dim)))
        out = self.encoder(x)
        q = self.q_head(out).view(bsz, n_agents, t_steps, self.action_dim).permute(0, 2, 1, 3).contiguous()
        if self.q_clip > 0:
            q = q.clamp(-self.q_clip, self.q_clip)
        return q

    @torch.no_grad()
    def _target_agent_qs(self, obs: torch.Tensor) -> torch.Tensor:
        bsz, t_steps, n_agents, obs_dim = obs.shape
        x = F.relu(self.target_obs_proj(obs.reshape(bsz * n_agents, t_steps, obs_dim)))
        out = self.target_encoder(x)
        q = self.target_q_head(out).view(bsz, n_agents, t_steps, self.action_dim).permute(0, 2, 1, 3).contiguous()
        if self.q_clip > 0:
            q = q.clamp(-self.q_clip, self.q_clip)
        return q

    def _td_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.td_loss == "huber":
            return F.smooth_l1_loss(pred, target, beta=self.huber_delta, reduction="none")
        if self.td_loss == "mse":
            return F.mse_loss(pred, target, reduction="none")
        raise ValueError(f"Unsupported td_loss={self.td_loss}")

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs, state, action, reward, done = self._parse_batch(batch)
        device = obs.device
        bsz, t_steps, n_agents, _ = obs.shape
        if n_agents != self.n_agents:
            raise ValueError(f"Batch n_agents={n_agents} does not match model.n_agents={self.n_agents}")

        mask_t = self._time_mask(batch, t_steps, device=device)
        if self.burn_in > 0 and self.burn_in < t_steps:
            mask_t[:, : self.burn_in] = 0.0

        next_obs = self._get(batch, "next_observation", "next_obs")
        next_state = self._get(batch, "next_state", "next_states")
        if next_obs is None:
            next_obs = obs[:, 1:, :, :].contiguous()
        else:
            next_obs = _nan_to_num_(next_obs.float())
            if next_obs.dim() == 3:
                if self.n_agents == 1:
                    next_obs = next_obs.unsqueeze(2)
                else:
                    next_obs = next_obs.view(next_obs.size(0), next_obs.size(1), self.n_agents, self.observation_dim)
            if next_obs.dim() != 4:
                raise ValueError(f"Unsupported next_obs rank {next_obs.dim()}")

        if next_state is None:
            next_state = state[:, 1:, :].contiguous()
        else:
            next_state = _nan_to_num_(next_state.float())

        t_minus_1 = t_steps - 1
        obs_t = obs[:, :t_minus_1, :, :]
        state_t = state[:, :t_minus_1, :]
        action_t = action[:, :t_minus_1, :]
        reward_t = reward[:, :t_minus_1]
        done_t = done[:, :t_minus_1]
        mask_tm1 = mask_t[:, :t_minus_1]

        q = self._agent_qs(obs_t)  # (B, T-1, N, A)
        action_space = self.action_dim

        as_loss = torch.zeros((), device=device)
        if self.lambda_as > 0.0 and ("action" in batch or "actions" in batch):
            acts_raw = action_t
            valid = (acts_raw >= 0) & (acts_raw < action_space)

            if valid.any():
                ce_input = q.reshape(bsz * t_minus_1 * n_agents, action_space)
                ce_target = acts_raw.clamp(0, action_space - 1).reshape(bsz * t_minus_1 * n_agents)

                ce_all = F.cross_entropy(ce_input, ce_target, reduction="none")
                valid_f = valid.reshape(-1).float()

                tm_f = mask_tm1.unsqueeze(-1).expand_as(valid).reshape(-1).float()
                valid_f = valid_f * tm_f

                denom = valid_f.sum().clamp_min(1.0)
                as_loss = (ce_all * valid_f).sum() / denom
            else:
                as_loss = torch.zeros((), device=device)

        action_valid = ((action_t >= 0) & (action_t < action_space)).float()
        acts = action_t.clamp(0, action_space - 1)
        q_taken = q.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        q_taken = q_taken * action_valid

        q_tot = self.mixer(
            q_taken.reshape(bsz * t_minus_1, n_agents),
            state_t.reshape(bsz * t_minus_1, -1),
        ).view(bsz, t_minus_1)

        with torch.no_grad():
            q_main_next_all = self._agent_qs(next_obs)
            q_tgt_next_all = self._target_agent_qs(next_obs)

            if self.use_double_q:
                greedy_next = q_main_next_all.argmax(dim=-1)
            else:
                greedy_next = q_tgt_next_all.argmax(dim=-1)

            greedy_valid = ((greedy_next >= 0) & (greedy_next < action_space)).float()
            greedy_next = greedy_next.clamp(0, action_space - 1)
            q_next_taken = q_tgt_next_all.gather(-1, greedy_next.unsqueeze(-1)).squeeze(-1)
            q_next_taken = q_next_taken * greedy_valid

            q_tot_next = self.target_mixer(
                q_next_taken.reshape(bsz * t_minus_1, n_agents),
                next_state.reshape(bsz * t_minus_1, -1),
            ).view(bsz, t_minus_1)

            not_done = (1.0 - done_t).clamp(0.0, 1.0)
            td_target = reward_t + self.gamma * not_done * q_tot_next

        td_elem = self._td_loss(q_tot, td_target)
        denom_td = mask_tm1.sum().clamp_min(1.0)
        td = (td_elem * mask_tm1).sum() / denom_td

        cql = torch.zeros((), device=device)
        if self.cql_alpha > 0.0:
            lse = torch.logsumexp(q, dim=-1)  # (B, T-1, N)
            cql_gap = (lse - q_taken).clamp_min(0.0) * action_valid
            cql_gap = cql_gap * mask_tm1.unsqueeze(-1)
            denom = (mask_tm1.unsqueeze(-1) * action_valid).sum().clamp_min(1.0)
            cql = self.cql_alpha * cql_gap.sum() / denom

        l1 = torch.zeros((), device=device)
        if self.lambda_l1 > 0.0:
            l1 = self.lambda_l1 * sum(p.abs().sum() for p in self.parameters())

        loss = td + cql + (self.lambda_as * as_loss) + l1

        if not torch.isfinite(loss):
            return torch.zeros((), device=device, requires_grad=True)

        self.log("td_loss", td, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if self.cql_alpha > 0.0:
            self.log("cql_loss", cql, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if self.lambda_as > 0.0:
            self.log("as_loss", as_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs: torch.Tensor, batch: Any, batch_idx: int) -> None:
        self._target_update_counter += 1
        if 0.0 < self.tau < 1.0:
            self._soft_update_targets(self.tau)
        elif self._target_update_counter % self.target_update_interval == 0:
            self._hard_update_targets()

    def configure_optimizers(self):
        if self._optimizer_config is None:
            return torch.optim.Adam(self.parameters(), lr=5e-4)

        optimizer_ = Optimizer.from_params(params_=self._optimizer_config, params=self.parameters())

        if self._scheduler_config is None:
            return optimizer_

        scheduler_ = LRScheduler.from_params(params_=self._scheduler_config, optimizer=optimizer_)
        return [optimizer_], [scheduler_]

    @torch.no_grad()
    def get_actions(self, obs: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        obs: (B, N, O)  single timestep
        returns actions: (B, N)
        """
        if obs.dim() != 3:
            raise ValueError(f"get_actions expects (B, N, O), got {tuple(obs.shape)}")

        bsz, n_agents, obs_dim = obs.shape
        if n_agents != self.n_agents or obs_dim != self.observation_dim:
            raise ValueError(
                f"get_actions shape mismatch: expected (B,{self.n_agents},{self.observation_dim}) got {tuple(obs.shape)}"
            )

        obs_seq = obs.unsqueeze(1)  # (B, 1, N, O)
        q = self._agent_qs(obs_seq)[:, 0, :, :]  # (B, N, A)

        greedy = q.argmax(dim=-1)
        if epsilon <= 0.0:
            return greedy

        rnd = torch.randint(low=0, high=self.action_dim, size=greedy.shape, device=greedy.device)
        explore = torch.rand_like(greedy.float()) < epsilon
        return torch.where(explore, rnd, greedy)
