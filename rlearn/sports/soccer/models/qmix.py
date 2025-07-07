from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rlearn.sports.soccer.models.q_model_base import QModelBase

class AgentNetwork(nn.Module):
    """DRQN for individual agents (shared parameters)"""
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        hidden_dim: int = 128, 
        rnn_dim: int = 64
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.GRU(hidden_dim, rnn_dim, batch_first=True)
        self.q_net = nn.Linear(rnn_dim, action_dim)
        
    def init_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(1, batch_size, self.rnn_dim)
    
    def forward(self, obs: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        x = self.fc(obs)
        out, h = self.rnn(x, hidden_state)
        q_values = self.q_net(out)
        return q_values, h

class MixingNetwork(nn.Module):
    """QMIX mixer with hypernetworks"""
    def __init__(
        self, 
        n_agents: int, 
        state_dim: int, 
        hyper_hidden: int = 64, 
        mixing_hidden: int = 32
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden)
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1)
        )

    def forward(self, agent_qs: Tensor, states: Tensor) -> Tensor:
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(bs, self.n_agents, -1)
        b1 = b1.view(bs, 1, -1)
        
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.relu(hidden)
        
        w2 = torch.abs(self.hyper_w2(states)).view(bs, -1, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze(-1)

@QModelBase.register("qmix")
class QMIXModel(QModelBase):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        agent_hidden_dim: int = 128,
        agent_rnn_dim: int = 64,
        hyper_hidden_dim: int = 64,
        mixing_hidden_dim: int = 32,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 0.001,
        grad_norm_clip: float = 10.0,
        use_sarsa: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize networks
        self.agent_net = AgentNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=agent_hidden_dim,
            rnn_dim=agent_rnn_dim
        )
        
        self.mixer_net = MixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            hyper_hidden=hyper_hidden_dim,
            mixing_hidden=mixing_hidden_dim
        )
        
        # Initialize target networks
        self.target_agent_net = AgentNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=agent_hidden_dim,
            rnn_dim=agent_rnn_dim
        )
        
        self.target_mixer_net = MixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            hyper_hidden=hyper_hidden_dim,
            mixing_hidden=mixing_hidden_dim
        )
        
        self._hard_update()
        self.automatic_optimization = False

    def _hard_update(self) -> None:
        """Synchronize target networks with main networks"""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

    def _soft_update(self) -> None:
        """Soft update target networks"""
        tau = self.hparams.tau
        for target_param, param in zip(self.target_agent_net.parameters(), 
                                      self.agent_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_mixer_net.parameters(), 
                                      self.mixer_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _compute_q_values(
        self, 
        net: AgentNetwork, 
        obs: Tensor, 
        actions: Tensor
    ) -> Tensor:
        """Compute Q values through agent and mixer networks"""
        B, T, N, O = obs.shape
        obs_flat = obs.view(B * N, T, O)
        hidden = net.init_hidden(B * N).to(obs.device)
        
        q_values, _ = net(obs_flat, hidden)
        q_values = q_values.view(B, N, T, -1).transpose(1, 2)
        
        actions_expanded = actions.unsqueeze(-1)
        q_taken = q_values.gather(-1, actions_expanded).squeeze(-1)
        
        q_total_list = []
        for t in range(T):
            agent_qs_t = q_taken[:, t, :]
            states_t = self.current_batch["states"][:, t, :]
            q_total_t = self.mixer_net(agent_qs_t, states_t)
            q_total_list.append(q_total_t)
            
        return torch.stack(q_total_list, dim=1)

    def _compute_targets(self) -> Tensor:
        """Compute targets based on SARSA/Q-learning"""
        next_obs = self.current_batch["next_obs"]
        rewards = self.current_batch["rewards"]
        dones = self.current_batch["dones"]
        gamma = self.hparams.gamma
        
        if self.hparams.use_sarsa:  # SARSA
            next_actions = self.current_batch["next_actions"]
            next_q = self._compute_q_values(
                self.target_agent_net,
                next_obs,
                next_actions
            )
        else:  # Q-learning
            B, T, N, O = next_obs.shape
            next_obs_flat = next_obs.view(B * N, T, O)
            hidden = self.target_agent_net.init_hidden(B * N).to(next_obs.device)
            
            q_next, _ = self.target_agent_net(next_obs_flat, hidden)
            q_next = q_next.view(B, N, T, -1).transpose(1, 2)
            max_q_next = q_next.max(dim=-1)[0]
            
            next_q_list = []
            for t in range(T):
                max_q_next_t = max_q_next[:, t, :]
                next_states_t = self.current_batch["next_states"][:, t, :]
                next_q_t = self.target_mixer_net(max_q_next_t, next_states_t)
                next_q_list.append(next_q_t)
                
            next_q = torch.stack(next_q_list, dim=1)
        
        return rewards + gamma * (1 - dones) * next_q.detach()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        self.current_batch = batch
        opt = self.optimizers()
        opt.zero_grad()
        
        # Compute current Q values
        q_total = self._compute_q_values(
            self.agent_net,
            batch["obs"],
            batch["actions"]
        )
        
        # Compute targets
        targets = self._compute_targets()
        
        # Compute loss
        loss = F.mse_loss(q_total, targets)
        self.log("train_loss", loss)
        
        # Backpropagation
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.agent_net.parameters(), 
            self.hparams.grad_norm_clip
        )
        torch.nn.utils.clip_grad_norm_(
            self.mixer_net.parameters(), 
            self.hparams.grad_norm_clip
        )
        opt.step()
        
        # Soft update target networks
        self._soft_update()
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        self.current_batch = batch
        
        # Compute current Q values
        q_total = self._compute_q_values(
            self.agent_net,
            batch["obs"],
            batch["actions"]
        )
        
        # Compute targets
        targets = self._compute_targets()
        
        # Compute loss
        loss = F.mse_loss(q_total, targets)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        return torch.optim.Adam(params, lr=self.hparams.lr)

    def get_actions(
        self, 
        obs: Tensor, 
        hidden: Optional[Tensor] = None,
        epsilon: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        """Get actions for environment interaction"""
        self.agent_net.eval()
        if hidden is None:
            batch_size = obs.size(0)
            hidden = self.agent_net.init_hidden(batch_size).to(obs.device)
            
        with torch.no_grad():
            q_values, new_hidden = self.agent_net(obs, hidden)
            
            if torch.rand(1).item() < epsilon:
                actions = torch.randint(0, self.hparams.action_dim, (q_values.size(0),)
            else:
                actions = q_values.argmax(dim=-1)
                
        self.agent_net.train()
        return actions, new_hidden
