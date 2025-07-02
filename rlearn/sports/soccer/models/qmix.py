import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

class AgentNetwork(nn.Module):
    """DRQN for individual soccer agents (shared parameters)"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128, rnn_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        
        # Store init args for target network creation
        self.init_args = (obs_dim, action_dim, hidden_dim, rnn_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.GRU(hidden_dim, rnn_dim, batch_first=True)
        self.q_net = nn.Linear(rnn_dim, action_dim)
        
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn_dim)
    
    def forward(self, obs, hidden_state):
        # obs: (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
        
        x = self.fc(obs)
        out, h = self.rnn(x, hidden_state)
        q_values = self.q_net(out)
        return q_values, h

class MixingNetwork(nn.Module):
    """QMIX mixer with hypernetworks for RoboCup2D"""
    def __init__(self, n_agents, state_dim, hyper_hidden=64, mixing_hidden=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_hidden = hyper_hidden
        self.mixing_hidden = mixing_hidden
        
        # Store init args for target network creation
        self.init_args = (n_agents, state_dim, hyper_hidden, mixing_hidden)
        
        # Hypernetwork for weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden)
        )
        
        # Hypernetwork for biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        
        # Final layer hypernetworks
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

    def forward(self, agent_qs, states):
        bs = states.size(0)
        
        # Generate weights and biases
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)  # Fixed syntax error
        w1 = w1.view(bs, self.n_agents, -1)
        b1 = b1.view(bs, 1, -1)
        
        # First layer mixing
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.relu(hidden)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states)).view(bs, -1, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)  # Fixed syntax error
        
        # Final output
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze(-1)

class QMIX:
    """Complete QMIX implementation with training logic"""
    
    def __init__(self, config, log_dir=None, device="auto"):
        self.device = self._get_device(device)
        self.config = config
        
        # Initialize agent network
        self.agent_net = AgentNetwork(
            obs_dim=config["obs_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config.get("hidden_dim", 128),
            rnn_dim=config.get("rnn_dim", 64)
        ).to(self.device)
        
        # Initialize mixer network
        self.mixer_net = MixingNetwork(
            n_agents=config["n_agents"],
            state_dim=config["state_dim"],
            hyper_hidden=config.get("hyper_hidden", 64),
            mixing_hidden=config.get("mixing_hidden", 32)
        ).to(self.device)
        
        # Initialize trainer
        self.trainer = QMIXTrainer(
            agent_net=self.agent_net,
            mixer_net=self.mixer_net,
            config=config,
            log_dir=log_dir,
            device=self.device
        )
    
    def _get_device(self, device_str):
        """Determine the best available device"""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
    
    def train(self, batch):
        """Perform one training step"""
        return self.trainer.update(batch)
    
    def get_actions(self, obs, hidden_states=None, epsilon=0.0):
        """Get actions from agent network for environment interaction"""
        self.agent_net.eval()
        with torch.no_grad():
            if hidden_states is None:
                batch_size = obs.size(0) if obs.dim() > 1 else 1
                hidden_states = self.agent_net.init_hidden(batch_size).to(self.device)
            
            obs = obs.to(self.device)
            q_values, new_hidden = self.agent_net(obs, hidden_states)
            
            # Epsilon-greedy action selection
            if torch.rand(1).item() < epsilon:
                actions = torch.randint(0, self.config["action_dim"], (q_values.size(0),))
            else:
                actions = q_values.argmax(dim=-1)
            
        self.agent_net.train()
        return actions, new_hidden
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            "agent_net": self.agent_net.state_dict(),
            "mixer_net": self.mixer_net.state_dict(),
            "config": self.config
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(checkpoint["agent_net"])
        self.mixer_net.load_state_dict(checkpoint["mixer_net"])
    
    def close(self):
        """Clean up resources"""
        self.trainer.close()

class QMIXTrainer:
    """QMIX training logic with TensorBoard logging"""
    
    def __init__(self, agent_net, mixer_net, config, log_dir=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.agent_net = agent_net
        self.mixer_net = mixer_net
        
        # Target networks
        self.target_agent_net = AgentNetwork(*agent_net.init_args).to(self.device)
        self.target_mixer_net = MixingNetwork(*mixer_net.init_args).to(self.device)
        self._hard_update()
        
        # Optimizer
        params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        self.optimizer = optim.Adam(params, lr=config["lr"])

        # Training parameters
        self.gamma = config["gamma"]
        self.tau = config.get("tau", 0.005)
        self.n_agents = config["n_agents"]
        self.grad_norm_clip = config.get("grad_norm_clip", 10.0)
        
        # Logging setup
        self._init_logging(log_dir)
        
        # Training stats
        self.global_step = 0
        self.start_time = time.time()
        self.loss_history = []
        self.q_history = []

    def _hard_update(self):
        """Initialize target networks with same weights"""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

    def _init_logging(self, log_dir):
        """Initialize TensorBoard logging"""
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("runs", f"qmix_{timestamp}")
        
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logging to: {log_dir}")

    def update(self, batch):
        """Perform one training update"""
        # Prepare batch
        states = batch["states"].float().to(self.device)
        obs = batch["obs"].float().to(self.device)
        actions = batch["actions"].long().to(self.device)
        rewards = batch["rewards"].float().to(self.device)
        next_states = batch["next_states"].float().to(self.device)
        next_obs = batch["next_obs"].float().to(self.device)
        dones = batch["dones"].float().to(self.device)

        # Compute Q values
        q_total = self._compute_q_values(obs, states, actions)
        
        # Compute targets
        with torch.no_grad():
            target_q = self._compute_targets(next_obs, next_states, rewards, dones)
        
        # Optimize
        loss = F.mse_loss(q_total, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.agent_net.parameters(), self.grad_norm_clip)
        torch.nn.utils.clip_grad_norm_(self.mixer_net.parameters(), self.grad_norm_clip)
        
        self.optimizer.step()
        
        # Soft update targets
        self._soft_update()
        
        # Logging
        self._log_stats(loss, q_total, target_q)
        
        return loss.item()

    def _compute_q_values(self, obs, states, actions):
        """Compute Q values through agent and mixer networks"""
        B, T, N, O = obs.shape  # batch, time, agents, obs_dim
        
        # Reshape for agent network processing
        obs_reshaped = obs.view(B * N, T, O)  # (batch*agents, time, obs_dim)
        
        # Initialize hidden states
        hidden = self.agent_net.init_hidden(B * N).to(self.device)
        
        # Process through agent network
        q_values, _ = self.agent_net(obs_reshaped, hidden)  # (B*N, T, action_dim)
        
        # Reshape back and select actions
        q_values = q_values.view(B, N, T, -1).transpose(1, 2)  # (B, T, N, action_dim)
        actions_expanded = actions.unsqueeze(-1)  # (B, T, N, 1)
        q_taken = q_values.gather(-1, actions_expanded).squeeze(-1)  # (B, T, N)
        
        # Mix Q values for each timestep
        q_total_list = []
        for t in range(T):
            agent_qs_t = q_taken[:, t, :]  # (B, N)
            states_t = states[:, t, :]  # (B, state_dim)
            q_total_t = self.mixer_net(agent_qs_t, states_t)
            q_total_list.append(q_total_t)
        
        return torch.stack(q_total_list, dim=1)  # (B, T)

    def _compute_targets(self, next_obs, next_states, rewards, dones):
        """Compute target Q values using target networks"""
        B, T, N, O = next_obs.shape
        
        # Reshape for agent network processing
        next_obs_reshaped = next_obs.view(B * N, T, O)
        
        # Initialize hidden states for target network
        hidden = self.target_agent_net.init_hidden(B * N).to(self.device)
        
        # Process through target agent network
        q_next, _ = self.target_agent_net(next_obs_reshaped, hidden)
        q_next = q_next.view(B, N, T, -1).transpose(1, 2)  # (B, T, N, action_dim)
        
        # Get max Q values
        max_q_next = q_next.max(dim=-1)[0]  # (B, T, N)
        
        # Mix target Q values for each timestep
        target_q_list = []
        for t in range(T):
            max_q_next_t = max_q_next[:, t, :]  # (B, N)
            next_states_t = next_states[:, t, :]  # (B, state_dim)
            target_q_t = self.target_mixer_net(max_q_next_t, next_states_t)
            target_q_list.append(target_q_t)
        
        target_q_total = torch.stack(target_q_list, dim=1)  # (B, T)
        
        # Compute TD targets
        targets = rewards + self.gamma * (1 - dones) * target_q_total
        return targets

    def _soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_mixer_net.parameters(), self.mixer_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _log_stats(self, loss, q_values, targets):
        """Log training statistics"""
        self.writer.add_scalar("Loss/td_error", loss.item(), self.global_step)
        self.writer.add_scalar("Q/average_q", q_values.mean().item(), self.global_step)
        self.writer.add_scalar("Q/target_q", targets.mean().item(), self.global_step)
        
        # Store history
        self.loss_history.append(loss.item())
        self.q_history.append(q_values.mean().item())
        
        # Print periodically
        if self.global_step % 100 == 0:
            elapsed = time.time() - self.start_time
            print(f"Step {self.global_step}: Loss={loss.item():.4f}, "
                  f"Q={q_values.mean().item():.4f}, "
                  f"Target={targets.mean().item():.4f}, "
                  f"Time={elapsed:.2f}s")
        
        self.global_step += 1

    def close(self):
        """Clean up resources"""
        self.writer.close()
        print(f"\nTraining completed in {time.time() - self.start_time:.2f} seconds")
        if self.loss_history:
            print(f"Final average loss: {sum(self.loss_history)/len(self.loss_history):.4f}")

# Example usage and configuration
def create_qmix_config():
    """Create a configuration for QMIX training"""
    return {
        "obs_dim": 20,          # Agent observation dimension
        "action_dim": 9,        # Number of possible actions  
        "state_dim": 50,        # Global state dimension
        "n_agents": 11,         # Number of agents (soccer team)
        "hidden_dim": 128,      # Agent network hidden dimension
        "rnn_dim": 64,          # RNN hidden dimension
        "hyper_hidden": 64,     # Mixer hypernetwork hidden dimension
        "mixing_hidden": 32,    # Mixer network hidden dimension
        "lr": 0.0005,           # Learning rate
        "gamma": 0.99,          # Discount factor
        "tau": 0.005,           # Target network update rate
        "grad_norm_clip": 10.0  # Gradient clipping
    }

# Example training loop
def example_training():
    """Example of how to use the integrated QMIX system"""
    config = create_qmix_config()
    qmix = QMIX(config, log_dir="./logs/qmix_soccer")
    
    # Example batch structure (you would get this from your replay buffer)
    batch_size, seq_len = 32, 10
    example_batch = {
        "states": torch.randn(batch_size, seq_len, config["state_dim"]),
        "obs": torch.randn(batch_size, seq_len, config["n_agents"], config["obs_dim"]),
        "actions": torch.randint(0, config["action_dim"], (batch_size, seq_len, config["n_agents"])),
        "rewards": torch.randn(batch_size, seq_len),
        "next_states": torch.randn(batch_size, seq_len, config["state_dim"]),
        "next_obs": torch.randn(batch_size, seq_len, config["n_agents"], config["obs_dim"]),
        "dones": torch.zeros(batch_size, seq_len)
    }
    
    # Training step
    loss = qmix.train(example_batch)
    print(f"Training loss: {loss}")
    
    # Example of getting actions for environment interaction
    current_obs = torch.randn(config["n_agents"], config["obs_dim"])
    actions, hidden_states = qmix.get_actions(current_obs, epsilon=0.1)
    print(f"Selected actions: {actions}")
    
    qmix.close()

if __name__ == "__main__":
    main_training()
