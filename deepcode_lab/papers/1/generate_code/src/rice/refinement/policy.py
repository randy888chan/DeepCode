import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Policy Network for RICE implementation using PPO architecture.
    Includes both actor (policy) and critic (value) networks.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        activation: nn.Module = nn.ReLU,
        log_std_init: float = 0.0,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation(),
            ])
            prev_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)

        # Policy head (actor)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

        # Value head (critic)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

        self.to(device)
        self.train()

    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            states: Batch of states (B, state_dim)
            deterministic: If True, return deterministic actions
            
        Returns:
            actions: Sampled or deterministic actions
            log_probs: Log probabilities of sampled actions
            values: Value function estimates
        """
        features = self.shared_net(states)
        
        # Get mean and constrained log std
        action_mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head,
            min=self.min_log_std,
            max=self.max_log_std
        )
        std = log_std.exp()

        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, std)

        if deterministic:
            actions = action_mean
        else:
            actions = dist.rsample()  # Reparameterization trick

        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.value_head(features).squeeze(-1)

        return actions, log_probs, values

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given state-action pairs.
        
        Args:
            states: Batch of states (B, state_dim)
            actions: Batch of actions (B, action_dim)
            
        Returns:
            log_probs: Log probabilities of the actions
            values: Value function estimates
            entropy: Policy entropy
        """
        features = self.shared_net(states)
        
        action_mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head,
            min=self.min_log_std,
            max=self.max_log_std
        )
        std = log_std.exp()

        dist = torch.distributions.Normal(action_mean, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.value_head(features).squeeze(-1)
        entropy = dist.entropy().mean()

        return log_probs, values, entropy

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        """Get value function estimates for given states."""
        features = self.shared_net(states)
        return self.value_head(features).squeeze(-1)

    def reset_parameters(self):
        """Initialize network parameters using orthogonal initialization."""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        self.shared_net.apply(init_weights)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def to(self, device: torch.device) -> 'PolicyNetwork':
        """Move the policy to the specified device."""
        self.device = device
        return super().to(device)