import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

class StateMask(nn.Module):
    """
    Enhanced StateMask implementation for RICE algorithm.
    Generates binary masks for state space exploration and explanation.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: List[int] = [128, 128],
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialize the StateMask network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function to use
        """
        super().__init__()
        
        # Build network layers
        layers = []
        prev_size = state_dim
        
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                activation(),
            ])
            prev_size = size
            
        # Output layer produces logits for each state dimension
        layers.append(nn.Linear(prev_size, state_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate masks.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            tuple: (logits, probabilities, binary_actions)
        """
        # Get logits from network
        logits = self.network(states)
        
        # Convert to probabilities using sigmoid
        probs = torch.sigmoid(logits)
        
        # Generate binary actions (0/1) using probabilities
        if self.training:
            # During training, sample from Bernoulli distribution
            dist = torch.distributions.Bernoulli(probs)
            actions = dist.sample()
        else:
            # During inference, use threshold of 0.5
            actions = (probs > 0.5).float()
            
        return logits, probs, actions
        
    def compute_explanation_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        alpha: float = 0.0001
    ) -> torch.Tensor:
        """
        Compute the explanation loss with blinding bonus.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            alpha: Blinding bonus coefficient
            
        Returns:
            Scalar loss value
        """
        # Get mask outputs
        logits, probs, mask_actions = self.forward(states)
        
        # Compute blinding bonus
        # Encourage minimal masks by penalizing high probabilities
        blinding_bonus = -alpha * probs.mean()
        
        # Compute main explanation loss
        # Use binary cross entropy between mask actions and original actions
        main_loss = F.binary_cross_entropy_with_logits(
            logits,
            actions.float(),
            reduction='mean'
        )
        
        # Weight the losses by rewards to focus on important states
        weighted_loss = main_loss * rewards.abs().mean()
        
        # Combine losses
        total_loss = weighted_loss + blinding_bonus
        
        return total_loss
        
    def get_critical_states(
        self,
        states: torch.Tensor,
        threshold: float = 0.8
    ) -> torch.Tensor:
        """
        Identify critical states based on mask probabilities.
        
        Args:
            states: Batch of states to evaluate
            threshold: Probability threshold for critical state identification
            
        Returns:
            Tensor containing critical states
        """
        with torch.no_grad():
            # Get mask probabilities
            _, probs, _ = self.forward(states)
            
            # Find states where any dimension has probability above threshold
            critical_mask = (probs > threshold).any(dim=1)
            
            # Return critical states
            critical_states = states[critical_mask]
            
        return critical_states