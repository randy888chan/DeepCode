"""
Random Network Distillation (RND) implementation for RICE exploration.
Provides target and predictor networks for novelty-based exploration.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class MLP(nn.Module):
    """Multi-layer perceptron network used for both target and predictor."""
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RNDModule(nn.Module):
    """
    Random Network Distillation module containing both target and predictor networks.
    The target network remains fixed while the predictor network is trained to match
    the target network's outputs.
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        learning_rate: float,
        device: str = "cpu"
    ):
        """
        Initialize RND module.
        
        Args:
            state_dim: Dimension of input state
            hidden_dims: Tuple of hidden layer dimensions
            output_dim: Dimension of output features
            learning_rate: Learning rate for predictor network
            device: Device to run the networks on
        """
        super().__init__()
        
        # Create target network (fixed random weights)
        self.target_network = MLP(state_dim, hidden_dims, output_dim)
        # Freeze target network weights
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        # Create predictor network (trainable)
        self.predictor_network = MLP(state_dim, hidden_dims, output_dim)
        
        # Setup optimizer for predictor network
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=learning_rate
        )
        
        self.to(device)
    
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks.
        
        Args:
            states: Input states tensor
            
        Returns:
            Tuple of (target_features, predicted_features)
        """
        with torch.no_grad():
            target_features = self.target_network(states)
        predicted_features = self.predictor_network(states)
        return target_features, predicted_features
    
    def compute_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between target and predictor outputs.
        
        Args:
            states: Input states tensor
            
        Returns:
            MSE loss value
        """
        target_features, predicted_features = self.forward(states)
        return torch.mean((target_features - predicted_features) ** 2)
    
    def update(self, states: torch.Tensor) -> float:
        """
        Update predictor network using MSE loss.
        
        Args:
            states: Batch of states to update on
            
        Returns:
            Loss value after update
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(states)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def get_exploration_bonus(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute exploration bonus as MSE between target and predictor.
        
        Args:
            states: Input states tensor
            
        Returns:
            Tensor of exploration bonuses
        """
        target_features, predicted_features = self.forward(states)
        return ((target_features - predicted_features) ** 2).mean(dim=-1)
    
    def state_dict(self) -> Dict:
        """Get state dict including target network, predictor network and optimizer."""
        return {
            'target_network': self.target_network.state_dict(),
            'predictor_network': self.predictor_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from provided state dict."""
        self.target_network.load_state_dict(state_dict['target_network'])
        self.predictor_network.load_state_dict(state_dict['predictor_network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])