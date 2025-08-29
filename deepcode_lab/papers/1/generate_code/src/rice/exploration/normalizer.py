"""
Normalizer implementation for the RICE framework.
Provides running normalization of inputs and rewards for stable training.
"""

import torch
import numpy as np
from typing import Optional, Dict, Union


class RunningNormalizer:
    """
    Implements running normalization for input states and rewards.
    Maintains running statistics (mean, variance) for normalization.
    """
    
    def __init__(
        self,
        shape: Union[int, tuple],
        clip_range: float = 5.0,
        epsilon: float = 1e-8,
        device: str = "cpu"
    ):
        """
        Initialize the running normalizer.
        
        Args:
            shape: Input shape (int) or tuple of input shapes
            clip_range: Maximum absolute value after normalization
            epsilon: Small constant for numerical stability
            device: Device to store statistics on
        """
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.device = device
        
        # Initialize running statistics
        self.mean = torch.zeros(self.shape, device=device)
        self.var = torch.ones(self.shape, device=device)
        self.count = torch.tensor(0, device=device, dtype=torch.float32)
        
        # For tracking updates
        self.mean_sq = torch.zeros(self.shape, device=device)
        
    def update(self, x: torch.Tensor) -> None:
        """
        Update running statistics with new data.
        
        Args:
            x: Input tensor of shape matching self.shape
        """
        batch_mean = x.mean(dim=0)
        batch_size = x.shape[0]
        batch_sq_mean = (x ** 2).mean(dim=0)
        
        # Update count
        new_count = self.count + batch_size
        
        # Update mean using welford's online algorithm
        delta = batch_mean - self.mean
        self.mean += (delta * batch_size) / new_count
        
        # Update mean_sq
        self.mean_sq = (self.mean_sq * self.count + batch_sq_mean * batch_size) / new_count
        
        # Update variance
        self.var = self.mean_sq - self.mean ** 2
        self.var = torch.clamp(self.var, min=self.epsilon)
        
        # Update count
        self.count = new_count
        
    def normalize(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Normalize input tensor using current statistics.
        
        Args:
            x: Input tensor to normalize
            update_stats: Whether to update running statistics
            
        Returns:
            Normalized tensor
        """
        if update_stats:
            self.update(x)
            
        # Normalize and clip
        x_norm = (x - self.mean) / torch.sqrt(self.var)
        return torch.clamp(x_norm, -self.clip_range, self.clip_range)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization transform.
        
        Args:
            x: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        return x * torch.sqrt(self.var) + self.mean
    
    def state_dict(self) -> Dict:
        """Get normalizer state for saving."""
        return {
            'mean': self.mean.cpu().numpy(),
            'var': self.var.cpu().numpy(),
            'count': self.count.cpu().item(),
            'mean_sq': self.mean_sq.cpu().numpy(),
            'shape': self.shape,
            'clip_range': self.clip_range,
            'epsilon': self.epsilon
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load normalizer state.
        
        Args:
            state_dict: Dictionary containing normalizer state
        """
        self.mean = torch.tensor(state_dict['mean'], device=self.device)
        self.var = torch.tensor(state_dict['var'], device=self.device)
        self.count = torch.tensor(state_dict['count'], device=self.device)
        self.mean_sq = torch.tensor(state_dict['mean_sq'], device=self.device)
        self.shape = state_dict['shape']
        self.clip_range = state_dict['clip_range']
        self.epsilon = state_dict['epsilon']
    
    def to(self, device: str) -> 'RunningNormalizer':
        """
        Move normalizer to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.device = device
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        self.mean_sq = self.mean_sq.to(device)
        return self