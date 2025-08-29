import torch
import numpy as np
from typing import Optional, Tuple

from .rnd_networks import RNDModule
from .normalizer import RunningNormalizer

class ExplorationBonus:
    """
    Manages exploration bonus calculation using RND and normalization.
    Implements the bonus calculation mechanism described in the RICE paper.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        output_dim: int = 128,
        learning_rate: float = 3e-4,
        bonus_coef: float = 0.01,
        normalize_bonus: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize the exploration bonus calculator.
        
        Args:
            state_dim: Dimension of the state space
            hidden_dims: Hidden layer dimensions for RND networks
            output_dim: Output dimension for RND networks
            learning_rate: Learning rate for predictor network
            bonus_coef: Coefficient for scaling the exploration bonus (λ in paper)
            normalize_bonus: Whether to normalize the exploration bonuses
            device: Device to run computations on
        """
        self.device = torch.device(device)
        self.bonus_coef = bonus_coef
        self.normalize_bonus = normalize_bonus
        
        # Initialize RND module
        self.rnd = RNDModule(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            learning_rate=learning_rate,
            device=device
        )
        
        # Initialize bonus normalizer if needed
        self.bonus_normalizer = RunningNormalizer(shape=(1,), device=device) if normalize_bonus else None
        
    def compute_bonus(
        self,
        states: torch.Tensor,
        update_networks: bool = True
    ) -> torch.Tensor:
        """
        Compute exploration bonuses for given states.
        
        Args:
            states: Batch of states to compute bonuses for
            update_networks: Whether to update the predictor network
            
        Returns:
            Tensor of exploration bonuses for each state
        """
        # Ensure states are on correct device
        states = states.to(self.device)
        
        # Compute prediction error (MSE between target and predictor)
        with torch.set_grad_enabled(update_networks):
            prediction_error = self.rnd.compute_prediction_error(states)
            
            # Update predictor network if requested
            if update_networks:
                loss = self.rnd.update(states)
        
        # Scale the bonus
        bonus = self.bonus_coef * prediction_error
        
        # Normalize if enabled
        if self.normalize_bonus:
            self.bonus_normalizer.update(bonus.detach())
            bonus = self.bonus_normalizer.normalize(bonus)
            
        return bonus
    
    def update(self, states: torch.Tensor) -> float:
        """
        Update the RND predictor network.
        
        Args:
            states: Batch of states to update on
            
        Returns:
            Training loss
        """
        return self.rnd.update(states)
    
    def state_dict(self) -> dict:
        """Get state dictionary for saving."""
        state = {
            'rnd': self.rnd.state_dict(),
            'bonus_coef': self.bonus_coef,
        }
        if self.normalize_bonus:
            state['bonus_normalizer'] = self.bonus_normalizer.state_dict()
        return state
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load from state dictionary."""
        self.rnd.load_state_dict(state_dict['rnd'])
        self.bonus_coef = state_dict['bonus_coef']
        if self.normalize_bonus and 'bonus_normalizer' in state_dict:
            self.bonus_normalizer.load_state_dict(state_dict['bonus_normalizer'])
            
    def to(self, device: torch.device) -> 'ExplorationBonus':
        """Move module to specified device."""
        self.device = device
        self.rnd = self.rnd.to(device)
        if self.normalize_bonus:
            self.bonus_normalizer = self.bonus_normalizer.to(device)
        return self