import torch
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

class StateMixer:
    """
    Implements the mixed state distribution mechanism for RICE.
    Combines regular environment states with critical states identified
    through the explanation mechanism.
    """
    
    def __init__(
        self,
        beta: float = 0.3,  # mixing probability (0.25-0.5 recommended)
        buffer_size: int = 10000,  # size of critical states buffer
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the StateMixer.
        
        Args:
            beta: Mixing probability between regular and critical states
            buffer_size: Maximum number of critical states to store
            device: Device to store tensors on
        """
        assert 0.0 <= beta <= 1.0, "beta must be between 0 and 1"
        self.beta = beta
        self.device = device
        
        # Initialize critical states buffer
        self.critical_states_buffer = deque(maxlen=buffer_size)
        self.critical_states_probs = deque(maxlen=buffer_size)
        
    def add_critical_state(
        self,
        state: torch.Tensor,
        importance_score: float
    ) -> None:
        """
        Add a critical state to the buffer with its importance score.
        
        Args:
            state: State tensor to store
            importance_score: Score indicating state importance (e.g., mask probability)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
            
        self.critical_states_buffer.append(state)
        self.critical_states_probs.append(importance_score)
        
    def add_critical_states_batch(
        self,
        states: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> None:
        """
        Add a batch of critical states with their importance scores.
        
        Args:
            states: Batch of state tensors
            importance_scores: Corresponding importance scores
        """
        for state, score in zip(states, importance_scores):
            self.add_critical_state(state, score.item())
            
    def sample_mixed_states(
        self,
        env_states: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a mixed batch of states combining environment and critical states.
        
        Args:
            env_states: Current batch of environment states
            batch_size: Optional size of batch to return (defaults to env_states size)
            
        Returns:
            Tuple of (mixed states tensor, mixing indicators tensor)
        """
        if batch_size is None:
            batch_size = len(env_states)
            
        # Determine number of states to sample from each source
        n_critical = int(self.beta * batch_size)
        n_regular = batch_size - n_critical
        
        # Sample regular environment states
        if n_regular > 0:
            regular_indices = torch.randperm(len(env_states))[:n_regular]
            regular_states = env_states[regular_indices]
        else:
            regular_states = torch.empty((0,) + env_states.shape[1:], device=self.device)
            
        # Sample critical states if available
        if n_critical > 0 and len(self.critical_states_buffer) > 0:
            # Convert buffers to tensors for sampling
            critical_states = torch.stack(list(self.critical_states_buffer))
            critical_probs = torch.tensor(list(self.critical_states_probs), device=self.device)
            critical_probs = critical_probs / critical_probs.sum()  # normalize probabilities
            
            # Sample critical states based on their importance scores
            critical_indices = torch.multinomial(
                critical_probs,
                min(n_critical, len(critical_probs)),
                replacement=True
            )
            critical_states = critical_states[critical_indices]
            
            # If we don't have enough critical states, pad with regular states
            if len(critical_indices) < n_critical:
                padding_size = n_critical - len(critical_indices)
                padding_indices = torch.randperm(len(env_states))[:padding_size]
                padding_states = env_states[padding_indices]
                critical_states = torch.cat([critical_states, padding_states])
        else:
            # If no critical states available, use regular states
            extra_indices = torch.randperm(len(env_states))[:n_critical]
            critical_states = env_states[extra_indices]
            
        # Combine and shuffle states
        mixed_states = torch.cat([regular_states, critical_states])
        shuffle_indices = torch.randperm(len(mixed_states))
        mixed_states = mixed_states[shuffle_indices]
        
        # Create mixing indicators (0 for regular, 1 for critical)
        mixing_indicators = torch.cat([
            torch.zeros(n_regular, device=self.device),
            torch.ones(n_critical, device=self.device)
        ])[shuffle_indices]
        
        return mixed_states, mixing_indicators
    
    def get_buffer_stats(self) -> dict:
        """
        Get statistics about the critical states buffer.
        
        Returns:
            Dictionary containing buffer statistics
        """
        return {
            "buffer_size": len(self.critical_states_buffer),
            "buffer_capacity": self.critical_states_buffer.maxlen,
            "avg_importance": np.mean(list(self.critical_states_probs)) if self.critical_states_probs else 0.0,
            "max_importance": np.max(list(self.critical_states_probs)) if self.critical_states_probs else 0.0,
        }
    
    def clear_buffer(self) -> None:
        """Clear the critical states buffer."""
        self.critical_states_buffer.clear()
        self.critical_states_probs.clear()