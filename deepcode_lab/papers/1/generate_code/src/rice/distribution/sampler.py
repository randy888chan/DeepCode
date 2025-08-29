import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
from .mixer import StateMixer

class CriticalStateSampler:
    """
    Implements critical state sampling logic for the RICE framework.
    Works in conjunction with StateMixer to provide efficient state sampling
    with importance-based prioritization.
    """
    
    def __init__(
        self,
        state_dim: int,
        importance_threshold: float = 0.7,
        min_samples: int = 32,
        max_samples: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the CriticalStateSampler.
        
        Args:
            state_dim: Dimension of the state space
            importance_threshold: Threshold for considering a state as critical
            min_samples: Minimum number of samples to maintain
            max_samples: Maximum number of samples to store
            device: Device to store tensors on
        """
        self.state_dim = state_dim
        self.importance_threshold = importance_threshold
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.device = device
        
        # Initialize storage for states and their importance scores
        self.states: List[torch.Tensor] = []
        self.importance_scores: List[float] = []
        
    def identify_critical_states(
        self,
        states: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify critical states based on importance scores.
        
        Args:
            states: Batch of states to evaluate (B x state_dim)
            importance_scores: Importance scores for each state (B)
            
        Returns:
            Tuple of (critical_states, critical_scores)
        """
        # Ensure inputs are on correct device
        states = states.to(self.device)
        importance_scores = importance_scores.to(self.device)
        
        # Find states above threshold
        critical_mask = importance_scores >= self.importance_threshold
        critical_states = states[critical_mask]
        critical_scores = importance_scores[critical_mask]
        
        return critical_states, critical_scores
    
    def update_critical_states(
        self,
        states: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> Dict[str, int]:
        """
        Update the internal storage of critical states.
        
        Args:
            states: New states to evaluate
            importance_scores: Importance scores for the states
            
        Returns:
            Statistics about the update operation
        """
        critical_states, critical_scores = self.identify_critical_states(
            states, importance_scores
        )
        
        stats = {
            "total_critical": len(critical_states),
            "added": 0,
            "removed": 0
        }
        
        if len(critical_states) == 0:
            return stats
            
        # Convert to list of tensors for storage
        new_states = [s.cpu() for s in critical_states]
        new_scores = critical_scores.cpu().tolist()
        
        # Add new critical states
        self.states.extend(new_states)
        self.importance_scores.extend(new_scores)
        stats["added"] = len(new_states)
        
        # Maintain maximum size constraint
        if len(self.states) > self.max_samples:
            # Remove lowest importance states
            paired = list(zip(self.states, self.importance_scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            
            self.states = [s for s, _ in paired[:self.max_samples]]
            self.importance_scores = [score for _, score in paired[:self.max_samples]]
            stats["removed"] = len(paired) - self.max_samples
            
        return stats
    
    def sample_states(
        self,
        batch_size: Optional[int] = None,
        with_replacement: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample states from the critical state storage.
        
        Args:
            batch_size: Number of states to sample (default: all)
            with_replacement: Whether to sample with replacement
            
        Returns:
            Tuple of (sampled_states, importance_scores)
        """
        if len(self.states) < self.min_samples:
            raise ValueError(
                f"Not enough critical states collected (have {len(self.states)}, "
                f"need minimum {self.min_samples})"
            )
            
        if batch_size is None:
            batch_size = len(self.states)
            
        if with_replacement:
            indices = np.random.choice(
                len(self.states),
                size=batch_size,
                p=np.array(self.importance_scores) / sum(self.importance_scores)
            )
        else:
            indices = np.random.choice(
                len(self.states),
                size=min(batch_size, len(self.states)),
                replace=False
            )
            
        sampled_states = torch.stack([self.states[i] for i in indices]).to(self.device)
        sampled_scores = torch.tensor(
            [self.importance_scores[i] for i in indices],
            device=self.device
        )
        
        return sampled_states, sampled_scores
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get statistics about the current critical state storage.
        
        Returns:
            Dictionary containing storage statistics
        """
        if not self.states:
            return {
                "num_states": 0,
                "mean_importance": 0.0,
                "min_importance": 0.0,
                "max_importance": 0.0
            }
            
        return {
            "num_states": len(self.states),
            "mean_importance": np.mean(self.importance_scores),
            "min_importance": min(self.importance_scores),
            "max_importance": max(self.importance_scores)
        }
    
    def clear(self) -> None:
        """Clear all stored critical states."""
        self.states.clear()
        self.importance_scores.clear()