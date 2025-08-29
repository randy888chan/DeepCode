import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..refinement.policy import PolicyNetwork

class PPO:
    """
    Proximal Policy Optimization implementation for RICE system.
    Implements PPO with clipped objective and combined rewards (environment + exploration bonus).
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        learning_rate: float = 3e-4,
        n_epochs: int = 10,
        batch_size: int = 64,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.015,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: Policy network instance
            learning_rate: Learning rate for optimizer
            n_epochs: Number of epochs to train on the same data
            batch_size: Minibatch size for training
            clip_range: PPO clip range for policy loss
            clip_range_vf: PPO clip range for value function
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence threshold
            device: Device to run computations on
        """
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf or clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_values: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        exploration_bonus: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform a training step using PPO algorithm.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_values: Batch of value estimates from old policy
            old_log_probs: Batch of log probabilities from old policy
            advantages: Batch of advantage estimates
            returns: Batch of discounted returns
            exploration_bonus: Optional exploration bonus to add to advantages
            
        Returns:
            Dictionary containing training metrics
        """
        # Move inputs to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_values = old_values.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Add exploration bonus if provided
        if exploration_bonus is not None:
            exploration_bonus = exploration_bonus.to(self.device)
            advantages = advantages + exploration_bonus
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training stats
        pg_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        
        # Optimize policy for n_epochs
        for _ in range(self.n_epochs):
            # Generate random permutation for minibatches
            batch_size = states.size(0)
            indices = torch.randperm(batch_size)
            
            # Train in minibatches
            for start_idx in range(0, batch_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get minibatch
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_values = old_values[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]
                
                # Evaluate actions and get new log probs and values
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    mb_states, mb_actions
                )
                
                # Calculate policy loss with clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range
                )
                policy_loss = torch.mean(torch.max(policy_loss1, policy_loss2))
                
                # Calculate value loss with clipping
                values_pred = new_values
                values_pred_clipped = mb_old_values + torch.clamp(
                    values_pred - mb_old_values,
                    -self.clip_range_vf,
                    self.clip_range_vf
                )
                value_loss1 = (values_pred - mb_returns) ** 2
                value_loss2 = (values_pred_clipped - mb_returns) ** 2
                value_loss = torch.mean(torch.max(value_loss1, value_loss2)) * 0.5
                
                # Calculate entropy loss
                entropy_loss = -torch.mean(entropy)
                
                # Calculate total loss
                loss = (
                    policy_loss 
                    + self.vf_coef * value_loss 
                    + self.ent_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Calculate approximate KL divergence
                with torch.no_grad():
                    kl_div = torch.mean(
                        mb_old_log_probs - new_log_probs
                    )
                
                # Store stats
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_divs.append(kl_div.item())
                
                # Early stopping if KL divergence is too high
                if self.target_kl is not None and kl_div > 1.5 * self.target_kl:
                    break
            
            # Early stopping if KL divergence is too high
            if self.target_kl is not None and kl_div > 1.5 * self.target_kl:
                break
                
        # Return training stats
        return {
            "policy_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "kl_div": np.mean(kl_divs)
        }
    
    def save(self, path: str):
        """Save policy network state."""
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path: str):
        """Load policy network state."""
        self.policy.load_state_dict(torch.load(path))