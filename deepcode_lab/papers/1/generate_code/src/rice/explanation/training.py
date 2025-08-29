import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
import numpy as np

from rice.explanation.mask_network import StateMask

class MaskTrainer:
    """
    Trainer class for the StateMask network using PPO-style optimization
    """
    def __init__(
        self,
        state_dim: int,
        learning_rate: float = 3e-4,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        alpha: float = 0.0001  # blinding bonus coefficient
    ):
        self.mask_network = StateMask(state_dim)
        self.optimizer = optim.Adam(self.mask_network.parameters(), lr=learning_rate)
        
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha

    def compute_returns(
        self,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute discounted returns for PPO training
        """
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return
            
        return returns

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update the mask network using PPO
        """
        # Compute returns and advantages
        returns = self.compute_returns(rewards, masks)
        advantages = returns
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update loop
        total_loss = 0
        total_explanation_loss = 0
        total_entropy = 0

        for _ in range(self.ppo_epochs):
            # Generate random permutation for batching
            permutation = torch.randperm(states.size(0))
            
            # Process in batches
            for start_idx in range(0, states.size(0), self.batch_size):
                # Get batch indices
                batch_indices = permutation[start_idx:start_idx + self.batch_size]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                logits, probs, new_actions = self.mask_network(batch_states)
                
                # Compute action log probabilities
                dist = torch.distributions.Bernoulli(probs)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                
                # Compute explanation loss
                explanation_loss = self.mask_network.compute_explanation_loss(
                    batch_states, batch_actions, batch_returns, self.alpha
                )
                
                # Compute entropy
                entropy = dist.entropy().mean()
                
                # Compute PPO ratio and clipped loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                # Compute final loss
                policy_loss = -torch.min(surr1, surr2).mean()
                loss = policy_loss + self.value_loss_coef * explanation_loss - self.entropy_coef * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.mask_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_explanation_loss += explanation_loss.item()
                total_entropy += entropy.item()

        # Compute averages
        avg_loss = total_loss / (self.ppo_epochs * (states.size(0) // self.batch_size))
        avg_explanation_loss = total_explanation_loss / (self.ppo_epochs * (states.size(0) // self.batch_size))
        avg_entropy = total_entropy / (self.ppo_epochs * (states.size(0) // self.batch_size))

        return {
            'loss': avg_loss,
            'explanation_loss': avg_explanation_loss,
            'entropy': avg_entropy
        }

    def train_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train on a batch of data
        """
        with torch.no_grad():
            _, probs, _ = self.mask_network(states)
            dist = torch.distributions.Bernoulli(probs)
            old_log_probs = dist.log_prob(actions).sum(-1)

        return self.update(states, actions, old_log_probs, rewards, masks)

    def save_model(self, path: str):
        """Save the mask network state"""
        torch.save(self.mask_network.state_dict(), path)

    def load_model(self, path: str):
        """Load the mask network state"""
        self.mask_network.load_state_dict(torch.load(path))