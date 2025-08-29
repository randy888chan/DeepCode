import torch
import torch.nn.functional as F
import numpy as np

class ExplanationObjectives:
    """
    Implements the modified explanation objectives for the RICE framework.
    Handles various loss computations and objective functions for training
    the explanation mechanism.
    """
    
    def __init__(self, alpha=0.0001, entropy_coef=0.01, value_loss_coef=0.5):
        """
        Initialize explanation objectives with hyperparameters.
        
        Args:
            alpha (float): Blinding bonus coefficient
            entropy_coef (float): Entropy coefficient for encouraging exploration
            value_loss_coef (float): Value loss coefficient for critic updates
        """
        self.alpha = alpha
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
    
    def compute_explanation_loss(self, mask_probs, mask_actions, rewards):
        """
        Compute the explanation loss incorporating the blinding bonus.
        
        Args:
            mask_probs (torch.Tensor): Probabilities from the mask network
            mask_actions (torch.Tensor): Binary mask actions
            rewards (torch.Tensor): Environment rewards
            
        Returns:
            torch.Tensor: Computed explanation loss
        """
        # Calculate blinding bonus based on mask actions
        blinding_bonus = self.alpha * (1 - mask_actions.float()).mean()
        
        # Compute modified reward incorporating blinding bonus
        modified_rewards = rewards + blinding_bonus
        
        # Calculate explanation loss using modified rewards
        explanation_loss = -(modified_rewards * mask_probs.log()).mean()
        
        return explanation_loss
    
    def compute_policy_loss(self, advantages, ratio, clip_param=0.2):
        """
        Compute the clipped PPO policy loss for mask network training.
        
        Args:
            advantages (torch.Tensor): Computed advantages
            ratio (torch.Tensor): Probability ratios between new and old policies
            clip_param (float): PPO clipping parameter
            
        Returns:
            torch.Tensor: Computed policy loss
        """
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss
    
    def compute_value_loss(self, values, returns):
        """
        Compute value function loss for the critic.
        
        Args:
            values (torch.Tensor): Predicted value estimates
            returns (torch.Tensor): Computed returns
            
        Returns:
            torch.Tensor: Value loss
        """
        value_loss = F.mse_loss(values, returns)
        return self.value_loss_coef * value_loss
    
    def compute_entropy_loss(self, mask_probs):
        """
        Compute entropy loss to encourage exploration in mask generation.
        
        Args:
            mask_probs (torch.Tensor): Mask probabilities
            
        Returns:
            torch.Tensor: Entropy loss
        """
        entropy = -(mask_probs * torch.log(mask_probs + 1e-10) + 
                   (1 - mask_probs) * torch.log(1 - mask_probs + 1e-10)).mean()
        return -self.entropy_coef * entropy
    
    def compute_total_loss(self, mask_probs, mask_actions, rewards, values, 
                          returns, ratio, advantages):
        """
        Compute the total combined loss for training.
        
        Args:
            mask_probs (torch.Tensor): Mask probabilities
            mask_actions (torch.Tensor): Binary mask actions
            rewards (torch.Tensor): Environment rewards
            values (torch.Tensor): Value predictions
            returns (torch.Tensor): Computed returns
            ratio (torch.Tensor): Policy ratios
            advantages (torch.Tensor): Computed advantages
            
        Returns:
            torch.Tensor: Total combined loss
            dict: Dictionary containing individual loss components
        """
        # Compute individual losses
        explanation_loss = self.compute_explanation_loss(
            mask_probs, mask_actions, rewards)
        policy_loss = self.compute_policy_loss(advantages, ratio)
        value_loss = self.compute_value_loss(values, returns)
        entropy_loss = self.compute_entropy_loss(mask_probs)
        
        # Combine all losses
        total_loss = (explanation_loss + policy_loss + 
                     value_loss + entropy_loss)
        
        # Return total loss and components for logging
        loss_components = {
            'explanation_loss': explanation_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def normalize_advantages(self, advantages):
        """
        Normalize advantages for stable training.
        
        Args:
            advantages (torch.Tensor): Raw advantages
            
        Returns:
            torch.Tensor: Normalized advantages
        """
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)