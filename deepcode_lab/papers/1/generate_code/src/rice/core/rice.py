import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from ..explanation.mask_network import StateMask
from ..exploration.rnd_networks import RNDModule
from ..distribution.mixer import StateMixer
from ..refinement.ppo import PPO
from ..refinement.policy import PolicyNetwork

class RICE:
    """
    RICE (Reinforcement learning with Integrated Critical Explanations) system.
    Combines explanation-based refinement with exploration for improved RL performance.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # Explanation parameters
        mask_hidden_dims: list = [256, 256],
        alpha: float = 0.0001,  # Blinding bonus coefficient
        # Distribution mixing parameters
        beta: float = 0.25,  # Mixing probability
        buffer_size: int = 10000,
        # RND parameters
        rnd_output_dim: int = 128,
        rnd_hidden_dims: list = [256, 256],
        rnd_learning_rate: float = 1e-3,
        # PPO parameters
        learning_rate: float = 3e-4,
        n_epochs: int = 10,
        batch_size: int = 64,
        clip_range: float = 0.2,
    ):
        """
        Initialize RICE system with all components.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions for policy network
            device: Device to run computations on
            mask_hidden_dims: Hidden dimensions for mask network
            alpha: Blinding bonus coefficient
            beta: State distribution mixing probability
            buffer_size: Size of critical state buffer
            rnd_output_dim: Output dimension for RND networks
            rnd_hidden_dims: Hidden dimensions for RND networks
            rnd_learning_rate: Learning rate for RND predictor
            learning_rate: Learning rate for PPO
            n_epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            clip_range: PPO clip range
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize components
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        self.mask_network = StateMask(
            state_dim=state_dim,
            hidden_sizes=mask_hidden_dims
        ).to(device)
        
        self.rnd = RNDModule(
            input_dim=state_dim,
            hidden_dims=rnd_hidden_dims,
            output_dim=rnd_output_dim,
            learning_rate=rnd_learning_rate,
            device=device
        )
        
        self.state_mixer = StateMixer(
            beta=beta,
            buffer_size=buffer_size,
            device=device
        )
        
        self.ppo = PPO(
            policy=self.policy,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            clip_range=clip_range,
            device=device
        )
        
        # Store hyperparameters
        self.alpha = alpha
        self.beta = beta
        
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        infos: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Update all RICE components using collected experience.
        
        Args:
            states: Batch of states (B, state_dim)
            actions: Batch of actions (B, action_dim)
            rewards: Batch of rewards (B,)
            next_states: Batch of next states (B, state_dim)
            dones: Batch of done flags (B,)
            infos: Additional information dictionary
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {}
        
        # 1. Update RND and compute exploration bonuses
        self.rnd.update(next_states)
        exploration_bonuses = self.rnd.compute_bonus(next_states)
        
        # 2. Generate and update explanations
        mask_loss = self.mask_network.compute_explanation_loss(
            states, actions, rewards, alpha=self.alpha
        )
        metrics['mask_loss'] = mask_loss.item()
        
        # 3. Identify critical states
        critical_states = self.mask_network.get_critical_states(states)
        if critical_states is not None:
            self.state_mixer.add_critical_states_batch(
                critical_states,
                torch.ones(critical_states.size(0), device=self.device)  # Uniform importance
            )
        
        # 4. Create mixed state distribution for policy update
        mixed_states, mixing_indicators = self.state_mixer.sample_mixed_states(
            states, batch_size=states.size(0)
        )
        
        # 5. Combine rewards with exploration bonuses
        combined_rewards = rewards + exploration_bonuses
        
        # 6. Update policy using PPO
        ppo_metrics = self.ppo.train_step(
            states=mixed_states,
            actions=actions,
            rewards=combined_rewards,
            next_states=next_states,
            dones=dones
        )
        metrics.update(ppo_metrics)
        
        return metrics
    
    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Select action for given state.
        
        Args:
            state: Current state (state_dim,)
            deterministic: Whether to sample deterministically
            
        Returns:
            action: Selected action
            info: Dictionary with additional information
        """
        with torch.no_grad():
            state = torch.as_tensor(state, device=self.device).float()
            if state.ndim == 1:
                state = state.unsqueeze(0)
                
            # Get action from policy
            action, log_prob, value = self.policy(state)
            
            if deterministic:
                action = action.mean
            
            # Get mask and exploration bonus
            mask_logits, mask_probs, _ = self.mask_network(state)
            exploration_bonus = self.rnd.compute_bonus(state, update_stats=False)
            
            info = {
                'log_prob': log_prob,
                'value': value,
                'mask_probs': mask_probs,
                'exploration_bonus': exploration_bonus
            }
            
            return action, info
    
    def save(self, path: str) -> None:
        """Save RICE components to disk."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'mask_network_state_dict': self.mask_network.state_dict(),
            'rnd_state_dict': self.rnd.state_dict(),
            'ppo_state_dict': self.ppo.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """Load RICE components from disk."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.mask_network.load_state_dict(checkpoint['mask_network_state_dict'])
        self.rnd.load_state_dict(checkpoint['rnd_state_dict'])
        self.ppo.load_state_dict(checkpoint['ppo_state_dict'])