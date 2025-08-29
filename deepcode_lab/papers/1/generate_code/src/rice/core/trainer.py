import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import gym
from ..core.rice import RICE
from ..utils.logging import Logger

class RICETrainer:
    """
    Trainer class for the RICE (Reinforcement learning with Integrated Critical Explanations) system.
    Handles the training loop, evaluation, and logging of results.
    """
    def __init__(
        self,
        env_name: str,
        rice_config: Dict,
        max_episodes: int = 1000,
        steps_per_episode: int = 1000,
        eval_frequency: int = 10,
        num_eval_episodes: int = 5,
        save_dir: str = "checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the RICE trainer.
        
        Args:
            env_name: Name of the Gym environment
            rice_config: Configuration dictionary for RICE system
            max_episodes: Maximum number of training episodes
            steps_per_episode: Maximum steps per episode
            eval_frequency: Evaluate every N episodes
            num_eval_episodes: Number of episodes for evaluation
            save_dir: Directory to save model checkpoints
            device: Device to run the training on
        """
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)
        
        # Get environment dimensions
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
        else:
            action_dim = self.env.action_space.n
            
        # Initialize RICE system
        self.rice = RICE(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **rice_config
        )
        
        self.max_episodes = max_episodes
        self.steps_per_episode = steps_per_episode
        self.eval_frequency = eval_frequency
        self.num_eval_episodes = num_eval_episodes
        self.save_dir = save_dir
        self.device = device
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = Logger()
        
    def train_episode(self) -> Tuple[float, int]:
        """
        Train for one episode.
        
        Returns:
            Tuple of (episode_reward, episode_length)
        """
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.steps_per_episode):
            # Get action from RICE system
            action = self.rice.act(state)
            
            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Update RICE system
            self.rice.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
                
        return episode_reward, episode_length
        
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the current policy.
        
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        eval_rewards = []
        
        for _ in range(self.num_eval_episodes):
            state = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.rice.act(state, eval=True)
                next_state, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
                state = next_state
                
            eval_rewards.append(episode_reward)
            
        return np.mean(eval_rewards), np.std(eval_rewards)
        
    def train(self) -> Dict:
        """
        Main training loop.
        
        Returns:
            Dictionary containing training statistics
        """
        best_eval_reward = float('-inf')
        training_stats = {
            'train_rewards': [],
            'eval_rewards': [],
            'eval_stds': [],
            'episode_lengths': []
        }
        
        for episode in range(self.max_episodes):
            # Train for one episode
            episode_reward, episode_length = self.train_episode()
            
            # Log training statistics
            training_stats['train_rewards'].append(episode_reward)
            training_stats['episode_lengths'].append(episode_length)
            
            # Evaluate if needed
            if (episode + 1) % self.eval_frequency == 0:
                eval_reward, eval_std = self.evaluate()
                training_stats['eval_rewards'].append(eval_reward)
                training_stats['eval_stds'].append(eval_std)
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_checkpoint(os.path.join(self.save_dir, 'best_model.pt'))
                    
                print(f"Episode {episode + 1}: Train reward: {episode_reward:.2f}, "
                      f"Eval reward: {eval_reward:.2f} ± {eval_std:.2f}")
                      
            # Regular checkpoint saving
            if (episode + 1) % 100 == 0:
                self.save_checkpoint(os.path.join(self.save_dir, f'checkpoint_{episode + 1}.pt'))
                
        return training_stats
        
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the RICE system.
        
        Args:
            path: Path to save the checkpoint
        """
        self.rice.save(path)
        
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint of the RICE system.
        
        Args:
            path: Path to the checkpoint
        """
        self.rice.load(path)