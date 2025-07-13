"""
Configuration module for RICE (Robust Imitation via Contingency-aware Explanation).

This module defines the core configuration classes and parameters used throughout
the RICE implementation for robust imitation learning with explainable AI.
"""

import dataclasses
from typing import Dict, Any, Optional, List
import numpy as np


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # PPO Training Parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training Schedule
    total_timesteps: int = 1000000
    n_steps: int = 2048
    n_envs: int = 1
    
    # Evaluation
    eval_freq: int = 10000
    n_eval_episodes: int = 10


@dataclasses.dataclass
class ExplanationConfig:
    """Configuration for explanation and interpretability components."""
    
    # Mask Network Parameters
    mask_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [64, 32])
    mask_activation: str = "relu"
    mask_learning_rate: float = 1e-3
    
    # State Importance Parameters
    importance_threshold: float = 0.1
    importance_window_size: int = 10
    
    # Explanation Methods
    use_gradient_based: bool = True
    use_attention_based: bool = True
    use_perturbation_based: bool = False


@dataclasses.dataclass
class DistributionConfig:
    """Configuration for state distribution and sampling strategies."""
    
    # State Mixing Parameters
    mixing_ratio: float = 0.5
    mixing_strategy: str = "uniform"  # "uniform", "gaussian", "adaptive"
    
    # Sampling Parameters
    sampling_temperature: float = 1.0
    sampling_top_k: int = 10
    sampling_strategy: str = "importance"  # "random", "importance", "diversity"
    
    # Buffer Parameters
    buffer_size: int = 100000
    min_buffer_size: int = 1000


@dataclasses.dataclass
class EnvironmentConfig:
    """Configuration for environment setup and reset strategies."""
    
    # Environment Parameters
    env_name: str = "HalfCheetah-v3"
    env_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    # Reset Strategy
    reset_strategy: str = "standard"  # "standard", "diverse", "adversarial"
    reset_frequency: int = 1000
    
    # Observation/Action Space
    normalize_obs: bool = True
    normalize_reward: bool = False
    clip_actions: bool = True


@dataclasses.dataclass
class RobustnessConfig:
    """Configuration for robustness and contingency handling."""
    
    # Contingency Parameters
    contingency_threshold: float = 0.8
    contingency_window: int = 50
    
    # Robustness Metrics
    use_worst_case: bool = True
    use_average_case: bool = True
    robustness_alpha: float = 0.1
    
    # Bounds and Assumptions
    state_bounds_type: str = "learned"  # "fixed", "learned", "adaptive"
    assumption_violation_penalty: float = 1.0


@dataclasses.dataclass
class RICEConfig:
    """Main configuration class for RICE implementation."""
    
    # Sub-configurations
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    explanation: ExplanationConfig = dataclasses.field(default_factory=ExplanationConfig)
    distribution: DistributionConfig = dataclasses.field(default_factory=DistributionConfig)
    environment: EnvironmentConfig = dataclasses.field(default_factory=EnvironmentConfig)
    robustness: RobustnessConfig = dataclasses.field(default_factory=RobustnessConfig)
    
    # General Parameters
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    verbose: int = 1
    
    # Logging and Checkpointing
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_freq: int = 50000
    
    # Experiment Tracking
    experiment_name: str = "rice_experiment"
    tags: List[str] = dataclasses.field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RICEConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'RICEConfig':
        """Update configuration with new parameters."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Training validation
        assert self.training.learning_rate > 0, "Learning rate must be positive"
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert 0 < self.training.gamma <= 1, "Gamma must be in (0, 1]"
        
        # Explanation validation
        assert self.explanation.importance_threshold >= 0, "Importance threshold must be non-negative"
        assert len(self.explanation.mask_hidden_dims) > 0, "Mask network must have hidden layers"
        
        # Distribution validation
        assert 0 <= self.distribution.mixing_ratio <= 1, "Mixing ratio must be in [0, 1]"
        assert self.distribution.buffer_size > self.distribution.min_buffer_size, "Buffer size must be larger than minimum"
        
        # Robustness validation
        assert 0 <= self.robustness.contingency_threshold <= 1, "Contingency threshold must be in [0, 1]"
        assert self.robustness.robustness_alpha >= 0, "Robustness alpha must be non-negative"


def get_default_config() -> RICEConfig:
    """Get default RICE configuration."""
    return RICEConfig()


def get_mujoco_config(env_name: str = "HalfCheetah-v3") -> RICEConfig:
    """Get configuration optimized for MuJoCo environments."""
    config = RICEConfig()
    config.environment.env_name = env_name
    config.training.n_steps = 2048
    config.training.batch_size = 64
    config.training.learning_rate = 3e-4
    return config


def get_atari_config(env_name: str = "BreakoutNoFrameskip-v4") -> RICEConfig:
    """Get configuration optimized for Atari environments."""
    config = RICEConfig()
    config.environment.env_name = env_name
    config.training.n_steps = 128
    config.training.batch_size = 32
    config.training.learning_rate = 2.5e-4
    config.environment.normalize_obs = False
    return config


def load_config_from_file(file_path: str) -> RICEConfig:
    """Load configuration from JSON file."""
    import json
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return RICEConfig.from_dict(config_dict)


def save_config_to_file(config: RICEConfig, file_path: str) -> None:
    """Save configuration to JSON file."""
    import json
    with open(file_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)