"""
UrbanGPT Model Configuration

This module contains all model configuration parameters for the UrbanGPT architecture,
including spatio-temporal encoder, alignment module, and LLM integration settings.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch


@dataclass
class SpatioTemporalEncoderConfig:
    """Configuration for the spatio-temporal dependency encoder."""
    
    # Dilated convolution parameters
    kernel_sizes: List[int] = None
    dilation_rates: List[int] = None
    hidden_dims: List[int] = None
    num_levels: int = 3
    
    # Gating mechanism
    use_gating: bool = True
    activation: str = "tanh"
    gate_activation: str = "sigmoid"
    
    # Dropout and regularization
    dropout_rate: float = 0.1
    batch_norm: bool = True
    
    def __post_init__(self):
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7]
        if self.dilation_rates is None:
            self.dilation_rates = [1, 2, 4, 8]
        if self.hidden_dims is None:
            self.hidden_dims = [128, 256, 512]


@dataclass
class AlignmentModuleConfig:
    """Configuration for the spatio-temporal-text alignment module."""
    
    # Projection layer parameters
    st_feature_dim: int = 512  # Output dimension from ST encoder
    llm_embedding_dim: int = 4096  # Vicuna-7B embedding dimension
    
    # Activation and regularization
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    
    # Learnable parameters
    use_bias: bool = True
    init_method: str = "xavier_uniform"


@dataclass
class LLMIntegrationConfig:
    """Configuration for LLM integration and regression head."""
    
    # Base LLM configuration
    model_name: str = "lmsys/vicuna-7b-v1.5"
    use_gradient_checkpointing: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Fine-tuning parameters
    freeze_base_model: bool = False
    lora_config: Optional[Dict[str, Any]] = None
    
    # Regression head
    regression_hidden_dim: int = 1024
    num_prediction_targets: int = 1
    regression_dropout: float = 0.1
    
    # Special tokens
    special_tokens: List[str] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ["<ST_start>", "<ST_HIS>", "<ST_end>", "<ST_PRE>"]
        
        if self.lora_config is None:
            self.lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }


@dataclass
class DataConfig:
    """Configuration for data preprocessing and formatting."""
    
    # Sequence parameters
    historical_steps: int = 12
    prediction_steps: int = 1
    
    # Spatial-temporal dimensions
    num_regions: int = 100  # Default, will be overridden by dataset
    num_features: int = 10  # Default, will be overridden by dataset
    
    # Sliding window
    window_stride: int = 1
    overlap_ratio: float = 0.0
    
    # Normalization
    normalize_features: bool = True
    normalization_method: str = "z_score"  # Options: "z_score", "min_max", "robust"
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Optimization
    learning_rate_encoder: float = 1e-4
    learning_rate_llm: float = 1e-5
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    max_steps: int = 50000
    
    # Batch configuration
    batch_size_train: int = 32
    batch_size_eval: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Training dynamics
    num_epochs: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 1e-4
    
    # Mixed precision
    use_fp16: bool = True
    use_bf16: bool = False


@dataclass
class InferenceConfig:
    """Configuration for inference and prediction."""
    
    # Batch processing
    batch_size: int = 64
    max_length: int = 512
    
    # Generation parameters (for text generation tasks)
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Zero-shot prediction
    enable_zero_shot: bool = True
    cross_region_threshold: float = 0.8
    
    # Output formatting
    return_attention_weights: bool = False
    return_hidden_states: bool = False


@dataclass
class UrbanGPTConfig:
    """Main configuration class combining all component configurations."""
    
    # Component configurations
    encoder: SpatioTemporalEncoderConfig = None
    alignment: AlignmentModuleConfig = None
    llm: LLMIntegrationConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    inference: InferenceConfig = None
    
    # Global settings
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    seed: int = 42
    experiment_name: str = "urbangpt_experiment"
    output_dir: str = "./outputs"
    
    # Logging and monitoring
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.encoder is None:
            self.encoder = SpatioTemporalEncoderConfig()
        if self.alignment is None:
            self.alignment = AlignmentModuleConfig()
        if self.llm is None:
            self.llm = LLMIntegrationConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        
        # Auto-detect device if needed
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "encoder": self.encoder.__dict__,
            "alignment": self.alignment.__dict__,
            "llm": self.llm.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "device": self.device,
            "seed": self.seed,
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "log_level": self.log_level,
            "wandb_project": self.wandb_project,
            "tensorboard_dir": self.tensorboard_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UrbanGPTConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update component configurations
        if "encoder" in config_dict:
            for key, value in config_dict["encoder"].items():
                setattr(config.encoder, key, value)
        
        if "alignment" in config_dict:
            for key, value in config_dict["alignment"].items():
                setattr(config.alignment, key, value)
        
        if "llm" in config_dict:
            for key, value in config_dict["llm"].items():
                setattr(config.llm, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                setattr(config.data, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                setattr(config.training, key, value)
        
        if "inference" in config_dict:
            for key, value in config_dict["inference"].items():
                setattr(config.inference, key, value)
        
        # Update global settings
        for key in ["device", "seed", "experiment_name", "output_dir", 
                   "log_level", "wandb_project", "tensorboard_dir"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config


# Default configuration instance
DEFAULT_CONFIG = UrbanGPTConfig()


def get_config(config_name: str = "default") -> UrbanGPTConfig:
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        UrbanGPTConfig instance
    """
    if config_name == "default":
        return DEFAULT_CONFIG
    else:
        raise ValueError(f"Unknown configuration name: {config_name}")


def create_experiment_config(
    experiment_name: str,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_epochs: int = 50,
    **kwargs
) -> UrbanGPTConfig:
    """
    Create a custom experiment configuration.
    
    Args:
        experiment_name: Name of the experiment
        batch_size: Training batch size
        learning_rate: Learning rate for encoder
        num_epochs: Number of training epochs
        **kwargs: Additional configuration parameters
        
    Returns:
        UrbanGPTConfig instance with custom settings
    """
    config = UrbanGPTConfig()
    config.experiment_name = experiment_name
    config.training.batch_size_train = batch_size
    config.training.learning_rate_encoder = learning_rate
    config.training.num_epochs = num_epochs
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to set in sub-configurations
            for sub_config_name in ["encoder", "alignment", "llm", "data", "training", "inference"]:
                sub_config = getattr(config, sub_config_name)
                if hasattr(sub_config, key):
                    setattr(sub_config, key, value)
                    break
    
    return config