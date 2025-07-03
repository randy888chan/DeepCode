"""
UrbanGPT Core Module

This module contains the core components of the UrbanGPT architecture:
- Spatio-temporal dependency encoder with gated dilated convolutions
- Spatio-temporal-text alignment module for feature projection
- LLM integration layer with regression head for prediction

The core module implements the mathematical formulations from the UrbanGPT paper:
- Gated Dilated Convolution: G(X) = σ(Conv1d(X, dilation=d)) ⊙ tanh(Conv1d(X, dilation=d))
- Multi-level Correlation: C_l = Σ(α_i * Conv1d(X, dilation=2^i))
- Projection Mapping: P(H_st) = W_p * H_st + b_p
- Regression Head: ŷ = W_r * h_hidden + b_r

Author: UrbanGPT Implementation Team
Version: 1.0.0
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor

# Version information
__version__ = "1.0.0"
__author__ = "UrbanGPT Implementation Team"

# Core component imports (will be available after implementation)
try:
    from .encoder import (
        SpatioTemporalEncoder,
        GatedDilatedConv1d,
        MultiLevelCorrelation,
        TemporalConvolutionalNetwork
    )
    _ENCODER_AVAILABLE = True
except ImportError:
    _ENCODER_AVAILABLE = False

try:
    from .alignment import (
        STTextAlignmentModule,
        ProjectionLayer,
        FeatureAlignment
    )
    _ALIGNMENT_AVAILABLE = True
except ImportError:
    _ALIGNMENT_AVAILABLE = False

try:
    from .llm_integration import (
        LLMIntegrationLayer,
        VicunaWrapper,
        RegressionHead,
        UrbanGPTModel
    )
    _LLM_INTEGRATION_AVAILABLE = True
except ImportError:
    _LLM_INTEGRATION_AVAILABLE = False

# Export lists for different availability states
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    
    # Availability flags
    "is_encoder_available",
    "is_alignment_available", 
    "is_llm_integration_available",
    "get_available_components",
    
    # Core factory functions
    "create_encoder",
    "create_alignment_module",
    "create_llm_integration",
    "create_full_model",
]

# Conditionally add imports to __all__
if _ENCODER_AVAILABLE:
    __all__.extend([
        "SpatioTemporalEncoder",
        "GatedDilatedConv1d", 
        "MultiLevelCorrelation",
        "TemporalConvolutionalNetwork"
    ])

if _ALIGNMENT_AVAILABLE:
    __all__.extend([
        "STTextAlignmentModule",
        "ProjectionLayer",
        "FeatureAlignment"
    ])

if _LLM_INTEGRATION_AVAILABLE:
    __all__.extend([
        "LLMIntegrationLayer",
        "VicunaWrapper",
        "RegressionHead", 
        "UrbanGPTModel"
    ])


def is_encoder_available() -> bool:
    """Check if encoder components are available."""
    return _ENCODER_AVAILABLE


def is_alignment_available() -> bool:
    """Check if alignment components are available."""
    return _ALIGNMENT_AVAILABLE


def is_llm_integration_available() -> bool:
    """Check if LLM integration components are available."""
    return _LLM_INTEGRATION_AVAILABLE


def get_available_components() -> Dict[str, bool]:
    """Get availability status of all core components."""
    return {
        "encoder": _ENCODER_AVAILABLE,
        "alignment": _ALIGNMENT_AVAILABLE,
        "llm_integration": _LLM_INTEGRATION_AVAILABLE
    }


def create_encoder(
    input_dim: int,
    hidden_dims: List[int] = [128, 256, 512],
    kernel_sizes: List[int] = [3, 5, 7],
    dilation_rates: List[int] = [1, 2, 4, 8],
    dropout_rate: float = 0.1,
    **kwargs
) -> Optional[nn.Module]:
    """
    Factory function to create spatio-temporal encoder.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: Hidden layer dimensions
        kernel_sizes: Convolution kernel sizes
        dilation_rates: Dilation rates for dilated convolutions
        dropout_rate: Dropout rate
        **kwargs: Additional arguments
        
    Returns:
        SpatioTemporalEncoder instance or None if not available
    """
    if not _ENCODER_AVAILABLE:
        raise ImportError("Encoder components not available. Please implement encoder.py first.")
    
    return SpatioTemporalEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        kernel_sizes=kernel_sizes,
        dilation_rates=dilation_rates,
        dropout_rate=dropout_rate,
        **kwargs
    )


def create_alignment_module(
    st_dim: int,
    llm_dim: int = 4096,
    activation: str = "relu",
    dropout_rate: float = 0.1,
    **kwargs
) -> Optional[nn.Module]:
    """
    Factory function to create ST-text alignment module.
    
    Args:
        st_dim: Spatio-temporal feature dimension
        llm_dim: LLM embedding dimension (4096 for Vicuna-7B)
        activation: Activation function name
        dropout_rate: Dropout rate
        **kwargs: Additional arguments
        
    Returns:
        STTextAlignmentModule instance or None if not available
    """
    if not _ALIGNMENT_AVAILABLE:
        raise ImportError("Alignment components not available. Please implement alignment.py first.")
    
    return STTextAlignmentModule(
        st_dim=st_dim,
        llm_dim=llm_dim,
        activation=activation,
        dropout_rate=dropout_rate,
        **kwargs
    )


def create_llm_integration(
    model_name: str = "vicuna-7b",
    hidden_dim: int = 4096,
    output_dim: int = 1,
    freeze_llm: bool = False,
    **kwargs
) -> Optional[nn.Module]:
    """
    Factory function to create LLM integration layer.
    
    Args:
        model_name: LLM model name
        hidden_dim: Hidden dimension
        output_dim: Output dimension for regression
        freeze_llm: Whether to freeze LLM parameters
        **kwargs: Additional arguments
        
    Returns:
        LLMIntegrationLayer instance or None if not available
    """
    if not _LLM_INTEGRATION_AVAILABLE:
        raise ImportError("LLM integration components not available. Please implement llm_integration.py first.")
    
    return LLMIntegrationLayer(
        model_name=model_name,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        freeze_llm=freeze_llm,
        **kwargs
    )


def create_full_model(
    config: Dict[str, Any]
) -> Optional[nn.Module]:
    """
    Factory function to create complete UrbanGPT model.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Complete UrbanGPT model or None if components not available
    """
    if not all([_ENCODER_AVAILABLE, _ALIGNMENT_AVAILABLE, _LLM_INTEGRATION_AVAILABLE]):
        missing = []
        if not _ENCODER_AVAILABLE:
            missing.append("encoder")
        if not _ALIGNMENT_AVAILABLE:
            missing.append("alignment")
        if not _LLM_INTEGRATION_AVAILABLE:
            missing.append("llm_integration")
        
        raise ImportError(f"Missing components: {', '.join(missing)}. Please implement all core modules first.")
    
    return UrbanGPTModel(config)


# Module-level configuration
DEFAULT_CONFIG = {
    "encoder": {
        "hidden_dims": [128, 256, 512],
        "kernel_sizes": [3, 5, 7],
        "dilation_rates": [1, 2, 4, 8],
        "dropout_rate": 0.1
    },
    "alignment": {
        "llm_dim": 4096,
        "activation": "relu",
        "dropout_rate": 0.1
    },
    "llm_integration": {
        "model_name": "vicuna-7b",
        "hidden_dim": 4096,
        "output_dim": 1,
        "freeze_llm": False
    }
}


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for UrbanGPT components."""
    return DEFAULT_CONFIG.copy()


# Validation functions
def validate_tensor_shapes(
    st_features: Tensor,
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> bool:
    """
    Validate tensor shapes for core components.
    
    Args:
        st_features: Input tensor
        expected_shape: Expected shape (use -1 for flexible dimensions)
        name: Tensor name for error messages
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    actual_shape = st_features.shape
    
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{name} shape mismatch: expected {len(expected_shape)} dimensions, "
            f"got {len(actual_shape)} dimensions"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} shape mismatch at dimension {i}: expected {expected}, got {actual}"
            )
    
    return True


# Component status summary
def print_component_status() -> None:
    """Print availability status of all core components."""
    status = get_available_components()
    print("UrbanGPT Core Components Status:")
    print("-" * 35)
    for component, available in status.items():
        status_str = "✓ Available" if available else "✗ Not Available"
        print(f"{component:15}: {status_str}")
    print("-" * 35)