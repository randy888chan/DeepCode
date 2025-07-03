"""
UrbanGPT Data Module

This module provides data preprocessing and instruction building functionality
for spatio-temporal urban data processing and LLM integration.

Components:
- preprocessing: Data preprocessing and tensor formatting
- instruction_builder: Natural language instruction generation
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# Import core components (lazy loading to avoid circular imports)
def get_preprocessor():
    """Get data preprocessor instance."""
    try:
        from .preprocessing import SpatioTemporalPreprocessor
        return SpatioTemporalPreprocessor
    except ImportError as e:
        logger.error(f"Failed to import SpatioTemporalPreprocessor: {e}")
        raise

def get_instruction_builder():
    """Get instruction builder instance."""
    try:
        from .instruction_builder import InstructionBuilder
        return InstructionBuilder
    except ImportError as e:
        logger.error(f"Failed to import InstructionBuilder: {e}")
        raise

# Data module configuration
DATA_CONFIG = {
    "default_sequence_length": 12,
    "default_prediction_steps": 1,
    "supported_data_formats": ["tensor", "numpy", "pandas"],
    "max_regions": 1000,
    "max_features": 100,
    "instruction_templates": {
        "temporal": "Based on the historical spatio-temporal data from {start_time} to {end_time}, predict the {target_variable} for the next time step.",
        "spatial": "Given the spatio-temporal patterns across {num_regions} regions, forecast the {target_variable} values.",
        "zero_shot": "Using the learned spatio-temporal patterns, predict {target_variable} for the unseen region/city."
    }
}

# Validation functions
def validate_data_shape(data: Union[List, Tuple], expected_dims: int = 3) -> bool:
    """
    Validate spatio-temporal data shape.
    
    Args:
        data: Input data to validate
        expected_dims: Expected number of dimensions (default: 3 for R×T×F)
        
    Returns:
        bool: True if shape is valid
    """
    try:
        import torch
        import numpy as np
        
        if isinstance(data, (list, tuple)):
            return len(data) > 0
        elif isinstance(data, torch.Tensor):
            return len(data.shape) == expected_dims
        elif isinstance(data, np.ndarray):
            return len(data.shape) == expected_dims
        else:
            return False
    except Exception as e:
        logger.warning(f"Data shape validation failed: {e}")
        return False

def validate_instruction_format(instruction: str) -> bool:
    """
    Validate instruction format for LLM processing.
    
    Args:
        instruction: Instruction string to validate
        
    Returns:
        bool: True if format is valid
    """
    if not isinstance(instruction, str) or len(instruction.strip()) == 0:
        return False
    
    # Check for required spatio-temporal tokens
    required_tokens = ["<ST_start>", "<ST_end>"]
    return any(token in instruction for token in required_tokens)

# Utility functions
def get_data_statistics(data: Union[List, Tuple, Any]) -> Dict[str, Any]:
    """
    Get basic statistics of spatio-temporal data.
    
    Args:
        data: Input spatio-temporal data
        
    Returns:
        Dict containing data statistics
    """
    try:
        import torch
        import numpy as np
        
        stats = {
            "type": type(data).__name__,
            "shape": None,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "std_value": None
        }
        
        if isinstance(data, torch.Tensor):
            stats["shape"] = list(data.shape)
            stats["min_value"] = float(data.min())
            stats["max_value"] = float(data.max())
            stats["mean_value"] = float(data.mean())
            stats["std_value"] = float(data.std())
        elif isinstance(data, np.ndarray):
            stats["shape"] = list(data.shape)
            stats["min_value"] = float(np.min(data))
            stats["max_value"] = float(np.max(data))
            stats["mean_value"] = float(np.mean(data))
            stats["std_value"] = float(np.std(data))
        elif isinstance(data, (list, tuple)):
            stats["shape"] = [len(data)]
            if len(data) > 0 and isinstance(data[0], (int, float)):
                stats["min_value"] = min(data)
                stats["max_value"] = max(data)
                stats["mean_value"] = sum(data) / len(data)
        
        return stats
    except Exception as e:
        logger.error(f"Failed to compute data statistics: {e}")
        return {"error": str(e)}

def create_data_loader_config(
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, Any]:
    """
    Create data loader configuration.
    
    Args:
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Dict containing data loader configuration
    """
    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,  # Drop incomplete batches
        "persistent_workers": num_workers > 0
    }

# Export public API
__all__ = [
    "get_preprocessor",
    "get_instruction_builder",
    "DATA_CONFIG",
    "validate_data_shape",
    "validate_instruction_format",
    "get_data_statistics",
    "create_data_loader_config"
]

# Module initialization
def initialize_data_module(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize data module with optional configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        bool: True if initialization successful
    """
    try:
        if config:
            DATA_CONFIG.update(config)
        
        logger.info("Data module initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize data module: {e}")
        return False

# Auto-initialize on import
if not initialize_data_module():
    logger.warning("Data module initialization completed with warnings")