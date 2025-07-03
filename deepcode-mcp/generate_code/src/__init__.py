"""
UrbanGPT: Spatio-Temporal Large Language Models for Urban Prediction

This package implements the UrbanGPT architecture for spatio-temporal urban prediction
using large language models with instruction tuning.

Core Components:
- Spatio-Temporal Dependency Encoder
- ST-Text Alignment Module  
- LLM Integration Layer
- Zero-Shot Prediction Engine

Author: UrbanGPT Implementation Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "UrbanGPT Implementation Team"
__email__ = "urbangpt@example.com"

# Core module imports
from .core import encoder, alignment, llm_integration
from .data import preprocessing, instruction_builder
from .training import trainer, loss_functions
from .inference import predictor
from .utils import tokens, metrics

# Main classes for easy access
from .core.encoder import SpatioTemporalEncoder
from .core.alignment import STTextAlignment
from .core.llm_integration import UrbanGPTModel
from .data.preprocessing import UrbanDataProcessor
from .data.instruction_builder import InstructionBuilder
from .training.trainer import UrbanGPTTrainer
from .inference.predictor import ZeroShotPredictor

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Core modules
    "encoder",
    "alignment", 
    "llm_integration",
    "preprocessing",
    "instruction_builder",
    "trainer",
    "loss_functions",
    "predictor",
    "tokens",
    "metrics",
    
    # Main classes
    "SpatioTemporalEncoder",
    "STTextAlignment",
    "UrbanGPTModel",
    "UrbanDataProcessor",
    "InstructionBuilder",
    "UrbanGPTTrainer",
    "ZeroShotPredictor",
]

# Package-level configuration
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"UrbanGPT v{__version__} initialized")

# Validate dependencies
def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'torch',
        'transformers', 
        'numpy',
        'pandas',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.warning("Install with: pip install -r requirements.txt")
    else:
        logger.info("All dependencies satisfied")
    
    return len(missing_packages) == 0

# Check dependencies on import
check_dependencies()