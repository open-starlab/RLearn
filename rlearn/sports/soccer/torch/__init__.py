"""
rlearn/sports/soccer/torch/__init__.py

This module serves as the entry point for the PyTorch-based components of our reinforcement
learning package for soccer analytics. It aggregates model classes, training functions, and 
utility methods into a clean, importable namespace. Global configuration variables and logging 
settings are also initialized here.
"""

import logging
import os

# Global soccer field parameters (in meters)
FIELD_LENGTH = 105  # Standard soccer field length
FIELD_WIDTH = 68    # Standard soccer field width

# Package version
__version__ = "1.1.0"

# Configure logging:
# Allow overriding of log level via environment variable 'RL_TORCH_LOG_LEVEL'
LOG_LEVEL = os.environ.get("RL_TORCH_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Initializing rlearn.sports.soccer.torch package, version %s", __version__)

# Import necessary modules and classes
from .model import TorchModel
from .training import train_model
from .utils import some_util_function, enhanced_util_function  # Extended utility functions as an example

# Explicitly define what will be accessible when the package is imported
__all__ = [
    "TorchModel",
    "train_model",
    "some_util_function",
    "enhanced_util_function",
    "FIELD_LENGTH",
    "FIELD_WIDTH",
]
