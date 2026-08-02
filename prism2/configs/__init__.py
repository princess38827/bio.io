"""Configuration files for PRISM2 training and inference."""

from .model_config import ModelConfig, get_default_config
from .training_config import TrainingConfig

__all__ = ["ModelConfig", "TrainingConfig", "get_default_config"]
