"""Training configuration for PRISM2."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Stage1Config:
    """Configuration for Stage 1 training."""
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_steps: int = 500000
    gradient_clip: float = 1.0
    
    # Loss weights
    contrastive_weight: float = 1.0
    autoregressive_weight: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    # Logging and checkpointing
    log_every: int = 100
    save_every: int = 5000
    eval_every: int = 1000


@dataclass
class Stage2Config:
    """Configuration for Stage 2 training (QA fine-tuning)."""
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_clip: float = 1.0
    
    # Freeze slide encoder
    freeze_slide_encoder: bool = True
    freeze_tile_encoder: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    
    # Logging and checkpointing
    log_every: int = 100
    save_every: int = 2000
    eval_every: int = 500


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    # Data paths
    wsi_data_path: str = "/path/to/wsi/data"
    report_data_path: str = "/path/to/reports"
    qa_data_path: str = "/path/to/qa_pairs"
    
    # Data processing
    num_workers: int = 8
    prefetch_factor: int = 2
    tile_size: int = 256
    max_tiles_per_slide: int = 10000
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5


@dataclass
class TrainingConfig:
    """Complete training configuration for PRISM2."""
    # Training stages
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    
    # Data configuration
    data: DataConfig = DataConfig()
    
    # General settings
    seed: int = 42
    mixed_precision: bool = True
    distributed: bool = True
    num_gpus: int = 8
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    
    # Logging
    wandb_project: Optional[str] = "prism2"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_gpus > 0, "num_gpus must be positive"
        assert self.stage1.batch_size > 0, "batch_size must be positive"
        assert self.stage2.batch_size > 0, "batch_size must be positive"
