"""Training script for PRISM2."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from ..models.prism2 import PRISM2Model
from ..configs import TrainingConfig


class PRISM2Trainer:
    """
    Trainer for PRISM2 model.
    
    Handles both Stage 1 (contrastive + autoregressive) and
    Stage 2 (QA fine-tuning) training.
    
    Args:
        model: PRISM2 model instance
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
    """
    
    def __init__(
        self,
        model: PRISM2Model,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup logging
        self.logger = logging.getLogger("PRISM2Trainer")
        
        # Training state
        self.current_stage = 1
        self.global_step = 0
        self.epoch = 0
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_stage1(self):
        """Train Stage 1: Slide encoder with contrastive and autoregressive objectives."""
        self.logger.info("Starting Stage 1 training...")
        self.model.set_training_stage(1)
        
        # Setup optimizer
        optimizer = self._create_optimizer(self.config.stage1.learning_rate)
        scheduler = self._create_scheduler(optimizer, self.config.stage1)
        
        self.model.train()
        
        for epoch in range(100):  # Adjust based on max_steps
            self.epoch = epoch
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                tiles = batch["tiles"].to(self.device)
                text_ids = batch["text_ids"].to(self.device)
                text_mask = batch["text_mask"].to(self.device)
                qa_ids = batch["qa_ids"].to(self.device) if "qa_ids" in batch else None
                qa_mask = batch["qa_mask"].to(self.device) if "qa_mask" in batch else None
                
                # Forward pass
                outputs = self.model(
                    tiles=tiles,
                    text_input_ids=text_ids,
                    text_attention_mask=text_mask,
                    qa_input_ids=qa_ids,
                    qa_attention_mask=qa_mask,
                    return_loss=True
                )
                
                # Compute total loss
                loss = (
                    self.config.stage1.contrastive_weight * outputs["contrastive_loss"] +
                    self.config.stage1.autoregressive_weight * outputs.get("autoregressive_loss", 0)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.stage1.gradient_clip
                )
                
                optimizer.step()
                scheduler.step()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.stage1.log_every == 0:
                    self._log_metrics({
                        "loss": loss.item(),
                        "contrastive_loss": outputs["contrastive_loss"].item(),
                        "autoregressive_loss": outputs.get("autoregressive_loss", torch.tensor(0)).item(),
                        "lr": scheduler.get_last_lr()[0],
                    })
                
                # Checkpointing
                if self.global_step % self.config.stage1.save_every == 0:
                    self._save_checkpoint(f"stage1_step{self.global_step}")
                
                # Evaluation
                if self.val_loader and self.global_step % self.config.stage1.eval_every == 0:
                    self._evaluate()
                
                if self.global_step >= self.config.stage1.max_steps:
                    self.logger.info("Reached max steps for Stage 1")
                    return
    
    def train_stage2(self):
        """Train Stage 2: Fine-tune language model with frozen slide encoder."""
        self.logger.info("Starting Stage 2 training...")
        self.model.set_training_stage(2)
        
        # Setup optimizer (only for language model parameters)
        optimizer = self._create_optimizer(
            self.config.stage2.learning_rate,
            filter_frozen=True
        )
        scheduler = self._create_scheduler(optimizer, self.config.stage2)
        
        self.model.train()
        self.global_step = 0  # Reset for stage 2
        
        for epoch in range(100):
            self.epoch = epoch
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                tiles = batch["tiles"].to(self.device)
                qa_ids = batch["qa_ids"].to(self.device)
                qa_mask = batch["qa_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    tiles=tiles,
                    qa_input_ids=qa_ids,
                    qa_attention_mask=qa_mask,
                    return_loss=True
                )
                
                loss = outputs["qa_loss"]
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.stage2.gradient_clip
                )
                
                optimizer.step()
                scheduler.step()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.stage2.log_every == 0:
                    self._log_metrics({
                        "qa_loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    })
                
                # Checkpointing
                if self.global_step % self.config.stage2.save_every == 0:
                    self._save_checkpoint(f"stage2_step{self.global_step}")
                
                # Evaluation
                if self.val_loader and self.global_step % self.config.stage2.eval_every == 0:
                    self._evaluate()
                
                if self.global_step >= self.config.stage2.max_steps:
                    self.logger.info("Reached max steps for Stage 2")
                    return
    
    def _create_optimizer(
        self,
        lr: float,
        filter_frozen: bool = False
    ) -> optim.Optimizer:
        """Create optimizer."""
        if filter_frozen:
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()
        
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.config.stage1.weight_decay
        )
    
    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        stage_config
    ):
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=stage_config.max_steps
        )
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        log_str = f"Step {self.global_step} - "
        log_str += " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(log_str)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{name}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "stage": self.current_stage,
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                tiles = batch["tiles"].to(self.device)
                # Evaluation logic here
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"Validation loss: {avg_loss:.4f}")
        
        self.model.train()
