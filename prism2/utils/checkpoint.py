"""Checkpoint saving and loading utilities."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        step: Current training step
        save_path: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint to
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    info = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
    }
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Epoch: {info['epoch']}, Step: {info['step']}")
    
    return info
