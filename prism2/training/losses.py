"""Loss functions for PRISM2 training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for image-text alignment.
    
    Uses InfoNCE objective with learnable temperature parameter.
    
    Args:
        temperature: Initial temperature value
        learnable: Whether temperature is learnable
    """
    
    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.ones([]) * temperature)
        else:
            self.register_buffer("temperature", torch.tensor(temperature))
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            image_features: Normalized image features (batch_size, dim)
            text_features: Normalized text features (batch_size, dim)
        
        Returns:
            Scalar contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Symmetric contrastive loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2


class AutoregressiveLoss(nn.Module):
    """
    Autoregressive language modeling loss.
    
    Standard next-token prediction loss for text generation.
    
    Args:
        ignore_index: Index to ignore in loss computation (padding)
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute autoregressive loss.
        
        Args:
            logits: Model logits (batch_size, seq_length, vocab_size)
            target_ids: Target token IDs (batch_size, seq_length)
        
        Returns:
            Scalar autoregressive loss
        """
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        # Flatten and compute cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index
        )
        
        return loss


class DualObjectiveLoss(nn.Module):
    """
    Combined loss for Stage 1 training.
    
    Combines contrastive and autoregressive objectives.
    
    Args:
        contrastive_weight: Weight for contrastive loss
        autoregressive_weight: Weight for autoregressive loss
        temperature: Initial temperature for contrastive loss
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        autoregressive_weight: float = 1.0,
        temperature: float = 0.07
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.autoregressive_weight = autoregressive_weight
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.autoregressive_loss = AutoregressiveLoss()
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            image_features: Image features for contrastive learning
            text_features: Text features for contrastive learning
            logits: Model logits for autoregressive learning
            target_ids: Target token IDs
        
        Returns:
            Dictionary with individual and total losses
        """
        contrastive = self.contrastive_loss(image_features, text_features)
        autoregressive = self.autoregressive_loss(logits, target_ids)
        
        total = (
            self.contrastive_weight * contrastive +
            self.autoregressive_weight * autoregressive
        )
        
        return {
            "contrastive_loss": contrastive,
            "autoregressive_loss": autoregressive,
            "total_loss": total,
        }
