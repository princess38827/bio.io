"""
Dual embedding outputs for PRISM2.

Provides both base embeddings (for general tasks) and diagnostic embeddings
(optimized for clinical diagnostic tasks).
"""

import torch
import torch.nn as nn
from typing import Tuple


class BaseEmbedding(nn.Module):
    """
    Base embedding layer for PRISM2.
    
    Lightweight embedding that transfers well to novel tasks like
    biomarker prediction and survival analysis.
    
    Args:
        latent_dim: Dimension of input latent vectors
        embedding_dim: Dimension of output base embeddings
        num_latents: Number of latent vectors to aggregate
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        embedding_dim: int = 512,
        num_latents: int = 256
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # Aggregation method: can be mean pooling, attention pooling, etc.
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Query vector for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        """
        Generate base embeddings from latent features.
        
        Args:
            latent_features: Latent features of shape (batch_size, num_latents, latent_dim)
        
        Returns:
            Base embeddings of shape (batch_size, embedding_dim)
        """
        batch_size = latent_features.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Attention pooling
        pooled, _ = self.attention_pool(query, latent_features, latent_features)
        pooled = pooled.squeeze(1)  # (batch_size, latent_dim)
        
        # Project to output dimension
        embeddings = self.output_proj(pooled)
        
        return embeddings


class DiagnosticEmbedding(nn.Module):
    """
    Diagnostic embedding layer for PRISM2.
    
    Derived from hidden states of the language model, optimized for
    clinical tasks like cancer detection and subtyping.
    
    Args:
        lm_hidden_dim: Hidden dimension of language model (e.g., 4B parameter Phi-3)
        embedding_dim: Dimension of output diagnostic embeddings
    """
    
    def __init__(
        self,
        lm_hidden_dim: int = 3072,
        embedding_dim: int = 1024
    ):
        super().__init__()
        self.lm_hidden_dim = lm_hidden_dim
        self.embedding_dim = embedding_dim
        
        # Projection from language model hidden states
        self.proj = nn.Sequential(
            nn.Linear(lm_hidden_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, lm_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Generate diagnostic embeddings from language model hidden states.
        
        Args:
            lm_hidden_states: Hidden states from language model
                             Shape: (batch_size, lm_hidden_dim) or (batch_size, seq_len, lm_hidden_dim)
        
        Returns:
            Diagnostic embeddings of shape (batch_size, embedding_dim)
        """
        # If we have a sequence, pool it
        if len(lm_hidden_states.shape) == 3:
            # Mean pooling over sequence dimension
            lm_hidden_states = lm_hidden_states.mean(dim=1)
        
        # Project to diagnostic embedding space
        embeddings = self.proj(lm_hidden_states)
        
        return embeddings


class DualEmbeddingHead(nn.Module):
    """
    Dual embedding head that produces both base and diagnostic embeddings.
    
    Args:
        latent_dim: Dimension of slide latent features
        lm_hidden_dim: Hidden dimension of language model
        base_dim: Dimension of base embeddings
        diagnostic_dim: Dimension of diagnostic embeddings
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        lm_hidden_dim: int = 3072,
        base_dim: int = 512,
        diagnostic_dim: int = 1024,
        num_latents: int = 256
    ):
        super().__init__()
        
        self.base_embedding = BaseEmbedding(
            latent_dim=latent_dim,
            embedding_dim=base_dim,
            num_latents=num_latents
        )
        
        self.diagnostic_embedding = DiagnosticEmbedding(
            lm_hidden_dim=lm_hidden_dim,
            embedding_dim=diagnostic_dim
        )
    
    def forward(
        self,
        latent_features: torch.Tensor,
        lm_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both base and diagnostic embeddings.
        
        Args:
            latent_features: Slide latent features (batch_size, num_latents, latent_dim)
            lm_hidden_states: Language model hidden states
        
        Returns:
            Tuple of (base_embeddings, diagnostic_embeddings)
        """
        base_emb = self.base_embedding(latent_features)
        diagnostic_emb = self.diagnostic_embedding(lm_hidden_states)
        
        return base_emb, diagnostic_emb
