"""
Perceiver-based slide encoder for PRISM2.

Aggregates thousands of tile embeddings into a fixed-size slide representation
using cross-attention mechanism inspired by Perceiver architecture.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PerceiverSlideEncoder(nn.Module):
    """
    Perceiver-based slide encoder that aggregates tile embeddings.
    
    Uses learned latent queries to aggregate variable-length sequences of
    tile embeddings into fixed-size slide representations.
    
    Args:
        tile_dim: Dimension of input tile embeddings
        num_latents: Number of latent query vectors
        latent_dim: Dimension of latent vectors
        num_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        tile_dim: int = 1280,
        num_latents: int = 256,
        latent_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tile_dim = tile_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        
        # Learned latent queries
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, latent_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(tile_dim, latent_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                query_dim=latent_dim,
                key_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Self-attention layers for latent processing
        self.self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(latent_dim)
    
    def forward(
        self,
        tile_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate tile embeddings into slide representation.
        
        Args:
            tile_embeddings: Tensor of shape (batch_size, num_tiles, tile_dim)
            attention_mask: Optional mask for valid tiles (batch_size, num_tiles)
        
        Returns:
            Slide embeddings of shape (batch_size, num_latents, latent_dim)
        """
        batch_size = tile_embeddings.size(0)
        
        # Project tile embeddings
        tile_features = self.input_proj(tile_embeddings)
        
        # Expand latent queries for batch
        latents = self.latent_queries.expand(batch_size, -1, -1)
        
        # Iteratively refine latents through cross-attention and self-attention
        for cross_attn, self_attn in zip(
            self.cross_attention_layers,
            self.self_attention_layers
        ):
            # Cross-attend to tile features
            latents = cross_attn(latents, tile_features, attention_mask)
            
            # Self-attend within latents
            latents = self_attn(latents)
        
        # Normalize output
        latents = self.output_norm(latents)
        
        return latents


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for perceiver."""
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(query_dim)
        self.ffn = FeedForward(query_dim, dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Cross-attention
        attn_out, _ = self.attention(query, key_value, key_value, key_padding_mask=mask)
        query = self.norm(query + attn_out)
        
        # Feed-forward
        query = self.ffn(query)
        
        return query


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for latent processing."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        # Feed-forward
        x = self.ffn(x)
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))
