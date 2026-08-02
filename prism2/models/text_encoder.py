"""
Text encoder for PRISM2.

Encodes pathology report text using BioGPT for contrastive alignment.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class TextEncoder(nn.Module):
    """
    Text encoder based on BioGPT for encoding pathology reports.
    
    Processes diagnostic text into embeddings for contrastive learning
    with slide representations.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of text embeddings
        hidden_dim: Hidden dimension for transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection for contrastive learning
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
        
        Returns:
            Text embeddings of shape (batch_size, embedding_dim)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = self.dropout(token_embeds + position_embeds)
        
        # Create padding mask for transformer
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None
        
        # Transform
        hidden_states = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask
        )
        
        # Pool to single vector (use [CLS] token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)
        
        # Project to output dimension
        output = self.output_proj(pooled)
        output = self.norm(output)
        
        return output
