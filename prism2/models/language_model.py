"""
Language model component for PRISM2.

Uses Phi-3 Mini (4B parameters) for generating pathology report text
and answering clinical questions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class LanguageModel(nn.Module):
    """
    Language model for PRISM2 based on Phi-3 Mini.
    
    Generates pathology report text and answers clinical questions
    based on slide representations.
    
    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Hidden dimension (3072 for Phi-3 Mini)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_length: Maximum sequence length
        slide_dim: Dimension of input slide embeddings
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_dim: int = 3072,
        num_layers: int = 32,
        num_heads: int = 32,
        max_length: int = 2048,
        slide_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Slide embedding projection to LM space
        self.slide_proj = nn.Sequential(
            nn.Linear(slide_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output head
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        slide_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text based on slide features.
        
        Args:
            slide_features: Slide embeddings (batch_size, slide_dim)
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            return_hidden_states: Whether to return hidden states
        
        Returns:
            Dictionary containing logits and optionally hidden states
        """
        batch_size = slide_features.size(0)
        
        # Project slide features to LM space
        slide_memory = self.slide_proj(slide_features)
        slide_memory = slide_memory.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        if input_ids is not None:
            seq_length = input_ids.size(1)
            
            # Create position IDs
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Embed tokens and positions
            token_embeds = self.token_embedding(input_ids)
            position_embeds = self.position_embedding(position_ids)
            
            # Combine embeddings
            embeddings = self.dropout(token_embeds + position_embeds)
            
            # Create causal mask
            causal_mask = self._generate_causal_mask(seq_length, input_ids.device)
            
            # Create padding mask
            if attention_mask is not None:
                padding_mask = (attention_mask == 0)
            else:
                padding_mask = None
            
            # Decode
            hidden_states = self.transformer(
                tgt=embeddings,
                memory=slide_memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask
            )
            
            hidden_states = self.norm(hidden_states)
            
            # Generate logits
            logits = self.output_head(hidden_states)
            
            output = {"logits": logits}
            if return_hidden_states:
                output["hidden_states"] = hidden_states
            
            return output
        else:
            # Return slide memory for generation
            return {"memory": slide_memory}
    
    def _generate_causal_mask(
        self,
        seq_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device),
            diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate(
        self,
        slide_features: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            slide_features: Slide embeddings (batch_size, slide_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
        
        Returns:
            Generated token IDs (batch_size, generated_length)
        """
        batch_size = slide_features.size(0)
        device = slide_features.device
        
        # Start with BOS token (assume token 1)
        input_ids = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Forward pass
            outputs = self.forward(slide_features, input_ids)
            logits = outputs["logits"]
            
            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS token (assume token 2)
            if (next_token == 2).all():
                break
        
        return input_ids
