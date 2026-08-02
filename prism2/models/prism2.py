"""
Main PRISM2 model implementation.

Combines all components into the full multimodal pathology foundation model
with dual-embedding architecture and prompt-based inference capabilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from .tile_encoder import TileEncoder
from .slide_encoder import PerceiverSlideEncoder
from .text_encoder import TextEncoder
from .language_model import LanguageModel
from .embeddings import DualEmbeddingHead


class PRISM2Model(nn.Module):
    """
    PRISM2: End-to-end multimodal pathology foundation model.
    
    Features:
    - Two-stage training approach
    - Dual-embedding architecture (base + diagnostic)
    - Prompt-based inference for clinical tasks
    - Trained on 2.3M WSIs and 14M QA pairs
    
    Args:
        tile_encoder_config: Configuration for tile encoder
        slide_encoder_config: Configuration for slide encoder
        text_encoder_config: Configuration for text encoder
        language_model_config: Configuration for language model
        embedding_config: Configuration for embedding heads
    """
    
    def __init__(
        self,
        tile_encoder_config: Optional[Dict[str, Any]] = None,
        slide_encoder_config: Optional[Dict[str, Any]] = None,
        text_encoder_config: Optional[Dict[str, Any]] = None,
        language_model_config: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Default configurations
        tile_encoder_config = tile_encoder_config or {}
        slide_encoder_config = slide_encoder_config or {}
        text_encoder_config = text_encoder_config or {}
        language_model_config = language_model_config or {}
        embedding_config = embedding_config or {}
        
        # Stage 1 components: Slide encoding and text alignment
        self.tile_encoder = TileEncoder(**tile_encoder_config)
        self.slide_encoder = PerceiverSlideEncoder(**slide_encoder_config)
        self.text_encoder = TextEncoder(**text_encoder_config)
        
        # Stage 2 component: Language model for dialogue
        self.language_model = LanguageModel(**language_model_config)
        
        # Dual embedding heads
        self.embedding_head = DualEmbeddingHead(**embedding_config)
        
        # Contrastive learning components
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # Training stage flag
        self.training_stage = 1  # 1 or 2
    
    def forward(
        self,
        tiles: Optional[torch.Tensor] = None,
        tile_embeddings: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        qa_input_ids: Optional[torch.Tensor] = None,
        qa_attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PRISM2.
        
        Args:
            tiles: Raw tile images (batch_size, num_tiles, channels, height, width)
            tile_embeddings: Pre-computed tile embeddings (batch_size, num_tiles, tile_dim)
            text_input_ids: Text token IDs for contrastive learning
            text_attention_mask: Text attention mask
            qa_input_ids: Question-answer token IDs for dialogue training
            qa_attention_mask: QA attention mask
            return_embeddings: Whether to return base and diagnostic embeddings
            return_loss: Whether to compute and return loss
        
        Returns:
            Dictionary containing outputs and optionally loss
        """
        outputs = {}
        
        # Encode tiles if not pre-computed
        if tile_embeddings is None:
            if tiles is None:
                raise ValueError("Either tiles or tile_embeddings must be provided")
            tile_embeddings = self.tile_encoder(tiles)
        
        # Aggregate tiles into slide representation
        slide_latents = self.slide_encoder(tile_embeddings)
        
        # Pool slide latents for embeddings
        slide_features = slide_latents.mean(dim=1)  # (batch_size, latent_dim)
        
        # Stage 1: Contrastive and autoregressive objectives
        if self.training_stage == 1 or text_input_ids is not None:
            # Text encoding for contrastive learning
            if text_input_ids is not None:
                text_embeddings = self.text_encoder(
                    text_input_ids,
                    text_attention_mask
                )
                
                # Contrastive loss
                if return_loss:
                    contrastive_loss = self._compute_contrastive_loss(
                        slide_features,
                        text_embeddings
                    )
                    outputs["contrastive_loss"] = contrastive_loss
            
            # Autoregressive text generation
            if qa_input_ids is not None:
                lm_outputs = self.language_model(
                    slide_features,
                    qa_input_ids,
                    qa_attention_mask,
                    return_hidden_states=True
                )
                
                outputs["logits"] = lm_outputs["logits"]
                
                # Autoregressive loss
                if return_loss:
                    autoregressive_loss = self._compute_autoregressive_loss(
                        lm_outputs["logits"],
                        qa_input_ids
                    )
                    outputs["autoregressive_loss"] = autoregressive_loss
                
                # Store hidden states for diagnostic embeddings
                outputs["lm_hidden_states"] = lm_outputs["hidden_states"]
        
        # Stage 2: Fine-tune language model with frozen slide encoder
        elif self.training_stage == 2:
            if qa_input_ids is not None:
                with torch.no_grad() if hasattr(self, 'freeze_slide_encoder') else torch.enable_grad():
                    # Slide encoder may be frozen in stage 2
                    pass
                
                lm_outputs = self.language_model(
                    slide_features,
                    qa_input_ids,
                    qa_attention_mask,
                    return_hidden_states=True
                )
                
                outputs["logits"] = lm_outputs["logits"]
                outputs["lm_hidden_states"] = lm_outputs["hidden_states"]
                
                if return_loss:
                    qa_loss = self._compute_autoregressive_loss(
                        lm_outputs["logits"],
                        qa_input_ids
                    )
                    outputs["qa_loss"] = qa_loss
        
        # Generate dual embeddings if requested
        if return_embeddings:
            if "lm_hidden_states" in outputs:
                base_emb, diagnostic_emb = self.embedding_head(
                    slide_latents,
                    outputs["lm_hidden_states"]
                )
            else:
                # Generate dummy LM hidden states for base embedding only
                lm_outputs = self.language_model(slide_features)
                dummy_hidden = lm_outputs["memory"]
                
                base_emb, diagnostic_emb = self.embedding_head(
                    slide_latents,
                    dummy_hidden
                )
            
            outputs["base_embedding"] = base_emb
            outputs["diagnostic_embedding"] = diagnostic_emb
        
        return outputs
    
    def _compute_contrastive_loss(
        self,
        slide_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between slide and text features.
        
        Uses InfoNCE loss with learnable temperature.
        """
        # Normalize features
        slide_features = nn.functional.normalize(slide_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(slide_features, text_features.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = slide_features.size(0)
        labels = torch.arange(batch_size, device=slide_features.device)
        
        # Symmetric loss
        loss_i2t = nn.functional.cross_entropy(logits, labels)
        loss_t2i = nn.functional.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def _compute_autoregressive_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute autoregressive language modeling loss.
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        # Flatten for cross-entropy
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100  # Ignore padding
        )
        
        return loss
    
    def get_base_embedding(
        self,
        tiles: Optional[torch.Tensor] = None,
        tile_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get base embedding for a slide.
        
        Best for biomarker prediction, survival analysis, and novel tasks.
        """
        outputs = self.forward(
            tiles=tiles,
            tile_embeddings=tile_embeddings,
            return_embeddings=True,
            return_loss=False
        )
        return outputs["base_embedding"]
    
    def get_diagnostic_embedding(
        self,
        tiles: Optional[torch.Tensor] = None,
        tile_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get diagnostic embedding for a slide.
        
        Best for cancer detection, subtyping, and clinical diagnostic tasks.
        """
        outputs = self.forward(
            tiles=tiles,
            tile_embeddings=tile_embeddings,
            return_embeddings=True,
            return_loss=False
        )
        return outputs["diagnostic_embedding"]
    
    def answer_question(
        self,
        tiles: Optional[torch.Tensor] = None,
        tile_embeddings: Optional[torch.Tensor] = None,
        question: str = "",
        max_length: int = 512,
        **generation_kwargs
    ) -> str:
        """
        Prompt-based inference: answer a clinical question about a slide.
        
        Args:
            tiles: Raw tile images
            tile_embeddings: Pre-computed tile embeddings
            question: Clinical question (e.g., "Is this prostate cancer?")
            max_length: Maximum answer length
            **generation_kwargs: Additional arguments for generation
        
        Returns:
            Generated answer text
        """
        # Encode tiles
        if tile_embeddings is None:
            tile_embeddings = self.tile_encoder(tiles)
        
        slide_latents = self.slide_encoder(tile_embeddings)
        slide_features = slide_latents.mean(dim=1)
        
        # Generate answer
        # Note: In practice, question would be tokenized first
        generated_ids = self.language_model.generate(
            slide_features,
            max_length=max_length,
            **generation_kwargs
        )
        
        # Decode to text (placeholder)
        # In practice, use tokenizer.decode(generated_ids)
        answer = f"Generated answer with {generated_ids.size(1)} tokens"
        
        return answer
    
    def set_training_stage(self, stage: int):
        """
        Set training stage (1 or 2).
        
        Stage 1: Train slide encoder with contrastive and autoregressive objectives
        Stage 2: Freeze slide encoder, fine-tune language model on QA
        """
        assert stage in [1, 2], "Stage must be 1 or 2"
        self.training_stage = stage
        
        if stage == 2:
            # Freeze slide encoder in stage 2
            for param in self.slide_encoder.parameters():
                param.requires_grad = False
            for param in self.tile_encoder.parameters():
                param.requires_grad = False
