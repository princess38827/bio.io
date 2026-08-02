"""Model configuration for PRISM2."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TileEncoderConfig:
    """Configuration for tile encoder (Virchow2-based)."""
    embedding_dim: int = 1280
    pretrained: bool = True
    freeze: bool = True


@dataclass
class SlideEncoderConfig:
    """Configuration for perceiver-based slide encoder."""
    tile_dim: int = 1280
    num_latents: int = 256
    latent_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class TextEncoderConfig:
    """Configuration for text encoder (BioGPT-based)."""
    vocab_size: int = 50000
    embedding_dim: int = 768
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_length: int = 512
    dropout: float = 0.1


@dataclass
class LanguageModelConfig:
    """Configuration for language model (Phi-3 Mini)."""
    vocab_size: int = 50000
    hidden_dim: int = 3072
    num_layers: int = 32
    num_heads: int = 32
    max_length: int = 2048
    slide_dim: int = 512
    dropout: float = 0.1


@dataclass
class EmbeddingConfig:
    """Configuration for dual embedding heads."""
    latent_dim: int = 512
    lm_hidden_dim: int = 3072
    base_dim: int = 512
    diagnostic_dim: int = 1024
    num_latents: int = 256


@dataclass
class ModelConfig:
    """Complete PRISM2 model configuration."""
    tile_encoder: TileEncoderConfig = TileEncoderConfig()
    slide_encoder: SlideEncoderConfig = SlideEncoderConfig()
    text_encoder: TextEncoderConfig = TextEncoderConfig()
    language_model: LanguageModelConfig = LanguageModelConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "tile_encoder_config": {
                "embedding_dim": self.tile_encoder.embedding_dim,
                "pretrained": self.tile_encoder.pretrained,
                "freeze": self.tile_encoder.freeze,
            },
            "slide_encoder_config": {
                "tile_dim": self.slide_encoder.tile_dim,
                "num_latents": self.slide_encoder.num_latents,
                "latent_dim": self.slide_encoder.latent_dim,
                "num_layers": self.slide_encoder.num_layers,
                "num_heads": self.slide_encoder.num_heads,
                "dropout": self.slide_encoder.dropout,
            },
            "text_encoder_config": {
                "vocab_size": self.text_encoder.vocab_size,
                "embedding_dim": self.text_encoder.embedding_dim,
                "hidden_dim": self.text_encoder.hidden_dim,
                "num_layers": self.text_encoder.num_layers,
                "num_heads": self.text_encoder.num_heads,
                "max_length": self.text_encoder.max_length,
                "dropout": self.text_encoder.dropout,
            },
            "language_model_config": {
                "vocab_size": self.language_model.vocab_size,
                "hidden_dim": self.language_model.hidden_dim,
                "num_layers": self.language_model.num_layers,
                "num_heads": self.language_model.num_heads,
                "max_length": self.language_model.max_length,
                "slide_dim": self.language_model.slide_dim,
                "dropout": self.language_model.dropout,
            },
            "embedding_config": {
                "latent_dim": self.embedding.latent_dim,
                "lm_hidden_dim": self.embedding.lm_hidden_dim,
                "base_dim": self.embedding.base_dim,
                "diagnostic_dim": self.embedding.diagnostic_dim,
                "num_latents": self.embedding.num_latents,
            },
        }


def get_default_config() -> ModelConfig:
    """Get default PRISM2 configuration."""
    return ModelConfig()


def get_small_config() -> ModelConfig:
    """Get smaller configuration for testing/debugging."""
    config = ModelConfig()
    
    # Reduce model sizes
    config.slide_encoder.num_latents = 64
    config.slide_encoder.num_layers = 3
    config.text_encoder.num_layers = 6
    config.language_model.num_layers = 12
    config.language_model.hidden_dim = 1024
    
    return config
