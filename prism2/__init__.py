"""
PRISM2: End-to-end multimodal pathology foundation model with clinical dialogue

A slide-level foundation model trained on whole-slide images and question-answer pairs
derived from pathology reports. Features dual-embedding architecture and prompt-based inference.
"""

__version__ = "0.1.0"
__author__ = "bio.io"

from .models.prism2 import PRISM2Model
from .models.embeddings import BaseEmbedding, DiagnosticEmbedding

__all__ = [
    "PRISM2Model",
    "BaseEmbedding",
    "DiagnosticEmbedding",
]
