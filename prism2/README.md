# PRISM2: End-to-end Multimodal Pathology Foundation Model

A PyTorch implementation of PRISM2, a multimodal slide-level foundation model for computational pathology with clinical dialogue capabilities.

## Overview

PRISM2 is a foundation model trained on 2.3 million whole-slide images (WSIs) and 14 million question-answer pairs derived from pathology reports. It features:

- **Dual-embedding architecture**: Base embeddings for general tasks (biomarkers, survival) and diagnostic embeddings for clinical tasks (cancer detection, subtyping)
- **Prompt-based inference**: Answer clinical questions without task-specific training
- **Two-stage training**: Contrastive + autoregressive learning followed by dialogue fine-tuning
- **Clinical-grade performance**: Matches or exceeds specialized commercial models

## Architecture

### Key Components

1. **Tile Encoder** (Virchow2-based)
   - Processes individual histopathology image tiles
   - Embedding dimension: 1280
   - Frozen during training

2. **Slide Encoder** (Perceiver-based)
   - Aggregates thousands of tile embeddings using cross-attention
   - 256 latent queries, 512-dimensional latent space
   - 6 layers of cross and self-attention

3. **Text Encoder** (BioGPT-based)
   - Encodes pathology report text
   - Used for contrastive alignment with slide representations

4. **Language Model** (Phi-3 Mini)
   - 4 billion parameters, 3072 hidden dimensions
   - Generates diagnostic text and answers clinical questions
   - 32 transformer layers

5. **Dual Embedding Heads**
   - Base embedding: 512-dim, general-purpose representation
   - Diagnostic embedding: 1024-dim, optimized for clinical tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/princess38827/bio.io.git
cd bio.io

# Install dependencies
pip install torch torchvision
pip install -r requirements.txt  # If available
```

## Quick Start

### Basic Inference

```python
import torch
from prism2 import PRISM2Model
from prism2.configs import get_default_config

# Create model
config = get_default_config()
model = PRISM2Model(**config.to_dict())
model.eval()

# Example: Get base embedding from tile embeddings
tile_embeddings = torch.randn(1, 1000, 1280)  # (batch, num_tiles, dim)

with torch.no_grad():
    base_emb = model.get_base_embedding(tile_embeddings=tile_embeddings)
    print(f"Base embedding shape: {base_emb.shape}")  # (1, 512)
```

### Prompt-Based Inference

```python
# Ask clinical questions
question = "Is this prostate cancer?"
answer = model.answer_question(
    tile_embeddings=tile_embeddings,
    question=question,
    max_length=128
)
print(f"Answer: {answer}")
```

### Downstream Task (Linear Probe)

```python
import torch.nn as nn

# Freeze PRISM2
for param in model.parameters():
    param.requires_grad = False

# Add task-specific head
classifier = nn.Linear(512, num_classes)

# Get embeddings
base_emb = model.get_base_embedding(tile_embeddings=tile_embeddings)

# Train classifier
logits = classifier(base_emb)
```

## Training

### Two-Stage Training Pipeline

**Stage 1: Slide Encoder Training**
- Contrastive loss: Align slide and text representations
- Autoregressive loss: Generate pathology report text
- Duration: 500K steps

**Stage 2: Language Model Fine-Tuning**
- Freeze slide encoder
- Fine-tune language model on question-answer pairs
- Duration: 100K steps

### Example Training Script

```python
from prism2.training import PRISM2Trainer
from prism2.configs import TrainingConfig

# Create configuration
config = TrainingConfig()
config.stage1.batch_size = 32
config.stage1.learning_rate = 1e-4

# Create trainer
trainer = PRISM2Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)

# Train Stage 1
trainer.train_stage1()

# Train Stage 2
trainer.train_stage2()
```

See `prism2/examples/training_example.py` for complete examples.

## Model Configuration

### Default Configuration

```python
from prism2.configs import get_default_config

config = get_default_config()
# Tile encoder: 1280-dim
# Slide encoder: 256 latents, 512-dim, 6 layers
# Language model: 3072-dim, 32 layers
```

### Custom Configuration

```python
from prism2.configs import ModelConfig, SlideEncoderConfig

config = ModelConfig()
config.slide_encoder.num_latents = 128  # Reduce latents
config.slide_encoder.num_layers = 3     # Fewer layers

model = PRISM2Model(**config.to_dict())
```

## Use Cases

### 1. Cancer Detection
```python
# Use diagnostic embeddings for best performance
diagnostic_emb = model.get_diagnostic_embedding(tile_embeddings=tiles)

# Or use prompt-based inference
answer = model.answer_question(tiles, "Is this cancer?")
```

### 2. Biomarker Prediction
```python
# Use base embeddings for best performance
base_emb = model.get_base_embedding(tile_embeddings=tiles)

# Train linear probe for specific biomarker
classifier = nn.Linear(512, 2)  # Binary classification
```

### 3. Survival Prediction
```python
# Base embeddings recommended
base_emb = model.get_base_embedding(tile_embeddings=tiles)

# Fine-tune for survival task or use linear probe
```

### 4. Report Generation
```python
# Generate pathology report
generated_report = model.language_model.generate(
    slide_features=slide_features,
    max_length=512
)
```

## Performance

PRISM2 achieves or exceeds clinical-grade performance:

- **Prostate Cancer Detection**: Matches Paige Prostate (clinical-grade)
- **Breast Cancer Detection**: Matches Paige Breast (clinical-grade)
- **Breast Lymph Node**: Outperforms Paige BLN (clinical-grade)
- **Pan-Cancer Detection**: 0.967 AUC (linear probe)
- **CAMELYON17**: Superior to baseline models
- **PANDA Grading**: Strong performance on external data

## Project Structure

```
prism2/
├── models/
│   ├── prism2.py              # Main model
│   ├── tile_encoder.py        # Tile encoding
│   ├── slide_encoder.py       # Perceiver aggregation
│   ├── text_encoder.py        # Text encoding
│   ├── language_model.py      # LM for generation
│   └── embeddings.py          # Dual embeddings
├── training/
│   ├── trainer.py             # Training loop
│   └── losses.py              # Loss functions
├── configs/
│   ├── model_config.py        # Model configurations
│   └── training_config.py     # Training configurations
├── examples/
│   ├── inference_examples.py  # Inference demos
│   └── training_example.py    # Training demos
└── utils/                     # Utilities
```

## Citation

If you use this implementation, please cite the original PRISM2 paper:

```bibtex
@article{prism2,
  title={End-to-end multimodal pathology foundation model with clinical dialogue},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Specify license here]

## Acknowledgments

This implementation is based on the PRISM2 paper. Key foundation models:
- Virchow2 for tile encoding
- BioGPT for text encoding  
- Phi-3 Mini for language modeling
- Perceiver architecture for slide aggregation

## Contact

For questions or issues, please open an issue on GitHub.
