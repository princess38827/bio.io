# bio.io

A repository for bioinformatics and computational biology projects.

## Projects

### PRISM2: Multimodal Pathology Foundation Model

An implementation of PRISM2, an end-to-end multimodal pathology foundation model with clinical dialogue capabilities.

**Key Features:**
- Dual-embedding architecture (base + diagnostic embeddings)
- Prompt-based inference for clinical questions
- Clinical-grade cancer detection performance
- Two-stage training with contrastive and autoregressive objectives

**Quick Start:**
```python
from prism2 import PRISM2Model
from prism2.configs import get_default_config

# Create model
config = get_default_config()
model = PRISM2Model(**config.to_dict())

# Get slide embeddings
embeddings = model.get_base_embedding(tile_embeddings=tiles)
```

See [prism2/README.md](prism2/README.md) for detailed documentation.

## Installation

```bash
git clone https://github.com/princess38827/bio.io.git
cd bio.io
pip install torch torchvision
```

## Structure

```
bio.io/
├── prism2/              # PRISM2 implementation
│   ├── models/          # Model architectures
│   ├── training/        # Training utilities
│   ├── configs/         # Configurations
│   ├── examples/        # Usage examples
│   └── README.md        # Detailed documentation
└── plugins/             # Additional plugins and tools
```

## License

[Specify license]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
