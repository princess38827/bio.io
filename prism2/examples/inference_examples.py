"""Example usage of PRISM2 model for inference."""

import torch
from prism2 import PRISM2Model
from prism2.configs import get_default_config

def example_basic_inference():
    """Example: Basic model inference with tile embeddings."""
    
    # Create model with default configuration
    config = get_default_config()
    model = PRISM2Model(**config.to_dict())
    model.eval()
    
    # Example input: pre-computed tile embeddings from Virchow2
    batch_size = 2
    num_tiles = 1000
    tile_dim = 1280
    
    # Random tile embeddings (in practice, from Virchow2)
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    
    # Get base embeddings (best for biomarkers, survival)
    with torch.no_grad():
        base_emb = model.get_base_embedding(tile_embeddings=tile_embeddings)
        print(f"Base embedding shape: {base_emb.shape}")
        # Output: torch.Size([2, 512])
    
    # Get diagnostic embeddings (best for cancer detection)
    with torch.no_grad():
        outputs = model.forward(
            tile_embeddings=tile_embeddings,
            return_embeddings=True,
            return_loss=False
        )
        diagnostic_emb = outputs["diagnostic_embedding"]
        print(f"Diagnostic embedding shape: {diagnostic_emb.shape}")
        # Output: torch.Size([2, 1024])


def example_prompt_based_inference():
    """Example: Prompt-based inference for clinical questions."""
    
    config = get_default_config()
    model = PRISM2Model(**config.to_dict())
    model.eval()
    
    # Example tile embeddings
    tile_embeddings = torch.randn(1, 500, 1280)
    
    # Ask clinical question
    question = "Is this prostate cancer?"
    
    with torch.no_grad():
        answer = model.answer_question(
            tile_embeddings=tile_embeddings,
            question=question,
            max_length=128
        )
        print(f"Question: {question}")
        print(f"Answer: {answer}")


def example_batch_processing():
    """Example: Process multiple slides in batch."""
    
    config = get_default_config()
    model = PRISM2Model(**config.to_dict())
    model.eval()
    
    # Simulate batch of slides with varying tile counts
    slides = [
        torch.randn(800, 1280),   # Slide 1: 800 tiles
        torch.randn(1200, 1280),  # Slide 2: 1200 tiles
        torch.randn(500, 1280),   # Slide 3: 500 tiles
    ]
    
    base_embeddings = []
    
    with torch.no_grad():
        for slide_tiles in slides:
            # Add batch dimension
            slide_tiles = slide_tiles.unsqueeze(0)
            
            # Get embedding
            emb = model.get_base_embedding(tile_embeddings=slide_tiles)
            base_embeddings.append(emb)
    
    # Stack all embeddings
    all_embeddings = torch.cat(base_embeddings, dim=0)
    print(f"All embeddings shape: {all_embeddings.shape}")
    # Output: torch.Size([3, 512])
    
    # Can now use for downstream tasks
    # e.g., linear probe for cancer detection


def example_downstream_task():
    """Example: Use embeddings for downstream classification."""
    
    import torch.nn as nn
    
    config = get_default_config()
    prism2 = PRISM2Model(**config.to_dict())
    prism2.eval()
    
    # Freeze PRISM2 (linear probe)
    for param in prism2.parameters():
        param.requires_grad = False
    
    # Create downstream classifier
    num_classes = 10  # e.g., 10 cancer types
    classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes)
    )
    
    # Example training loop (pseudocode)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Get embeddings from PRISM2
    tile_embeddings = torch.randn(4, 1000, 1280)
    labels = torch.randint(0, num_classes, (4,))
    
    with torch.no_grad():
        base_emb = prism2.get_base_embedding(tile_embeddings=tile_embeddings)
    
    # Train classifier
    logits = classifier(base_emb)
    loss = criterion(logits, labels)
    
    print(f"Classification loss: {loss.item():.4f}")


if __name__ == "__main__":
    print("=== Example 1: Basic Inference ===")
    example_basic_inference()
    
    print("\n=== Example 2: Prompt-Based Inference ===")
    example_prompt_based_inference()
    
    print("\n=== Example 3: Batch Processing ===")
    example_batch_processing()
    
    print("\n=== Example 4: Downstream Task ===")
    example_downstream_task()
