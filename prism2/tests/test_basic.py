"""Basic tests for PRISM2 implementation."""

import torch
import sys
sys.path.insert(0, '/home/runner/work/bio.io/bio.io')

from prism2 import PRISM2Model
from prism2.configs import get_default_config, get_small_config


def test_model_creation():
    """Test basic model creation."""
    print("Test 1: Model creation...")
    
    config = get_small_config()  # Use smaller config for testing
    model = PRISM2Model(**config.to_dict())
    
    print(f"✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    

def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTest 2: Forward pass...")
    
    config = get_small_config()
    model = PRISM2Model(**config.to_dict())
    model.eval()
    
    # Create dummy tile embeddings
    batch_size = 2
    num_tiles = 100
    tile_dim = 1280
    
    tile_embeddings = torch.randn(batch_size, num_tiles, tile_dim)
    
    # Test base embedding
    with torch.no_grad():
        base_emb = model.get_base_embedding(tile_embeddings=tile_embeddings)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {tile_embeddings.shape}")
    print(f"  Base embedding shape: {base_emb.shape}")
    
    assert base_emb.shape == (batch_size, 512), "Base embedding shape mismatch"


def test_dual_embeddings():
    """Test dual embedding generation."""
    print("\nTest 3: Dual embeddings...")
    
    config = get_small_config()
    model = PRISM2Model(**config.to_dict())
    model.eval()
    
    tile_embeddings = torch.randn(1, 50, 1280)
    
    with torch.no_grad():
        outputs = model.forward(
            tile_embeddings=tile_embeddings,
            return_embeddings=True,
            return_loss=False
        )
    
    base_emb = outputs["base_embedding"]
    diagnostic_emb = outputs["diagnostic_embedding"]
    
    print(f"✓ Dual embeddings generated")
    print(f"  Base embedding: {base_emb.shape}")
    print(f"  Diagnostic embedding: {diagnostic_emb.shape}")
    
    assert base_emb.shape == (1, 512), "Base embedding shape mismatch"
    assert diagnostic_emb.shape == (1, 1024), "Diagnostic embedding shape mismatch"


def test_training_stage():
    """Test training stage setting."""
    print("\nTest 4: Training stages...")
    
    config = get_small_config()
    model = PRISM2Model(**config.to_dict())
    
    # Test stage 1
    model.set_training_stage(1)
    assert model.training_stage == 1
    print(f"✓ Stage 1 set")
    
    # Test stage 2
    model.set_training_stage(2)
    assert model.training_stage == 2
    print(f"✓ Stage 2 set (slide encoder frozen)")
    
    # Verify slide encoder is frozen
    for param in model.slide_encoder.parameters():
        assert not param.requires_grad, "Slide encoder should be frozen in stage 2"


def test_configurations():
    """Test configuration loading."""
    print("\nTest 5: Configurations...")
    
    default_config = get_default_config()
    small_config = get_small_config()
    
    print(f"✓ Default config loaded")
    print(f"  Slide encoder latents: {default_config.slide_encoder.num_latents}")
    print(f"  LM hidden dim: {default_config.language_model.hidden_dim}")
    
    print(f"✓ Small config loaded")
    print(f"  Slide encoder latents: {small_config.slide_encoder.num_latents}")
    print(f"  LM hidden dim: {small_config.language_model.hidden_dim}")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("PRISM2 Implementation Tests")
    print("="*60)
    
    try:
        test_model_creation()
        test_forward_pass()
        test_dual_embeddings()
        test_training_stage()
        test_configurations()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
