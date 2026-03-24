import pytest
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.text_cnn import TextCNN
def test_textcnn_initialization():
    """Test if the TextCNN model initializes correctly with given parameters."""
    vocab_size = 1000
    embed_dim = 50
    num_classes = 2
    kernel_sizes = [2, 3, 4]
    num_filters = 100
    
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters
    )
    
    assert model.embedding.num_embeddings == vocab_size
    assert model.embedding.embedding_dim == embed_dim
    assert len(model.convs) == len(kernel_sizes)
    assert model.fc.out_features == num_classes
    assert model.fc.in_features == len(kernel_sizes) * num_filters

def test_textcnn_forward_pass():
    """Test if the forward pass of the TextCNN produces the correct output shape."""
    vocab_size = 1000
    embed_dim = 50
    num_classes = 2
    batch_size = 16
    seq_length = 20
    
    model = TextCNN(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)
    
    # Create dummy input: random token IDs
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    logits = model(dummy_input)
    
    # Check output shape
    assert logits.shape == (batch_size, num_classes)
