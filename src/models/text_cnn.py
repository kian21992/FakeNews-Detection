import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, 
                 kernel_sizes: list[int] = [2, 3, 4], num_filters: int = 100, 
                 dropout_p: float = 0.5, padding_idx: int = 0):
        """
        Text-CNN model for 1D convolution over sequence data (text).
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of the word embeddings
            num_classes: Number of target classes (Binary classification = 2)
            kernel_sizes: List of n-gram sizes for the convolutional filters
            num_filters: Number of filters per kernel size
            dropout_p: Dropout probability
            padding_idx: Vocabulary index used for padding
        """
        super(TextCNN, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        
        # 1D Convolutional layers for different kernel sizes
        # In PyTorch, text inputs for Conv1d should be [batch_size, embed_dim, seq_length]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            ) for k in kernel_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Fully connected output layer
        # Output features match num_filters * number of different kernel sizes
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        
    def conv_and_pool(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """
        Helper method to apply convolution -> ReLU -> Max pooling.
        """
        # x is [batch_size, embed_dim, seq_length]
        x_conv = F.relu(conv(x))
        # x_conv is [batch_size, num_filters, seq_length - kernel_size + 1]
        
        # 1D Max Pooling over time (the sequence dimension)
        x_pooled = F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]).squeeze(2)
        # x_pooled is [batch_size, num_filters]
        return x_pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length] containing token IDs
            
        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # 1. Look up embeddings
        # x_embed: [batch_size, seq_length, embed_dim]
        x_embed = self.embedding(x)
        
        # 2. Permute to match Conv1d expected input shape [batch_size, in_channels, seq_length]
        # x_reshaped: [batch_size, embed_dim, seq_length]
        x_reshaped = x_embed.permute(0, 2, 1)
        
        # 3. Apply convolutions & pooling for each kernel size
        pooled_outputs = [self.conv_and_pool(x_reshaped, conv) for conv in self.convs]
        
        # 4. Concatenate features along the feature dimension
        # x_cat: [batch_size, num_filters * len(kernel_sizes)]
        x_cat = torch.cat(pooled_outputs, dim=1)
        
        # 5. Apply dropout and pass to fully connected output layer
        x_drop = self.dropout(x_cat)
        logits = self.fc(x_drop)
        
        return logits
