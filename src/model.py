"""
Deep Learning Model for Personnel Norm Prediction
==================================================
This module defines the neural network architectures for predicting
optimal personnel allocation in Turkish agricultural districts.

Models:
- PersonnelMLP: Multi-Layer Perceptron for regression
- PersonnelResNet: Residual Network with skip connections
- PersonnelAttentionNet: Network with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonnelMLP(nn.Module):
    """
    Multi-Layer Perceptron for personnel norm prediction.

    A feedforward neural network with configurable hidden layers,
    dropout for regularization, and batch normalization.

    Architecture:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Output
    """

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.3):
        """
        Initialize the MLP model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
        """
        super(PersonnelMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.network(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Architecture:
        x -> Linear -> BN -> ReLU -> Linear -> BN -> (+x) -> ReLU
    """

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        """
        Initialize residual block.

        Args:
            dim: Dimension of the block (input and output)
            dropout_rate: Dropout probability
        """
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass with skip connection."""
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class PersonnelResNet(nn.Module):
    """
    Residual Network for personnel norm prediction.

    Uses skip connections to enable training of deeper networks
    and prevent vanishing gradients.

    Architecture:
        Input -> Linear -> [ResidualBlock] x N -> Linear -> Output
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_blocks: int = 3, dropout_rate: float = 0.2):
        """
        Initialize the ResNet model.

        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of hidden layers
            num_blocks: Number of residual blocks
            dropout_rate: Dropout probability
        """
        super(PersonnelResNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass through the ResNet."""
        x = self.input_layer(x)

        for block in self.res_blocks:
            x = block(x)

        return self.output_layer(x).squeeze(-1)


class AttentionLayer(nn.Module):
    """
    Self-attention layer for feature weighting.

    Learns to weight different input features based on their
    importance for the prediction task.
    """

    def __init__(self, dim: int):
        """
        Initialize attention layer.

        Args:
            dim: Feature dimension
        """
        super(AttentionLayer, self).__init__()

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        """
        Apply self-attention.

        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            Attention-weighted output
        """
        # For 1D input, we treat each feature as a "token"
        # Reshape to (batch, 1, dim) for attention computation
        x = x.unsqueeze(1)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        return out.squeeze(1)


class PersonnelAttentionNet(nn.Module):
    """
    Neural Network with Attention for personnel norm prediction.

    Uses attention mechanism to learn feature importance weights,
    followed by MLP layers for final prediction.

    Architecture:
        Input -> Attention -> MLP -> Output
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 dropout_rate: float = 0.3):
        """
        Initialize the Attention Network.

        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout probability
        """
        super(PersonnelAttentionNet, self).__init__()

        self.input_dim = input_dim

        # Feature embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Attention layer
        self.attention = AttentionLayer(hidden_dim)

        # MLP after attention
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        """Forward pass through the attention network."""
        x = self.embedding(x)
        x = self.attention(x)
        return self.mlp(x).squeeze(-1)


def get_model(model_name: str, input_dim: int, **kwargs):
    """
    Factory function to create models by name.

    Args:
        model_name: One of 'mlp', 'resnet', 'attention'
        input_dim: Number of input features
        **kwargs: Additional model parameters

    Returns:
        Initialized model
    """
    models = {
        'mlp': PersonnelMLP,
        'resnet': PersonnelResNet,
        'attention': PersonnelAttentionNet
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(models.keys())}")

    return models[model_name.lower()](input_dim, **kwargs)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...")

    input_dim = 14
    batch_size = 32

    # Create sample input
    x = torch.randn(batch_size, input_dim)

    # Test MLP
    print("\n" + "="*50)
    print("MLP Model")
    print("="*50)
    mlp = PersonnelMLP(input_dim)
    out = mlp(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(mlp):,}")

    # Test ResNet
    print("\n" + "="*50)
    print("ResNet Model")
    print("="*50)
    resnet = PersonnelResNet(input_dim)
    out = resnet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(resnet):,}")

    # Test Attention Net
    print("\n" + "="*50)
    print("Attention Network")
    print("="*50)
    attn_net = PersonnelAttentionNet(input_dim)
    out = attn_net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(attn_net):,}")
