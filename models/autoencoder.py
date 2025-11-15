"""
Autoencoder Model cho Log Anomaly Detection
Kiến trúc: Encoder → Latent Space → Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """
    Autoencoder model cho log anomaly detection
    
    Architecture:
    - Encoder: Input → Hidden layers → Latent space
    - Decoder: Latent space → Hidden layers → Output
    - Loss: Mean Squared Error (MSE)
    """
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dims: List[int] = [256, 128, 64],
                 latent_dim: int = 32,
                 activation: str = "relu",
                 dropout: float = 0.1):
        """
        Khởi tạo Autoencoder
        
        Args:
            input_dim: Kích thước input embedding (128 cho Word2Vec)
            hidden_dims: List các hidden dimensions cho encoder/decoder
            latent_dim: Kích thước latent space (bottleneck)
            activation: Activation function ("relu", "tanh", "sigmoid")
            dropout: Dropout rate
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.activation_name = activation
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self.activation)
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Latent layer (bottleneck)
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(self.activation)
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        # Reverse hidden_dims for decoder
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(self.activation)
            decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # Sigmoid cho output nếu cần normalize
        # decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"Initialized Autoencoder:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Latent dim: {latent_dim}")
        logger.info(f"  Activation: {activation}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input vào latent space
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Latent representation (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode từ latent space về output
        
        Args:
            z: Latent tensor (batch_size, latent_dim)
        
        Returns:
            Reconstructed output (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode → decode
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (reconstructed, latent)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tính reconstruction error (MSE)
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Reconstruction error per sample (batch_size,)
        """
        reconstructed, _ = self.forward(x)
        error = F.mse_loss(reconstructed, x, reduction='none')
        # Sum over features, mean or sum over batch
        error = error.mean(dim=1)  # Mean over features
        return error
    
    def predict_anomaly(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Predict anomaly dựa trên reconstruction error
        
        Args:
            x: Input tensor (batch_size, input_dim)
            threshold: Threshold cho reconstruction error
        
        Returns:
            Binary predictions (batch_size,): 1 = anomaly, 0 = normal
        """
        error = self.get_reconstruction_error(x)
        predictions = (error > threshold).long()
        return predictions
    
    def get_latent_representation(self, x: torch.Tensor) -> np.ndarray:
        """
        Get latent representation cho visualization
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Latent representations as numpy array (batch_size, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            latent = self.encode(x)
            return latent.cpu().numpy()


def create_autoencoder(input_dim: int = 128,
                       hidden_dims: Optional[List[int]] = None,
                       latent_dim: int = 32,
                       **kwargs) -> Autoencoder:
    """
    Factory function để tạo Autoencoder
    
    Args:
        input_dim: Input dimension
        hidden_dims: Hidden dimensions (default: [256, 128, 64])
        latent_dim: Latent dimension
        **kwargs: Additional arguments cho Autoencoder
    
    Returns:
        Autoencoder model
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    return Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        **kwargs
    )


def count_parameters(model: nn.Module) -> int:
    """
    Đếm số lượng parameters trong model
    
    Args:
        model: PyTorch model
    
    Returns:
        Tổng số parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Test Autoencoder model
    """
    print("\n" + "="*70)
    print("TEST AUTOENCODER MODEL")
    print("="*70)
    
    # Test với sample data
    batch_size = 32
    input_dim = 128
    
    # Create model
    model = create_autoencoder(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        activation="relu",
        dropout=0.1
    )
    
    print(f"\n✅ Model created:")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    reconstructed, latent = model(x)
    
    print(f"\n✅ Forward pass:")
    print(f"   Input shape: {x.shape}")
    print(f"   Latent shape: {latent.shape}")
    print(f"   Reconstructed shape: {reconstructed.shape}")
    
    # Test reconstruction error
    error = model.get_reconstruction_error(x)
    print(f"\n✅ Reconstruction error:")
    print(f"   Error shape: {error.shape}")
    print(f"   Mean error: {error.mean().item():.4f}")
    print(f"   Std error: {error.std().item():.4f}")
    
    # Test anomaly prediction
    threshold = error.mean().item() + 2 * error.std().item()
    predictions = model.predict_anomaly(x, threshold)
    print(f"\n✅ Anomaly prediction (threshold={threshold:.4f}):")
    print(f"   Normal: {(predictions == 0).sum().item()}")
    print(f"   Anomaly: {(predictions == 1).sum().item()}")
    
    print("\n" + "="*70)
    print("✅ Autoencoder model test completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

