"""
Feature fusion head that combines spatial and frequency features.

Fusion strategy:
- Concatenate spatial (512) + FFT (256) + DCT (256) = 1024 dimensions
- Pass through fusion head: 1024 -> 512 -> 256 -> 64 -> 1 (binary output)
- Each layer includes ReLU, BatchNorm, and Dropout for regularization
- Output: Sigmoid probability for AI-generated class

Rationale:
- Concatenation allows downstream layers to learn feature importance weights
- Progressive dimensionality reduction avoids overfitting in fusion layers
- Dropout (5%, 4%, 3%) prevents co-adaptation of features
- Final sigmoid outputs probability in [0, 1] range
"""

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """
    Feature fusion module that combines spatial and frequency features.
    
    Architecture:
    - Input: [spatial_features (512) || fft_features (256) || dct_features (256)]
    - Hidden layers with progressive dimension reduction
    - Output: Binary classification logit (scalar)
    
    Args:
        spatial_dim (int): Dimension of spatial branch output (default 512)
        fft_dim (int): Dimension of FFT branch output (default 256)
        dct_dim (int): Dimension of DCT branch output (default 256)
        hidden_dim1 (int): First hidden layer dimension (default 512)
        hidden_dim2 (int): Second hidden layer dimension (default 256)
        hidden_dim3 (int): Third hidden layer dimension (default 64)
        dropout_rates (tuple): Dropout rates for layers (default (0.5, 0.4, 0.3))
    """
    
    def __init__(
        self,
        spatial_dim: int = 512,
        fft_dim: int = 256,
        dct_dim: int = 256,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        hidden_dim3: int = 64,
        dropout_rates: tuple = (0.5, 0.4, 0.3),
    ):
        super().__init__()
        
        # Input dimension is concatenation of all feature branches
        input_dim = spatial_dim + fft_dim + dct_dim  # 1024
        
        # Fusion layers with progressive dimensionality reduction
        self.fusion = nn.Sequential(
            # Layer 1: 1024 -> 512
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout_rates[0]),
            
            # Layer 2: 512 -> 256
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout_rates[1]),
            
            # Layer 3: 256 -> 64
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[2]),
            
            # Output layer: 64 -> 1 (binary logit)
            nn.Linear(hidden_dim3, 1),
        )
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        fft_features: torch.Tensor,
        dct_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass combining spatial and frequency features.
        
        Args:
            spatial_features (torch.Tensor): Spatial features from EfficientNet-B4
                                            of shape (batch_size, 512)
            fft_features (torch.Tensor): FFT features of shape (batch_size, 256)
            dct_features (torch.Tensor): DCT features of shape (batch_size, 256)
        
        Returns:
            torch.Tensor: Binary classification logit of shape (batch_size, 1)
                         Values are unbounded; pass through sigmoid for probability
        """
        # Concatenate all feature branches
        fused_features = torch.cat(
            [spatial_features, fft_features, dct_features],
            dim=1
        )  # (batch_size, 1024)
        
        # Process through fusion layers
        logit = self.fusion(fused_features)  # (batch_size, 1)
        
        return logit


class SimpleFusionHead(nn.Module):
    """
    Simpler fusion head for ablation studies (baseline).
    
    Useful for understanding whether the complexity of the full fusion head
    is necessary for good performance.
    
    Args:
        spatial_dim (int): Dimension of spatial features (512)
        fft_dim (int): Dimension of FFT features (256)
        dct_dim (int): Dimension of DCT features (256)
    """
    
    def __init__(
        self,
        spatial_dim: int = 512,
        fft_dim: int = 256,
        dct_dim: int = 256,
    ):
        super().__init__()
        
        input_dim = spatial_dim + fft_dim + dct_dim  # 1024
        
        # Simplified: single hidden layer
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        fft_features: torch.Tensor,
        dct_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (same as FusionHead)."""
        fused_features = torch.cat(
            [spatial_features, fft_features, dct_features],
            dim=1
        )
        logit = self.fusion(fused_features)
        return logit


class WeightedFusionHead(nn.Module):
    """
    Fusion head with learnable feature weighting.
    
    Each feature branch gets a learned attention weight before concatenation,
    allowing the model to learn which modalities are most important.
    
    Args:
        spatial_dim (int): Dimension of spatial features (512)
        fft_dim (int): Dimension of FFT features (256)
        dct_dim (int): Dimension of DCT features (256)
    """
    
    def __init__(
        self,
        spatial_dim: int = 512,
        fft_dim: int = 256,
        dct_dim: int = 256,
    ):
        super().__init__()
        
        # Learnable weights for each feature branch
        # Initialize with equal weights
        self.spatial_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.fft_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.dct_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        # Apply softmax to ensure weights sum to 1
        # But we'll do this in forward pass using functional softmax
        
        input_dim = spatial_dim + fft_dim + dct_dim
        
        # Main fusion layers (same as FusionHead)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1),
        )
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        fft_features: torch.Tensor,
        dct_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with learned feature weighting."""
        # Compute normalized weights via softmax
        weights = torch.stack([self.spatial_weight, self.fft_weight, self.dct_weight])
        weights = torch.softmax(weights, dim=0)
        
        # Apply weights to each feature branch
        spatial_weighted = spatial_features * weights[0]
        fft_weighted = fft_features * weights[1]
        dct_weighted = dct_features * weights[2]
        
        # Concatenate weighted features
        fused_features = torch.cat(
            [spatial_weighted, fft_weighted, dct_weighted],
            dim=1
        )
        
        # Process through fusion layers
        logit = self.fusion(fused_features)
        
        return logit
