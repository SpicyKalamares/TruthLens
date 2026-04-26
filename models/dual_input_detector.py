"""
Complete dual-input deepfake detector combining spatial and frequency branches.

Architecture:
    Input RGB image (3, 224, 224)
         ↓
    ┌────┴────┬──────────┐
    ↓         ↓          ↓
  Spatial   FFT        DCT
  Branch    Branch     Branch
  (512)     (256)      (256)
    ↓         ↓          ↓
    └────┬────┴──────────┘
         ↓
    Fusion Head
   (1024→512→256→64→1)
         ↓
    Sigmoid Output
    [0, 1] probability
    (0=Real, 1=AI-generated)

Forward pass time: ~100-150ms per image (GPU)
Total parameters: ~4.5M (manageable for multi-GPU training)

Loss function: BCEWithLogitsLoss (numerically stable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .spatial_branch import SpatialBranch
from .frequency_branch import FFTBranch, DCTBranch
from .fusion_head import FusionHead, SimpleFusionHead


class DualInputDetector(nn.Module):
    """
    Complete dual-input detector combining spatial and frequency features.
    
    Architecture:
    - Spatial branch: EfficientNet-B4 → 512-dim features
    - FFT branch: 2D FFT + small CNN → 256-dim features
    - DCT branch: block DCT + small CNN → 256-dim features
    - Fusion head: Concatenation + MLP → binary logit
    
    Args:
        spatial_dim (int): Dimension of spatial features (512)
        fft_dim (int): Dimension of FFT features (256)
        dct_dim (int): Dimension of DCT features (256)
        image_size (int): Expected input image size (224)
        use_attention (bool): Whether to use spatial attention in spatial branch
        fusion_type (str): Type of fusion head ('standard' or 'simple')
    """
    
    def __init__(
        self,
        spatial_dim: int = 512,
        fft_dim: int = 256,
        dct_dim: int = 256,
        image_size: int = 224,
        use_attention: bool = False,
        fusion_type: str = 'standard',
    ):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.fft_dim = fft_dim
        self.dct_dim = dct_dim
        self.image_size = image_size
        
        # Initialize branches
        self.spatial_branch = SpatialBranch(pretrained=True, freeze_backbone=True)
        self.fft_branch = FFTBranch(image_size=image_size)
        self.dct_branch = DCTBranch(image_size=image_size)
        
        # Initialize fusion head
        if fusion_type == 'standard':
            self.fusion_head = FusionHead(
                spatial_dim=spatial_dim,
                fft_dim=fft_dim,
                dct_dim=dct_dim,
            )
        elif fusion_type == 'simple':
            self.fusion_head = SimpleFusionHead(
                spatial_dim=spatial_dim,
                fft_dim=fft_dim,
                dct_dim=dct_dim,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.fusion_type = fusion_type
    
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from all branches.
        
        Args:
            x (torch.Tensor): RGB image of shape (batch_size, 3, 224, 224)
        
        Returns:
            dict: Dictionary with keys:
                - 'spatial': Spatial features (batch_size, 512)
                - 'fft': FFT features (batch_size, 256)
                - 'dct': DCT features (batch_size, 256)
        """
        spatial_features = self.spatial_branch(x)
        fft_features = self.fft_branch(x)
        dct_features = self.dct_branch(x)
        
        return {
            'spatial': spatial_features,
            'fft': fft_features,
            'dct': dct_features,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete model.
        
        Args:
            x (torch.Tensor): RGB image of shape (batch_size, 3, 224, 224)
                             Assumed to be normalized with ImageNet statistics:
                             μ = [0.485, 0.456, 0.406]
                             σ = [0.229, 0.224, 0.225]
        
        Returns:
            torch.Tensor: Binary classification logit of shape (batch_size, 1)
                         Unbounded output; pass through sigmoid() for probability
                         
                         For loss computation, use BCEWithLogitsLoss which applies
                         sigmoid internally for numerical stability.
        """
        # Extract features from all branches
        features = self.extract_features(x)
        
        spatial_features = features['spatial']
        fft_features = features['fft']
        dct_features = features['dct']
        
        # Fuse and classify
        logit = self.fusion_head(spatial_features, fft_features, dct_features)
        
        return logit
    
    def forward_with_details(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning logit and intermediate features.
        
        Useful for analysis, visualization, and ablation studies.
        
        Args:
            x (torch.Tensor): Input image
        
        Returns:
            dict: Contains keys:
                - 'logit': Final classification logit (batch_size, 1)
                - 'spatial_features': Spatial branch output (batch_size, 512)
                - 'fft_features': FFT branch output (batch_size, 256)
                - 'dct_features': DCT branch output (batch_size, 256)
                - 'prob': Sigmoid probability (batch_size, 1)
        """
        features = self.extract_features(x)
        
        spatial_features = features['spatial']
        fft_features = features['fft']
        dct_features = features['dct']
        
        logit = self.fusion_head(spatial_features, fft_features, dct_features)
        prob = torch.sigmoid(logit)
        
        return {
            'logit': logit,
            'spatial_features': spatial_features,
            'fft_features': fft_features,
            'dct_features': dct_features,
            'prob': prob,
        }
    
    def freeze_spatial_backbone(self):
        """Freeze spatial branch backbone (not fusion head)."""
        for param in self.spatial_branch.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_spatial_backbone(self):
        """Unfreeze entire spatial branch."""
        for param in self.spatial_branch.parameters():
            param.requires_grad = True
    
    def progressive_unfreeze(self, epoch: int):
        """
        Progressive unfreezing schedule during training.
        
        Args:
            epoch (int): Current training epoch
        """
        self.spatial_branch.unfreeze_progressive(epoch)
    
    def get_trainable_params_count(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class DualInputDetectorLightweight(nn.Module):
    """
    Lightweight variant of DualInputDetector for faster inference.
    
    Reduces:
    - EfficientNet-B4 → EfficientNet-B2
    - Drop some frequency branch complexity
    - Simpler fusion head
    
    Intended for resource-constrained deployment while maintaining accuracy.
    """
    
    def __init__(
        self,
        spatial_dim: int = 352,  # EfficientNet-B2 features
        fft_dim: int = 128,
        dct_dim: int = 128,
        image_size: int = 224,
    ):
        super().__init__()
        
        import torchvision.models as models
        
        self.spatial_dim = spatial_dim
        self.fft_dim = fft_dim
        self.dct_dim = dct_dim
        
        # Lightweight spatial branch (EfficientNet-B2)
        backbone = models.efficientnet_b2(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        
        self.spatial_head = nn.Sequential(
            nn.Linear(352, spatial_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(spatial_dim),
        )
        
        # Simplified frequency branches
        self.fft_branch = FFTBranch(image_size=image_size, num_fft_channels=16)
        self.dct_branch = DCTBranch(image_size=image_size, num_channels=16)
        
        # Simple fusion
        input_dim = spatial_dim + fft_dim + dct_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Spatial
        spatial = self.feature_extractor(x)
        spatial = spatial.view(spatial.size(0), -1)
        spatial = self.spatial_head(spatial)
        
        # Frequency
        fft = self.fft_branch(x)
        dct = self.dct_branch(x)
        
        # Fuse and classify
        fused = torch.cat([spatial, fft, dct], dim=1)
        logit = self.classifier(fused)
        
        return logit
