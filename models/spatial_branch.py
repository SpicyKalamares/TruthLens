"""
Spatial feature extraction branch using EfficientNet-B4.

EfficientNet-B4 provides:
- Stronger feature representation than B3 (more capacity)
- Better transfer learning from ImageNet
- Efficient architecture for multi-branch fusion (not too parameter-heavy)

Output: 1792-dimensional feature vector before classification head
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class SpatialBranch(nn.Module):
    """
    EfficientNet-B4 backbone with extraction of intermediate features.
    
    Architecture:
    - ImageNet pretrained EfficientNet-B4
    - Extract features after final pooling (1792-dimensional)
    - Add a small projection head to reduce to 512 dimensions
    
    Args:
        pretrained (bool): Load ImageNet pretrained weights
        freeze_backbone (bool): Initially freeze backbone (thaw during training)
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        
        # Load EfficientNet-B4
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Get feature dimension from final classifier input
        self.feature_dim = self.backbone.classifier[1].in_features  # 1792
        
        # Create feature extraction head (removes original classification layer)
        # Keep everything up to the final pooling in backbone
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head: reduce 1792 -> 512
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )
        
        # Initially freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters except projection head."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self, num_blocks_to_unfreeze: int = 2):
        """
        Progressively unfreeze backbone layers.
        
        EfficientNet-B4 structure:
        - features[0]: stem (always keep frozen initially)
        - features[1-8]: blocks
        
        Args:
            num_blocks_to_unfreeze: Number of final blocks to unfreeze (1-8)
        """
        # features is a Sequential module
        features = self.feature_extractor[0]
        
        # Unfreeze last num_blocks_to_unfreeze blocks
        for block in features[-num_blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True
    
    def unfreeze_progressive(self, epoch: int):
        """
        Progressive unfreezing schedule during training.
        
        Args:
            epoch (int): Current training epoch
        """
        if epoch == 3:
            # Unfreeze last 2 blocks at epoch 3
            self._unfreeze_backbone(num_blocks_to_unfreeze=2)
            print("Unfreezing EfficientNet-B4 top 2 blocks at epoch 3")
        
        elif epoch == 8:
            # Unfreeze last 4 blocks at epoch 8
            self._unfreeze_backbone(num_blocks_to_unfreeze=4)
            print("Unfreezing EfficientNet-B4 top 4 blocks at epoch 8")
        
        elif epoch == 12:
            # Unfreeze last 6 blocks (keep early blocks frozen to save GPU memory)
            self._unfreeze_backbone(num_blocks_to_unfreeze=6)
            print("Unfreezing EfficientNet-B4 top 6 blocks at epoch 12")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial branch.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Spatial features of shape (batch_size, 512)
        """
        # Extract backbone features
        features = self.feature_extractor(x)  # (batch_size, 1792, 1, 1)
        
        # Flatten
        features = features.view(features.size(0), -1)  # (batch_size, 1792)
        
        # Project to 512 dimensions
        spatial_features = self.projection_head(features)  # (batch_size, 512)
        
        return spatial_features


class SpatialBranchWithAttention(nn.Module):
    """
    Alternative SpatialBranch with channel attention mechanism.
    
    Adds Squeeze-and-Excitation (SE) attention to the projection head
    to weight spatial features adaptively.
    
    Args:
        pretrained (bool): Load ImageNet pretrained weights
        freeze_backbone (bool): Initially freeze backbone
        reduction_ratio (int): Reduction ratio for SE module
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        reduction_ratio: int = 16
    ):
        super().__init__()
        
        # Load EfficientNet-B4
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        self.feature_dim = self.backbone.classifier[1].in_features  # 1792
        
        # Feature extractor (no classification layer)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head with SE attention
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )
        
        # Squeeze-and-Excitation (SE) module
        self.se_module = nn.Sequential(
            nn.Linear(512, 512 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(512 // reduction_ratio, 512),
            nn.Sigmoid(),
        )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x (torch.Tensor): Input of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Spatial features of shape (batch_size, 512)
        """
        # Extract features
        features = self.feature_extractor(x)  # (batch_size, 1792, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 1792)
        
        # Project
        spatial_features = self.projection_head(features)  # (batch_size, 512)
        
        # Apply SE attention
        attention = self.se_module(spatial_features)  # (batch_size, 512)
        spatial_features = spatial_features * attention  # Element-wise multiplication
        
        return spatial_features
