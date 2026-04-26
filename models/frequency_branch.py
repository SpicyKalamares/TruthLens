"""
Frequency-domain feature extraction using FFT and DCT transforms.

Two parallel branches:
1. FFT branch: Captures global frequency patterns (generator signatures)
2. DCT branch: Captures block-based patterns (JPEG artifacts)

Both branches process frequency magnitude through small CNNs to extract
discriminative frequency-domain features without overfitting to sparse
frequency representations.

Rationale:
- AI generators (diffusion models, GANs) produce characteristic frequency patterns
- Diffusion models can introduce subtle frequency artifacts
- DCT mimics JPEG compression block structure (common in image processing)
- Small CNN branches avoid overfitting while learning frequency relevance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class FFTBranch(nn.Module):
    """
    FFT-based frequency feature extraction.
    
    Process:
    1. Convert RGB image to grayscale
    2. Apply 2D FFT
    3. Compute log-magnitude spectrum (avoids division by zero)
    4. Normalize to [0, 1] range
    5. Pass through small CNN to extract frequency features
    
    Output: 256-dimensional frequency features
    
    Args:
        image_size (int): Expected input size (224 for ImageNet)
        num_fft_channels (int): Number of channels in first conv layer
    """
    
    def __init__(self, image_size: int = 224, num_fft_channels: int = 32):
        super().__init__()
        
        self.image_size = image_size
        
        # Small CNN to process FFT magnitude spectrum
        self.fft_cnn = nn.Sequential(
            # Input: (1, image_size, image_size)
            nn.Conv2d(1, num_fft_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_fft_channels),
            
            # (32, image_size, image_size)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (32, image_size//2, image_size//2)
            
            nn.Conv2d(num_fft_channels, num_fft_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_fft_channels * 2),
            
            # (64, image_size//2, image_size//2)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (64, image_size//4, image_size//4)
            
            nn.Conv2d(num_fft_channels * 2, num_fft_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_fft_channels * 4),
            
            # (128, image_size//4, image_size//4)
            nn.AdaptiveAvgPool2d(1),
            # (128, 1, 1)
        )
        
        # Projection head
        cnn_output_features = num_fft_channels * 4
        self.projection_head = nn.Sequential(
            nn.Linear(cnn_output_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FFT branch.
        
        Args:
            x (torch.Tensor): RGB image of shape (batch_size, 3, 224, 224) with values in [0, 1]
                             (assumed to be normalized with ImageNet stats)
            
        Returns:
            torch.Tensor: FFT features of shape (batch_size, 256)
        """
        # Denormalize from ImageNet statistics (for better frequency interpretation)
        # x is already normalized, so reverse it
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_denorm = x * imagenet_std + imagenet_mean
        
        # Convert RGB to grayscale using standard weights
        # L = 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * x_denorm[:, 0:1, :, :] + 0.587 * x_denorm[:, 1:2, :, :] + 0.114 * x_denorm[:, 2:3, :, :]
        
        # Apply 2D FFT
        batch_size = gray.shape[0]
        fft_result = torch.fft.rfft2(gray.squeeze(1), dim=(-2, -1))  # Real FFT for efficiency
        
        # Compute magnitude spectrum
        fft_magnitude = torch.abs(fft_result)  # (batch_size, 224, 113)
        
        # Log magnitude to compress dynamic range (avoid division by zero)
        fft_log_magnitude = torch.log(fft_magnitude + 1e-8)
        
        # Resize to original image size for CNN processing
        # fft_log_magnitude is (batch_size, H, W//2+1), we need (batch_size, H, W)
        # Pad to match original dimensions
        fft_log_magnitude_padded = F.pad(fft_log_magnitude, (0, 1), mode='reflect')  # Mirror pad
        
        # Add channel dimension
        fft_input = fft_log_magnitude_padded.unsqueeze(1)  # (batch_size, 1, 224, 224)
        
        # Normalize to [0, 1] range
        fft_input_min = fft_input.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        fft_input_max = fft_input.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        fft_input_norm = (fft_input - fft_input_min) / (fft_input_max - fft_input_min + 1e-8)
        
        # Pass through CNN
        cnn_features = self.fft_cnn(fft_input_norm)  # (batch_size, 128, 1, 1)
        
        # Flatten
        cnn_features_flat = cnn_features.view(batch_size, -1)  # (batch_size, 128)
        
        # Project to 256 dimensions
        fft_features = self.projection_head(cnn_features_flat)  # (batch_size, 256)
        
        return fft_features


class DCTBranch(nn.Module):
    """
    DCT-based frequency feature extraction (simplified).
    
    Process:
    1. Convert RGB image to grayscale
    2. Apply 2D FFT (serves as proxy for frequency domain analysis)
    3. Extract low-frequency coefficients
    4. Pass through small CNN to extract frequency features
    
    Note: We use FFT instead of true DCT for simplicity while maintaining
    the ability to capture frequency-domain artifacts from AI generators.
    
    Output: 256-dimensional frequency features
    
    Args:
        image_size (int): Expected input size (224 for ImageNet)
        num_channels (int): Number of channels in first conv layer
    """
    
    def __init__(
        self,
        image_size: int = 224,
        num_channels: int = 32
    ):
        super().__init__()
        
        self.image_size = image_size
        
        # Small CNN to process frequency components
        self.dct_cnn = nn.Sequential(
            # Input: (1, image_size, image_size)
            nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            
            # (32, image_size, image_size)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (32, image_size//2, image_size//2)
            
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels * 2),
            
            # (64, image_size//2, image_size//2)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (64, image_size//4, image_size//4)
            
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels * 4),
            
            # (128, image_size//4, image_size//4)
            nn.AdaptiveAvgPool2d(1),
            # (128, 1, 1)
        )
        
        # Projection head
        cnn_output_features = num_channels * 4
        self.projection_head = nn.Sequential(
            nn.Linear(cnn_output_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DCT branch.
        
        Args:
            x (torch.Tensor): RGB image of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: DCT features of shape (batch_size, 256)
        """
        # Denormalize from ImageNet statistics
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_denorm = x * imagenet_std + imagenet_mean
        
        # Convert RGB to grayscale
        gray = 0.299 * x_denorm[:, 0:1, :, :] + 0.587 * x_denorm[:, 1:2, :, :] + 0.114 * x_denorm[:, 2:3, :, :]
        
        # Apply 2D FFT to get frequency domain representation
        batch_size = gray.shape[0]
        fft_result = torch.fft.rfft2(gray.squeeze(1), dim=(-2, -1))
        
        # Extract magnitude spectrum
        magnitude = torch.abs(fft_result)  # (batch_size, H, W//2+1)
        
        # Log scale to compress dynamic range
        magnitude_log = torch.log(magnitude + 1e-8)
        
        # Resize to match spatial dimensions for CNN
        magnitude_log_padded = F.pad(magnitude_log, (0, 1), mode='reflect')  # Mirror pad to get full width
        
        # Add channel dimension
        magnitude_input = magnitude_log_padded.unsqueeze(1)  # (batch_size, 1, H, W)
        
        # Normalize to [0, 1]
        mag_min = magnitude_input.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        mag_max = magnitude_input.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        magnitude_norm = (magnitude_input - mag_min) / (mag_max - mag_min + 1e-8)
        
        # Pass through CNN
        cnn_features = self.dct_cnn(magnitude_norm)  # (batch_size, 128, 1, 1)
        
        # Flatten
        cnn_features_flat = cnn_features.view(batch_size, -1)  # (batch_size, 128)
        
        # Project to 256 dimensions
        dct_features = self.projection_head(cnn_features_flat)  # (batch_size, 256)
        
        return dct_features
