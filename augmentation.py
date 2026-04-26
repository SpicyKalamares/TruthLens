"""
Enhanced augmentation transforms for deepfake detection.

Designed to prevent model from learning shortcuts based on:
- Image style differences (via color jitter, brightness/contrast variation)
- Compression artifacts (via JPEG simulation)
- Resolution differences (via random resize/crop)
- Blur/noise patterns (via motion blur, Gaussian blur, noise injection)

Architecture: All transforms are composition-safe and work with both PIL and torch tensors.
"""

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance
from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomRotation, ColorJitter,
    RandomCrop, Resize, ToTensor, Normalize, RandomVerticalFlip,
    RandomAffine, RandomPerspective, GaussianBlur, RandomErasing
)


class JPEGCompressionSimulation:
    """
    Simulate JPEG compression artifacts by saving to JPEG quality and reloading.
    This targets the compression-artifact shortcut that real images might exhibit.
    
    Args:
        quality_range: tuple (min_quality, max_quality). Default (50, 90) simulates
                      varying compression levels that might appear in datasets.
    """
    
    def __init__(self, quality_range=(50, 90), p=0.7):
        self.quality_range = quality_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if not isinstance(img, Image.Image):
            return img
        
        quality = random.randint(*self.quality_range)
        
        # Save to JPEG buffer and reload
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
        img = img.copy()  # Detach from buffer
        
        return img


class GaussianNoiseInjection:
    """
    Add Gaussian noise to tensor to simulate sensor noise or transmission artifacts.
    Works on torch tensors (post-ToTensor).
    
    Args:
        std_range: tuple (min_std, max_std) for noise standard deviation.
                  Default (0.02, 0.05) adds ~2-5% noise to [0,1] range.
    """
    
    def __init__(self, std_range=(0.02, 0.05), p=0.5):
        self.std_range = std_range
        self.p = p
    
    def __call__(self, x):
        if random.random() > self.p:
            return x
        
        if not isinstance(x, torch.Tensor):
            return x
        
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)


class PoissonNoiseInjection:
    """
    Add Poisson noise (shot noise) to simulate photon noise in images.
    Works on torch tensors.
    
    Args:
        lam: Poisson lambda parameter (number of photons). Default 2 for subtle noise.
    """
    
    def __init__(self, lam=2.0, p=0.3):
        self.lam = lam
        self.p = p
    
    def __call__(self, x):
        if random.random() > self.p:
            return x
        
        if not isinstance(x, torch.Tensor):
            return x
        
        # Poisson noise: more aggressive in bright regions
        lam = random.uniform(self.lam, self.lam * 2)
        noise = torch.poisson(torch.ones_like(x) * lam) / (lam + 1e-8)
        noise = (noise - noise.mean()) * 0.02  # Scale to ~2% of intensity
        return torch.clamp(x + noise, 0, 1)


class MotionBlurSimulation:
    """
    Simulate motion blur by applying directional blur.
    Works on PIL images.
    
    Args:
        kernel_size_range: tuple (min, max) for blur kernel size.
        angle_range: tuple (min_angle, max_angle) in degrees.
    """
    
    def __init__(self, kernel_size_range=(3, 15), angle_range=(0, 360), p=0.3):
        self.kernel_size_range = kernel_size_range
        self.angle_range = angle_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if not isinstance(img, Image.Image):
            return img
        
        # For simplicity, approximate motion blur using directional Gaussian blur
        # True motion blur would require more complex kernel generation
        kernel_size = random.randint(*self.kernel_size_range)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Use GaussianBlur as motion blur approximation
        try:
            img = img.filter(ImageFilter.GaussianBlur(radius=kernel_size // 3))
        except:
            pass
        
        return img


class AdaptiveGaussianBlur:
    """
    Apply Gaussian blur with adaptive sigma to simulate defocus.
    Works on PIL images.
    """
    
    def __init__(self, sigma_range=(0.5, 2.0), p=0.3):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if not isinstance(img, Image.Image):
            return img
        
        sigma = random.uniform(*self.sigma_range)
        try:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        except:
            pass
        
        return img


class BrightnessContrastUnequal:
    """
    Apply unequal brightness/contrast (different channels can be affected differently).
    Works on PIL images.
    """
    
    def __init__(self, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1), p=0.3):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if not isinstance(img, Image.Image):
            return img
        
        brightness_factor = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)
        
        try:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness_factor)
            
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)
        except:
            pass
        
        return img


def get_train_transforms(image_size=224):
    """
    Strong augmentation pipeline for training.
    
    Strategy:
    1. Resize to (image_size + 4, image_size + 4) to enable RandomCrop
    2. Apply JPEG compression simulation (targets compression artifact shortcuts)
    3. Apply geometric transforms (rotation, crop, flip)
    4. Apply color transforms (ColorJitter, brightness/contrast)
    5. Apply blur/noise (motion blur, Gaussian blur, Gaussian/Poisson noise)
    6. Convert to tensor and normalize with ImageNet statistics
    
    Returns:
        Compose object that chains all transforms
    """
    
    return Compose([
        # Pre-crop augmentation
        Resize((image_size + 4, image_size + 4)),
        
        # JPEG compression (PIL)
        JPEGCompressionSimulation(quality_range=(50, 90), p=0.7),
        
        # Geometric augmentations (PIL)
        RandomCrop((image_size, image_size)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomRotation(degrees=15),
        RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Blur and noise (PIL, before ToTensor)
        MotionBlurSimulation(kernel_size_range=(3, 9), p=0.3),
        AdaptiveGaussianBlur(sigma_range=(0.5, 1.5), p=0.3),
        
        # Color augmentation (PIL)
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        BrightnessContrastUnequal(brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1), p=0.3),
        
        # Convert to tensor
        ToTensor(),
        
        # Noise injection (tensor-based)
        GaussianNoiseInjection(std_range=(0.02, 0.05), p=0.5),
        PoissonNoiseInjection(lam=2.0, p=0.2),
        
        # Random erasing (cutout-style)
        RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        
        # Normalization with ImageNet statistics
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(image_size=224):
    """
    Light augmentation for validation (mainly resizing + normalization).
    No aggressive augmentations to maintain distribution similarity to test set.
    
    Returns:
        Compose object that chains all transforms
    """
    
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_test_transforms(image_size=224):
    """
    No augmentation for test set evaluation (standard benchmark protocol).
    
    Returns:
        Compose object that chains all transforms
    """
    
    return Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# Alternative: minimal augmentation for baseline comparison
def get_baseline_transforms(image_size=224):
    """
    Minimal augmentation pipeline for baseline model comparison.
    Used in ablation studies to isolate augmentation impact.
    
    Returns:
        Compose object with only geometric transforms
    """
    
    return Compose([
        Resize((image_size + 4, image_size + 4)),
        RandomCrop((image_size, image_size)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
