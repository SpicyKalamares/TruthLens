"""
PyTorch Dataset classes for binary classification of real vs AI-generated images.

Features:
- StratifiedImageDataset: Implements stratified train/val/test split with reproducible splits
- No data leakage: Explicit assertions verify zero intersection between splits
- Windows compatibility: All dataset classes defined at module level for pickle serialization
- Robustness: Handles truncated images, RGBA→RGB conversion, missing files
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, List, Optional, Callable, Dict
import warnings

# Enable truncated image handling
Image.MAX_IMAGE_PIXELS = None
ImageFile = Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    """
    Basic image dataset for binary classification (Real vs AI-generated).
    
    Args:
        root_dir (str): Root directory containing 'real' and 'fake' subdirectories
        transform (callable): Image transformations (augmentation/normalization)
        balance_classes (bool): If True, weight samples to balance class distribution
        
    Structure expected:
        root_dir/
            real/  (Class 0)
            fake/  (Class 1)
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        balance_classes: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.balance_classes = balance_classes
        
        self.samples = []
        self.labels = []
        self.class_to_idx = {'real': 0, 'fake': 1}
        self.idx_to_class = {0: 'real', 1: 'fake'}
        
        # Load samples from directory structure
        self._load_samples()
        
        if balance_classes:
            self._compute_class_weights()
        else:
            self.class_weights = None
    
    def _load_samples(self):
        """Load all valid image paths and their labels."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                warnings.warn(f"Class directory not found: {class_dir}")
                continue
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    try:
                        # Quick validation: try opening
                        with Image.open(img_path) as img:
                            pass
                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)
                    except Exception as e:
                        warnings.warn(f"Skipping corrupted image {img_path}: {e}")
        
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {self.root_dir}")
        
        print(f"Loaded {len(self.samples)} images: {np.bincount(self.labels)}")
    
    def _compute_class_weights(self):
        """Compute inverse class frequency weights for balanced sampling."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        self.class_weights = torch.ones(len(self.labels), dtype=torch.float32)
        for class_idx, count in zip(unique, counts):
            weight = total / (len(unique) * count)
            self.class_weights[self.labels == class_idx] = weight
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (image_tensor, label)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Open and convert to RGB (handle RGBA)
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return blank image on failure (avoid training crash)
            warnings.warn(f"Failed to load {img_path}: {e}. Returning blank image.")
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class StratifiedImageDataset(Dataset):
    """
    Image dataset with stratified train/validation/test split.
    
    Implements StratifiedShuffleSplit to maintain class distribution across splits
    and ensure no data leakage between splits.
    
    Args:
        root_dir (str): Root directory containing 'real' and 'fake' subdirectories
        split_type (str): One of 'train', 'val', 'test'
        train_ratio (float): Proportion for training set (e.g., 0.7)
        val_ratio (float): Proportion for validation set (e.g., 0.15)
        transform (callable): Image transformations
        seed (int): Random seed for reproducibility
        balance_classes (bool): Whether to compute class weights
    
    Stratification ensures:
        - Each split has same class ratio as full dataset (within statistical limits)
        - No sample appears in multiple splits (verified explicitly)
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    def __init__(
        self,
        root_dir: str,
        split_type: str = 'train',
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        transform: Optional[Callable] = None,
        seed: int = 42,
        balance_classes: bool = False,
        max_samples: Optional[int] = None,  # Limit total samples (for memory-constrained training)
    ):
        assert split_type in ['train', 'val', 'test'], "split_type must be 'train', 'val', or 'test'"
        assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
        assert 0 <= val_ratio < 1, "val_ratio must be in [0, 1)"
        
        test_ratio = 1.0 - train_ratio - val_ratio
        assert test_ratio > 0, f"test_ratio = {test_ratio} must be > 0"
        
        self.root_dir = Path(root_dir)
        self.split_type = split_type
        self.transform = transform
        self.seed = seed
        self.balance_classes = balance_classes
        
        self.class_to_idx = {'real': 0, 'fake': 1}
        self.idx_to_class = {0: 'real', 1: 'fake'}
        
        # Load all samples
        self.all_samples = []
        self.all_labels = []
        self._load_all_samples()
        
        # Optionally limit to max_samples (balanced per class)
        if max_samples is not None and len(self.all_samples) > max_samples:
            all_samples_array = np.array(self.all_samples)
            all_labels_array = np.array(self.all_labels)
            
            # Balance sampling: limit per class
            samples_per_class = max_samples // 2  # Assume binary classification
            
            sampled_indices = []
            for class_idx in [0, 1]:
                class_mask = all_labels_array == class_idx
                class_indices = np.where(class_mask)[0]
                
                if len(class_indices) > samples_per_class:
                    selected = np.random.RandomState(seed).choice(
                        class_indices, size=samples_per_class, replace=False
                    )
                    sampled_indices.extend(selected)
                else:
                    sampled_indices.extend(class_indices)
            
            sampled_indices = np.array(sampled_indices)
            self.all_samples = all_samples_array[sampled_indices].tolist()
            self.all_labels = all_labels_array[sampled_indices].tolist()
            print(f"Limited dataset to {len(self.all_samples)} samples (max_samples={max_samples})")
        
        # Check if pre-existing train/test structure exists
        has_split_structure = (self.root_dir / "train").exists() and (self.root_dir / "test").exists()
        
        if has_split_structure and split_type == 'test':
            # Use pre-existing test set as-is (no further splitting)
            all_samples_array = np.array(self.all_samples)
            all_labels_array = np.array(self.all_labels)
            
            self.samples = all_samples_array
            self.labels = all_labels_array
            self.indices = np.arange(len(all_samples_array))
            train_idx = np.array([])
            val_idx = np.array([])
            test_idx = self.indices
        else:
            # Perform stratified split on training data
            all_samples_array = np.array(self.all_samples)
            all_labels_array = np.array(self.all_labels)
            
            # First split: separate train from (val+test)
            splitter_1 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=(val_ratio + test_ratio),
                random_state=seed
            )
            
            train_idx, temp_idx = next(splitter_1.split(all_samples_array, all_labels_array))
            
            # Second split: separate val from test
            # Rebalance: val_ratio vs test_ratio from remaining data
            val_ratio_normalized = val_ratio / (val_ratio + test_ratio)
            
            splitter_2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - val_ratio_normalized,
                random_state=seed + 1  # Different seed for second split
            )
            
            val_idx, test_idx = next(splitter_2.split(all_samples_array[temp_idx], all_labels_array[temp_idx]))
            val_idx = temp_idx[val_idx]
            test_idx = temp_idx[test_idx]
            
            # Assign samples based on split type
            if split_type == 'train':
                self.samples = all_samples_array[train_idx]
                self.labels = all_labels_array[train_idx]
                self.indices = train_idx
            elif split_type == 'val':
                self.samples = all_samples_array[val_idx]
                self.labels = all_labels_array[val_idx]
                self.indices = val_idx
            else:  # test
                self.samples = all_samples_array[test_idx]
                self.labels = all_labels_array[test_idx]
                self.indices = test_idx
        
        # Verify no leakage (only if splits were created)
        if len(train_idx) > 0 or len(val_idx) > 0:
            self._verify_no_leakage(train_idx, val_idx, test_idx)
        
        # Compute class weights if requested
        if balance_classes:
            self._compute_class_weights()
        else:
            self.class_weights = None
        
        # Log split information
        print(f"\n{'='*60}")
        print(f"StratifiedImageDataset split: {split_type}")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_samples_array)}")
        print(f"Train samples: {len(train_idx)} ({len(train_idx)/len(all_samples_array)*100:.1f}%)")
        print(f"Val samples: {len(val_idx)} ({len(val_idx)/len(all_samples_array)*100:.1f}%)")
        print(f"Test samples: {len(test_idx)} ({len(test_idx)/len(all_samples_array)*100:.1f}%)")
        print(f"\n{split_type.upper()} split class distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"  {self.idx_to_class[class_idx]}: {count} ({count/len(self.labels)*100:.1f}%)")
        print(f"{'='*60}\n")
    
    def _load_all_samples(self):
        """Load all valid image paths and labels.
        
        Handles two directory structures:
        1. Dataset/real/, Dataset/fake/ (flat structure)
        2. Dataset/train/real/, Dataset/train/fake/, Dataset/test/real/, Dataset/test/fake/ (split structure)
        """
        root = self.root_dir
        
        # Check for existing train/test split structure
        train_dir = root / "train"
        test_dir = root / "test"
        has_split_structure = train_dir.exists() and test_dir.exists()
        
        if has_split_structure:
            # Load from appropriate split directory based on split_type
            if self.split_type == 'test':
                # Load from Dataset/test/
                source_dir = test_dir
            else:
                # Load from Dataset/train/ (will be further split into train/val)
                source_dir = train_dir
        else:
            # Load from flat structure Dataset/real/, Dataset/fake/
            source_dir = root
        
        # Load images from all class directories (skip validation for speed with large datasets)
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = source_dir / class_name
            
            if not class_dir.exists():
                warnings.warn(f"Class directory not found: {class_dir}")
                continue
            
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    self.all_samples.append(str(img_path))
                    self.all_labels.append(class_idx)
        
        if len(self.all_samples) == 0:
            raise ValueError(f"No valid images found in {source_dir}")
    
    @staticmethod
    def _verify_no_leakage(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
        """Verify that train/val/test splits have no overlap."""
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        
        assert len(all_indices) == len(np.unique(all_indices)), \
            "Data leakage detected: indices appear in multiple splits!"
        
        assert len(np.intersect1d(train_idx, val_idx)) == 0, \
            "Data leakage: train and val sets overlap!"
        assert len(np.intersect1d(train_idx, test_idx)) == 0, \
            "Data leakage: train and test sets overlap!"
        assert len(np.intersect1d(val_idx, test_idx)) == 0, \
            "Data leakage: val and test sets overlap!"
        
        print("[OK] No data leakage detected: train/val/test splits are disjoint.")
    
    def _compute_class_weights(self):
        """Compute inverse class frequency weights for balanced sampling."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        self.class_weights = torch.ones(len(self.labels), dtype=torch.float32)
        for class_idx, count in zip(unique, counts):
            weight = total / (len(unique) * count)
            self.class_weights[self.labels == class_idx] = weight
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (image_tensor, label)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            warnings.warn(f"Failed to load {img_path}: {e}. Returning blank image.")
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    seed: int = 42,
    balance_classes: bool = False,
    max_samples: Optional[int] = None,  # Limit total samples for memory-constrained training
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with stratified splits.
    
    Args:
        data_dir: Root directory with 'real' and 'fake' subdirectories
        batch_size: Batch size for all loaders
        num_workers: Number of dataloader workers (0 for Windows)
        train_transform: Transform pipeline for training (with augmentation)
        val_transform: Transform pipeline for validation (light)
        test_transform: Transform pipeline for testing (no augmentation)
        seed: Random seed for reproducibility
        balance_classes: Whether to use class weights for sampling
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # Use provided transforms or fallback to identity
    if train_transform is None:
        train_transform = lambda x: x
    if val_transform is None:
        val_transform = lambda x: x
    if test_transform is None:
        test_transform = lambda x: x
    
    # Create datasets
    train_dataset = StratifiedImageDataset(
        data_dir,
        split_type='train',
        transform=train_transform,
        seed=seed,
        balance_classes=balance_classes,
        max_samples=max_samples,
    )
    
    val_dataset = StratifiedImageDataset(
        data_dir,
        split_type='val',
        transform=val_transform,
        seed=seed,
        balance_classes=False,  # No balancing on validation
        max_samples=max_samples,
    )
    
    test_dataset = StratifiedImageDataset(
        data_dir,
        split_type='test',
        transform=test_transform,
        seed=seed,
        balance_classes=False,  # No balancing on test
        max_samples=max_samples,
    )
    
    # Create sampler for class balancing if requested
    if balance_classes:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=train_dataset.class_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    # Validation and test loaders (no shuffling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
