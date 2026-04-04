"""
TruthLens - AI-Generated Media Detection Training Script
Trains a CNN model to classify images/videos as REAL or FAKE (AI-generated)
Using PyTorch with transfer learning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFile
import numpy as np

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Disable decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
IMG_SIZE = 224  # MobileNetV2 native resolution
BATCH_SIZE = 64  # Smaller batch for larger images
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = 'Dataset/train'  # Updated to use new dataset location
TEST_DIR = 'Dataset/test'  # Separate test directory
MAX_TRAIN_SAMPLES = None  # Use all samples for better generalization
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class imbalance handling
USE_CLASS_WEIGHTS = True  # Use weighted loss to handle imbalance

# Enable cuDNN benchmarking for faster training
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class ImageDataset(Dataset):
    """Custom dataset for loading images with augmentation."""

    def __init__(self, root_dir, transform=None, max_samples=None, balance_classes=False):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'Fake': 0, 'Real': 1}

        # Collect all samples per class
        class_samples = {'Fake': [], 'Real': []}
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        class_samples[class_name].append((img_path, idx))

        # Balance classes by undersampling the majority class
        if balance_classes:
            min_count = min(len(class_samples['Fake']), len(class_samples['Real']))
            class_samples['Fake'] = class_samples['Fake'][:min_count]
            class_samples['Real'] = class_samples['Real'][:min_count]
            print(f"  [Balanced] Fake: {len(class_samples['Fake'])}, Real: {len(class_samples['Real'])}")

        # Combine samples
        for class_name in self.class_to_idx.keys():
            self.samples.extend(class_samples[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGBA').convert('RGB')
            # Fast thumbnail resize before transforms (avoids decoding full resolution)
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            # Return a blank image if loading fails
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label


class TrainDataset(Dataset):
    """Training dataset - defined at module level for Windows pickle compatibility."""
    def __init__(self, indices, root_dir, transform):
        self.indices = indices
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {'Fake': 0, 'Real': 1}
        self.samples = []
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.samples[self.indices[idx]]
        try:
            image = Image.open(img_path).convert('RGBA').convert('RGB')
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label


class TestDataset(Dataset):
    """Test dataset - defined at module level for Windows pickle compatibility."""
    def __init__(self, indices, root_dir, transform):
        self.indices = indices
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {'Fake': 0, 'Real': 1}
        self.samples = []
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.samples[self.indices[idx]]
        try:
            image = Image.open(img_path).convert('RGBA').convert('RGB')
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label


class ValDataset(Dataset):
    """Validation dataset wrapper - uses TestDataset pattern with test_transform."""
    def __init__(self, indices, root_dir, transform):
        self.indices = indices
        self.root_dir = root_dir
        self.transform = transform  # test_transform: Resize + ToTensor + Normalize (no augmentation)
        self.class_to_idx = {'Fake': 0, 'Real': 1}
        self.samples = []
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.samples[self.indices[idx]]
        try:
            image = Image.open(img_path).convert('RGBA').convert('RGB')
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label

def create_dataloaders():
    """Create data loaders with proper train/test split and class imbalance handling."""

    # Training transforms with moderate augmentation
    # Tuned to avoid confusing filtered real photos with AI-generated images
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        # Gentle color jitter - preserves natural photo characteristics
        # Reduced to avoid making real filtered photos look like AI artifacts
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        # Random grayscale at lower probability
        transforms.RandomGrayscale(p=0.1),
        # Slight affine transforms (natural camera variation)
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        # Mild Gaussian blur (natural focus variation, not heavy filter simulation)
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms - only resizing and normalization
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load train and test datasets from separate directories
    # Balance train dataset by undersampling majority class
    train_dataset = ImageDataset(DATA_DIR, transform=None, balance_classes=True)
    test_dataset = ImageDataset(TEST_DIR, transform=None, balance_classes=False)  # Keep test set as-is

    # Count samples per class for class weight calculation
    fake_count = sum(1 for _, label in train_dataset.samples if label == 0)
    real_count = sum(1 for _, label in train_dataset.samples if label == 1)
    total_count = fake_count + real_count

    print(f"  - Train samples: {len(train_dataset)} (Fake: {fake_count:,}, Real: {real_count:,})")
    print(f"  - Test samples: {len(test_dataset)}")

    # No class weights needed - dataset is now balanced
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

    # Split train into train/val (90%/10%)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders with proper transforms
    train_loader = DataLoader(
        TrainDataset(train_subset.indices, DATA_DIR, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        ValDataset(val_subset.indices, DATA_DIR, test_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        TestDataset(list(range(len(test_dataset))), TEST_DIR, test_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_weights

def create_model():
    """
    Create a CNN model for deepfake detection.
    Uses transfer learning with MobileNetV2 for better feature extraction.
    Improved architecture with deeper classifier head.
    """
    # Load pretrained MobileNetV2 with ImageNet weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze base model initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier with deeper head for better feature learning
    num_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 2)  # Binary classification (2 classes)
    )

    return model

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def train_model():
    """Main training function."""

    print("=" * 60)
    print("TruthLens - AI-Generated Media Detection Training")
    print("=" * 60)
    print(f"\nUsing device: {DEVICE}")

    # Create dataloaders
    print("\n[1/4] Loading dataset...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders()

    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print(f"  - Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")

    # Create model
    print("\n[2/4] Creating model...")
    model = create_model().to(DEVICE)

    # Loss and optimizer - use class weights for imbalanced dataset
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    # Train full model with gradual unfreezing
    print(f"\n[3/4] Training...")
    for epoch in range(EPOCHS):
        # Unfreeze more layers as training progresses
        if epoch == 2:
            for param in model.features[-10:].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            print("  [Unfreezing] Last 10 backbone layers")
        if epoch == 5:
            for param in model.features[-20:].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
            print("  [Unfreezing] Last 20 backbone layers")
        if epoch == 10:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
            print("  [Unfreezing] All backbone layers")

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"  [Saved] Best model with val_acc: {val_acc:.2f}%")

        # Early stopping: stop if val_acc is 100% (no room for improvement)
        if val_acc >= 99.99:
            print("Early stopping triggered (validation accuracy saturated at ~100%)")
            break

        # Early stopping based on validation loss plateau
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset on any improvement
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered (no improvement for {max_patience} epochs)")
                break

    # Load best model
    model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))

    return model, history, test_loader

def evaluate_model(model, test_loader, device):
    """Evaluate the trained model on test data."""

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    return accuracy

def plot_training_history(history):
    """Plot and save training history."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history['train_acc'], label='Train Acc')
    axes[0].plot(history['val_acc'], label='Val Acc')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/training_history.png', dpi=150)
    print("\nTraining history plot saved to: models/training_history.png")
    plt.close()

def main():
    # Train model
    model, history, test_loader = train_model()

    # Evaluate
    evaluate_model(model, test_loader, DEVICE)

    # Save final model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/fine_tuned_model.pth')
    print("\nModel saved to: models/fine_tuned_model.pth")

    # Plot history
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
