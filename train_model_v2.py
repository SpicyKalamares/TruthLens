"""
TruthLens v2 - Improved AI-Generated Media Detection Training
Fixes:
1. Consistent architecture between training and inference
2. Better validation threshold calibration
3. Persistent metrics logging
4. Improved data augmentation
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFile
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = 'Dataset/train'
TEST_DIR = 'Dataset/test'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable cuDNN benchmarking
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class ImageDataset(Dataset):
    """Dataset for loading images."""

    def __init__(self, root_dir, transform=None, balance_classes=False):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'Fake': 0, 'Real': 1}

        class_samples = {'Fake': [], 'Real': []}
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        class_samples[class_name].append((img_path, idx))

        if balance_classes:
            min_count = min(len(class_samples['Fake']), len(class_samples['Real']))
            class_samples['Fake'] = class_samples['Fake'][:min_count]
            class_samples['Real'] = class_samples['Real'][:min_count]
            print(f"  [Balanced] Fake: {len(class_samples['Fake'])}, Real: {len(class_samples['Real'])}")

        for class_name in self.class_to_idx.keys():
            self.samples.extend(class_samples[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label


class TrainDataset(Dataset):
    """Training dataset - module level for Windows pickle compatibility."""
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
            image = Image.open(img_path).convert('RGB')
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
    """Validation dataset - module level for Windows pickle compatibility."""
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
            image = Image.open(img_path).convert('RGB')
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label


def create_model():
    """
    Create model with architecture matching app.py exactly.
    This ensures weights can be loaded without issues.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze base model initially
    for param in model.parameters():
        param.requires_grad = False

    # Classifier head - MUST match app.py exactly
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
        nn.Linear(64, 2)
    )

    return model


def create_dataloaders():
    """Create data loaders with balanced training."""

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms - no augmentation
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = ImageDataset(DATA_DIR, transform=None, balance_classes=True)
    test_dataset = ImageDataset(TEST_DIR, transform=None, balance_classes=False)

    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")

    # Split train into train/val (90%/10%)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        TrainDataset(train_subset.indices, DATA_DIR, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        ValDataset(val_subset.indices, DATA_DIR, test_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        ValDataset(list(range(len(test_dataset))), TEST_DIR, test_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
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


def find_optimal_threshold(model, val_loader, device, target_fp_rate=5.0):
    """Find threshold that achieves target false positive rate."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 0].cpu().numpy())  # Fake probability
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Get fake probs for real images (class 1)
    real_probs = all_probs[all_labels == 1]

    # Find threshold where FP rate <= target
    thresholds = np.arange(0.50, 0.99, 0.01)
    for thresh in thresholds:
        fp_rate = (real_probs > thresh).sum() / len(real_probs) * 100
        if fp_rate <= target_fp_rate:
            return float(thresh)

    return 0.99  # Default to very strict threshold


def train_model():
    """Main training function."""

    print("=" * 60)
    print("TruthLens v2 - AI-Generated Media Detection Training")
    print("=" * 60)
    print(f"\nUsing device: {DEVICE}")

    # Create dataloaders
    print("\n[1/4] Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # Create model
    print("\n[2/4] Creating model...")
    model = create_model().to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5

    # Train with gradual unfreezing
    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # Unfreeze layers progressively
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

        # Early stopping
        if val_acc >= 99.5 and epoch >= 5:
            print("Early stopping triggered (validation accuracy converged)")
            break

        patience_counter += 1 if val_loss >= min(history['val_loss']) else 0
        if patience_counter >= max_patience and epoch >= 10:
            print(f"Early stopping triggered (no improvement for {max_patience} epochs)")
            break

    # Load best model
    model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))

    # Find optimal threshold
    print("\n[4/4] Calibrating decision threshold...")
    optimal_threshold = find_optimal_threshold(model, val_loader, DEVICE, target_fp_rate=5.0)
    print(f"  Optimal threshold for <=5% FP rate: {optimal_threshold:.2f}")

    # Save config
    config = {
        'model_path': 'models/best_model.pth',
        'recommended_threshold_fp5': optimal_threshold,
        'best_val_accuracy': best_val_acc,
        'training_date': datetime.now().isoformat(),
        'epochs_trained': epoch + 1
    }

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to: config.json")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(DEVICE), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Fake    Real")
    print(f"Actual Fake   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Real   {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Save metrics
    metrics = {
        'test_accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True),
        'optimal_threshold': optimal_threshold,
        'history': history
    }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: models/metrics.json")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_acc'], label='Train Acc')
    axes[0].plot(history['val_acc'], label='Val Acc')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150)
    print(f"Training history plot saved to: models/training_history.png")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: models/best_model.pth")
    print(f"Threshold for inference: {optimal_threshold:.2f}")
    print(f"  - This gives <=5% false positive rate on real images")
    print(f"  - Update app.py to use this threshold")

    return model, history


if __name__ == '__main__':
    train_model()
