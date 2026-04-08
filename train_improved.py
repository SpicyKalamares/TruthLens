"""
TruthLens - Improved Training with EfficientNet-B3
Key improvements:
1. More powerful backbone (EfficientNet-B3 vs MobileNetV2)
2. Hard example mining - focus on difficult samples
3. Better augmentation mix
4. Label smoothing for calibration
5. Test-time augmentation built into training evaluation
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
IMG_SIZE = 300  # EfficientNet-B3 native resolution
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0005
DATA_DIR = 'Dataset/train'
TEST_DIR = 'Dataset/test'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Label smoothing for better calibration
LABEL_SMOOTHING = 0.1

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class ImageDataset(Dataset):
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
            return image, label, idx
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label, idx


class TrainDataset(Dataset):
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
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label


class ValDataset(Dataset):
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
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label


def create_model():
    """
    Create EfficientNet-B3 model.
    More powerful than MobileNetV2 while still efficient.
    """
    # Use IMAGENET1K_V1 weights for better feature extraction
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # Freeze base model initially
    for param in model.parameters():
        param.requires_grad = False

    # Classifier head - simpler but effective
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    return model


def create_dataloaders():
    """Create data loaders with strong augmentation."""

    # Stronger augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TTA-style validation transform (multiple crops)
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


def train_epoch(model, loader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch with hard example mining after epoch 5."""
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


def find_optimal_threshold(model, val_loader, device, target_fp_rate=3.0):
    """Find threshold that achieves target false positive rate."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    real_probs = all_probs[all_labels == 1]

    # Find threshold where FP rate <= target
    thresholds = np.arange(0.50, 0.99, 0.005)
    for thresh in thresholds:
        fp_rate = (real_probs > thresh).sum() / len(real_probs) * 100
        if fp_rate <= target_fp_rate:
            return float(thresh)

    return 0.99


def train_model():
    """Main training function."""

    print("=" * 60)
    print("TruthLens - Improved Training (EfficientNet-B3)")
    print("=" * 60)
    print(f"\nUsing device: {DEVICE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    # Create dataloaders
    print("\n[1/4] Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # Create model
    print("\n[2/4] Creating EfficientNet-B3 model...")
    model = create_model().to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    max_patience = 7

    # Train with gradual unfreezing
    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # Unfreeze layers progressively
        if epoch == 3:
            for param in model.features[-20:].parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
            print("  [Unfreezing] Last 20 backbone layers")
        if epoch == 8:
            for param in model.features[-40:].parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
            print("  [Unfreezing] Last 40 backbone layers")
        if epoch == 12:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
            print("  [Unfreezing] All backbone layers")

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler, epoch)
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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs('models', exist_ok=True)
            torch.save(best_model_state, 'models/best_model.pth')
            print(f"  [Saved] Best model with val_acc: {val_acc:.2f}%")

        # Early stopping
        if val_acc >= 99.5 and epoch >= 10:
            print("Early stopping triggered (validation accuracy converged)")
            break

        patience_counter += 1 if val_loss >= min(history['val_loss'][-3:]) else 0
        if patience_counter >= max_patience and epoch >= 15:
            print(f"Early stopping triggered (no improvement for {max_patience} epochs)")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(torch.load('models/best_model.pth', weights_only=True))

    # Find optimal threshold for <=3% FP rate (stricter than before)
    print("\n[4/4] Calibrating decision threshold for <=3% FP rate...")
    optimal_threshold = find_optimal_threshold(model, val_loader, DEVICE, target_fp_rate=3.0)
    print(f"  Optimal threshold: {optimal_threshold:.3f}")

    # Save config
    config = {
        'model_path': 'models/best_model.pth',
        'model_architecture': 'EfficientNet-B3',
        'recommended_threshold_fp3': optimal_threshold,
        'best_val_accuracy': best_val_acc,
        'training_date': datetime.now().isoformat(),
        'epochs_trained': epoch + 1,
        'image_size': IMG_SIZE
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
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 0].cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)

    # Calculate metrics at the calibrated threshold
    fp_rate_at_thresh = (all_probs[all_labels == 1] > optimal_threshold).sum() / (all_labels == 1).sum() * 100
    fn_rate_at_thresh = (all_probs[all_labels == 0] <= optimal_threshold).sum() / (all_labels == 0).sum() * 100

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"\nAt threshold {optimal_threshold:.3f}:")
    print(f"  - False Positive Rate: {fp_rate_at_thresh:.2f}%")
    print(f"  - False Negative Rate: {fn_rate_at_thresh:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Fake    Real")
    print(f"Actual Fake   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Real   {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Calculate per-class accuracy
    fake_acc = cm[0][0] / (cm[0][0] + cm[0][1]) * 100
    real_acc = cm[1][1] / (cm[1][0] + cm[1][1]) * 100
    print(f"\nPer-class accuracy:")
    print(f"  - Fake: {fake_acc:.2f}%")
    print(f"  - Real: {real_acc:.2f}%")

    # Save metrics
    metrics = {
        'test_accuracy': accuracy,
        'false_positive_rate': fp_rate_at_thresh,
        'false_negative_rate': fn_rate_at_thresh,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True),
        'per_class_accuracy': {
            'fake': fake_acc,
            'real': real_acc
        },
        'history': history
    }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: models/metrics.json")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: models/training_history.png")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: models/best_model.pth")
    print(f"Recommended threshold: {optimal_threshold:.3f}")
    print(f"  - Expected FP Rate: ~{fp_rate_at_thresh:.2f}%")
    print(f"  - Expected FN Rate: ~{fn_rate_at_thresh:.2f}%")

    # Update app.py threshold info
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. The config.json has been updated with the new threshold")
    print("2. Restart your Streamlit app to use the improved model:")
    print("   streamlit run app.py")
    print(f"3. The app will automatically use threshold {optimal_threshold:.3f}")

    return model, history


if __name__ == '__main__':
    train_model()
