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
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
IMG_SIZE = 224  # MobileNetV2 native resolution
BATCH_SIZE = 64  # Smaller batch for larger images
EPOCHS = 15
LEARNING_RATE = 0.001
DATA_DIR = 'Dataset/deepfake-vs-real-60k'
TRAIN_SPLIT = 0.8  # 80% train, 20% test
MAX_TRAIN_SAMPLES = 30000  # Stratified subset — faster training, minimal accuracy loss
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable cuDNN benchmarking for faster training
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class ImageDataset(Dataset):
    """Custom dataset for loading images with augmentation."""

    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'Fake': 0, 'Real': 1}

        # Collect all samples
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloaders():
    """Create data loaders with proper train/test split."""

    # Training transforms with strong augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms - only resizing and normalization
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = ImageDataset(DATA_DIR, transform=None)

    # Sample subset if MAX_TRAIN_SAMPLES is set
    if MAX_TRAIN_SAMPLES and len(full_dataset) > MAX_TRAIN_SAMPLES:
        all_indices = list(range(len(full_dataset)))
        np.random.seed(42)
        np.random.shuffle(all_indices)
        full_dataset_indices = all_indices[:MAX_TRAIN_SAMPLES]
    else:
        full_dataset_indices = list(range(len(full_dataset)))

    # Split into train (80%) and test (20%)
    train_size = int(TRAIN_SPLIT * len(full_dataset_indices))
    test_size = len(full_dataset_indices) - train_size

    # Use indices for proper splitting
    indices = list(range(len(full_dataset_indices)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create train and test datasets with appropriate transforms
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
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    class TestDataset(Dataset):
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
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    train_dataset = TrainDataset(train_indices, DATA_DIR, train_transform)
    test_dataset = TestDataset(test_indices, DATA_DIR, test_transform)

    # Split train into train/val (90%/10% of train set)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap val_subset to apply test_transform
    class ValDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    val_dataset = ValDataset(val_subset, test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def create_model():
    """
    Create a CNN model for deepfake detection.
    Uses transfer learning with MobileNetV2 for better feature extraction.
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze base model initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    num_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 2)  # Binary classification (2 classes)
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

def train_model():
    """Main training function."""

    print("=" * 60)
    print("TruthLens - AI-Generated Media Detection Training")
    print("=" * 60)
    print(f"\nUsing device: {DEVICE}")

    # Create dataloaders
    print("\n[1/4] Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    print(f"  - Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Train/Val/Test split: {int(TRAIN_SPLIT*100)}%/{int((1-TRAIN_SPLIT)*100)}%/{int((1-TRAIN_SPLIT)*100)}%")

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
            for param in model.features[-15:].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
            print("  [Unfreezing] Last 15 backbone layers")

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

        # Early stopping based on validation loss plateau
        if len(history['val_loss']) > 5:
            if val_loss >= min(history['val_loss'][-5:]):
                patience_counter += 1
                if patience_counter >= max_patience:
                    print("Early stopping triggered")
                    break
            else:
                patience_counter = 0

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
    torch.save(model.state_dict(), 'models/deepfake_detector.pth')
    print("\nModel saved to: models/deepfake_detector.pth")

    # Plot history
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
