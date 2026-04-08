"""
TruthLens - Threshold Calibration Script
Find optimal decision threshold to minimize false positives on real images
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFile
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 64
TEST_DIR = 'Dataset/test'
MODEL_PATH = 'models/fine_tuned_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model():
    """Create model matching the training script architecture."""
    model = models.mobilenet_v2(weights=None)
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


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'Fake': 0, 'Real': 1}

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


def calibrate_threshold():
    print("=" * 60)
    print("TruthLens - Threshold Calibration")
    print("=" * 60)
    print(f"\nUsing device: {DEVICE}")

    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    print("\n[1/3] Loading test dataset...")
    test_dataset = ImageDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )
    print(f"  - Test samples: {len(test_dataset)}")

    # Load model
    print("\n[2/3] Loading model...")
    model = create_model().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print(f"  Loaded: {MODEL_PATH}")
    else:
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        return

    model.eval()

    # Get predictions
    print("\n[3/3] Running evaluation...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 0].cpu().numpy())  # Fake probability
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Separate by class
    real_probs = all_probs[all_labels == 1]  # Fake prob for real images
    fake_probs = all_probs[all_labels == 0]  # Fake prob for fake images

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    # Test different thresholds
    thresholds = np.arange(0.50, 0.95, 0.01)
    results = []

    print("\nThreshold Analysis:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'FP Rate':<12} {'FN Rate':<12} {'Accuracy':<12}")
    print("-" * 60)

    for thresh in thresholds:
        preds_fake = all_probs > thresh
        preds_real = all_probs <= thresh

        # False Positive Rate: real images classified as fake
        fp_rate = (real_probs > thresh).sum() / len(real_probs) * 100

        # False Negative Rate: fake images classified as real
        fn_rate = (fake_probs <= thresh).sum() / len(fake_probs) * 100

        # Overall accuracy
        accuracy = ((all_probs > thresh) == (all_labels == 0)).sum() / len(all_labels) * 100

        results.append({
            'threshold': float(thresh),
            'fp_rate': float(fp_rate),
            'fn_rate': float(fn_rate),
            'accuracy': float(accuracy)
        })

        if thresh % 0.05 < 0.01:  # Print every 0.05
            print(f"{thresh:<12.2f} {fp_rate:<12.2f} {fn_rate:<12.2f} {accuracy:<12.2f}")

    # Find optimal thresholds
    print("\n" + "=" * 60)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 60)

    # Threshold for <5% false positive rate
    fp5_results = [r for r in results if r['fp_rate'] <= 5.0]
    if fp5_results:
        thresh_fp5 = max(fp5_results, key=lambda x: x['accuracy'])
        print(f"\nFor <=5% False Positive Rate:")
        print(f"  Threshold: {thresh_fp5['threshold']:.2f}")
        print(f"  FP Rate: {thresh_fp5['fp_rate']:.2f}%")
        print(f"  FN Rate: {thresh_fp5['fn_rate']:.2f}%")
        print(f"  Accuracy: {thresh_fp5['accuracy']:.2f}%")
    else:
        print("\nNo threshold achieves <=5% FP rate")

    # Threshold for <3% false positive rate (stricter)
    fp3_results = [r for r in results if r['fp_rate'] <= 3.0]
    if fp3_results:
        thresh_fp3 = max(fp3_results, key=lambda x: x['accuracy'])
        print(f"\nFor <=3% False Positive Rate:")
        print(f"  Threshold: {thresh_fp3['threshold']:.2f}")
        print(f"  FP Rate: {thresh_fp3['fp_rate']:.2f}%")
        print(f"  FN Rate: {thresh_fp3['fn_rate']:.2f}%")
        print(f"  Accuracy: {thresh_fp3['accuracy']:.2f}%")

    # Threshold for balanced error rates
    balanced_results = sorted(results, key=lambda x: abs(x['fp_rate'] - x['fn_rate']))
    thresh_balanced = balanced_results[0]
    print(f"\nFor Balanced Error Rates:")
    print(f"  Threshold: {thresh_balanced['threshold']:.2f}")
    print(f"  FP Rate: {thresh_balanced['fp_rate']:.2f}%")
    print(f"  FN Rate: {thresh_balanced['fn_rate']:.2f}%")
    print(f"  Accuracy: {thresh_balanced['accuracy']:.2f}%")

    # Save config
    config = {
        'model_path': MODEL_PATH,
        'current_threshold': 0.80,
        'recommended_threshold_fp5': thresh_fp5['threshold'] if fp5_results else None,
        'recommended_threshold_fp3': thresh_fp3['threshold'] if fp3_results else None,
        'recommended_threshold_balanced': thresh_balanced['threshold'],
        'current_fp_rate': float((real_probs > 0.80).sum() / len(real_probs) * 100),
        'current_fn_rate': float((fake_probs <= 0.80).sum() / len(fake_probs) * 100),
        'full_results': results
    }

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: config.json")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: FP/FN rates vs threshold
    ax1 = axes[0]
    fp_rates = [r['fp_rate'] for r in results]
    fn_rates = [r['fn_rate'] for r in results]

    ax1.plot(thresholds, fp_rates, 'r-', label='False Positive Rate (Real->Fake)', linewidth=2)
    ax1.plot(thresholds, fn_rates, 'b-', label='False Negative Rate (Fake->Real)', linewidth=2)
    ax1.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% target')
    ax1.axhline(y=3, color='r', linestyle=':', alpha=0.5, label='3% target')
    ax1.axvline(x=0.80, color='gray', linestyle='--', alpha=0.5, label='Current (0.80)')
    if fp5_results:
        ax1.axvline(x=thresh_fp5['threshold'], color='green', linestyle='-', linewidth=2,
                   label=f"Recommended ({thresh_fp5['threshold']:.2f})")
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Error Rate (%)')
    ax1.set_title('Error Rates vs Decision Threshold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of probabilities
    ax2 = axes[1]
    ax2.hist(fake_probs, bins=50, alpha=0.7, label='Fake Images', color='red', density=True)
    ax2.hist(real_probs, bins=50, alpha=0.7, label='Real Images', color='blue', density=True)
    ax2.axvline(x=0.80, color='gray', linestyle='--', linewidth=2, label='Current (0.80)')
    if fp5_results:
        ax2.axvline(x=thresh_fp5['threshold'], color='green', linestyle='-', linewidth=2,
                   label=f"Recommended ({thresh_fp5['threshold']:.2f})")
    ax2.set_xlabel('Fake Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Prediction Probabilities')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/threshold_calibration.png', dpi=150)
    print(f"Calibration plot saved to: analysis/threshold_calibration.png")

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)

    # Summary
    print("\nSUMMARY:")
    print(f"  Current threshold (0.80) produces {config['current_fp_rate']:.2f}% false positives")
    if fp5_results:
        print(f"  Changing to {thresh_fp5['threshold']:.2f} will reduce FP to ~{thresh_fp5['fp_rate']:.2f}%")
        print(f"  But will increase FN from {config['current_fn_rate']:.2f}% to {thresh_fp5['fn_rate']:.2f}%")
    print("\nNEXT STEP:")
    print("  Update app.py to use the calibrated threshold from config.json")


if __name__ == '__main__':
    calibrate_threshold()
