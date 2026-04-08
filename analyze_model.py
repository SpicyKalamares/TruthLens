"""
TruthLens - Model Analysis Script
Diagnose current model performance and identify failure modes
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFile
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 64
TEST_DIR = 'Dataset/test'
MODEL_PATH = 'models/fine_tuned_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture matching train_model.py (how the model was actually saved)
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
        self.image_paths = []

        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        self.samples.append((img_path, idx))
                        self.image_paths.append(img_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label, img_path


def analyze_model():
    print("=" * 60)
    print("TruthLens - Model Performance Analysis")
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
        print("  Please train the model first with: python train_model.py")
        return

    model.eval()

    # Run evaluation
    print("\n[3/3] Running evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Per-class accuracy
    fake_mask = all_labels == 0
    real_mask = all_labels == 1

    fake_accuracy = 100. * (all_preds[fake_mask] == all_labels[fake_mask]).sum() / fake_mask.sum()
    real_accuracy = 100. * (all_preds[real_mask] == all_labels[real_mask]).sum() / real_mask.sum()

    print(f"\nPer-Class Accuracy:")
    print(f"  - Fake images: {fake_accuracy:.2f}% ({fake_mask.sum()} samples)")
    print(f"  - Real images: {real_accuracy:.2f}% ({real_mask.sum()} samples)")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Fake    Real")
    print(f"Actual Fake   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Real   {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Calculate error rates
    false_negative_rate = cm[0][1] / (cm[0][0] + cm[0][1]) * 100  # Fake classified as Real
    false_positive_rate = cm[1][0] / (cm[1][0] + cm[1][1]) * 100  # Real classified as Fake

    print(f"\nError Analysis:")
    print(f"  - False Negative Rate (Fake->Real): {false_negative_rate:.2f}%")
    print(f"  - False Positive Rate (Real->Fake): {false_positive_rate:.2f}%")

    # Find hardest examples
    print("\n" + "=" * 60)
    print("HARDEST EXAMPLES (Low confidence predictions)")
    print("=" * 60)

    # Get confidence for correct predictions
    correct_mask = all_preds == all_labels
    correct_confidences = all_probs[correct_mask]
    correct_paths = [all_paths[i] for i in range(len(all_paths)) if correct_mask[i]]

    # Sort by confidence (ascending)
    sorted_indices = np.argsort(correct_confidences.max(axis=1))

    print("\n10 Hardest Correct Predictions:")
    for i, idx in enumerate(sorted_indices[:10]):
        conf = correct_confidences[idx]
        path = correct_paths[idx]
        pred_label = "Fake" if all_preds[correct_mask][idx] == 0 else "Real"
        true_label = "Fake" if all_labels[correct_mask][idx] == 0 else "Real"
        print(f"  {i+1}. {os.path.basename(path)[:50]}")
        print(f"     True: {true_label}, Pred: {pred_label}, Conf: {conf.max():.4f}")
        print(f"     [Fake: {conf[0]:.4f}, Real: {conf[1]:.4f}]")

    # Find misclassified examples
    incorrect_mask = all_preds != all_labels
    incorrect_count = incorrect_mask.sum()

    if incorrect_count > 0:
        print(f"\n{incorrect_count} Misclassified Examples:")
        incorrect_indices = np.where(incorrect_mask)[0]

        for i, idx in enumerate(incorrect_indices[:10]):
            path = all_paths[idx]
            pred_label = "Fake" if all_preds[idx] == 0 else "Real"
            true_label = "Fake" if all_labels[idx] == 0 else "Real"
            conf = all_probs[idx]
            print(f"  {i+1}. {os.path.basename(path)[:50]}")
            print(f"     True: {true_label}, Pred: {pred_label}")
            print(f"     [Fake: {conf[0]:.4f}, Real: {conf[1]:.4f}]")

    # Save analysis report
    report = {
        'model_path': MODEL_PATH,
        'test_samples': len(test_dataset),
        'accuracy': accuracy,
        'fake_accuracy': fake_accuracy,
        'real_accuracy': real_accuracy,
        'confusion_matrix': cm.tolist(),
        'false_negative_rate': false_negative_rate,
        'false_positive_rate': false_positive_rate,
        'classification_report': classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True)
    }

    os.makedirs('analysis', exist_ok=True)
    with open('analysis/model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: analysis/model_report.json")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Fake', 'Real'])
    plt.yticks([0, 1], ['Fake', 'Real'])
    plt.tight_layout()
    plt.savefig('analysis/confusion_matrix.png', dpi=150)
    print(f"Confusion matrix plot saved to: analysis/confusion_matrix.png")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if false_positive_rate > 10:
        print("  ⚠ High False Positive Rate (Real→Fake):")
        print("    - Model is too aggressive at classifying images as Fake")
        print("    - Consider raising the decision threshold")
        print("    - Add more diverse real images to training")

    if false_negative_rate > 10:
        print("  ⚠ High False Negative Rate (Fake→Real):")
        print("    - Model is missing AI-generated images")
        print("    - Consider lowering the decision threshold")
        print("    - Add more varied fake images to training")

    if accuracy < 90:
        print("  ⚠ Low Overall Accuracy:")
        print("    - Model may be underfitting")
        print("    - Consider training longer or using a larger model")
        print("    - Check for data quality issues")


if __name__ == '__main__':
    analyze_model()
