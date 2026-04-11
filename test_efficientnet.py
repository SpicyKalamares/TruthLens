"""Test EfficientNet-B3 model from train_improved.py"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image, ImageFile
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 300
BATCH_SIZE = 64
TEST_DIR = 'Dataset/test'
MODEL_PATH = 'models/best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model():
    model = models.efficientnet_b3(weights=None)
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
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            if self.transform:
                image = self.transform(image)
            return image, label


print("=" * 60)
print("Testing EfficientNet-B3 Model")
print("=" * 60)
print(f"Device: {DEVICE}")

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
print("\nLoading test dataset...")
test_dataset = ImageDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"  Test samples: {len(test_dataset)}")

# Load model
print("\nLoading model...")
model = create_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
print(f"  Loaded: {MODEL_PATH}")

model.eval()

# Evaluate
print("\nEvaluating...")
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

cm = confusion_matrix(all_labels, all_preds)
fake_acc = cm[0][0] / (cm[0][0] + cm[0][1]) * 100 if (cm[0][0] + cm[0][1]) > 0 else 0
real_acc = cm[1][1] / (cm[1][0] + cm[1][1]) * 100 if (cm[1][0] + cm[1][1]) > 0 else 0
fp_rate = cm[1][0] / (cm[1][0] + cm[1][1]) * 100 if (cm[1][0] + cm[1][1]) > 0 else 0
fn_rate = cm[0][1] / (cm[0][0] + cm[0][1]) * 100 if (cm[0][0] + cm[0][1]) > 0 else 0

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Overall Accuracy: {accuracy:.2f}%")
print(f"\nPer-class accuracy:")
print(f"  - Fake: {fake_acc:.2f}%")
print(f"  - Real: {real_acc:.2f}%")
print(f"\nError rates:")
print(f"  - False Positive Rate (Real->Fake): {fp_rate:.2f}%")
print(f"  - False Negative Rate (Fake->Real): {fn_rate:.2f}%")
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"                Fake    Real")
print(f"Actual Fake   {cm[0][0]:6d}  {cm[0][1]:6d}")
print(f"       Real   {cm[1][0]:6d}  {cm[1][1]:6d}")
