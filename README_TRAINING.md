# TruthLens - AI-Generated Media Detection Training

## Overview

This training pipeline detects AI-generated images and videos (deepfakes) using deep learning with PyTorch.

## Dataset Structure

```
Dataset/
├── train/
│   ├── REAL/      # Real images/videos
│   └── FAKE/      # AI-generated images/videos
├── test/
│   ├── REAL/
│   └── FAKE/
└── videos/        # Optional: raw videos for frame extraction
    ├── REAL/
    └── FAKE/
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Extract Frames from Videos (Optional)

If you have videos instead of images:

```bash
python extract_frames.py --input Dataset/videos --output Dataset/extracted_frames --interval 30
```

Then copy extracted frames to `Dataset/train/REAL` and `Dataset/train/FAKE`.

### Step 2: Train the Model

```bash
python train_model.py
```

This will:
- Load and augment your dataset
- Train a MobileNetV2-based CNN
- Fine-tune the model
- Save the best model to `models/best_model.pth`
- Save final model to `models/deepfake_detector.pth`
- Generate training history plots

### Step 3: Make Predictions

**Single image:**
```bash
python predict.py --input path/to/image.jpg
```

**Video:**
```bash
python predict.py --input path/to/video.mp4
```

**Batch folder:**
```bash
python predict.py --input path/to/folder/
```

## Model Architecture

- **Base:** MobileNetV2 (ImageNet pretrained)
- **Head:** Dropout + Linear(128) + ReLU + BatchNorm + Dropout + Linear(2)
- **Output:** Softmax (binary classification)

## Training Features

- Two-phase training (frozen base → fine-tuning)
- Data augmentation (rotation, flip, zoom, color jitter)
- Early stopping
- Learning rate scheduling
- Best model checkpointing

## Output Files

After training:
- `models/deepfake_detector.pth` - Final model
- `models/best_model.pth` - Best validation accuracy model
- `models/training_history.png` - Training curves
