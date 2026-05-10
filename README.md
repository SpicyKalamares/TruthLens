# TruthLens - AI-Generated Media Detection

A Streamlit web application for detecting AI-generated images using deep learning.

## What app.py Does

**app.py** is the main web interface for analyzing images:

1. **Loads Configuration** - Reads `config.json` to get model path, architecture type, and detection threshold
2. **Detects Faces** - Uses OpenCV Haar Cascade to automatically find and crop faces from uploaded images
3. **Loads Model** - Supports multiple architectures:
   - **EfficientNet-B3** (default, 98.96% accuracy)
   - **MobileNetV2** (faster alternative)
   - **DualInputDetector** (advanced dual-branch model)
4. **Applies Test-Time Augmentation (TTA)** - Runs 10 different image variations (rotations, crops, flips) and averages predictions for more robust results
5. **Makes Predictions** - Outputs confidence score (0.0-1.0) indicating probability the image is AI-generated
6. **Displays Results** - Shows prediction, confidence, and visual analysis in the Streamlit interface

## How to Run

```bash
streamlit run app.py
```

Open browser to `http://localhost:8501`

### Features
- Upload images (JPG, PNG, GIF)
- Automatic face detection and preprocessing
- Real-time predictions with confidence scores
- GPU acceleration (auto-detects CUDA)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.json`:
```json
{
  "model_path": "models/best_model_efficientnet_b3.pth",
  "model_architecture": "EfficientNet-B3",
  "image_size": 224,
  "recommended_threshold_fp3": 0.5
}
```

**Available options:**
- `model_architecture`: "EfficientNet-B3" | "MobileNetV2" | "DualInputDetector"
- `recommended_threshold_fp3`: Detection threshold (0.0-1.0)
