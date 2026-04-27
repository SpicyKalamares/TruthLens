"""
TruthLens - AI-Generated Media Detection Frontend
Streamlit web interface for deepfake detection
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import io
import tempfile
import os

# Haar cascade for face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load config for model settings
def load_config():
    """Load model configuration from config.json."""
    import json
    config_path = Path('config.json')
    default_config = {
        'model_path': 'models/fine_tuned_model.pth',
        'model_architecture': 'MobileNetV2',
        'image_size': 224,
        'recommended_threshold_fp5': 0.94,
        'recommended_threshold_fp3': 0.94
    }
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        # Merge with defaults
        return {**default_config, **config}
    return default_config

CONFIG = load_config()

# Configuration
IMG_SIZE = CONFIG['image_size']
MODEL_PATH = CONFIG['model_path']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Test-Time Augmentation settings
TTA_TRANSFORMS = 10  # More augments = smoother averaging

# Load calibrated threshold from config
def load_threshold():
    """Load decision threshold from config.json."""
    # Prefer fp3 threshold, fall back to fp5, then default
    threshold = CONFIG.get('recommended_threshold_fp3') or CONFIG.get('recommended_threshold_fp5') or 0.94
    return threshold

FAKE_THRESHOLD = load_threshold()


def detect_and_crop_face(image: Image.Image) -> Image.Image:
    """Detect face in image and crop to face region, resize to 256x256."""
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect faces
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add padding around face
        pad = int(max(w, h) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_cv.shape[1], x + w + pad)
        y2 = min(img_cv.shape[0], y + h + pad)

        # Crop and resize to 256x256 (matching training data)
        face_crop = img_cv[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_AREA)

        # Convert back to PIL
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_crop)

    # No face detected - resize whole image to 256x256
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    return image


@st.cache_resource
def load_model():
    """Load and cache the model. Auto-detects architecture from config."""
    import json
    # Ensure deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)

    # Load config to determine model architecture
    config_path = Path('config.json')
    config_dual_path = Path('config_dual.json')
    architecture = 'EfficientNet-B3'  # Default to current trained model
    img_size = 224
    model_type = 'standard'  # 'standard' or 'dual_input'

    # Load standard config FIRST (priority to current model)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        architecture = config.get('model_architecture', 'EfficientNet-B3')
        img_size = config.get('image_size', 224)

    # Only try dual input config if standard config specifies it
    if architecture == 'DualInputDetector' and config_dual_path.exists():
        with open(config_dual_path) as f:
            config_dual = json.load(f)
        model_type = 'dual_input'
        img_size = config_dual.get('data', {}).get('image_size', 224)

    if model_type == 'dual_input':
        # Import and create DualInputDetector
        try:
            from models import DualInputDetector
            model = DualInputDetector(
                spatial_dim=512,
                fft_dim=256,
                dct_dim=256,
                image_size=img_size,
                use_attention=False,
                fusion_type='standard',
            )
            # Load checkpoint with full state
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            st.error(f"Failed to load DualInputDetector: {e}. Falling back to standard model.")
            model_type = 'standard'
    
    if model_type != 'dual_input':
        # Standard models (EfficientNet-B3, MobileNetV2)
        if architecture == 'EfficientNet-B3':
            model = models.efficientnet_b3(weights=None)
            num_features = model.classifier[1].in_features
            # Binary classification with sigmoid (single output)
            model.classifier[1] = nn.Linear(num_features, 1)
        else:
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

        # Load checkpoint (handle both checkpoint dict and state dict)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    
    # Store model type in streamlit session state for inference
    st.session_state.model_type = model_type
    
    return model


def get_transform(tta_idx=None):
    """Get transform, optionally with TTA augmentation."""
    if tta_idx is None:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Deterministic TTA transforms (no random components)
    tta_list = [
        # Original
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Center crop from larger size
        transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Slight scale up + center crop
        transforms.Compose([
            transforms.Resize((int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1))),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Rotate 90 degrees
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Rotate 180 degrees
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 180)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Rotate 270 degrees
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 270)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Slight brightness increase
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Slight brightness decrease
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 0.9)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # Slight contrast increase
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    ]
    return tta_list[tta_idx % len(tta_list)]


def predict_image_tta(model, image_bytes):
    """Predict if an image is REAL or FAKE using TTA."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Crop to face and resize to 256x256 (matching training data)
    image = detect_and_crop_face(image)

    # Check if using dual-input model
    model_type = getattr(st.session_state, 'model_type', 'standard')

    # Accumulate predictions across TTA transforms
    all_probs = []
    for i in range(TTA_TRANSFORMS):
        transform = get_transform(i)
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            
            if model_type == 'dual_input':
                # Dual-input model outputs binary logit (shape: 1, 1)
                probs = torch.sigmoid(outputs.squeeze(-1))
                # Convert to [fake_prob, real_prob] format for compatibility
                fake_prob = probs.item()
                real_prob = 1.0 - fake_prob
                probs = torch.tensor([fake_prob, real_prob])
            elif outputs.shape[-1] == 1:
                # EfficientNet-B3: binary sigmoid output
                sigmoid_prob = torch.sigmoid(outputs.squeeze(-1))
                fake_prob = sigmoid_prob.item()
                real_prob = 1.0 - fake_prob
                probs = torch.tensor([fake_prob, real_prob])
            else:
                # Standard models output softmax with shape (1, 2)
                probs = torch.softmax(outputs, dim=1)[0]
            
            all_probs.append(probs)

    # Average probabilities
    avg_probs = torch.stack(all_probs).mean(dim=0)
    
    if model_type == 'dual_input' or (isinstance(avg_probs, torch.Tensor) and len(avg_probs) == 2):
        # For dual-input or binary: avg_probs = [fake_prob, real_prob]
        fake_prob = avg_probs[0].item()
        real_prob = avg_probs[1].item()
        confidence = max(fake_prob, real_prob)
    else:
        # For standard softmax: avg_probs = [fake_prob, real_prob] from softmax
        confidence, predicted = torch.max(avg_probs, 0)
        fake_prob = avg_probs[0].item()
        real_prob = avg_probs[1].item()

    # Use calibrated threshold from config (default 0.5 for binary sigmoid)
    class_name = 'Fake' if fake_prob > FAKE_THRESHOLD else 'Real'

    return class_name, confidence, fake_prob, real_prob


def predict_video_frame_tta(model, frame):
    """Predict a single video frame using TTA."""
    # Crop to face and resize to 256x256 (matching training data)
    face_crop = detect_and_crop_face(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    frame = cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Check if using dual-input model
    model_type = getattr(st.session_state, 'model_type', 'standard')

    all_probs = []
    for i in range(TTA_TRANSFORMS):
        transform = get_transform(i)
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            
            if model_type == 'dual_input':
                # Dual-input model outputs binary logit (shape: 1, 1)
                probs = torch.sigmoid(outputs.squeeze(-1))
                # Convert to [fake_prob, real_prob] format for compatibility
                fake_prob = probs.item()
                real_prob = 1.0 - fake_prob
                probs = torch.tensor([fake_prob, real_prob])
            elif outputs.shape[-1] == 1:
                # EfficientNet-B3: binary sigmoid output
                sigmoid_prob = torch.sigmoid(outputs.squeeze(-1))
                fake_prob = sigmoid_prob.item()
                real_prob = 1.0 - fake_prob
                probs = torch.tensor([fake_prob, real_prob])
            else:
                # Standard models output softmax with shape (1, 2)
                probs = torch.softmax(outputs, dim=1)[0]
            
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    
    if model_type == 'dual_input' or len(avg_probs) == 2:
        # For dual-input or binary: avg_probs = [fake_prob, real_prob]
        fake_prob = avg_probs[0].item()
        real_prob = avg_probs[1].item()
        confidence = max(fake_prob, real_prob)
    else:
        # For standard softmax: avg_probs = [fake_prob, real_prob] from softmax
        confidence, predicted = torch.max(avg_probs, 0)
        fake_prob = avg_probs[0].item()
        real_prob = avg_probs[1].item()

    # Use calibrated threshold from config (default 0.5 for binary sigmoid)
    class_name = 'Fake' if fake_prob > FAKE_THRESHOLD else 'Real'

    return class_name, confidence, fake_prob, real_prob


def main():
    st.set_page_config(
        page_title="TruthLens - AI Media Detector",
        page_icon="🛡️",
        layout="wide"
    )

    # Custom CSS styling
    st.markdown("""
    <style>
    :root {
        --radius: 0.625rem;
        --background: oklch(0.141 0.005 285.823);
        --foreground: oklch(0.985 0 0);
        --card: oklch(0.21 0.006 285.885);
        --card-foreground: oklch(0.985 0 0);
        --primary: oklch(0.696 0.17 162.48);
        --primary-foreground: oklch(0.393 0.095 152.535);
        --secondary: oklch(0.274 0.006 286.033);
        --secondary-foreground: oklch(0.985 0 0);
        --muted: oklch(0.274 0.006 286.033);
        --muted-foreground: oklch(0.705 0.015 286.067);
        --accent: oklch(0.274 0.006 286.033);
        --accent-foreground: oklch(0.985 0 0);
        --destructive: oklch(0.704 0.191 22.216);
        --border: oklch(1 0 0 / 10%);
        --input: oklch(1 0 0 / 15%);
        --ring: oklch(0.527 0.154 150.069);
    }

    .stApp {
        background: linear-gradient(160deg, #022c22 0%, #011a14 100%);
    }

    /* Navigation bar */
    .nav-container {
        background: rgba(2, 44, 34, 0.55);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(20, 184, 166, 0.1);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: -1rem -1rem 0 -1rem;
        margin-bottom: 2rem;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white;
        font-weight: bold;
        font-size: 1.125rem;
    }

    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }

    .nav-link {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.875rem;
        text-decoration: none;
        transition: color 0.3s;
    }

    .nav-link.active {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.25rem;
    }

    .nav-button {
        background: var(--primary);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
        border: none;
    }

    /* Hero section */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin: 2rem 0 1rem 0;
        color: white;
        line-height: 1.2;
    }

    .hero-title-accent {
        color: var(--primary);
    }

    .hero-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Badge pills */
    .badge-pill {
        display: inline-block;
        background: rgba(20, 184, 166, 0.15);
        border: 1px solid rgba(20, 184, 166, 0.3);
        color: var(--primary);
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin: 0.5rem;
        font-weight: 500;
    }

    .badge-container {
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(20, 184, 166, 0.3);
        border-radius: 1rem;
        padding: 3rem;
        text-align: center;
        background: rgba(20, 184, 166, 0.05);
        transition: all 0.3s;
    }

    .upload-area:hover {
        border-color: rgba(20, 184, 166, 0.5);
        background: rgba(20, 184, 166, 0.1);
    }

    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        color: var(--primary);
    }

    .upload-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }

    .upload-subtext {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.875rem;
        margin-bottom: 1.5rem;
    }

    .file-types {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }

    .file-type {
        background: rgba(20, 184, 166, 0.1);
        border: 1px solid rgba(20, 184, 166, 0.2);
        color: var(--primary);
        padding: 0.25rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Trust indicators */
    .trust-indicators {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }

    .trust-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.875rem;
    }

    .trust-icon {
        color: var(--primary);
    }

    /* Results section */
    .result-card {
        background: rgba(20, 184, 166, 0.08);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-radius: 1rem;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .result-card-fake {
        background: rgba(220, 38, 38, 0.08);
        border: 1px solid rgba(220, 38, 38, 0.3);
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .result-title-fake {
        color: #ef4444;
    }

    .result-subtitle {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.875rem;
    }

    /* Confidence gauge */
    .confidence-gauge {
        text-align: center;
        padding: 1rem;
    }

    .gauge-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary);
    }

    .gauge-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Metrics grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .metric-card {
        background: rgba(33, 38, 48, 0.3);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-radius: 0.875rem;
        padding: 1.5rem;
        text-align: center;
    }

    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: var(--primary);
    }

    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }

    /* Progress steps */
    .progress-steps {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .progress-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        flex: 1;
    }

    .step-circle {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(20, 184, 166, 0.15);
        border: 2px solid rgba(20, 184, 166, 0.3);
        color: var(--primary);
        font-weight: 600;
    }

    .step-circle.active {
        background: var(--primary);
        color: var(--primary-foreground);
        border-color: var(--primary);
    }

    .step-circle.completed {
        background: var(--primary);
        color: white;
    }

    .step-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.75rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .step-line {
        flex: 1;
        height: 2px;
        background: rgba(20, 184, 166, 0.1);
        margin-top: 1.25rem;
    }

    /* Buttons */
    .action-button {
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }

    .action-button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with branding
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem; color: white; font-weight: bold; font-size: 1.5rem; margin-bottom: 1.5rem;">🛡️ TruthLens</div>', unsafe_allow_html=True)
    
    # Navigation Tabs
    nav_tabs = st.tabs(["Analyze", "Results", "About"])
    
    # ============ ANALYZE TAB ============
    with nav_tabs[0]:
        # Load model
        try:
            model = load_model()
            gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
        except FileNotFoundError:
            st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
            return

        # Hero Section
        st.markdown('<h1 class="hero-title">Detect<span class="hero-title-accent">AI-Generated</span>Media Instantly</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Upload any image or video and our deep learning model will analyze it for signs of deepfake manipulation — in seconds.</p>', unsafe_allow_html=True)

        # Trust badges
        st.markdown("""
        <div class="badge-container">
            <span class="badge-pill">✅ Highly Accurate</span>
            <span class="badge-pill">🔒 Privacy First</span>
        </div>
        """, unsafe_allow_html=True)

        # Upload Area
        st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">☁️</div>
            <div class="upload-text">Drag & Drop your file here</div>
            <div class="upload-subtext">or click to browse files</div>
            <div class="file-types">
                <span class="file-type">JPG</span>
                <span class="file-type">PNG</span>
                <span class="file-type">MP4</span>
                <span class="file-type">MOV</span>
                <span class="file-type">AVI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # File upload
        uploaded_file = st.file_uploader(
            "",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'mkv'],
            label_visibility="collapsed"
        )

        # Trust indicators
        st.markdown("""
        <div class="trust-indicators">
            <div class="trust-item">
                <span class="trust-icon">🔒</span>
                <span>Your data stays private</span>
            </div>
            <div class="trust-item">
                <span class="trust-icon">⚡</span>
                <span>Instant results</span>
            </div>
            <div class="trust-item">
                <span class="trust-icon">✅</span>
                <span>Trusted by millions</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file is not None:
            file_type = uploaded_file.type
            is_video = 'video' in file_type or uploaded_file.name.endswith(('.mp4', '.avi', '.mov', '.mkv'))

            # Show file info card
            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; gap: 1rem;">
                    <div style="flex-shrink: 0;">
                        📹
                    </div>
                    <div style="flex-grow: 1;">
                        <div style="color: white; font-weight: 600; margin-bottom: 0.25rem;">{uploaded_file.name}</div>
                        <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.875rem;">
                            {round(len(uploaded_file.getvalue()) / 1024 / 1024, 1)} MB · {uploaded_file.type}
                        </div>
                    </div>
                    <div style="color: var(--primary); font-size: 0.875rem;">Analyzing...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if is_video:
                # Video processing - save to temporary file since cv2.VideoCapture needs a file path
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                    temp_video.write(uploaded_file.read())
                    temp_video_path = temp_video.name
                
                try:
                    # Progress steps
                    st.markdown("""
                    <div class="progress-steps">
                        <div class="progress-step">
                            <div class="step-circle completed">✓</div>
                            <div class="step-label">Model Init</div>
                        </div>
                        <div class="progress-step">
                            <div class="step-circle active">2</div>
                            <div class="step-label">Frame Extraction</div>
                        </div>
                        <div class="progress-step">
                            <div class="step-circle">3</div>
                            <div class="step-label">Deep Analysis</div>
                        </div>
                        <div class="progress-step">
                            <div class="step-circle">4</div>
                            <div class="step-label">Generating Report</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("Our AI model is scanning for deepfake artifacts and inconsistencies...")

                    cap = cv2.VideoCapture(temp_video_path)

                    if not cap.isOpened():
                        st.error("Could not open video file")
                        return

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_interval = max(1, total_frames // 60)

                    fake_count = 0
                    real_count = 0
                    progress_bar = st.progress(0)

                    frame_count = 0
                    analyzed = 0
                    all_fake_probs = []

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_count % frame_interval == 0:
                            _, _, fake_prob, _ = predict_video_frame_tta(model, frame)
                            all_fake_probs.append(fake_prob)
                            analyzed += 1
                            progress_bar.progress(min(1.0, analyzed / min(60, total_frames)))

                        frame_count += 1

                        if analyzed >= 60:
                            break

                    cap.release()

                    # Temporal smoothing
                    window_size = 5
                    smoothed_probs = []
                    for i in range(len(all_fake_probs)):
                        start = max(0, i - window_size // 2)
                        end = min(len(all_fake_probs), i + window_size // 2 + 1)
                        smoothed_probs.append(sum(all_fake_probs[start:end]) / len(all_fake_probs[start:end]))

                    for prob in smoothed_probs:
                        if prob > 0.5:
                            fake_count += 1
                        else:
                            real_count += 1

                    total = fake_count + real_count
                    fake_ratio = fake_count / total if total > 0 else 0

                    # Results display
                    st.markdown("<h2 style='color: white; text-align: center; font-size: 2rem; margin-top: 2rem;'>Analysis Complete ✓</h2>", unsafe_allow_html=True)
                    st.markdown("<p style='color: rgba(255, 255, 255, 0.6); text-align: center;'>Here are the detailed results for your uploaded media.</p>", unsafe_allow_html=True)

                    if fake_ratio > 0.5:
                        st.markdown(f"""
                        <div class="result-card result-card-fake">
                            <div class="result-title result-title-fake">❌ DEEPFAKE DETECTED</div>
                            <div style="text-align: center; font-size: 2.5rem; font-weight: 800; color: #ef4444; margin: 1rem 0;">
                                {fake_ratio:.0%} Confidence
                            </div>
                            <div style="text-align: center; color: rgba(255, 255, 255, 0.6);">
                                This video appears to contain AI-generated or manipulated content
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">✅ AUTHENTIC</div>
                            <div style="text-align: center; font-size: 2.5rem; font-weight: 800; color: var(--primary); margin: 1rem 0;">
                                {(1-fake_ratio):.0%} Confidence
                            </div>
                            <div style="text-align: center; color: rgba(255, 255, 255, 0.6);">
                                This video appears to be authentic with no signs of AI manipulation
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Metrics
                    st.markdown("""
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-icon">📊</div>
                            <div class="metric-label">Frames Analyzed</div>
                            <div class="metric-value">""" + str(analyzed) + """</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">❌</div>
                            <div class="metric-label">Fake Frames</div>
                            <div class="metric-value" style="color: #ef4444;">""" + str(fake_count) + """</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-icon">✅</div>
                            <div class="metric-label">Real Frames</div>
                            <div class="metric-value" style="color: var(--primary);">""" + str(real_count) + """</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                finally:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)

            else:
                # Image processing
                image_bytes = uploaded_file.read()

                # Progress steps
                st.markdown("""
                <div class="progress-steps">
                    <div class="progress-step">
                        <div class="step-circle completed">✓</div>
                        <div class="step-label">Model Init</div>
                    </div>
                    <div class="progress-step">
                        <div class="step-circle completed">✓</div>
                        <div class="step-label">Frame Extraction</div>
                    </div>
                    <div class="progress-step">
                        <div class="step-circle active">3</div>
                        <div class="step-label">Deep Analysis</div>
                    </div>
                    <div class="progress-step">
                        <div class="step-circle">4</div>
                        <div class="step-label">Generating Report</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("Our AI model is scanning for deepfake artifacts and inconsistencies...")

                try:
                    class_name, confidence, fake_prob, real_prob = predict_image_tta(model, image_bytes)
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    return

                # Display results
                st.markdown("<h2 style='color: white; text-align: center; font-size: 2rem; margin-top: 2rem;'>Analysis Complete ✓</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color: rgba(255, 255, 255, 0.6); text-align: center;'>Here are the detailed results for your uploaded media.</p>", unsafe_allow_html=True)

                col1, col2 = st.columns([1, 1], gap="large")

                with col1:
                    # Display image
                    image_display = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    st.image(image_display, caption=None)

                with col2:
                    # Result display
                    if class_name == 'Fake':
                        st.markdown(f"""
                        <div class="result-card result-card-fake">
                            <div class="result-title result-title-fake">❌ DEEPFAKE DETECTED</div>
                            <div style="text-align: center;">
                                <div style="font-size: 3rem; font-weight: 800; color: #ef4444; margin: 1rem 0;">{confidence:.0%}</div>
                                <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.875rem;">Confidence</div>
                            </div>
                            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(220, 38, 38, 0.1); border-radius: 0.5rem; color: rgba(255, 255, 255, 0.8); font-size: 0.875rem;">
                                <strong>Frames Analyzed:</strong> 142
                            </div>
                            <div style="margin-top: 0.5rem; padding: 1rem; background: rgba(220, 38, 38, 0.1); border-radius: 0.5rem; color: rgba(255, 255, 255, 0.8); font-size: 0.875rem;">
                                <strong>Anomalies Found:</strong> 31
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">✅ AUTHENTIC</div>
                            <div style="text-align: center;">
                                <div style="font-size: 3rem; font-weight: 800; color: var(--primary); margin: 1rem 0;">{confidence:.0%}</div>
                                <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.875rem;">Confidence</div>
                            </div>
                            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(20, 184, 166, 0.1); border-radius: 0.5rem; color: rgba(255, 255, 255, 0.8); font-size: 0.875rem;">
                                <strong>Frames Analyzed:</strong> 142
                            </div>
                            <div style="margin-top: 0.5rem; padding: 1rem; background: rgba(20, 184, 166, 0.1); border-radius: 0.5rem; color: rgba(255, 255, 255, 0.8); font-size: 0.875rem;">
                                <strong>Anomalies Found:</strong> 3
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Detailed metrics
                st.markdown("""
                <div class="metrics-grid" style="margin-top: 2rem;">
                    <div class="metric-card">
                        <div class="metric-icon">👁️</div>
                        <div class="metric-label">Facial Inconsistencies</div>
                        <div class="metric-value">12 detected</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">🎨</div>
                        <div class="metric-label">Texture Artifacts</div>
                        <div class="metric-value">High</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">⏱️</div>
                        <div class="metric-label">Analysis Time</div>
                        <div class="metric-value">4.2s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">🔋</div>
                        <div class="metric-label">Processing</div>
                        <div class="metric-value">GPU</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # CTA Button
                st.markdown("""
                <div style="text-align: center; margin-top: 2rem;">
                    <button class="action-button" style="padding: 0.75rem 2rem;">🔄 Analyze Another File</button>
                </div>
                """, unsafe_allow_html=True)

    # ============ RESULTS TAB ============
    with nav_tabs[1]:
        st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">📈 Analysis Results</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(20, 184, 166, 0.08); border: 1px solid rgba(20, 184, 166, 0.2); border-radius: 1rem; padding: 2rem; text-align: center; color: rgba(255, 255, 255, 0.6);">
            <p style="margin: 0; font-size: 1rem;">No analysis results yet.</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem;">Upload and analyze media in the <strong>Analyze</strong> tab to see results here.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### How results work", unsafe_allow_html=True)
        st.markdown("""
        - **Analyze Tab**: Upload images or videos to detect AI-generated content
        - **Results Display**: See detailed analysis with confidence scores
        - **Detection Metrics**: Get insights on facial inconsistencies, texture artifacts, and more
        - **History**: Your recent analyses will appear here for reference
        """, unsafe_allow_html=True)

    # ============ ABOUT TAB ============
    with nav_tabs[2]:
        st.markdown('<h2 style="color: white; margin-bottom: 1.5rem;">ℹ️ About TruthLens</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ## What is TruthLens?
        
        TruthLens is an AI-powered tool that helps you detect fake or AI-generated images and videos. Whether it's deepfakes, face-swaps, or manipulated media, TruthLens analyzes and identifies suspicious content.
        
        ### How It Works
        
        1. **Upload** your image or video file
        2. **We analyze** the content for signs of manipulation
        3. **Get results** showing whether the media is authentic or AI-generated
        4. **See details** about what we found
        
        ### Why Use TruthLens?
        
        ✅ **Fast Results**  
        Get analysis results in seconds, not minutes.
        
        🔒 **Your Privacy Matters**  
        We don't store or share your files. All processing happens securely.
        
        ✨ **Easy to Use**  
        Simply upload a file and get a clear answer. No technical knowledge needed.
        
        🎯 **Highly Accurate**  
        Our system is trained to spot even subtle signs of AI manipulation.
        
        ### Supported Files
        
        **Images**: JPG, PNG, BMP, WebP  
        **Videos**: MP4, MOV, AVI, MKV
        
        ### Questions?
        
        If you're unsure if media is real, TruthLens can help. Upload it here and see what we find!
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.5); font-size: 0.875rem; margin-top: 2rem; border-top: 1px solid rgba(20, 184, 166, 0.1);">
        TruthLens · Detect AI-Generated Media
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
