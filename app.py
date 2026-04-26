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
        page_title="TruthLens - Deepfake Detector",
        page_icon="🔍",
        layout="centered"
    )

    st.title("🔍 TruthLens")
    st.markdown("**AI-Generated Media Detection**")
    st.markdown("---")

    # Load model
    try:
        model = load_model()
        gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
        st.success(f"Model loaded successfully | Device: {gpu_status}")
    except FileNotFoundError:
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return

    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image or video",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'mkv']
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type
        is_video = 'video' in file_type or uploaded_file.name.endswith(('.mp4', '.avi', '.mov', '.mkv'))

        if is_video:
            # Video processing - save to temporary file since cv2.VideoCapture needs a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name
            
            try:
                cap = cv2.VideoCapture(temp_video_path)

                if not cap.isOpened():
                    st.error("Could not open video file")
                    return

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_interval = max(1, total_frames // 60)  # Analyze up to 60 frames for better consistency

                fake_count = 0
                real_count = 0
                progress_bar = st.progress(0)

                st.markdown("**Analyzing video frames...**")

                frame_count = 0
                analyzed = 0
                all_fake_probs = []  # Store all probabilities for smoothing

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        _, _, fake_prob, _ = predict_video_frame_tta(model, frame)
                        all_fake_probs.append(fake_prob)
                        analyzed += 1
                        progress_bar.progress(analyzed / min(60, total_frames))

                    frame_count += 1

                    if analyzed >= 60:
                        break

                cap.release()

                # Temporal smoothing: apply rolling average window
                window_size = 5
                smoothed_probs = []
                for i in range(len(all_fake_probs)):
                    start = max(0, i - window_size // 2)
                    end = min(len(all_fake_probs), i + window_size // 2 + 1)
                    smoothed_probs.append(sum(all_fake_probs[start:end]) / len(all_fake_probs[start:end]))

                # Count smoothed predictions
                for prob in smoothed_probs:
                    if prob > 0.5:
                        fake_count += 1
                    else:
                        real_count += 1

                total = fake_count + real_count
                fake_ratio = fake_count / total if total > 0 else 0

                st.markdown("---")
                st.markdown("### Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("FAKE frames", fake_count)
                with col2:
                    st.metric("REAL frames", real_count)

                if fake_ratio > 0.5:
                    st.error(f"**Video classified as: FAKE** ({fake_ratio:.1%} AI-generated frames)")
                else:
                    st.success(f"**Video classified as: REAL** ({(1-fake_ratio):.1%} authentic frames)")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

        else:
            # Image processing
            image_bytes = uploaded_file.read()

            try:
                class_name, confidence, fake_prob, real_prob = predict_image_tta(model, image_bytes)
            except Exception as e:
                st.error(f"Error processing image: {e}")
                return

            # Display image
            st.image(uploaded_file, caption="Uploaded Image", width=350)

            # Results
            st.markdown("---")
            st.markdown("### Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", class_name)
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("Model Device", gpu_status)

            # Probability bars
            st.markdown("**Confidence breakdown:**")
            st.progress(fake_prob, text=f"FAKE: {fake_prob:.1%}")
            st.progress(real_prob, text=f"REAL: {real_prob:.1%}")

            if class_name == 'FAKE':
                st.error("⚠️ This image is likely **AI-generated**")
            else:
                st.success("✓ This image appears to be **authentic**")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "TruthLens Deepfake Detector | Built with PyTorch & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
