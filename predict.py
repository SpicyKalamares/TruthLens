"""
TruthLens - AI-Generated Media Detection (Prediction)
Use trained model to classify images/videos as REAL or FAKE
Using PyTorch
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'models/deepfake_detector.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepfakeDetector:
    def __init__(self, model_path=MODEL_PATH):
        """Load the trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Please train the model first using: python train_model.py"
            )

        self.device = DEVICE
        self.class_names = ['Fake', 'Real']

        # Create model architecture
        model = models.mobilenet_v2(weights=None)
        num_features = model.last_channel
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()

        self.model = model

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """Load and preprocess an image for prediction."""
        image = Image.open(str(image_path)).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_image(self, image_path):
        """
        Predict if an image is REAL or FAKE.

        Returns:
            tuple: (class_name, confidence, is_fake)
        """
        img = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(img)
            probabilities = torch.softmax(outputs, dim=1)[0]

        confidence, predicted = torch.max(probabilities, 0)
        class_name = self.class_names[predicted.item()]
        is_fake = class_name == 'FAKE'

        return class_name, confidence.item(), is_fake

    def predict_video(self, video_path, frame_interval=30):
        """
        Predict if a video is REAL or FAKE by analyzing frames.

        Returns:
            tuple: (class_name, confidence, is_fake, frame_results)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_results = []
        fake_count = 0

        print(f"Analyzing video: {video_path}")

        with tqdm(total=total_frames, desc="Analyzing") as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    img_tensor = self.transform(image).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(img_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0]

                    confidence, predicted = torch.max(probabilities, 0)
                    is_fake = predicted.item() == 0

                    if is_fake:
                        fake_count += 1

                    frame_results.append({
                        'frame': frame_count,
                        'is_fake': is_fake,
                        'confidence': confidence.item()
                    })

                    pbar.set_postfix({'fake': fake_count})

                frame_count += 1
                pbar.update(1)

        cap.release()

        # Aggregate results
        total_analyzed = len(frame_results)
        fake_ratio = fake_count / total_analyzed if total_analyzed > 0 else 0
        is_fake = fake_ratio > 0.5
        confidence = max(fake_ratio, 1 - fake_ratio)
        class_name = 'FAKE' if is_fake else 'REAL'

        return class_name, confidence, is_fake, frame_results

    def predict_batch(self, folder_path):
        """
        Predict all images in a folder.

        Returns:
            dict: Summary of predictions
        """
        folder_path = Path(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        results = []
        fake_count = 0
        real_count = 0

        image_files = list(folder_path.rglob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in image_extensions]

        for img_path in tqdm(image_files, desc='Processing'):
            try:
                class_name, confidence, is_fake = self.predict_image(str(img_path))
                results.append({
                    'path': str(img_path),
                    'class': class_name,
                    'confidence': confidence,
                    'is_fake': is_fake
                })

                if is_fake:
                    fake_count += 1
                else:
                    real_count += 1

            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")

        return {
            'total': len(results),
            'fake': fake_count,
            'real': real_count,
            'results': results
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='TruthLens - Deepfake Detector')
    parser.add_argument('--input', '-i', required=True,
                        help='Input image, video, or folder')
    parser.add_argument('--model', '-m', default=MODEL_PATH,
                        help='Path to trained model')
    parser.add_argument('--interval', '-n', type=int, default=30,
                        help='Frame interval for video analysis')

    args = parser.parse_args()

    # Load detector
    print("Loading model...")
    detector = DeepfakeDetector(args.model)

    input_path = Path(args.input)

    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            # Video prediction
            class_name, confidence, is_fake, frame_results = detector.predict_video(
                str(input_path), args.interval
            )
            print(f"\n{'='*60}")
            print(f"Result: {class_name}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Frames analyzed: {len(frame_results)}")
            print(f"Fake frames: {sum(1 for r in frame_results if r['is_fake'])}")
        else:
            # Single image prediction
            class_name, confidence, is_fake = detector.predict_image(str(input_path))
            print(f"\n{'='*60}")
            print(f"Result: {class_name}")
            print(f"Confidence: {confidence:.2%}")
            print(f"AI-generated probability: {is_fake}")

    elif input_path.is_dir():
        # Batch prediction
        print(f"\nAnalyzing folder: {input_path}")
        results = detector.predict_batch(str(input_path))

        print(f"\n{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Fake: {results['fake']} ({results['fake']/results['total']:.2%})")
        print(f"Real: {results['real']} ({results['real']/results['total']:.2%})")


if __name__ == '__main__':
    main()
