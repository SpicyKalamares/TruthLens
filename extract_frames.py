"""
TruthLens - Video Frame Extractor
Extracts frames from videos for training the deepfake detection model.
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30 = ~1fps for 30fps video)
    """
    video_path = Path(video_path)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"  Video: {video_path.name}")
    print(f"  FPS: {fps}, Total frames: {total_frames}")

    frame_count = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="  Extracting") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = f"{video_path.stem}_frame_{saved_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"  Saved: {saved_count} frames")
    return saved_count

def process_video_folder(video_folder, output_folder, frame_interval=30):
    """
    Process all videos in a folder.

    Args:
        video_folder: Folder containing videos
        output_folder: Output folder structure (mirrors video folder)
        frame_interval: Extract every Nth frame
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_folder = Path(video_folder)

    total_extracted = 0

    for video_file in video_folder.rglob('*'):
        if video_file.suffix.lower() in video_extensions:
            # Determine output directory based on video location
            rel_path = video_file.relative_to(video_folder)
            output_dir = Path(output_folder) / rel_path.parent / rel_path.stem

            print(f"\nProcessing: {video_file}")
            extracted = extract_frames_from_video(
                str(video_file),
                str(output_dir),
                frame_interval=frame_interval
            )
            total_extracted += extracted

    print(f"\n{'='*60}")
    print(f"Total frames extracted: {total_extracted}")
    return total_extracted

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract frames from videos for training')
    parser.add_argument('--input', '-i', default='Dataset/videos',
                        help='Input folder containing videos (default: Dataset/videos)')
    parser.add_argument('--output', '-o', default='Dataset/extracted_frames',
                        help='Output folder for frames (default: Dataset/extracted_frames)')
    parser.add_argument('--interval', '-n', type=int, default=30,
                        help='Extract every Nth frame (default: 30)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input folder not found: {args.input}")
        print("Please create this folder and add your videos, organized as:")
        print("  Dataset/videos/REAL/*.mp4")
        print("  Dataset/videos/FAKE/*.mp4")
        exit(1)

    process_video_folder(args.input, args.output, args.interval)
    print("\nFrames extracted! You can now use them for training by copying to Dataset/train/")
