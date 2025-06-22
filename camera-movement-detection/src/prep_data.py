import cv2
import os
from typing import List, Tuple, Optional

def load_image_sequence(
    directory: str,
    resize_dim: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None
) -> Tuple[List, List[str]]:
    """
    Load sorted images from a directory, convert to grayscale, and return
    both the frame list and the corresponding filenames.
    """
    files = sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    )

    frames, names = [], []
    for idx, fname in enumerate(files):
        if max_frames and idx >= max_frames:
            break
        path = os.path.join(directory, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        if resize_dim:
            img = cv2.resize(img, resize_dim)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        names.append(fname)

    return frames, names

def load_video(
    video_path: str,
    resize_dim: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None
) -> Tuple[List, List[int]]:
    """
    Load grayscale frames from video; filenames are just frame indices.
    """
    cap = cv2.VideoCapture(video_path)
    frames, names = [], []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and count >= max_frames):
            break
        if resize_dim:
            frame = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        names.append(count)
        count += 1

    cap.release()
    return frames, names

def capture_from_webcam(
    num_frames: int = 20,
    resize_dim: Optional[Tuple[int, int]] = None
) -> Tuple[List, List[int]]:
    """
    Capture frames from the default webcam.
    """
    cap = cv2.VideoCapture(0)
    frames, names = [], []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if resize_dim:
            frame = cv2.resize(frame, resize_dim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        names.append(i)

    cap.release()
    return frames, names

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Load frames and return grayscale images + their identifiers'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_dir', type=str, help='Directory of images')
    group.add_argument('--video',     type=str, help='Path to video file')
    group.add_argument('--webcam',    action='store_true', help='Use webcam')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('W','H'))
    parser.add_argument('--max_frames', type=int)
    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize else None
    if args.input_dir:
        frames, names = load_image_sequence(args.input_dir, resize, args.max_frames)
    elif args.video:
        frames, names = load_video(args.video, resize, args.max_frames)
    else:
        frames, names = capture_from_webcam(args.max_frames or 20, resize)

    print(f'Loaded {len(frames)} frames: {names}')
