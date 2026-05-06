"""
LSA64 Dataset Preprocessor
===========================

Dataset: LSA64 — 64 Argentine Sign Language classes
         3,200 videos | 10 subjects × 5 repetitions × 64 signs
         Some signs use BOTH hands → 126-feature extraction

Supported Kaggle structures (auto-detected):

  Structure A — Flat directory, numeric filename:
      lsa64/
          001_001_001.avi   (class 001, subject 001, rep 001)
          001_001_002.avi
          002_001_001.avi
          ...

  Structure B — Subfolder per class (numeric or named):
      lsa64/
          001/
              001_001_001.avi
          002/
              ...

  Structure C — Subfolder with class name:
      lsa64/
          opaque/
              video1.avi
          red/
              video1.avi

Output: final_dataset/
            opaque/
                keypoints.npy   shape (N_videos, 20, 126)  — dual-hand
                video.mp4
                label.txt       "opaque"

Usage:
    python ml/preprocess_lsa64.py --lsa64_dir C:\\path\\to\\lsa64 --output_dir final_dataset

Note: Set NUM_HANDS=2 in backend/.env (default) before training.
"""

import argparse
import shutil
import sys
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "logs" / "matplotlib"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import mediapipe as mp

SEQUENCE_LENGTH = 20
FEATURE_DIM = 126  # 21 landmarks × 3 coords × 2 hands

mp_hands = mp.solutions.hands

LSA64_CLASS_NAMES = {
    1: "opaque", 2: "red", 3: "green", 4: "yellow", 5: "bright",
    6: "light-blue", 7: "colors", 8: "where", 9: "birthday", 10: "pizza",
    11: "uruguay", 12: "coat", 13: "hunger", 14: "yarn", 15: "thirst",
    16: "help", 17: "forget", 18: "memory", 19: "none", 20: "name",
    21: "tired", 22: "sorry", 23: "permit", 24: "bread", 25: "sheep",
    26: "fruit", 27: "apple", 28: "argentina", 29: "mate", 30: "milk",
    31: "water", 32: "food", 33: "fat", 34: "thin", 35: "worried",
    36: "happy", 37: "sad", 38: "want", 39: "give", 40: "meat",
    41: "chicken", 42: "fish", 43: "sugar", 44: "egg", 45: "cheese",
    46: "butter", 47: "yogurt", 48: "potato", 49: "tomato", 50: "pepper",
    51: "oil", 52: "good-afternoon", 53: "good-morning", 54: "good-evening",
    55: "how-are-you", 56: "fine", 57: "thanks", 58: "no", 59: "i",
    60: "you", 61: "he-she", 62: "we", 63: "you-plural", 64: "they",
}


def normalize_hand(hand_landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    coords -= coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords /= scale
    return coords.flatten()


def extract_dual_keypoints(frame_rgb, hands_model) -> np.ndarray:
    """Extract right(63) + left(63) = 126-d. Zeros if hand absent."""
    right = np.zeros(63, dtype=np.float32)
    left = np.zeros(63, dtype=np.float32)

    if frame_rgb is None:
        return np.concatenate([right, left])

    results = hands_model.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return np.concatenate([right, left])

    for i, h in enumerate(results.multi_handedness):
        label = h.classification[0].label
        vec = normalize_hand(results.multi_hand_landmarks[i])
        if label == "Right":
            right = vec
        else:
            left = vec

    return np.concatenate([right, left])


def sample_frames(video_path: Path, n: int = SEQUENCE_LENGTH):
    """Sample n evenly-spaced RGB frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total == 0:
        cap.release()
        return frames

    indices = np.linspace(0, total - 1, n, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None)
    cap.release()
    return frames


def process_video(video_path: Path, hands_model) -> np.ndarray | None:
    """Process one video → (20, 126). Returns None if too many blank frames."""
    frames = sample_frames(video_path, SEQUENCE_LENGTH)
    if len(frames) < SEQUENCE_LENGTH:
        return None

    sequence = np.stack([extract_dual_keypoints(f, hands_model) for f in frames])
    zero_rows = np.all(sequence == 0, axis=1).sum()
    if zero_rows > SEQUENCE_LENGTH * 0.6:
        return None
    return sequence


def detect_structure(lsa64_dir: Path):
    """
    Auto-detect dataset structure. Returns list of (class_name, [video_paths]).
    """
    all_videos = list(lsa64_dir.rglob("*.avi")) + list(lsa64_dir.rglob("*.mp4"))
    if not all_videos:
        print("ERROR: No .avi or .mp4 files found in", lsa64_dir)
        sys.exit(1)

    class_map: dict[str, list[Path]] = {}

    subdirs = [d for d in lsa64_dir.iterdir() if d.is_dir()]

    if subdirs:
        for d in subdirs:
            vids = list(d.glob("*.avi")) + list(d.glob("*.mp4"))
            if not vids:
                continue
            try:
                class_id = int(d.name)
                class_name = LSA64_CLASS_NAMES.get(class_id, f"class_{class_id:03d}")
            except ValueError:
                class_name = d.name.lower().replace(" ", "-")
            class_map[class_name] = vids
    else:
        for v in all_videos:
            stem = v.stem
            parts = stem.split("_")
            try:
                class_id = int(parts[0])
                class_name = LSA64_CLASS_NAMES.get(class_id, f"class_{class_id:03d}")
            except (ValueError, IndexError):
                class_name = "unknown"
            class_map.setdefault(class_name, []).append(v)

    return sorted(class_map.items())


def preprocess(lsa64_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    class_entries = detect_structure(lsa64_dir)
    print(f"\nDetected {len(class_entries)} classes. Processing with dual-hand extraction…\n")

    all_labels = []
    skipped = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        model_complexity=1,
    ) as hands_model:

        for class_name, videos in tqdm(class_entries, desc="Classes"):
            out_dir = output_dir / class_name
            existing_keypoints = out_dir / "keypoints.npy"
            if existing_keypoints.exists():
                all_labels.append(class_name)
                continue

            sequences = []
            for v in sorted(videos):
                seq = process_video(v, hands_model)
                if seq is not None:
                    sequences.append(seq)

            if not sequences:
                print(f"  [SKIP] {class_name} — no usable videos")
                skipped += 1
                continue

            keypoints = np.stack(sequences, axis=0)

            out_dir.mkdir(exist_ok=True)

            np.save(str(out_dir / "keypoints.npy"), keypoints)

            best_video = max(videos, key=lambda v: int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            out_video = out_dir / "video.mp4"

            if best_video.suffix == ".mp4":
                shutil.copy2(str(best_video), str(out_video))
            else:
                _convert_avi_to_mp4(best_video, out_video)

            (out_dir / "label.txt").write_text(class_name, encoding="utf-8")
            all_labels.append(class_name)

    labels_path = output_dir.parent / "backend" / "model" / "labels.txt"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(all_labels), encoding="utf-8")

    env_path = output_dir.parent / "backend" / ".env"
    env_path.write_text("NUM_HANDS=2\n", encoding="utf-8")

    print(f"\n✅ Done!")
    print(f"   Classes processed : {len(all_labels)}")
    print(f"   Classes skipped   : {skipped}")
    print(f"   Feature dim       : 126 (dual-hand)")
    print(f"   Output dir        : {output_dir}")
    print(f"   Labels file       : {labels_path}")
    print(f"   .env updated      : NUM_HANDS=2")


def _convert_avi_to_mp4(src: Path, dst: Path):
    """Convert .avi to .mp4 using OpenCV (no ffmpeg needed)."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LSA64 dataset")
    parser.add_argument("--lsa64_dir", required=True,
                        help="Path to LSA64 dataset folder (contains videos or class subfolders)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <project_root>/final_dataset)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    lsa64 = Path(args.lsa64_dir)
    output = Path(args.output_dir) if args.output_dir else base / "final_dataset"

    if not lsa64.exists():
        print(f"ERROR: Path not found: {lsa64}")
        sys.exit(1)

    preprocess(lsa64, output)
