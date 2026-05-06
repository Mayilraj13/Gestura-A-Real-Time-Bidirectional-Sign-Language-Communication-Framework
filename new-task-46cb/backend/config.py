import os
from pathlib import Path
from dotenv import load_dotenv

# Always prefer values from .env over any stale process env
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "final_dataset"
MODEL_DIR = BASE_DIR / "backend" / "model"
LOG_DIR = BASE_DIR / "logs"

# Primary inference model file produced by ml/train_model.py
MODEL_PATH = MODEL_DIR / "final_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

# ── Frame / Feature settings ──────────────────────────────
SEQUENCE_LENGTH = 20          # sliding window (20 frames = lower latency)
NUM_LANDMARKS = 21
NUM_COORDS = 3
NUM_HANDS = int(os.getenv("NUM_HANDS", "1"))       # 1=right hand only, 2=both hands (LSA64)
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS * NUM_HANDS  # 63 (1 hand) or 126 (2 hands)

# ── Model / Inference ─────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
LIVE_CONFIDENCE_THRESHOLD = float(os.getenv("LIVE_CONFIDENCE_THRESHOLD", "0.60"))
MIN_PREDICTION_MARGIN = float(os.getenv("MIN_PREDICTION_MARGIN", "0.18"))
MIN_VOTE_RATIO = float(os.getenv("MIN_VOTE_RATIO", "0.70"))
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "5"))
EMA_ALPHA = float(os.getenv("EMA_ALPHA", "0.6"))
DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "1.2"))

# ── Whisper STT ───────────────────────────────────────────
WHISPER_MODEL_SIZE = "tiny"   # small|base|medium — balance speed/accuracy
WHISPER_DEVICE = "cuda"       # GPU inference

# ── Text-to-Speech ────────────────────────────────────────
TTS_RATE = 150
TTS_VOLUME = 1.0

# ── Server ────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
