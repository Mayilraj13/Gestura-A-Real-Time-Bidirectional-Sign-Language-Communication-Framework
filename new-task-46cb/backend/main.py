"""
Gestura FastAPI application entry point.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

import config
from modules import gru_model, speech_recognizer, tts_engine
from modules.text_processor import load_spacy, load_labels
from routers import sign_to_speech, speech_to_sign, dataset as dataset_router

LOG_DIR = config.LOG_DIR
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(LOG_DIR / "app.log", rotation="10 MB", retention="7 days", level="DEBUG")

MODEL_CANDIDATES = [
    config.MODEL_PATH,                               # default: final_model.h5
    config.MODEL_DIR / "best_model.h5",              # training checkpoint
    config.MODEL_DIR / "gru_attention_model.h5",     # legacy name
]


def _pick_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Gestura - Starting up")
    logger.info("=" * 50)

    config.MODEL_DIR.mkdir(exist_ok=True)

    model_path = _pick_first_existing(MODEL_CANDIDATES)
    labels_path = config.LABELS_PATH if config.LABELS_PATH.exists() else None

    if model_path and labels_path:
        gru_model.load_model(model_path, labels_path)
        logger.info(f"Model loaded from: {model_path.name}")
    else:
        logger.warning(
            "Model or labels not found. Train the model first:\n"
            "  cd ml && python train_model.py\n"
            "This creates backend/model/final_model.h5 and labels.txt"
        )

    speech_recognizer.load_whisper()

    load_spacy()
    load_labels(config.LABELS_PATH)

    _ = tts_engine.get_tts()

    logger.info("All modules loaded — server ready")
    yield

    logger.info("Shutting down…")
    tts_engine.get_tts().shutdown()


app = FastAPI(
    title="Gestura API",
    description="Real-time bidirectional communication for deaf and hearing users",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_SIGNS = config.DATASET_DIR
if STATIC_SIGNS.exists():
    app.mount("/static/signs", StaticFiles(directory=str(STATIC_SIGNS)), name="signs")
else:
    logger.warning(f"Dataset dir not found: {STATIC_SIGNS} — /static/signs not mounted")

app.include_router(sign_to_speech.router)
app.include_router(speech_to_sign.router)
app.include_router(dataset_router.router)


@app.get("/")
def root():
    return {
        "app": "Gestura",
        "version": "1.0.0",
        "docs": "/docs",
        "websockets": ["/ws/sign-to-speech", "/ws/speech-to-sign"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
