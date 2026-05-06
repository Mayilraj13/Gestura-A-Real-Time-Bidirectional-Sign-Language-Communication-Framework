"""
Whisper-based speech recognizer.
Loads the model once and exposes a synchronous transcribe() function
that runs in a thread pool (called via asyncio.run_in_executor).
"""
import numpy as np
import torch
from loguru import logger
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE

try:
    import whisper
    if not hasattr(whisper, "load_model"):
        raise ImportError("Wrong 'whisper' package installed")
except (ImportError, AttributeError):
    raise ImportError(
        "\n❌ Wrong 'whisper' package installed.\n"
        "   Fix it:\n"
        "     pip uninstall whisper -y\n"
        "     pip install openai-whisper\n"
    )

_whisper_model = None


# 🔥 LOAD MODEL (FAST CONFIG)
def load_whisper() -> None:
    global _whisper_model

    device = WHISPER_DEVICE if torch.cuda.is_available() else "cpu"

    if device == "cpu" and WHISPER_DEVICE == "cuda":
        logger.warning("CUDA not available — running on CPU")

    logger.info(f"⚡ Loading Whisper '{WHISPER_MODEL_SIZE}' on {device}")

    _whisper_model = whisper.load_model(
        WHISPER_MODEL_SIZE,
        device=device
    )

    logger.info("✅ Whisper ready")


# 🚀 FAST TRANSCRIBE
def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    if _whisper_model is None:
        raise RuntimeError("Whisper model not loaded. Call load_whisper() first.")

    # 🔹 Convert audio
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

    if audio_array.size == 0:
        return ""

    # 🔥 Normalize (important for accuracy)
    audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-6)

    # 🔥 Trim silence (reduce processing time)
    audio_array = whisper.pad_or_trim(audio_array)

    # 🔥 Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_array).to(_whisper_model.device)

    # 🔥 FAST decoding options
    options = whisper.DecodingOptions(
        language="en",
        fp16=torch.cuda.is_available(),
        without_timestamps=True,
        beam_size=1,                  # 🔥 faster
        temperature=0.0,       # 🔥 stable
    )

    # 🔥 Decode
    result = whisper.decode(_whisper_model, mel, options)

    text = result.text.strip().lower()

    logger.debug(f"⚡ Transcript: '{text}'")

    return text


def is_loaded() -> bool:
    return _whisper_model is not None