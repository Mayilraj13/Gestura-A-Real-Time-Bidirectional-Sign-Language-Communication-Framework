"""
WebSocket endpoint: Speech → Sign Language

Flow:
  Client streams raw PCM float32 audio chunks via WebSocket
  Server accumulates chunks until silence / end-of-utterance
  Whisper transcribes accumulated audio
  spaCy extracts content-word keywords
  Keywords matched against dataset labels
  Server returns ordered list of video URLs to display
"""

from __future__ import annotations

import json
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from modules import speech_recognizer
from modules.text_processor import process_text
from config import DATASET_DIR

router = APIRouter()

SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_S = 0.5
MAX_AUDIO_DURATION_S = 10.0


def _build_video_url(label: str) -> str | None:
    """Return the static URL for a sign video, or None if not found."""
    video_path = DATASET_DIR / label / "video.mp4"
    if video_path.exists():
        return f"/static/signs/{label}/video.mp4"
    return None


@router.websocket("/ws/speech-to-sign")
async def speech_to_sign_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected: speech-to-sign")

    audio_chunks: list[bytes] = []
    total_samples = 0
    max_samples = int(MAX_AUDIO_DURATION_S * SAMPLE_RATE)

    try:
        while True:
            message = await asyncio.wait_for(websocket.receive(), timeout=30.0)

            if "text" in message:
                cmd = json.loads(message["text"])
                action = cmd.get("action")

                if action == "start":
                    audio_chunks.clear()
                    total_samples = 0
                    await websocket.send_json({"status": "recording"})
                    logger.debug("Recording started")

                elif action == "stop":
                    if not audio_chunks:
                        await websocket.send_json({"status": "error", "message": "No audio received"})
                        continue

                    await _process_and_respond(websocket, audio_chunks)
                    audio_chunks.clear()
                    total_samples = 0

                elif action == "ping":
                    await websocket.send_json({"status": "pong"})

            elif "bytes" in message:
                chunk = message["bytes"]
                audio_chunks.append(chunk)
                chunk_samples = len(chunk) // 4
                total_samples += chunk_samples

                if total_samples >= max_samples:
                    logger.info("Max audio duration reached — auto-processing")
                    await _process_and_respond(websocket, audio_chunks)
                    audio_chunks.clear()
                    total_samples = 0

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: speech-to-sign")
    except asyncio.TimeoutError:
        logger.warning("WebSocket timeout: speech-to-sign")
        await websocket.close()
    except Exception as exc:
        logger.error(f"speech-to-sign error: {exc}")
        await websocket.close()


async def _process_and_respond(websocket: WebSocket, audio_chunks: list[bytes]):
    combined = b"".join(audio_chunks)
    min_bytes = int(MIN_AUDIO_DURATION_S * SAMPLE_RATE * 4)
    if len(combined) < min_bytes:
        await websocket.send_json({"status": "too_short", "message": "Speak longer"})
        return

    await websocket.send_json({"status": "processing"})

    loop = asyncio.get_event_loop()

    try:
        transcript = await loop.run_in_executor(
            None, speech_recognizer.transcribe_audio, combined, SAMPLE_RATE
        )
    except Exception as exc:
        logger.error(f"Whisper error: {exc}")
        await websocket.send_json({"status": "error", "message": "Speech recognition failed"})
        return

    logger.info(f"Transcript: '{transcript}'")

    if not transcript:
        await websocket.send_json({"status": "no_speech", "transcript": ""})
        return

    matched_labels = process_text(transcript)
    logger.info(f"Matched labels: {matched_labels}")

    sign_sequence = []
    missing = []
    for label in matched_labels:
        url = _build_video_url(label)
        if url:
            sign_sequence.append({"label": label, "video_url": url})
        else:
            missing.append(label)

    await websocket.send_json({
        "status": "result",
        "transcript": transcript,
        "matched_labels": matched_labels,
        "sign_sequence": sign_sequence,
        "missing_signs": missing,
    })
