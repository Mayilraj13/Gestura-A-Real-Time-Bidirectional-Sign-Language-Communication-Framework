"""
WebSocket endpoint: Sign Language -> Speech.

Flow:
  Client sends keypoints JSON per frame
  Server buffers 20-frame sequences
  GRU+Attention model predicts sign labels
  Prediction smoother stabilizes output for speech/history
  Server sends both live and stable predictions back to the client
"""

import asyncio
import json
from collections import deque

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from config import (
    CONFIDENCE_THRESHOLD,
    FEATURE_DIM,
    LIVE_CONFIDENCE_THRESHOLD,
    MIN_PREDICTION_MARGIN,
    SEQUENCE_LENGTH,
)
from modules import gru_model
from modules.prediction_smoother import PredictionSmoother

router = APIRouter()


@router.websocket("/ws/sign-to-speech")
async def sign_to_speech_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected: sign-to-speech")

    frame_buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    smoother = PredictionSmoother()

    try:
        while True:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            data = json.loads(raw)

            if data.get("type") == "ping":
                continue

            if data.get("type") == "no_hand":
                frame_buffer.clear()
                smoother.clear_tracking()
                await websocket.send_json(
                    {
                        "status": "waiting",
                        "message": "No hand detected",
                    }
                )
                continue

            keypoints = data.get("keypoints")
            if keypoints is None:
                continue

            kp_array = np.array(keypoints, dtype=np.float32).flatten()

            # Auto-heal shape mismatches (63 <-> 126) so the session keeps running.
            if kp_array.shape[0] != FEATURE_DIM:
                if kp_array.shape[0] == 63 and FEATURE_DIM == 126:
                    kp_array = np.concatenate([kp_array, np.zeros_like(kp_array)])
                    logger.warning(
                        "Received 63-d points while expecting 126; padded left hand with zeros."
                    )
                elif kp_array.shape[0] == 126 and FEATURE_DIM == 63:
                    kp_array = kp_array[:FEATURE_DIM]
                    logger.warning(
                        "Received 126-d points while expecting 63; truncated to right hand."
                    )
                else:
                    logger.warning(f"Bad keypoints shape: {kp_array.shape}, expected {FEATURE_DIM}")
                    continue

            frame_buffer.append(kp_array)

            if len(frame_buffer) < SEQUENCE_LENGTH:
                await websocket.send_json(
                    {
                        "status": "buffering",
                        "buffer_size": len(frame_buffer),
                        "required": SEQUENCE_LENGTH,
                    }
                )
                continue

            sequence = np.stack(list(frame_buffer), axis=0)

            loop = asyncio.get_event_loop()
            raw_label, raw_conf, prob_map = await loop.run_in_executor(
                None, gru_model.predict, sequence
            )

            logger.debug(f"Raw prediction: {raw_label} ({raw_conf:.2f})")

            top_probs = list(prob_map.values())
            runner_up_conf = float(top_probs[1]) if len(top_probs) > 1 else 0.0
            prediction_margin = max(0.0, raw_conf - runner_up_conf)

            smoothed_label, smoothed_conf = smoother.update(
                raw_label,
                raw_conf,
                prob_map,
                prediction_margin,
            )
            is_stable = bool(smoothed_label)
            show_live_prediction = (
                raw_conf >= LIVE_CONFIDENCE_THRESHOLD
                and prediction_margin >= MIN_PREDICTION_MARGIN
            )

            response = {
                "status": "ok",
                "raw_label": raw_label,
                "raw_confidence": round(raw_conf, 4),
                "runner_up_confidence": round(runner_up_conf, 4),
                "prediction_margin": round(prediction_margin, 4),
                "smoothed_label": smoothed_label,
                "smoothed_confidence": round(smoothed_conf, 4),
                "display_label": smoothed_label or (raw_label if show_live_prediction else None),
                "display_confidence": round(
                    smoothed_conf if is_stable else (raw_conf if show_live_prediction else 0.0),
                    4,
                ),
                "is_stable": is_stable,
                "live_visible": show_live_prediction,
                "meets_stable_threshold": raw_conf >= CONFIDENCE_THRESHOLD,
                "top5": prob_map,
                "should_speak": is_stable,
            }

            if is_stable:
                logger.info(f"Sign recognized: '{smoothed_label}' ({smoothed_conf:.2f})")

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: sign-to-speech")
    except asyncio.TimeoutError:
        logger.warning("WebSocket timeout: sign-to-speech")
        await websocket.close()
    except Exception as exc:
        logger.error(f"sign-to-speech error: {exc}")
        await websocket.close()
    finally:
        smoother.reset()
