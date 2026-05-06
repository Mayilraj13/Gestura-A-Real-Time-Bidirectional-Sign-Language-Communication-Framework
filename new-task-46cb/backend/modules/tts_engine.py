"""
Text-to-Speech engine using pyttsx3.
Runs in a dedicated thread to avoid blocking the async event loop.
"""

from __future__ import annotations

import threading
import queue
import pyttsx3
from loguru import logger
from config import TTS_RATE, TTS_VOLUME


class TTSEngine:
    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("TTS engine started (dedicated thread)")

    def speak(self, text: str) -> None:
        if text and text.strip():
            self._q.put(text.strip())
            logger.debug(f"TTS queued: '{text}'")

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)

        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)

        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                logger.error(f"TTS error: {exc}")
            finally:
                self._q.task_done()

    def shutdown(self):
        self._q.put(None)
        self._thread.join(timeout=3)
        logger.info("TTS engine shut down")


_tts: TTSEngine | None = None


def get_tts() -> TTSEngine:
    global _tts
    if _tts is None:
        _tts = TTSEngine()
    return _tts
