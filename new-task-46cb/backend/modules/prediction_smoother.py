"""
Prediction stabilization using:
1. Majority voting over a sliding window of predictions
2. Exponential moving average on class probabilities
3. Confidence and margin gating
4. Vote-ratio gating to reject unstable guesses
5. Debounce to suppress repeated identical output
"""

from __future__ import annotations

from collections import deque
import time

from config import (
    CONFIDENCE_THRESHOLD,
    DEBOUNCE_SECONDS,
    EMA_ALPHA,
    MIN_PREDICTION_MARGIN,
    MIN_VOTE_RATIO,
    SMOOTHING_WINDOW,
)


class PredictionSmoother:
    def __init__(
        self,
        smoothing_window: int = SMOOTHING_WINDOW,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        ema_alpha: float = EMA_ALPHA,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        min_prediction_margin: float = MIN_PREDICTION_MARGIN,
        min_vote_ratio: float = MIN_VOTE_RATIO,
    ):
        self._window = deque(maxlen=smoothing_window)
        self._ema_probs: dict[str, float] = {}
        self._alpha = ema_alpha
        self._confidence_threshold = confidence_threshold
        self._debounce = debounce_seconds
        self._min_prediction_margin = min_prediction_margin
        self._min_vote_ratio = min_vote_ratio
        self._last_spoken: str | None = None
        self._last_spoken_at: float = 0.0

    def update(
        self,
        label: str,
        confidence: float,
        prob_map: dict[str, float],
        prediction_margin: float,
    ) -> tuple[str | None, float]:
        """
        Feed a new raw prediction and return a stable label only when the
        sequence is confident, separated from the runner-up class, and
        consistent across the full smoothing window.
        """
        if (
            confidence < self._confidence_threshold
            or prediction_margin < self._min_prediction_margin
        ):
            self.clear_tracking()
            return None, 0.0

        self._window.append(label)
        self._update_ema(prob_map)

        smoothed_label, smoothed_confidence, vote_ratio = self._majority_vote()
        if smoothed_label is None:
            return None, 0.0

        if len(self._window) < self._window.maxlen:
            return None, 0.0

        if vote_ratio < self._min_vote_ratio:
            return None, 0.0

        if smoothed_confidence < self._confidence_threshold:
            return None, 0.0

        now = time.time()
        if (
            smoothed_label == self._last_spoken
            and (now - self._last_spoken_at) < self._debounce
        ):
            return None, 0.0

        self._last_spoken = smoothed_label
        self._last_spoken_at = now
        return smoothed_label, smoothed_confidence

    def _majority_vote(self) -> tuple[str | None, float, float]:
        if not self._window:
            return None, 0.0, 0.0

        labels = list(self._window)
        most_common = max(set(labels), key=labels.count)
        vote_ratio = labels.count(most_common) / len(labels)
        ema_conf = self._ema_probs.get(most_common, 0.0)
        combined = 0.6 * vote_ratio + 0.4 * ema_conf
        return most_common, combined, vote_ratio

    def _update_ema(self, prob_map: dict[str, float]) -> None:
        for label, prob in prob_map.items():
            prev = self._ema_probs.get(label, 0.0)
            self._ema_probs[label] = self._alpha * prob + (1 - self._alpha) * prev

    def clear_tracking(self) -> None:
        self._window.clear()
        self._ema_probs.clear()

    def reset(self) -> None:
        self.clear_tracking()
        self._last_spoken = None
        self._last_spoken_at = 0.0
