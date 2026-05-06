import { useCallback, useEffect, useRef, useState } from "react";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { HAND_CONNECTIONS } from "@mediapipe/hands";
import { WS_SIGN_TO_SPEECH } from "../config";
import { useWebSocket } from "../hooks/useWebSocket";
import ConfidenceBar from "./ConfidenceBar";
import StatusIndicator from "./StatusIndicator";

const HAND_DIM = 21 * 3;
const ZERO_HAND = Array(HAND_DIM).fill(0);
const SEND_INTERVAL_MS = 150;
const NO_HAND_SEND_INTERVAL_MS = 350;
const STATUS_RESET_MS = 1400;
const SPEECH_DEBOUNCE_MS = 1200;

function normalizeLandmarks(landmarks) {
  if (!landmarks?.length) return null;

  const base = landmarks[0];
  if (!base || typeof base.x !== "number") return null;

  const coords = landmarks.map((lm) => [lm.x - base.x, lm.y - base.y, lm.z - base.z]);
  const maxDist = Math.max(...coords.map(([x, y, z]) => Math.hypot(x, y, z))) || 1;
  if (!isFinite(maxDist) || maxDist <= 0) return null;

  const normalized = coords.flatMap(([x, y, z]) => [x / maxDist, y / maxDist, z / maxDist]);
  return normalized.some((value) => !isFinite(value)) ? null : normalized;
}

function buildKeypoints(multiHandLandmarks, multiHandedness) {
  if (!multiHandLandmarks?.length || !multiHandedness?.length) return null;

  let right = null;
  let left = null;

  for (let i = 0; i < multiHandedness.length; i += 1) {
    const label = multiHandedness[i]?.classification?.[0]?.label;
    const normalized = normalizeLandmarks(multiHandLandmarks[i]);
    if (!normalized) continue;

    if (label === "Right") right = normalized;
    else if (label === "Left") left = normalized;
  }

  if (!right && multiHandLandmarks[0]) {
    right = normalizeLandmarks(multiHandLandmarks[0]);
  }
  if (!left && multiHandLandmarks[1]) {
    left = normalizeLandmarks(multiHandLandmarks[1]);
  }

  return [...(right || ZERO_HAND), ...(left || ZERO_HAND)];
}

function formatLabel(label) {
  if (!label) return "";
  return label.replace(/-/g, " ");
}

export default function WebcamCapture({ onPrediction }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handsRef = useRef(null);
  const rafRef = useRef(null);
  const streamRef = useRef(null);
  const lastSendRef = useRef(0);
  const lastNoHandSentRef = useRef(0);
  const activeRef = useRef(false);
  const statusTimeoutRef = useRef(null);
  const lastStableLabelRef = useRef("");
  const lastSpokenAtRef = useRef(0);
  const speakingRef = useRef(false);

  const [status, setStatus] = useState("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [active, setActive] = useState(false);

  const { connected, send, subscribe } = useWebSocket(WS_SIGN_TO_SPEECH);
  const sendRef = useRef(send);

  useEffect(() => {
    sendRef.current = send;
  }, [send]);

  const clearStatusReset = useCallback(() => {
    if (statusTimeoutRef.current) {
      clearTimeout(statusTimeoutRef.current);
      statusTimeoutRef.current = null;
    }
  }, []);

  const queueStatusReset = useCallback(() => {
    clearStatusReset();
    statusTimeoutRef.current = setTimeout(() => {
      if (activeRef.current) {
        setStatus("detecting");
        setStatusMessage("");
      }
    }, STATUS_RESET_MS);
  }, [clearStatusReset]);

  const speakPrediction = useCallback((label) => {
    if (!label || typeof window === "undefined" || !window.speechSynthesis) {
      return;
    }

    const now = Date.now();
    if (
      lastStableLabelRef.current === label &&
      now - lastSpokenAtRef.current < SPEECH_DEBOUNCE_MS
    ) {
      return;
    }

    const utterance = new SpeechSynthesisUtterance(formatLabel(label));
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;
    utterance.onend = () => {
      speakingRef.current = false;
      if (activeRef.current) {
        setStatus("recognized");
        setStatusMessage("Prediction locked");
        queueStatusReset();
      }
    };

    window.speechSynthesis.cancel();
    speakingRef.current = true;
    window.speechSynthesis.speak(utterance);
    lastStableLabelRef.current = label;
    lastSpokenAtRef.current = now;
    setStatus("speaking");
    setStatusMessage(`Speaking: ${formatLabel(label)}`);
  }, [queueStatusReset]);

  useEffect(() => {
    return subscribe((msg) => {
      if (msg.status === "buffering") {
        setStatus("buffering");
        setStatusMessage(`Collecting frames ${msg.buffer_size}/${msg.required}`);
        return;
      }

      if (msg.status === "waiting") {
        setPrediction("");
        setConfidence(0);
        setStatus("detecting");
        setStatusMessage(msg.message || "Show your hand to the camera");
        return;
      }

      if (msg.status !== "ok") {
        return;
      }

      const nextLabel = msg.display_label || "";
      const nextConfidence = msg.display_confidence ?? 0;

      if (nextLabel) {
        setPrediction(nextLabel);
        setConfidence(nextConfidence);
      } else {
        setPrediction("");
        setConfidence(0);
      }

      if (msg.is_stable && msg.smoothed_label) {
        setStatus("recognized");
        setStatusMessage("Prediction locked");
        onPrediction?.(msg.smoothed_label, msg.smoothed_confidence ?? nextConfidence);

        if (msg.should_speak) {
          speakPrediction(msg.smoothed_label);
        } else {
          queueStatusReset();
        }
        return;
      }

      if (activeRef.current) {
        setStatus("detecting");
        setStatusMessage("Live prediction updating");
      }
    });
  }, [onPrediction, queueStatusReset, speakPrediction, subscribe]);

  const drawFrame = useCallback((results) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (results.image) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
    }

    if (results.multiHandLandmarks?.length) {
      results.multiHandLandmarks.forEach((hand) => {
        drawConnectors(ctx, hand, HAND_CONNECTIONS, { color: "#00ff88", lineWidth: 3 });
        drawLandmarks(ctx, hand, { color: "#ff3366", lineWidth: 1, radius: 3 });
      });
    }

    if (results.multiHandLandmarks?.length && results.multiHandedness?.length) {
      const keypoints = buildKeypoints(results.multiHandLandmarks, results.multiHandedness);
      if (!keypoints) return;

      const now = Date.now();
      lastNoHandSentRef.current = 0;
      if (now - lastSendRef.current >= SEND_INTERVAL_MS) {
        lastSendRef.current = now;
        sendRef.current({ type: "frame", keypoints });
      }

      if (activeRef.current && !speakingRef.current) {
        setStatus("detecting");
        setStatusMessage("Hand detected");
      }
      return;
    }

    if (activeRef.current && connected) {
      const now = Date.now();
      if (now - lastNoHandSentRef.current >= NO_HAND_SEND_INTERVAL_MS) {
        lastNoHandSentRef.current = now;
        sendRef.current({ type: "no_hand" });
      }
      setStatus("detecting");
      setStatusMessage("Show your hand to the camera");
    }
  }, [connected]);

  const processLoop = useCallback(async () => {
    if (!activeRef.current) return;

    const hands = handsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (canvas && video && video.readyState >= 2) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
    }

    if (hands && video && video.readyState >= 2) {
      try {
        await hands.send({ image: video });
      } catch (error) {
        console.warn("[Hands] send failed:", error);
      }
    }

    rafRef.current = requestAnimationFrame(processLoop);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      clearStatusReset();
      setPrediction("");
      setConfidence(0);
      lastNoHandSentRef.current = 0;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 480, height: 360 },
        audio: false,
      });

      streamRef.current = stream;

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const { Hands } = await import("@mediapipe/hands");
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`,
      });

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        selfieMode: false,
      });

      hands.onResults(drawFrame);
      await hands.initialize();

      handsRef.current = hands;
      activeRef.current = true;
      setActive(true);
      setStatus("buffering");
      setStatusMessage("Starting camera");

      rafRef.current = requestAnimationFrame(processLoop);
    } catch (error) {
      console.error(error);
      setStatus("error");
      setStatusMessage("Unable to start camera");
    }
  }, [clearStatusReset, drawFrame, processLoop]);

  const stopCamera = useCallback(() => {
    clearStatusReset();
    cancelAnimationFrame(rafRef.current);

    if (handsRef.current) {
      try {
        handsRef.current.close();
      } catch (error) {
        console.warn("[Hands] close skipped:", error?.message);
      } finally {
        handsRef.current = null;
      }
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    speakingRef.current = false;
    lastStableLabelRef.current = "";
    lastSpokenAtRef.current = 0;
    lastNoHandSentRef.current = 0;
    activeRef.current = false;
    setActive(false);
    setStatus("idle");
    setStatusMessage("");
  }, [clearStatusReset]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[11px] font-bold uppercase text-slate-500">Camera workspace</p>
          <h2 className="mt-0.5 text-base font-bold text-slate-950">Sign to Speech</h2>
        </div>
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${connected ? "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200" : "bg-rose-50 text-rose-700 ring-1 ring-rose-200"}`}>
          {connected ? "WebSocket connected" : "WebSocket disconnected"}
        </span>
      </div>

      <video ref={videoRef} className="hidden" />

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
        <canvas
          ref={canvasRef}
          width={480}
          height={360}
          className="mx-auto aspect-[4/3] w-full max-w-md object-cover"
          style={{ transform: "scaleX(-1)" }}
        />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {!active ? (
          <button
            type="button"
            onClick={startCamera}
            className="rounded-lg bg-teal-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-teal-700"
          >
            Start Camera
          </button>
        ) : (
          <button
            type="button"
            onClick={stopCamera}
            className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-slate-900"
          >
            Stop Camera
          </button>
        )}

        <StatusIndicator status={status} message={statusMessage} />
      </div>

      <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-[11px] font-bold uppercase text-slate-500">Current Prediction</p>
            <h3 className="mt-0.5 text-lg font-bold text-slate-950">
              {prediction ? formatLabel(prediction) : "Waiting for a sign"}
            </h3>
          </div>
          {prediction ? (
            <span className="rounded-full bg-white px-3 py-1 text-xs font-bold text-slate-700 ring-1 ring-slate-200">
              {Math.round(confidence * 100)}%
            </span>
          ) : null}
        </div>

        <ConfidenceBar confidence={confidence} label={prediction ? formatLabel(prediction) : ""} />
      </div>
    </div>
  );
}
