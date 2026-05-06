import { useCallback, useEffect, useState, useRef } from "react";
import { useMicrophone } from "../hooks/useMicrophone";
import { useWebSocket } from "../hooks/useWebSocket";
import { WS_SPEECH_TO_SIGN } from "../config";
import StatusIndicator from "./StatusIndicator";

export default function MicrophoneRecorder({ onResult }) {
  const [status, setStatus] = useState("idle");
  const [transcript, setTranscript] = useState("");
  const [missing, setMissing] = useState([]);
  const { connected, send, sendBinary, subscribe } = useWebSocket(WS_SPEECH_TO_SIGN);
  const activeRef = useRef(false);

  const handleChunk = useCallback(
    (buf) => {
      if (activeRef.current) sendBinary(buf);
    },
    [sendBinary]
  );

  const { recording, start: startMic, stop: stopMic, error: micError } = useMicrophone(handleChunk);

  useEffect(() => {
    return subscribe((msg) => {
      if (msg.status === "recording") setStatus("recording");
      else if (msg.status === "processing") setStatus("processing");
      else if (msg.status === "result") {
        setStatus("idle");
        setTranscript(msg.transcript || "");
        setMissing(msg.missing_signs || []);
        onResult?.(msg.sign_sequence || [], msg.transcript || "");
      } else if (msg.status === "no_speech") {
        setStatus("idle");
        setTranscript("");
      } else if (msg.status === "too_short") {
        setStatus("idle");
      } else if (msg.status === "error") {
        setStatus("error");
      }
    });
  }, [subscribe, onResult]);

  const startRecording = useCallback(async () => {
    if (!connected) return;
    activeRef.current = true;
    send({ action: "start" });
    await startMic();
    setStatus("recording");
    setTranscript("");
    setMissing([]);
  }, [connected, send, startMic]);

  const stopRecording = useCallback(() => {
    stopMic();
    activeRef.current = false;
    send({ action: "stop" });
    setStatus("processing");
  }, [stopMic, send]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-bold uppercase text-slate-500">Audio workspace</p>
          <h2 className="mt-1 text-2xl font-bold text-slate-950">Speech to Sign</h2>
        </div>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-500" : "bg-rose-500"}`} />
          <span className="text-sm font-semibold text-slate-500">{connected ? "Connected" : "Disconnected"}</span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <StatusIndicator status={status} />
        {!recording ? (
          <button
            type="button"
            onClick={startRecording}
            disabled={!connected}
            className="flex items-center gap-2 rounded-lg bg-rose-600 px-5 py-2.5 text-base font-semibold text-white transition-colors hover:bg-rose-700 disabled:opacity-40"
          >
            <span className="h-2 w-2 rounded-full bg-white" />
            Start Speaking
          </button>
        ) : (
          <button
            type="button"
            onClick={stopRecording}
            className="rounded-lg bg-slate-800 px-5 py-2.5 text-base font-semibold text-white transition-colors hover:bg-slate-900"
          >
            Stop & Translate
          </button>
        )}
      </div>

      {micError && (
        <p className="text-xs font-medium text-rose-600">Microphone error: {micError}</p>
      )}

      {transcript && (
        <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-4">
          <p className="text-sm font-bold uppercase text-slate-500">Recognized Speech</p>
          <p className="text-xl font-semibold leading-relaxed text-slate-950">"{transcript}"</p>
          {missing.length > 0 && (
            <p className="text-xs font-medium text-amber-700">
              No sign video for: {missing.join(", ")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
