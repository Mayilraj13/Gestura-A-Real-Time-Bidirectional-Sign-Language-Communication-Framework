import { useRef, useState, useCallback } from "react";
import { AUDIO_SAMPLE_RATE } from "../config";

const NOISE_GATE_THRESHOLD = 0.01;

function calculateRMS(audioData) {
  let sum = 0;
  for (let i = 0; i < audioData.length; i++) {
    sum += audioData[i] * audioData[i];
  }
  return Math.sqrt(sum / audioData.length);
}

export function useMicrophone(onChunk) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState(null);
  const contextRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { 
          sampleRate: AUDIO_SAMPLE_RATE, 
          channelCount: 1, 
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        },
      });
      streamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
      contextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const pcm = e.inputBuffer.getChannelData(0);
        
        const rms = calculateRMS(pcm);
        
        if (rms > NOISE_GATE_THRESHOLD) {
          const buf = new Float32Array(pcm).buffer;
          onChunk(buf);
        } else {
          console.debug(`[Mic] Filtered silence (RMS: ${rms.toFixed(5)})`);
        }
      };

      source.connect(processor);
      processor.connect(ctx.destination);
      setRecording(true);
      console.log("[Mic] Recording started with noise gate");
    } catch (err) {
      setError(err.message);
      console.error("[Mic]", err);
    }
  }, [onChunk]);

  const stop = useCallback(() => {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    contextRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setRecording(false);
    console.log("[Mic] Recording stopped");
  }, []);

  return { recording, start, stop, error };
}
