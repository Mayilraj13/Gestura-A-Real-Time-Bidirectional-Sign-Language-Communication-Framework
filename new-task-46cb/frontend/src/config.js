const WS_BASE = process.env.REACT_APP_WS_URL || "ws://localhost:8000";
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const WS_SIGN_TO_SPEECH = `${WS_BASE}/ws/sign-to-speech`;
export const WS_SPEECH_TO_SIGN = `${WS_BASE}/ws/speech-to-sign`;
export const API_LABELS = `${API_BASE}/api/labels`;
export const API_SIGNS = `${API_BASE}/api/signs`;
export const API_HEALTH = `${API_BASE}/api/health`;

export const SEQUENCE_LENGTH = 20;
export const SEND_INTERVAL_MS = 10;
export const AUDIO_SAMPLE_RATE = 16000;

