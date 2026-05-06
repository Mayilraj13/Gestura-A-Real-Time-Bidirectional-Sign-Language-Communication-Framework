import { useEffect, useRef, useCallback, useState } from "react";

export function useWebSocket(url) {
  const ws = useRef(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const reconnectTimer = useRef(null);
  const listeners = useRef([]);

  const pingTimer = useRef(null);

  const connect = useCallback(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) return;

    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      setConnected(true);
      console.log(`[WS] Connected: ${url}`);
      clearInterval(pingTimer.current);
      pingTimer.current = setInterval(() => {
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: "ping" }));
        }
      }, 20000);
    };

    ws.current.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setLastMessage(data);
        listeners.current.forEach((fn) => fn(data));
      } catch {}
    };

    ws.current.onerror = (e) => console.error("[WS] Error", e);

    ws.current.onclose = () => {
      setConnected(false);
      clearInterval(pingTimer.current);
      console.log("[WS] Disconnected — reconnecting in 2s");
      reconnectTimer.current = setTimeout(connect, 2000);
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      clearInterval(pingTimer.current);
      ws.current?.close();
    };
  }, [connect]);

  const send = useCallback((data) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(typeof data === "string" ? data : JSON.stringify(data));
    }
  }, []);

  const sendBinary = useCallback((buffer) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(buffer);
    }
  }, []);

  const subscribe = useCallback((fn) => {
    listeners.current.push(fn);
    return () => {
      listeners.current = listeners.current.filter((f) => f !== fn);
    };
  }, []);

  return { connected, send, sendBinary, subscribe, lastMessage };
}
