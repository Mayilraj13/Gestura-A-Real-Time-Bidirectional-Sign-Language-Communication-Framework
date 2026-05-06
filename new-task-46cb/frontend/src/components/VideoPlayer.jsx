import { useRef, useEffect, useState, useCallback } from "react";

export default function VideoPlayer({ signSequence = [] }) {
  const videoRef = useRef(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loadError, setLoadError] = useState(null);

  const playNext = useCallback(() => {
    if (currentIdx < signSequence.length - 1) {
      setCurrentIdx((i) => i + 1);
      setLoadError(null);
    } else {
      setPlaying(false);
    }
  }, [currentIdx, signSequence.length]);

  useEffect(() => {
    if (signSequence.length > 0) {
      setCurrentIdx(0);
      setPlaying(true);
      setLoadError(null);
    } else {
      setPlaying(false);
    }
  }, [signSequence]);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid || !playing || !signSequence[currentIdx]) return;

    const url = `http://localhost:8000${signSequence[currentIdx].video_url}`;
    
    const handleCanPlay = () => {
      console.log(`[VideoPlayer] Playing: ${signSequence[currentIdx].label}`);
      setLoadError(null);
      vid.play().catch((err) => {
        console.error("[VideoPlayer] Play error:", err);
        setLoadError(`Cannot play video: ${err.message}`);
      });
    };

    const handleError = (e) => {
      const error = vid.error;
      let errMsg = "Unknown error";
      if (error) {
        if (error.code === 4) errMsg = "Video format not supported";
        else if (error.code === 2) errMsg = "Network error";
        else if (error.code === 3) errMsg = "Video loading aborted";
      }
      console.error(`[VideoPlayer] Load error for ${url}: ${errMsg}`);
      setLoadError(`Failed to load sign video: ${errMsg}`);
      setTimeout(playNext, 1000);
    };

    const handleEnded = () => {
      console.log(`[VideoPlayer] Finished: ${signSequence[currentIdx].label}`);
      playNext();
    };

    vid.addEventListener("canplay", handleCanPlay);
    vid.addEventListener("error", handleError);
    vid.addEventListener("ended", handleEnded);

    console.log(`[VideoPlayer] Loading: ${signSequence[currentIdx].label} from ${url}`);
    vid.src = url;
    vid.load();

    return () => {
      vid.removeEventListener("canplay", handleCanPlay);
      vid.removeEventListener("error", handleError);
      vid.removeEventListener("ended", handleEnded);
    };
  }, [currentIdx, playing, signSequence, playNext]);

  if (!signSequence.length) {
    return (
      <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-sm font-medium text-slate-500">
        Sign videos will appear here
      </div>
    );
  }

  const current = signSequence[currentIdx];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {signSequence.map((s, i) => (
          <button
            key={i}
            onClick={() => { setCurrentIdx(i); setPlaying(true); }}
            className={`rounded-lg px-3 py-1 text-sm font-semibold transition-colors ${
              i === currentIdx
                ? "bg-teal-600 text-white"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="relative aspect-video overflow-hidden rounded-lg bg-slate-950">
        <video
          ref={videoRef}
          className="w-full h-full object-contain"
          controls={false}
          playsInline
          muted={false}
        />
        {loadError && (
          <div className="absolute inset-0 flex items-center justify-center bg-rose-950/75">
            <p className="px-4 text-center text-sm text-rose-100">{loadError}</p>
          </div>
        )}
        <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
          <span className="rounded bg-black/60 px-2 py-1 text-xs text-white">
            {current?.label}
          </span>
          <span className="rounded bg-black/60 px-2 py-1 text-xs text-white">
            {currentIdx + 1} / {signSequence.length}
          </span>
        </div>
      </div>
      {loadError && (
        <p className="mt-2 text-xs font-medium text-rose-600">Error: {loadError}</p>
      )}
    </div>
  );
}
