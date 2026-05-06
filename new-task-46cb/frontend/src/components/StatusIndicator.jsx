import clsx from "clsx";

const STATUS_CONFIG = {
  idle: { color: "bg-slate-100 text-slate-700 ring-slate-200", dot: "bg-slate-400", label: "Idle" },
  buffering: { color: "bg-amber-50 text-amber-800 ring-amber-200", dot: "bg-amber-400", label: "Buffering..." },
  detecting: { color: "bg-sky-50 text-sky-800 ring-sky-200", dot: "bg-sky-500", label: "Detecting" },
  recognized: { color: "bg-emerald-50 text-emerald-800 ring-emerald-200", dot: "bg-emerald-500", label: "Sign Recognized" },
  speaking: { color: "bg-violet-50 text-violet-800 ring-violet-200", dot: "bg-violet-500", label: "Speaking" },
  recording: { color: "bg-rose-50 text-rose-800 ring-rose-200", dot: "bg-rose-500", label: "Recording" },
  processing: { color: "bg-orange-50 text-orange-800 ring-orange-200", dot: "bg-orange-500", label: "Processing" },
  error: { color: "bg-red-50 text-red-800 ring-red-200", dot: "bg-red-500", label: "Error" },
};

export default function StatusIndicator({ status = "idle", message = "" }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.idle;
  const isActive = ["detecting", "recording", "processing"].includes(status);

  return (
    <div className={clsx("flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-semibold ring-1", cfg.color)}>
      <span className={clsx("h-2 w-2 rounded-full", cfg.dot, isActive && "animate-pulse")} />
      <span>{message || cfg.label}</span>
    </div>
  );
}
