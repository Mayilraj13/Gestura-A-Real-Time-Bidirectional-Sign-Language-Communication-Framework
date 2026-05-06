import { useState, useCallback } from "react";
import { Activity, Camera, Hand, History, Mic, Radio, Video } from "lucide-react";
import WebcamCapture from "../components/WebcamCapture";
import MicrophoneRecorder from "../components/MicrophoneRecorder";
import VideoPlayer from "../components/VideoPlayer";

export default function Dashboard() {
  const [signHistory, setSignHistory] = useState([]);
  const [signSequence, setSignSequence] = useState([]);
  const [transcript, setTranscript] = useState("");
  const [activeTab, setActiveTab] = useState("sign");

  const handlePrediction = useCallback((label, confidence) => {
    setSignHistory((prev) => [
      { label, confidence, time: new Date().toLocaleTimeString() },
      ...prev.slice(0, 19),
    ]);
  }, []);

  const handleSpeechResult = useCallback((sequence, text) => {
    setSignSequence(sequence);
    setTranscript(text);
  }, []);

  const latestSign = signHistory[0];
  const avgConfidence = signHistory.length
    ? Math.round(
        (signHistory.reduce((sum, item) => sum + item.confidence, 0) / signHistory.length) * 100
      )
    : 0;

  const stats = [
    {
      label: "Current Mode",
      value: activeTab === "sign" ? "Sign Live" : "Speech Live",
      helper: activeTab === "sign" ? "Camera recognition" : "Voice translation",
      icon: activeTab === "sign" ? Camera : Mic,
    },
    {
      label: "Recognitions",
      value: signHistory.length,
      helper: "Recent session",
      icon: History,
    },
    {
      label: "Avg Confidence",
      value: signHistory.length ? `${avgConfidence}%` : "--",
      helper: latestSign ? `Last: ${latestSign.label}` : "Waiting for input",
      icon: Activity,
    },
  ];

  return (
    <div className="min-h-screen bg-[#f6f8fb] text-slate-950">
      <div className="flex min-h-screen">
        <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-white px-5 py-6 lg:block">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-teal-600 text-white">
              <Hand size={21} />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-950">Gestura</h1>
              <p className="text-xs font-medium text-slate-500">Communication Console</p>
            </div>
          </div>

          <nav className="mt-8 space-y-2">
            <button
              type="button"
              onClick={() => setActiveTab("sign")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left text-sm font-semibold transition-colors ${
                activeTab === "sign"
                  ? "bg-teal-50 text-teal-700 ring-1 ring-teal-100"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-950"
              }`}
            >
              <Camera size={18} />
              Sign to Speech
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("speech")}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left text-sm font-semibold transition-colors ${
                activeTab === "speech"
                  ? "bg-teal-50 text-teal-700 ring-1 ring-teal-100"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-950"
              }`}
            >
              <Mic size={18} />
              Speech to Sign
            </button>
          </nav>

          <div className="mt-8 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
              <Radio size={16} className="text-emerald-600" />
              Live services
            </div>
            <p className="mt-2 text-xs leading-5 text-slate-500">
              Backend model and websocket channels are used by the active workspace.
            </p>
          </div>
        </aside>

        <main className="flex-1 px-4 py-5 sm:px-6 lg:px-8">
          <header className="flex flex-col gap-4 border-b border-slate-200 pb-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-bold uppercase text-teal-700">Real-time translation</p>
              <h2 className="mt-1 text-2xl font-bold text-slate-950">Dashboard</h2>
              <p className="mt-1 text-sm text-slate-500">
                Monitor camera recognition, speech input, and generated sign output from one clean workspace.
              </p>
            </div>

            <div className="grid grid-cols-2 rounded-lg border border-slate-200 bg-white p-1 shadow-sm sm:flex">
              <button
                type="button"
                onClick={() => setActiveTab("sign")}
                className={`rounded-md px-4 py-2 text-sm font-semibold transition-colors ${
                  activeTab === "sign" ? "bg-slate-950 text-white" : "text-slate-600 hover:bg-slate-50"
                }`}
              >
                Sign to Speech
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("speech")}
                className={`rounded-md px-4 py-2 text-sm font-semibold transition-colors ${
                  activeTab === "speech" ? "bg-slate-950 text-white" : "text-slate-600 hover:bg-slate-50"
                }`}
              >
                Speech to Sign
              </button>
            </div>
          </header>

          <section className="mt-6 grid gap-4 md:grid-cols-3">
            {stats.map((stat) => {
              const Icon = stat.icon;
              return (
                <div key={stat.label} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-bold uppercase text-slate-500">{stat.label}</p>
                    <span className="grid h-8 w-8 place-items-center rounded-md bg-slate-100 text-slate-700">
                      <Icon size={17} />
                    </span>
                  </div>
                  <p className="mt-3 text-2xl font-bold text-slate-950">{stat.value}</p>
                  <p className="mt-1 truncate text-sm text-slate-500">{stat.helper}</p>
                </div>
              );
            })}
          </section>

          <section className="mt-6 grid grid-cols-1 gap-5 xl:grid-cols-12">
            {activeTab === "sign" ? (
              <>
                <div className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm xl:col-span-6">
                  <WebcamCapture onPrediction={handlePrediction} />
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm xl:col-span-6">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-bold uppercase text-slate-500">Recognition History</p>
                      <h3 className="mt-1 text-lg font-bold text-slate-950">Recent signs</h3>
                    </div>
                    <History size={20} className="text-slate-400" />
                  </div>

                  {signHistory.length === 0 ? (
                    <div className="mt-5 rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-8 text-center">
                      <p className="text-sm font-medium text-slate-700">No signs recognized yet</p>
                      <p className="mt-1 text-xs text-slate-500">Start the camera and begin signing.</p>
                    </div>
                  ) : (
                    <div className="mt-5 max-h-[520px] space-y-2 overflow-y-auto pr-1">
                      {signHistory.map((item, i) => (
                        <div
                          key={`${item.label}-${item.time}-${i}`}
                          className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-3"
                        >
                          <div className="min-w-0">
                            <p className="truncate text-sm font-bold uppercase text-slate-950">{item.label}</p>
                            <p className="text-xs text-slate-500">{item.time}</p>
                          </div>
                          <span className="rounded-md bg-emerald-100 px-2.5 py-1 text-xs font-bold text-emerald-700">
                            {Math.round(item.confidence * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm xl:col-span-5">
                  <MicrophoneRecorder onResult={handleSpeechResult} />
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm xl:col-span-7">
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-bold uppercase text-slate-500">Sign Output</p>
                      <h3 className="mt-1 text-lg font-bold text-slate-950">Generated playback</h3>
                    </div>
                    <Video size={20} className="text-slate-400" />
                  </div>
                  {transcript && (
                    <p className="mb-4 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
                      Heard: <span className="font-semibold text-slate-950">"{transcript}"</span>
                    </p>
                  )}
                  <VideoPlayer signSequence={signSequence} />
                </div>
              </>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
