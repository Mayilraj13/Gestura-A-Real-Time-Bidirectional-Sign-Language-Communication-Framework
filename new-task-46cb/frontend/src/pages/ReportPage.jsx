import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const CHAPTERS = [
  { id: "toc",  title: "Table of Contents",      file: "/report/TABLE_OF_CONTENTS.md" },
  { id: "ch1",  title: "1. Introduction",         file: "/report/CHAPTER_1_INTRODUCTION.md" },
  { id: "ch2",  title: "2. Literature Survey",    file: "/report/CHAPTER_2_LITERATURE_SURVEY.md" },
  { id: "ch3",  title: "3. Existing System",      file: "/report/CHAPTER_3_EXISTING_SYSTEM.md" },
  { id: "ch4",  title: "4. Proposed System",      file: "/report/CHAPTER_4_PROPOSED_SYSTEM.md" },
  { id: "ch5",  title: "5. System Requirements",  file: "/report/CHAPTER_5_SYSTEM_REQUIREMENTS.md" },
  { id: "ch6",  title: "6. Implementation",       file: "/report/CHAPTER_6_SYSTEM_IMPLEMENTATION.md" },
  { id: "ch7",  title: "7. Experimental Results", file: "/report/CHAPTER_7_EXPERIMENTAL_RESULTS.md" },
  { id: "ch8",  title: "8. Conclusion & Future",  file: "/report/CHAPTER_8_CONCLUSION_FUTURE_WORK.md" },
  { id: "figs", title: "Figures & Diagrams",      file: "/report/FIGURES_AND_DIAGRAMS.md" },
];

export default function ReportPage() {
  const [activeId, setActiveId] = useState("toc");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const contentRef = useRef(null);

  const loadChapter = useCallback((id) => {
    const chapter = CHAPTERS.find((c) => c.id === id);
    if (!chapter) return;
    setActiveId(id);
    setLoading(true);
    setError(null);
    setContent("");
    fetch(chapter.file)
      .then((res) => {
        if (!res.ok) throw new Error(`Could not load file (HTTP ${res.status})`);
        return res.text();
      })
      .then((text) => {
        setContent(text);
        setLoading(false);
        if (contentRef.current) contentRef.current.scrollTop = 0;
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    loadChapter("toc");
  }, [loadChapter]);

  return (
    <div className="flex" style={{ height: "calc(100vh - 65px)" }}>
      <aside className="hidden lg:flex flex-col w-64 flex-shrink-0 bg-slate-800 border-r border-slate-700 overflow-y-auto">
        <div className="p-4 border-b border-slate-700">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            Project Report
          </p>
          <p className="text-xs text-slate-500 mt-1">Gestura</p>
        </div>
        <nav className="p-3 space-y-1 flex-1">
          {CHAPTERS.map((ch) => (
            <button
              key={ch.id}
              onClick={() => loadChapter(ch.id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                activeId === ch.id
                  ? "bg-blue-600 text-white font-medium"
                  : "text-slate-300 hover:bg-slate-700 hover:text-white"
              }`}
            >
              {ch.title}
            </button>
          ))}
        </nav>
      </aside>

      <div className="lg:hidden sticky top-0 z-10 bg-slate-800 border-b border-slate-700 px-4 py-2 w-full">
        <select
          value={activeId}
          onChange={(e) => loadChapter(e.target.value)}
          className="w-full bg-slate-700 border border-slate-600 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {CHAPTERS.map((ch) => (
            <option key={ch.id} value={ch.id}>
              {ch.title}
            </option>
          ))}
        </select>
      </div>

      <main
        ref={contentRef}
        className="flex-1 overflow-y-auto bg-slate-900 px-6 py-8 lg:px-12 lg:py-10"
      >
        {loading && (
          <div className="flex items-center gap-3 text-slate-400">
            <span className="w-4 h-4 rounded-full border-2 border-slate-500 border-t-blue-400 animate-spin" />
            <span className="text-sm">Loading chapter…</span>
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-red-700 bg-red-900/30 px-5 py-4 text-red-300 text-sm">
            <p className="font-semibold mb-1">Failed to load chapter</p>
            <p className="text-xs text-red-400">{error}</p>
            <p className="text-xs text-red-500 mt-2">
              Make sure the markdown files are in{" "}
              <code className="bg-red-900/50 px-1 rounded">frontend/public/report/</code>
            </p>
          </div>
        )}

        {!loading && !error && content && (
          <article className="prose prose-invert prose-slate max-w-4xl
            prose-headings:text-white prose-headings:font-bold
            prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg
            prose-p:text-slate-300 prose-p:leading-relaxed
            prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
            prose-strong:text-white
            prose-code:text-emerald-300 prose-code:bg-slate-800 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none
            prose-pre:bg-slate-800 prose-pre:border prose-pre:border-slate-700
            prose-table:text-slate-300
            prose-th:text-white prose-th:bg-slate-700 prose-th:border prose-th:border-slate-600
            prose-td:border prose-td:border-slate-700
            prose-tr:even:bg-slate-800/50
            prose-li:text-slate-300
            prose-hr:border-slate-700
            prose-blockquote:border-blue-500 prose-blockquote:text-slate-400
          ">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </article>
        )}

        {!loading && !error && !content && (
          <p className="text-slate-500 text-sm">Select a chapter from the sidebar.</p>
        )}
      </main>
    </div>
  );
}
