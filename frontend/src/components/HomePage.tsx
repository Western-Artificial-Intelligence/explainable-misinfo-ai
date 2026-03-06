import { useState } from "react";
import { ArrowRight, Puzzle, Shield, Zap, Globe, CheckCircle, XCircle, AlertCircle } from "lucide-react";
import { AnalyzerSection } from "./AnalyzerSection";
import { HistoryPanel } from "./HistoryPanel";
import { useHistory } from "../hooks/useHistory";
import type { HistoryEntry } from "../hooks/useHistory";

export type Prediction = "TRUE" | "FALSE" | "MIXED";

export interface AnalysisResult {
  prediction: Prediction;
  confidence: number;
}

export function HomePage() {
  const { entries, addEntry, deleteEntry, clearAll } = useHistory();
  const [restoredEntry, setRestoredEntry] = useState<HistoryEntry | null>(null);
  const [localSaveNotice, setLocalSaveNotice] = useState("");

  const handleRestore = (entry: HistoryEntry) => {
    setRestoredEntry({ ...entry });
    // Scroll to top of analyzer
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleResultSaved = (entry: Omit<HistoryEntry, "id" | "timestamp">) => {
    const saved = addEntry(entry);
    setLocalSaveNotice(`Saved locally on this device at ${new Date(saved.timestamp).toLocaleTimeString()}`);
    window.setTimeout(() => setLocalSaveNotice(""), 3000);
  };

  const features = [
    {
      icon: Zap,
      title: "Instant Analysis",
      desc: "Get results in seconds using state-of-the-art transformer models trained on verified datasets.",
      color: "text-blue-500",
      bg: "bg-blue-50 dark:bg-blue-950/40",
    },
    {
      icon: Globe,
      title: "Browser Extension",
      desc: "Analyze any article or claim on the web without leaving your browser tab.",
      color: "text-orange-500",
      bg: "bg-orange-50 dark:bg-orange-950/40",
    },
    {
      icon: Shield,
      title: "Privacy First",
      desc: "History is stored in your browser on this device, with optional backend backup if enabled.",
      color: "text-indigo-500",
      bg: "bg-indigo-50 dark:bg-indigo-950/40",
    },
  ];

  const verdicts = [
    { icon: CheckCircle, label: "TRUE", desc: "Verified, factual claims", color: "text-emerald-500", bg: "bg-emerald-50 dark:bg-emerald-950/40" },
    { icon: XCircle, label: "FALSE", desc: "Misinformation detected", color: "text-red-500", bg: "bg-red-50 dark:bg-red-950/40" },
    { icon: AlertCircle, label: "MIXED", desc: "Opinion or partial facts", color: "text-amber-500", bg: "bg-amber-50 dark:bg-amber-950/40" },
  ];

  return (
    <main>
      {/* Hero + Analyzer combined */}
      <section className="relative overflow-hidden">
        {/* Background gradient blobs */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-40 -left-40 w-80 h-80 bg-blue-400/20 dark:bg-blue-600/10 rounded-full blur-3xl" />
          <div className="absolute top-20 right-0 w-96 h-96 bg-indigo-400/15 dark:bg-indigo-600/10 rounded-full blur-3xl" />
          <div className="absolute bottom-0 left-1/2 w-64 h-64 bg-orange-400/10 dark:bg-orange-600/10 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-3xl mx-auto px-4 sm:px-6 pt-14 pb-16">
          {/* Centered headline */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-100 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 text-xs mb-5 border border-blue-200 dark:border-blue-800" style={{ fontWeight: 600 }}>
              <Zap className="size-3" />
              Powered by AI Transformer Models
            </div>

            <h1 className="text-gray-900 dark:text-white mb-3" style={{ fontSize: "clamp(1.9rem, 4.5vw, 3rem)", lineHeight: 1.15, fontWeight: 800 }}>
              Stop Misinformation{" "}
              <span className="bg-gradient-to-r from-blue-600 to-indigo-500 bg-clip-text text-transparent">
                Before It Spreads
              </span>
            </h1>

            <p className="text-gray-600 dark:text-gray-400 max-w-lg mx-auto" style={{ fontSize: "1rem", lineHeight: 1.65 }}>
              Paste any claim, article, or post — get an instant{" "}
              <span className="text-emerald-600 dark:text-emerald-400" style={{ fontWeight: 600 }}>TRUE</span>,{" "}
              <span className="text-red-600 dark:text-red-400" style={{ fontWeight: 600 }}>FALSE</span>, or{" "}
              <span className="text-amber-600 dark:text-amber-400" style={{ fontWeight: 600 }}>MIXED</span>{" "}
              verdict powered by AI.
            </p>
          </div>

          {/* Analyzer — front and center */}
          <AnalyzerSection
            onResult={handleResultSaved}
            restoredEntry={restoredEntry}
          />

          {localSaveNotice && (
            <div className="mt-3 rounded-xl border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/30 px-3 py-2">
              <p className="text-sm text-emerald-700 dark:text-emerald-300" style={{ fontWeight: 600 }}>
                {localSaveNotice}
              </p>
            </div>
          )}

          {/* Chrome extension nudge below analyzer */}
          <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-3">
            <a
              href="https://chrome.google.com/webstore"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white shadow-md hover:shadow-orange-500/30 transition-all duration-300 group text-sm"
              style={{ fontWeight: 600 }}
            >
              <Puzzle className="size-4 group-hover:rotate-12 transition-transform duration-300" />
              Add to Chrome — Free
              <ArrowRight className="size-3.5 opacity-70 group-hover:translate-x-0.5 transition-transform duration-200" />
            </a>
            <span className="text-xs text-gray-400 dark:text-gray-600">Analyze directly in your browser, no tab switching</span>
          </div>
        </div>
      </section>

      {/* History */}
      <section className="py-12 border-t border-gray-200 dark:border-gray-800">
        <div className="max-w-3xl mx-auto px-4 sm:px-6">
          <HistoryPanel
            entries={entries}
            onDelete={deleteEntry}
            onClearAll={clearAll}
            onRestore={handleRestore}
          />
        </div>
      </section>

      {/* Verdict legend */}
      <section className="py-10 border-t border-gray-200 dark:border-gray-800">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="flex flex-wrap justify-center gap-3">
            {verdicts.map((v) => {
              const Icon = v.icon;
              return (
                <div key={v.label} className={`flex items-center gap-2 px-4 py-2 rounded-full ${v.bg} border border-gray-200 dark:border-gray-700 shadow-sm`}>
                  <Icon className={`size-4 ${v.color}`} />
                  <span className={`text-sm ${v.color}`} style={{ fontWeight: 700 }}>{v.label}</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">{v.desc}</span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 border-t border-gray-200 dark:border-gray-800">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-10">
            <h2 className="text-gray-900 dark:text-white mb-2" style={{ fontSize: "1.75rem", fontWeight: 700 }}>Why TruthLens?</h2>
            <p className="text-gray-500 dark:text-gray-400 text-sm">Built for speed, accuracy, and privacy.</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {features.map((f) => {
              const Icon = f.icon;
              return (
                <div key={f.title} className="group p-6 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
                  <div className={`w-11 h-11 rounded-xl ${f.bg} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className={`size-5 ${f.color}`} />
                  </div>
                  <h3 className="text-gray-900 dark:text-white mb-2" style={{ fontWeight: 600, fontSize: "1rem" }}>{f.title}</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">{f.desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Chrome Extension CTA */}
      <section className="py-16 border-t border-gray-200 dark:border-gray-800">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-orange-500 via-amber-500 to-orange-600 p-8 md:p-12 text-white shadow-xl shadow-orange-400/20">
            <div className="absolute -top-10 -right-10 w-48 h-48 bg-white/10 rounded-full blur-2xl pointer-events-none" />
            <div className="absolute -bottom-8 -left-8 w-40 h-40 bg-white/10 rounded-full blur-2xl pointer-events-none" />
            <div className="relative flex flex-col md:flex-row items-center justify-between gap-6">
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Puzzle className="size-6" />
                  <span className="text-sm tracking-widest uppercase opacity-80" style={{ fontWeight: 600 }}>Chrome Extension</span>
                </div>
                <h2 className="mb-2 text-white" style={{ fontSize: "1.75rem", fontWeight: 800, lineHeight: 1.2 }}>
                  Fact-check anywhere,<br />without switching tabs
                </h2>
                <p className="opacity-80 text-sm max-w-md leading-relaxed">The TruthLens extension analyzes highlighted text directly in your browser. One click. Instant verdict.</p>
              </div>
              <div className="flex flex-col gap-3 flex-shrink-0">
                <a href="https://chrome.google.com/webstore" target="_blank" rel="noopener noreferrer"
                  className="flex items-center justify-center gap-2 px-7 py-3.5 rounded-xl bg-white text-orange-600 hover:bg-orange-50 shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all duration-300 group"
                  style={{ fontWeight: 700, fontSize: "0.95rem", minWidth: 200 }}
                >
                  <Puzzle className="size-5 group-hover:rotate-12 transition-transform duration-300" />
                  Add to Chrome — Free
                </a>
                <p className="text-center text-xs opacity-70">Works on Chrome 88+ · No sign-in needed</p>
              </div>
            </div>
            <div className="relative mt-8 flex flex-wrap gap-3">
              {[
                { label: "TRUE", color: "bg-emerald-400/30 text-emerald-100 border-emerald-400/40" },
                { label: "FALSE", color: "bg-red-400/30 text-red-100 border-red-400/40" },
                { label: "MIXED", color: "bg-amber-400/30 text-amber-100 border-amber-400/40" },
              ].map((b) => (
                <span key={b.label} className={`px-3 py-1 rounded-full border text-xs ${b.color}`} style={{ fontWeight: 700 }}>{b.label}</span>
              ))}
              <span className="text-xs opacity-60 self-center ml-1">· Verdicts appear inline on any page</span>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
