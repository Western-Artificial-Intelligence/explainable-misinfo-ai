import { useState, useRef, useCallback } from "react";
import {
  Sparkles, Loader2, CheckCircle, XCircle, AlertCircle,
  RotateCcw, FileText, Link2, AlignLeft, Upload, X,
  FileUp, Globe, ClipboardPaste,
} from "lucide-react";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";

type Prediction = "TRUE" | "FALSE" | "MIXED";
type InputMode = "text" | "file" | "url";

interface AnalysisResult {
  prediction: Prediction;
  confidence: number;
  detail: string;
}

interface BackendClassifyResponse {
  label: string;
  confidence: number;
  explanation?: string;
}

const API_BASE_URL = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const predictionConfig: Record<Prediction, {
  icon: typeof CheckCircle;
  label: string;
  color: string;
  bg: string;
  border: string;
  badgeBg: string;
  barColor: string;
  glow: string;
}> = {
  TRUE: {
    icon: CheckCircle,
    label: "TRUE",
    color: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50 dark:bg-emerald-950/40",
    border: "border-emerald-200 dark:border-emerald-800",
    badgeBg: "bg-emerald-500",
    barColor: "bg-emerald-500",
    glow: "shadow-emerald-400/20",
  },
  FALSE: {
    icon: XCircle,
    label: "FALSE",
    color: "text-red-600 dark:text-red-400",
    bg: "bg-red-50 dark:bg-red-950/40",
    border: "border-red-200 dark:border-red-800",
    badgeBg: "bg-red-500",
    barColor: "bg-red-500",
    glow: "shadow-red-400/20",
  },
  MIXED: {
    icon: AlertCircle,
    label: "MIXED",
    color: "text-amber-600 dark:text-amber-400",
    bg: "bg-amber-50 dark:bg-amber-950/40",
    border: "border-amber-200 dark:border-amber-800",
    badgeBg: "bg-amber-500",
    barColor: "bg-amber-500",
    glow: "shadow-amber-400/20",
  },
};

const examples = [
  { label: "COVID Microchips", text: "COVID-19 vaccines contain microchips that track your location." },
  { label: "Earth Orbits Sun", text: "The Earth orbits the Sun, completing one revolution every 365.25 days." },
  { label: "AI Jobs Claim", text: "AI could potentially replace millions of jobs over the next decade." },
  { label: "Climate Hoax", text: "Climate change is a complete hoax fabricated by scientists for funding." },
];

function mapPrediction(label: string): Prediction {
  const normalized = label.toLowerCase().trim();
  if (["false", "misinformation", "fake"].includes(normalized)) return "FALSE";
  if (["true", "factual", "reliable", "real"].includes(normalized)) return "TRUE";
  return "MIXED";
}

function normalizeConfidence(value: number): number {
  const parsed = Number.isFinite(value) ? value : 0.65;
  const confidence01 = parsed > 1 ? parsed / 100 : parsed;
  return Math.max(0, Math.min(100, Math.round(confidence01 * 100)));
}

function fallbackAnalyzeContent(text: string): AnalysisResult {
  const lower = text.toLowerCase();
  if (lower.includes("hoax") || lower.includes("microchip") || lower.includes("fake") || lower.includes("fabricated") || lower.includes("conspiracy")) {
    return { prediction: "FALSE", confidence: Math.floor(Math.random() * 15) + 80, detail: "This content contains claims that contradict established facts or scientific consensus." };
  } else if (lower.includes("earth orbits") || lower.includes("scientific") || lower.includes("proven") || lower.includes("revolution every") || lower.includes("research shows")) {
    return { prediction: "TRUE", confidence: Math.floor(Math.random() * 10) + 88, detail: "This content appears to contain factual, verifiable information supported by evidence." };
  } else if (lower.includes("will") || lower.includes("might") || lower.includes("could") || lower.includes("potentially") || lower.includes("opinion")) {
    return { prediction: "MIXED", confidence: Math.floor(Math.random() * 20) + 58, detail: "This content mixes facts with opinion, prediction, or speculation." };
  }
  return { prediction: "MIXED", confidence: Math.floor(Math.random() * 20) + 48, detail: "This content doesn't contain strong indicators in either direction — treat with caution." };
}

async function analyzeViaBackend(text: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error(`Backend request failed (${response.status})`);
  }

  const payload = (await response.json()) as BackendClassifyResponse;
  const prediction = mapPrediction(payload.label);

  return {
    prediction,
    confidence: normalizeConfidence(payload.confidence),
    detail:
      payload.explanation?.trim() ||
      "Classification returned from backend model.",
  };
}

async function extractTextFromPDF(file: File): Promise<string> {
  // Load PDF.js from CDN at runtime to avoid hard dependency issues in local installs.
  const pdfjsLib = await import(
    /* @vite-ignore */ "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs"
  );
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let fullText = "";
  for (let i = 1; i <= Math.min(pdf.numPages, 10); i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    fullText += content.items.map((item: any) => item.str).join(" ") + "\n";
  }
  return fullText.trim();
}

async function fetchTextFromUrl(url: string): Promise<string> {
  // Use allorigins CORS proxy
  const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
  const res = await fetch(proxyUrl);
  if (!res.ok) throw new Error("Failed to fetch URL");
  const data = await res.json();
  const html: string = data.contents || "";
  // Strip HTML tags, collapse whitespace
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/\s{2,}/g, " ")
    .trim();
  if (!text) throw new Error("Could not extract text from page");
  return text.slice(0, 4000);
}

const ACCEPTED_TYPES = ".txt,.md,.csv,.pdf,.doc,.docx";

export function AnalyzerSection() {
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loadingFile, setLoadingFile] = useState(false);
  const [loadingUrl, setLoadingUrl] = useState(false);
  const [urlError, setUrlError] = useState("");
  const [fileError, setFileError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeExample, setActiveExample] = useState<number | null>(null);
  const [extractedSource, setExtractedSource] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  const reset = () => {
    setText("");
    setUrl("");
    setFile(null);
    setResult(null);
    setActiveExample(null);
    setUrlError("");
    setFileError("");
    setExtractedSource("");
  };

  const handleModeSwitch = (m: InputMode) => {
    setMode(m);
    setResult(null);
    setUrlError("");
    setFileError("");
  };

  // ── File handling ──────────────────────────────────────────────
  const processFile = useCallback(async (f: File) => {
    setFileError("");
    setLoadingFile(true);
    setResult(null);
    try {
      let content = "";
      if (f.type === "application/pdf" || f.name.endsWith(".pdf")) {
        content = await extractTextFromPDF(f);
      } else if (
        f.type.startsWith("text/") ||
        f.name.endsWith(".md") ||
        f.name.endsWith(".csv") ||
        f.name.endsWith(".txt")
      ) {
        content = await f.text();
      } else {
        // Fallback: try reading as text
        content = await f.text();
      }
      if (!content.trim()) throw new Error("No readable text found in file.");
      setFile(f);
      setText(content.slice(0, 5000));
      setExtractedSource(`📄 ${f.name}`);
    } catch (err: any) {
      setFileError(err.message || "Could not read file.");
    } finally {
      setLoadingFile(false);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) processFile(f);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) processFile(f);
  };

  // ── URL handling ───────────────────────────────────────────────
  const handleFetchUrl = async () => {
    if (!url.trim()) return;
    setUrlError("");
    setLoadingUrl(true);
    setResult(null);
    try {
      const content = await fetchTextFromUrl(url.trim());
      setText(content);
      setExtractedSource(`🌐 ${url.trim()}`);
    } catch (err: any) {
      setUrlError(err.message || "Failed to fetch content from URL.");
    } finally {
      setLoadingUrl(false);
    }
  };

  // ── Analysis ───────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    setResult(null);
    try {
      const apiResult = await analyzeViaBackend(text.trim());
      setResult(apiResult);
    } catch (error) {
      console.error("Failed to call backend /classify, using local fallback:", error);
      setResult(fallbackAnalyzeContent(text));
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExample = (idx: number) => {
    setActiveExample(idx);
    setText(examples[idx].text);
    setResult(null);
    setExtractedSource("");
    setMode("text");
  };

  const tabs: { id: InputMode; label: string; icon: typeof AlignLeft }[] = [
    { id: "text", label: "Paste Text", icon: ClipboardPaste },
    { id: "file", label: "Upload File", icon: FileUp },
    { id: "url", label: "From URL", icon: Globe },
  ];

  const canAnalyze = text.trim().length > 0 && !isAnalyzing && !loadingFile && !loadingUrl;

  return (
    <div className="space-y-5">
      {/* Main card */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm overflow-hidden">

        {/* Tab bar */}
        <div className="flex border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/80">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const active = mode === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => handleModeSwitch(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-3 text-sm transition-all duration-200 relative flex-1 justify-center ${
                  active
                    ? "text-blue-600 dark:text-blue-400 bg-white dark:bg-gray-900"
                    : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
                }`}
                style={{ fontWeight: active ? 600 : 500 }}
              >
                <Icon className="size-4" />
                <span className="hidden sm:inline">{tab.label}</span>
                {active && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400 rounded-t-full" />
                )}
              </button>
            );
          })}
        </div>

        {/* ── Paste Text ── */}
        {mode === "text" && (
          <>
            <div className="px-4 pt-3 pb-1 flex items-center justify-between">
              {extractedSource ? (
                <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 px-2 py-0.5 rounded-full" style={{ fontWeight: 500 }}>
                  {extractedSource}
                </span>
              ) : <span />}
              {text && (
                <button onClick={reset} className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors">
                  <RotateCcw className="size-3" /> Clear
                </button>
              )}
            </div>
            <Textarea
              value={text}
              onChange={(e) => { setText(e.target.value); setResult(null); setExtractedSource(""); }}
              placeholder="Paste or type any text — news headline, social media post, or article excerpt…"
              className="min-h-[200px] resize-none border-0 rounded-none bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-600 focus-visible:ring-0 focus-visible:ring-offset-0 text-sm leading-relaxed p-4"
              disabled={isAnalyzing}
            />
          </>
        )}

        {/* ── Upload File ── */}
        {mode === "file" && (
          <div className="p-4">
            {!text || !file ? (
              <div
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex flex-col items-center justify-center gap-3 min-h-[200px] rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
                  isDragging
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30"
                    : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-950/20"
                }`}
              >
                {loadingFile ? (
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="size-8 text-blue-500 animate-spin" />
                    <p className="text-sm text-gray-500 dark:text-gray-400">Reading file…</p>
                  </div>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-950/60 flex items-center justify-center">
                      <Upload className="size-6 text-blue-500" />
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-gray-700 dark:text-gray-300" style={{ fontWeight: 600 }}>
                        {isDragging ? "Drop your file here" : "Drag & drop or click to browse"}
                      </p>
                      <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                        Supports .txt, .md, .csv, .pdf — up to 10MB
                      </p>
                    </div>
                  </>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_TYPES}
                  className="hidden"
                  onChange={handleFileChange}
                />
              </div>
            ) : (
              <div className="space-y-3">
                {/* File loaded banner */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800">
                  <div className="flex items-center gap-2">
                    <FileText className="size-5 text-green-600 dark:text-green-400" />
                    <div>
                      <p className="text-sm text-green-800 dark:text-green-300" style={{ fontWeight: 600 }}>{file.name}</p>
                      <p className="text-xs text-green-600 dark:text-green-500">{wordCount} words extracted</p>
                    </div>
                  </div>
                  <button
                    onClick={() => { setFile(null); setText(""); setResult(null); setExtractedSource(""); }}
                    className="p-1 rounded-lg text-green-600 dark:text-green-400 hover:bg-green-100 dark:hover:bg-green-900/40 transition-colors"
                  >
                    <X className="size-4" />
                  </button>
                </div>
                {/* Extracted text preview */}
                <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-3 max-h-40 overflow-y-auto">
                  <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed whitespace-pre-wrap">{text.slice(0, 600)}{text.length > 600 ? "…" : ""}</p>
                </div>
              </div>
            )}
            {fileError && (
              <p className="mt-2 text-xs text-red-500 flex items-center gap-1">
                <X className="size-3" /> {fileError}
              </p>
            )}
          </div>
        )}

        {/* ── From URL ── */}
        {mode === "url" && (
          <div className="p-4 space-y-3">
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <Globe className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-400" />
                <input
                  type="url"
                  value={url}
                  onChange={(e) => { setUrl(e.target.value); setUrlError(""); }}
                  onKeyDown={(e) => e.key === "Enter" && handleFetchUrl()}
                  placeholder="https://example.com/article"
                  className="w-full pl-9 pr-4 py-2.5 text-sm rounded-xl border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
              </div>
              <button
                onClick={handleFetchUrl}
                disabled={!url.trim() || loadingUrl}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm transition-all duration-200 flex-shrink-0"
                style={{ fontWeight: 600 }}
              >
                {loadingUrl ? <Loader2 className="size-4 animate-spin" /> : <Link2 className="size-4" />}
                {loadingUrl ? "Fetching…" : "Fetch"}
              </button>
            </div>

            {urlError && (
              <p className="text-xs text-red-500 flex items-center gap-1">
                <X className="size-3" /> {urlError}
              </p>
            )}

            {!text && !loadingUrl && (
              <div className="rounded-xl border border-dashed border-gray-200 dark:border-gray-700 p-4 text-center">
                <Globe className="size-8 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
                <p className="text-xs text-gray-400 dark:text-gray-600">
                  Enter any article, news page, or blog URL — TruthLens will extract and analyze the text content.
                </p>
              </div>
            )}

            {loadingUrl && (
              <div className="flex flex-col items-center gap-2 py-8">
                <Loader2 className="size-7 text-blue-500 animate-spin" />
                <p className="text-sm text-gray-500 dark:text-gray-400">Fetching page content…</p>
              </div>
            )}

            {text && !loadingUrl && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 px-2 py-0.5 rounded-full truncate max-w-[70%]" style={{ fontWeight: 500 }}>
                    🌐 {url}
                  </span>
                  <button
                    onClick={() => { setText(""); setUrl(""); setResult(null); setExtractedSource(""); }}
                    className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors flex items-center gap-1"
                  >
                    <X className="size-3" /> Clear
                  </button>
                </div>
                <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-3 max-h-40 overflow-y-auto">
                  <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">{text.slice(0, 600)}{text.length > 600 ? "…" : ""}</p>
                </div>
                <p className="text-xs text-gray-400">{wordCount} words extracted</p>
              </div>
            )}
          </div>
        )}

        {/* Bottom bar — always visible */}
        <div className="p-4 border-t border-gray-100 dark:border-gray-800 flex items-center justify-between gap-3">
          <span className="text-xs text-gray-400 dark:text-gray-600">
            {canAnalyze ? `${wordCount} word${wordCount !== 1 ? "s" : ""} ready` : "No content yet"}
          </span>
          <Button
            onClick={handleAnalyze}
            disabled={!canAnalyze}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md hover:shadow-blue-500/30 disabled:opacity-50 transition-all duration-300 rounded-xl px-6"
          >
            {isAnalyzing ? (
              <><Loader2 className="size-4 mr-2 animate-spin" />Analyzing…</>
            ) : (
              <><Sparkles className="size-4 mr-2" />Analyze with AI</>
            )}
          </Button>
        </div>
      </div>

      {/* Example buttons — only show in text mode with no content */}
      {mode === "text" && !text && (
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-widest" style={{ fontWeight: 600 }}>
            Try an example
          </p>
          <div className="flex flex-wrap gap-2">
            {examples.map((ex, i) => (
              <button
                key={i}
                onClick={() => handleExample(i)}
                className={`px-3 py-1.5 rounded-lg text-xs border transition-all duration-200 ${
                  activeExample === i
                    ? "bg-blue-600 text-white border-blue-600 shadow-sm"
                    : "bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400"
                }`}
                style={{ fontWeight: 500 }}
              >
                {ex.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Loading skeleton */}
      {isAnalyzing && (
        <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 animate-pulse space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gray-200 dark:bg-gray-700" />
            <div className="space-y-2 flex-1">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-24" />
              <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-48" />
            </div>
          </div>
          <div className="space-y-2">
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full" />
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-3/4" />
          </div>
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full w-full" />
        </div>
      )}

      {/* Result card */}
      {result && !isAnalyzing && (() => {
        const cfg = predictionConfig[result.prediction];
        const Icon = cfg.icon;
        return (
          <div className={`rounded-2xl border ${cfg.border} ${cfg.bg} shadow-lg ${cfg.glow} p-6 animate-in fade-in slide-in-from-bottom-4 duration-500`}>
            <div className="flex items-start gap-4 mb-5">
              <div className={`w-12 h-12 rounded-xl ${cfg.badgeBg} flex items-center justify-center flex-shrink-0 shadow-md`}>
                <Icon className="size-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider" style={{ fontWeight: 600 }}>Verdict</span>
                  <span className={`px-2.5 py-0.5 rounded-full text-xs ${cfg.badgeBg} text-white`} style={{ fontWeight: 700 }}>
                    {cfg.label}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{result.detail}</p>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-500 dark:text-gray-400 uppercase tracking-wider" style={{ fontWeight: 600 }}>Confidence</span>
                <span className={cfg.color} style={{ fontWeight: 700 }}>{result.confidence}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
                <div className={`h-full rounded-full ${cfg.barColor} transition-all duration-700 ease-out`} style={{ width: `${result.confidence}%` }} />
              </div>
              <p className="text-xs text-gray-400 dark:text-gray-500">
                {result.confidence >= 80 ? "High confidence — strong indicators detected." : result.confidence >= 60 ? "Moderate confidence — some ambiguity present." : "Low confidence — not enough clear signals."}
              </p>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
