import { useState, useRef, useCallback, useEffect } from "react";
import { createPortal } from "react-dom";
import {
  Sparkles, Loader2, CheckCircle, XCircle, AlertCircle,
  RotateCcw, FileText, Link2, Upload, X,
  FileUp, Globe, ClipboardPaste, ChevronDown, Cpu,
} from "lucide-react";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import type { ModelId, HistoryEntry } from "../hooks/useHistory";

export type Prediction = "TRUE" | "FALSE" | "MIXED";
type InputMode = "text" | "file" | "url";

export interface AnalysisResult {
  prediction: Prediction;
  confidence: number;
  detail: string;
  reasoning: string;
  detectedClaims: string[];
  supportingEvidence: Array<{ title: string; url: string; snippet: string }>;
  rawResult: Record<string, unknown>;
}

interface Citation {
  title?: string;
  url?: string;
}

interface LegacyAnalysisPayload {
  prediction?: string;
  label?: string;
  confidence?: number;
  explanation?: string;
  detail?: string;
  sources?: Citation[];
}

interface ProductionSummary {
  intro?: string;
  body1?: string;
  body2?: string;
  conclusion?: string;
  verdict?: string;
  confidence?: number;
  citations?: Citation[];
}

interface ProductionAnalysisPayload {
  final_prediction?: string;
  confidence?: number;
  summary?: ProductionSummary;
  roberta?: {
    confidence?: number;
    label?: {
      class_name?: string;
    };
  };
  last_stage_output?: {
    summary?: ProductionSummary;
    roberta?: {
      confidence?: number;
      label?: {
        class_name?: string;
      };
    };
  };
  meta?: {
    warnings?: Array<{
      message?: string;
    }>;
  };
}

// ── Model definitions ──────────────────────────────────────────────────────
export const MODELS: { id: ModelId; label: string; shortLabel: string; desc: string; badge: string; badgeColor: string }[] = [
  {
    id: "roberta",
    label: "RoBERTa",
    shortLabel: "RoBERTa",
    desc: "Robustly Optimized BERT — fast, accurate, fine-tuned on misinformation datasets.",
    badge: "Recommended",
    badgeColor: "bg-blue-100 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400",
  },
  {
    id: "llm-encoder",
    label: "LLM as Encoder",
    shortLabel: "LLM Enc.",
    desc: "Uses a large language model's encoder layers for deeper contextual understanding.",
    badge: "Experimental",
    badgeColor: "bg-purple-100 dark:bg-purple-950/60 text-purple-600 dark:text-purple-400",
  },
];

const predictionConfig: Record<Prediction, {
  icon: typeof CheckCircle; label: string; color: string; bg: string;
  border: string; badgeBg: string; barColor: string; glow: string;
}> = {
  TRUE: { icon: CheckCircle, label: "TRUE", color: "text-emerald-600 dark:text-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-950/40", border: "border-emerald-200 dark:border-emerald-800", badgeBg: "bg-emerald-500", barColor: "bg-emerald-500", glow: "shadow-emerald-400/20" },
  FALSE: { icon: XCircle, label: "FALSE", color: "text-red-600 dark:text-red-400", bg: "bg-red-50 dark:bg-red-950/40", border: "border-red-200 dark:border-red-800", badgeBg: "bg-red-500", barColor: "bg-red-500", glow: "shadow-red-400/20" },
  MIXED: { icon: AlertCircle, label: "MIXED", color: "text-amber-600 dark:text-amber-400", bg: "bg-amber-50 dark:bg-amber-950/40", border: "border-amber-200 dark:border-amber-800", badgeBg: "bg-amber-500", barColor: "bg-amber-500", glow: "shadow-amber-400/20" },
};

const examples = [
  { label: "COVID Microchips", text: "COVID-19 vaccines contain microchips that track your location." },
  { label: "Earth Orbits Sun", text: "The Earth orbits the Sun, completing one revolution every 365.25 days." },
  { label: "AI Jobs Claim", text: "AI could potentially replace millions of jobs over the next decade." },
  { label: "Climate Hoax", text: "Climate change is a complete hoax fabricated by scientists for funding." },
];

const API_BASE_URL = ((import.meta.env.VITE_API_BASE_URL as string | undefined) || "").replace(/\/$/, "");
const FRONTEND_SESSION_KEY = "truthlens-web-session-id";

function apiUrl(path: string) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

function normalizePrediction(value: string): Prediction {
  const raw = String(value || "").trim().toUpperCase();
  if (raw === "TRUE" || raw === "FALSE" || raw === "MIXED") return raw;
  if (raw === "REAL" || raw === "RELIABLE" || raw === "FACTUAL") return "TRUE";
  if (raw === "FAKE" || raw === "MISINFORMATION" || raw === "NOT_REAL") return "FALSE";
  return "MIXED";
}

function normalizeConfidence(value: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const asPercent = numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, Number(asPercent.toFixed(2))));
}

function sourceToUrl(source: string): string {
  const trimmed = source.trim();
  if (trimmed.startsWith("🌐")) {
    return trimmed.replace(/^🌐\s*/, "");
  }
  return trimmed.startsWith("http://") || trimmed.startsWith("https://") ? trimmed : "";
}

async function getErrorMessage(response: Response): Promise<string> {
  try {
    const text = await response.text();
    const parsed = JSON.parse(text);
    if (typeof parsed?.detail === "string") {
      return parsed.detail;
    }
    if (typeof parsed?.detail?.message === "string") {
      return parsed.detail.message;
    }
    if (typeof parsed?.message === "string") {
      return parsed.message;
    }
    if (typeof parsed?.error === "string") {
      return parsed.error;
    }
    return text || `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

function getOrCreateFrontendSessionId(): string {
  try {
    const existing = localStorage.getItem(FRONTEND_SESSION_KEY);
    if (existing && existing.trim()) return existing;
    const generated = `web_${Date.now()}_${crypto.randomUUID()}`;
    localStorage.setItem(FRONTEND_SESSION_KEY, generated);
    return generated;
  } catch {
    return `web_${Date.now()}`;
  }
}

function mapInputType(mode: InputMode): string {
  if (mode === "url") return "page_analysis";
  if (mode === "file") return "document";
  return "selected_text";
}

function extractReasoning(payload: any): string {
  const summary = payload?.summary || {};
  return String(summary?.conclusion || summary?.intro || payload?.backend_log || "No reasoning returned by pipeline.");
}

function extractDetectedClaims(payload: any): string[] {
  const claims = [payload?.normalized_claim, payload?.user_claim]
    .map((item) => String(item || "").trim())
    .filter(Boolean);
  return [...new Set(claims)].slice(0, 5);
}

function extractSupportingEvidence(payload: any): Array<{ title: string; url: string; snippet: string }> {
  const summaryCitations = Array.isArray(payload?.summary?.citations) ? payload.summary.citations : [];
  if (summaryCitations.length > 0) {
    return summaryCitations
      .map((citation: any) => ({
        title: String(citation?.title || citation?.source || "Evidence source"),
        url: String(citation?.url || ""),
        snippet: String(citation?.snippet || ""),
      }))
      .slice(0, 5);
  }

  const topItems = Array.isArray(payload?.evidence_topk?.items) ? payload.evidence_topk.items : [];
  return topItems
    .map((item: any) => ({
      title: String(item?.doc?.title || item?.doc?.source || "Evidence source"),
      url: String(item?.doc?.url || ""),
      snippet: String(item?.text || ""),
    }))
    .filter((item: { title: string; url: string; snippet: string }) => item.snippet || item.title || item.url)
    .slice(0, 5);
}

async function analyzeWithBackend(
  text: string,
  mode: InputMode,
  pageUrl: string,
): Promise<AnalysisResult> {
  const response = await fetch(apiUrl("/api/process"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_claim: text,
      session_id: getOrCreateFrontendSessionId(),
      source_context: "frontend_website",
      input_type: mapInputType(mode),
      input_text: text,
      transcript: mode === "text" ? text : "",
      page_url: pageUrl || "",
      persist_result: false,
      timestamp: Date.now(),
    }),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const payload = await response.json();
  const roberta = payload?.roberta || {};
  const summary = payload?.summary || {};
  const label = roberta?.label || {};
  const verdictRaw = summary?.verdict || label?.class_name || "";
  const confidenceRaw = summary?.confidence ?? roberta?.confidence ?? label?.confidence ?? 0;
  const reasoning = extractReasoning(payload);
  const detectedClaims = extractDetectedClaims(payload);
  const supportingEvidence = extractSupportingEvidence(payload);

  return {
    prediction: normalizePrediction(String(verdictRaw)),
    confidence: normalizeConfidence(Number(confidenceRaw)),
    detail: reasoning,
    reasoning,
    detectedClaims,
    supportingEvidence,
    rawResult: payload,
  };
}

async function persistAnalysisToMongo(args: {
  mode: InputMode;
  text: string;
  url: string;
  result: AnalysisResult;
}) {
  const { mode, text, url, result } = args;
  try {
    await fetch(apiUrl("/analysis"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: getOrCreateFrontendSessionId(),
        input_type: mapInputType(mode),
        input_text: text,
        transcript: mode === "text" ? text : "",
        page_url: mode === "url" ? url.trim() : "",
        analysis_result: result.rawResult,
        confidence: Number((result.confidence / 100).toFixed(4)),
        reasoning: result.reasoning,
        verdict: result.prediction,
        timestamp: Date.now(),
        source_context: "frontend_website",
      }),
    });
  } catch {
    // Best effort only. Local history save remains intact.
  }
}

async function extractTextFromPDF(file: File): Promise<string> {
  const pdfjsLib = await import("pdfjs-dist");
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;
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
  const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
  const res = await fetch(proxyUrl);
  if (!res.ok) throw new Error("Failed to fetch URL");
  const data = await res.json();
  const html: string = data.contents || "";
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
    .replace(/\s{2,}/g, " ").trim();
  if (!text) throw new Error("Could not extract text from page");
  return text.slice(0, 4000);
}

const ACCEPTED_TYPES = ".txt,.md,.csv,.pdf,.doc,.docx";

// ── Model Selector Dropdown ────────────────────────────────────────────────
function ModelSelector({ selected, onChange }: { selected: ModelId; onChange: (id: ModelId) => void }) {
  const [open, setOpen] = useState(false);
  const [dropdownStyle, setDropdownStyle] = useState<React.CSSProperties>({});
  const buttonRef = useRef<HTMLButtonElement>(null);
  const current = MODELS.find((m) => m.id === selected)!;

  // Recalculate position whenever the dropdown opens
  useEffect(() => {
    if (!open || !buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    setDropdownStyle({
      position: "fixed",
      top: rect.bottom + 8,
      left: rect.left,
      width: 288,
      zIndex: 9999,
    });
  }, [open]);

  // Close on scroll/resize so position stays accurate
  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs transition-all duration-200 group"
        style={{ fontWeight: 600 }}
      >
        <Cpu className="size-3.5 text-blue-500" />
        <span className="hidden sm:inline">{current.shortLabel}</span>
        <ChevronDown className={`size-3 text-gray-400 transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>

      {open && createPortal(
        <>
          {/* Click-outside backdrop */}
          <div className="fixed inset-0" style={{ zIndex: 9998 }} onClick={() => setOpen(false)} />

          {/* Dropdown panel */}
          <div
            style={dropdownStyle}
            className="rounded-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 shadow-xl overflow-hidden animate-in slide-in-from-top-2 duration-200"
          >
            <div className="px-3 pt-3 pb-2 border-b border-gray-100 dark:border-gray-800">
              <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-widest" style={{ fontWeight: 600 }}>Select Model</p>
            </div>
            {MODELS.map((model) => {
              const active = model.id === selected;
              return (
                <button
                  key={model.id}
                  onClick={() => { onChange(model.id); setOpen(false); }}
                  className={`w-full text-left px-3 py-3 transition-colors duration-150 ${active ? "bg-blue-50 dark:bg-blue-950/40" : "hover:bg-gray-50 dark:hover:bg-gray-800"}`}
                >
                  <div className="flex items-start gap-2">
                    <div className={`mt-0.5 w-4 h-4 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${active ? "border-blue-500" : "border-gray-300 dark:border-gray-600"}`}>
                      {active && <div className="w-2 h-2 rounded-full bg-blue-500" />}
                    </div>
                    <div>
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-sm text-gray-900 dark:text-white" style={{ fontWeight: 600 }}>{model.label}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${model.badgeColor}`} style={{ fontWeight: 600 }}>{model.badge}</span>
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">{model.desc}</p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </>,
        document.body
      )}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────
interface AnalyzerSectionProps {
  onResult?: (entry: Omit<HistoryEntry, "id" | "timestamp">) => void;
  restoredEntry?: HistoryEntry | null;
}

export function AnalyzerSection({ onResult, restoredEntry }: AnalyzerSectionProps) {
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loadingFile, setLoadingFile] = useState(false);
  const [loadingUrl, setLoadingUrl] = useState(false);
  const [urlError, setUrlError] = useState("");
  const [fileError, setFileError] = useState("");
  const [analyzeError, setAnalyzeError] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeExample, setActiveExample] = useState<number | null>(null);
  const [extractedSource, setExtractedSource] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<ModelId>("roberta");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;

  const reset = () => { setText(""); setUrl(""); setFile(null); setResult(null); setActiveExample(null); setUrlError(""); setFileError(""); setAnalyzeError(""); setExtractedSource(""); };
  const handleModeSwitch = (m: InputMode) => { setMode(m); setResult(null); setUrlError(""); setFileError(""); setAnalyzeError(""); };

  useEffect(() => {
    if (!restoredEntry) return;
    setText(restoredEntry.text || "");
    setResult({
      prediction: restoredEntry.prediction,
      confidence: restoredEntry.confidence,
      detail: restoredEntry.detail,
      reasoning: restoredEntry.detail,
      detectedClaims: [restoredEntry.text].filter(Boolean),
      supportingEvidence: [],
      rawResult: {},
    });
    setActiveExample(null);
    setSelectedModel(restoredEntry.model);
    setExtractedSource(restoredEntry.source || "Pasted text");
    setUrl(sourceToUrl(restoredEntry.source || ""));
    setFile(null);
    setMode("text");
    setUrlError("");
    setFileError("");
    setAnalyzeError("");
  }, [restoredEntry]);

  const processFile = useCallback(async (f: File) => {
    setFileError(""); setAnalyzeError(""); setLoadingFile(true); setResult(null);
    try {
      let content = "";
      if (f.type === "application/pdf" || f.name.endsWith(".pdf")) content = await extractTextFromPDF(f);
      else content = await f.text();
      if (!content.trim()) throw new Error("No readable text found in file.");
      setFile(f); setText(content.slice(0, 5000)); setExtractedSource(`📄 ${f.name}`);
    } catch (err: any) { setFileError(err.message || "Could not read file."); }
    finally { setLoadingFile(false); }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => { const f = e.target.files?.[0]; if (f) processFile(f); };
  const handleDrop = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); const f = e.dataTransfer.files?.[0]; if (f) processFile(f); };

  const handleFetchUrl = async () => {
    if (!url.trim()) return;
    setUrlError(""); setAnalyzeError(""); setLoadingUrl(true); setResult(null);
    try { const content = await fetchTextFromUrl(url.trim()); setText(content); setExtractedSource(`🌐 ${url.trim()}`); }
    catch (err: any) { setUrlError(err.message || "Failed to fetch content from URL."); }
    finally { setLoadingUrl(false); }
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    setAnalyzeError("");
    setResult(null);
    try {
      const res = await analyzeWithBackend(text, mode, mode === "url" ? url : "");
      setResult(res);
      const modelMeta = MODELS.find((m) => m.id === selectedModel)!;
      onResult?.({
        text: text.slice(0, 5000),
        source: extractedSource || (mode === "url" ? url : mode === "file" && file ? `📄 ${file.name}` : "Pasted text"),
        prediction: res.prediction,
        confidence: res.confidence,
        detail: res.detail,
        model: selectedModel,
        modelLabel: modelMeta.label,
      });
      void persistAnalysisToMongo({
        mode,
        text,
        url,
        result: res,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Analysis failed. Please try again.";
      setAnalyzeError(message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExample = (idx: number) => { setActiveExample(idx); setText(examples[idx].text); setResult(null); setAnalyzeError(""); setExtractedSource(""); setMode("text"); };

  const tabs = [
    { id: "text" as InputMode, label: "Paste Text", icon: ClipboardPaste },
    { id: "file" as InputMode, label: "Upload File", icon: FileUp },
    { id: "url" as InputMode, label: "From URL", icon: Globe },
  ];

  const canAnalyze = text.trim().length > 0 && !isAnalyzing && !loadingFile && !loadingUrl;

  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm overflow-hidden">
        {/* Tab bar */}
        <div className="flex border-b border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/80">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const active = mode === tab.id;
            return (
              <button key={tab.id} onClick={() => handleModeSwitch(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-3 text-sm transition-all duration-200 relative flex-1 justify-center ${active ? "text-blue-600 dark:text-blue-400 bg-white dark:bg-gray-900" : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"}`}
                style={{ fontWeight: active ? 600 : 500 }}
              >
                <Icon className="size-4" />
                <span className="hidden sm:inline">{tab.label}</span>
                {active && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-400 rounded-t-full" />}
              </button>
            );
          })}
        </div>

        {/* ── Paste Text ── */}
        {mode === "text" && (
          <>
            <div className="px-4 pt-3 pb-1 flex items-center justify-between">
              {extractedSource ? (
                <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 px-2 py-0.5 rounded-full" style={{ fontWeight: 500 }}>{extractedSource}</span>
              ) : <span />}
              {text && <button onClick={reset} className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"><RotateCcw className="size-3" /> Clear</button>}
            </div>
            <Textarea value={text} onChange={(e) => { setText(e.target.value); setResult(null); setAnalyzeError(""); setExtractedSource(""); }}
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
              <div onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex flex-col items-center justify-center gap-3 min-h-[200px] rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 ${isDragging ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30" : "border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-950/20"}`}
              >
                {loadingFile ? (
                  <div className="flex flex-col items-center gap-2"><Loader2 className="size-8 text-blue-500 animate-spin" /><p className="text-sm text-gray-500 dark:text-gray-400">Reading file…</p></div>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-xl bg-blue-100 dark:bg-blue-950/60 flex items-center justify-center"><Upload className="size-6 text-blue-500" /></div>
                    <div className="text-center">
                      <p className="text-sm text-gray-700 dark:text-gray-300" style={{ fontWeight: 600 }}>{isDragging ? "Drop your file here" : "Drag & drop or click to browse"}</p>
                      <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">Supports .txt, .md, .csv, .pdf — up to 10MB</p>
                    </div>
                  </>
                )}
                <input ref={fileInputRef} type="file" accept={ACCEPTED_TYPES} className="hidden" onChange={handleFileChange} />
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 rounded-xl bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-800">
                  <div className="flex items-center gap-2">
                    <FileText className="size-5 text-green-600 dark:text-green-400" />
                    <div>
                      <p className="text-sm text-green-800 dark:text-green-300" style={{ fontWeight: 600 }}>{file.name}</p>
                      <p className="text-xs text-green-600 dark:text-green-500">{wordCount} words extracted</p>
                    </div>
                  </div>
                  <button onClick={() => { setFile(null); setText(""); setResult(null); setAnalyzeError(""); setExtractedSource(""); }} className="p-1 rounded-lg text-green-600 dark:text-green-400 hover:bg-green-100 dark:hover:bg-green-900/40 transition-colors"><X className="size-4" /></button>
                </div>
                <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-3 max-h-40 overflow-y-auto">
                  <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed whitespace-pre-wrap">{text.slice(0, 600)}{text.length > 600 ? "…" : ""}</p>
                </div>
              </div>
            )}
            {fileError && <p className="mt-2 text-xs text-red-500 flex items-center gap-1"><X className="size-3" /> {fileError}</p>}
          </div>
        )}

        {/* ── From URL ── */}
        {mode === "url" && (
          <div className="p-4 space-y-3">
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <Globe className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-400" />
                <input type="url" value={url} onChange={(e) => { setUrl(e.target.value); setUrlError(""); }} onKeyDown={(e) => e.key === "Enter" && handleFetchUrl()}
                  placeholder="https://example.com/article"
                  className="w-full pl-9 pr-4 py-2.5 text-sm rounded-xl border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                />
              </div>
              <button onClick={handleFetchUrl} disabled={!url.trim() || loadingUrl}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-sm transition-all flex-shrink-0"
                style={{ fontWeight: 600 }}
              >
                {loadingUrl ? <Loader2 className="size-4 animate-spin" /> : <Link2 className="size-4" />}
                {loadingUrl ? "Fetching…" : "Fetch"}
              </button>
            </div>
            {urlError && <p className="text-xs text-red-500 flex items-center gap-1"><X className="size-3" /> {urlError}</p>}
            {!text && !loadingUrl && (
              <div className="rounded-xl border border-dashed border-gray-200 dark:border-gray-700 p-4 text-center">
                <Globe className="size-8 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
                <p className="text-xs text-gray-400 dark:text-gray-600">Enter any article, news page, or blog URL — TruthLens will extract and analyze the text content.</p>
              </div>
            )}
            {loadingUrl && <div className="flex flex-col items-center gap-2 py-8"><Loader2 className="size-7 text-blue-500 animate-spin" /><p className="text-sm text-gray-500 dark:text-gray-400">Fetching page content…</p></div>}
            {text && !loadingUrl && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-800 px-2 py-0.5 rounded-full truncate max-w-[70%]" style={{ fontWeight: 500 }}>🌐 {url}</span>
                  <button onClick={() => { setText(""); setUrl(""); setResult(null); setAnalyzeError(""); setExtractedSource(""); }} className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors flex items-center gap-1"><X className="size-3" /> Clear</button>
                </div>
                <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-3 max-h-40 overflow-y-auto">
                  <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">{text.slice(0, 600)}{text.length > 600 ? "…" : ""}</p>
                </div>
                <p className="text-xs text-gray-400">{wordCount} words extracted</p>
              </div>
            )}
          </div>
        )}

        {/* ── Bottom bar ── */}
        <div className="p-3 border-t border-gray-100 dark:border-gray-800 flex items-center gap-2">
          <span className="text-xs text-gray-400 dark:text-gray-600 hidden sm:block flex-1">
            {canAnalyze ? `${wordCount} word${wordCount !== 1 ? "s" : ""} ready` : "No content yet"}
          </span>

          <div className="flex-1 sm:flex-none" />

          <Button onClick={handleAnalyze} disabled={!canAnalyze}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md hover:shadow-blue-500/30 disabled:opacity-50 transition-all duration-300 rounded-xl px-5"
          >
            {isAnalyzing ? <><Loader2 className="size-4 mr-2 animate-spin" />Analyzing…</> : <><Sparkles className="size-4 mr-2" />Analyze with AI</>}
          </Button>
        </div>
        {analyzeError && (
          <div className="px-3 pb-3">
            <p className="text-xs text-red-500 flex items-center gap-1">
              <X className="size-3" />
              {analyzeError}
            </p>
          </div>
        )}
      </div>

      {/* Model selector — between Analyze button and Try an example */}
      <div className="flex items-center gap-2">
        <ModelSelector selected={selectedModel} onChange={setSelectedModel} />
      </div>

      {/* Example buttons */}
      {mode === "text" && !text && (
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-widest" style={{ fontWeight: 600 }}>Try an example</p>
          <div className="flex flex-wrap gap-2">
            {examples.map((ex, i) => (
              <button key={i} onClick={() => handleExample(i)}
                className={`px-3 py-1.5 rounded-lg text-xs border transition-all duration-200 ${activeExample === i ? "bg-blue-600 text-white border-blue-600 shadow-sm" : "bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400"}`}
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
            <div className="space-y-2 flex-1"><div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-24" /><div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-48" /></div>
          </div>
          <div className="space-y-2"><div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full" /><div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-3/4" /></div>
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full w-full" />
        </div>
      )}

      {/* Result card */}
      {result && !isAnalyzing && (() => {
        const cfg = predictionConfig[result.prediction];
        const Icon = cfg.icon;
        const modelMeta = MODELS.find((m) => m.id === selectedModel)!;
        return (
          <div className={`rounded-2xl border ${cfg.border} ${cfg.bg} shadow-lg ${cfg.glow} p-6 animate-in fade-in slide-in-from-bottom-4 duration-500`}>
            <div className="flex items-start gap-4 mb-5">
              <div className={`w-12 h-12 rounded-xl ${cfg.badgeBg} flex items-center justify-center flex-shrink-0 shadow-md`}>
                <Icon className="size-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider" style={{ fontWeight: 600 }}>Verdict</span>
                  <span className={`px-2.5 py-0.5 rounded-full text-xs ${cfg.badgeBg} text-white`} style={{ fontWeight: 700 }}>{cfg.label}</span>
                  <span className={`px-2 py-0.5 rounded-full text-xs flex items-center gap-1 ${modelMeta.badgeColor}`} style={{ fontWeight: 600 }}>
                    <Cpu className="size-3" />{modelMeta.label}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{result.reasoning}</p>
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
            {result.detectedClaims.length > 0 && (
              <div className="mt-5">
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2" style={{ fontWeight: 600 }}>Detected Claims</p>
                <div className="space-y-2">
                  {result.detectedClaims.map((claim, idx) => (
                    <p key={`${claim}-${idx}`} className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                      {claim}
                    </p>
                  ))}
                </div>
              </div>
            )}
            {result.supportingEvidence.length > 0 && (
              <div className="mt-5">
                <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2" style={{ fontWeight: 600 }}>Supporting Evidence</p>
                <div className="space-y-3">
                  {result.supportingEvidence.map((item, idx) => (
                    <div key={`${item.title}-${idx}`} className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-gray-900/40 p-3">
                      <p className="text-sm text-gray-800 dark:text-gray-200" style={{ fontWeight: 600 }}>{item.title || "Evidence source"}</p>
                      {item.url && (
                        <a href={item.url} target="_blank" rel="noreferrer" className="text-xs text-blue-600 dark:text-blue-400 break-all">
                          {item.url}
                        </a>
                      )}
                      {item.snippet && (
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1 leading-relaxed">
                          {item.snippet.length > 220 ? `${item.snippet.slice(0, 220)}...` : item.snippet}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}
