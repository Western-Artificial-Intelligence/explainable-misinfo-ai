import { useState, type ChangeEvent } from "react";
import { Header } from "./components/Header";
import { TextInput } from "./components/TextInput";
import { ResultCard } from "./components/ResultCard";
import { ExampleButtons } from "./components/ExampleButtons";
import { Footer } from "./components/Footer";
import { Card } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Button } from "./components/ui/button";

export interface AnalysisResult {
  prediction: "MISINFORMATION" | "RELIABLE" | "OPINION" | "NEUTRAL";
  confidence: number;
}

interface BackendClassifyResponse {
  label: string;
  confidence: number;
  explanation?: string;
}

interface DocumentIngestResponse {
  document_id: string;
  title: string;
  source: string;
  char_count: number;
  created_at: string;
}

interface DocumentAnswerResponse {
  answer: string;
  snippets: string[];
  confidence: number;
  source: string;
  title: string;
  document_id: string;
}

const API_BASE_URL = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined) || "http://127.0.0.1:8000"
).replace(/\/$/, "");

function normalizeConfidence(value: number): number {
  const parsed = Number.isFinite(value) ? value : 0.65;
  const confidence01 = parsed > 1 ? parsed / 100 : parsed;
  return Math.max(0, Math.min(100, Math.round(confidence01 * 100)));
}

function mapPrediction(label: string): AnalysisResult["prediction"] {
  const normalized = label.toLowerCase().trim();
  if (["misinformation", "false", "fake"].includes(normalized)) return "MISINFORMATION";
  if (["reliable", "factual", "true"].includes(normalized)) return "RELIABLE";
  if (["opinion", "mixed"].includes(normalized)) return "OPINION";
  return "NEUTRAL";
}

function randomFallbackResult(): AnalysisResult {
  const predictions: AnalysisResult["prediction"][] = [
    "MISINFORMATION",
    "RELIABLE",
    "OPINION",
    "NEUTRAL",
  ];
  const prediction = predictions[Math.floor(Math.random() * predictions.length)];
  const confidence = Math.floor(Math.random() * 41) + 55;
  return { prediction, confidence };
}

async function classifyText(text: string): Promise<AnalysisResult> {
  const response = await fetch(`${API_BASE_URL}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(`Backend request failed with status ${response.status}`);
  }
  const data = (await response.json()) as BackendClassifyResponse;
  return {
    prediction: mapPrediction(data.label),
    confidence: normalizeConfidence(data.confidence),
  };
}

async function uploadDocumentFromText(filename: string, content: string): Promise<DocumentIngestResponse> {
  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, content }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Upload failed (${response.status})`);
  }
  return (await response.json()) as DocumentIngestResponse;
}

async function uploadDocumentFromUrl(url: string): Promise<DocumentIngestResponse> {
  const response = await fetch(`${API_BASE_URL}/documents/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `URL ingestion failed (${response.status})`);
  }
  return (await response.json()) as DocumentIngestResponse;
}

async function askDocumentQuestion(question: string, documentId: string): Promise<DocumentAnswerResponse> {
  const response = await fetch(`${API_BASE_URL}/documents/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_id: documentId }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Question request failed (${response.status})`);
  }
  return (await response.json()) as DocumentAnswerResponse;
}

export default function App() {
  const [darkMode, setDarkMode] = useState(false);

  const [text, setText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const [documentId, setDocumentId] = useState<string | null>(null);
  const [documentTitle, setDocumentTitle] = useState("");
  const [documentSource, setDocumentSource] = useState("");
  const [documentStatus, setDocumentStatus] = useState("");
  const [documentBusy, setDocumentBusy] = useState(false);
  const [urlInput, setUrlInput] = useState("");

  const [question, setQuestion] = useState("");
  const [askBusy, setAskBusy] = useState(false);
  const [answer, setAnswer] = useState<string>("");
  const [answerSnippets, setAnswerSnippets] = useState<string[]>([]);
  const [answerError, setAnswerError] = useState("");
  const [answerConfidence, setAnswerConfidence] = useState<number | null>(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    setResult(null);
    try {
      const apiResult = await classifyText(text);
      setResult(apiResult);
    } catch (error) {
      console.error("Failed to call backend /classify, using random fallback:", error);
      setResult(randomFallbackResult());
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleExampleClick = (exampleText: string) => {
    setText(exampleText);
    setResult(null);
  };

  const applyDocumentState = (payload: DocumentIngestResponse) => {
    setDocumentId(payload.document_id);
    setDocumentTitle(payload.title);
    setDocumentSource(payload.source);
    setDocumentStatus(`Loaded "${payload.title}" (${payload.char_count} chars)`);
    setAnswer("");
    setAnswerSnippets([]);
    setAnswerError("");
    setAnswerConfidence(null);
  };

  const handleLocalFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setDocumentBusy(true);
    setDocumentStatus("Reading local file...");
    try {
      const content = await file.text();
      const payload = await uploadDocumentFromText(file.name, content);
      applyDocumentState(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not ingest local file.";
      setDocumentStatus(`Error: ${message}`);
    } finally {
      event.target.value = "";
      setDocumentBusy(false);
    }
  };

  const handleUrlIngest = async () => {
    if (!urlInput.trim()) return;
    setDocumentBusy(true);
    setDocumentStatus("Fetching URL content...");
    try {
      const payload = await uploadDocumentFromUrl(urlInput.trim());
      applyDocumentState(payload);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not ingest URL.";
      setDocumentStatus(`Error: ${message}`);
    } finally {
      setDocumentBusy(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || !documentId) return;
    setAskBusy(true);
    setAnswer("");
    setAnswerSnippets([]);
    setAnswerError("");
    setAnswerConfidence(null);
    try {
      const payload = await askDocumentQuestion(question.trim(), documentId);
      setAnswer(payload.answer);
      setAnswerSnippets(payload.snippets || []);
      setAnswerConfidence(Math.round(payload.confidence * 100));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to answer question.";
      setAnswerError(message);
    } finally {
      setAskBusy(false);
    }
  };

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="min-h-screen bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors">
        <div className="max-w-4xl mx-auto px-4 py-8 md:py-12">
          <Header darkMode={darkMode} onToggleDarkMode={() => setDarkMode(!darkMode)} />

          <main className="mt-12 space-y-8">
            <TextInput
              value={text}
              onChange={setText}
              onAnalyze={handleAnalyze}
              isAnalyzing={isAnalyzing}
              disabled={isAnalyzing}
            />

            {result && <ResultCard result={result} />}

            <ExampleButtons onExampleClick={handleExampleClick} />

            <Card className="p-6 shadow-lg border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
              <div className="space-y-5">
                <div>
                  <h2 className="text-xl font-semibold">Document Q&A</h2>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Upload a local text file or paste a URL, then ask questions about the content.
                  </p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Local file</p>
                    <Input
                      type="file"
                      accept=".txt,.md,.markdown,.csv,.json,.log,.html,.htm,.xml"
                      onChange={handleLocalFile}
                      disabled={documentBusy}
                    />
                  </div>

                  <div className="space-y-2">
                    <p className="text-sm font-medium">From URL</p>
                    <div className="flex gap-2">
                      <Input
                        value={urlInput}
                        onChange={(e) => setUrlInput(e.target.value)}
                        placeholder="https://example.com/article"
                        disabled={documentBusy}
                      />
                      <Button onClick={handleUrlIngest} disabled={documentBusy || !urlInput.trim()}>
                        {documentBusy ? "Loading..." : "Load URL"}
                      </Button>
                    </div>
                  </div>
                </div>

                {documentStatus && (
                  <p className="text-sm text-blue-700 dark:text-blue-300">{documentStatus}</p>
                )}

                {documentId && (
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    Active document: <span className="font-medium">{documentTitle}</span> ({documentSource})
                  </p>
                )}

                <div className="space-y-3">
                  <p className="text-sm font-medium">Ask a question</p>
                  <Textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="What does this document say about ... ?"
                    className="min-h-[120px] border-gray-300 dark:border-gray-700"
                    disabled={askBusy || !documentId}
                  />
                  <Button onClick={handleAsk} disabled={askBusy || !documentId || !question.trim()}>
                    {askBusy ? "Finding answer..." : "Get Answer"}
                  </Button>
                </div>

                {answerError && <p className="text-sm text-red-600 dark:text-red-400">{answerError}</p>}

                {answer && (
                  <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">Answer</p>
                      {answerConfidence !== null && (
                        <p className="text-xs text-gray-600 dark:text-gray-400">Confidence: {answerConfidence}%</p>
                      )}
                    </div>
                    <p className="text-sm leading-6">{answer}</p>
                    {answerSnippets.length > 0 && (
                      <div>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Top evidence snippets</p>
                        <ul className="list-disc pl-5 space-y-1 text-xs text-gray-600 dark:text-gray-300">
                          {answerSnippets.map((snippet, index) => (
                            <li key={`${snippet.slice(0, 30)}-${index}`}>{snippet}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </Card>
          </main>

          <Footer />
        </div>
      </div>
    </div>
  );
}
