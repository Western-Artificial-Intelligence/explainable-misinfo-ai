import { useState, useEffect, useCallback } from "react";

export type Prediction = "TRUE" | "FALSE" | "MIXED";
export type ModelId = "roberta" | "llm-encoder";

export interface HistoryEntry {
  id: string;
  text: string;
  source: string; // "text" | filename | url
  prediction: Prediction;
  confidence: number;
  detail: string;
  model: ModelId;
  modelLabel: string;
  timestamp: number;
}

const STORAGE_KEY = "truthlens-history";
const MAX_ENTRIES = 100;
const API_BASE_URL = ((import.meta.env.VITE_API_BASE_URL as string | undefined) || "").replace(/\/$/, "");
const HISTORY_SYNC_TO_BACKEND = String(import.meta.env.VITE_HISTORY_SYNC_TO_BACKEND ?? "true").toLowerCase() === "true";
const HISTORY_LOAD_REMOTE = String(import.meta.env.VITE_HISTORY_LOAD_REMOTE ?? "false").toLowerCase() === "true";

function apiUrl(path: string) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

function loadFromStorage(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveToStorage(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {}
}

function normalizePrediction(value: string): Prediction {
  const raw = String(value || "").toUpperCase();
  if (raw === "TRUE" || raw === "FALSE" || raw === "MIXED") return raw;
  if (raw === "REAL" || raw === "FACTUAL" || raw === "RELIABLE") return "TRUE";
  if (raw === "FAKE" || raw === "MISINFORMATION" || raw === "NOT_REAL") return "FALSE";
  return "MIXED";
}

function normalizeConfidence(value: number): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const asPercent = numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, Number(asPercent.toFixed(2))));
}

function normalizeEntry(entry: Partial<HistoryEntry>): HistoryEntry {
  return {
    id: String(entry.id || crypto.randomUUID()),
    text: String(entry.text || ""),
    source: String(entry.source || "Pasted text"),
    prediction: normalizePrediction(String(entry.prediction || "MIXED")),
    confidence: normalizeConfidence(Number(entry.confidence || 0)),
    detail: String(entry.detail || ""),
    model: (entry.model === "llm-encoder" ? "llm-encoder" : "roberta"),
    modelLabel: String(entry.modelLabel || "RoBERTa"),
    timestamp: Number(entry.timestamp || Date.now()),
  };
}

async function fetchRemoteHistory(): Promise<HistoryEntry[] | null> {
  try {
    const response = await fetch(apiUrl(`/history?limit=${MAX_ENTRIES}&include_central=true`));
    if (!response.ok) return null;
    const data = await response.json();
    const entries = Array.isArray(data?.entries) ? data.entries : [];
    return entries.map((entry: Partial<HistoryEntry>) => normalizeEntry(entry));
  } catch {
    return null;
  }
}

async function persistRemoteEntry(entry: HistoryEntry): Promise<void> {
  try {
    await fetch(apiUrl("/history"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
  } catch {}
}

async function deleteRemoteEntry(id: string): Promise<void> {
  try {
    await fetch(apiUrl(`/history/${encodeURIComponent(id)}`), { method: "DELETE" });
  } catch {}
}

async function clearRemoteEntries(): Promise<void> {
  try {
    await fetch(apiUrl("/history"), { method: "DELETE" });
  } catch {}
}

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(loadFromStorage);

  useEffect(() => {
    saveToStorage(entries);
  }, [entries]);

  useEffect(() => {
    if (!HISTORY_LOAD_REMOTE) return;
    void (async () => {
      const remoteEntries = await fetchRemoteHistory();
      if (remoteEntries) {
        setEntries((prev) => {
          const merged = [...prev, ...remoteEntries];
          const byId = new Map<string, HistoryEntry>();
          merged.forEach((entry) => byId.set(entry.id, normalizeEntry(entry)));
          return [...byId.values()]
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, MAX_ENTRIES);
        });
      }
    })();
  }, []);

  const addEntry = useCallback((entry: Omit<HistoryEntry, "id" | "timestamp">) => {
    const newEntry = normalizeEntry({
      ...entry,
      id: crypto.randomUUID(),
      timestamp: Date.now(),
    });
    setEntries((prev) => [newEntry, ...prev].slice(0, MAX_ENTRIES));
    if (HISTORY_SYNC_TO_BACKEND) {
      void persistRemoteEntry(newEntry);
    }
    return newEntry;
  }, []);

  const deleteEntry = useCallback((id: string) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
    if (HISTORY_SYNC_TO_BACKEND) {
      void deleteRemoteEntry(id);
    }
  }, []);

  const clearAll = useCallback(() => {
    setEntries([]);
    if (HISTORY_SYNC_TO_BACKEND) {
      void clearRemoteEntries();
    }
  }, []);

  return { entries, addEntry, deleteEntry, clearAll };
}
