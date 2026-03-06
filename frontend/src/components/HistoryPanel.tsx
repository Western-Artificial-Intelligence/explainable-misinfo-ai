import { useState, useMemo } from "react";
import {
  History, Search, Trash2, X, CheckCircle, XCircle, AlertCircle,
  ChevronDown, ChevronUp, Clock, Cpu, FileText, Globe, AlignLeft,
} from "lucide-react";
import type { HistoryEntry, Prediction } from "../hooks/useHistory";

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onDelete: (id: string) => void;
  onClearAll: () => void;
  onRestore: (entry: HistoryEntry) => void;
}

const predictionMeta: Record<Prediction, { icon: typeof CheckCircle; color: string; bg: string; border: string; badgeBg: string }> = {
  TRUE: {
    icon: CheckCircle,
    color: "text-emerald-600 dark:text-emerald-400",
    bg: "bg-emerald-50 dark:bg-emerald-950/40",
    border: "border-emerald-200 dark:border-emerald-800",
    badgeBg: "bg-emerald-500",
  },
  FALSE: {
    icon: XCircle,
    color: "text-red-600 dark:text-red-400",
    bg: "bg-red-50 dark:bg-red-950/40",
    border: "border-red-200 dark:border-red-800",
    badgeBg: "bg-red-500",
  },
  MIXED: {
    icon: AlertCircle,
    color: "text-amber-600 dark:text-amber-400",
    bg: "bg-amber-50 dark:bg-amber-950/40",
    border: "border-amber-200 dark:border-amber-800",
    badgeBg: "bg-amber-500",
  },
};

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const s = Math.floor(diff / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function sourceIcon(source: string) {
  if (source.startsWith("🌐") || source.startsWith("http")) return <Globe className="size-3 text-blue-500" />;
  if (source.startsWith("📄")) return <FileText className="size-3 text-indigo-500" />;
  return <AlignLeft className="size-3 text-gray-400" />;
}

function HistoryCard({ entry, onDelete, onRestore }: { entry: HistoryEntry; onDelete: () => void; onRestore: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const meta = predictionMeta[entry.prediction];
  const Icon = meta.icon;

  return (
    <div className={`rounded-xl border ${meta.border} bg-white dark:bg-gray-900 overflow-hidden transition-all duration-200 hover:shadow-md`}>
      <div className="p-3">
        <div className="flex items-start gap-2">
          {/* Verdict dot */}
          <div className={`mt-0.5 w-7 h-7 rounded-lg ${meta.badgeBg} flex items-center justify-center flex-shrink-0`}>
            <Icon className="size-3.5 text-white" />
          </div>

          <div className="flex-1 min-w-0">
            {/* Top row */}
            <div className="flex items-center gap-2 flex-wrap mb-0.5">
              <span className={`text-xs ${meta.color}`} style={{ fontWeight: 700 }}>{entry.prediction}</span>
              <span className="text-xs text-gray-400 dark:text-gray-600">·</span>
              <span className="text-xs text-gray-500 dark:text-gray-400">{entry.confidence}% confidence</span>
              <span className="text-xs text-gray-400 dark:text-gray-600">·</span>
              <span className="flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500">
                <Cpu className="size-3" />{entry.modelLabel}
              </span>
            </div>

            {/* Text preview */}
            <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed line-clamp-2">
              {entry.text}
            </p>

            {/* Source + time */}
            <div className="flex items-center gap-1.5 mt-1">
              {sourceIcon(entry.source)}
              <span className="text-xs text-gray-400 dark:text-gray-500 truncate max-w-[140px]">{entry.source || "Pasted text"}</span>
              <span className="text-xs text-gray-300 dark:text-gray-700">·</span>
              <Clock className="size-3 text-gray-300 dark:text-gray-700" />
              <span className="text-xs text-gray-400 dark:text-gray-500">{timeAgo(entry.timestamp)}</span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={() => setExpanded(!expanded)}
              className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title={expanded ? "Collapse" : "Expand"}
            >
              {expanded ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
            </button>
            <button
              onClick={onDelete}
              className="p-1 rounded-lg text-gray-300 dark:text-gray-700 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors"
              title="Delete"
            >
              <Trash2 className="size-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className={`px-3 pb-3 pt-0 border-t border-gray-100 dark:border-gray-800 ${meta.bg} space-y-2 animate-in slide-in-from-top-1 duration-200`}>
          <p className="text-xs text-gray-500 dark:text-gray-400 pt-2">{entry.detail}</p>
          <div className="h-1.5 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
            <div className={`h-full rounded-full ${meta.badgeBg} opacity-80`} style={{ width: `${entry.confidence}%` }} />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">{new Date(entry.timestamp).toLocaleString()}</span>
            <button
              onClick={onRestore}
              className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
              style={{ fontWeight: 600 }}
            >
              Restore →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export function HistoryPanel({ entries, onDelete, onClearAll, onRestore }: HistoryPanelProps) {
  const [query, setQuery] = useState("");
  const [filterVerdict, setFilterVerdict] = useState<Prediction | "ALL">("ALL");
  const [confirmClear, setConfirmClear] = useState(false);

  const filtered = useMemo(() => {
    let list = entries;
    if (filterVerdict !== "ALL") list = list.filter((e) => e.prediction === filterVerdict);
    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter(
        (e) =>
          e.text.toLowerCase().includes(q) ||
          e.source.toLowerCase().includes(q) ||
          e.prediction.toLowerCase().includes(q) ||
          e.modelLabel.toLowerCase().includes(q)
      );
    }
    return list;
  }, [entries, query, filterVerdict]);

  const verdictFilters: { label: string; value: Prediction | "ALL"; color: string }[] = [
    { label: "All", value: "ALL", color: "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400" },
    { label: "True", value: "TRUE", color: "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400" },
    { label: "False", value: "FALSE", color: "bg-red-100 dark:bg-red-950/60 text-red-700 dark:text-red-400" },
    { label: "Mixed", value: "MIXED", color: "bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400" },
  ];

  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <History className="size-4 text-blue-500" />
          <span className="text-sm text-gray-900 dark:text-white" style={{ fontWeight: 700 }}>Analysis History</span>
          {entries.length > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 text-xs" style={{ fontWeight: 600 }}>
              {entries.length}
            </span>
          )}
        </div>
        {entries.length > 0 && (
          confirmClear ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Clear all?</span>
              <button onClick={() => { onClearAll(); setConfirmClear(false); }} className="text-xs text-red-500 hover:text-red-700 transition-colors" style={{ fontWeight: 600 }}>Yes</button>
              <button onClick={() => setConfirmClear(false)} className="text-xs text-gray-400 hover:text-gray-600 transition-colors">No</button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmClear(true)}
              className="text-xs text-gray-400 hover:text-red-500 transition-colors flex items-center gap-1"
            >
              <Trash2 className="size-3" /> Clear all
            </button>
          )
        )}
      </div>

      {entries.length === 0 ? (
        <div className="p-8 text-center">
          <History className="size-10 text-gray-200 dark:text-gray-700 mx-auto mb-3" />
          <p className="text-sm text-gray-400 dark:text-gray-600">No analyses yet.</p>
          <p className="text-xs text-gray-300 dark:text-gray-700 mt-1">Results will appear here after you analyze text.</p>
        </div>
      ) : (
        <>
          {/* Search + filter */}
          <div className="p-3 border-b border-gray-100 dark:border-gray-800 space-y-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-3.5 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search history…"
                className="w-full pl-8 pr-8 py-2 text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              />
              {query && (
                <button onClick={() => setQuery("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">
                  <X className="size-3.5" />
                </button>
              )}
            </div>
            <div className="flex gap-1.5 flex-wrap">
              {verdictFilters.map((f) => (
                <button
                  key={f.value}
                  onClick={() => setFilterVerdict(f.value)}
                  className={`px-2.5 py-1 rounded-lg text-xs transition-all duration-150 ${
                    filterVerdict === f.value
                      ? f.color + " ring-2 ring-offset-1 ring-blue-400 dark:ring-offset-gray-900"
                      : "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                  }`}
                  style={{ fontWeight: 600 }}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          {/* List */}
          <div className="overflow-y-auto max-h-[480px] p-3 space-y-2">
            {filtered.length === 0 ? (
              <div className="text-center py-6">
                <Search className="size-7 text-gray-200 dark:text-gray-700 mx-auto mb-2" />
                <p className="text-xs text-gray-400 dark:text-gray-600">No results for "{query}"</p>
              </div>
            ) : (
              filtered.map((entry) => (
                <HistoryCard
                  key={entry.id}
                  entry={entry}
                  onDelete={() => onDelete(entry.id)}
                  onRestore={() => onRestore(entry)}
                />
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
