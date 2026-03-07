const DEFAULT_API_BASE = "http://127.0.0.1:8011";
const LEGACY_API_BASES = new Set(["http://localhost:8000", "http://127.0.0.1:8000"]);
const MENU_ID_ANALYZE_SELECTION = "truthlens-analyze-selection";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_ID_ANALYZE_SELECTION,
      title: "Analyze with TruthLens",
      contexts: ["selection"]
    });
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID_ANALYZE_SELECTION || !tab?.id) {
    return;
  }

  const selectedText = (info.selectionText || "").trim();
  if (!selectedText) {
    await safeSendToTab(tab, {
      type: "SHOW_SELECTION_RESULT",
      payload: {
        prediction: "FALSE",
        confidence: 0,
        explanation: "No selected text found.",
        sources: []
      }
    });
    return;
  }

  try {
    const raw = await callProcess(selectedText);
    const prediction = normalizeProcessResponse(raw);
    await safeSendToTab(tab, {
      type: "SHOW_SELECTION_RESULT",
      payload: prediction
    });
  } catch (error) {
    await safeSendToTab(tab, {
      type: "SHOW_SELECTION_RESULT",
      payload: {
        prediction: "FALSE",
        confidence: 0,
        explanation: `TruthLens backend error: ${error.message}`,
        sources: []
      }
    });
  }
});

async function safeSendToTab(tab, message) {
  if (!tab?.id) {
    return;
  }
  try {
    await chrome.tabs.sendMessage(tab.id, message);
  } catch (error) {
    const messageText = String(error?.message || "");
    const missingReceiver =
      messageText.includes("Receiving end does not exist") ||
      messageText.includes("Could not establish connection");

    if (!missingReceiver || !canInjectIntoTab(tab)) {
      return;
    }

    await chrome.scripting.insertCSS({
      target: { tabId: tab.id },
      files: ["styles.css"]
    });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"]
    });
    await chrome.tabs.sendMessage(tab.id, message);
  }
}

function canInjectIntoTab(tab) {
  const url = String(tab?.url || "");
  return url.startsWith("http://") || url.startsWith("https://");
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message?.type) {
    return;
  }

  if (message.type === "LIVE_TRANSCRIPT_CLEAR") {
    (async () => {
      try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab?.id || !tab?.url) {
          sendResponse({ ok: false, error: "No active tab." });
          return;
        }

        const storageKey = `tlx_live_${tab.url}`;
        try {
          await chrome.storage.local.set({ [storageKey]: "" });
        } catch (_) {}

        // Best-effort: also clear the in-page live state immediately.
        try {
          await chrome.tabs.sendMessage(tab.id, { type: "SET_LIVE_BASELINE", payload: { text: "" } });
        } catch (_) {}

        sendResponse({ ok: true });
      } catch (error) {
        sendResponse({ ok: false, error: String(error?.message || error) });
      }
    })();
    return true;
  }

  if (message.type === "EVALUATE_TRANSCRIPT") {
    const STORAGE_EVALUATING = "truthlens_live_evaluating";
    const STORAGE_RESULT = "truthlens_live_evaluate_result";
    function safeSendResponse(obj) {
      try {
        sendResponse(obj);
      } catch (_) {
        // Popup closed; result is already in storage for when user reopens
      }
    }
    (async () => {
      try {
        const text = String(message?.payload?.text || "").trim();
        if (!text) {
          safeSendResponse({ ok: false, error: "Transcript is empty." });
          return;
        }
        await chrome.storage.local.set({
          [STORAGE_EVALUATING]: true,
          truthlens_live_evaluate_started_at: Date.now()
        });
        try {
          const result = await callProcessStreamWithRetry(
            text,
            (line) => {
              try {
                chrome.runtime.sendMessage({ type: "LIVE_EVALUATE_LOG_LINE", payload: { line } });
              } catch (_) {}
            },
            (attempt, maxAttempts) => {
              try {
                chrome.runtime.sendMessage({
                  type: "LIVE_EVALUATE_LOG_LINE",
                  payload: { line: `[TruthLens] Stream ended without result. Retrying (${attempt}/${maxAttempts - 1})…` }
                });
              } catch (_) {}
            }
          );
          await chrome.storage.local.set({ [STORAGE_RESULT]: result });
          safeSendResponse({ ok: true, result });
        } finally {
          await chrome.storage.local.set({ [STORAGE_EVALUATING]: false });
        }
      } catch (error) {
        await chrome.storage.local.set({ [STORAGE_EVALUATING]: false });
        safeSendResponse({ ok: false, error: String(error?.message || error) });
      }
    })();
    return true;
  }

  if (message.type === "PREDICT_TEXT") {
    (async () => {
      try {
        const result = await callPredict(message.text || "");
        sendResponse({ ok: true, result });
      } catch (error) {
        sendResponse({ ok: false, error: error.message });
      }
    })();
    return true;
  }

  if (message.type === "ANALYZE_TEXT") {
    (async () => {
      try {
        const raw = await callProcess(message.text || "");
        const result = normalizeProcessResponse(raw);
        sendResponse({ ok: true, result });
      } catch (error) {
        sendResponse({ ok: false, error: error.message });
      }
    })();
    return true;
  }

  if (message.type === "ANALYZE_BATCH") {
    (async () => {
      try {
        const items = Array.isArray(message.items) ? message.items : [];
        const tabId = sender.tab?.id;
        const results = await analyzeBatch(items, tabId);
        try {
          sendResponse({ ok: true, results });
        } catch (_) {
          /* Channel may be closed if popup/tab closed; ignore */
        }
      } catch (error) {
        try {
          sendResponse({ ok: false, error: error.message });
        } catch (_) {}
      }
    })();
    return true;
  }
});

async function analyzeBatch(items, tabId) {
  const output = [];
  const concurrency = 4;
  let cursor = 0;

  function pushItemResult(payload) {
    if (tabId) {
      try {
        chrome.tabs.sendMessage(tabId, { type: "ANALYZE_PAGE_ITEM_RESULT", payload });
      } catch (_) {}
    }
  }

  async function worker() {
    while (cursor < items.length) {
      const index = cursor++;
      const item = items[index];
      const id = item?.id;
      const text = String(item?.text || "").trim();

      if (!id || !text) {
        const entry = { id, ok: false, error: "Invalid analysis item." };
        output[index] = entry;
        pushItemResult(entry);
        continue;
      }

      try {
        const raw = await callProcess(text);
        const result = normalizeProcessResponse(raw);
        const entry = { id, ok: true, result };
        output[index] = entry;
        pushItemResult(entry);
      } catch (error) {
        const entry = { id, ok: false, error: error.message };
        output[index] = entry;
        pushItemResult(entry);
      }
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, items.length) }, () => worker());
  await Promise.all(workers);
  return output;
}

async function getApiBase() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ backendUrl: DEFAULT_API_BASE }, (items) => {
      const raw = String(items.backendUrl || DEFAULT_API_BASE).replace(/\/$/, "");
      resolve(LEGACY_API_BASES.has(raw) ? DEFAULT_API_BASE : raw);
    });
  });
}

async function callPredict(text) {
  const cleanText = String(text || "").trim();
  if (!cleanText) {
    throw new Error("Text cannot be empty.");
  }

  const base = await getApiBase();
  const response = await fetch(`${base}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: cleanText })
  });

  if (!response.ok) {
    const errMsg = await formatBackendError(response);
    throw new Error(errMsg);
  }

  const data = await response.json();
  return normalizePredictResponse(data);
}

/** Ollama + web search for Analyze Selected Text. Slower, richer reasoning. */
async function callAnalyze(text) {
  const cleanText = String(text || "").trim();
  if (!cleanText) {
    throw new Error("Text cannot be empty.");
  }

  const base = await getApiBase();
  const response = await fetch(`${base}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: cleanText })
  });

  if (!response.ok) {
    const errMsg = await formatBackendError(response);
    throw new Error(errMsg);
  }

  const data = await response.json();
  return normalizePredictResponse(data);
}

async function callProcess(userClaim) {
  const cleanText = String(userClaim || "").trim();
  if (!cleanText) {
    throw new Error("text cannot be empty.");
  }

  const base = await getApiBase();
  const response = await fetch(`${base}/api/pipeline/process-text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: cleanText })
  });

  if (!response.ok) {
    const details = await safeReadText(response);
    throw new Error(`Backend ${response.status}: ${details || "Unknown error"}`);
  }

  return await response.json();
}

/** Map /api/process response to UI shape: { prediction, confidence, explanation, sources }. */
function normalizeProcessResponse(data) {
  const roberta = data?.roberta || {};
  const label = roberta?.label || {};
  const className = String(label?.class_name || "mixed").toLowerCase();
  const rawPrediction =
    className === "true" ? "REAL"
    : className === "false" ? "FAKE"
    : "MIXED";
  const prediction = rawPrediction;

  let confidence = Number(roberta?.confidence);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    confidence = 0;
  }

  const summary = data?.summary || {};
  const explanation =
    (summary.conclusion || summary.intro || "").trim()
    || (data?.backend_log ? (data.backend_log.slice(0, 500) + (data.backend_log.length > 500 ? "…" : "")) : "")
    || "Processed via full pipeline.";

  const citations = summary.citations || [];
  const sources = Array.isArray(citations)
    ? citations.map((c) => ({
        title: c?.title || c?.source || "",
        url: c?.url || "",
        snippet: c?.snippet || ""
      }))
    : [];

  const intro = (summary.intro || "").trim();
  const body1 = (summary.body1 || "").trim();
  const body2 = (summary.body2 || "").trim();
  const conclusion = (summary.conclusion || "").trim();

  return { prediction, confidence, explanation, sources, intro, body1, body2, conclusion };
}

const STREAM_NO_RESULT_MSG = "Stream ended without result.";
const EVALUATE_MAX_ATTEMPTS = 3;

/**
 * Call streaming /api/process-stream with retry. On "Stream ended without result."
 * retries up to 2 times (3 attempts total); then throws.
 * onLogLine(line) is called for each log line; onRetry(attempt, maxAttempts) is called before a retry.
 */
async function callProcessStreamWithRetry(userClaim, onLogLine, onRetry) {
  let lastError = null;
  for (let attempt = 1; attempt <= EVALUATE_MAX_ATTEMPTS; attempt++) {
    try {
      return await callProcessStream(userClaim, onLogLine);
    } catch (e) {
      lastError = e;
      const msg = String(e?.message || e);
      const isNoResult = msg.includes("Stream ended without result");
      if (attempt < EVALUATE_MAX_ATTEMPTS && isNoResult) {
        if (typeof onRetry === "function") {
          try {
            onRetry(attempt, EVALUATE_MAX_ATTEMPTS);
          } catch (_) {}
        }
        continue;
      }
      throw e;
    }
  }
  throw lastError || new Error(STREAM_NO_RESULT_MSG);
}

/**
 * Call streaming /api/process-stream; onLogLine(line) is called for each log line in real time.
 * Returns the final result object or throws.
 */
async function callProcessStream(userClaim, onLogLine) {
  const cleanText = String(userClaim || "").trim();
  if (!cleanText) {
    throw new Error("user_claim cannot be empty.");
  }

  const base = await getApiBase();
  const response = await fetch(`${base}/api/process-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_claim: cleanText })
  });

  if (!response.ok) {
    const details = await safeReadText(response);
    throw new Error(`Backend ${response.status}: ${details || "Unknown error"}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;
  let errorMessage = null;

  function processMessage(msg) {
    const lines = msg.split("\n");
    let eventType = null;
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5));
      }
    }
    const dataStr = dataLines.join("\n").trim();
    if (!dataStr) return;
    try {
      const payload = JSON.parse(dataStr);
      if (payload.log !== undefined && typeof onLogLine === "function") {
        onLogLine(payload.log);
      }
      if (eventType === "result") {
        result = payload;
      }
      if (eventType === "error" && payload.error) {
        errorMessage = payload.error;
      }
    } catch (_) {
      if (eventType === "result") {
        result = dataStr;
      }
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const message = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      processMessage(message);
    }
  }
  if (buffer.trim()) {
    processMessage(buffer);
  }

  if (errorMessage) {
    throw new Error(errorMessage);
  }
  if (result === null) {
    throw new Error(STREAM_NO_RESULT_MSG);
  }
  return result;
}

function normalizePredictResponse(data) {
  const rawPrediction = String(data?.prediction || data?.label || "").toUpperCase();
  const prediction = rawPrediction === "TRUE" || rawPrediction === "MIXED" || rawPrediction === "FALSE"
    ? rawPrediction
    : isFakePrediction(rawPrediction) ? "FAKE" : "REAL";

  const confidenceRaw = Number(data?.confidence);
  let confidence = Number.isFinite(confidenceRaw) ? confidenceRaw : 0;
  if (confidence > 1) {
    confidence = confidence / 100;
  }
  confidence = Math.max(0, Math.min(1, confidence));

  const explanation = String(data?.explanation || "No explanation provided by backend.");

  const sources = Array.isArray(data?.sources) ? data.sources : [];

  return {
    prediction,
    confidence,
    explanation,
    sources
  };
}

function isFakePrediction(rawPrediction) {
  return (
    rawPrediction === "FAKE" ||
    rawPrediction === "FALSE" ||
    rawPrediction === "MISINFORMATION" ||
    rawPrediction === "NOT_REAL"
  );
}

async function safeReadText(response) {
  try {
    return await response.text();
  } catch (_) {
    return "";
  }
}

async function formatBackendError(response) {
  const raw = await safeReadText(response);
  try {
    const json = JSON.parse(raw);
    const detail = json?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.join(". ");
  } catch (_) {}
  return raw ? `Backend ${response.status}: ${raw}` : `Backend ${response.status}: Unknown error`;
}
