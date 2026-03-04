const DEFAULT_API_BASE = "http://localhost:8000";
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
        prediction: "FAKE",
        confidence: 0,
        explanation: "No selected text found."
      }
    });
    return;
  }

  try {
    const prediction = await callAnalyze(selectedText);
    await safeSendToTab(tab, {
      type: "SHOW_SELECTION_RESULT",
      payload: prediction
    });
  } catch (error) {
    await safeSendToTab(tab, {
      type: "SHOW_SELECTION_RESULT",
      payload: {
        prediction: "FAKE",
        confidence: 0,
        explanation: `TruthLens backend error: ${error.message}`
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

  if (message.type === "PREDICT_TEXT") {
    (async () => {
      try {
        const result = await callAnalyze(message.text || "");
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
        const results = await analyzeBatch(items);
        sendResponse({ ok: true, results });
      } catch (error) {
        sendResponse({ ok: false, error: error.message });
      }
    })();
    return true;
  }
});

async function analyzeBatch(items) {
  const output = [];
  const concurrency = 4;
  let cursor = 0;

  async function worker() {
    while (cursor < items.length) {
      const index = cursor++;
      const item = items[index];
      const id = item?.id;
      const text = String(item?.text || "").trim();

      if (!id || !text) {
        output[index] = {
          id,
          ok: false,
          error: "Invalid analysis item."
        };
        continue;
      }

      try {
        const result = await callPredict(text);
        output[index] = {
          id,
          ok: true,
          result
        };
      } catch (error) {
        output[index] = {
          id,
          ok: false,
          error: error.message
        };
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
      resolve(items.backendUrl.replace(/\/$/, ""));
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
    const details = await safeReadText(response);
    throw new Error(`Backend ${response.status}: ${details || "Unknown error"}`);
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
    const details = await safeReadText(response);
    throw new Error(`Backend ${response.status}: ${details || "Unknown error"}`);
  }

  const data = await response.json();
  return normalizePredictResponse(data);
}

function normalizePredictResponse(data) {
  const rawPrediction = String(data?.prediction || data?.label || "").toUpperCase();
  const prediction = isFakePrediction(rawPrediction) ? "FAKE" : "REAL";

  const confidenceRaw = Number(data?.confidence);
  let confidence = Number.isFinite(confidenceRaw) ? confidenceRaw : 0;
  if (confidence > 1) {
    confidence = confidence / 100;
  }
  confidence = Math.max(0, Math.min(1, confidence));

  const explanation = String(data?.explanation || "No explanation provided by backend.");

  return {
    prediction,
    confidence,
    explanation
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
