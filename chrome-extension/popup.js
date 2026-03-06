const MAX_DOCUMENT_PARAGRAPHS = 80;
const MIN_PARAGRAPH_LEN = 25;

const analyzeSelectionBtn = document.getElementById("analyzeSelectionBtn");
const selectionResult = document.getElementById("selectionResult");
const analyzePageBtn = document.getElementById("analyzePageBtn");
const toggleOverlayBtn = document.getElementById("toggleOverlayBtn");
const pageStatus = document.getElementById("pageStatus");

const fileInput = document.getElementById("fileInput");
const manualTextInput = document.getElementById("manualTextInput");
const analyzeDocumentBtn = document.getElementById("analyzeDocumentBtn");
const documentStatus = document.getElementById("documentStatus");
const documentResults = document.getElementById("documentResults");

const startAudioBtn = document.getElementById("startAudioBtn");
const stopAudioBtn = document.getElementById("stopAudioBtn");
const audioStatus = document.getElementById("audioStatus");
const transcriptionResult = document.getElementById("transcriptionResult");

const liveTranscriptText = document.getElementById("liveTranscriptText");
const liveStartPauseBtn = document.getElementById("liveStartPauseBtn");
const liveStopBtn = document.getElementById("liveStopBtn");
const liveEvaluateBtn = document.getElementById("liveEvaluateBtn");
const liveTranscriptStatus = document.getElementById("liveTranscriptStatus");
const liveLogPanel = document.getElementById("liveLogPanel");
const liveLogHeaderLeft = document.getElementById("liveLogHeaderLeft");
const liveLogHeaderRight = document.getElementById("liveLogHeaderRight");
const liveLogContent = document.getElementById("liveLogContent");
const liveEssayContainer = document.getElementById("liveEssayContainer");
const liveEssayContent = document.getElementById("liveEssayContent");
const liveEssayToggleBtn = document.getElementById("liveEssayToggleBtn");

const autoDetectCaptureBtn = document.getElementById("autoDetectCaptureBtn");
const startImageCaptureBtn = document.getElementById("startImageCaptureBtn");
const stopImageCaptureBtn = document.getElementById("stopImageCaptureBtn");
const imageStatus = document.getElementById("imageStatus");
const imageAnalysisResult = document.getElementById("imageAnalysisResult");

let overlaysEnabled = false;
let audioCapturing = false;
let imageCapturing = false;
let liveCapturing = false;
let livePaused = false; // legacy; kept for minimal diff

// Live transcript word-by-word rendering (UI only).
let liveWordTimerId = null;
let liveWordQueue = [];
let liveLastRenderedWordKey = "";
const LIVE_WORD_INTERVAL_MS = 55;
const LIVE_MAX_CHARS = 500;
const STORAGE_KEY_LIVE_EVALUATE_RESULT = "truthlens_live_evaluate_result";
const STORAGE_KEY_LIVE_EVALUATING = "truthlens_live_evaluating";
const STORAGE_KEY_EVALUATE_STARTED_AT = "truthlens_live_evaluate_started_at";
const LIVE_EVALUATE_POLL_MS = 800;
const LIVE_EVALUATE_POLL_TIMEOUT_MS = 45000; // 45 s: if still "evaluating", assume worker was suspended
const LIVE_EVALUATE_STALE_MS = 90000; // 90 s: if "evaluating" started this long ago, treat as stale on open
let liveEvaluatePollId = null;
let liveEvaluatePollStartedAt = 0;
/** True while we are showing "Evaluating..." and waiting for storage (so syncLiveEvaluateAvailability keeps button disabled) */
let liveEvaluateUiLock = false;
let liveLogExpanded = false;

initPopup();

function normalizeWordKey(word) {
  return String(word || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function getLastWordKeyFromText(text) {
  const words = String(text || "").trim().split(/\s+/g).filter(Boolean);
  if (words.length === 0) return "";
  return normalizeWordKey(words[words.length - 1]);
}

function stopLiveWordAnimation() {
  if (liveWordTimerId) {
    clearInterval(liveWordTimerId);
    liveWordTimerId = null;
  }
  liveWordQueue = [];
}

function appendLiveWord(word) {
  if (!liveTranscriptText) return;
  const raw = String(word || "").trim();
  if (!raw) return;

  // Word-level de-dupe guard (handles occasional repeated delta boundaries).
  const key = normalizeWordKey(raw);
  if (key && key === liveLastRenderedWordKey) return;

  liveTranscriptText.classList.remove("tlx-hidden");
  if (liveTranscriptText.textContent && !/\s$/.test(liveTranscriptText.textContent)) {
    liveTranscriptText.textContent += " ";
  }
  liveTranscriptText.textContent += raw;
  // Enforce max length by trimming from the start (oldest characters).
  while (liveTranscriptText.textContent.length > LIVE_MAX_CHARS) {
    liveTranscriptText.textContent = liveTranscriptText.textContent.slice(1);
  }
  liveTranscriptText.scrollTop = liveTranscriptText.scrollHeight;
  if (key) liveLastRenderedWordKey = key;
}

function kickLiveWordAnimation() {
  if (liveWordTimerId) return;
  liveWordTimerId = setInterval(() => {
    if (!liveWordQueue.length) {
      stopLiveWordAnimation();
      return;
    }
    const next = liveWordQueue.shift();
    appendLiveWord(next);
  }, LIVE_WORD_INTERVAL_MS);
}

function enqueueLiveTranscriptWords(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return;
  const words = trimmed.split(/\s+/g).filter(Boolean);
  if (!words.length) return;
  liveWordQueue.push(...words);
  kickLiveWordAnimation();
}

function syncLiveTranscriptEditability() {
  if (!liveTranscriptText) return;
  // Editable only when capture is stopped.
  const editable = !liveCapturing;
  liveTranscriptText.setAttribute("contenteditable", editable ? "true" : "false");
  liveTranscriptText.setAttribute("role", "textbox");
  liveTranscriptText.setAttribute("aria-multiline", "true");
  liveTranscriptText.style.outline = editable ? "1px solid rgba(148, 163, 184, 0.6)" : "";
}

function syncLiveClearAvailability() {
  if (!liveStopBtn) return;
  // Clear should only be available when NOT recording.
  const canClear = !liveCapturing;
  liveStopBtn.disabled = !canClear;
  liveStopBtn.classList.toggle("tlx-hidden", !canClear);
}

function syncLiveEvaluateAvailability() {
  if (!liveEvaluateBtn) return;
  const hasText = Boolean(liveTranscriptText && String(liveTranscriptText.textContent || "").trim());
  liveEvaluateBtn.disabled = liveCapturing || !hasText || liveEvaluateUiLock;
  liveEvaluateBtn.classList.toggle("tlx-hidden", liveCapturing);
  if (liveEvaluateUiLock) {
    liveEvaluateBtn.textContent = "Evaluating...";
  } else {
    liveEvaluateBtn.textContent = "Evaluate";
  }
}

async function initPopup() {
  analyzeSelectionBtn.addEventListener("click", onAnalyzeSelection);
  analyzePageBtn.addEventListener("click", onAnalyzePage);
  toggleOverlayBtn.addEventListener("click", onToggleOverlays);
  analyzeDocumentBtn.addEventListener("click", onAnalyzeDocument);
  startAudioBtn.addEventListener("click", onStartAudioCapture);
  stopAudioBtn.addEventListener("click", onStopAudioCapture);
  if (liveStartPauseBtn) liveStartPauseBtn.addEventListener("click", onLiveStartPause);
  if (liveStopBtn) liveStopBtn.addEventListener("click", onLiveStop);
  if (liveEvaluateBtn) liveEvaluateBtn.addEventListener("click", onLiveEvaluate);
  if (liveEssayToggleBtn) liveEssayToggleBtn.addEventListener("click", onLiveEssayToggle);
  if (liveLogPanel) {
    liveLogPanel.addEventListener("click", (e) => { e.preventDefault(); toggleLiveLogExpanded(); });
    liveLogPanel.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleLiveLogExpanded(); } });
  }
  if (liveTranscriptText) liveTranscriptText.addEventListener("input", onLiveTranscriptEdited);
  autoDetectCaptureBtn.addEventListener("click", onAutoDetectCapture);
  startImageCaptureBtn.addEventListener("click", onStartImageCapture);
  stopImageCaptureBtn.addEventListener("click", onStopImageCapture);

  // Listen for transcription results from content script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "TRANSCRIPTION_COMPLETE") {
      renderTranscriptionResult(message.payload);
    } else if (message.type === "TRANSCRIPTION_ERROR") {
      renderTranscriptionError(message.payload?.error || "Unknown error");
    } else if (message.type === "TRANSCRIPT_CHUNK") {
      appendLiveTranscriptChunk(message.payload?.text);
    } else if (message.type === "LIVE_TRANSCRIPT_ERROR") {
      const errMsg = message.payload?.error || "Chunk error";
      setLiveTranscriptStatus(errMsg, true);
      if (liveTranscriptText) {
        const prefix = liveTranscriptText.textContent.trim() ? "\n\n" : "";
        liveTranscriptText.textContent += prefix + "[Transcription failed] " + errMsg;
        liveTranscriptText.scrollTop = liveTranscriptText.scrollHeight;
      }
    } else if (message.type === "LIVE_TRANSCRIPT_NO_AUDIO") {
      const hint = message.payload?.hint || "No audio in chunk. Is the video playing and unmuted?";
      setLiveTranscriptStatus(hint, false);
      if (liveTranscriptText && !liveTranscriptText.textContent.trim()) {
        liveTranscriptText.textContent = "[Waiting for audio] " + hint;
        liveTranscriptText.scrollTop = 0;
      }
    } else if (message.type === "LIVE_EVALUATE_LOG_LINE") {
      appendLiveEvaluateLogLine(message.payload?.line);
    } else if (message.type === "IMAGE_ANALYSIS_COMPLETE") {
      renderImageAnalysisResult(message.payload);
    } else if (message.type === "IMAGE_ANALYSIS_ERROR") {
      renderImageAnalysisError(message.payload?.error || "Unknown error");
    }
  });

  const tab = await getActiveTab();
  if (!tab?.id) {
    pageStatus.textContent = "No active tab available.";
    return;
  }

  const status = await sendToContentScript(tab, { type: "GET_OVERLAY_STATUS" }, { injectIfNeeded: true }).catch(() => null);
  overlaysEnabled = Boolean(status?.enabled);
  syncOverlayButtonText();

  // Sync live transcript button with actual capture state on the content side
  try {
    const liveStatus = await sendToContentScript(tab, { type: "GET_LIVE_STATUS" }, { injectIfNeeded: true }).catch(
      () => null
    );
    const isLiveActive = Boolean(liveStatus?.active);
    liveCapturing = isLiveActive;
    if (isLiveActive) {
      liveStartPauseBtn.textContent = "Stop";
      setLiveTranscriptStatus("🔴 Capturing — transcript updates every few seconds.");
    } else {
      liveStartPauseBtn.textContent = "Start capture";
      setLiveTranscriptStatus("");
    }

    // Restore existing transcript text (including while capturing), so popup
    // collapse / tab switches don't lose what was captured so far.
    const liveTranscriptState = await sendToContentScript(
      tab,
      { type: "GET_LIVE_TRANSCRIPT" },
      { injectIfNeeded: true }
    ).catch(() => null);
    const existingText = String(liveTranscriptState?.text || "").trim();
    if (liveTranscriptText) {
      if (existingText) {
        liveTranscriptText.textContent = existingText;
        liveTranscriptText.classList.remove("tlx-hidden");
      } else {
        liveTranscriptText.textContent = "";
        liveTranscriptText.classList.add("tlx-hidden");
      }
    }
    liveLastRenderedWordKey = getLastWordKeyFromText(existingText);
    syncLiveTranscriptEditability();
    syncLiveClearAvailability();
    syncLiveEvaluateAvailability();
  } catch {
    // ignore; live capture not available for this tab
  }

  // If evaluation is in progress (e.g. panel was closed while evaluating), show "Evaluating..." and poll for result
  try {
    const stored = await chrome.storage.local.get([
      STORAGE_KEY_LIVE_EVALUATING,
      STORAGE_KEY_LIVE_EVALUATE_RESULT,
      STORAGE_KEY_EVALUATE_STARTED_AT
    ]);
    let isEvaluating = Boolean(stored[STORAGE_KEY_LIVE_EVALUATING]);
    const startedAt = Number(stored[STORAGE_KEY_EVALUATE_STARTED_AT]) || 0;
    const age = startedAt ? Date.now() - startedAt : 0;
    // Stale: job was "started" long ago; worker was likely suspended, clear and don't show Evaluating
    if (isEvaluating && age > LIVE_EVALUATE_STALE_MS) {
      await chrome.storage.local.set({ [STORAGE_KEY_LIVE_EVALUATING]: false });
      liveEvaluateUiLock = false;
      isEvaluating = false;
    }
    if (isEvaluating && liveEvaluateBtn) {
      liveEvaluateUiLock = true;
      liveEvaluateBtn.disabled = true;
      liveEvaluateBtn.textContent = "Evaluating...";
      setLiveTranscriptStatus("Evaluation in progress…");
      showLiveLogPanel("Evaluating...", "", "", false);
      startLiveEvaluatePolling();
      return;
    }
    const savedResult = stored[STORAGE_KEY_LIVE_EVALUATE_RESULT];
    if (savedResult && liveEssayContainer && liveEssayContent) {
      renderLiveEssay(savedResult);
      updateLiveLogPanelFromResult(savedResult);
    }
    // So after collapse/reload: button reflects enabled (if has transcript and not capturing) and label "Evaluate"
    syncLiveEvaluateAvailability();
  } catch (_) {}
}

function stopLiveEvaluatePolling() {
  if (liveEvaluatePollId) {
    clearInterval(liveEvaluatePollId);
    liveEvaluatePollId = null;
  }
}

function startLiveEvaluatePolling() {
  stopLiveEvaluatePolling();
  liveEvaluatePollStartedAt = Date.now();
  liveEvaluatePollId = setInterval(async () => {
    try {
      const elapsed = Date.now() - liveEvaluatePollStartedAt;
      if (elapsed >= LIVE_EVALUATE_POLL_TIMEOUT_MS) {
        stopLiveEvaluatePolling();
        liveEvaluateUiLock = false;
        chrome.storage.local.set({ [STORAGE_KEY_LIVE_EVALUATING]: false }).catch(() => {});
        if (liveEvaluateBtn) {
          liveEvaluateBtn.textContent = "Evaluate";
          syncLiveEvaluateAvailability();
        }
        showLiveLogPanel("Evaluation was interrupted (panel was closed). Click Evaluate to try again.", "", "", false);
        setLiveTranscriptStatus("Evaluation was interrupted (panel was closed). Click Evaluate to try again.", true);
        return;
      }
      const stored = await chrome.storage.local.get([STORAGE_KEY_LIVE_EVALUATING, STORAGE_KEY_LIVE_EVALUATE_RESULT]);
      if (!stored[STORAGE_KEY_LIVE_EVALUATING]) {
        stopLiveEvaluatePolling();
        liveEvaluateUiLock = false;
        if (liveEvaluateBtn) {
          liveEvaluateBtn.textContent = "Evaluate";
          syncLiveEvaluateAvailability();
        }
        const result = stored[STORAGE_KEY_LIVE_EVALUATE_RESULT];
        if (result && liveEssayContainer && liveEssayContent) {
          renderLiveEssay(result);
          updateLiveLogPanelFromResult(result);
        }
        setLiveTranscriptStatus("");
      }
    } catch (_) {
      stopLiveEvaluatePolling();
      liveEvaluateUiLock = false;
      syncLiveEvaluateAvailability();
    }
  }, LIVE_EVALUATE_POLL_MS);
}

async function onAnalyzeSelection() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    renderSelectionResultError("Could not access active tab.");
    return;
  }

  setBusy(analyzeSelectionBtn, true, "Analyzing...");
  clearSelectionResult();

  try {
    const selectionPayload = await sendToContentScript(tab, { type: "GET_SELECTED_TEXT" }, { injectIfNeeded: true });
    const text = String(selectionPayload?.text || "").trim();
    if (!text) {
      throw new Error("No highlighted text found. Highlight text on the page first.");
    }

    const response = await chrome.runtime.sendMessage({
      type: "PREDICT_TEXT",
      text
    });
    if (!response?.ok) {
      throw new Error(response?.error || "Prediction failed.");
    }

    renderSelectionResult(response.result);
  } catch (error) {
    renderSelectionResultError(error.message || "Unknown error");
  } finally {
    setBusy(analyzeSelectionBtn, false, "Analyze Selected Text");
  }
}

async function onAnalyzePage() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    pageStatus.textContent = "Could not access active tab.";
    return;
  }

  setBusy(analyzePageBtn, true, "Analyzing...");
  pageStatus.textContent = "";

  try {
    const response = await sendToContentScript(tab, { type: "ANALYZE_PAGE" }, { injectIfNeeded: true });
    if (!response?.ok) {
      throw new Error(response?.error || "Failed to analyze page.");
    }
    overlaysEnabled = true;
    syncOverlayButtonText();
    pageStatus.textContent = `Analyzed ${response.summary?.analyzed || 0} text blocks on this page.`;
  } catch (error) {
    pageStatus.textContent = `Error: ${error.message || "Unknown error"}`;
  } finally {
    setBusy(analyzePageBtn, false, "Analyze This Page");
  }
}

async function onToggleOverlays() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    pageStatus.textContent = "Could not access active tab.";
    return;
  }

  try {
    if (overlaysEnabled) {
      await sendToContentScript(tab, { type: "CLEAR_OVERLAYS" }, { injectIfNeeded: true });
      overlaysEnabled = false;
      pageStatus.textContent = "Overlays removed.";
    } else {
      await onAnalyzePage();
      return;
    }
    syncOverlayButtonText();
  } catch (error) {
    pageStatus.textContent = `Error: ${error.message || "Unknown error"}`;
  }
}

async function onAnalyzeDocument() {
  const file = fileInput.files?.[0] || null;
  const manualText = String(manualTextInput.value || "").trim();

  if (!file && !manualText) {
    documentStatus.textContent = "Add a file or paste text first.";
    return;
  }

  setBusy(analyzeDocumentBtn, true, "Analyzing...");
  documentStatus.textContent = "Preparing document...";
  documentResults.innerHTML = "";

  try {
    const parts = [];

    if (file) {
      const fileText = await extractTextFromFile(file);
      if (fileText.trim()) {
        parts.push(fileText.trim());
      }
    }
    if (manualText) {
      parts.push(manualText);
    }

    const fullText = parts.join("\n\n");
    const paragraphs = splitIntoParagraphs(fullText);
    if (paragraphs.length === 0) {
      throw new Error("No valid paragraphs found in the provided content.");
    }

    documentStatus.textContent = `Sending ${paragraphs.length} paragraph(s) to backend...`;

    const items = paragraphs.map((paragraph, index) => ({
      id: `doc-${index}`,
      text: paragraph
    }));

    const response = await chrome.runtime.sendMessage({
      type: "ANALYZE_BATCH",
      items
    });

    if (!response?.ok) {
      throw new Error(response?.error || "Document analysis failed.");
    }

    renderDocumentResults(paragraphs, response.results || []);
    documentStatus.textContent = `Analysis complete. ${paragraphs.length} paragraph(s) processed.`;
  } catch (error) {
    documentStatus.textContent = `Error: ${error.message || "Unknown error"}`;
  } finally {
    setBusy(analyzeDocumentBtn, false, "Analyze Document");
  }
}

function renderSelectionResult(result) {
  selectionResult.classList.remove("tlx-hidden");
  const isFake = String(result?.prediction || "").toUpperCase() === "FAKE";
  const confidencePct = Math.round(Number(result?.confidence || 0) * 100);
  const explanation = escapeHtml(String(result?.explanation || ""));

  selectionResult.innerHTML = `
    <div class="tlx-result-header">
      <span class="tlx-pill ${isFake ? "tlx-pill-fake" : "tlx-pill-real"}">
        ${isFake ? "FAKE" : "REAL"}
      </span>
      <span class="tlx-result-confidence">${confidencePct}%</span>
    </div>
    <p class="tlx-result-explanation">${explanation}</p>
  `;
}

function renderSelectionResultError(message) {
  selectionResult.classList.remove("tlx-hidden");
  selectionResult.innerHTML = `<p class="tlx-result-error">${escapeHtml(message)}</p>`;
}

function clearSelectionResult() {
  selectionResult.innerHTML = "";
  selectionResult.classList.add("tlx-hidden");
}

function renderDocumentResults(paragraphs, batchResults) {
  const byId = new Map(batchResults.map((item) => [item.id, item]));

  const cards = paragraphs.map((paragraph, index) => {
    const id = `doc-${index}`;
    const item = byId.get(id) || batchResults[index];
    const ok = Boolean(item?.ok && item?.result);

    if (!ok) {
      return `
        <article class="tlx-doc-card">
          <p class="tlx-doc-paragraph">${escapeHtml(paragraph)}</p>
          <p class="tlx-result-error">Error: ${escapeHtml(item?.error || "Analysis failed.")}</p>
        </article>
      `;
    }

    const result = item.result;
    const isFake = String(result.prediction).toUpperCase() === "FAKE";
    const confidencePct = Math.round(Number(result.confidence || 0) * 100);

    return `
      <article class="tlx-doc-card">
        <p class="tlx-doc-paragraph">${escapeHtml(paragraph)}</p>
        <div class="tlx-result-header">
          <span class="tlx-pill ${isFake ? "tlx-pill-fake" : "tlx-pill-real"}">${isFake ? "FAKE" : "REAL"}</span>
          <span class="tlx-result-confidence">${confidencePct}%</span>
        </div>
        <p class="tlx-result-explanation">${escapeHtml(String(result.explanation || ""))}</p>
      </article>
    `;
  });

  documentResults.innerHTML = cards.join("\n");
}

async function extractTextFromFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".txt") || name.endsWith(".md")) {
    return await file.text();
  }
  if (name.endsWith(".pdf")) {
    return await extractTextFromPdf(file);
  }
  if (name.endsWith(".docx")) {
    return await extractTextFromDocx(file);
  }
  throw new Error("Unsupported file type. Please upload .txt, .pdf, or .docx.");
}

async function extractTextFromPdf(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const content = new TextDecoder("latin1").decode(bytes);

  const textBlocks = [];
  const textObjectPattern = /BT([\s\S]*?)ET/g;
  let textObjectMatch;

  while ((textObjectMatch = textObjectPattern.exec(content)) !== null) {
    const block = textObjectMatch[1];
    const literalMatches = block.match(/\((?:\\.|[^\\()])*\)\s*Tj/g) || [];
    for (const entry of literalMatches) {
      const raw = entry.replace(/\)\s*Tj$/, "").replace(/^\(/, "");
      textBlocks.push(unescapePdfString(raw));
    }

    const arrayMatches = block.match(/\[(.*?)\]\s*TJ/g) || [];
    for (const arr of arrayMatches) {
      const literals = arr.match(/\((?:\\.|[^\\()])*\)/g) || [];
      const joined = literals
        .map((str) => unescapePdfString(str.slice(1, -1)))
        .join(" ");
      if (joined.trim()) {
        textBlocks.push(joined);
      }
    }
  }

  const output = normalizeText(textBlocks.join("\n"));
  if (!output) {
    throw new Error("Could not extract text from PDF. Try a text-based PDF file.");
  }
  return output;
}

async function extractTextFromDocx(file) {
  const buffer = await file.arrayBuffer();
  const documentXml = await extractZipEntry(buffer, "word/document.xml");
  if (!documentXml) {
    throw new Error("Could not read DOCX content.");
  }

  const xmlText = new TextDecoder("utf-8").decode(documentXml);
  const parser = new DOMParser();
  const xml = parser.parseFromString(xmlText, "application/xml");
  const paragraphs = Array.from(xml.getElementsByTagName("w:p"))
    .map((paragraph) =>
      Array.from(paragraph.getElementsByTagName("w:t"))
        .map((node) => node.textContent || "")
        .join("")
    )
    .map((entry) => normalizeText(entry))
    .filter(Boolean);

  const output = paragraphs.join("\n\n");
  if (!output) {
    throw new Error("Could not extract text from DOCX.");
  }
  return output;
}

function unescapePdfString(value) {
  return value
    .replace(/\\\(/g, "(")
    .replace(/\\\)/g, ")")
    .replace(/\\\\/g, "\\")
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "\r")
    .replace(/\\t/g, "\t")
    .replace(/\\[0-7]{1,3}/g, "");
}

function splitIntoParagraphs(text) {
  const normalized = normalizeText(String(text || ""));
  return normalized
    .split(/\n{2,}|(?<=[.?!])\s+(?=[A-Z])/g)
    .map((part) => normalizeText(part))
    .filter((part) => part.length >= MIN_PARAGRAPH_LEN)
    .slice(0, MAX_DOCUMENT_PARAGRAPHS);
}

function normalizeText(value) {
  return String(value || "")
    .replace(/\r/g, "\n")
    .replace(/\u0000/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function setBusy(button, busy, busyText) {
  button.disabled = busy;
  if (busy) {
    button.dataset.originalLabel = button.textContent;
    button.textContent = busyText;
  } else if (button.dataset.originalLabel) {
    button.textContent = button.dataset.originalLabel;
  }
}

function syncOverlayButtonText() {
  toggleOverlayBtn.textContent = overlaysEnabled ? "Remove Overlays" : "Show Overlays";
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

function canInjectIntoTab(tab) {
  if (!tab?.id || !tab?.url) {
    return false;
  }
  return tab.url.startsWith("http://") || tab.url.startsWith("https://");
}

async function ensureContentScriptInjected(tab) {
  if (!canInjectIntoTab(tab)) {
    throw new Error("TruthLens only runs on regular http/https pages.");
  }

  await chrome.scripting.insertCSS({
    target: { tabId: tab.id },
    files: ["styles.css"]
  });
  // Inject all content scripts (image/audio capture are needed for video analysis)
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ["content.js", "audio-capture.js", "image-capture.js"]
  });
}

async function sendToContentScript(tab, message, { injectIfNeeded = false } = {}) {
  try {
    return await chrome.tabs.sendMessage(tab.id, message);
  } catch (error) {
    const messageText = String(error?.message || "");
    const missingReceiver =
      messageText.includes("Receiving end does not exist") ||
      messageText.includes("Could not establish connection");

    if (!injectIfNeeded || !missingReceiver) {
      throw error;
    }

    await ensureContentScriptInjected(tab);
    return await chrome.tabs.sendMessage(tab.id, message);
  }
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function extractZipEntry(zipBuffer, targetPath) {
  const bytes = new Uint8Array(zipBuffer);
  const view = new DataView(zipBuffer);
  const decoder = new TextDecoder("utf-8");

  const eocdOffset = findEndOfCentralDirectory(bytes);
  if (eocdOffset < 0) {
    return null;
  }

  const centralDirOffset = view.getUint32(eocdOffset + 16, true);
  const totalEntries = view.getUint16(eocdOffset + 10, true);

  let pointer = centralDirOffset;
  for (let i = 0; i < totalEntries; i += 1) {
    const signature = view.getUint32(pointer, true);
    if (signature !== 0x02014b50) {
      break;
    }

    const compression = view.getUint16(pointer + 10, true);
    const compressedSize = view.getUint32(pointer + 20, true);
    const fileNameLength = view.getUint16(pointer + 28, true);
    const extraLength = view.getUint16(pointer + 30, true);
    const commentLength = view.getUint16(pointer + 32, true);
    const localHeaderOffset = view.getUint32(pointer + 42, true);

    const nameStart = pointer + 46;
    const nameBytes = bytes.slice(nameStart, nameStart + fileNameLength);
    const fileName = decoder.decode(nameBytes);

    if (fileName === targetPath) {
      const localSignature = view.getUint32(localHeaderOffset, true);
      if (localSignature !== 0x04034b50) {
        return null;
      }

      const localNameLength = view.getUint16(localHeaderOffset + 26, true);
      const localExtraLength = view.getUint16(localHeaderOffset + 28, true);
      const dataStart = localHeaderOffset + 30 + localNameLength + localExtraLength;
      const dataEnd = dataStart + compressedSize;
      const compressed = bytes.slice(dataStart, dataEnd);

      if (compression === 0) {
        return compressed;
      }
      if (compression === 8) {
        if (typeof DecompressionStream === "undefined") {
          throw new Error("DOCX decompression is not supported in this browser.");
        }
        const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
        const decompressed = await new Response(stream).arrayBuffer();
        return new Uint8Array(decompressed);
      }
      throw new Error(`Unsupported DOCX compression method: ${compression}`);
    }

    pointer += 46 + fileNameLength + extraLength + commentLength;
  }

  return null;
}

// ============================================
// Audio Capture & Transcription Functions
// ============================================

async function onStartAudioCapture() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    renderTranscriptionError("Could not access active tab.");
    return;
  }

  audioStatus.textContent = "Starting audio capture...";
  setBusy(startAudioBtn, true, "Starting...");

  try {
    const response = await sendToContentScript(
      tab,
      { type: "START_AUDIO_CAPTURE" },
      { injectIfNeeded: true }
    );

    if (!response?.ok) {
      throw new Error(response?.error || "Failed to start audio capture");
    }

    audioCapturing = true;
    audioStatus.textContent = "🔴 Recording audio from video...";
    startAudioBtn.classList.add("tlx-hidden");
    stopAudioBtn.classList.remove("tlx-hidden");
    clearTranscriptionResult();

  } catch (error) {
    audioStatus.textContent = `Error: ${error.message}`;
    setBusy(startAudioBtn, false, "Start Audio Capture");
  }
}

async function onStopAudioCapture() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    renderTranscriptionError("Could not access active tab.");
    return;
  }

  audioStatus.textContent = "Stopping audio capture and sending for transcription...";
  setBusy(stopAudioBtn, true, "Processing...");

  try {
    const response = await sendToContentScript(
      tab,
      { type: "STOP_AUDIO_CAPTURE" },
      { injectIfNeeded: true }
    );

    if (!response?.ok) {
      throw new Error(response?.error || "Failed to stop audio capture");
    }

    audioCapturing = false;
    audioStatus.textContent = "Transcribing audio... This may take a moment.";

  } catch (error) {
    audioStatus.textContent = `Error: ${error.message}`;
    audioCapturing = false;
    stopAudioBtn.classList.add("tlx-hidden");
    startAudioBtn.classList.remove("tlx-hidden");
    setBusy(stopAudioBtn, false, "Stop & Transcribe");
  }
}

function renderTranscriptionResult(result) {
  stopAudioBtn.classList.add("tlx-hidden");
  startAudioBtn.classList.remove("tlx-hidden");
  setBusy(stopAudioBtn, false, "Stop & Transcribe");

  if (!result?.transcription?.full_text) {
    audioStatus.textContent = "✅ Transcription complete, but no text was detected.";
    return;
  }

  audioStatus.textContent = "✅ Transcription complete!";
  
  const transcribedText = result.transcription.full_text;
  const duration = (result.transcription.duration_seconds || 0).toFixed(1);
  const language = result.transcription.language_detected || "unknown";
  const segmentCount = (result.transcription.segments || []).length;

  const html = `
    <div class="tlx-result-header">
      <strong>Transcribed Text</strong>
      <small>${duration}s • ${language}</small>
    </div>
    <div class="tlx-transcription-text">${escapeHtml(transcribedText)}</div>
    <div class="tlx-result-meta">
      <small>Segments: ${segmentCount} | Duration: ${duration}s</small>
    </div>
  `;

  transcriptionResult.innerHTML = html;
  transcriptionResult.classList.remove("tlx-hidden");
}

function renderTranscriptionError(error) {
  stopAudioBtn.classList.add("tlx-hidden");
  startAudioBtn.classList.remove("tlx-hidden");
  setBusy(stopAudioBtn, false, "Stop & Transcribe");

  audioStatus.textContent = `❌ Error: ${error}`;
  clearTranscriptionResult();
}

function clearTranscriptionResult() {
  transcriptionResult.innerHTML = "";
  transcriptionResult.classList.add("tlx-hidden");
}

// ============================================
// Live transcript panel (real-time capture + transcript)
// ============================================

function appendLiveTranscriptChunk(text) {
  if (!liveTranscriptText || !String(text || "").trim()) return;
  // Render delta "word-by-word" in the popup to reduce perceived overlaps.
  enqueueLiveTranscriptWords(text);
}

function setLiveTranscriptStatus(message, isError = false) {
  if (!liveTranscriptStatus) return;
  liveTranscriptStatus.textContent = message || "";
  liveTranscriptStatus.style.color = isError ? "#b91c1c" : "";
}

function showLiveLogPanel(leftText, rightText, logText, expanded) {
  if (!liveLogPanel || !liveLogHeaderLeft || !liveLogHeaderRight || !liveLogContent) return;
  liveLogPanel.classList.remove("tlx-hidden");
  liveLogHeaderLeft.textContent = leftText || "";
  liveLogHeaderRight.textContent = rightText || "";
  liveLogContent.textContent = logText || "";
  liveLogExpanded = Boolean(expanded);
  liveLogContent.classList.toggle("tlx-hidden", !liveLogExpanded);
  liveLogPanel.setAttribute("aria-expanded", liveLogExpanded);
}

function hideLiveLogPanel() {
  if (liveLogPanel) liveLogPanel.classList.add("tlx-hidden");
  liveLogExpanded = false;
}

function updateLiveLogPanelFromResult(data) {
  if (!data || typeof data !== "object") return;
  const roberta = data.roberta || {};
  const label = roberta.label || {};
  const className = String(label.class_name || label.label || "").trim();
  const conf = Number(roberta.confidence);
  const confPct = Number.isFinite(conf) ? `${Math.round(conf * 100)}%` : "";
  const src = roberta.inference_source ? ` ${roberta.inference_source}` : "";
  const right = className ? `${className} (${confPct})${src}` : "";
  const logText = typeof data.backend_log === "string" ? data.backend_log : "";
  showLiveLogPanel("Evaluation complete", right, logText, false);
}

function toggleLiveLogExpanded() {
  if (!liveLogPanel || !liveLogContent) return;
  liveLogExpanded = !liveLogExpanded;
  liveLogContent.classList.toggle("tlx-hidden", !liveLogExpanded);
  liveLogPanel.setAttribute("aria-expanded", liveLogExpanded);
}

function appendLiveEvaluateLogLine(line) {
  if (!liveLogContent || line == null) return;
  const text = String(line).trim();
  if (!text) return;
  console.log("[TruthLens]", text);
  const hadContent = liveLogContent.textContent.length > 0;
  if (hadContent) {
    liveLogContent.textContent += "\n" + text;
  } else {
    liveLogContent.textContent = text;
  }
  liveLogContent.scrollTop = liveLogContent.scrollHeight;
  if (!hadContent && liveLogPanel && !liveLogExpanded) {
    liveLogExpanded = true;
    liveLogContent.classList.remove("tlx-hidden");
    liveLogPanel.setAttribute("aria-expanded", "true");
  }
}

async function onLiveStartPause() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    setLiveTranscriptStatus("Could not access active tab.", true);
    return;
  }

  try {
    if (!liveCapturing) {
      setLiveTranscriptStatus("Starting capture...");
      liveStartPauseBtn.disabled = true;
      stopLiveWordAnimation();
      liveLastRenderedWordKey = getLastWordKeyFromText(liveTranscriptText ? liveTranscriptText.textContent : "");
      const initialText = liveTranscriptText ? liveTranscriptText.textContent : "";
      const response = await sendToContentScript(
        tab,
        { type: "START_LIVE_TRANSCRIPT", payload: { initialText } },
        { injectIfNeeded: true }
      );
      if (!response?.ok) throw new Error(response?.error || "Failed to start live capture");
      liveCapturing = true;
      livePaused = false;
      liveStartPauseBtn.textContent = "Stop";
      setLiveTranscriptStatus("🔴 Capturing — transcript updates every few seconds.");
      syncLiveTranscriptEditability();
      syncLiveClearAvailability();
      syncLiveEvaluateAvailability();
    } else {
      const response = await sendToContentScript(tab, { type: "STOP_LIVE_TRANSCRIPT" });
      if (!response?.ok) throw new Error(response?.error || "Failed to stop");
      livePaused = false;
      liveCapturing = false;
      liveStartPauseBtn.textContent = "Start capture";
      setLiveTranscriptStatus("Stopped.");
      stopLiveWordAnimation();
      syncLiveTranscriptEditability();
      syncLiveClearAvailability();
      syncLiveEvaluateAvailability();
    }
  } catch (error) {
    setLiveTranscriptStatus(`Error: ${error.message}`, true);
    liveCapturing = false;
    livePaused = false;
    liveStartPauseBtn.textContent = "Start capture";
    stopLiveWordAnimation();
    syncLiveTranscriptEditability();
    syncLiveClearAvailability();
    syncLiveEvaluateAvailability();
  } finally {
    liveStartPauseBtn.disabled = false;
  }
}

async function onLiveTranscriptEdited() {
  // Only treat edits as authoritative baseline when capture is stopped.
  if (liveCapturing) return;
  stopLiveWordAnimation();
  liveLastRenderedWordKey = getLastWordKeyFromText(liveTranscriptText ? liveTranscriptText.textContent : "");
  syncLiveEvaluateAvailability();
  syncLiveEvaluateAvailability();
  const tab = await getActiveTab();
  if (!tab?.id || !liveTranscriptText) return;
  const text = liveTranscriptText.textContent || "";
  try {
    await sendToContentScript(tab, { type: "SET_LIVE_BASELINE", payload: { text } });
  } catch (_) {
    // ignore; not fatal for UI editing
  }
}

async function onLiveStop() {
  const tab = await getActiveTab();
  stopLiveWordAnimation();
  if (liveTranscriptText) {
    liveTranscriptText.textContent = "";
    liveTranscriptText.classList.remove("tlx-hidden");
  }
  liveLastRenderedWordKey = "";
  syncLiveEvaluateAvailability();
  syncLiveEvaluateAvailability();
  // Reset baseline (persisted) even if popup collapses immediately.
  try {
    chrome.runtime.sendMessage({ type: "LIVE_TRANSCRIPT_CLEAR" });
  } catch (_) {}

  // Best-effort direct update to the content script too (for immediate UI consistency).
  if (tab?.id) {
    try {
      await sendToContentScript(tab, { type: "SET_LIVE_BASELINE", payload: { text: "" } });
    } catch (_) {}
  }
  setLiveTranscriptStatus("Cleared.");
}

function clearLiveEvaluateResult() {
  if (liveEssayContainer) liveEssayContainer.classList.add("tlx-hidden");
  if (liveEssayContent) liveEssayContent.innerHTML = "";
  hideLiveLogPanel();
  chrome.storage.local.remove(STORAGE_KEY_LIVE_EVALUATE_RESULT).catch(() => {});
}

async function onLiveEvaluate() {
  if (!liveEvaluateBtn || !liveTranscriptText) return;
  const text = String(liveTranscriptText.textContent || "").trim();
  if (!text) {
    setLiveTranscriptStatus("Nothing to evaluate (transcript is empty).", true);
    syncLiveEvaluateAvailability();
    return;
  }
  if (liveCapturing) {
    setLiveTranscriptStatus("Stop capture before evaluating.", true);
    return;
  }

  // Remove previous result and stored result; show new one when it arrives
  clearLiveEvaluateResult();

  liveEvaluateUiLock = true;
  const prevLabel = liveEvaluateBtn.textContent;
  liveEvaluateBtn.disabled = true;
  liveEvaluateBtn.textContent = "Evaluating...";
  setLiveTranscriptStatus("Evaluating transcript…");
  showLiveLogPanel("Evaluating...", "", "", false);

  try {
    const resp = await chrome.runtime.sendMessage({ type: "EVALUATE_TRANSCRIPT", payload: { text } });
    if (!resp?.ok) {
      throw new Error(resp?.error || "Evaluation failed.");
    }

    const data = resp.result || {};
    renderLiveEssay(data);
    try {
      await chrome.storage.local.set({ [STORAGE_KEY_LIVE_EVALUATE_RESULT]: data });
    } catch (_) {}
    updateLiveLogPanelFromResult(data);
    setLiveTranscriptStatus("");
    console.log("[TruthLens] /api/process result:", data);
  } catch (error) {
    setLiveTranscriptStatus(`Evaluate error: ${error.message || "Unknown error"}`, true);
  } finally {
    liveEvaluateUiLock = false;
    liveEvaluateBtn.textContent = prevLabel;
    syncLiveEvaluateAvailability();
  }
}

function extractEssaySections(result) {
  if (!result || typeof result !== "object") return null;
  const essay =
    result.essay ||
    result.llm_summary ||
    result.summary_essay ||
    null;
  if (essay && typeof essay === "object") {
    const intro = String(essay.intro || "").trim();
    const body1 = String(essay.body1 || "").trim();
    const body2 = String(essay.body2 || "").trim();
    const conclusion = String(essay.conclusion || "").trim();
    if (intro || body1 || body2 || conclusion) {
      return { intro, body1, body2, conclusion };
    }
  }
  // Fallback: build a minimal explanation from roberta label and evidence_topk, if available.
  const roberta = result.roberta || {};
  const label = roberta.label || {};
  const className = String(label.class_name || label.label || "").trim();
  const conf = Number(roberta.confidence);
  const confPct = Number.isFinite(conf) ? `${Math.round(conf * 100)}%` : "";
  const status = className ? `The claim is classified as ${className}${confPct ? ` with confidence ${confPct}.` : "."}` : "";
  const evidenceItems = (result.evidence_topk && Array.isArray(result.evidence_topk.items)) ? result.evidence_topk.items : [];
  const topEvidence = evidenceItems.slice(0, 3).map((item) => {
    const src = item.doc?.source || "";
    const title = item.doc?.title || "";
    return `• ${title}${src ? ` (${src})` : ""}`;
  });
  const intro = status || "";
  const body1 = topEvidence.length ? `Key evidence:\n${topEvidence.join("\n")}` : "";
  return { intro, body1, body2: "", conclusion: "" };
}

function renderLiveEssay(result) {
  if (!liveEssayContainer || !liveEssayContent) return;
  const sections = extractEssaySections(result);
  if (!sections) {
    liveEssayContainer.classList.add("tlx-hidden");
    liveEssayContent.innerHTML = "";
    return;
  }

  const parts = [];
  if (sections.intro) {
    parts.push(
      `<div class="tlx-live-essay-section"><h3>Introduction</h3><p>${escapeHtml(sections.intro)}</p></div>`
    );
  }
  if (sections.body1) {
    parts.push(
      `<div class="tlx-live-essay-section"><h3>Body 1</h3><p>${escapeHtml(sections.body1)}</p></div>`
    );
  }
  if (sections.body2) {
    parts.push(
      `<div class="tlx-live-essay-section"><h3>Body 2</h3><p>${escapeHtml(sections.body2)}</p></div>`
    );
  }
  if (sections.conclusion) {
    parts.push(
      `<div class="tlx-live-essay-section"><h3>Conclusion</h3><p>${escapeHtml(sections.conclusion)}</p></div>`
    );
  }

  liveEssayContent.innerHTML = parts.join("");
  liveEssayContainer.classList.remove("tlx-hidden");
  liveEssayContent.style.display = "block";
  if (liveEssayToggleBtn) {
    liveEssayToggleBtn.textContent = "Collapse";
  }
}

function onLiveEssayToggle() {
  if (!liveEssayContainer || !liveEssayContent || !liveEssayToggleBtn) return;
  const isHidden = liveEssayContent.style.display === "none";
  liveEssayContent.style.display = isHidden ? "block" : "none";
  liveEssayToggleBtn.textContent = isHidden ? "Collapse" : "Expand";
}

async function onAutoDetectCapture() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    renderImageAnalysisError("Could not access active tab.");
    return;
  }

  imageStatus.textContent = "Analyzing video audio levels...";
  setBusy(autoDetectCaptureBtn, true, "Analyzing...");

  try {
    const response = await sendToContentScript(
      tab,
      { type: "SHOULD_USE_IMAGE_CAPTURE" },
      { injectIfNeeded: true }
    );

    if (!response?.ok) {
      throw new Error(response?.error || "Failed to analyze video");
    }

    const { shouldCapture, audioAnalysis, reason } = response.result;
    const msg = reason || audioAnalysis?.reason || "Use audio capture for videos with voice.";

    if (shouldCapture) {
      imageStatus.textContent = `📸 ${msg} - Starting image capture...`;
      await startImageCaptureInternal();
    } else {
      imageStatus.textContent = `🔊 ${msg} - Use audio capture instead.`;
      setBusy(autoDetectCaptureBtn, false, "Auto-Detect & Start");
    }

  } catch (error) {
    renderImageAnalysisError(error.message);
    setBusy(autoDetectCaptureBtn, false, "Auto-Detect & Start");
  }
}

async function onStartImageCapture() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    renderImageAnalysisError("Could not access active tab.");
    return;
  }

  imageStatus.textContent = "Starting image capture (forced)...";
  setBusy(startImageCaptureBtn, true, "Starting...");

  try {
    const response = await sendToContentScript(
      tab,
      { type: "START_IMAGE_CAPTURE", payload: { forceCapture: true } },
      { injectIfNeeded: true }
    );

    if (!response?.ok) {
      throw new Error(response?.error || "Failed to start image capture");
    }

    imageCapturing = true;
    imageStatus.textContent = "📸 Capturing frames and text...";
    startImageCaptureBtn.classList.add("tlx-hidden");
    autoDetectCaptureBtn.classList.add("tlx-hidden");
    stopImageCaptureBtn.classList.remove("tlx-hidden");
    clearImageAnalysisResult();

  } catch (error) {
    renderImageAnalysisError(error.message);
    setBusy(startImageCaptureBtn, false, "Force Image Capture");
  }
}

async function startImageCaptureInternal() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    renderImageAnalysisError("Could not access active tab.");
    return;
  }

  setBusy(autoDetectCaptureBtn, true, "Capturing...");

  try {
    const response = await sendToContentScript(
      tab,
      { type: "START_IMAGE_CAPTURE" },
      { injectIfNeeded: true }
    );

    if (!response?.ok) {
      throw new Error(response?.error || "Failed to start image capture");
    }

    imageCapturing = true;
    imageStatus.textContent = "📸 Capturing frames and text...";
    startImageCaptureBtn.classList.add("tlx-hidden");
    autoDetectCaptureBtn.classList.add("tlx-hidden");
    stopImageCaptureBtn.classList.remove("tlx-hidden");
    clearImageAnalysisResult();

  } catch (error) {
    renderImageAnalysisError(error.message);
    setBusy(autoDetectCaptureBtn, false, "Auto-Detect & Start");
  }
}

async function onStopImageCapture() {
  const tab = await getActiveTab();
  if (!tab?.id) {
    renderImageAnalysisError("Could not access active tab.");
    return;
  }

  imageStatus.textContent = "Stopping capture and analyzing...";
  setBusy(stopImageCaptureBtn, true, "Processing...");

  try {
    const response = await sendToContentScript(
      tab,
      { type: "STOP_IMAGE_CAPTURE" },
      { injectIfNeeded: true }
    );

    if (!response?.ok) {
      throw new Error(response?.error || "Failed to stop image capture");
    }

    imageCapturing = false;
    imageStatus.textContent = "Analyzing frames and text... This may take a moment.";

  } catch (error) {
    renderImageAnalysisError(error.message);
    imageCapturing = false;
    stopImageCaptureBtn.classList.add("tlx-hidden");
    startImageCaptureBtn.classList.remove("tlx-hidden");
    autoDetectCaptureBtn.classList.remove("tlx-hidden");
    setBusy(stopImageCaptureBtn, false, "Stop & Analyze");
  }
}

function renderImageAnalysisResult(result) {
  stopImageCaptureBtn.classList.add("tlx-hidden");
  startImageCaptureBtn.classList.remove("tlx-hidden");
  autoDetectCaptureBtn.classList.remove("tlx-hidden");
  setBusy(stopImageCaptureBtn, false, "Stop & Analyze");

  if (result.status === "no_content" || (result.frame_count === 0 && !result.captured_text?.length && !result.extracted_text)) {
    imageStatus.textContent = "No text or frames captured.";
    imageAnalysisResult.innerHTML = `
      <p class="tlx-result-explanation">${escapeHtml(result.analysis?.reason || "Play the video, ensure it is visible and not muted, then try again.")}</p>
    `;
    imageAnalysisResult.classList.remove("tlx-hidden");
    return;
  }

  imageStatus.textContent = "✅ Video/image analysis complete!";

  const frameCount = result.frame_count || 0;
  const capturedText = result.captured_text || result.extracted_text;
  const textDisplay = Array.isArray(capturedText)
    ? capturedText.join("\n")
    : (capturedText || result.extracted_text || "No text extracted");
  const predictions = result.misinfo_predictions || [];

  let predictionsHtml = "";
  if (predictions.length > 0) {
    predictionsHtml = `
      <div class="tlx-result-meta">
        <p><strong>Misinformation Analysis:</strong></p>
        ${predictions
          .map(
            (p) => {
            const isFake = String(p.prediction || "").toUpperCase() === "FAKE";
            const confPct = Math.round((Number(p.confidence) || 0) * 100);
            return `
              <div class="tlx-doc-card" style="margin: 0.5em 0;">
                <p class="tlx-doc-paragraph">${escapeHtml((p.text || "").slice(0, 120))}${(p.text || "").length > 120 ? "…" : ""}</p>
                <div class="tlx-result-header">
                  <span class="tlx-pill ${isFake ? "tlx-pill-fake" : "tlx-pill-real"}">${isFake ? "MISINFO" : "REAL"}</span>
                  <span class="tlx-result-confidence">${confPct}%</span>
                </div>
                <p class="tlx-result-explanation">${escapeHtml(String(p.explanation || ""))}</p>
              </div>
            `;
          }
          )
          .join("")}
      </div>
    `;
  }

  const html = `
    <div class="tlx-result-header">
      <strong>Video/Image Analysis</strong>
      <small>${frameCount} frames • ${predictions.length} claim(s) analyzed</small>
    </div>
    <div class="tlx-result-meta">
      <p><strong>Extracted Text:</strong></p>
      <div class="tlx-extracted-text">${escapeHtml(textDisplay)}</div>
    </div>
    ${predictionsHtml}
  `;

  imageAnalysisResult.innerHTML = html;
  imageAnalysisResult.classList.remove("tlx-hidden");
}

function renderImageAnalysisError(error) {
  stopImageCaptureBtn.classList.add("tlx-hidden");
  startImageCaptureBtn.classList.remove("tlx-hidden");
  autoDetectCaptureBtn.classList.remove("tlx-hidden");
  setBusy(stopImageCaptureBtn, false, "Stop & Analyze");

  imageStatus.textContent = `❌ Error: ${error}`;
  clearImageAnalysisResult();
}

function clearImageAnalysisResult() {
  imageAnalysisResult.innerHTML = "";
  imageAnalysisResult.classList.add("tlx-hidden");
}

function findEndOfCentralDirectory(bytes) {
  const minOffset = Math.max(0, bytes.length - 65557);
  for (let i = bytes.length - 22; i >= minOffset; i -= 1) {
    if (
      bytes[i] === 0x50 &&
      bytes[i + 1] === 0x4b &&
      bytes[i + 2] === 0x05 &&
      bytes[i + 3] === 0x06
    ) {
      return i;
    }
  }
  return -1;
}
