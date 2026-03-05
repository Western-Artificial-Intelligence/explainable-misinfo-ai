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

const backendStatus = document.getElementById("backendStatus");
const startAudioBtn = document.getElementById("startAudioBtn");
const stopAudioBtn = document.getElementById("stopAudioBtn");
const audioStatus = document.getElementById("audioStatus");
const transcriptionResult = document.getElementById("transcriptionResult");

const autoDetectCaptureBtn = document.getElementById("autoDetectCaptureBtn");
const startImageCaptureBtn = document.getElementById("startImageCaptureBtn");
const stopImageCaptureBtn = document.getElementById("stopImageCaptureBtn");
const imageStatus = document.getElementById("imageStatus");
const imageAnalysisResult = document.getElementById("imageAnalysisResult");

let overlaysEnabled = false;
let audioCapturing = false;
let imageCapturing = false;

initPopup();

async function initPopup() {
  analyzeSelectionBtn.addEventListener("click", onAnalyzeSelection);
  analyzePageBtn.addEventListener("click", onAnalyzePage);
  toggleOverlayBtn.addEventListener("click", onToggleOverlays);
  analyzeDocumentBtn.addEventListener("click", onAnalyzeDocument);
  startAudioBtn.addEventListener("click", onStartAudioCapture);
  stopAudioBtn.addEventListener("click", onStopAudioCapture);
  autoDetectCaptureBtn.addEventListener("click", onAutoDetectCapture);
  startImageCaptureBtn.addEventListener("click", onStartImageCapture);
  stopImageCaptureBtn.addEventListener("click", onStopImageCapture);

  // Listen for transcription results from content script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "TRANSCRIPTION_COMPLETE") {
      renderTranscriptionResult(message.payload);
    } else if (message.type === "TRANSCRIPTION_ERROR") {
      renderTranscriptionError(message.payload?.error || "Unknown error");
    } else if (message.type === "IMAGE_ANALYSIS_COMPLETE") {
      renderImageAnalysisResult(message.payload);
    } else if (message.type === "IMAGE_ANALYSIS_ERROR") {
      renderImageAnalysisError(message.payload?.error || "Unknown error");
    }
  });

  checkBackendStatus();

  const tab = await getActiveTab();
  if (!tab?.id) {
    pageStatus.textContent = "No active tab available.";
    return;
  }

  const status = await sendToContentScript(tab, { type: "GET_OVERLAY_STATUS" }, { injectIfNeeded: true }).catch(
    () => null
  );
  overlaysEnabled = Boolean(status?.enabled);
  syncOverlayButtonText();
}

async function checkBackendStatus() {
  if (!backendStatus) return;
  const base = await new Promise((resolve) => {
    chrome.storage.sync.get({ backendUrl: "http://localhost:8000" }, (items) => {
      resolve(String(items.backendUrl || "http://localhost:8000").replace(/\/$/, ""));
    });
  });
  try {
    const r = await fetch(`${base}/healthz`, { method: "GET", signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      backendStatus.textContent = "Backend connected";
      backendStatus.className = "tlx-backend-status tlx-backend-ok";
    } else {
      backendStatus.textContent = `Backend error ${r.status}. Check server.`;
      backendStatus.className = "tlx-backend-status tlx-backend-error";
    }
  } catch (e) {
    backendStatus.textContent = "Backend not reachable. Start: uvicorn api.main:app --reload";
    backendStatus.className = "tlx-backend-status tlx-backend-error";
  }
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
    let selectionPayload = await sendToContentScript(tab, { type: "GET_SELECTED_TEXT" }, { injectIfNeeded: true });
    let text = String(selectionPayload?.text || "").trim();
    if (!text) {
      const visiblePayload = await sendToContentScript(tab, { type: "GET_VISIBLE_PRIMARY_TEXT" }, { injectIfNeeded: true });
      text = String(visiblePayload?.text || "").trim();
    }
    if (!text) {
      throw new Error("No text found. Highlight text on the page, or ensure a tweet/post is visible.");
    }

    const response = await chrome.runtime.sendMessage({
      type: "ANALYZE_TEXT",
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
  const pred = String(result?.prediction || "").toUpperCase();
  const pillClass = pred === "FALSE" || pred === "FAKE" ? "tlx-pill-fake"
    : pred === "MIXED" ? "tlx-pill-mixed"
    : "tlx-pill-real";
  const label = pred === "TRUE" || pred === "REAL" ? "TRUE" : pred === "MIXED" ? "MIXED" : "FALSE";
  const confidencePct = Math.round(Number(result?.confidence || 0) * 100);
  const explanation = escapeHtml(String(result?.explanation || ""));
  const sources = Array.isArray(result?.sources) ? result.sources : [];

  const sourcesHtml = sources.length > 0
    ? `<div class="tlx-sources-section">
         <div class="tlx-sources-title">Sources</div>
         ${sources.slice(0, 5).map(s =>
           `<a class="tlx-source-link" href="${escapeHtml(s.url || "#")}" target="_blank" rel="noopener">
              ${escapeHtml(s.title || "Link")}
            </a>`
         ).join("")}
       </div>`
    : "";

  selectionResult.innerHTML = `
    <div class="tlx-result-header">
      <span class="tlx-pill ${pillClass}">${label}</span>
      <span class="tlx-result-confidence">${confidencePct}%</span>
    </div>
    <p class="tlx-result-explanation">${explanation}</p>
    ${sourcesHtml}
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
