/**
 * Audio Capture Module for TikTok Voice-to-Text
 * 
 * This module handles:
 * 1. Detecting video elements on TikTok
 * 2. Capturing audio from videos
 * 3. Sending audio to backend for transcription
 * 4. Displaying transcription results
 */

const TLX_AUDIO_STATE = {
  isCapturing: false,
  currentVideoElement: null,
  audioContext: null,
  mediaStream: null,
  scriptProcessor: null,
  audioChunks: [],
  sampleRate: 44100,
  requestId: null,
  claimId: null
};

const TLX_LIVE_STATE = {
  active: false,
  chunkIntervalId: null,
  chunkIntervalMs: 2000, // live transcript updates every 2 seconds
  noAudioCount: 0,
  // Keep a rolling audio window (more reliable than pure 2s chunks for Whisper)
  rollingWindowSeconds: 8,
  minWindowSeconds: 3,
  rollingPcmChunks: [],
  rollingSamples: 0,
  emittedText: "",
  lastWindowText: ""
};

const TLX_LIVE_MAX_CHARS = 500;

/**
 * Initialize audio capture listeners
 * Called when extension loads on TikTok
 */
function initAudioCapture() {
  // Restore any persisted live transcript baseline for this page URL so that
  // reloads do not lose the accumulated text.
  try {
    const storageKey = `tlx_live_${location.href}`;
    if (chrome?.storage?.local) {
      chrome.storage.local.get({ [storageKey]: "" }, (items) => {
        const saved = items?.[storageKey];
        if (typeof saved === "string" && saved.trim()) {
          TLX_LIVE_STATE.emittedText = saved;
        }
      });
    }
  } catch (e) {
    console.warn("[TruthLens] Failed to restore live transcript baseline:", e);
  }

  // Listen for requests to start audio capture
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "START_AUDIO_CAPTURE") {
      startAudioCapture(message.payload || {})
        .then(result => sendResponse({ ok: true, result }))
        .catch(error => sendResponse({ ok: false, error: error.message }));
      return true; // Keep channel open for async response
    }

    if (message.type === "STOP_AUDIO_CAPTURE") {
      stopAudioCapture()
        .then(() => sendResponse({ ok: true }))
        .catch(error => sendResponse({ ok: false, error: error.message }));
      return true;
    }

    if (message.type === "GET_AUDIO_STATUS") {
      sendResponse({ 
        isCapturing: TLX_AUDIO_STATE.isCapturing,
        requestId: TLX_AUDIO_STATE.requestId
      });
    }

    if (message.type === "START_LIVE_TRANSCRIPT") {
      startLiveTranscript(message.payload || {})
        .then(result => sendResponse({ ok: true, result }))
        .catch(error => sendResponse({ ok: false, error: error.message }));
      return true;
    }
    if (message.type === "PAUSE_LIVE_TRANSCRIPT") {
      pauseLiveTranscript();
      sendResponse({ ok: true });
      return false;
    }
    if (message.type === "STOP_LIVE_TRANSCRIPT") {
      stopLiveTranscript();
      sendResponse({ ok: true });
      return false;
    }
    if (message.type === "SET_LIVE_BASELINE") {
      let baseline = normalizeText(message.payload?.text || "");
      if (baseline.length > TLX_LIVE_MAX_CHARS) {
        baseline = baseline.slice(-TLX_LIVE_MAX_CHARS);
      }
      TLX_LIVE_STATE.emittedText = baseline;
      // Persist updated baseline so it survives reloads
      try {
        const storageKey = `tlx_live_${location.href}`;
        if (chrome?.storage?.local) {
          chrome.storage.local.set({ [storageKey]: TLX_LIVE_STATE.emittedText });
        }
      } catch (e) {
        console.warn("[TruthLens] Failed to persist live baseline:", e);
      }
      sendResponse({ ok: true });
      return false;
    }
    if (message.type === "GET_LIVE_TRANSCRIPT") {
      sendResponse({
        ok: true,
        text: TLX_LIVE_STATE.emittedText || "",
      });
      return false;
    }
    if (message.type === "GET_LIVE_STATUS") {
      sendResponse({
        active: TLX_LIVE_STATE.active,
      });
      return false;
    }
  });
}

/**
 * Start capturing audio from the current video element
 */
async function startAudioCapture(payload = {}) {
  try {
    if (TLX_AUDIO_STATE.isCapturing) {
      throw new Error("Audio capture already in progress");
    }

    // Generate request and claim IDs
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const claimId = `claim_${Date.now()}`;

    TLX_AUDIO_STATE.requestId = requestId;
    TLX_AUDIO_STATE.claimId = claimId;
    TLX_AUDIO_STATE.isCapturing = true;
    TLX_AUDIO_STATE.audioChunks = [];

    console.log(`[TruthLens] Starting audio capture: ${requestId}`);

    // Find TikTok video element
    const videoElement = findVideoElement();
    if (!videoElement) {
      throw new Error("No video element found on page");
    }

    TLX_AUDIO_STATE.currentVideoElement = videoElement;

    // Create audio context and capture audio
    await captureAudioFromVideo(videoElement);

    return {
      requestId,
      claimId,
      message: "Audio capture started"
    };

  } catch (error) {
    TLX_AUDIO_STATE.isCapturing = false;
    console.error("[TruthLens] Audio capture error:", error);
    throw error;
  }
}

/**
 * Stop capture only (stream + context). Does not send audio for transcription.
 */
function stopCaptureOnly() {
  if (TLX_AUDIO_STATE.mediaStream) {
    TLX_AUDIO_STATE.mediaStream.getTracks().forEach(track => track.stop());
    TLX_AUDIO_STATE.mediaStream = null;
  }
  if (TLX_AUDIO_STATE.audioContext) {
    TLX_AUDIO_STATE.audioContext.close().catch(() => {});
    TLX_AUDIO_STATE.audioContext = null;
  }
  TLX_AUDIO_STATE.isCapturing = false;
  TLX_AUDIO_STATE.scriptProcessor = null;
  TLX_AUDIO_STATE.currentVideoElement = null;
  TLX_AUDIO_STATE.audioChunks = [];
}

/**
 * Stop capturing audio and send for transcription
 */
async function stopAudioCapture() {
  try {
    if (!TLX_AUDIO_STATE.isCapturing) {
      throw new Error("Audio capture not in progress");
    }

    console.log("[TruthLens] Stopping audio capture");

    const chunks = TLX_AUDIO_STATE.audioChunks.slice();
    stopCaptureOnly();

    if (chunks.length > 0) {
      const audioBlob = createWavBlob(chunks);
      if (audioBlob) {
        await sendAudioForTranscription(audioBlob);
      } else {
        chrome.runtime.sendMessage({
          type: "TRANSCRIPTION_ERROR",
          payload: { error: "Failed to create audio file. Ensure the video is playing and not muted." }
        }).catch(() => {});
      }
    } else {
      chrome.runtime.sendMessage({
        type: "TRANSCRIPTION_ERROR",
        payload: { error: "No audio captured. Play the video and ensure it is not muted, then try again." }
      }).catch(() => {});
    }

    return { message: "Audio capture stopped" };

  } catch (error) {
    console.error("[TruthLens] Error stopping audio capture:", error);
    stopCaptureOnly();
    throw error;
  }
}

/**
 * Live transcript: chunk-based only. Sends audio every TLX_LIVE_STATE.chunkIntervalMs.
 */
async function startLiveTranscript(options = {}) {
  if (TLX_LIVE_STATE.active) return { message: "Live transcript already active" };

  await startAudioCapture({ live: true });
  TLX_LIVE_STATE.active = true;
  TLX_LIVE_STATE.noAudioCount = 0;
  TLX_LIVE_STATE.rollingPcmChunks = [];
  TLX_LIVE_STATE.rollingSamples = 0;
  TLX_LIVE_STATE.emittedText = normalizeText(options?.initialText || "");
  TLX_LIVE_STATE.lastWindowText = tailWords(TLX_LIVE_STATE.emittedText, 60);

  TLX_LIVE_STATE.chunkIntervalId = setInterval(async () => {
    if (!TLX_LIVE_STATE.active) return;

    if (!TLX_AUDIO_STATE.isCapturing) {
      TLX_LIVE_STATE.noAudioCount += 1;
      if (TLX_LIVE_STATE.noAudioCount === 2) {
        chrome.runtime.sendMessage({
          type: "LIVE_TRANSCRIPT_NO_AUDIO",
          payload: {
            hint: "No audio captured yet. On TikTok/YouTube, ensure the main video is playing (not paused), then try Stop and Start capture again."
          }
        }).catch(() => {});
      }
      return;
    }

    TLX_LIVE_STATE.noAudioCount = 0;

    // Move newly captured PCM into the rolling window
    const newChunks = TLX_AUDIO_STATE.audioChunks.slice();
    TLX_AUDIO_STATE.audioChunks = [];
    if (newChunks.length === 0) {
      TLX_LIVE_STATE.noAudioCount += 1;
      if (TLX_LIVE_STATE.noAudioCount === 2) {
        chrome.runtime.sendMessage({
          type: "LIVE_TRANSCRIPT_NO_AUDIO",
          payload: {
            hint: "No audio captured yet. On TikTok/YouTube, ensure the main video is playing (not paused), then try Stop and Start capture again."
          }
        }).catch(() => {});
      }
      return;
    }

    for (const arr of newChunks) {
      TLX_LIVE_STATE.rollingPcmChunks.push(arr);
      TLX_LIVE_STATE.rollingSamples += arr.length;
    }

    // Trim old audio beyond rolling window
    const sampleRate = TLX_AUDIO_STATE.sampleRate || 44100;
    const maxSamples = Math.floor(sampleRate * TLX_LIVE_STATE.rollingWindowSeconds);
    while (TLX_LIVE_STATE.rollingSamples > maxSamples && TLX_LIVE_STATE.rollingPcmChunks.length > 0) {
      const dropped = TLX_LIVE_STATE.rollingPcmChunks.shift();
      TLX_LIVE_STATE.rollingSamples -= dropped.length;
    }

    const minSamples = Math.floor(sampleRate * TLX_LIVE_STATE.minWindowSeconds);
    if (TLX_LIVE_STATE.rollingSamples < minSamples) {
      return;
    }

    const audioBlob = createWavBlob(TLX_LIVE_STATE.rollingPcmChunks);
    if (audioBlob) {
      try {
        await sendLiveChunkForTranscription(audioBlob);
      } catch (e) {
        chrome.runtime.sendMessage({
          type: "LIVE_TRANSCRIPT_ERROR",
          payload: { error: e.message || "Chunk transcription failed" }
        }).catch(() => {});
      }
    }
  }, TLX_LIVE_STATE.chunkIntervalMs);

  return { message: "Live transcript started" };
}

function pauseLiveTranscript() {
  if (TLX_LIVE_STATE.chunkIntervalId) {
    clearInterval(TLX_LIVE_STATE.chunkIntervalId);
    TLX_LIVE_STATE.chunkIntervalId = null;
  }
  TLX_LIVE_STATE.active = false;
  TLX_LIVE_STATE.rollingPcmChunks = [];
  TLX_LIVE_STATE.rollingSamples = 0;
  stopCaptureOnly();
}

function stopLiveTranscript() {
  pauseLiveTranscript();
}

/**
 * Send a single audio chunk for transcription and broadcast TRANSCRIPT_CHUNK (no TRANSCRIPTION_COMPLETE)
 */
async function sendLiveChunkForTranscription(audioBlob) {
  const backendUrl = await getBackendUrl();
  const apiUrl = `${backendUrl}/api/audio/transcribe-file`;
  const requestId = `live_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const claimId = `chunk_${Date.now()}`;

  const formData = new FormData();
  formData.append("request_id", requestId);
  formData.append("claim_id", claimId);
  formData.append("file", audioBlob, "chunk.wav");
  formData.append("language", "en");

  const response = await fetch(apiUrl, { method: "POST", body: formData });
  if (!response.ok) {
    throw new Error(`Backend ${response.status}`);
  }

  const result = await response.json();
  console.log("[TruthLens] Live chunk result:", result);
  if (result && result.error && !result.transcription) {
    throw new Error(result.error);
  }

  const tx = result?.transcription || result;
  let text = tx?.full_text || tx?.text || "";
  if (!text && Array.isArray(tx?.segments)) {
    text = tx.segments.map((s) => s.text || "").join(" ");
  }
  text = normalizeText(text || "");
  if (!text) return;

  // Deduplicate overlap between rolling windows (prevents repeated text spam)
  const historyContext = tailWords(TLX_LIVE_STATE.emittedText, 120);
  const delta = computeDeltaTranscript(historyContext, text);
  if (!delta) return;
  let merged = mergeTranscript(TLX_LIVE_STATE.emittedText, delta);
  if (merged.length > TLX_LIVE_MAX_CHARS) {
    merged = merged.slice(-TLX_LIVE_MAX_CHARS);
  }
  TLX_LIVE_STATE.emittedText = merged;
  TLX_LIVE_STATE.lastWindowText = text;
  // Persist the cumulative transcript so it survives reloads
  try {
    const storageKey = `tlx_live_${location.href}`;
    if (chrome?.storage?.local) {
      chrome.storage.local.set({ [storageKey]: TLX_LIVE_STATE.emittedText });
    }
  } catch (e) {
    console.warn("[TruthLens] Failed to persist live transcript:", e);
  }

  chrome.runtime.sendMessage({
    type: "TRANSCRIPT_CHUNK",
    payload: { text: delta }
  }).catch(() => {});
}

function normalizeText(s) {
  return String(s || "")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenizeForMatch(text) {
  const rawTokens = normalizeText(text).split(" ").filter(Boolean);
  return rawTokens
    .map((raw) => {
      const key = raw.toLowerCase().replace(/[^a-z0-9]+/g, "");
      return { raw, key };
    })
    .filter((t) => t.key);
}

function tailWords(text, n) {
  const toks = normalizeText(text).split(" ").filter(Boolean);
  if (toks.length <= n) return toks.join(" ");
  return toks.slice(-n).join(" ");
}

function jaccardSimilarity(keysA, keysB) {
  const a = new Set(keysA.filter(Boolean));
  const b = new Set(keysB.filter(Boolean));
  if (a.size === 0 && b.size === 0) return 1;
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  for (const k of a) {
    if (b.has(k)) inter += 1;
  }
  const union = a.size + b.size - inter;
  return union ? inter / union : 0;
}

function computeDeltaTranscript(prevText, newText) {
  const prevToks = tokenizeForMatch(prevText);
  const nextToks = tokenizeForMatch(newText);
  if (nextToks.length === 0) return "";
  if (prevToks.length === 0) return nextToks.map((t) => t.raw).join(" ").trim();

  // Compare suffix of previous context with prefix of next window.
  // Allow small mismatches to handle ASR re-phrasing across overlapping windows.
  const prevKeys = prevToks.map((t) => t.key);
  const nextKeys = nextToks.map((t) => t.key);

  const maxOverlap = Math.min(60, prevKeys.length, nextKeys.length);
  for (let k = maxOverlap; k >= 4; k--) {
    const mismatchBudget = Math.max(1, Math.floor(k * 0.15));
    let mismatches = 0;
    for (let i = 0; i < k; i++) {
      if (prevKeys[prevKeys.length - k + i] !== nextKeys[i]) {
        mismatches += 1;
        if (mismatches > mismatchBudget) break;
      }
    }
    if (mismatches <= mismatchBudget) {
      let deltaToks = nextToks.slice(k);
      // If the boundary repeats the last token, drop it.
      const lastPrevKey = prevKeys[prevKeys.length - 1] || "";
      if (deltaToks.length > 0 && deltaToks[0].key === lastPrevKey) {
        deltaToks = deltaToks.slice(1);
      }
      // Cap huge deltas (helps when Whisper outputs a long re-transcription).
      if (deltaToks.length > 30) {
        deltaToks = deltaToks.slice(-30);
      }
      return deltaToks.map((t) => t.raw).join(" ").trim();
    }
  }

  // If next window is effectively a subset of prev window, emit nothing.
  const prevStr = prevKeys.join(" ");
  const nextStr = nextKeys.join(" ");
  if (prevStr.includes(nextStr)) return "";

  // Anchor-based fallback: if the beginning of the next window appears in our recent context,
  // emit only what comes after that anchor.
  const maxAnchor = Math.min(10, nextKeys.length);
  for (let len = maxAnchor; len >= 5; len--) {
    const anchor = nextKeys.slice(0, len).join(" ");
    if (anchor && prevStr.includes(anchor)) {
      const remainder = nextToks.slice(len).map((t) => t.raw).join(" ").trim();
      if (!remainder) return "";
      return remainder;
    }
  }

  // Otherwise, emit only a short tail and suppress if it looks like a repeat of recent output.
  const prevTailKeys = prevKeys.slice(-40);
  const tailCount = Math.min(16, nextToks.length);
  const tailToks = nextToks.slice(-tailCount);
  const tailKeys = tailToks.map((t) => t.key);
  const sim = jaccardSimilarity(tailKeys, prevTailKeys);
  if (sim > 0.7) return "";
  return tailToks.map((t) => t.raw).join(" ").trim();
}

function mergeTranscript(prevText, delta) {
  const prev = normalizeText(prevText);
  const d = normalizeText(delta);
  if (!prev) return d;
  if (!d) return prev;
  return `${prev} ${d}`.trim();
}

/**
 * Create a valid WAV blob from PCM Int16 chunks.
 * Sample rate comes from AudioContext (typically 44100 or 48000).
 */
function createWavBlob(pcmChunks) {
  if (!pcmChunks || pcmChunks.length === 0) return null;

  const sampleRate = TLX_AUDIO_STATE.sampleRate || 44100;
  const numChannels = 1;
  const bitsPerSample = 16;

  let totalLength = 0;
  for (let i = 0; i < pcmChunks.length; i++) {
    totalLength += pcmChunks[i].length * 2;
  }

  const dataSize = totalLength;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  let offset = 0;

  function writeStr(off, s) {
    for (let i = 0; i < s.length; i++) {
      view.setUint8(off + i, s.charCodeAt(i));
    }
  }

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
  view.setUint16(32, numChannels * (bitsPerSample / 8), true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  offset = 44;
  for (let i = 0; i < pcmChunks.length; i++) {
    const arr = pcmChunks[i];
    for (let j = 0; j < arr.length; j++) {
      view.setInt16(offset, arr[j], true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/**
 * Find the video element on the page (TikTok, YouTube, etc.)
 * Prefers the video that is actually playing (TikTok feed has multiple videos).
 */
function findVideoElement() {
  const videoElements = Array.from(document.querySelectorAll('video'));
  if (videoElements.length === 0) return null;

  // Prefer a video that is currently playing (not paused, has data)
  const playing = videoElements.filter(
    (v) => !v.paused && v.readyState >= 2
  );
  if (playing.length > 0) {
    const byVisible = playing.slice().sort((a, b) => visibleArea(b) - visibleArea(a));
    return byVisible[0];
  }

  // Fallback: largest visible (e.g. video not yet playing or muted attribute on element)
  let best = null;
  let maxArea = 0;
  videoElements.forEach((video) => {
    const area = visibleArea(video);
    if (area > maxArea) {
      maxArea = area;
      best = video;
    }
  });
  return best;
}

function visibleArea(el) {
  const rect = el.getBoundingClientRect();
  const h = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(0, rect.top));
  const w = Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(0, rect.left));
  return h * w;
}

/**
 * Capture audio from video element.
 * Tries createMediaElementSource first (best audio); if the element is already connected
 * (e.g. by YouTube), falls back to captureStream() + createMediaStreamSource.
 */
async function captureAudioFromVideo(videoElement) {
  try {
    TLX_AUDIO_STATE.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    TLX_AUDIO_STATE.sampleRate = TLX_AUDIO_STATE.audioContext.sampleRate;

    let source = null;
    const alreadyConnected = /already connected|MediaElementSourceNode/i;

    try {
      source = TLX_AUDIO_STATE.audioContext.createMediaElementSource(videoElement);
      TLX_AUDIO_STATE.mediaStream = null;
    } catch (e) {
      if (alreadyConnected.test(String(e.message))) {
        const stream = videoElement.captureStream
          ? videoElement.captureStream()
          : (videoElement.mozCaptureStream && videoElement.mozCaptureStream());
        if (!stream) {
          throw new Error("This page does not support stream capture. Try refreshing and start capture after the video is playing.");
        }
        const audioTracks = stream.getAudioTracks ? stream.getAudioTracks() : [];
        if (audioTracks.length === 0) {
          throw new Error("This page does not expose audio in the capture. Try starting capture only after the video has been playing for a few seconds, or use a different video.");
        }
        TLX_AUDIO_STATE.mediaStream = stream;
        source = TLX_AUDIO_STATE.audioContext.createMediaStreamSource(stream);
      } else {
        throw e;
      }
    }

    const sampleRate = TLX_AUDIO_STATE.audioContext.sampleRate;

    if (TLX_LIVE_STATE.useLiveWs && TLX_LIVE_STATE.ws) {
      try {
        const workletUrl = chrome.runtime.getURL("capture-worklet.js");
        await TLX_AUDIO_STATE.audioContext.audioWorklet.addModule(workletUrl);
        const workletNode = new AudioWorkletNode(TLX_AUDIO_STATE.audioContext, "capture-worklet-processor", {
          processorOptions: { sampleRate }
        });
        workletNode.port.onmessage = (event) => {
          if (TLX_LIVE_STATE.ws && TLX_LIVE_STATE.ws.readyState === WebSocket.OPEN) {
            TLX_LIVE_STATE.ws.send(event.data);
          }
        };
        source.connect(workletNode);
        workletNode.connect(TLX_AUDIO_STATE.audioContext.destination);
        TLX_LIVE_STATE.workletNode = workletNode;
      } catch (e) {
        console.warn("[TruthLens] AudioWorklet failed, using chunk fallback:", e);
        TLX_LIVE_STATE.useLiveWs = false;
      }
    }

    if (!TLX_LIVE_STATE.workletNode) {
      const bufferSize = 4096;
      TLX_AUDIO_STATE.scriptProcessor = TLX_AUDIO_STATE.audioContext.createScriptProcessor(
        bufferSize,
        1,
        1
      );
      TLX_AUDIO_STATE.scriptProcessor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        const outputData = event.outputBuffer.getChannelData(0);
        const pcmData = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const v = inputData[i];
          // Pass-through audio so user still hears sound
          outputData[i] = v;
          // Capture as 16-bit PCM
          pcmData[i] = Math.max(-1, Math.min(1, v)) < 0 ? v * 0x8000 : v * 0x7fff;
        }
        TLX_AUDIO_STATE.audioChunks.push(pcmData);
      };
      source.connect(TLX_AUDIO_STATE.scriptProcessor);
      TLX_AUDIO_STATE.scriptProcessor.connect(TLX_AUDIO_STATE.audioContext.destination);
    }

    if (TLX_AUDIO_STATE.audioContext.state === "suspended") {
      await TLX_AUDIO_STATE.audioContext.resume();
    }

    console.log("[TruthLens] Audio capture initialized");

  } catch (error) {
    console.error("[TruthLens] Error initializing audio capture:", error);
    throw error;
  }
}

/**
 * Send captured audio to backend for transcription
 */
async function sendAudioForTranscription(audioBlob) {
  try {
    const backendUrl = await getBackendUrl();
    const apiUrl = `${backendUrl}/api/audio/transcribe-file`;

    console.log(`[TruthLens] Sending audio to: ${apiUrl}`);

    // Prepare form data
    const formData = new FormData();
    formData.append('request_id', TLX_AUDIO_STATE.requestId);
    formData.append('claim_id', TLX_AUDIO_STATE.claimId);
    formData.append('file', audioBlob, 'tiktok_audio.wav');
    formData.append('language', 'en'); // Default to English; could be configurable

    // Send to backend
    const response = await fetch(apiUrl, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    console.log("[TruthLens] Transcription result:", result);

    // Check for backend error (e.g., Whisper not installed)
    if (result && result.error && !result.transcription) {
      chrome.runtime.sendMessage({
        type: "TRANSCRIPTION_ERROR",
        payload: { error: result.error }
      }).catch(e => console.log("Could not send to popup:", e));
      throw new Error(result.error);
    }

    // Send result to popup/background
    chrome.runtime.sendMessage({
      type: "TRANSCRIPTION_COMPLETE",
      payload: result
    }).catch(e => console.log("Could not send to popup:", e));

    return result;

  } catch (error) {
    console.error("[TruthLens] Error sending audio for transcription:", error);
    
    // Notify of error
    chrome.runtime.sendMessage({
      type: "TRANSCRIPTION_ERROR",
      payload: { error: error.message }
    }).catch(e => console.log("Could not send error to popup:", e));

    throw error;
  }
}

/**
 * Get backend URL from extension settings
 */
async function getBackendUrl() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ backendUrl: 'http://127.0.0.1:8000' }, (items) => {
      resolve(items.backendUrl);
    });
  });
}

// Initialize audio capture on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAudioCapture);
} else {
  initAudioCapture();
}
