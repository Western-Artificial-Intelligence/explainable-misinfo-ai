/**
 * Image & Text Capture Module for TikTok/YouTube Shorts
 * 
 * This module handles:
 * 1. Intelligent silence detection using RMS analysis
 * 2. Smart frame capture (1 frame per 1.5s or on text change)
 * 3. Text extraction via OCR and DOM scraping
 * 4. Comprehensive text cleanup
 * 5. Integration with misinfo detection pipeline
 */

const TLX_IMAGE_STATE = {
  isCapturing: false,
  currentVideoElement: null,
  canvasContext: null,
  extractedFrames: [],
  extractedText: [],
  requestId: null,
  claimId: null,
  audioAnalysis: null,
  captureInterval: null,
  frameBuffer: null,
  lastFrameHash: null,
  audioContext: null,
  analyser: null,
};

// ==========================================
// Audio Analysis for Silence Detection
// ==========================================

/**
 * Analyze audio levels using RMS (Root Mean Square) energy.
 * This determines if there is meaningful voice content.
 * 
 * Algorithm:
 * - Capture audio context from video element
 * - Sample frequency data every 100ms
 * - Calculate RMS energy from frequency domain
 * - Compare against thresholds
 * - If < VOICE_THRESHOLD for extended period -> use image capture
 */
async function analyzeAudioLevelsAdvanced(videoElement) {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContext();
    // Resume if suspended (required after user gesture on strict sites like TikTok)
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.8;

    const source = audioContext.createMediaElementAudioSource(videoElement);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    // RMS thresholds (normalized 0-1)
    const SILENCE_THRESHOLD = 0.02; // ~-34 dB
    const VOICE_THRESHOLD = 0.05;   // ~-26 dB
    const MIN_VOICE_RATIO = 0.1;    // At least 10% voice
    const SAMPLE_DURATION_MS = 1000; // Sample for 1 second

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    
    return new Promise((resolve) => {
      let samples = [];
      let silenceCount = 0;
      let voiceCount = 0;
      
      const sampleInterval = setInterval(() => {
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate RMS from frequency bins
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const normalized = dataArray[i] / 255.0;
          sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / dataArray.length);
        samples.push(rms);
        
        // Categorize sample
        if (rms < SILENCE_THRESHOLD) {
          silenceCount++;
        } else if (rms >= VOICE_THRESHOLD) {
          voiceCount++;
        }
        
        // After ~1 second of sampling, evaluate
        if (samples.length * 100 >= SAMPLE_DURATION_MS) {
          clearInterval(sampleInterval);
          
          const avgEnergy = samples.reduce((a, b) => a + b, 0) / samples.length;
          const voiceRatio = voiceCount / samples.length;
          const hasVoice = voiceRatio >= MIN_VOICE_RATIO && avgEnergy >= VOICE_THRESHOLD;
          
          console.log(
            `[TruthLens] Audio Analysis - Avg RMS: ${avgEnergy.toFixed(3)}, ` +
            `Voice Ratio: ${(voiceRatio * 100).toFixed(1)}%, ` +
            `Has Voice: ${hasVoice}`
          );
          
          // Attempt to close audio context
          try {
            audioContext.close();
          } catch (e) {
            // Context may already be closed
          }
          
          resolve({
            hasVoice: hasVoice,
            averageEnergy: avgEnergy,
            voiceRatio: voiceRatio,
            shouldCaptureImage: !hasVoice,
            sampleCount: samples.length,
          });
        }
      }, 100);
      
      // Fallback timeout (2 seconds)
      setTimeout(() => {
        clearInterval(sampleInterval);
        if (samples.length > 0) {
          const avgEnergy = samples.reduce((a, b) => a + b, 0) / samples.length;
          const voiceRatio = voiceCount / samples.length;
          const hasVoice = voiceRatio >= MIN_VOICE_RATIO && avgEnergy >= VOICE_THRESHOLD;
          
          try {
            audioContext.close();
          } catch (e) {}
          
          resolve({
            hasVoice: hasVoice,
            averageEnergy: avgEnergy,
            voiceRatio: voiceRatio,
            shouldCaptureImage: !hasVoice,
            sampleCount: samples.length,
          });
        } else {
          resolve({
            hasVoice: false,
            averageEnergy: 0,
            voiceRatio: 0,
            shouldCaptureImage: true,
            sampleCount: 0,
          });
        }
      }, 2500);
    });
  } catch (error) {
    console.warn("[TruthLens] Audio analysis failed:", error);
    return {
      hasVoice: false,
      averageEnergy: 0,
      voiceRatio: 0,
      shouldCaptureImage: true,
      sampleCount: 0,
    };
  }
}

// ==========================================
// Frame Capture with Smart Sampling
// ==========================================

/**
 * Calculate frame hash to detect significant visual changes.
 * Uses simple pixel sampling to avoid expensive full hashing.
 */
function calculateFrameHash(canvas) {
  try {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Sample every 1000th pixel to keep it fast
    let hash = 0;
    for (let i = 0; i < data.length; i += 4000) {
      hash = ((hash << 5) - hash) + (data[i] + data[i+1] + data[i+2]);
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return hash;
  } catch (e) {
    return Date.now();
  }
}

/**
 * Determine if frame is significantly different from previous.
 * Helps avoid capturing redundant similar frames.
 */
function hasFrameChanged(canvas, threshold = 0.1) {
  const currentHash = calculateFrameHash(canvas);
  
  if (TLX_IMAGE_STATE.lastFrameHash === null) {
    TLX_IMAGE_STATE.lastFrameHash = currentHash;
    return true;
  }
  
  const changed = Math.abs(currentHash - TLX_IMAGE_STATE.lastFrameHash) > 1000;
  TLX_IMAGE_STATE.lastFrameHash = currentHash;
  
  return changed;
}

// ==========================================
// Initialization
// ==========================================

/**
 * Initialize image capture listeners
 */
function initImageCapture() {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "START_IMAGE_CAPTURE") {
      startImageCapture(message.payload || {})
        .then(result => sendResponse({ ok: true, result }))
        .catch(error => sendResponse({ ok: false, error: String(error?.message || error || "Unknown error") }));
      return true;
    }

    if (message.type === "STOP_IMAGE_CAPTURE") {
      stopImageCapture()
        .then(() => sendResponse({ ok: true }))
        .catch(error => sendResponse({ ok: false, error: error.message }));
      return true;
    }

    if (message.type === "SHOULD_USE_IMAGE_CAPTURE") {
      shouldUseImageCapture(message.payload || {})
        .then(result => sendResponse({ ok: true, result }))
        .catch(error => sendResponse({ ok: false, error: error.message }));
      return true;
    }

    if (message.type === "GET_IMAGE_STATUS") {
      sendResponse({
        isCapturing: TLX_IMAGE_STATE.isCapturing,
        requestId: TLX_IMAGE_STATE.requestId,
        frameCount: TLX_IMAGE_STATE.extractedFrames.length,
        textCount: TLX_IMAGE_STATE.extractedText.length
      });
    }
  });
}

/**
 * Determine if image capture should be used instead of audio
 */
async function shouldUseImageCapture(payload = {}) {
  try {
    const videoElement = findVideoElement();
    if (!videoElement) {
      return { shouldCapture: false, reason: "No video element found" };
    }

    const audioAnalysis = await analyzeAudioLevelsAdvanced(videoElement);
    
    return {
      shouldCapture: audioAnalysis.shouldCaptureImage,
      audioAnalysis: audioAnalysis,
      reason: audioAnalysis.shouldCaptureImage 
        ? `Minimal audio detected (${(audioAnalysis.voiceRatio * 100).toFixed(1)}% voice)`
        : `Voice content detected (${(audioAnalysis.voiceRatio * 100).toFixed(1)}% voice)`
    };

  } catch (error) {
    console.error("[TruthLens] Error determining capture method:", error);
    return { shouldCapture: false, reason: error.message };
  }
}

/**
 * Start capturing image and text from video
 */
async function startImageCapture(payload = {}) {
  try {
    // If already capturing (e.g. popup was closed and reopened), clean up and allow fresh start
    if (TLX_IMAGE_STATE.isCapturing) {
      if (payload.forceCapture) {
        if (TLX_IMAGE_STATE.captureInterval) {
          clearInterval(TLX_IMAGE_STATE.captureInterval);
          TLX_IMAGE_STATE.captureInterval = null;
        }
        TLX_IMAGE_STATE.isCapturing = false;
        TLX_IMAGE_STATE.extractedFrames = [];
        TLX_IMAGE_STATE.extractedText = [];
      } else {
        throw new Error("Image capture already in progress. Close and reopen the popup, then try Force Image Capture.");
      }
    }

    const requestId = `req_img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const claimId = `claim_${Date.now()}`;

    TLX_IMAGE_STATE.requestId = requestId;
    TLX_IMAGE_STATE.claimId = claimId;
    TLX_IMAGE_STATE.isCapturing = true;
    TLX_IMAGE_STATE.extractedFrames = [];
    TLX_IMAGE_STATE.extractedText = [];
    TLX_IMAGE_STATE.lastFrameHash = null;

    console.log(`[TruthLens] Starting image capture: ${requestId}`);

    const videoElement = findVideoElement();
    if (!videoElement) {
      throw new Error("No video element found on page");
    }

    TLX_IMAGE_STATE.currentVideoElement = videoElement;

    // Analyze audio to confirm we should use image capture
    if (!payload.forceCapture) {
      const audioAnalysis = await analyzeAudioLevelsAdvanced(videoElement);
      TLX_IMAGE_STATE.audioAnalysis = audioAnalysis;

      if (!audioAnalysis.shouldCaptureImage) {
        throw new Error("Video has voice content - use audio capture instead");
      }
    }

    // Try to load Tesseract (optional - backend does OCR if client fails)
    await ensureTesseractLoaded();

    // Extract on-screen text elements
    await extractOnScreenText();

    // Start frame capture loop
    startFrameCaptureLoop(videoElement);

    return {
      requestId,
      claimId,
      audioAnalysis: TLX_IMAGE_STATE.audioAnalysis,
      message: "Image capture started"
    };

  } catch (error) {
    TLX_IMAGE_STATE.isCapturing = false;
    console.error("[TruthLens] Image capture error:", error);
    throw error;
  }
}

/**
 * Stop capturing and send for processing
 */
async function stopImageCapture() {
  try {
    if (!TLX_IMAGE_STATE.isCapturing) {
      throw new Error("Image capture not in progress");
    }

    console.log("[TruthLens] Stopping image capture");

    if (TLX_IMAGE_STATE.captureInterval) {
      clearInterval(TLX_IMAGE_STATE.captureInterval);
      TLX_IMAGE_STATE.captureInterval = null;
    }

    TLX_IMAGE_STATE.isCapturing = false;

    if (TLX_IMAGE_STATE.extractedFrames.length > 0 || TLX_IMAGE_STATE.extractedText.length > 0) {
      await sendImageAndTextForAnalysis();
    } else {
      chrome.runtime.sendMessage({
        type: "IMAGE_ANALYSIS_COMPLETE",
        payload: {
          request_id: TLX_IMAGE_STATE.requestId,
          claim_id: TLX_IMAGE_STATE.claimId,
          frame_count: 0,
          captured_text: [],
          extracted_text: "",
          misinfo_predictions: [],
          status: "no_content",
          analysis: { reason: "No frames or text captured. Play the video and ensure it is visible." }
        }
      }).catch(() => {});
    }

    return { message: "Image capture stopped" };

  } catch (error) {
    console.error("[TruthLens] Error stopping image capture:", error);
    TLX_IMAGE_STATE.isCapturing = false;
    throw error;
  }
}

/**
 * Smart frame capture loop: 1 frame per 1.5s OR on significant visual change
 */
function startFrameCaptureLoop(videoElement) {
  const canvas = document.createElement("canvas");
  const w = videoElement.videoWidth || videoElement.clientWidth || 640;
  const h = videoElement.videoHeight || videoElement.clientHeight || 360;
  canvas.width = Math.max(w, 320);
  canvas.height = Math.max(h, 180);
  TLX_IMAGE_STATE.canvasContext = canvas.getContext("2d");
  
  let lastCaptureTime = 0;
  const MIN_CAPTURE_INTERVAL_MS = 1500; // 1.5 seconds
  const MAX_FRAMES = 15; // Limit frames to avoid excessive data

  TLX_IMAGE_STATE.captureInterval = setInterval(async () => {
    if (!TLX_IMAGE_STATE.isCapturing) return;
    if (TLX_IMAGE_STATE.extractedFrames.length >= MAX_FRAMES) {
      clearInterval(TLX_IMAGE_STATE.captureInterval);
      return;
    }

    const now = Date.now();
    
    // Draw current frame to canvas (may fail for cross-origin video)
    try {
      try {
        TLX_IMAGE_STATE.canvasContext.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      } catch (drawErr) {
        if (drawErr.name === "SecurityError" || drawErr.message?.includes("tainted")) {
          return;
        }
        throw drawErr;
      }

      const shouldCapture = (now - lastCaptureTime > MIN_CAPTURE_INTERVAL_MS) || hasFrameChanged(canvas);
      if (!shouldCapture) return;

      lastCaptureTime = now;

      canvas.toBlob(async (blob) => {
        if (blob && TLX_IMAGE_STATE.extractedFrames.length < MAX_FRAMES) {
          const frameData = {
            timestamp: now,
            blob: blob,
            text: null,
          };

          try {
            const ocrText = await performOCROnFrame(blob);
            frameData.text = ocrText;
            
            if (ocrText && ocrText.trim().length > 0) {
              if (!TLX_IMAGE_STATE.extractedText.includes(ocrText.trim())) {
                TLX_IMAGE_STATE.extractedText.push(ocrText.trim());
              }
            }
          } catch (ocrError) {
            console.warn("[TruthLens] OCR error:", ocrError);
          }

          TLX_IMAGE_STATE.extractedFrames.push(frameData);
          console.log(
            `[TruthLens] Frame ${TLX_IMAGE_STATE.extractedFrames.length}: ` +
            `${frameData.text?.length || 0} chars from OCR`
          );
        }
      }, 'image/jpeg', 0.65); // Slightly lower quality to reduce payload

    } catch (error) {
      console.error("[TruthLens] Frame capture error:", error);
    }
  }, 300); // Check every 300ms, but only save every 1.5s+

  console.log("[TruthLens] Frame capture loop started");
}

/**
 * Perform OCR on a single frame using Tesseract.js
 */
async function performOCROnFrame(imageBlob) {
  try {
    const TesseractLib = window.Tesseract || (typeof globalThis !== "undefined" && globalThis.Tesseract);
    if (!TesseractLib || !TesseractLib.recognize) {
      throw new Error("Tesseract not loaded");
    }

    const { data } = await TesseractLib.recognize(imageBlob, "eng", {
      logger: msg => {
        if (msg.status === 'recognizing') {
          // Only log major progress
          const progress = (msg.progress * 100).toFixed(0);
          if (progress % 25 === 0) {
            console.log(`[TruthLens] OCR: ${progress}%`);
          }
        }
      }
    });

    return data.text || "";

  } catch (error) {
    console.error("[TruthLens] OCR error:", error);
    return "";
  }
}

/**
 * Extract text from on-screen elements (captions, overlays, etc.)
 */
async function extractOnScreenText() {
  try {
    const textElements = document.querySelectorAll(
      'p, span, div, h1, h2, h3, h4, h5, h6, [data-caption], .caption, .subtitle, .text-overlay, video ~ *'
    );

    const extractedTexts = new Set();

    textElements.forEach(element => {
      const text = element.textContent?.trim();
      if (text && text.length > 0 && text.length < 500) {
        if (!isCommonUIText(text) && !isLikelyUIControl(text)) {
          extractedTexts.add(text);
        }
      }
    });

    extractedTexts.forEach(text => {
      if (!TLX_IMAGE_STATE.extractedText.includes(text)) {
        TLX_IMAGE_STATE.extractedText.push(text);
      }
    });

    console.log(`[TruthLens] Extracted ${extractedTexts.size} DOM text elements`);

  } catch (error) {
    console.error("[TruthLens] Error extracting on-screen text:", error);
  }
}

/**
 * Filter out common UI text that's not content
 */
function isCommonUIText(text) {
  const commonPatterns = [
    /^\d+[kmb]?\s*$/i, // Numbers like "1.2M", "5K"
    /^v\d+\.\d+/, // Version numbers
    /^rate|review|share|follow|subscribe|like|comment|buy|shop/i,
  ];
  
  return commonPatterns.some(pattern => pattern.test(text));
}

/**
 * Detect if text is from UI controls rather than content
 */
function isLikelyUIControl(text) {
  const uiKeywords = [
    'menu', 'settings', 'profile', 'home', 'trending', 'search',
    'notifications', 'messages', 'explore', 'close', 'back', 'next'
  ];
  
  return uiKeywords.some(kw => text.toLowerCase().includes(kw));
}

/**
 * Recursively find all video elements, including inside shadow DOM
 */
function findAllVideos(root = document) {
  const videos = Array.from(root.querySelectorAll('video'));
  const walk = (node) => {
    if (node.shadowRoot) {
      videos.push(...node.shadowRoot.querySelectorAll('video'));
      node.shadowRoot.querySelectorAll('*').forEach(walk);
    }
  };
  root.querySelectorAll('*').forEach(walk);
  return videos;
}

/**
 * Find the most visible video element on page (including shadow DOM)
 */
function findVideoElement() {
  const videoElements = findAllVideos();
  if (videoElements.length === 0) {
    return null;
  }

  let mostVisibleVideo = null;
  let maxVisibleArea = 0;

  videoElements.forEach(video => {
    const rect = video.getBoundingClientRect();
    const visibleHeight = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(0, rect.top));
    const visibleWidth = Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(0, rect.left));
    const visibleArea = visibleHeight * visibleWidth;

    if (visibleArea > maxVisibleArea) {
      maxVisibleArea = visibleArea;
      mostVisibleVideo = video;
    }
  });

  return mostVisibleVideo;
}

/**
 * Ensure Tesseract.js library is loaded in content script context.
 * Tries extension-bundled script first (CSP-safe on TikTok), then CDN.
 * Returns true if loaded, false otherwise. Never throws - capture continues
 * without client OCR; backend will do server-side OCR on frames.
 */
async function ensureTesseractLoaded() {
  if (typeof Tesseract !== "undefined" && Tesseract.recognize) {
    return true;
  }
  if (typeof (window.Tesseract || (typeof globalThis !== "undefined" && globalThis.Tesseract)) !== "undefined") {
    const T = window.Tesseract || globalThis.Tesseract;
    if (T && T.recognize) {
      if (!window.Tesseract) window.Tesseract = T;
      return true;
    }
  }

  const loaders = [
    async () => {
      // Load from extension (CSP-safe on TikTok - chrome-extension is allowed)
      const url = typeof chrome !== "undefined" && chrome.runtime?.getURL
        ? chrome.runtime.getURL("tesseract.min.js")
        : null;
      if (!url) return false;
      return new Promise((resolve) => {
        const script = document.createElement("script");
        script.src = url;
        script.onload = () => {
          const T = window.Tesseract || (typeof globalThis !== "undefined" && globalThis.Tesseract);
          if (T && T.recognize) {
            if (!window.Tesseract) window.Tesseract = T;
            console.log("[TruthLens] Tesseract.js loaded from extension");
            resolve(true);
          } else resolve(false);
        };
        script.onerror = () => resolve(false);
        (document.head || document.documentElement).appendChild(script);
      });
    },
    async () => {
      try {
        const res = await fetch(
          "https://cdn.jsdelivr.net/npm/tesseract.js@5.0.2/dist/tesseract.min.js",
          { mode: "cors" }
        );
        if (!res.ok) return false;
        const code = await res.text();
        new Function(code).call(typeof window !== "undefined" ? window : globalThis);
        const T = window.Tesseract || globalThis.Tesseract;
        if (T && T.recognize) {
          if (!window.Tesseract) window.Tesseract = T;
          console.log("[TruthLens] Tesseract.js loaded from CDN");
          return true;
        }
      } catch (_) {}
      return false;
    },
  ];

  for (const load of loaders) {
    try {
      if (await load()) return true;
    } catch (_) {}
  }

  console.warn("[TruthLens] Tesseract.js not available. Frames will be sent for server-side OCR.");
  return false;
}

/**
 * Send images and extracted text to backend for analysis
 */
async function sendImageAndTextForAnalysis() {
  try {
    const backendUrl = await getBackendUrl();
    const apiUrl = `${backendUrl}/api/image/analyze-claim`;

    console.log(`[TruthLens] Sending ${TLX_IMAGE_STATE.extractedFrames.length} frames to: ${apiUrl}`);

    const formData = new FormData();
    formData.append('request_id', TLX_IMAGE_STATE.requestId);
    formData.append('claim_id', TLX_IMAGE_STATE.claimId);
    formData.append('frame_count', TLX_IMAGE_STATE.extractedFrames.length);
    formData.append('audio_analysis', JSON.stringify(TLX_IMAGE_STATE.audioAnalysis));
    formData.append('captured_text', JSON.stringify(TLX_IMAGE_STATE.extractedText));

    // Append frame images
    TLX_IMAGE_STATE.extractedFrames.forEach((frame, index) => {
      if (frame.blob) {
        formData.append(`frame_${index}`, frame.blob, `frame_${index}.jpg`);
      }
      if (frame.text) {
        formData.append(`frame_${index}_text`, frame.text);
      }
    });

    const response = await fetch(apiUrl, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    console.log("[TruthLens] Image analysis complete:", result);

    chrome.runtime.sendMessage({
      type: "IMAGE_ANALYSIS_COMPLETE",
      payload: result
    }).catch(e => console.log("Could not send to popup:", e));

    return result;

  } catch (error) {
    console.error("[TruthLens] Error sending analysis:", error);
    
    chrome.runtime.sendMessage({
      type: "IMAGE_ANALYSIS_ERROR",
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

// Initialize image capture on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initImageCapture);
} else {
  initImageCapture();
}
