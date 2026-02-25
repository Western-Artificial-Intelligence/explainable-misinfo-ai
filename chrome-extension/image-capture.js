/**
 * Image & Text Capture Module for TikTok/YouTube Shorts
 * 
 * This module handles:
 * 1. Detecting if video has minimal/no voice
 * 2. Capturing frames from video elements
 * 3. Extracting text via OCR (Tesseract.js)
 * 4. Capturing on-screen text overlays
 * 5. Sending image+text to backend for analysis
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
  captureInterval: null
};

/**
 * Analyze audio levels to determine if video has meaningful voice content
 * Returns true if audio is minimal/no voice detected
 */
async function analyzeAudioLevels(videoElement) {
  try {
    // Create audio context
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;

    const source = audioContext.createMediaElementAudioSource(videoElement);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    // Get frequency data
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    
    return new Promise((resolve) => {
      // Sample audio analysis over a brief period
      let samples = 0;
      let totalEnergy = 0;
      const sampleInterval = setInterval(() => {
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate energy across frequency bands
        let energy = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        totalEnergy += energy;
        samples++;

        // After 5 samples (~500ms), determine if voice exists
        if (samples >= 5) {
          clearInterval(sampleInterval);
          audioContext.close();
          
          const avgEnergy = totalEnergy / samples;
          const hasNoVoice = avgEnergy < 30; // Threshold for minimal audio
          
          console.log(`[TruthLens] Audio analysis - Avg energy: ${avgEnergy.toFixed(2)}, Has voice: ${!hasNoVoice}`);
          
          resolve({
            hasVoice: !hasNoVoice,
            averageEnergy: avgEnergy,
            shouldCaptureImage: hasNoVoice
          });
        }
      }, 100);
    });
  } catch (error) {
    console.warn("[TruthLens] Audio analysis failed, defaulting to image capture:", error);
    return { hasVoice: false, shouldCaptureImage: true };
  }
}

/**
 * Initialize image capture listeners
 */
function initImageCapture() {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "START_IMAGE_CAPTURE") {
      startImageCapture(message.payload || {})
        .then(result => sendResponse({ ok: true, result }))
        .catch(error => sendResponse({ ok: false, error: error.message }));
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

    const audioAnalysis = await analyzeAudioLevels(videoElement);
    
    return {
      shouldCapture: audioAnalysis.shouldCaptureImage,
      audioAnalysis: audioAnalysis,
      reason: audioAnalysis.shouldCaptureImage ? "Minimal audio detected" : "Voice content detected"
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
    if (TLX_IMAGE_STATE.isCapturing) {
      throw new Error("Image capture already in progress");
    }

    // Generate request and claim IDs
    const requestId = `req_img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const claimId = `claim_${Date.now()}`;

    TLX_IMAGE_STATE.requestId = requestId;
    TLX_IMAGE_STATE.claimId = claimId;
    TLX_IMAGE_STATE.isCapturing = true;
    TLX_IMAGE_STATE.extractedFrames = [];
    TLX_IMAGE_STATE.extractedText = [];

    console.log(`[TruthLens] Starting image capture: ${requestId}`);

    // Find video element
    const videoElement = findVideoElement();
    if (!videoElement) {
      throw new Error("No video element found on page");
    }

    TLX_IMAGE_STATE.currentVideoElement = videoElement;

    // Analyze audio to confirm we should use image capture
    const audioAnalysis = await analyzeAudioLevels(videoElement);
    TLX_IMAGE_STATE.audioAnalysis = audioAnalysis;

    if (!audioAnalysis.shouldCaptureImage && !payload.forceCapture) {
      throw new Error("Video has voice content - use audio capture instead");
    }

    // Ensure Tesseract.js is loaded
    await ensureTesseractLoaded();

    // Start frame capture
    startFrameCapture(videoElement);

    // Extract on-screen text elements
    await extractOnScreenText();

    return {
      requestId,
      claimId,
      audioAnalysis: audioAnalysis,
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

    // Stop frame capture
    if (TLX_IMAGE_STATE.captureInterval) {
      clearInterval(TLX_IMAGE_STATE.captureInterval);
      TLX_IMAGE_STATE.captureInterval = null;
    }

    TLX_IMAGE_STATE.isCapturing = false;

    // Send captured data to backend
    if (TLX_IMAGE_STATE.extractedFrames.length > 0 || TLX_IMAGE_STATE.extractedText.length > 0) {
      await sendImageAndTextForAnalysis();
    }

    return { message: "Image capture stopped" };

  } catch (error) {
    console.error("[TruthLens] Error stopping image capture:", error);
    throw error;
  }
}

/**
 * Capture frames from video at regular intervals
 */
function startFrameCapture(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth || 640;
  canvas.height = videoElement.videoHeight || 360;
  TLX_IMAGE_STATE.canvasContext = canvas.getContext('2d');

  // Capture every 500ms (can be adjusted)
  TLX_IMAGE_STATE.captureInterval = setInterval(async () => {
    try {
      if (!TLX_IMAGE_STATE.isCapturing) return;

      // Draw current frame to canvas
      TLX_IMAGE_STATE.canvasContext.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

      // Convert to blob
      canvas.toBlob(async (blob) => {
        if (blob && TLX_IMAGE_STATE.extractedFrames.length < 20) { // Limit to 20 frames
          const frameData = {
            timestamp: Date.now(),
            blob: blob,
            text: null // Will be populated by OCR
          };

          // Run OCR on frame
          try {
            const ocrText = await performOCROnFrame(blob);
            frameData.text = ocrText;
            if (ocrText && ocrText.trim().length > 0) {
              TLX_IMAGE_STATE.extractedText.push(ocrText);
            }
          } catch (ocrError) {
            console.warn("[TruthLens] OCR error on frame:", ocrError);
          }

          TLX_IMAGE_STATE.extractedFrames.push(frameData);
          console.log(`[TruthLens] Captured frame ${TLX_IMAGE_STATE.extractedFrames.length}, OCR text length: ${frameData.text?.length || 0}`);
        }
      }, 'image/jpeg', 0.7);

    } catch (error) {
      console.error("[TruthLens] Error capturing frame:", error);
    }
  }, 500);

  console.log("[TruthLens] Frame capture started");
}

/**
 * Perform OCR on a single frame using Tesseract.js
 */
async function performOCROnFrame(imageBlob) {
  try {
    if (!window.Tesseract) {
      throw new Error("Tesseract not loaded");
    }

    const { data } = await window.Tesseract.recognize(imageBlob, 'eng', {
      logger: msg => console.log(`[TruthLens] OCR progress:`, msg)
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
      'p, span, div, h1, h2, h3, h4, h5, h6, [data-caption], .caption, .subtitle, .text-overlay'
    );

    const extractedTexts = new Set();

    textElements.forEach(element => {
      const text = element.textContent?.trim();
      if (text && text.length > 0 && text.length < 500) {
        // Filter out common non-content text
        if (!isCommonUIText(text)) {
          extractedTexts.add(text);
        }
      }
    });

    // Add extracted texts to state
    extractedTexts.forEach(text => {
      if (!TLX_IMAGE_STATE.extractedText.includes(text)) {
        TLX_IMAGE_STATE.extractedText.push(text);
      }
    });

    console.log(`[TruthLens] Extracted ${extractedTexts.size} unique text elements from DOM`);

  } catch (error) {
    console.error("[TruthLens] Error extracting on-screen text:", error);
  }
}

/**
 * Filter out common UI text that's not content
 */
function isCommonUIText(text) {
  const commonTexts = [
    'like', 'comment', 'share', 'follow', 'subscribe',
    'menu', 'settings', 'profile', 'home', 'trending',
    'search', 'notifications', 'messages', 'explore'
  ];
  return commonTexts.some(common => text.toLowerCase().includes(common));
}

/**
 * Find the video element (same as audio capture)
 */
function findVideoElement() {
  const videoElements = document.querySelectorAll('video');
  
  if (videoElements.length === 0) {
    return null;
  }

  let mostVisibleVideo = null;
  let maxVisibleArea = 0;

  videoElements.forEach(video => {
    const rect = video.getBoundingClientRect();
    const visibleArea = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(0, rect.top)) *
                       Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(0, rect.left));

    if (visibleArea > maxVisibleArea) {
      maxVisibleArea = visibleArea;
      mostVisibleVideo = video;
    }
  });

  return mostVisibleVideo;
}

/**
 * Ensure Tesseract.js library is loaded
 */
async function ensureTesseractLoaded() {
  return new Promise((resolve, reject) => {
    if (window.Tesseract) {
      resolve();
      return;
    }

    // Load Tesseract.js from CDN
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/tesseract.js@5.0.2/dist/tesseract.min.js';
    script.onload = () => {
      console.log("[TruthLens] Tesseract.js loaded");
      resolve();
    };
    script.onerror = () => {
      console.error("[TruthLens] Failed to load Tesseract.js");
      reject(new Error("Failed to load Tesseract.js"));
    };
    document.head.appendChild(script);
  });
}

/**
 * Send images and extracted text to backend for analysis
 */
async function sendImageAndTextForAnalysis() {
  try {
    const backendUrl = await getBackendUrl();
    const apiUrl = `${backendUrl}/api/image/analyze-claim`;

    console.log(`[TruthLens] Sending image+text analysis to: ${apiUrl}`);

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
    console.log("[TruthLens] Image analysis result:", result);

    // Send result to popup
    chrome.runtime.sendMessage({
      type: "IMAGE_ANALYSIS_COMPLETE",
      payload: result
    }).catch(e => console.log("Could not send to popup:", e));

    return result;

  } catch (error) {
    console.error("[TruthLens] Error sending image/text for analysis:", error);
    
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
    chrome.storage.sync.get({ backendUrl: 'http://localhost:8000' }, (items) => {
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
