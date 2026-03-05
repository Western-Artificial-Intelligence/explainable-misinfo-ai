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

/**
 * Initialize audio capture listeners
 * Called when extension loads on TikTok
 */
function initAudioCapture() {
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
 * Stop capturing audio and send for transcription
 */
async function stopAudioCapture() {
  try {
    if (!TLX_AUDIO_STATE.isCapturing) {
      throw new Error("Audio capture not in progress");
    }

    console.log("[TruthLens] Stopping audio capture");

    // Stop audio stream
    if (TLX_AUDIO_STATE.mediaStream) {
      TLX_AUDIO_STATE.mediaStream.getTracks().forEach(track => track.stop());
      TLX_AUDIO_STATE.mediaStream = null;
    }

    // Close audio context
    if (TLX_AUDIO_STATE.audioContext) {
      await TLX_AUDIO_STATE.audioContext.close();
      TLX_AUDIO_STATE.audioContext = null;
    }

    TLX_AUDIO_STATE.isCapturing = false;

    if (TLX_AUDIO_STATE.audioChunks.length > 0) {
      const audioBlob = createWavBlob(TLX_AUDIO_STATE.audioChunks);
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
    throw error;
  }
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
 */
function findVideoElement() {
  // TikTok uses video elements in their feed
  const videoElements = document.querySelectorAll('video');
  
  if (videoElements.length === 0) {
    return null;
  }

  // Find the most visible video (usually the one in focus)
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
 * Capture audio from video element
 * Uses Web Audio API to extract audio stream
 */
async function captureAudioFromVideo(videoElement) {
  try {
    // Create audio context
    TLX_AUDIO_STATE.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    TLX_AUDIO_STATE.sampleRate = TLX_AUDIO_STATE.audioContext.sampleRate;

    // Create media element audio source
    const source = TLX_AUDIO_STATE.audioContext.createMediaElementAudioSource(videoElement);

    // Create script processor for audio capture
    const sampleRate = TLX_AUDIO_STATE.audioContext.sampleRate;
    const bufferSize = 4096;
    
    TLX_AUDIO_STATE.scriptProcessor = TLX_AUDIO_STATE.audioContext.createScriptProcessor(
      bufferSize,
      1, // input channels
      1  // output channels
    );

    // Collect audio data
    TLX_AUDIO_STATE.scriptProcessor.onaudioprocess = (event) => {
      const inputData = event.inputBuffer.getChannelData(0);
      
      // Convert float32 to PCM16
      const pcmData = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) < 0
          ? inputData[i] * 0x8000
          : inputData[i] * 0x7fff;
      }

      TLX_AUDIO_STATE.audioChunks.push(pcmData);
    };

    // Connect nodes
    source.connect(TLX_AUDIO_STATE.scriptProcessor);
    TLX_AUDIO_STATE.scriptProcessor.connect(TLX_AUDIO_STATE.audioContext.destination);

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
    chrome.storage.sync.get({ backendUrl: 'http://localhost:8000' }, (items) => {
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
