/**
 * AudioWorklet: captures mono float32 at context sample rate,
 * resamples to 16 kHz and sends Int16 PCM to main thread via postMessage.
 * Main thread forwards to backend WebSocket for live-transcribe.
 */
const TARGET_SAMPLE_RATE = 16000;

let inputSampleRate = 44100;
const inputBuffer = [];
let inputDuration = 0;

function resampleTo16k() {
  if (inputBuffer.length < 2) return null;
  const outLength = Math.floor((inputDuration * TARGET_SAMPLE_RATE));
  if (outLength <= 0) return null;
  const ratio = (inputBuffer.length - 1) / Math.max(1, outLength - 1);
  const int16 = new Int16Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIdx = i * ratio;
    const i0 = Math.floor(srcIdx);
    const i1 = Math.min(i0 + 1, inputBuffer.length - 1);
    const t = srcIdx - i0;
    const v = inputBuffer[i0] * (1 - t) + inputBuffer[i1] * t;
    const s = Math.max(-1, Math.min(1, v));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16.buffer;
}

class CaptureWorkletProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    inputSampleRate = options.processorOptions?.sampleRate || 44100;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input.length) return true;
    const ch = input[0];
    if (!ch || ch.length === 0) return true;
    for (let i = 0; i < ch.length; i++) {
      inputBuffer.push(ch[i]);
    }
    inputDuration = inputBuffer.length / inputSampleRate;
    const chunkDuration = 0.1;
    if (inputDuration >= chunkDuration) {
      const out = resampleTo16k();
      if (out) {
        this.port.postMessage(out, [out]);
        const toRemove = Math.floor(chunkDuration * inputSampleRate);
        inputBuffer.splice(0, toRemove);
        inputDuration = inputBuffer.length / inputSampleRate;
      }
    }
    return true;
  }
}

registerProcessor("capture-worklet-processor", CaptureWorkletProcessor);
