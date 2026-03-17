"""
Live ASR WebSocket: stream 16 kHz mono PCM from the extension and stream back
partial + confirmed transcript using faster-whisper. No production_pipeline dependency.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)

# Optional: faster-whisper for live decode
_faster_whisper = None

try:
    from faster_whisper import WhisperModel

    _faster_whisper = "WhisperModel"
except ImportError:
    pass

# Defaults: decode every 300ms; confirm after 3 stable decodes (~1s). 8s rolling window.
WINDOW_S = 8.0
DECODE_EVERY_S = 0.3
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def _get_model():
    if _faster_whisper is None:
        return None
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        logger.warning("faster_whisper model load failed: %s", e)
        return None


def _build_live_transcribe_handler(path: str, register_fn):
    """Internal helper to register the live-transcribe WebSocket on a router or app.

    register_fn is either router.websocket or app.websocket.
    """
    from fastapi import WebSocket, WebSocketDisconnect

    logger.info("[live_asr_ws] Registering WebSocket handler at path: %s", path)

    @register_fn(path)
    async def live_transcribe(websocket: WebSocket):
        logger.info("[live_asr_ws] WebSocket connection attempt from %s | headers: %s",
                    websocket.client, dict(websocket.headers))
        await websocket.accept()
        logger.info("[live_asr_ws] WebSocket accepted")
        model = _get_model()
        logger.info("[live_asr_ws] Model loaded: %s", model)
        if model is None:
            try:
                await websocket.send_json({
                    "error": "Live transcription requires faster-whisper. Install with: pip install faster-whisper",
                    "partial": "",
                    "confirmed": "",
                })
            except Exception:
                pass
            await websocket.close()
            return

        buffer = []
        buffer_duration_s = 0.0
        last_confirmed = ""
        decode_interval = DECODE_EVERY_S
        window_samples = int(SAMPLE_RATE * WINDOW_S)

        def bytes_to_float32(pcm_bytes: bytes):
            n = len(pcm_bytes) // BYTES_PER_SAMPLE
            ints = struct.unpack_from(f"<{n}h", pcm_bytes)
            return [x / 32768.0 for x in ints]

        async def receiver():
            nonlocal buffer_duration_s
            try:
                while True:
                    data = await websocket.receive_bytes()
                    floats = bytes_to_float32(data)
                    buffer.extend(floats)
                    buffer_duration_s += len(floats) / SAMPLE_RATE
                    # Keep only last WINDOW_S
                    if len(buffer) > window_samples:
                        buffer[:] = buffer[-window_samples:]
                        buffer_duration_s = len(buffer) / SAMPLE_RATE
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.debug("live_transcribe receiver: %s", e)

        stable_count = 0
        last_partial_text = ""

        async def sender():
            nonlocal last_confirmed, last_partial_text, stable_count, buffer
            while True:
                await asyncio.sleep(decode_interval)
                if not buffer or len(buffer) < int(SAMPLE_RATE * 0.5):
                    continue
                try:
                    import numpy as np
                    arr = np.array(buffer, dtype=np.float32)
                    def _run():
                        segs, _ = model.transcribe(
                            arr, language="en", vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=300),
                        )
                        return " ".join(s.text for s in segs).strip()
                    loop = asyncio.get_running_loop()
                    text = await loop.run_in_executor(None, _run)
                    if text != last_partial_text:
                        last_partial_text = text
                        stable_count = 0
                    else:
                        stable_count += 1
                    partial = text
                    if stable_count >= 3:
                        last_confirmed = text
                    try:
                        await websocket.send_json({
                            "partial": partial,
                            "confirmed": last_confirmed,
                        })
                    except Exception:
                        break
                except Exception as e:
                    logger.debug("live_transcribe decode: %s", e)

        try:
            await asyncio.gather(receiver(), sender())
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.debug("live_transcribe: %s", e)
        try:
            await websocket.close()
        except Exception:
            pass


def register_live_transcribe_ws(router):
    """Register WebSocket route /live-transcribe on the given router (prefix /api/audio)."""
    _build_live_transcribe_handler("/live-transcribe", router.websocket)


def register_live_transcribe_app(app):
    """Register WebSocket route /api/live-asr directly on the FastAPI app."""
    _build_live_transcribe_handler("/api/live-asr", app.websocket)
