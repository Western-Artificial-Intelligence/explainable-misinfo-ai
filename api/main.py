# Import FastAPI framework
import os  # noqa: F401
import warnings
from pathlib import Path
import logging

# Suppress NumPy MINGW-W64 warnings on Windows (experimental float128 path).
# To fix properly: use a NumPy wheel built with MSVC, e.g.:
#   pip install --force-reinstall --only-binary :all: "numpy>=1.24,<2"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core.getlimits")
warnings.filterwarnings("ignore", message=".*MINGW-W64.*")

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Load repo env files before importing routes so pipeline steps see runtime config.
_repo_root = Path(__file__).resolve().parent.parent
_env_path = _repo_root / ".env"
_env_local_path = _repo_root / ".env.local"
if _env_path.is_file():
    load_dotenv(_env_path, override=False)
if _env_local_path.is_file():
    load_dotenv(_env_local_path, override=True)
if not _env_path.is_file() and not _env_local_path.is_file():
    load_dotenv()

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .routes import analysis, classify, documents, health, history
from .utils.history_store import get_store_mode

load_dotenv()
load_dotenv(".env.local", override=True)  # Local overrides (e.g. API keys)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(title="TruthLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.include_router(health.router)
app.include_router(classify.router)
app.include_router(documents.router)
app.include_router(history.router)
app.include_router(analysis.router)


@app.on_event("startup")
def startup_checks() -> None:
    mode = get_store_mode()
    logger.info("History storage mode: %s", mode)

# Audio extraction and voice-to-text route
try:
    from .routes import audio
    app.include_router(audio.router)
except (ModuleNotFoundError, ImportError) as exc:
    logger.warning(
        "Skipping /api/audio routes because dependency is missing: %s. "
        "Install optional audio deps to enable it.",
        getattr(exc, "name", exc),
    )

# Live ASR WebSocket (faster-whisper streaming) — registered inline to avoid
# function-based registration timing issues with Starlette's middleware stack.
from fastapi import WebSocket as _WS
from starlette.websockets import WebSocketDisconnect as _WSD
import asyncio as _asyncio, struct as _struct, logging as _logging
_asr_logger = _logging.getLogger("api.utils.live_asr_ws")

@app.websocket("/api/live-asr")
async def live_asr_endpoint(websocket: _WS):
    _asr_logger.info("[live_asr_ws] WebSocket connection attempt")
    from .utils.live_asr_ws import _get_model, WINDOW_S, DECODE_EVERY_S, SAMPLE_RATE, BYTES_PER_SAMPLE
    await websocket.accept()
    _asr_logger.info("[live_asr_ws] WebSocket accepted")
    model = _get_model()
    if model is None:
        await websocket.send_json({"error": "faster-whisper not available. Run: pip install faster-whisper"})
        await websocket.close()
        return

    buffer: list = []
    window_samples = int(SAMPLE_RATE * WINDOW_S)
    total_received: int = 0    # total PCM samples received (monotonically increasing)
    committed_abs: int = 0     # absolute sample index of last committed word's end
    STABILITY_MARGIN_S = 1.5  # words ending before this many secs from buffer end are stable

    def bytes_to_float32(pcm_bytes: bytes):
        n = len(pcm_bytes) // BYTES_PER_SAMPLE
        ints = _struct.unpack_from(f"<{n}h", pcm_bytes)
        return [x / 32768.0 for x in ints]

    chunks_received = 0

    async def receiver():
        nonlocal chunks_received, total_received
        try:
            while True:
                data = await websocket.receive_bytes()
                floats = bytes_to_float32(data)
                buffer.extend(floats)
                total_received += len(floats)
                chunks_received += 1
                if chunks_received % 50 == 0:
                    _asr_logger.info("[live_asr_ws] received %d chunks, buffer=%.1fs",
                                     chunks_received, len(buffer) / SAMPLE_RATE)
                if len(buffer) > window_samples:
                    buffer[:] = buffer[-window_samples:]
        except _WSD:
            pass
        except Exception as e:
            _asr_logger.debug("receiver: %s", e)

    async def sender():
        nonlocal committed_abs
        loop = _asyncio.get_running_loop()
        while True:
            await _asyncio.sleep(DECODE_EVERY_S)
            if len(buffer) < int(SAMPLE_RATE * 0.5):
                continue
            try:
                import numpy as np
                arr = np.array(buffer, dtype=np.float32)
                buffer_start_abs = total_received - len(arr)   # absolute sample of buffer[0]
                buffer_duration = len(arr) / SAMPLE_RATE
                stable_cutoff = buffer_duration - STABILITY_MARGIN_S

                def _run():
                    segs, _ = model.transcribe(
                        arr, language="en",
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300),
                    )
                    return [w for seg in segs if seg.words for w in seg.words]

                words = await loop.run_in_executor(None, _run)

                # Convert committed absolute position to time within this buffer snapshot
                committed_in_buffer_s = max(0.0, (committed_abs - buffer_start_abs) / SAMPLE_RATE)

                # New words: start at or after committed position AND end before stability cutoff
                new_words = [w for w in words
                             if w.start >= committed_in_buffer_s - 0.05   # 50ms overlap for boundary safety
                             and w.end <= stable_cutoff]
                if not new_words:
                    continue

                delta = " ".join(w.word.strip() for w in new_words)
                committed_abs = buffer_start_abs + int(new_words[-1].end * SAMPLE_RATE)

                _asr_logger.info("[live_asr_ws] emitting delta: %r", delta[:80])
                try:
                    await websocket.send_json({"word": delta})
                except Exception:
                    break
            except Exception as e:
                _asr_logger.debug("sender: %s", e)

    try:
        await _asyncio.gather(receiver(), sender())
    except (_WSD, _asyncio.CancelledError):
        pass
    except Exception as e:
        _asr_logger.debug("live_asr_endpoint: %s", e)
    try:
        await websocket.close()
    except Exception:
        pass

# Image processing and OCR route
try:
    from .routes import imageprocessing
    app.include_router(imageprocessing.router)
except ModuleNotFoundError as exc:
    logger.warning(
        "Skipping /api/image routes because dependency is missing: %s. "
        "Install optional image processing deps to enable it.",
        exc.name,
    )

# Unified production pipeline route
try:
    from .routes import pipeline
    app.include_router(pipeline.router)
except ModuleNotFoundError as exc:
    logger.warning(
        "Skipping /api/pipeline routes because dependency is missing: %s. "
        "Install optional deps to enable it.",
        exc.name,
    )

# Production route is optional because it relies on extra dependencies.
try:
    from .routes import production
except ModuleNotFoundError as exc:
    logger.warning(
        "Skipping /api production routes because dependency is missing: %s. "
        "Install optional deps to enable it.",
        exc.name,
    )
else:
    app.include_router(production.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "TruthLens backend running!"}


@app.websocket("/ws-test")
async def ws_test(websocket: WebSocket):
    logger.info("[ws-test] connection attempt")
    await websocket.accept()
    logger.info("[ws-test] accepted")
    await websocket.send_text("ok")
    await websocket.close()
