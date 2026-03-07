from __future__ import annotations

import importlib.util
import io
import json
import logging
import queue
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..production_pipeline.pipeline_runner import PipelineRunner
from ..production_pipeline.middlewares.llm_blackbox import LLMBlackbox
from ..utils.analysis_store import add_analysis_record

router = APIRouter()
logger = logging.getLogger(__name__)


class _LineBufferedStdout:
    """Thread-safe stdout proxy: print each line to server console first, then enqueue for SSE to frontend."""

    def __init__(self, log_queue: queue.Queue, original_stdout: io.TextIOBase):
        self._queue = log_queue
        self._original = original_stdout
        self._buffer: list[str] = []
        self._lock = threading.Lock()

    def _emit_line(self, line: str) -> None:
        if not line:
            return
        if self._original:
            self._original.write(line + "\n")
            self._original.flush()
        self._queue.put(line)

    def write(self, s: str) -> int:
        with self._lock:
            for c in s:
                if c == "\n":
                    line = "".join(self._buffer).strip()
                    self._buffer.clear()
                    self._emit_line(line)
                else:
                    self._buffer.append(c)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                line = "".join(self._buffer).strip()
                self._buffer.clear()
                self._emit_line(line)
        if self._original:
            self._original.flush()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


# ----------------------------
# Dynamic step loader
# ----------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register to support module-level caches + avoid duplicate module instances.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_API_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _API_ROOT / "production_pipeline"


# ----------------------------
# Load Step 1
# ----------------------------
_step1 = _load_module(
    "step1_ingest_claim",
    _PIPELINE / "1_Ingest_claim" / "1_ingest_claim.py",
)
process_user_claim = _step1.process_user_claim
IngestClaimError = getattr(_step1, "IngestClaimError", Exception)


# ----------------------------
# Load Step 2
# ----------------------------
_step2 = _load_module(
    "step2_roberta_inference",
    _PIPELINE / "2_RoBERTa_inference" / "2_RoBERTa_inference.py",
)
process_step1_output = _step2.process_step1_output
RobertaInferenceError = getattr(_step2, "RobertaInferenceError", Exception)


# ----------------------------
# Load Step 3
# ----------------------------
_step3 = _load_module(
    "step3_routing_policy",
    _PIPELINE / "3_Routing_policy" / "3_routing_policy.py",
)
process_step2_output = _step3.process_step2_output
RoutingPolicyError = getattr(_step3, "RoutingPolicyError", Exception)


# ----------------------------
# Load Step 4
# ----------------------------
_step4 = _load_module(
    "step4_query_building",
    _PIPELINE / "4_Query_Building" / "4_query_building.py",
)
process_step3_output = _step4.process_step3_output
QueryBuildingError = getattr(_step4, "QueryBuildingError", Exception)


# ----------------------------
# Load Step 5
# ----------------------------
_step5 = _load_module(
    "step5_rag_retrieval",
    _PIPELINE / "5_RAG_retrieval" / "5_rag_retrieval.py",
)
process_step4_output = _step5.process_step4_output
RAGRetrievalError = getattr(_step5, "RAGRetrievalError", Exception)


# ----------------------------
# Load Step 6
# ----------------------------
_step6 = _load_module(
    "step6_mmr_selection",
    _PIPELINE / "6_MMR_selection" / "6_mmr_selection.py",
)
process_step5_output = _step6.process_step5_output
MMRSelectionError = getattr(_step6, "MMRSelectionError", Exception)


# ----------------------------
# Load Step 7
# ----------------------------
_step7 = _load_module(
    "step7_light_relevance_gate",
    _PIPELINE / "7_Light_relevance_gate" / "7_light_relevance_gate.py",
)
process_step6_output = _step7.process_step6_output
LightRelevanceGateError = getattr(_step7, "LightRelevanceGateError", Exception)


# ----------------------------
# Load Step 8
# ----------------------------
_step8 = _load_module(
    "step8_explainability_shap",
    _PIPELINE / "8_Explainability_SHAP" / "8_explainability_shap.py",
)
process_step7_output = _step8.process_step7_output
ExplainabilitySHAPError = getattr(_step8, "ExplainabilitySHAPError", Exception)


# ----------------------------
# Load Step 9 (optional)
# ----------------------------
class _LLMSummarizationErrorPlaceholder(Exception):
    """Placeholder when step 9 is not loaded; never raised."""

LLMSummarizationError = _LLMSummarizationErrorPlaceholder
_step9_path = _PIPELINE / "9_LLM_summarization" / "9_llm_summarization.py"
if _step9_path.exists():
    try:
        _step9 = _load_module("step9_llm_summarization", _step9_path)
        process_step8_output = _step9.process_step8_output
        LLMSummarizationError = getattr(_step9, "LLMSummarizationError", _LLMSummarizationErrorPlaceholder)
        _STEP9_AVAILABLE = True
    except Exception:
        process_step8_output = None
        _STEP9_AVAILABLE = False
else:
    process_step8_output = None
    _STEP9_AVAILABLE = False


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_verdict(value: object) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"TRUE", "FALSE", "MIXED"}:
        return raw
    if raw in {"REAL", "FACTUAL", "RELIABLE"}:
        return "TRUE"
    if raw in {"FAKE", "MISINFORMATION", "NOT_REAL"}:
        return "FALSE"
    return "MIXED"


def _extract_confidence_reasoning_verdict(out: dict) -> tuple[float, str, str]:
    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    roberta = out.get("roberta") if isinstance(out.get("roberta"), dict) else {}
    label = roberta.get("label") if isinstance(roberta.get("label"), dict) else {}

    confidence = _safe_float(summary.get("confidence"), default=-1.0)
    if confidence < 0:
        confidence = _safe_float(roberta.get("confidence"), default=-1.0)
    if confidence < 0:
        confidence = _safe_float(label.get("confidence"), default=0.0)

    reasoning = str(
        summary.get("conclusion")
        or summary.get("intro")
        or out.get("backend_log")
        or ""
    ).strip()

    verdict = _normalize_verdict(
        summary.get("verdict")
        or label.get("class_name")
        or roberta.get("prediction")
        or ""
    )
    return confidence, reasoning, verdict


def _should_persist(meta: dict) -> bool:
    explicit = meta.get("persist_result")
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str):
        return explicit.strip().lower() in {"1", "true", "yes", "y"}
    return str(meta.get("source_context") or "").strip().lower() == "chrome_extension"


def _build_analysis_record(meta: dict, out: dict, user_claim: str) -> dict:
    confidence, reasoning, verdict = _extract_confidence_reasoning_verdict(out)
    return {
        "session_id": meta.get("session_id") or "unknown-session",
        "input_type": meta.get("input_type") or "text",
        "input_text": meta.get("input_text") or user_claim,
        "transcript": meta.get("transcript") or "",
        "page_url": meta.get("page_url") or "",
        "analysis_result": out,
        "confidence": confidence,
        "reasoning": reasoning,
        "verdict": verdict,
        "timestamp": meta.get("timestamp"),
        "source_context": meta.get("source_context") or "unknown",
    }


async def _persist_analysis_if_requested(meta: dict, out: object, user_claim: str) -> None:
    if not _should_persist(meta) or not isinstance(out, dict):
        return
    record = _build_analysis_record(meta, out, user_claim)
    try:
        await run_in_threadpool(add_analysis_record, record)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist /api/process result for source=%s", meta.get("source_context"))


@router.post("/process")
async def process(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_REQUEST", "message": "Request body must be a JSON object"},
        )
    user_claim = body.get("user_claim")
    metadata = body

    if not isinstance(user_claim, str) or not user_claim.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_REQUEST", "message": "user_claim must be a non-empty string"},
        )

    log_buffer = io.StringIO()
    try:
        with redirect_stdout(log_buffer):
            runner = PipelineRunner()
            out = await runner.run(text_input=user_claim.strip(), request_id=request_id)

        if isinstance(out, dict) and out.get("error"):
            raise HTTPException(
                status_code=400,
                detail={"code": "PIPELINE_ERROR", "message": str(out["error"])},
            )

        backend_log = (log_buffer.getvalue() or "").strip()
        if isinstance(out, dict):
            out = _merge_runner_result(out)
            out["backend_log"] = backend_log
        await _persist_analysis_if_requested(metadata, out, user_claim)
        return out

    except IngestClaimError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "INGEST_ERROR"), "message": getattr(e, "message", str(e))})

    except RobertaInferenceError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "ROBERTA_ERROR"), "message": getattr(e, "message", str(e))})

    except RoutingPolicyError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "ROUTING_ERROR"), "message": getattr(e, "message", str(e))})

    except QueryBuildingError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "QUERY_BUILDING_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except RAGRetrievalError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "RAG_RETRIEVAL_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except MMRSelectionError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "MMR_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except LightRelevanceGateError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "RELEVANCE_GATE_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except ExplainabilitySHAPError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "SHAP_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})

    except LLMSummarizationError as e:
        raise HTTPException(status_code=400, detail={"code": getattr(e, "code", "SUMMARY_ERROR"), "message": getattr(e, "message", str(e)), "details": getattr(e, "details", None)})


async def _run_pipeline_async(user_claim: str):
    """Run the same pipeline as process(); used by streaming endpoint. Raises on error."""
    step1_out = await run_in_threadpool(process_user_claim, user_claim)
    step2_out = await run_in_threadpool(process_step1_output, step1_out)
    step3_out = await run_in_threadpool(process_step2_output, step2_out)
    step4_out = await process_step3_output(step3_out)
    step5_out = await process_step4_output(step4_out)
    step6_out = await process_step5_output(step5_out)
    step7_out = await process_step6_output(step6_out)
    step8_out = await process_step7_output(step7_out)
    if _STEP9_AVAILABLE and process_step8_output is not None:
        step9_out = await process_step8_output(step8_out)
        return step9_out
    return step8_out


def _run_pipeline_in_thread(
    user_claim: str,
    log_queue: queue.Queue,
    proxy: _LineBufferedStdout,
    result_holder: list,
    exc_holder: list,
) -> None:
    """Run the pipeline in a dedicated thread so the async generator can drain
    the log queue in real time without being blocked by the pipeline."""
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with redirect_stdout(proxy):
            result_holder.append(loop.run_until_complete(_run_pipeline_async(user_claim)))
    except Exception as e:
        exc_holder.append(e)
    finally:
        loop.close()


async def _stream_process_generator(user_claim: str, metadata: dict):
    import asyncio

    log_queue: queue.Queue = queue.Queue()
    proxy = _LineBufferedStdout(log_queue, getattr(sys, "stdout", None))
    collected: list[str] = []
    result_holder: list = []
    exc_holder: list = []

    thread = threading.Thread(
        target=_run_pipeline_in_thread,
        args=(user_claim, log_queue, proxy, result_holder, exc_holder),
        daemon=True,
    )
    thread.start()

    # Drain log queue in real time and yield each line so the client updates live
    loop = asyncio.get_event_loop()
    while thread.is_alive() or not log_queue.empty():
        try:
            line = await loop.run_in_executor(
                None, lambda: log_queue.get(timeout=0.15)
            )
            collected.append(line)
            yield f"data: {json.dumps({'log': line})}\n\n"
        except queue.Empty:
            await asyncio.sleep(0.05)

    # Drain any remaining lines
    while True:
        try:
            line = log_queue.get_nowait()
            collected.append(line)
            yield f"data: {json.dumps({'log': line})}\n\n"
        except queue.Empty:
            break

    if exc_holder:
        yield f"event: error\ndata: {json.dumps({'error': str(exc_holder[0])})}\n\n"
        return

    result = result_holder[0] if result_holder else None
    backend_log = "\n".join(collected)
    if isinstance(result, dict):
        result = _merge_runner_result(result)
        result["backend_log"] = backend_log
    await _persist_analysis_if_requested(metadata, result, user_claim)
    yield f"event: result\ndata: {json.dumps(result)}\n\n"


@router.post("/process-stream")
async def process_stream(request: Request):
    """Stream backend log lines as SSE, then send final result. For real-time log in the extension."""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_REQUEST", "message": "Request body must be a JSON object"},
        )
    user_claim = body.get("user_claim")
    metadata = body
    if not isinstance(user_claim, str) or not user_claim.strip():
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_REQUEST", "message": "user_claim must be a non-empty string"},
        )
    return StreamingResponse(
        _stream_process_generator(user_claim.strip(), metadata),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class LLMTestRequest(BaseModel):
    user_context: str


@router.post("/llm_test")
async def llm_test(request: LLMTestRequest):
    system_context = "You are a helpful assistant. Answer in 1 or 2 sentences."
    user_context = request.user_context

    if not isinstance(user_context, str) or not user_context.strip():
        raise HTTPException(status_code=400, detail="user_context must be provided")

    try:
        llm = LLMBlackbox()
        result = await llm.generate_async(system_context=system_context, user_context=user_context)
        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
