"""
Unified Pipeline Runner

Coordinates the entire end-to-end pipeline:
0.  Audio Extraction -> Text Transcription
0b. LLM Claim Extraction (for media-sourced text)
1.  Ingest & Normalize Claim
2.  RoBERTa Inference (Claim Classification)
3.  Routing Policy (Determine analysis path)
4.  Query Building
5.  RAG Retrieval
6.  MMR Selection
7.  Light Relevance Gate
8.  Explainability (SHAP)

Usage:
    runner = PipelineRunner()
    result = await runner.run(
        media_source="...",
        media_type="video",
        request_id="req_123"
    )
"""

import asyncio
import importlib
import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve the production_pipeline directory once
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helper: load a stage module by file path (handles numeric dir names)
# ---------------------------------------------------------------------------

def _load_stage_module(name: str, filepath: Path):
    """Load a Python module from *filepath* and register it under *name*."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load stage module from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Stage 0 – Audio extraction (optional, needs whisper)
# ---------------------------------------------------------------------------
try:
    _audio_mod = importlib.import_module(
        "api.production_pipeline.0_Audio_Extraction_VoiceToText"
    )
    AudioExtractionPipeline = _audio_mod.AudioExtractionPipeline
    AUDIO_EXTRACTION_AVAILABLE = True
except (ImportError, AttributeError):
    logger.warning("Audio extraction pipeline not available")
    AUDIO_EXTRACTION_AVAILABLE = False

# ---------------------------------------------------------------------------
# Real model inference (optional, needs checkpoint)
# ---------------------------------------------------------------------------
try:
    from api.services.model_inference import is_model_available, predict as model_predict
    REAL_MODEL_AVAILABLE = True
except ImportError:
    logger.warning("model_inference service not available")
    REAL_MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# LLM claim extraction (optional, needs Ollama)
# ---------------------------------------------------------------------------
try:
    from api.services.claim_extractor import extract_claim
    CLAIM_EXTRACTOR_AVAILABLE = True
except ImportError:
    logger.warning("claim_extractor service not available")
    CLAIM_EXTRACTOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Load stage modules 1-8 via importlib (same modules production.py uses)
# Each exposes a process_* function that takes previous stage output.
# ---------------------------------------------------------------------------
_STAGES_LOADED = False

# Stage functions (populated by _ensure_stages_loaded)
_process_user_claim = None       # Stage 1: sync
_process_step1_output = None     # Stage 2: sync
_process_step2_output = None     # Stage 3: sync
_process_step3_output = None     # Stage 4: async
_process_step4_output = None     # Stage 5: async
_process_step5_output = None     # Stage 6: async
_process_step6_output = None     # Stage 7: async
_process_step7_output = None     # Stage 8: async
_process_step8_output = None     # Stage 9: async


def _ensure_stages_loaded():
    """Lazily load all stage modules. Called once on first pipeline run."""
    global _STAGES_LOADED
    global _process_user_claim, _process_step1_output, _process_step2_output
    global _process_step3_output, _process_step4_output, _process_step5_output
    global _process_step6_output, _process_step7_output, _process_step8_output

    if _STAGES_LOADED:
        return

    stage_defs = [
        ("step1_ingest_claim",        "1_Ingest_claim",          "1_ingest_claim.py"),
        ("step2_roberta_inference",   "2_RoBERTa_inference",     "2_RoBERTa_inference.py"),
        ("step3_routing_policy",      "3_Routing_policy",        "3_routing_policy.py"),
        ("step4_query_building",      "4_Query_Building",        "4_query_building.py"),
        ("step5_rag_retrieval",       "5_RAG_retrieval",         "5_rag_retrieval.py"),
        ("step6_mmr_selection",       "6_MMR_selection",         "6_mmr_selection.py"),
        ("step7_light_relevance_gate","7_Light_relevance_gate",  "7_light_relevance_gate.py"),
        ("step8_explainability_shap", "8_Explainability_SHAP",   "8_explainability_shap.py"),
        ("step9_llm_summarization",   "9_LLM_summarization",     "9_llm_summarization.py"),
    ]

    modules = {}
    for mod_name, dirname, filename in stage_defs:
        path = _PIPELINE_DIR / dirname / filename
        if not path.exists():
            logger.warning("Stage module not found: %s", path)
            modules[mod_name] = None
            continue
        try:
            modules[mod_name] = _load_stage_module(mod_name, path)
        except Exception:
            logger.warning("Failed to load stage module %s", mod_name, exc_info=True)
            modules[mod_name] = None

    def _get_fn(mod_name, fn_name):
        mod = modules.get(mod_name)
        return getattr(mod, fn_name, None) if mod else None

    _process_user_claim   = _get_fn("step1_ingest_claim",        "process_user_claim")
    _process_step1_output = _get_fn("step2_roberta_inference",   "process_step1_output")
    _process_step2_output = _get_fn("step3_routing_policy",      "process_step2_output")
    _process_step3_output = _get_fn("step4_query_building",      "process_step3_output")
    _process_step4_output = _get_fn("step5_rag_retrieval",       "process_step4_output")
    _process_step5_output = _get_fn("step6_mmr_selection",       "process_step5_output")
    _process_step6_output = _get_fn("step7_light_relevance_gate","process_step6_output")
    _process_step7_output = _get_fn("step8_explainability_shap", "process_step7_output")
    _process_step8_output = _get_fn("step9_llm_summarization",   "process_step8_output")

    _STAGES_LOADED = True
    logger.info("Pipeline stage modules loaded")


class PipelineRunner:
    """
    Runs the complete misinformation detection pipeline.

    Supports both:
    - Direct text input (existing flow)
    - Media input -> Audio extraction -> Text (new flow)
    """

    def __init__(self):
        self.request_id = None
        self.claim_id = None
        self.pipeline_stages = {}
        self.execution_log = []

        # Initialize audio extraction if available
        if AUDIO_EXTRACTION_AVAILABLE:
            try:
                self.audio_extractor = AudioExtractionPipeline(model_size="base")
                logger.info("Audio extraction pipeline initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio extraction: {e}")
                self.audio_extractor = None
        else:
            self.audio_extractor = None

    async def run(
        self,
        media_source: Optional[str] = None,
        media_type: Optional[str] = None,
        text_input: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the complete pipeline (async).

        Args:
            media_source: File path, URL, or media reference (for audio extraction)
            media_type: 'video', 'audio', 'url' (for audio extraction)
            text_input: Direct text input (skips audio extraction)
            request_id: Optional request identifier
            **kwargs: Additional options passed to steps

        Returns:
            Complete pipeline result with all stage outputs
        """
        _ensure_stages_loaded()

        self.request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        self.claim_id = f"claim_{uuid.uuid4().hex[:12]}"

        logger.info(f"Starting pipeline: {self.request_id}")

        from_media = False

        # ---- Stage 0: Audio Extraction ----
        if media_source and media_type:
            try:
                logger.info("Stage 0: Audio Extraction -> Text")
                transcription_result = self._stage_0_audio_extraction(
                    media_source, media_type, kwargs.get("language")
                )
                text_input = transcription_result["transcription"]["full_text"]
                self.pipeline_stages["0_audio_extraction"] = transcription_result
                from_media = True
            except Exception as e:
                logger.error(f"Audio extraction failed: {e}")
                return self._error_response(f"Audio extraction failed: {e}")

        if not text_input:
            return self._error_response("No text input or media source provided")

        # ---- Stage 0b: LLM Claim Extraction (media only) ----
        if from_media and CLAIM_EXTRACTOR_AVAILABLE:
            try:
                logger.info("Stage 0b: LLM Claim Extraction")
                source_type = kwargs.get("source_type", "transcript")
                extracted = await extract_claim(text_input, source_type)
                self.pipeline_stages["0b_claim_extraction"] = {
                    "request_id": self.request_id,
                    "raw_text": text_input,
                    "extracted_claim": extracted,
                    "source_type": source_type,
                }
                text_input = extracted
            except Exception as e:
                logger.warning(f"Claim extraction failed, using raw text: {e}")

        # ---- Stage 1: Ingest & Normalize ----
        try:
            logger.info("Stage 1: Ingest Claim")
            if _process_user_claim:
                step1_out = _process_user_claim(text_input)
            else:
                step1_out = self._fallback_stage_1(text_input)
            self.pipeline_stages["1_ingest_claim"] = step1_out
            self._log_stage("1_ingest_claim", step1_out)
        except Exception as e:
            logger.error(f"Ingest claim failed: {e}")
            return self._error_response(f"Ingest claim failed: {e}")

        # ---- Stage 2: RoBERTa Inference ----
        try:
            logger.info("Stage 2: RoBERTa Inference")
            if _process_step1_output:
                step2_out = _process_step1_output(step1_out)
            else:
                step2_out = self._fallback_stage_2(step1_out)
            self.pipeline_stages["2_roberta_inference"] = step2_out
            self._log_stage("2_roberta_inference", step2_out)
        except Exception as e:
            logger.error(f"RoBERTa inference failed: {e}")
            return self._error_response(f"RoBERTa inference failed: {e}")

        # ---- Stage 3: Routing Policy ----
        try:
            logger.info("Stage 3: Routing Policy")
            if _process_step2_output:
                step3_out = _process_step2_output(step2_out)
            else:
                step3_out = self._fallback_stage_3(step2_out)
            self.pipeline_stages["3_routing_policy"] = step3_out
            self._log_stage("3_routing_policy", step3_out)
        except Exception as e:
            logger.error(f"Routing policy failed: {e}")
            return self._error_response(f"Routing policy failed: {e}")

        # ---- Stages 4-8: Full analysis path ----
        routing_decision = step3_out.get("routing_decision", "FULL_ANALYSIS")
        last_output = step3_out

        if routing_decision != "FAST_REAL":
            async_stages = [
                ("4_query_building",       _process_step3_output),
                ("5_rag_retrieval",        _process_step4_output),
                ("6_mmr_selection",        _process_step5_output),
                ("7_light_relevance_gate", _process_step6_output),
                ("8_explainability_shap",  _process_step7_output),
                ("9_llm_summarization",    _process_step8_output),
            ]
            for stage_name, stage_fn in async_stages:
                if stage_fn is None:
                    logger.info("Stage %s: module not loaded, skipping", stage_name)
                    continue
                try:
                    logger.info("Stage %s", stage_name)
                    result = await stage_fn(last_output)
                    self.pipeline_stages[stage_name] = result
                    self._log_stage(stage_name, result)
                    last_output = result
                except Exception as e:
                    logger.error(f"Stage {stage_name} failed: {e}")
                    return self._error_response(f"Stage {stage_name} failed: {e}")

        # ---- Build final result ----
        final_result = self._build_final_result(step2_out, last_output)
        logger.info(f"Pipeline completed: {self.request_id}")
        return final_result

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _stage_0_audio_extraction(
        self, media_source: str, media_type: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.audio_extractor:
            raise RuntimeError("Audio extraction not available")
        result = self.audio_extractor.process(
            request_id=self.request_id,
            claim_id=self.claim_id,
            media_source=media_source,
            media_type=media_type,
            language=language or "en",
        )
        self._log_stage("0_audio_extraction", result)
        return result

    def _fallback_stage_1(self, claim_text: str) -> Dict[str, Any]:
        """Inline ingest when Stage 1 module is not available."""
        import unicodedata

        normalized = " ".join(unicodedata.normalize("NFKC", claim_text).split())
        return {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "user_claim": claim_text,
            "normalized_claim": normalized,
            "meta": {
                "received_at": datetime.utcnow().isoformat() + "Z",
                "version": "1.0",
            },
        }

    def _fallback_stage_2(self, step1_out: Dict[str, Any]) -> Dict[str, Any]:
        """Inline RoBERTa inference when Stage 2 module is not available."""
        claim_text = step1_out.get("normalized_claim", "")

        if REAL_MODEL_AVAILABLE and is_model_available():
            pred = model_predict(claim=claim_text)
            return {
                **step1_out,
                "roberta": {
                    "label": {
                        "logits": pred["logits"],
                        "probs": pred["probs"],
                        "class_id": pred["class_id"],
                        "class_name": pred["label"],
                    },
                    "confidence": pred["confidence"],
                    "model": {"name": "roberta-base(+lora)", "revision": "dev", "ctx_tokens": 512},
                    "tokenization": {"truncated": False, "input_tokens": len(claim_text.split())},
                },
                "meta": {
                    **step1_out.get("meta", {}),
                    "real_model": True,
                },
            }

        logger.warning("Real model not available; Stage 2 using placeholder")
        return {
            **step1_out,
            "roberta": {
                "label": {
                    "logits": [0.0, 0.0, 0.0],
                    "probs": [0.33, 0.34, 0.33],
                    "class_id": 1,
                    "class_name": "mixed",
                },
                "confidence": 0.34,
                "model": {"name": "roberta-base(+lora)", "revision": "dev", "ctx_tokens": 512},
                "tokenization": {"truncated": False, "input_tokens": len(claim_text.split())},
            },
            "meta": {
                **step1_out.get("meta", {}),
                "real_model": False,
            },
        }

    def _fallback_stage_3(self, step2_out: Dict[str, Any]) -> Dict[str, Any]:
        """Inline routing when Stage 3 module is not available."""
        roberta = step2_out.get("roberta", {})
        confidence = roberta.get("confidence", 0.5)
        class_name = roberta.get("label", {}).get("class_name", "mixed")

        if class_name == "true" and confidence > 0.9:
            routing_decision = "FAST_REAL"
        else:
            routing_decision = "FULL_ANALYSIS"

        return {
            **step2_out,
            "routing_decision": routing_decision,
        }

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_final_result(
        self,
        step2_out: Dict[str, Any],
        last_stage_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the final pipeline response."""
        roberta = step2_out.get("roberta", {})
        label_info = roberta.get("label", {})
        class_name = label_info.get("class_name", "mixed")
        confidence = roberta.get("confidence", 0.5)

        label_to_prediction = {"false": "FAKE", "mixed": "REQUIRES_VERIFICATION", "true": "REAL"}
        prediction = label_to_prediction.get(class_name, "REQUIRES_VERIFICATION")

        return {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "final_prediction": prediction,
            "confidence": confidence,
            "stages": self.pipeline_stages,
            "last_stage_output": last_stage_out,
            "meta": {
                "total_stages_executed": len(self.pipeline_stages),
                "pipeline_completed_at": datetime.utcnow().isoformat() + "Z",
                "execution_log": self.execution_log,
            },
        }

    # ------------------------------------------------------------------
    # Logging / errors
    # ------------------------------------------------------------------

    def _log_stage(self, stage_name: str, result: Dict[str, Any]) -> None:
        log_entry = {
            "stage": stage_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success",
            "result_keys": list(result.keys()) if isinstance(result, dict) else None,
        }
        self.execution_log.append(log_entry)
        logger.debug(f"Stage {stage_name} completed")

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "error": error_message,
            "status": "failed",
            "stages": self.pipeline_stages,
            "meta": {
                "failed_at": datetime.utcnow().isoformat() + "Z",
                "execution_log": self.execution_log,
            },
        }


def get_runner() -> PipelineRunner:
    """Get or create the pipeline runner."""
    return PipelineRunner()
