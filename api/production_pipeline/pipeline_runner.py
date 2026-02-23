"""
Unified Pipeline Runner

Coordinates the entire end-to-end pipeline:
0. Audio Extraction → Text Transcription
1. Ingest & Normalize Claim
2. RoBERTa Inference (Claim Classification)
3. Routing Policy (Determine analysis path)
4. Query Building
5. RAG Retrieval
6. MMR Selection
7. Light Relevance Gate
8. Explainability (SHAP)
9. LLM Summarization
10. Output Formatting

Usage:
    runner = PipelineRunner()
    result = runner.run(
        media_source="...",
        media_type="video",
        request_id="req_123"
    )
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Import pipeline steps
try:
    from api.production_pipeline.0_Audio_Extraction_VoiceToText import AudioExtractionPipeline
    AUDIO_EXTRACTION_AVAILABLE = True
except ImportError:
    logger.warning("Audio extraction pipeline not available")
    AUDIO_EXTRACTION_AVAILABLE = False

# These would be imported as they're implemented
# from api.production_pipeline.1_Ingest_claim import IngestClaim
# from api.production_pipeline.2_RoBERTa_inference import RoBERTaInference
# etc.


class PipelineRunner:
    """
    Runs the complete misinformation detection pipeline
    
    Supports both:
    - Direct text input (existing flow)
    - Media input → Audio extraction → Text (new flow)
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
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
    
    def run(
        self,
        media_source: Optional[str] = None,
        media_type: Optional[str] = None,
        text_input: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Args:
            media_source: File path, URL, or media reference (for audio extraction)
            media_type: 'video', 'audio', 'url' (for audio extraction)
            text_input: Direct text input (skips audio extraction)
            request_id: Optional request identifier
            **kwargs: Additional options passed to steps
            
        Returns:
            Complete pipeline result with all stage outputs
        """
        # Generate IDs if not provided
        self.request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        self.claim_id = f"claim_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Starting pipeline: {self.request_id}")
        
        # Stage 0: Audio Extraction (if needed)
        if media_source and media_type:
            try:
                logger.info("Stage 0: Audio Extraction → Text")
                transcription_result = self._stage_0_audio_extraction(
                    media_source, media_type, kwargs.get('language')
                )
                text_input = transcription_result['transcription']['full_text']
                self.pipeline_stages['0_audio_extraction'] = transcription_result
            except Exception as e:
                logger.error(f"Audio extraction failed: {e}")
                return self._error_response(f"Audio extraction failed: {e}")
        
        # Validate we have text input
        if not text_input:
            return self._error_response("No text input or media source provided")
        
        # Stage 1: Ingest & Normalize Claim
        try:
            logger.info("Stage 1: Ingest Claim")
            ingest_result = self._stage_1_ingest_claim(text_input)
            self.pipeline_stages['1_ingest_claim'] = ingest_result
        except Exception as e:
            logger.error(f"Ingest claim failed: {e}")
            return self._error_response(f"Ingest claim failed: {e}")
        
        # Stage 2: RoBERTa Inference
        try:
            logger.info("Stage 2: RoBERTa Inference")
            inference_result = self._stage_2_roberta_inference(
                ingest_result['normalized_claim']
            )
            self.pipeline_stages['2_roberta_inference'] = inference_result
        except Exception as e:
            logger.error(f"RoBERTa inference failed: {e}")
            return self._error_response(f"RoBERTa inference failed: {e}")
        
        # Stage 3: Routing Policy
        try:
            logger.info("Stage 3: Routing Policy")
            routing_result = self._stage_3_routing_policy(inference_result)
            self.pipeline_stages['3_routing_policy'] = routing_result
        except Exception as e:
            logger.error(f"Routing policy failed: {e}")
            return self._error_response(f"Routing policy failed: {e}")
        
        # Stages 4-10: Conditional based on routing
        # (These would be implemented as needed)
        
        # Build final result
        final_result = self._build_final_result(inference_result)
        
        logger.info(f"Pipeline completed: {self.request_id}")
        return final_result
    
    def _stage_0_audio_extraction(
        self,
        media_source: str,
        media_type: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stage 0: Extract audio from media and convert to text"""
        if not self.audio_extractor:
            raise RuntimeError("Audio extraction not available")
        
        result = self.audio_extractor.process(
            request_id=self.request_id,
            claim_id=self.claim_id,
            media_source=media_source,
            media_type=media_type,
            language=language or "en"
        )
        
        self._log_stage("0_audio_extraction", result)
        return result
    
    def _stage_1_ingest_claim(self, claim_text: str) -> Dict[str, Any]:
        """Stage 1: Ingest and normalize claim"""
        result = {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "user_claim": claim_text,
            "normalized_claim": self._normalize_text(claim_text),
            "meta": {
                "received_at": datetime.utcnow().isoformat() + "Z",
                "version": "1.0"
            }
        }
        
        self._log_stage("1_ingest_claim", result)
        return result
    
    def _stage_2_roberta_inference(self, claim_text: str) -> Dict[str, Any]:
        """
        Stage 2: RoBERTa inference for claim classification
        
        TODO: Connect to actual RoBERTa model
        For now returns placeholder
        """
        # This would call the actual RoBERTa model
        # For now, placeholder that matches expected output format
        result = {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "classification": {
                "prediction": "REQUIRES_VERIFICATION",  # or REAL/FAKE
                "confidence": 0.85,
                "logits": {
                    "real": 0.45,
                    "fake": 0.30,
                    "requires_verification": 0.25
                }
            },
            "meta": {
                "model": "roberta-misinformation-classifier",
                "version": "1.0",
                "processed_at": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self._log_stage("2_roberta_inference", result)
        return result
    
    def _stage_3_routing_policy(
        self,
        inference_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 3: Determine routing based on classification
        
        Routes to:
        - High confidence REAL → Light gate only
        - High confidence FAKE → Full RAG + evidence retrieval
        - Uncertain → Full RAG + evidence retrieval
        """
        confidence = inference_result['classification']['confidence']
        prediction = inference_result['classification']['prediction']
        
        # Routing logic
        if prediction == "REAL" and confidence > 0.9:
            routing_path = "FAST_REAL"  # Skip to light gate
        elif prediction == "FAKE" and confidence > 0.9:
            routing_path = "FULL_ANALYSIS"  # Full RAG
        else:
            routing_path = "FULL_ANALYSIS"  # Default to full analysis
        
        result = {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "routing_decision": routing_path,
            "prediction": prediction,
            "confidence": confidence,
            "meta": {
                "decided_at": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        self._log_stage("3_routing_policy", result)
        return result
    
    def _build_final_result(
        self,
        inference_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build final result from all pipeline stages"""
        return {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "final_prediction": inference_result['classification']['prediction'],
            "confidence": inference_result['classification']['confidence'],
            "stages": self.pipeline_stages,
            "meta": {
                "total_stages_executed": len(self.pipeline_stages),
                "pipeline_completed_at": datetime.utcnow().isoformat() + "Z",
                "execution_log": self.execution_log
            }
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (unicode, whitespace, etc.)"""
        import unicodedata
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKC', text)
        
        # Strip and collapse whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _log_stage(self, stage_name: str, result: Dict[str, Any]) -> None:
        """Log stage execution"""
        log_entry = {
            "stage": stage_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success",
            "result_keys": list(result.keys()) if isinstance(result, dict) else None
        }
        self.execution_log.append(log_entry)
        logger.debug(f"Stage {stage_name} completed")
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Build error response"""
        return {
            "request_id": self.request_id,
            "claim_id": self.claim_id,
            "error": error_message,
            "status": "failed",
            "meta": {
                "failed_at": datetime.utcnow().isoformat() + "Z"
            }
        }


def get_runner() -> PipelineRunner:
    """Get or create the pipeline runner (singleton-like)"""
    # In production, this could be cached as a singleton
    return PipelineRunner()
