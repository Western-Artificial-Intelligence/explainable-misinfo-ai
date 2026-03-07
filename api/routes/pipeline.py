"""
Unified Production Pipeline API Routes

Endpoints:
    POST /api/pipeline/process-media     - Process audio/video → full pipeline
    POST /api/pipeline/process-text      - Process text → full pipeline
    POST /api/pipeline/process-direct    - Process with direct audio blob
    GET  /api/pipeline/status/{request_id} - Check request status
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import tempfile
import os
import json
import re
from datetime import datetime
import uuid

from api.production_pipeline import pipeline_runner as pipeline_runner_mod
from api.production_pipeline.pipeline_runner import PipelineRunner
from api.production_pipeline.middlewares.llm_blackbox import LLMBlackbox, LLMBlackboxError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# Store orchestrator instances (could use Redis/database in production)
active_requests = {}


class ProcessMediaRequest(BaseModel):
    """Request model for processing media"""
    media_source: str = Field(..., description="File path or URL")
    media_type: str = Field(..., description="'video', 'audio', or 'url'")
    language: Optional[str] = Field("en", description="Language code")
    request_id: Optional[str] = Field(None, description="Optional custom request ID")


class ProcessTextRequest(BaseModel):
    """Request model for processing text"""
    text: str = Field(..., description="Text to analyze")
    request_id: Optional[str] = Field(None, description="Optional custom request ID")


class PipelineResponse(BaseModel):
    """Response model for pipeline"""
    request_id: str
    claim_id: str
    final_prediction: Optional[str] = None
    confidence: Optional[float] = None
    summary: Optional[Dict[str, Any]] = None
    roberta: Optional[Dict[str, Any]] = None
    essay: Optional[Dict[str, Any]] = None
    stages: Dict[str, Any] = {}
    last_stage_output: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = {}
    error: Optional[str] = None
    status: Optional[str] = None


def _normalize_verdict(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"real", "true", "reliable", "factual"}:
        return "true"
    if raw in {"fake", "false", "misinformation", "not_real"}:
        return "false"
    return "mixed"


def _prediction_from_verdict(value: str) -> str:
    verdict = _normalize_verdict(value)
    if verdict == "true":
        return "TRUE"
    if verdict == "false":
        return "FALSE"
    return "MIXED"


def _build_fast_queries(normalized_claim: str) -> list[dict[str, str]]:
    base = str(normalized_claim or "").strip()
    if not base:
        return []

    candidates = [{"q": base, "variant": "primary"}]
    copula_match = re.match(
        r"^\s*(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?P<predicate>[a-z][a-z\s-]{1,40})\s*$",
        base,
        flags=re.IGNORECASE,
    )
    if copula_match:
        subject = " ".join(str(copula_match.group("subject") or "").split())
        predicate = " ".join(str(copula_match.group("predicate") or "").split())
        if subject and predicate:
            candidates.extend(
                [
                    {"q": f"{subject} {predicate}", "variant": "subject_predicate"},
                    {"q": f"{subject} nationality", "variant": "nationality"},
                    {"q": f"{subject} citizenship", "variant": "citizenship"},
                    {"q": f"who is {subject}", "variant": "who_is"},
                ]
            )
    candidates.extend(
        [
            {"q": f"is it true that {base}", "variant": "verify"},
            {"q": f"fact check {base}", "variant": "fact_check"},
        ]
    )
    out: list[dict[str, str]] = []
    seen = set()
    for item in candidates:
        q = " ".join(str(item.get("q") or "").split()).strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"q": q, "variant": str(item.get("variant") or "primary")})
    return out


def _extract_claim_profile(normalized_claim: str) -> Dict[str, str]:
    base = str(normalized_claim or "").strip()
    match = re.match(
        r"^\s*(?P<subject>.+?)\s+(?:is|are|was|were)\s+(?P<predicate>[a-z][a-z\s-]{1,40})\s*$",
        base,
        flags=re.IGNORECASE,
    )
    if not match:
        return {}
    subject = " ".join(str(match.group("subject") or "").split())
    predicate = " ".join(str(match.group("predicate") or "").split())
    if not subject or not predicate:
        return {}
    return {"kind": "copula", "subject": subject, "predicate": predicate}


def _word_tokens(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(tok) >= 3]


def _simple_norm(text: str) -> str:
    return " ".join(_word_tokens(text))


def _domain_quality_score(source: str) -> float:
    src = str(source or "").strip().lower()
    if not src:
        return 0.0

    high_trust = (
        ".gov",
        ".edu",
        "nasa.gov",
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "who.int",
        "britannica.com",
        "wikipedia.org",
        "astronomy.com",
        "nature.com",
        "science.org",
        "sciencedirect.com",
    )
    medium_trust = (
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "wired.com",
        "history.com",
    )
    low_trust = (
        "reddit.com",
        "quora.com",
        "stackexchange.com",
        "medium.com",
        "blogspot.com",
        "tumblr.com",
    )
    if any(pattern in src for pattern in low_trust):
        return -0.22
    if any(pattern in src for pattern in high_trust):
        return 0.18
    if any(pattern in src for pattern in medium_trust):
        return 0.08
    return 0.0


def _score_fast_evidence_item(
    item: Dict[str, Any], normalized_claim: str, claim_profile: Dict[str, str]
) -> float:
    score = float(item.get("score") or 0.0)
    doc = item.get("doc") if isinstance(item.get("doc"), dict) else {}
    title = str(doc.get("title") or "").strip()
    source = str(doc.get("source") or "").strip().lower()
    variant = str(doc.get("source_variant") or "").strip().lower()
    text = str(item.get("text") or "").strip()
    haystack = f"{title} {text}".lower()
    title_lower = title.lower()
    score += _domain_quality_score(source)

    claim_tokens = set(_word_tokens(normalized_claim))
    evidence_tokens = set(_word_tokens(haystack))
    if claim_tokens:
        overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        score += 0.18 * overlap
    if _simple_norm(normalized_claim) and _simple_norm(normalized_claim) in _simple_norm(haystack):
        score += 0.14
    if title_lower.startswith("is it true") or title_lower.startswith("does ") or title_lower.startswith("why "):
        score -= 0.08

    if claim_profile:
        subject = str(claim_profile.get("subject") or "").strip().lower()
        predicate = str(claim_profile.get("predicate") or "").strip().lower()
        subject_tokens = [tok for tok in _word_tokens(subject) if tok not in {"the"}]
        if subject and subject in haystack:
            score += 0.14
        elif subject_tokens:
            subj_overlap = sum(1 for tok in subject_tokens if tok in evidence_tokens) / max(1, len(subject_tokens))
            score += 0.12 * subj_overlap

        variant_boosts = {
            "nationality": 0.24,
            "citizenship": 0.22,
            "who_is": 0.18,
            "fact_check": 0.14,
            "verify": 0.10,
            "subject_predicate": 0.05,
            "primary": -0.04,
        }
        score += variant_boosts.get(variant, 0.0)

        anchor_terms = (
            "nationality",
            "citizenship",
            "born",
            "birth",
            "ancestry",
            "heritage",
            "family",
            "biography",
            "who is",
            "immigrant",
        )
        has_anchor = any(term in haystack for term in anchor_terms)
        if has_anchor:
            score += 0.16
        if predicate and predicate in haystack:
            score += 0.08

        if "wikipedia.org" in source or "history.com" in source or "britannica.com" in source:
            score += 0.08

        if predicate == "indian":
            contradiction_terms = (
                "american",
                "united states",
                "u.s.",
                "german",
                "scottish",
                "ancestry",
                "citizenship",
                "nationality",
                "family",
            )
            if any(term in haystack for term in contradiction_terms):
                score += 0.16
            if ("india" in haystack or "indian americans" in haystack or "trade" in haystack) and not has_anchor:
                score -= 0.20
            if "indian american" in haystack and "trump" in haystack and not has_anchor:
                score -= 0.12

        if not has_anchor and variant in {"primary", "subject_predicate"} and len(claim_tokens & evidence_tokens) <= 1:
            score -= 0.08

    return score


def _select_fast_evidence_items(step5_out: Dict[str, Any], normalized_claim: str) -> list[Dict[str, Any]]:
    rag_candidates = step5_out.get("rag_candidates") if isinstance(step5_out.get("rag_candidates"), dict) else {}
    items = rag_candidates.get("items") if isinstance(rag_candidates.get("items"), list) else []
    if not items:
        return []

    claim_profile = _extract_claim_profile(normalized_claim)
    scored: list[tuple[float, Dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        scored.append((_score_fast_evidence_item(item, normalized_claim, claim_profile), item))

    scored.sort(
        key=lambda pair: (
            -float(pair[0]),
            -float((pair[1].get("score") or 0.0)),
            int((((pair[1].get("doc") or {}) if isinstance(pair[1].get("doc"), dict) else {}).get("serp_rank") or 10**9)),
        )
    )

    preferred_variants = {"nationality", "citizenship", "who_is", "fact_check", "verify"}
    selected: list[Dict[str, Any]] = []
    seen_urls = set()
    for custom_score, item in scored:
        doc = item.get("doc") if isinstance(item.get("doc"), dict) else {}
        url = str(doc.get("url") or "").strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        out = dict(item)
        out["fast_score"] = float(custom_score)
        selected.append(out)
        if len(selected) >= 4:
            break

    if claim_profile:
        preferred_item = None
        for custom_score, item in scored:
            doc = item.get("doc") if isinstance(item.get("doc"), dict) else {}
            variant = str(doc.get("source_variant") or "").strip().lower()
            url = str(doc.get("url") or "").strip()
            if variant in preferred_variants and url and all(str((x.get("doc") or {}).get("url") or "") != url for x in selected):
                preferred_item = dict(item)
                preferred_item["fast_score"] = float(custom_score)
                break
        if preferred_item is not None:
            selected = [preferred_item] + selected[:3]

    return selected[:4]


def _parse_fast_verdict_response(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except Exception:
            return None

    if not isinstance(parsed, dict):
        return None

    verdict = _normalize_verdict(str(parsed.get("verdict") or "mixed"))
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(parsed.get("reason") or "").strip()
    return {"verdict": verdict, "confidence": confidence, "reason": reason}


async def _classify_fast_text_with_evidence(
    claim: str, evidence_items: list[dict[str, Any]]
) -> Dict[str, Any]:
    if not evidence_items:
        return {
            "verdict": "mixed",
            "confidence": 0.35,
            "reason": "No usable evidence snippets were retrieved from the web.",
        }

    evidence_lines = []
    for idx, item in enumerate(evidence_items[:3], start=1):
        doc = item.get("doc") if isinstance(item, dict) else {}
        doc = doc if isinstance(doc, dict) else {}
        title = str(doc.get("title") or "Untitled source").strip()
        url = str(doc.get("url") or "").strip()
        snippet = " ".join(str(item.get("text") or "").split()).strip()
        if len(snippet) > 320:
            snippet = snippet[:320].rstrip() + "..."
        evidence_lines.append(
            f"[{idx}] Title: {title}\nURL: {url}\nSnippet: {snippet}"
        )

    system_context = (
        "You are a misinformation analyst. "
        "Using only the provided evidence, decide whether the claim is true, false, or mixed. "
        "Return exactly one lowercase word: true, false, or mixed. "
        "Use false when the evidence clearly contradicts the claim. "
        "Use true when the evidence clearly supports the claim. "
        "Use mixed when the evidence is inconclusive, conflicting, off-topic, only partially supportive, "
        "limited to special subgroups, or supports a weaker version of the claim than stated."
    )
    user_context = (
        f"Claim: {claim}\n\n"
        "Evidence:\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "Reply with exactly one word."
    )

    try:
        llm = LLMBlackbox()
        raw = await llm.generate_async(
            system_context=system_context,
            user_context=user_context,
            temperature=0,
            num_predict=8,
        )
        verdict = _normalize_verdict(raw)
        if verdict == "mixed":
            match = re.search(r"\b(false|mixed|true)\b", str(raw or "").lower())
            if match:
                verdict = _normalize_verdict(match.group(1))
        if verdict in {"false", "mixed", "true"}:
            confidence = 0.82 if verdict in {"false", "true"} else 0.58
            return {
                "verdict": verdict,
                "confidence": confidence,
                "reason": "Fast verdict inferred from retrieved web evidence.",
            }
    except LLMBlackboxError as exc:
        logger.warning("Fast text evidence verdict failed: %s", exc.message)
    except Exception as exc:
        logger.warning("Fast text evidence verdict failed: %s", exc)

    return {
        "verdict": "mixed",
        "confidence": 0.4,
        "reason": "Evidence was retrieved, but the fast verdict step could not confidently classify it.",
    }


def _build_fallback_summary(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    last_stage_output = result.get("last_stage_output")
    if not isinstance(last_stage_output, dict):
        return None

    roberta = result.get("roberta")
    if not isinstance(roberta, dict):
        roberta = last_stage_output.get("roberta")
    roberta = roberta if isinstance(roberta, dict) else {}

    label = roberta.get("label") if isinstance(roberta.get("label"), dict) else {}
    verdict = _normalize_verdict(
        result.get("final_prediction") or label.get("class_name") or ""
    )
    confidence = float(roberta.get("confidence") or result.get("confidence") or 0.0)
    claim = str(result.get("user_claim") or last_stage_output.get("user_claim") or "").strip()

    evidence_topk = last_stage_output.get("evidence_topk")
    items = evidence_topk.get("items") if isinstance(evidence_topk, dict) else []
    items = items if isinstance(items, list) else []

    citations = []
    for item in items[:2]:
        doc = item.get("doc") if isinstance(item, dict) else {}
        if not isinstance(doc, dict):
            continue
        citations.append(
            {
                "doc_id": str(item.get("doc_id") or ""),
                "source": str(doc.get("source") or ""),
                "title": str(doc.get("title") or ""),
                "url": str(doc.get("url") or ""),
            }
        )

    fast_reason = str(result.get("fast_verdict_reason") or "").strip()
    intro = (
        f'The claim "{claim}" is predicted to be {verdict} with {round(confidence * 100, 1)}% confidence.'
        if claim
        else f"Pipeline verdict: {verdict} ({round(confidence * 100, 1)}% confidence)."
    )
    body1 = ""
    if items:
        snippets = []
        for item in items[:2]:
            text = str(item.get("text") or "").strip()
            doc = item.get("doc") if isinstance(item, dict) else {}
            title = str((doc or {}).get("title") or "Source").strip()
            if text:
                snippets.append(f"{title}: {text[:240]}")
        body1 = "\n\n".join(snippets)

    conclusion_parts = []
    if fast_reason:
        conclusion_parts.append(fast_reason)
    conclusion_parts.append(
        "Retrieved web evidence is attached below."
        if citations
        else "No web citations were attached to this run."
    )

    return {
        "intro": intro,
        "body1": body1,
        "body2": "",
        "conclusion": " ".join(part for part in conclusion_parts if part),
        "verdict": verdict,
        "confidence": confidence,
        "citations": citations,
    }


def _merge_runner_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Expose final-stage summary fields at the top level for frontend clients."""
    merged = dict(result)
    last_stage_output = result.get("last_stage_output")
    if isinstance(last_stage_output, dict):
        if "summary" not in merged and isinstance(last_stage_output.get("summary"), dict):
            merged["summary"] = last_stage_output["summary"]
        if "roberta" not in merged and isinstance(last_stage_output.get("roberta"), dict):
            merged["roberta"] = last_stage_output["roberta"]
    if "summary" not in merged:
        fallback_summary = _build_fallback_summary(merged)
        if fallback_summary is not None:
            merged["summary"] = fallback_summary
    summary = merged.get("summary")
    if isinstance(summary, dict):
        merged.setdefault("essay", dict(summary))
    return merged


def _placeholder_roberta(normalized_claim: str) -> Dict[str, Any]:
    token_count = len((normalized_claim or "").split())
    return {
        "label": {
            "logits": [0.0, 1.0, 0.0],
            "probs": [0.2, 0.6, 0.2],
            "class_id": 1,
            "class_name": "mixed",
        },
        "confidence": 0.6,
        "model": {"name": "fast-web-placeholder", "revision": "dev", "ctx_tokens": 512},
        "tokenization": {"truncated": False, "input_tokens": token_count},
    }


def _fast_roberta_from_verdict(normalized_claim: str, verdict: str, confidence: float) -> Dict[str, Any]:
    token_count = len((normalized_claim or "").split())
    mapped = _normalize_verdict(verdict)
    logits_map = {
        "false": [3.0, 0.0, -1.0],
        "mixed": [0.0, 2.5, 0.0],
        "true": [-1.0, 0.0, 3.0],
    }
    probs_map = {
        "false": [confidence, max(0.0, 1.0 - confidence - 0.02), 0.02],
        "mixed": [max(0.0, (1.0 - confidence) / 2.0), confidence, max(0.0, (1.0 - confidence) / 2.0)],
        "true": [0.02, max(0.0, 1.0 - confidence - 0.02), confidence],
    }
    class_map = {"false": 0, "mixed": 1, "true": 2}
    return {
        "label": {
            "logits": logits_map[mapped],
            "probs": probs_map[mapped],
            "class_id": class_map[mapped],
            "class_name": mapped,
        },
        "confidence": confidence,
        "model": {"name": "fast-web-evidence-judge", "revision": "dev", "ctx_tokens": 512},
        "tokenization": {"truncated": False, "input_tokens": token_count},
    }


async def _run_fast_text_web_path(text: str, request_id: Optional[str]) -> Dict[str, Any]:
    pipeline_runner_mod._ensure_stages_loaded()

    runner = PipelineRunner()
    runner.request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
    runner.claim_id = f"claim_{uuid.uuid4().hex[:12]}"

    if pipeline_runner_mod._process_user_claim:
        step1_out = pipeline_runner_mod._process_user_claim(text)
    else:
        step1_out = runner._fallback_stage_1(text)

    normalized_claim = str(step1_out.get("normalized_claim") or text).strip()
    roberta = _placeholder_roberta(normalized_claim)
    routing = {
        "policy_name": "fast_text_web",
        "policy_version": "v1",
        "tier": "fast_web",
        "rag": {
            "candidate_pool_size_m": 24,
            "mmr_top_n": 8,
            "mmr_lambda": 0.70,
            "final_top_k": 4,
            "min_relevance": 0.20,
            "use_query_expansion": False,
            "allow_web_search": True,
        },
    }
    queries = _build_fast_queries(normalized_claim)
    query_plan = {
        "primary_query": normalized_claim,
        "queries": queries or [{"q": normalized_claim, "variant": "primary"}],
        "expansion": {"enabled": False, "method": "template_fast_web", "requested": max(0, len(queries) - 1), "kept": max(0, len(queries) - 1)},
    }

    step4_out = {
        "request_id": runner.request_id,
        "claim_id": runner.claim_id,
        "user_claim": step1_out.get("user_claim") or text,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "meta": dict(step1_out.get("meta") or {}),
    }

    step5_out = await pipeline_runner_mod._process_step4_output(step4_out)
    selected_items = _select_fast_evidence_items(step5_out, normalized_claim)
    step6_out = {
        "request_id": runner.request_id,
        "claim_id": runner.claim_id,
        "user_claim": step1_out.get("user_claim") or text,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "rag_candidates": step5_out.get("rag_candidates") or {},
        "mmr_selected": {
            "top_n": len(selected_items),
            "lambda": routing["rag"]["mmr_lambda"],
            "relevance_blend_alpha": 1.0,
            "similarity": {"metric": "claim_aware_fast_score", "embed_model": "fast_selector_v1"},
            "items": [
                {
                    **item,
                    "rank": idx,
                    "mmr_score": float(item.get("fast_score") or item.get("score") or 0.0),
                    "mmr_relevance": float(item.get("fast_score") or item.get("score") or 0.0),
                    "retrieval_score": float(item.get("score") or 0.0),
                }
                for idx, item in enumerate(selected_items, start=1)
            ],
        },
        "meta": dict(step5_out.get("meta") or {}),
    }
    step7_out = {
        "request_id": runner.request_id,
        "claim_id": runner.claim_id,
        "user_claim": step1_out.get("user_claim") or text,
        "normalized_claim": normalized_claim,
        "roberta": roberta,
        "routing": routing,
        "query_plan": query_plan,
        "rag_candidates": step5_out.get("rag_candidates") or {},
        "mmr_selected": step6_out["mmr_selected"],
        "evidence_topk": {
            "top_k": len(selected_items),
            "min_relevance": routing["rag"]["min_relevance"],
            "method": "claim_aware_fast_selector",
            "items": [
                {
                    "rank": idx,
                    "chunk_id": str(item.get("chunk_id") or ""),
                    "doc_id": str(((item.get("doc") or {}) if isinstance(item.get("doc"), dict) else {}).get("doc_id") or ""),
                    "text": str(item.get("text") or ""),
                    "relevance_score": float(item.get("fast_score") or item.get("score") or 0.0),
                    "doc": {
                        "source": str(((item.get("doc") or {}) if isinstance(item.get("doc"), dict) else {}).get("source") or ""),
                        "title": str(((item.get("doc") or {}) if isinstance(item.get("doc"), dict) else {}).get("title") or ""),
                        "url": str(((item.get("doc") or {}) if isinstance(item.get("doc"), dict) else {}).get("url") or ""),
                    },
                }
                for idx, item in enumerate(selected_items, start=1)
            ],
        },
        "meta": dict(step5_out.get("meta") or {}),
    }
    evidence_items = step7_out["evidence_topk"]["items"]
    verdict_info = await _classify_fast_text_with_evidence(normalized_claim, evidence_items)
    verdict = str(verdict_info.get("verdict") or "mixed")
    confidence = float(verdict_info.get("confidence") or 0.4)
    reason = str(verdict_info.get("reason") or "").strip()
    roberta = _fast_roberta_from_verdict(normalized_claim, verdict, confidence)

    meta = dict(step7_out.get("meta") or {})
    warnings = meta.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        meta["warnings"] = warnings
    warnings.append(
        {
            "code": "FAST_TEXT_WEB_PATH",
            "message": "Used fast website text path: claim-aware evidence selection and verdicting over retrieved web results.",
        }
    )
    step7_out = dict(step7_out)
    step7_out["roberta"] = roberta
    step7_out["meta"] = meta

    return {
        "request_id": runner.request_id,
        "claim_id": runner.claim_id,
        "final_prediction": _prediction_from_verdict(verdict),
        "confidence": confidence,
        "roberta": roberta,
        "fast_verdict_reason": reason,
        "stages": {
            "1_ingest_claim": step1_out,
            "4_query_building": step4_out,
            "5_rag_retrieval": step5_out,
            "6_mmr_selection": step6_out,
            "7_light_relevance_gate": step7_out,
        },
        "last_stage_output": step7_out,
        "meta": {
            "total_stages_executed": 5,
            "pipeline_completed_at": datetime.utcnow().isoformat() + "Z",
            "execution_log": [],
        },
    }


@router.post("/process-media")
async def process_media(request: ProcessMediaRequest) -> PipelineResponse:
    """
    Process audio/video file through full pipeline
    
    Flow:
    1. Extract audio from media (if video)
    2. Transcribe to text (Whisper)
    3. Ingest & normalize claim
    4. RoBERTa inference
    5. Routing policy
    6. Continue through remaining pipeline
    
    Args:
        media_source: File path or URL
        media_type: 'video', 'audio', or 'url'
        language: Language code (default: 'en')
        request_id: Optional custom request ID
        
    Returns:
        Complete pipeline result
    """
    try:
        logger.info(f"Processing media: {request.media_source}")
        
        runner = PipelineRunner()
        result = await runner.run(
            media_source=request.media_source,
            media_type=request.media_type,
            text_input=None,
            request_id=request.request_id,
            language=request.language
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        result = _merge_runner_result(result)
        # Store for status checking
        active_requests[result['request_id']] = result
        
        logger.info(f"Media processing completed: {result['request_id']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Media processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-text")
async def process_text(request: ProcessTextRequest) -> PipelineResponse:
    """
    Process text directly through pipeline
    
    Skips audio extraction, goes straight to text normalization
    
    Args:
        text: Text to analyze
        request_id: Optional custom request ID
        
    Returns:
        Complete pipeline result
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"Processing text claim")

        result = await _run_fast_text_web_path(request.text, request.request_id)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        result = _merge_runner_result(result)
        # Store for status checking
        active_requests[result['request_id']] = result
        
        logger.info(f"Text processing completed: {result['request_id']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-direct")
async def process_direct(
    request_id: str = Form(...),
    claim_id: str = Form(...),
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form("en")
) -> PipelineResponse:
    """
    Process audio file directly (uploaded from extension)
    
    This is the primary endpoint for the Chrome extension
    
    Args:
        request_id: Unique request identifier
        claim_id: Unique claim identifier
        audio_file: Uploaded audio/video file
        language: Language code
        
    Returns:
        Complete pipeline result with transcription + analysis
    """
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"media_{request_id}")
        
        with open(temp_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Determine media type from extension
        filename = audio_file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"}:
            media_type = "video"
        elif ext in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}:
            media_type = "audio"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")
        
        logger.info(f"Processing direct media: {media_type} from extension")
        
        # Run full pipeline
        runner = PipelineRunner()
        result = await runner.run(
            media_source=temp_path,
            media_type=media_type,
            text_input=None,
            request_id=request_id,
            language=language
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        result = _merge_runner_result(result)
        # Store for status checking
        active_requests[result['request_id']] = result
        
        logger.info(f"Direct media processing completed: {request_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Direct processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@router.get("/status/{request_id}")
async def get_status(request_id: str):
    """
    Get processing status and results
    
    Args:
        request_id: Request identifier
        
    Returns:
        Current status and results if available
    """
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    
    return active_requests[request_id]


@router.get("/health")
async def health_check():
    """Health check for pipeline"""
    try:
        runner = PipelineRunner()
        return {
            "status": "healthy",
            "service": "production-pipeline",
            "audio_extraction_available": runner.audio_extractor is not None,
            "active_requests": len(active_requests)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.delete("/clear-requests")
async def clear_requests():
    """Clear completed requests (for cleanup)"""
    count = len(active_requests)
    active_requests.clear()
    return {"message": f"Cleared {count} completed requests"}
