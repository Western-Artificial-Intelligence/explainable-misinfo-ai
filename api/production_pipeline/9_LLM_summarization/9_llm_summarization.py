# 9_llm_summarization.py
"""
Step 9: LLM summarization — produce a concise, user-facing explanation of why the
claim was classified true/mixed/false, grounded in RoBERTa verdict, evidence (including
paragraphs for chunk_ids used in SHAP), and SHAP explainability.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Constants (tunable via env or override)
MAX_BULLETS = int(os.getenv("SUMMARY_MAX_BULLETS", "4") or "4")
MAX_BULLETS = max(0, min(8, MAX_BULLETS))
MAX_EVIDENCE_SNIPPETS = int(os.getenv("SUMMARY_MAX_EVIDENCE_SNIPPETS", "4") or "4")
MAX_EVIDENCE_SNIPPETS = max(1, min(10, MAX_EVIDENCE_SNIPPETS))
MAX_SNIP_CHARS = int(os.getenv("SUMMARY_MAX_SNIP_CHARS", "400") or "400")
MAX_SNIP_CHARS = max(50, min(2000, MAX_SNIP_CHARS))
TOPK_TOKENS = int(os.getenv("SUMMARY_TOPK_TOKENS", "5") or "5")
TOPK_TOKENS = max(1, min(15, TOPK_TOKENS))

# Essay-style output: intro, body1, body2, conclusion.
LLM_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intro", "body1", "body2", "conclusion"],
    "properties": {
        "intro": {"type": "string", "minLength": 1},
        "body1": {"type": "string", "minLength": 1},
        "body2": {"type": "string", "minLength": 1},
        "conclusion": {"type": "string", "minLength": 1},
    },
}

ESSAY_STRUCTURE = """
- intro: Opening paragraph. State the claim and the model verdict with confidence (e.g. "The claim \"...\" is predicted to be true/false/mixed with X% confidence (True A% · Mixed B% · False C%).").
- body1: First evidence paragraph. Evidence 1 (from SOURCE1_NAME): where you found it (SOURCE1_URL, paragraph P1_ID), what it says in plain English (P1_PLAIN_SUMMARY), why it supports/contradicts the claim (P1_REASON), and key phrases that influenced the model (SHAP: + pos1, + pos2, – neg1).
- body2: Second evidence paragraph. Same structure for Evidence 2 (SOURCE2_NAME, URL, P2_ID, P2_PLAIN_SUMMARY, supports/contradicts because P2_REASON, SHAP phrases).
- conclusion: Final reasoning. One sentence that matches the verdict: if true "Across the sources above, the evidence consistently supports the claim."; if false "Across the sources above, the evidence consistently contradicts the claim."; if mixed "The sources support part of the claim but contradict/limit another part: [briefly what differs]."
""".strip()

SYS_INSTRUCTION = (
    "You are a fact-check explanation assistant. "
    "Use ONLY the provided claim, model verdict, evidence snippets, and SHAP phrases. "
    "Do not invent facts or sources. "
    "Your task is to write a short essay with exactly four parts. "
    "Output STRICT JSON with exactly four keys: \"intro\", \"body1\", \"body2\", \"conclusion\". "
    "Each value is a string (one or more sentences for that section). "
    "intro = claim + verdict + confidence breakdown. "
    "body1 = first evidence block (source, URL, paragraph id, what it says, why it supports/contradicts, SHAP key phrases). "
    "body2 = second evidence block (same structure). "
    "conclusion = one final sentence matching the verdict (true/false/mixed). "
    "Temperature=0, no extra commentary."
)

SYS_REPAIR_INSTRUCTION = (
    "You must output ONLY valid JSON (no markdown, no commentary). "
    "The JSON must have exactly four keys: \"intro\", \"body1\", \"body2\", \"conclusion\", each a non-empty string. "
    "Fix the user's provided JSON so it conforms. Do not add other keys or trailing text."
)

MAX_JSON_REPAIR_RETRIES = 1


class LLMSummarizationError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _copy_meta(meta_in: Dict[str, Any]) -> Dict[str, Any]:
    meta_out: Dict[str, Any] = {"received_at": meta_in.get("received_at"), "version": meta_in.get("version")}
    for k in ("warnings", "latency_ms", "stage_latencies_ms", "total_latency_ms", "stage_errors"):
        if isinstance(meta_in, dict) and k in meta_in:
            meta_out[k] = meta_in[k]
    return meta_out


def _ensure_warnings(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    w = meta.get("warnings")
    if isinstance(w, list):
        return w
    meta["warnings"] = []
    return meta["warnings"]


def _mark_latency(meta: Dict[str, Any], stage_name: str, ms: int) -> None:
    sl = meta.get("stage_latencies_ms")
    if not isinstance(sl, dict):
        sl = {}
        meta["stage_latencies_ms"] = sl
    sl[stage_name] = max(0, int(ms))
    try:
        meta["total_latency_ms"] = int(sum(int(v) for v in sl.values() if isinstance(v, (int, float))))
    except Exception:
        pass


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # Strip markdown code fence if present
    if s.startswith("```"):
        lines = s.split("\n")
        out = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                out.append(line)
        s = "\n".join(out)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _validate_schema(obj: Any, schema: Dict[str, Any]) -> bool:
    """Best-effort validation: required keys and types."""
    if not isinstance(obj, dict):
        return False
    required = schema.get("required") or []
    for key in required:
        if key not in obj:
            return False
    props = schema.get("properties") or {}
    for key, prop in props.items():
        if key not in obj:
            continue
        val = obj[key]
        if prop.get("type") == "string":
            if not isinstance(val, str):
                return False
            min_len = prop.get("minLength", 0)
            if len(val) < min_len:
                return False
        elif prop.get("type") == "array":
            if not isinstance(val, list):
                return False
            max_items = prop.get("maxItems")
            if max_items is not None and len(val) > max_items:
                return False
            enum = prop.get("items", {}).get("enum") if isinstance(prop.get("items"), dict) else None
            if enum and val and not all(isinstance(x, str) for x in val):
                return False
        elif prop.get("type") == "object":
            if not isinstance(val, dict):
                return False
    if schema.get("additionalProperties") is False:
        allowed = set(required) | set((props or {}).keys())
        if not set(obj.keys()) <= allowed:
            return False
    return True


def _unique_preserve_order(lst: List[Any]) -> List[Any]:
    seen: Set[Any] = set()
    out: List[Any] = []
    for x in lst:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _chunk_id_to_doc_id(chunk_id: str) -> str:
    """Map chunk_id to doc_id (e.g. chunk_doc_abc123 -> doc_doc_abc123)."""
    if not chunk_id:
        return ""
    if chunk_id.startswith("chunk_"):
        return chunk_id[6:]  # strip "chunk_" -> "doc_xxx"
    return chunk_id


def _build_chunk_lookup(step8_out: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build chunk_id -> { text, doc_id, doc, relevance_score } from evidence_topk,
    then fill in from rag_candidates / mmr_selected for any evidence_used chunk_ids
    not already present.
    """
    lookup: Dict[str, Dict[str, Any]] = {}

    def add(chunk_id: str, text: str, doc: Any, relevance: float = 0.0, doc_id: Optional[str] = None) -> None:
        if not chunk_id:
            return
        doc = doc or {}
        doc_id = doc_id or str(doc.get("doc_id") or _chunk_id_to_doc_id(chunk_id))
        if chunk_id in lookup:
            return
        lookup[chunk_id] = {
            "text": _strip_html((text or "").strip()),
            "doc_id": doc_id,
            "doc": {
                "source": str(doc.get("source") or "unknown"),
                "title": str(doc.get("title") or "Unknown"),
                "url": str(doc.get("url") or "about:blank"),
                "published_at": doc.get("published_at") or doc.get("age"),
            },
            "relevance_score": relevance,
        }

    # evidence_topk.items
    etop = step8_out.get("evidence_topk") or {}
    for it in etop.get("items") or []:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("chunk_id") or "")
        text = it.get("text") or ""
        doc = it.get("doc") or {}
        rel = float(it.get("relevance_score") or 0.0)
        add(cid, text, doc, rel, str(doc.get("doc_id") or ""))

    # evidence_used from SHAP (may have chunk_ids not in evidence_topk if we only have top-2)
    shap_meta = (step8_out.get("shap_explainability") or {}).get("meta") or {}
    for eu in shap_meta.get("evidence_used") or []:
        if not isinstance(eu, dict):
            continue
        cid = str(eu.get("chunk_id") or "")
        if cid in lookup:
            continue
        doc = eu.get("doc") or {}
        # Try rag_candidates.items and mmr_selected.items for text
        text = ""
        for source in ("rag_candidates", "mmr_selected"):
            cont = step8_out.get(source) or {}
            for item in cont.get("items") or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("chunk_id")) == cid:
                    text = item.get("text") or ""
                    break
            if text:
                break
        add(cid, text, doc, float(eu.get("sim") or 0.0))

    return lookup


def _take_unique_by_doc(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep best chunk per doc_id (highest relevance_score)."""
    by_doc: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        doc_id = c.get("doc_id") or ""
        if not doc_id:
            continue
        existing = by_doc.get(doc_id)
        rel = float(c.get("relevance_score") or 0.0)
        if existing is None or rel > float(existing.get("relevance_score") or 0.0):
            by_doc[doc_id] = c
    return list(by_doc.values())


def _get_evidence_snippets_and_citations(
    step8_out: Dict[str, Any],
    chunk_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build evidence_snips (for LLM) and citations (sources). Prefer chunks that appear
    in shap_explainability.meta.evidence_used (by chunk_id), then evidence_topk, sorted
    by relevance, unique by doc.
    """
    evidence_used_ids = set()
    for eu in ((step8_out.get("shap_explainability") or {}).get("meta") or {}).get("evidence_used") or []:
        if isinstance(eu, dict):
            cid = eu.get("chunk_id")
            if cid:
                evidence_used_ids.add(str(cid))

    # Build list: (prefer_evidence_used, -relevance, rank, chunk_id, entry)
    entries: List[Tuple[float, float, int, str, Dict[str, Any]]] = []
    etop_items = (step8_out.get("evidence_topk") or {}).get("items") or []
    for i, it in enumerate(etop_items):
        if not isinstance(it, dict):
            continue
        cid = str(it.get("chunk_id") or "")
        rec = chunk_lookup.get(cid)
        if not rec:
            continue
        rel = float(it.get("relevance_score") or rec.get("relevance_score") or 0.0)
        rank = int(it.get("rank") or i + 1)
        prefer = 0.0 if cid in evidence_used_ids else 1.0
        entries.append((prefer, -rel, rank, cid, rec))

    # Add evidence_used chunks not in evidence_topk
    for cid in evidence_used_ids:
        if cid in chunk_lookup and not any(e[3] == cid for e in entries):
            rec = chunk_lookup[cid]
            rel = float(rec.get("relevance_score") or 0.0)
            entries.append((0.0, -rel, 999, cid, rec))

    entries.sort(key=lambda x: (x[0], x[1], x[2]))
    chosen = _take_unique_by_doc([x[4] for x in entries])[:MAX_EVIDENCE_SNIPPETS]

    evidence_snips: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    seen_doc: Set[str] = set()

    for c in chosen:
        doc_id = c.get("doc_id") or ""
        doc = c.get("doc") or {}
        snip = _truncate(c.get("text") or "", MAX_SNIP_CHARS)
        evidence_snips.append({
            "doc_id": doc_id,
            "source": doc.get("source"),
            "title": doc.get("title"),
            "url": doc.get("url"),
            "published_at": doc.get("published_at"),
            "relevance_score": c.get("relevance_score"),
            "snippet": snip,
        })
        if doc_id and doc_id not in seen_doc:
            seen_doc.add(doc_id)
            citations.append({
                "doc_id": doc_id,
                "source": doc.get("source"),
                "title": doc.get("title"),
                "url": doc.get("url"),
                "published_at": doc.get("published_at"),
            })

    return evidence_snips, citations


def _get_shap_top_tokens(step8_out: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Return (top_positive_tokens, top_negative_tokens) from either
    shap tokens+top_tokens or segments+top_segments.
    """
    shap = step8_out.get("shap_explainability") or {}
    top_pos: List[str] = []
    top_neg: List[str] = []

    # SHAP path: tokens + top_tokens
    tokens = shap.get("tokens")
    top_tokens = shap.get("top_tokens") or {}
    if isinstance(tokens, list) and tokens:
        pos_idx = top_tokens.get("positive") or []
        neg_idx = top_tokens.get("negative") or []
        k = min(TOPK_TOKENS, len(tokens))
        for i in (pos_idx[:k] if isinstance(pos_idx, list) else []):
            if isinstance(i, int) and 0 <= i < len(tokens):
                t = tokens[i]
                tok = t.get("token") if isinstance(t, dict) else str(t)
                if tok:
                    top_pos.append(str(tok))
        for i in (neg_idx[:k] if isinstance(neg_idx, list) else []):
            if isinstance(i, int) and 0 <= i < len(tokens):
                t = tokens[i]
                tok = t.get("token") if isinstance(t, dict) else str(t)
                if tok:
                    top_neg.append(str(tok))
        return (top_pos[:TOPK_TOKENS], top_neg[:TOPK_TOKENS])

    # Fallback: segments + top_segments
    segments = shap.get("segments")
    top_segments = shap.get("top_segments") or {}
    if isinstance(segments, list) and segments:
        pos_idx = top_segments.get("positive") or []
        neg_idx = top_segments.get("negative") or []
        k = min(TOPK_TOKENS, len(segments))
        for i in (pos_idx[:k] if isinstance(pos_idx, list) else []):
            if isinstance(i, int) and 0 <= i < len(segments):
                s = segments[i]
                text = s.get("text") if isinstance(s, dict) else str(s)
                if text:
                    top_pos.append(str(text).strip()[:80])
        for i in (neg_idx[:k] if isinstance(neg_idx, list) else []):
            if isinstance(i, int) and 0 <= i < len(segments):
                s = segments[i]
                text = s.get("text") if isinstance(s, dict) else str(s)
                if text:
                    top_neg.append(str(text).strip()[:80])
        return (top_pos[:TOPK_TOKENS], top_neg[:TOPK_TOKENS])

    return ([], [])


def _get_llm():
    try:
        from api.production_pipeline.middlewares.llm_blackbox import LLMBlackbox
        return LLMBlackbox()
    except Exception:
        import importlib.util
        from pathlib import Path
        here = Path(__file__).resolve().parent.parent / "middlewares" / "llm_blackbox.py"
        spec = importlib.util.spec_from_file_location("llm_blackbox", here)
        if spec is None or spec.loader is None:
            raise LLMSummarizationError("LLM_UNAVAILABLE", "Could not load llm_blackbox.")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "LLMBlackbox")()


async def process_step8_output(step8_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: Step 8 output (passthrough from step 7).
    Output: Step 8 output plus "summary" (verdict, confidence, short, bullets, citations, explainability).
    """
    t0 = time.perf_counter()
    meta_out = _copy_meta(step8_out.get("meta") or {})
    warnings = _ensure_warnings(meta_out)

    try:
        request_id = step8_out["request_id"]
        claim_id = step8_out["claim_id"]
        user_claim = step8_out["user_claim"]
        normalized_claim = step8_out.get("normalized_claim") or user_claim
        roberta = step8_out.get("roberta") or {}
        evidence_topk = step8_out.get("evidence_topk") or {}
        shap_explainability = step8_out.get("shap_explainability") or {}
    except Exception as e:
        raise LLMSummarizationError("INVALID_INPUT", "Step9 expected Step8 output shape.", {"error": str(e)})

    # 1) Verdict + confidence
    label = roberta.get("label") or {}
    verdict = str(label.get("class_name") or "mixed")
    confidence = float(roberta.get("confidence") or 0.0)

    # 2) Chunk lookup and evidence snippets + citations (include paragraphs for evidence_used chunk_ids)
    chunk_lookup = _build_chunk_lookup(step8_out)
    evidence_snips, citations = _get_evidence_snippets_and_citations(step8_out, chunk_lookup)

    # 3) SHAP top tokens (or segments as token-like strings)
    top_pos_tokens, top_neg_tokens = _get_shap_top_tokens(step8_out)

    # Probabilities for template (True %, Mixed %, False %)
    probs = list(label.get("probs") or [0.0, 0.0, 0.0])
    while len(probs) < 3:
        probs.append(0.0)
    p_true_pct = round(100.0 * float(probs[2]), 1)
    p_mixed_pct = round(100.0 * float(probs[1]), 1)
    p_false_pct = round(100.0 * float(probs[0]), 1)
    top_pct = round(100.0 * confidence, 1)

    # Exactly 2 evidence items for the template (Evidence 1, Evidence 2)
    ev1 = evidence_snips[0] if len(evidence_snips) > 0 else {"snippet": "(No evidence)", "source": "—", "title": "—", "url": "—", "doc_id": "—"}
    ev2 = evidence_snips[1] if len(evidence_snips) > 1 else {"snippet": "(No second evidence)", "source": "—", "title": "—", "url": "—", "doc_id": "—"}

    # 4) Build LLM payload for essay (intro, body1, body2, conclusion)
    payload = (
        "ESSAY STRUCTURE:\n"
        + ESSAY_STRUCTURE
        + "\n\n"
        "VALUES TO USE:\n"
        "CLAIM = " + json.dumps(user_claim) + "\n"
        "TOP_LABEL = " + verdict + ", TOP_PCT = " + str(top_pct) + "\n"
        "P_TRUE = " + str(p_true_pct) + "%, P_MIXED = " + str(p_mixed_pct) + "%, P_FALSE = " + str(p_false_pct) + "%\n\n"
        "EVIDENCE 1 (for body1):\n"
        "SOURCE1_NAME = " + str(ev1.get("title") or ev1.get("source") or "—") + "\n"
        "SOURCE1_URL = " + str(ev1.get("url") or "—") + "\n"
        "P1_ID = " + str(ev1.get("doc_id") or "—") + "\n"
        "Snippet: " + json.dumps((ev1.get("snippet") or "")[:500]) + "\n"
        "SHAP phrases (positive then negative): " + json.dumps(top_pos_tokens[:3]) + ", " + json.dumps(top_neg_tokens[:2]) + "\n\n"
        "EVIDENCE 2 (for body2):\n"
        "SOURCE2_NAME = " + str(ev2.get("title") or ev2.get("source") or "—") + "\n"
        "SOURCE2_URL = " + str(ev2.get("url") or "—") + "\n"
        "P2_ID = " + str(ev2.get("doc_id") or "—") + "\n"
        "Snippet: " + json.dumps((ev2.get("snippet") or "")[:500]) + "\n"
        "SHAP phrases: " + json.dumps(top_pos_tokens[:3]) + ", " + json.dumps(top_neg_tokens[:2]) + "\n\n"
        "CONCLUSION: Write one sentence that matches the verdict (true/false/mixed). If mixed, briefly say what parts differ.\n\n"
        "OUTPUT: JSON with exactly four keys: \"intro\", \"body1\", \"body2\", \"conclusion\". Each value is a string (that paragraph)."
    )

    # 5) Call LLM
    llm_raw = ""
    try:
        llm = _get_llm()
        llm_raw = await llm.generate_async(
            SYS_INSTRUCTION,
            payload,
            temperature=0.0,
            num_predict=2048,
        )
    except Exception as e:
        warnings.append({
            "code": "SUMMARY_LLM_FAILED",
            "message": "LLM summarization call failed; using fallback.",
        })
        llm_raw = ""

    # 6) Parse + validate, with optional repair
    llm_obj: Optional[Dict[str, Any]] = None
    attempt = 0
    while attempt <= MAX_JSON_REPAIR_RETRIES:
        llm_obj = _safe_json_parse(llm_raw)
        if llm_obj is not None and _validate_schema(llm_obj, LLM_JSON_SCHEMA):
            break
        if attempt >= MAX_JSON_REPAIR_RETRIES:
            llm_obj = None
            break
        try:
            repair_payload = (
                "INVALID_OUTPUT:\n" + (llm_raw or "") + "\n\n"
                "SCHEMA:\n" + json.dumps(LLM_JSON_SCHEMA)
            )
            llm_raw = await llm.generate_async(SYS_REPAIR_INSTRUCTION, repair_payload, temperature=0.0, num_predict=512)
        except Exception:
            llm_obj = None
            break
        attempt += 1

    # 7) Fallback if invalid
    fallback_intro = (
        f'The claim "{user_claim}" is predicted to be {verdict} with {top_pct}% confidence '
        f"(True {p_true_pct}% · Mixed {p_mixed_pct}% · False {p_false_pct}%)."
    )
    fallback_body = "Evidence limited; review sources."
    if llm_obj is None:
        llm_obj = {
            "intro": fallback_intro,
            "body1": fallback_body,
            "body2": "—",
            "conclusion": "See citations for details.",
        }

    # 8) Essay fields (ensure non-empty strings)
    def _str(s: str) -> str:
        return (s or "").strip() or "—"

    intro = _str(llm_obj.get("intro"))
    body1 = _str(llm_obj.get("body1"))
    body2 = _str(llm_obj.get("body2"))
    conclusion = _str(llm_obj.get("conclusion"))
    if not intro:
        intro = fallback_intro
    if not body1:
        body1 = fallback_body
    if not body2:
        body2 = "—"
    if not conclusion:
        conclusion = "See citations for details."

    # 9) Summary block: essay-style intro, body1, body2, conclusion
    summary = {
        "intro": intro,
        "body1": body1,
        "body2": body2,
        "conclusion": conclusion,
        "verdict": verdict,
        "confidence": confidence,
        "citations": citations[:2],
        "explainability": {
            "top_positive_tokens": top_pos_tokens[:TOPK_TOKENS],
            "top_negative_tokens": top_neg_tokens[:TOPK_TOKENS],
        },
    }

    _mark_latency(meta_out, "9_LLM_summarization", int(round((time.perf_counter() - t0) * 1000)))

    out = dict(step8_out)
    out["summary"] = summary
    out["meta"] = meta_out
    return out
