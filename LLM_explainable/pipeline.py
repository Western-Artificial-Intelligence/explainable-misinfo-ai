from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .schemas import EvidenceSnippet, FinalResult, GuardrailResult, GeneratorResult, VerifierResult
from .utils.cache import DiskCache

from .step1_verifier import VerifierPipeline
from .step2_rag import gather_evidence
from .step3_guardrail import evidence_guardrail
from .step4_generator import GeneratorLLM


def _normalize_claim(claim: str) -> str:
    return " ".join(str(claim).split()).strip()


def _to_snippets(rag_out: Dict[str, Any]) -> List[EvidenceSnippet]:
    snippets: List[EvidenceSnippet] = []
    for s in rag_out.get("snippets", []):
        text = str(s.get("text") or s.get("snippet") or "")
        snippets.append(
            EvidenceSnippet(
                eid=str(s.get("eid", "")),
                url=str(s.get("url", "")),
                title=str(s.get("title", "")),
                text=text,
                score=float(s.get("score", 0.0)),
            )
        )
    return snippets


def _normalize_citations(citations: Any) -> List[str]:
    if citations is None:
        return []
    if isinstance(citations, str):
        return [citations.strip()]
    if isinstance(citations, (list, tuple)):
        return [str(x).strip() for x in citations if str(x).strip()]
    return [str(citations).strip()]


def run_claim(
    claim: str,
    *,
    verifier_model_path: str,
    generator_model_name: str,
    cache_dir: str = ".cache/rag",
    use_cache: bool = True,
    rag_kwargs: Optional[Dict[str, Any]] = None,
) -> FinalResult:
    """
    End-to-end verification pipeline:
    0 normalize -> 1 verifier(prior) -> 2 RAG(always) -> 3 guardrail -> 4 generator(optional) -> 5 pack
    """
    rag_kwargs = rag_kwargs or {}
    claim_n = _normalize_claim(claim)

    # 1) Verifier (prior)
    verifier = VerifierPipeline(model_path=verifier_model_path)
    v = verifier.predict(claim_n)
    verifier_res = VerifierResult(
        label_id=int(v.label_id),
        label=str(v.label),
        conf=float(v.conf),
        score=float(v.score),
        probs=dict(v.probs),
    )

    # 2) RAG (always) with caching
    cache = DiskCache(cache_dir)
    cache_key = cache.key_for_claim(claim_n)

    rag_out: Dict[str, Any]
    if use_cache:
        cached = cache.get_json(cache_key)
        if cached is not None:
            rag_out = cached
        else:
            rag_out = gather_evidence(claim_n, **rag_kwargs)
            cache.set_json(cache_key, rag_out)
    else:
        rag_out = gather_evidence(claim_n, **rag_kwargs)

    evidence = _to_snippets(rag_out)
    urls = sorted({e.url for e in evidence if e.url})

    # 3) Guardrail
    # If no evidence at all, hard stop
    if not evidence:
        guard = GuardrailResult(
            status="INSUFFICIENT_EVIDENCE",
            has_contradiction=False,
            used_eids=[],
            metrics={"reason": "no_snippets_returned"},
        )
        return FinalResult(
            claim=claim_n,
            label_id=verifier_res.label_id,
            label=verifier_res.label,
            conf=verifier_res.conf,
            score=verifier_res.score,
            verdict_status="INSUFFICIENT_EVIDENCE",
            citations=[],
            evidence=evidence,
            urls=urls,
            debug={
                "verifier": asdict(verifier_res),
                "guardrail": asdict(guard),
                "rag_meta": rag_out.get("meta", {}),
            },
        )

    g = evidence_guardrail(claim_n, [asdict(e) for e in evidence])
    guard = GuardrailResult(
        status=str(getattr(g, "status", "INSUFFICIENT_EVIDENCE")),
        has_contradiction=bool(getattr(g, "has_contradiction", False)),
        used_eids=list(getattr(g, "used_eids", [])),
        metrics=dict(getattr(g, "metrics", {})),
    )

    if guard.status != "OK":
        return FinalResult(
            claim=claim_n,
            label_id=verifier_res.label_id,
            label=verifier_res.label,
            conf=verifier_res.conf,
            score=verifier_res.score,
            verdict_status="INSUFFICIENT_EVIDENCE",
            citations=_normalize_citations(guard.used_eids),
            evidence=evidence,
            urls=urls,
            debug={
                "verifier": asdict(verifier_res),
                "guardrail": asdict(guard),
                "rag_meta": rag_out.get("meta", {}),
            },
        )

    # 4) Generator
    # Robust init in case your GeneratorLLM signature differs
    try:
        gen = GeneratorLLM(model_name=generator_model_name)
    except TypeError:
        gen = GeneratorLLM(generator_model_name)  # fallback positional

    gen_out = gen.generate(
        claim=claim_n,
        snippets=[asdict(e) for e in evidence],
        verifier_label_id=verifier_res.label_id,
        contradiction=guard.has_contradiction,
    )

    generator_res = GeneratorResult(mode=str(gen_out.mode), content=dict(gen_out.content))

    # 5) Packager
    explanation = generator_res.content.get("explanation")
    correction = generator_res.content.get("correction")
    citations = _normalize_citations(generator_res.content.get("citations")) or _normalize_citations(guard.used_eids)

    return FinalResult(
        claim=claim_n,
        label_id=verifier_res.label_id,
        label=verifier_res.label,
        conf=verifier_res.conf,
        score=verifier_res.score,
        verdict_status="OK",
        explanation=explanation,
        correction=correction,
        citations=citations,
        evidence=evidence,
        urls=urls,
        debug={
            "verifier": asdict(verifier_res),
            "guardrail": asdict(guard),
            "generator": {"mode": generator_res.mode},
            "rag_meta": rag_out.get("meta", {}),
        },
    )
