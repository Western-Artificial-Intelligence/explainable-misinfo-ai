# TruthLens Pipeline — Implementation Guide

This guide walks you through implementing the 10-stage production pipeline and integrating it into your existing project.

---

## Current Project Structure (What You Have)

```
explainable-misinfo-ai-1/
├── api/
│   ├── main.py                    # FastAPI app
│   ├── routes/
│   │   ├── classify.py            # Dummy classify endpoint
│   │   ├── health.py
│   │   └── ollama_blackboxes.py   # Ollama LLM calls
│   └── utils/
│       └── cache.py
├── baseline_model/
│   ├── configs/
│   │   └── baseline.yaml
│   ├── data_utils/
│   │   ├── load_dataset.py
│   │   ├── tokenize.py
│   │   └── ...
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── infer_example.py
│   │   └── ...
│   └── models/                    # (may need to add)
│       └── domain_adversarial.py
├── data/
│   ├── unified_schema/
│   │   ├── cleanup_text.py
│   │   ├── article_scraper.py
│   │   └── ...
│   └── processed/
├── production_pipeline_instruction/
│   ├── 1_Ingest_claim/
│   ├── 2_RoBERTa_inference/
│   ├── ...
│   └── 10_Output/
├── notebooks/
├── requirements.txt
└── ...
```

---

## Target File Structure (What You're Building)

```
explainable-misinfo-ai-1/
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── classify.py            # REPLACE: full pipeline endpoint
│   │   ├── health.py
│   │   └── ollama_blackboxes.py
│   └── utils/
│       └── cache.py
│
├── pipeline/                      # NEW: production pipeline package
│   ├── __init__.py
│   ├── types.py                  # Shared data types (Request, Response, etc.)
│   ├── orchestrator.py           # Runs stages 1→10 in sequence
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage_01_ingest.py
│   │   ├── stage_02_roberta.py
│   │   ├── stage_03_routing.py
│   │   ├── stage_04_query_build.py
│   │   ├── stage_05_rag.py
│   │   ├── stage_06_mmr.py
│   │   ├── stage_07_relevance_gate.py
│   │   ├── stage_08_explainability.py
│   │   ├── stage_09_llm_summary.py
│   │   └── stage_10_output.py
│   │
│   ├── blackboxes/               # Wrappers around external services
│   │   ├── __init__.py
│   │   ├── roberta.py            # RoBERTa inference
│   │   ├── ollama.py             # LLM calls (or import from api)
│   │   └── embeddings.py         # Sentence transformers for MMR/RAG
│   │
│   └── config.py                 # Pipeline config (thresholds, model paths)
│
├── baseline_model/               # EXISTING: training & model
│   └── ... (unchanged)
│
├── production_pipeline_instruction/  # EXISTING: spec docs
│   └── ...
│
├── configs/                      # Optional: pipeline config YAML
│   └── pipeline.yaml
│
└── scripts/                      # Optional: CLI runners
    └── run_pipeline.py
```

---

## Implementation Phases

| Phase | Scope | Est. Time |
|-------|-------|------------|
| **Phase 1** | Stages 1–2 + 10, minimal API | 1–2 days |
| **Phase 2** | Stages 3–4 (routing, query) | 0.5 day |
| **Phase 3** | Stages 5–7 (RAG, MMR, gate) | 2–3 days |
| **Phase 4** | Stage 8 (SHAP) + Stage 9 (LLM summary) | 1–2 days |
| **Phase 5** | Polish, caching, error handling | 0.5 day |

---

## Phase 1: MVP (Stages 1, 2, 10)

**Goal:** Ingest claim → RoBERTa classify → return verdict + confidence.

### Step 1.1: Create `pipeline/` package

```bash
mkdir -p pipeline/stages pipeline/blackboxes
touch pipeline/__init__.py pipeline/types.py pipeline/config.py pipeline/orchestrator.py
touch pipeline/stages/__init__.py
touch pipeline/blackboxes/__init__.py
```

### Step 1.2: `pipeline/types.py` — Shared data structures

```python
"""Shared types for the pipeline. Each stage receives a dict and returns a dict (passthrough + new fields)."""
from typing import Any, TypedDict, Optional

# Minimal pipeline state (grows as it moves through stages)
PipelineState = dict[str, Any]

# Final public output (what the user sees)
class PublicOutput(TypedDict, total=False):
    claim: str
    verdict: str          # "false" | "mixed" | "true"
    confidence: float
    summary: str
    bullets: list[str]
    citations: list[dict]
    explainability: dict
```

### Step 1.3: `pipeline/config.py` — Config constants

```python
"""Pipeline configuration."""
import os

PIPELINE_VERSION = "0.1.0"
ROBERTA_CHECKPOINT = os.getenv("ROBERTA_CHECKPOINT", "baseline_outputs/baseline/checkpoints/best_model.pt")
ROBERTA_BACKBONE = "roberta-base"
T_CONTEXT = 256
LABELS_3WAY = ["false", "mixed", "true"]
```

### Step 1.4: `pipeline/stages/stage_01_ingest.py`

```python
"""Stage 1: Ingest and normalize user claim."""
import unicodedata
import re
from datetime import datetime, timezone

def collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def run(state: dict) -> dict:
    user_claim = state.get("user_claim", "").strip()
    if not user_claim:
        raise ValueError("INVALID_CLAIM_TEXT")
    
    raw = user_claim
    normalized = unicodedata.normalize("NFKC", raw)
    normalized = collapse_whitespace(normalized)
    if not normalized:
        raise ValueError("INVALID_CLAIM_TEXT")
    
    warnings = []
    if raw != normalized:
        warnings.append({"code": "TEXT_NORMALIZED", "message": "Text was normalized"})
    if len(normalized) > 2000:
        warnings.append({"code": "VERY_LONG_CLAIM", "message": "Claim is very long"})
    
    state["normalized_claim"] = normalized
    state["meta"] = state.get("meta", {}) | {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
        "warnings": warnings if warnings else None,
    }
    return state
```

### Step 1.5: `pipeline/blackboxes/roberta.py`

```python
"""RoBERTa inference blackbox. Supports claim-only (production) and claim+article (baseline)."""
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline_model.data_utils.tokenize import build_tokenizer
from baseline_model.models.domain_adversarial import DomainAdversarialClassifier

_model = None
_tokenizer = None

def _load_model(checkpoint: str, backbone: str = "roberta-base", device: str = "cpu"):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    _tokenizer = build_tokenizer(backbone)
    state = torch.load(checkpoint, map_location=device)
    _model = DomainAdversarialClassifier(
        backbone_name=backbone, num_sources=3, lambda_adv=0.5
    )
    _model.load_state_dict(state["model_state"])
    _model.to(device)
    _model.eval()
    return _model, _tokenizer

def roberta_infer(
    claim: str,
    article: str | None = None,
    checkpoint: str = "baseline_outputs/baseline/checkpoints/best_model.pt",
    backbone: str = "roberta-base",
    max_len: int = 256,
    device: str = "cpu",
) -> dict:
    """Returns: {h: tensor, label_logits: list, label_probs: list}"""
    model, tokenizer = _load_model(checkpoint, backbone, device)
    text = f"<CLAIM> {claim} </CLAIM>"
    if article:
        text += f" <ARTICLE> {article} </ARTICLE>"
    else:
        text += " <ARTICLE> </ARTICLE>"  # Empty article for claim-only
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits_label"].cpu().numpy()[0].tolist()
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy().tolist()
    h = out.get("pooled")  # or last_hidden_state[:,0,:] if needed
    return {
        "h": h,
        "label_logits": logits,
        "label_probs": probs,
    }
```

**Note:** If `DomainAdversarialClassifier` or claim-only mode doesn't exist, use a **fallback** that returns random/dummy probs until the model is ready:

```python
def roberta_infer(claim: str, article=None, **kwargs) -> dict:
    # Fallback for missing model
    return {
        "h": None,
        "label_logits": [0.0, 0.0, 0.0],
        "label_probs": [0.33, 0.34, 0.33],
    }
```

### Step 1.6: `pipeline/stages/stage_02_roberta.py`

```python
"""Stage 2: RoBERTa inference."""
from pipeline.blackboxes.roberta import roberta_infer
from pipeline.config import ROBERTA_CHECKPOINT, ROBERTA_BACKBONE, T_CONTEXT, LABELS_3WAY

def run(state: dict) -> dict:
    normalized = state["normalized_claim"]
    pred = roberta_infer(
        claim=normalized,
        article=None,
        checkpoint=ROBERTA_CHECKPOINT,
        backbone=ROBERTA_BACKBONE,
        max_len=T_CONTEXT,
    )
    probs = pred["label_probs"]
    class_id = int(max(range(3), key=lambda i: probs[i]))
    class_name = LABELS_3WAY[class_id]
    confidence = probs[class_id]
    
    state["roberta"] = {
        "label": {
            "logits": pred["label_logits"],
            "probs": probs,
            "class_id": class_id,
            "class_name": class_name,
        },
        "confidence": confidence,
        "model": {"name": "roberta-base", "revision": "prod", "ctx_tokens": 512},
        "tokenization": {"truncated": False, "input_tokens": len(normalized.split())},
    }
    return state
```

### Step 1.7: `pipeline/stages/stage_10_output.py`

```python
"""Stage 10: Final output packaging."""
from datetime import datetime, timezone

def run(state: dict) -> dict:
    roberta = state["roberta"]
    public = {
        "claim": state["user_claim"],
        "verdict": roberta["label"]["class_name"],
        "confidence": roberta["confidence"],
        "summary": f"{roberta['label']['class_name']} (confidence {roberta['confidence']:.2f})",
        "bullets": [],
        "citations": [],
        "explainability": {"top_positive_tokens": [], "top_negative_tokens": []},
    }
    return {
        "request_id": state["request_id"],
        "claim_id": state["claim_id"],
        "public": public,
        "meta": {
            "version": state["meta"]["version"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
```

### Step 1.8: `pipeline/orchestrator.py` — MVP version

```python
"""Pipeline orchestrator. Runs stages in sequence."""
import uuid
from pipeline.stages import stage_01_ingest, stage_02_roberta, stage_10_output

STAGES = [
    ("ingest", stage_01_ingest.run),
    ("roberta", stage_02_roberta.run),
    ("output", stage_10_output.run),
]

def run_pipeline(user_claim: str) -> dict:
    request_id = str(uuid.uuid4())
    claim_id = str(uuid.uuid4())
    state = {
        "request_id": request_id,
        "claim_id": claim_id,
        "user_claim": user_claim,
    }
    for name, stage_fn in STAGES:
        state = stage_fn(state)
        if name == "output":
            return state  # Final stage returns output, not state
    return state
```

### Step 1.9: `pipeline/stages/__init__.py`

```python
from . import stage_01_ingest
from . import stage_02_roberta
from . import stage_10_output
```

### Step 1.10: Wire up `api/routes/classify.py`

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pipeline.orchestrator import run_pipeline

router = APIRouter()

class InputText(BaseModel):
    text: str

@router.post("/classify")
def classify_text(payload: InputText):
    try:
        result = run_pipeline(payload.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Step 1.11: Add pipeline to `PYTHONPATH`

If you run from project root:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# or on Windows:
set PYTHONPATH=%PYTHONPATH%;%CD%
```

Or add to `api/main.py` at top:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

---

## Phase 2: Routing + Query Building

### Step 2.1: `pipeline/stages/stage_03_routing.py`

Implement logic from `3_Routing_policy/process_pseudo.txt`:
- Read `roberta.confidence`, `roberta.tokenization.truncated`
- Set `routing.tier` (standard/expanded/conservative)
- Set `routing.rag` knobs (M, N, K, MIN_REL, use_query_expansion, allow_web_search)

### Step 2.2: `pipeline/stages/stage_04_query_build.py`

- If `use_query_expansion`: call Ollama to paraphrase claim 3×
- Filter paraphrases (numbers match, cosine ≥ 0.85 if embed available)
- Add `query_plan.queries` = [original] + kept paraphrases

**Dependency:** You need `api.routes.ollama_blackboxes.ollama_logic` or a shared `ollama_client` module.

---

## Phase 3: RAG (Stages 5–7)

### Step 3.1: Add dependencies

```
# requirements.txt additions
sentence-transformers>=2.2.0
requests>=2.28.0
googlesearch-python>=1.2.0   # or SerpAPI / Custom Search API
beautifulsoup4>=4.12.0
```

### Step 3.2: `pipeline/blackboxes/embeddings.py`

- Load `sentence-transformers/all-MiniLM-L6-v2`
- `embed(text: str) -> list[float]`
- Optional: cache by text hash

### Step 3.3: `pipeline/stages/stage_05_rag.py`

- Google Search API (or SerpAPI) for each query in `query_plan.queries`
- Fetch URLs, extract paragraphs (BeautifulSoup or readability-lxml)
- Score chunks (cosine(embed(claim), embed(chunk)))
- Build `rag_candidates.items` (top M)

**Simplification:** Skip real web fetch initially; use a mock that returns a few hardcoded chunks. Add real fetch later.

### Step 3.4: `pipeline/stages/stage_06_mmr.py`

- Embed claim + all candidates
- MMR loop: λ*sim(d,Q) - (1-λ)*max sim(d,selected)
- Output `mmr_selected.items` (top N)

### Step 3.5: `pipeline/stages/stage_07_relevance_gate.py`

- Filter by `cosine(claim_emb, chunk_emb) >= min_relevance`
- Sort, take top K → `evidence_topk`

---

## Phase 4: Explainability + LLM Summary

### Step 4.1: `pipeline/stages/stage_08_explainability.py`

- Use SHAP (`shap.Explainer`) on RoBERTa
- Input: normalized claim
- Output: `shap_explainability.tokens`, `top_tokens.positive`, `top_tokens.negative`

**Dependency:** `shap>=0.42.0` (already in requirements)

### Step 4.2: `pipeline/stages/stage_09_llm_summary.py`

- Build prompt: claim + verdict + confidence + evidence snippets + SHAP tokens
- Call Ollama with JSON schema
- Parse, validate, fallback if invalid
- Output `summary` block

---

## Phase 5: Full Orchestrator

Update `pipeline/orchestrator.py` to run all 10 stages:

```python
STAGES = [
    ("ingest", stage_01_ingest.run),
    ("roberta", stage_02_roberta.run),
    ("routing", stage_03_routing.run),
    ("query_build", stage_04_query_build.run),
    ("rag", stage_05_rag.run),
    ("mmr", stage_06_mmr.run),
    ("relevance_gate", stage_07_relevance_gate.run),
    ("explainability", stage_08_explainability.run),
    ("llm_summary", stage_09_llm_summary.run),
    ("output", stage_10_output.run),
]
```

---

## Quick Reference: Stage Input/Output

| Stage | Input (state) | Adds to state |
|-------|----------------|---------------|
| 1 | user_claim | normalized_claim, meta |
| 2 | normalized_claim | roberta |
| 3 | roberta | routing |
| 4 | routing, query_plan? | query_plan |
| 5 | query_plan, routing | rag_candidates |
| 6 | rag_candidates | mmr_selected |
| 7 | mmr_selected | evidence_topk |
| 8 | roberta, normalized_claim | shap_explainability |
| 9 | roberta, evidence_topk, shap | summary |
| 10 | summary, ... | — (returns final output) |

---

## Checklist

- [ ] Create `pipeline/` folder structure
- [ ] Implement `types.py`, `config.py`
- [ ] Implement `stage_01_ingest.py`
- [ ] Implement `roberta.py` blackbox (or fallback)
- [ ] Implement `stage_02_roberta.py`
- [ ] Implement `stage_10_output.py` (MVP)
- [ ] Implement `orchestrator.py` (MVP: 1→2→10)
- [ ] Wire `classify` route to `run_pipeline`
- [ ] Test: `POST /classify` with `{"text": "Eating carrots fixes eyesight"}`
- [ ] Add stages 3–9 incrementally
- [ ] Add caching, error handling, logging

---

## Running the API

```bash
# Terminal 1: (Optional) Start Ollama for Stage 9 later
ollama run qwen3:4b

# Terminal 2: Run API
cd explainable-misinfo-ai-1
uvicorn api.main:app --reload
```

Test:
```bash
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d "{\"text\": \"Vaccines cause autism\"}"
```

---

## What Was Created (MVP Files)

The following files were created for you:

```
pipeline/
├── __init__.py
├── config.py
├── orchestrator.py
├── types.py
├── stages/
│   ├── __init__.py
│   ├── stage_01_ingest.py
│   ├── stage_02_roberta.py
│   └── stage_10_output.py
└── blackboxes/
    ├── __init__.py
    └── roberta.py
```

And updated:
- `api/main.py` — adds project root to path
- `api/routes/classify.py` — uses `run_pipeline` instead of dummy logic

**Note:** Stage 2 uses a fallback (dummy probs) if:
- `baseline_model.models.domain_adversarial` doesn't exist, or
- `baseline_outputs/.../best_model.pt` doesn't exist

Once you train a model and have the checkpoint, set `ROBERTA_CHECKPOINT` or place the file at the expected path.
