# TruthLens Pipeline Architecture Analysis

> **TruthLens** is an NLP-based misinformation detection tool that classifies short-form content (tweets, WhatsApp, TikTok, etc.) as misinformation, conspiracy, or factual—and explains *why* using explainable AI and RAG-sourced counter-evidence.

---

## Pipeline Overview (10 Stages)

```
User Claim → [1] Ingest → [2] RoBERTa → [3] Routing → [4] Query Build → [5] RAG → [6] MMR → [7] Relevance Gate → [8] SHAP → [9] LLM Summary → [10] Output
```

---

## Stage-by-Stage Breakdown

### **1. Ingest Claim** (`1_Ingest_claim`)

**Purpose:** Validate and normalize raw user input.

**What it does:**
- Validates that `user_claim` exists and is non-empty
- Unicode normalizes (NFKC), trims, collapses whitespace
- Optionally removes zero-width chars / normalizes newlines
- Emits warnings for normalization, very long claims
- Assigns `request_id`, `claim_id` (server-side)

**Schema:** `user_claim` (string) → `normalized_claim`, `meta`, `warnings`

**TruthLens relevance:**
- Handles informal text: emoji, slang, casual grammar (via normalization)
- Prepares content for downstream models
- Enables **Informal Text Mode** (your TikTok/WhatsApp edge)

---

### **2. RoBERTa Inference** (`2_RoBERTa_inference`)

**Purpose:** Core misinformation classifier. The main "blackbox" model.

**What it does:**
- Wraps claim in `<CLAIM>...</CLAIM>` sentinel tokens
- Tokenizes (max 256/512 tokens), runs RoBERTa
- Produces **3-way label**: `false` | `mixed` | `true`
- Outputs: logits, probs, class_id, confidence, CLS embedding
- Optional domain prediction (5-way) for routing/risk
- Tracks tokenization (truncated?, input_tokens)

**Schema:** `normalized_claim` → `roberta.label`, `roberta.confidence`, `roberta.embedding`

**TruthLens relevance:**
- **Main classification** (misinfo vs factual vs mixed)
- **Risk meter** can use `confidence` + domain (e.g., medical = higher risk)
- Embedding used later for MMR and relevance

---

### **3. Routing Policy** (`3_Routing_policy`)

**Purpose:** Set RAG knobs based on RoBERTa confidence and flags.

**What it does:**
- Uses confidence thresholds: `LO_CONF=0.55`, `HI_CONF=0.80`
- Three tiers:
  - **expanded** (conf &lt; 0.55): more candidates, broader search
  - **conservative** (between or truncated): moderate search
  - **standard** (conf ≥ 0.80, not truncated): tighter search
- Knobs: `M` (pool size), `N` (MMR top), `K` (final top-K), `MIN_REL`, `QEXP`, `WEB`
- Label tweaks: `false` → enable query expansion; `true` + high conf → reduce M

**Schema:** Adds `routing.tier`, `routing.rag.*`

**TruthLens relevance:**
- **Intent-aware** behavior: uncertain claims get more evidence; confident ones get less
- Keeps latency and cost under control

---

### **4. Query Building** (`4_Query_Building`)

**Purpose:** Build search queries—original + meaning-preserving paraphrases.

**What it does:**
- Base query = `normalized_claim`
- If `use_query_expansion`: calls LLM to generate 3 paraphrases
- Filters paraphrases by:
  - Numbers must match
  - Cosine similarity ≥ 0.85 to original (meaning-preservation)
- Produces 1–4 queries for RAG

**Schema:** Adds `query_plan.queries`, `query_plan.paraphrase_meta`

**TruthLens relevance:**
- Improves retrieval for casual/indirect wording
- Supports **Informal Text Mode** by covering alternate phrasings

---

### **5. RAG Retrieval** (`5_RAG`)

**Purpose:** Web search → fetch pages → extract text chunks. The **Source Tracker** stage.

**What it does:**
- Uses Google Search API (or similar) to get URLs from queries
- Fetches pages with SSRF hardening, robots.txt, rate limits
- Extracts paragraphs (or fallback windows)
- Scores chunks vs claim (bi-encoder or heuristic)
- Builds candidate pool of size M (~50–100)
- Handles paywalls, JS-heavy pages, PDFs (optional)

**Schema:** Adds `rag_candidates.items` with `chunk_id`, `text`, `doc`, `score`

**TruthLens relevance:**
- **Source Tracker:** links to Snopes, Wikipedia, WHO, etc. via `doc.url`, `doc.source`, `doc.title`
- Real counter-evidence from trusted sources

---

### **6. MMR Selection** (`6_MMR_selection`)

**Purpose:** Pick diverse, relevant chunks via Maximal Marginal Relevance.

**What it does:**
- Embeds claim + all candidates (e.g., `all-MiniLM-L6-v2`)
- MMR: `λ * sim(d,Q) - (1-λ) * max sim(d,selected)`
- Balances relevance to claim vs diversity (avoid redundant snippets)
- Caches embeddings for Stage 7
- Selects top-N chunks

**Schema:** Adds `mmr_selected.items`

**TruthLens relevance:**
- Avoids multiple near-duplicate Snopes results
- Improves quality of citations shown to users

---

### **7. Light Relevance Gate** (`7_Light_relevance_gate`)

**Purpose:** Filter out obviously irrelevant chunks; keep top-K.

**What it does:**
- Reuses embeddings from Stage 6
- Keeps chunks with `cosine(claim_emb, chunk_emb) >= min_relevance` (T)
- Sorts by relevance, takes top-K
- Output: `evidence_topk`

**Schema:** Adds `evidence_topk.items` (ranked, with relevance_score)

**TruthLens relevance:**
- Clean **citations** list for report
- Reduces noise before LLM summarization

---

### **8. Explainability (SHAP)** (`8_Explainability`)

**Purpose:** Explain *why* RoBERTa predicted what it did. The **Explainable AI** stage.

**What it does:**
- Uses SHAP (partition explainer) on RoBERTa
- Token-level attributions for the predicted class
- Returns top-K positive and negative tokens (most influential)
- Caches explainer for performance

**Schema:** Adds `shap_explainability.tokens`, `top_tokens.positive`, `top_tokens.negative`

**TruthLens relevance:**
- **Explainable AI:** “These words pushed the model toward FALSE”
- Educational: users see which phrases triggered the verdict
- **Report Export:** highlights in shareable breakdowns

---

### **9. LLM Summarization** (`9_LLM_summarization`)

**Purpose:** Turn evidence + SHAP into a user-facing summary. Uses Chat AI blackbox (Ollama/Qwen).

**What it does:**
- Takes: verdict, confidence, evidence snippets, SHAP top tokens
- Prompts LLM for structured JSON:
  - `short`: one-line summary
  - `bullets`: 0–4 explanation bullets
  - `used_doc_ids`: which citations were used
  - `evidence_quality`: strong | medium | weak
- Validates JSON schema strictly; retry with repair prompt if needed
- Fallback if invalid: verdict + confidence + “evidence limited”
- Restricts citations to provided evidence only (no hallucination)

**Schema:** Adds `summary.verdict`, `summary.short`, `summary.bullets`, `summary.citations`, `summary.explainability`

**TruthLens relevance:**
- **Report Export:** bullets + citations → PDF/infographic
- Transparent about evidence strength
- Grounded in retrieved sources only

---

### **10. Output** (`10_Output`)

**Purpose:** Final API response for the frontend.

**What it does:**
- Builds `public` block: claim, verdict, confidence, summary, bullets, citations, explainability
- Deduplicates citations by URL
- Optional `debug` block for internal diagnostics
- `meta`: version, completed_at, latency_ms_total

**Schema:** `{ request_id, claim_id, public, meta, debug? }`

**TruthLens relevance:**
- Clean shape for **Discord Bot** or **Chrome Extension**
- `public` is what users see; `debug` for admins

---

## Blackboxes (from README)

| Blackbox | Input | Output |
|----------|-------|--------|
| **RoBERTa** | ONE string (claim) | label logits/probs, CLS embedding, domain (optional) |
| **Chat AI** | TWO strings (system instruction, input) | ONE string (result) |
| **Storage** | Final JSON from Stage 10 | (stores for retraining) |

---

## TruthLens Feature ↔ Pipeline Mapping

| Feature | Pipeline Stage(s) |
|---------|-------------------|
| **Intent Classifier** | Stage 2 (domain), future extension for fearmongering/satire |
| **Risk Meter** | Stage 2 (confidence) + Stage 3 (routing) |
| **Source Tracker** | Stages 5–7 (RAG → MMR → evidence_topk) |
| **Informal Text Mode** | Stages 1, 4 (normalization + query expansion) |
| **Report Export** | Stages 9, 10 (summary + public) |
| **Explainability** | Stage 8 (SHAP) |
| **Discord/Chrome** | Stage 10 (public output) |

---

## Data Flow Summary

```
user_claim
  → normalized_claim (Stage 1)
  → roberta.label {false|mixed|true} + confidence + embedding (Stage 2)
  → routing.rag.* (Stage 3)
  → query_plan.queries (Stage 4)
  → rag_candidates (Stage 5)
  → mmr_selected (Stage 6)
  → evidence_topk (Stage 7)
  → shap_explainability (Stage 8)
  → summary (Stage 9)
  → public {claim, verdict, confidence, summary, bullets, citations, explainability} (Stage 10)
```

---

## Notes for Implementation

1. **Determinism:** All stages define tie-breaking rules (e.g., EPS=1e-9, stable sort keys).
2. **Caching:** Embeddings (Stage 6) and SHAP explainer (Stage 8) are cached for speed.
3. **Safety:** Stage 5 has SSRF hardening, robots.txt, rate limits, paywall handling.
4. **Ollama:** README recommends Qwen3:4b for the Chat AI blackbox (Stage 9).
