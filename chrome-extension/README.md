# TruthLens Chrome Extension

TruthLens is a Manifest V3 Chrome extension for real-time misinformation analysis using your FastAPI backend.

Backend target:

- `POST http://localhost:8000/predict`
- Request body: `{"text":"..."}`
- Expected response:
  - `prediction`: `REAL` or `FAKE` (or equivalent values)
  - `confidence`: float (0..1 or 0..100)
  - `explanation`: string

## File Structure

```text
chrome-extension/
├── manifest.json
├── background.js
├── content.js
├── audio-capture.js      # Realtime tab-audio capture + transcript merge/dedupe
├── capture-worklet.js    # AudioWorklet processor (16 kHz mono PCM)
├── popup.html
├── popup.js
├── styles.css
└── README.md
```

## Realtime transcript – tech stack

- **Capture:** Web Audio API (content script); `createMediaElementSource` / `createMediaStreamSource(captureStream())`; AudioWorklet or ScriptProcessorNode → 16 kHz mono Int16 PCM; rolling buffer, WAV chunks every ~2 s → `POST /api/audio/transcribe-file`.
- **Backend:** Whisper (openai-whisper, base) for each chunk; returns `full_text` (and segments).
- **Dedupe / merge (no external libs):** Implemented in `audio-capture.js` only, vanilla JavaScript. **No embeddings, no cosine similarity** — overlap is found with token alignment and one set-based similarity.
  - **Tokenization:** `normalizeText` (collapse whitespace, trim) → split on spaces; each word normalized to a **key**: lowercase, strip non‑alphanumeric (`[^a-z0-9]+` removed) for matching; **raw** form kept for display.
  - **Context:** Last 120 words of current transcript (`tailWords(emittedText, 120)`) passed as previous context for each new chunk.
  - **Overlap detection (main):** **Suffix-of-prev vs prefix-of-next** — position-by-position token comparison (no vectors). Try overlap lengths `k` from `min(60, prevLen, nextLen)` down to 4; allow up to 15% token mismatches (`mismatchBudget = floor(k * 0.15)`) to handle ASR rephrasing; first valid `k` → **delta** = tokens after the overlap (drop one more if duplicate boundary token); delta length capped at 30 tokens.
  - **Fallbacks:** (1) If normalized `next` is substring of `prev` → emit nothing. (2) Anchor: find prefix of `next` (5–10 tokens) in `prev`, emit remainder. (3) **Jaccard similarity** (set overlap: |A∩B|/|A∪B|) on last 16 tokens of next vs last 40 of prev — if > 0.7, suppress (treat as repeat). This is the only similarity metric used; no cosine.
  - **Merge:** `mergeTranscript(prev, delta)` = `normalize(prev) + " " + normalize(delta)`; result length capped at `TLX_LIVE_MAX_CHARS` (keep trailing).
  - **State:** `TLX_LIVE_STATE.emittedText` (cumulative), `lastWindowText`; persisted per tab in `chrome.storage.local` under `tlx_live_${url}`.

## What It Does

1. Highlight + right click:
   - Select text on any page
   - Right-click -> `Analyze with TruthLens`
   - Floating result card appears near selection

2. Analyze entire page:
   - Popup button `Analyze This Page`
   - Parses visible headings, paragraphs, and tweet-like text blocks
   - Calls backend per block
   - Injects REAL/FAKE badges + confidence + tooltip explanation
   - Uses `MutationObserver` for dynamic pages (including X/Twitter-like feeds)
   - Toggle with `Remove Overlays` / `Show Overlays`

3. Analyze documents in popup:
   - Upload `.txt`, `.pdf`, `.docx`
   - Optional manual textarea input
   - Splits into paragraphs and sends each paragraph to backend
   - Displays paragraph-level prediction, confidence, explanation

## Load Unpacked Extension

1. Open Chrome and go to `chrome://extensions`.
2. Enable `Developer mode` (top-right).
3. Click `Load unpacked`.
4. Select:
   - `/Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai/chrome-extension`

## Run Backend

From repo root:

```bash
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai
source .venv/bin/activate
uvicorn api.main:app --reload
```

Ensure backend serves `POST /predict` at `http://localhost:8000/predict`.

## Test Checklist

1. Highlight feature:
   - Open any webpage
   - Highlight text
   - Right-click -> `Analyze with TruthLens`
   - Verify floating result card appears

2. Popup selected text feature:
   - Highlight text on page
   - Open extension popup
   - Click `Analyze Selected Text`
   - Verify result in popup

3. Page analysis:
   - Open article page or X/Twitter feed
   - In popup click `Analyze This Page`
   - Verify colored overlays and badges
   - Click `Remove Overlays` and verify cleanup

4. Document analysis:
   - Upload `.txt`, `.pdf`, or `.docx`
   - Optionally add manual text
   - Click `Analyze Document`
   - Verify paragraph-level result cards

