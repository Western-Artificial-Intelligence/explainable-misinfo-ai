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
├── popup.html
├── popup.js
├── styles.css
└── README.md
```

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

