
# TruthLens Frontend

This frontend is a Vite + React app with two backend-powered features:

1. Text classification (`/classify`)
2. Document Q&A over uploaded/local files or URL content (`/documents/*`)

## How it works

1. User enters text and clicks **Analyze with AI**.
2. Frontend sends `POST /classify` to FastAPI.
3. Backend returns:
   - `label` (for example: `false`, `mixed`, `factual`)
   - `confidence` (0..1 or 0..100)
   - `explanation` (string)
4. Frontend maps backend labels into UI categories:
   - `false`/`misinformation` -> `MISINFORMATION`
   - `factual`/`reliable`/`true` -> `RELIABLE`
   - `mixed`/`opinion` -> `OPINION`
   - anything else -> `NEUTRAL`

## Document Q&A flow

1. In the **Document Q&A** section, load content using one of:
   - local file upload (text-like files: `.txt`, `.md`, `.csv`, `.json`, `.html`, `.xml`, `.log`)
   - URL ingestion
2. Frontend sends:
   - `POST /documents/upload` for local file text
   - `POST /documents/url` for URL scraping/parsing
3. Backend stores parsed text in memory and returns a `document_id`.
4. Ask a question and frontend sends `POST /documents/ask` with `question` + `document_id`.
5. Backend returns:
   - `answer`
   - `snippets` (top supporting sentences)
   - `confidence`

## Run locally

## 1. Start backend (FastAPI)

From repo root:

```bash
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Backend should run at `http://127.0.0.1:8000`.

If you also need advanced routes (audio/image/explainability), install:

```bash
pip install -r requirements-optional.txt
```

## 2. Start frontend (Vite)

In a new terminal:

```bash
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai/frontend
npm install
npm run dev
```

Frontend should run at `http://localhost:3000`.

By default, frontend calls `http://127.0.0.1:8000` for both classify and document endpoints.

## Optional: configure API base URL

Create `frontend/.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then restart frontend.

## Current fallback behavior (no model yet)

Backend classification uses a fallback predictor when no model is configured.
That fallback returns random but valid outputs, so frontend still works end-to-end.

You will also see terminal logs in backend like:

- `INPUT: 'your text'`
- `OUTPUT: {...}`

## Connect a real AI model later

When model inference is ready:

1. Implement a Python module with a callable function that accepts `text: str` and returns:

```python
{
  "label": "false" | "mixed" | "factual" | "reliable" | "opinion" | "neutral",
  "confidence": 0.0,  # can be 0..1 or 0..100
  "explanation": "..."
}
```

2. Example file (repo root): `api/model_inference.py`

```python
def predict(text: str) -> dict:
    # replace with real model inference
    return {
        "label": "factual",
        "confidence": 0.91,
        "explanation": "Model-based explanation",
    }
```

3. Set env vars before running backend:

```bash
export TRUTHLENS_MODEL_MODULE=api.model_inference
export TRUTHLENS_MODEL_FUNCTION=predict
uvicorn api.main:app --reload
```

If loading/inference fails, backend automatically falls back to random outputs so the UI does not break.
  
