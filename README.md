# explainable-misinfo-ai

cd explainable-misinfo-ai 
source .venv/bin/activate  
git status
git log --oneline -5

Open Command Palette → Cmd + Shift + P 
Search: Python: Select Interpreter
Choose the one that points to your venv


# TruthLens — Misinfo Classifier + Debunker (MVP)

Shining a light on misinformation — with explainable AI.

## Quick Start (Dev Setup)

### Requirements
- Python 3.10+ (recommended 3.11)
- Node 18+ (only if using the React UI; skip for Streamlit)
- Git, Git LFS (for models)
- (Optional) CUDA 11+ if you have an NVIDIA GPU

### 1) Clone & set up
```bash
git clone https://github.com/<org>/<repo>.git
cd <repo>
# Python env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# Pre-commit hooks (lint/format on commit)
pre-commit install

**2) Create a .env in the repo root:**
HF_HOME=.cache/huggingface
HF_TOKEN=<optional_hf_token_if_private>
MODEL_NAME=distilbert-base-uncased
NUM_LABELS=3
MAX_LEN=256
LR=2e-5
BATCH=16
EPOCHS=3
API_SECRET=<choose_a_secret_for_backend_auth>

**3) Put CSVs here:**
data/
  train.csv    # text,label
  valid.csv
  test.csv
**
4)To generate them from source datasets, run:**
python scripts/prepare_liar.py
python scripts/prepare_fakehealth.py

**5)Tech Summary**

Model: DistilBERT/RoBERTa (Hugging Face Transformers)

Explainability: LIME/SHAP → token/phrase importances

Backend: FastAPI + Uvicorn; CORS enabled; simple HMAC auth

Cache: In-memory (MVP) → Redis (Phase 2)

UI: Streamlit (MVP) or React (Phase 2)

Hosting: HF Spaces / Render / Railway

**6) Add these files to your repo root:**

**`requirements.txt`**
transformers>=4.44
datasets
accelerate
evaluate
scikit-learn
lime
shap
fastapi
uvicorn[standard]
python-dotenv
pydantic
pre-commit
ruff
black
isort
pytest
yaml

**7) add these to .gitignore**

.venv/
.env
__pycache__/
.cache/
runs/
models/
data/*
!data/.gitkeep

**8) PULL_REQUEST_TEMPLATE.md**
## What
- 

## Why
- 

## Test Plan
- [ ] Unit tests pass
- [ ] Manual check (endpoint / UI)

## Screenshots
(If UI)

## Risks / Rollback
```
##Fast API Framework
api/
├── main.py # Core FastAPI app
├── routes/
│ ├── classify.py # /classify endpoint
│ └── health.py # /healthz endpoint
├── utils/
│ └── cache.py # In-memory caching logic
└── init.py

- `main.py` → Launches the API and connects routes  
- `routes/classify.py` → Handles incoming classification requests  
- `utils/cache.py` → Stores previously computed results to save time  

/healthz (GET)
Returns a simple message: "TruthLens backend is healthy!"
Useful for monitoring if the API is running

/classify (POST)
Input: JSON with a text field
Output: JSON with:
label: factual | mixed | false
confidence: float between 0 and 1

e.g.
POST /classify
{
    "text": "Is AI dangerous?"
}

Response:
{
    "label": "factual",
    "confidence": 0.9,
    "explanation": "Dummy placeholder explanation"
}

explanation: short text explaining the decision
## How Caching Works
A user asks a question (`"Is AI dangerous?"`).
2. The computer first checks a **database** (cache) to see if they've answered it before.  
   - ✅ If found → quickly gives the answer from the database.  
   - ❌ If not → looks it up (runs the model), writes the answer in the database, then gives it to the user.

##How to Start 
source .venv/bin/activate
uvicorn api.main:app --reload

### To Test /classify endpoint 
Type into new terminal
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"text":"Is AI dangerous?"}'

Expected output (dummy):
{
  "label": "factual",
  "confidence": 0.92,
  "explanation": "This is a placeholder explanation for 'factual' classification."
}

##To Preview Markdown
Ctrl+Shift+V