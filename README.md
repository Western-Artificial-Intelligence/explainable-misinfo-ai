# 🧠 TruthLens — Explainable Misinformation AI (MVP)

> **Shining a light on misinformation — with explainable AI.**

---

## 🚀 Quick Start (Developer Setup)

### 🔧 Requirements

* **Python** 3.10+ (recommended: 3.11)
* **Git** + **Git LFS** (for model weights)
* *(Optional)* **Node.js 18+** (if using React UI)
* *(Optional)* **CUDA 11+** (for GPU acceleration)

---

### 🏁 1. Clone & Setup Environment

```bash
(optional) cd downloads
git clone https://github.com/Western-Artificial-Intelligence/explainable-misinfo-ai.git
cd explainable-misinfo-ai

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Set up auto-formatting / linting hooks
pre-commit install
```

---

### ⚙️ 2. Create Environment Variables

In the **repo root**, create a `.env` file:

```bash
HF_HOME=.cache/huggingface
HF_TOKEN=<optional_hf_token_if_private>
MODEL_NAME=distilbert-base-uncased
NUM_LABELS=3
MAX_LEN=256
LR=2e-5
BATCH=16
EPOCHS=3
API_SECRET=<your_secret_for_backend_auth>
```

---

### 📊 3. Prepare Datasets

Mak sure your data under the `data/` folder:

```
data/
├── train.csv    # text,label
├── valid.csv
└── test.csv
```

---

### 🧠 4. Tech Overview

| Component          | Description                                          |
| ------------------ | ---------------------------------------------------- |
| **Model**          | DistilBERT / RoBERTa (Hugging Face Transformers)     |
| **Explainability** | LIME / SHAP → token-level importances                |
| **Backend**        | FastAPI + Uvicorn; CORS enabled; HMAC auth           |
| **Cache**          | In-memory (MVP), Redis planned for Phase 2           |
| **UI**             | Streamlit (MVP) → React + Chrome Extension (Phase 2) |
| **Hosting**        | Hugging Face Spaces / Render / Railway               |

---

### 📦 5. Required Files (Double Check)

**`requirements.txt`**

```
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

#if it doesn't work force download them e.g.
pip install fastapi uvicorn[standard] lime shap python-dotenv pydantic ruff black isort pytest PyYAML
```

**`.gitignore`**

```
.venv/
.env
__pycache__/
.cache/
runs/
models/
data/*
!data/.gitkeep
```

---

### 🧩 6. Project Structure (Currently)

```
api/
├── main.py             # Core FastAPI app
├── routes/
│   ├── classify.py     # /classify endpoint
│   └── health.py       # /healthz endpoint
├── utils/
│   └── cache.py        # In-memory caching logic
└── __init__.py
```

* **`main.py`** → Launches the API and connects routes
* **`routes/classify.py`** → Handles incoming classification requests
* **`utils/cache.py`** → Stores cached results to speed up responses

---

### 🔍 7. API Endpoints

#### **GET `/healthz`**

Checks if the backend is running.
Response: `"TruthLens backend is healthy!"`

#### **POST `/classify`**

Input:

```json
{
  "text": "Is AI dangerous?"
}
```

Response:

```json
{
  "label": "factual",
  "confidence": 0.9,
  "explanation": "Dummy placeholder explanation"
}
```

---

### 💾 8. How Caching Works

When a user sends text (e.g., `"Is AI dangerous?"`):

1. The system first checks if the response is already in cache.

   * ✅ **Found:** returns the cached answer instantly
   * ❌ **Not found:** runs the model, saves the result, and returns it

This reduces repeated computation and speeds up responses.

---

### ▶️ 9. Run the Backend

Start the FastAPI server:

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
```

---

### 🧪 10. Test the `/classify` Endpoint

In a new terminal, run:

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"text":"Is AI dangerous?"}'
```

Expected output:

```json
{
  "label": "factual",
  "confidence": 0.92,
  "explanation": "This is a placeholder explanation for 'factual' classification."
}
```

> You can also go to http://127.0.0.1:8000/docs to test the same thing!

---

### 🧰 11. Developer Tools

#### View Git Info

```bash
git status
git log --oneline -5
```

### How to Push
```bash
Steps Taken to Push:
git status
git branch
git checkout -b
git add .
git commit -m "Describe your changes" # git commit -m "Describe your changes" --no-verify cause it wasn't working the other way
git push origin
```
#### Set Python Interpreter in VS Code

* Open Command Palette → **Cmd + Shift + P**
* Search “Python: Select Interpreter”
* Choose the one inside `.venv`

#### Preview Markdown

**Ctrl + Shift + V**

---

### 🧩 12. Pull Request Template

**`PULL_REQUEST_TEMPLATE.md`**

```markdown
## What
-

## Why
-

## Test Plan
- [ ] Unit tests pass
- [ ] Manual check (endpoint / UI)

## Screenshots
(If applicable)

## Risks / Rollback
-
```
# Terminal 1: backend
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-dotenv pydantic
uvicorn api.main:app --reload
http://127.0.0.1:8000


# Terminal 2: frontend
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai/frontend
npm install
npm run dev

# Test
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"Is AI dangerous?"}'

Frontend notes:
1. needs a Navbar and a homepage just like an actual website
2. Change the prediction texts to True/False or Mixed.
3. Include a button that links to our chrome extension
4. UI changes for the button on the chrome extension
5. Change the favicon for the chrome extension

---

## Full Stack Run (Backend + Frontend + Extension + MongoDB)

### 1. Environment setup

Create `.env.local` in the repo root (or copy from `.env.example`) and set:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT_S=60

MONGO_URI=mongodb+srv://khimaniaryan_db_user:<db_password>@waicluster0.eeyuita.mongodb.net/?appName=WAICluster0
MONGODB_DB_NAME=truthlens
MONGODB_HISTORY_COLLECTION=history
MONGODB_CONNECT_TIMEOUT_MS=5000
```

If `MONGO_URI` is missing/unreachable, backend history automatically falls back to in-memory.

### 2. Run backend

```bash
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Backend URLs:
- `http://127.0.0.1:8000/healthz`
- `http://127.0.0.1:8000/docs`

### 3. Run frontend

```bash
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai/frontend
npm install
npm run dev
```

Frontend default:
- `http://localhost:3000`

The frontend now calls:
- `POST /predict` (RoBERTa mode)
- `POST /analyze` (LLM encoder mode)
- `GET/POST/DELETE /history` (history persistence)

### 4. Load Chrome extension

1. Open `chrome://extensions`
2. Enable `Developer mode`
3. Click `Load unpacked`
4. Select:
   - `/Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai/chrome-extension`

Default backend for extension is `http://localhost:8000`.

Backedn
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai
source .venv/bin/activate
pip install -r requirements.txt

# set real password in MONGO_URI
uvicorn api.main:app --reload

Frontend
cd /Users/aryankhimani/Downloads/WAI_Project/explainable-misinfo-ai/frontend
npm install
npm run dev


