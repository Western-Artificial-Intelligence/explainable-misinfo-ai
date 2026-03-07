# TruthLens Backend Deployment Guide

Host your FastAPI backend so the frontend at Netlify can call `https://api.truthlens.app`.

---

## Prerequisites

- **Brave Search API key** – [Get one](https://brave.com/search/api/)
- **Ollama** – For LLM verdicts. Options:
  - Run Ollama on a separate GPU VM (e.g. RunPod, Vast.ai) and set `OLLAMA_BASE_URL` to that URL
  - Or use a serverless LLM API (requires code changes)
- **MongoDB Atlas** (optional) – For persistent history; leave unset for in-memory

---

## Option 1: Railway (recommended)

1. **Sign up** at [railway.app](https://railway.app).

2. **Create project** → **Deploy from GitHub** → choose this repo.

3. **Add service**:
   - Source: GitHub repo
   - Root directory: `/` (repo root)
   - Build: Dockerfile (auto-detected)
   - No need to set build command

4. **Set environment variables** (Settings → Variables):

   | Variable | Value |
   |----------|--------|
   | `BRAVE_API_KEY` | Your Brave API key |
   | `OLLAMA_BASE_URL` | URL of your Ollama server (e.g. `https://ollama.yourrunpod.net`) |
   | `OLLAMA_MODEL` | `llama3.2:latest` |
   | `OLLAMA_TIMEOUT_S` | `120` |
   | `MONGO_URI` | (optional) MongoDB Atlas connection string |
   | `HF_HOME` | `/app/.cache/huggingface` |

5. **Deploy** – Railway builds the image and assigns a URL like `https://your-app.up.railway.app`.

6. **Custom domain** – Settings → Domains → Add custom domain `api.truthlens.app`. Add the CNAME record Railway shows in your DNS.

---

## Option 2: Render

1. **Sign up** at [render.com](https://render.com).

2. **New** → **Web Service** → connect this GitHub repo.

3. **Settings**:
   - Build command: *(leave empty – Dockerfile handles it)*
   - Start command: *(leave empty – CMD in Dockerfile)*
   - Ensure "Docker" is selected as environment.

4. **Environment variables** (same as Railway above).

5. **Deploy** – Render gives a URL like `https://truthlens-api.onrender.com`.

6. **Custom domain** – Settings → Custom Domains → Add `api.truthlens.app`.

---

## Option 3: Fly.io

1. Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/).

2. From repo root:
   ```bash
   fly launch --no-deploy
   ```
   - App name: e.g. `truthlens-api`
   - Region: choose nearest users
   - Decline PostgreSQL if prompted

3. Set secrets:
   ```bash
   fly secrets set BRAVE_API_KEY=your_key
   fly secrets set OLLAMA_BASE_URL=https://your-ollama-url
   fly secrets set OLLAMA_MODEL=llama3.2:latest
   ```

4. Deploy:
   ```bash
   fly deploy
   ```

5. Custom domain: `fly certs add api.truthlens.app`.

---

## Custom domain (api.truthlens.app)

1. In your domain DNS (e.g. truthlens.app), add:
   - **Type**: CNAME  
   - **Name**: `api`  
   - **Value**: the host from your host (e.g. `your-app.up.railway.app` or `your-app.fly.dev`)

2. Wait for propagation (often 5–30 minutes). Check with:
   ```bash
   nslookup api.truthlens.app
   ```

3. Open https://api.truthlens.app/docs – you should see the FastAPI Swagger docs.

---

## Verify deployment

1. `https://api.truthlens.app/docs` – Swagger UI loads
2. `https://api.truthlens.app/healthz` – Returns healthy
3. Frontend on Netlify calls `https://api.truthlens.app/api/...` – analyze requests work

---

## Notes

- **Memory**: PyTorch + transformers need ~2–4 GB RAM. Railway/Render paid tiers or Fly.io usually work; free tiers may fail.
- **Ollama**: Must be reachable from the host. Use a GPU VM (RunPod, etc.) or another service; local Ollama is not reachable from Railway/Render.
- **Secrets**: Never commit `.env` or `.env.local`. Use each platform’s environment variable UI.
