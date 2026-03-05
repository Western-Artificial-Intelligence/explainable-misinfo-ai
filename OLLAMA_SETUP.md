# Ollama Setup for TruthLens

Ollama powers the **LLM claim extraction** step in the pipeline: it extracts structured factual claims from transcripts and OCR text before misinformation classification.

## 1. Install Ollama

Run in PowerShell (may require admin for system install):

```powershell
irm https://ollama.com/install.ps1 | iex
```

On Windows, Ollama is installed and typically runs as a background service. If it doesn't start automatically, open **Ollama** from the Start menu.

## 2. Pull the Model

The claim extractor uses `qwen3:4b` by default. Pull it:

```powershell
ollama pull qwen3:4b
```

This downloads ~2.5GB. Alternative smaller models (faster, less accurate):

- `ollama pull phi3:mini` (~2GB)
- `ollama pull llama3.2:3b` (~2GB)

## 3. Configure (Optional)

Create or edit `.env` in the project root.

**Local Ollama** (no API key needed):

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT_S=60
```

**Ollama Cloud** (using your API key from [ollama.com/settings/keys](https://ollama.com/settings/keys)):

```env
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=qwen3:4b
OLLAMA_API_KEY=your_api_key_here
OLLAMA_TIMEOUT_S=60
```

## 4. Verify

```powershell
# Check Ollama is running
curl http://127.0.0.1:11434/api/tags

# Test a chat
ollama run qwen3:4b "Extract the main claim from: The Earth is flat and NASA is hiding it."
```

## 5. Backend & Frontend setup

To run TruthLens end-to-end you also need to start the Python backend and the web frontend. The repository includes tasks and sample commands to make this straightforward.

1. **Create a Python virtual environment and install dependencies** (run from workspace root):

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\pip install --upgrade pip
    .\.venv\Scripts\pip install -r requirements.txt
    ```

   You may also execute the VS Code task **Python: create venv & install deps**.

2. **Configure environment variables**

   Edit `api/.env` and/or root `.env` (both are loaded) with the values shown earlier under the "Configure" section. Typical local settings:

    ```env
    # api/.env (or workspace root .env)
    OLLAMA_BASE_URL=http://127.0.0.1:11434
    OLLAMA_MODEL=qwen3:4b
    OLLAMA_TIMEOUT_S=60
    HOST=127.0.0.1
    PORT=8000
    ```

   And in the frontend folder create/modify `frontend/.env`:

    ```env
    VITE_API_BASE_URL=http://127.0.0.1:8000
    VITE_OLLAMA_KEY=                  # only required if you call Ollama cloud directly
    ```

3. **Start the backend**

    ```powershell
    cd api
    . .\.venv\Scripts\Activate.ps1     # activate the venv
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
    ```

   Or run the **Backend: run uvicorn** VS Code task (it keeps the server in a background terminal).

4. **Launch the frontend**

    ```powershell
    cd frontend
    npm install              # one-time
    npm run dev              # start dev server
    ```

   Alternatively use the **Frontend: npm install & dev** task.

5. **Load the browser extension**

   Use the **Extension: load unpacked reminder** task or manually open `chrome://extensions` → toggle developer mode → Load unpacked → point at `chrome-extension/`.

With Ollama, backend, and frontend running you can click the extension's button and select text; the request will go through the backend `analyze` endpoint and you should no longer see “Failed to fetch” errors.

## 6. Usage in TruthLens

When Ollama is running with a compatible model:

- **Image/Video analysis** (Chrome extension "Force Image Capture"): Raw text → Ollama extracts claim → RoBERTa/heuristic classifies
- **Claim extractor**: `api.services.claim_extractor` uses the system prompts in `_SYSTEM_PROMPT_OCR` and `_SYSTEM_PROMPT_TRANSCRIPT`

If Ollama is unavailable, the pipeline falls back to using the raw text directly for classification.
