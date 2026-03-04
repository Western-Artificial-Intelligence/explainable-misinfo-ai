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

## 5. Usage in TruthLens

When Ollama is running with a compatible model:

- **Image/Video analysis** (Chrome extension "Force Image Capture"): Raw text → Ollama extracts claim → RoBERTa/heuristic classifies
- **Claim extractor**: `api.services.claim_extractor` uses the system prompts in `_SYSTEM_PROMPT_OCR` and `_SYSTEM_PROMPT_TRANSCRIPT`

If Ollama is unavailable, the pipeline falls back to using the raw text directly for classification.
