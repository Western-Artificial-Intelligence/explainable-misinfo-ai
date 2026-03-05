# TruthLens Extension – Why Nothing Works (Checklist)

If **nothing** works in the Chrome extension, it’s almost always one of these.

---

## 1. Backend not running (most common)

**Symptom:** "Backend not reachable" in the popup, or "Backend 500" / connection errors when you click any button.

**Fix:** Start the API on your machine:

```bash
cd c:\Users\Subhr\explainable-misinfo-ai-1
.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

Leave this terminal open. The popup should show **"Backend connected"** when you open it.

---

## 2. Analyze Selected Text / right‑click "Analyze with TruthLens" fails (503 or 500)

**Symptom:** Error like "Ollama unavailable" or "Backend 500" when analyzing selected text.

**Cause:** This feature uses **Ollama** (LLM) on the backend. If Ollama isn’t running or the model isn’t loaded, the `/analyze` call fails.

**Fix:**

1. Install Ollama from https://ollama.com if needed.
2. In a terminal where `ollama` works (or after fixing PATH):
   ```powershell
   ollama run llama3.2
   ```
3. In the project, ensure `.env.local` uses **local** Ollama:
   - `OLLAMA_BASE_URL=http://127.0.0.1:11434`
   - `OLLAMA_MODEL=llama3.2`
4. Restart the API (`uvicorn api.main:app --reload`) after changing `.env.local`.

---

## 3. Wrong tab / restricted page

**Symptom:** "Could not access active tab" or "TruthLens only runs on regular http/https pages."

**Cause:** The extension only works on normal web pages (`http://` or `https://`). It cannot run on:

- `chrome://` (e.g. Extensions page)
- New Tab page
- File URLs

**Fix:** Open a normal site (e.g. https://twitter.com, a news article), then open the TruthLens popup or use the context menu.

---

## 4. Video / Image capture: "No video element found"

**Symptom:** Image capture or Auto‑Detect says "No video element found on page".

**Cause:** There is no **playing** `<video>` on the page. Image capture is meant for pages where a video is actually playing (e.g. TikTok, YouTube).

**Fix:** Use Image capture only on a page with a **visible, playing** video. On a feed with no video (e.g. Twitter feed with no playing clip), this is expected to fail.

---

## 5. Audio capture / transcription fails

**Symptom:** "Audio analysis failed" in console, or transcription returns an error.

**Possible causes:**

- **Backend:** If the real audio pipeline isn’t installed, the server uses a fallback that returns a message like "Audio transcription requires OpenAI Whisper".
- **Browser / page:** Some sites block microphone/audio capture or use CORS in a way that breaks capture.

**Fix:** Ensure the backend is running and, if you need real transcription, install the optional audio dependencies (e.g. Whisper) as in the main project README.

---

## 6. Analyze Entire Page / Analyze Document

**Symptom:** "Failed to analyze page" or "Document analysis failed".

**Cause:** These use the **/predict** endpoint (no Ollama). So they only need the **backend** to be running and reachable.

**Fix:** Ensure the backend is running (see §1). If the popup shows "Backend connected", try again; if it still fails, check the browser console and the API terminal for the real error.

---

## Quick checklist

| Check | What to do |
|-------|------------|
| Popup shows "Backend connected" | If not, start `uvicorn api.main:app --reload` and fix URL in extension if you changed it. |
| Analyze Selected Text works | Backend + Ollama running, `.env.local` points to local Ollama and correct model. |
| Image capture works | Use on a page with a **playing** video. |
| Right tab | Use a normal https/http page, not chrome:// or new tab. |

After changing backend URL, restarting the API, or fixing Ollama, **reload the extension** (chrome://extensions → TruthLens → Reload) and try again.
