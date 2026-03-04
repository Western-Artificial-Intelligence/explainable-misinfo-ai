const TLX_MAX_PAGE_ITEMS = 120;

const tlxState = {
  selectionRect: null,
  overlayEnabled: false,
  observer: null,
  observerTimer: null,
  elementById: new Map(),
  analyzedElementIds: new Set(),
  selectionCard: null
};

trackSelectionPosition();

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message?.type) {
    return;
  }

  if (message.type === "GET_SELECTED_TEXT") {
    sendResponse({
      text: getSelectedText()
    });
    return;
  }

  if (message.type === "SHOW_SELECTION_RESULT") {
    showSelectionResultCard(message.payload || {});
    return;
  }

  if (message.type === "ANALYZE_PAGE") {
    analyzeVisiblePage()
      .then((summary) => sendResponse({ ok: true, summary }))
      .catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }

  if (message.type === "CLEAR_OVERLAYS") {
    clearAllOverlays();
    sendResponse({ ok: true });
    return;
  }

  if (message.type === "GET_OVERLAY_STATUS") {
    sendResponse({ enabled: tlxState.overlayEnabled });
  }
});

function trackSelectionPosition() {
  const capture = () => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
      return;
    }
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    if (rect && rect.width > 0 && rect.height > 0) {
      tlxState.selectionRect = {
        left: rect.left + window.scrollX,
        top: rect.top + window.scrollY,
        width: rect.width,
        height: rect.height
      };
    }
  };

  document.addEventListener("mouseup", capture, true);
  document.addEventListener("keyup", capture, true);
  document.addEventListener("selectionchange", capture, true);
}

function getSelectedText() {
  const selection = window.getSelection();
  return (selection ? selection.toString() : "").trim();
}

function showSelectionResultCard(payload) {
  if (tlxState.selectionCard) {
    tlxState.selectionCard.remove();
    tlxState.selectionCard = null;
  }

  const prediction = String(payload.prediction || "REAL").toUpperCase();
  const confidence = Number(payload.confidence || 0);
  const explanation = String(payload.explanation || "");
  const confidencePct = Math.round(confidence * 100);

  const card = document.createElement("div");
  card.className = `tlx-floating-card ${prediction === "FAKE" ? "tlx-fake" : "tlx-real"}`;
  card.innerHTML = `
    <div class="tlx-card-header">
      <span class="tlx-card-badge">${prediction}</span>
      <button class="tlx-close-button" aria-label="Close">×</button>
    </div>
    <div class="tlx-card-confidence">Confidence: ${confidencePct}%</div>
    <div class="tlx-card-explanation">${escapeHtml(explanation)}</div>
  `;

  const closeButton = card.querySelector(".tlx-close-button");
  closeButton?.addEventListener("click", () => {
    card.remove();
    if (tlxState.selectionCard === card) {
      tlxState.selectionCard = null;
    }
  });

  const fallbackLeft = window.scrollX + window.innerWidth - 380;
  const fallbackTop = window.scrollY + 20;
  const anchor = tlxState.selectionRect;
  const left = anchor ? anchor.left : fallbackLeft;
  const top = anchor ? anchor.top + anchor.height + 10 : fallbackTop;

  card.style.left = `${Math.max(window.scrollX + 12, left)}px`;
  card.style.top = `${Math.max(window.scrollY + 12, top)}px`;

  document.documentElement.appendChild(card);
  tlxState.selectionCard = card;
}

async function analyzeVisiblePage() {
  tlxState.overlayEnabled = true;
  const candidates = collectAnalyzableElements();
  if (candidates.length === 0) {
    startMutationObserver();
    return { analyzed: 0 };
  }

  const payloadItems = [];
  for (const candidate of candidates) {
    tlxState.elementById.set(candidate.id, candidate.element);
    payloadItems.push({ id: candidate.id, text: candidate.text });
  }

  const response = await chrome.runtime.sendMessage({
    type: "ANALYZE_BATCH",
    items: payloadItems
  });

  if (!response?.ok) {
    throw new Error(response?.error || "Could not analyze page.");
  }

  applyAnalysisResults(response.results || []);
  startMutationObserver();
  return { analyzed: candidates.length };
}

function collectAnalyzableElements() {
  const selectors = [
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "article p",
    "[data-testid='tweetText']",
    "article [lang]"
  ];

  const all = Array.from(document.querySelectorAll(selectors.join(",")));
  const unique = new Set();
  const items = [];
  const now = Date.now();

  for (let i = 0; i < all.length; i += 1) {
    if (items.length >= TLX_MAX_PAGE_ITEMS) {
      break;
    }

    const element = all[i];
    if (!(element instanceof HTMLElement)) {
      continue;
    }
    if (element.closest(".tlx-floating-card")) {
      continue;
    }
    if (!isElementVisible(element)) {
      continue;
    }

    const text = normalizeText(element.innerText || element.textContent || "");
    if (text.length < 35) {
      continue;
    }

    const fingerprint = `${element.tagName}:${text.slice(0, 120)}`;
    if (unique.has(fingerprint)) {
      continue;
    }
    unique.add(fingerprint);

    const existingId = element.dataset.tlxElementId;
    if (existingId && tlxState.analyzedElementIds.has(existingId)) {
      continue;
    }

    const id = existingId || `tlx-${now}-${i}-${Math.random().toString(36).slice(2, 8)}`;
    element.dataset.tlxElementId = id;

    items.push({
      id,
      element,
      text
    });
  }

  return items;
}

function applyAnalysisResults(results) {
  for (const entry of results) {
    if (!entry?.id) {
      continue;
    }
    const element = tlxState.elementById.get(entry.id);
    tlxState.elementById.delete(entry.id);
    if (!element || !(element instanceof HTMLElement)) {
      continue;
    }
    if (!entry.ok || !entry.result) {
      continue;
    }
    paintOverlay(element, entry.id, entry.result);
  }
}

function paintOverlay(element, id, result) {
  const prediction = String(result.prediction || "REAL").toUpperCase();
  const isFake = prediction === "FAKE";
  const confidence = Number(result.confidence || 0);
  const explanation = String(result.explanation || "");
  const confidencePct = Math.round(confidence * 100);

  tlxState.analyzedElementIds.add(id);
  element.classList.add("tlx-analyzed");
  element.classList.toggle("tlx-analyzed-real", !isFake);
  element.classList.toggle("tlx-analyzed-fake", isFake);

  if (!element.dataset.tlxOriginalPosition) {
    element.dataset.tlxOriginalPosition = element.style.position || "";
  }
  if (getComputedStyle(element).position === "static") {
    element.style.position = "relative";
  }

  const previousBadge = element.querySelector(":scope > .tlx-inline-badge");
  if (previousBadge) {
    previousBadge.remove();
  }

  const badge = document.createElement("button");
  badge.type = "button";
  badge.className = `tlx-inline-badge ${isFake ? "tlx-badge-fake" : "tlx-badge-real"}`;
  badge.textContent = `${prediction} ${confidencePct}%`;
  badge.title = explanation;
  badge.setAttribute("aria-label", `${prediction}. Confidence ${confidencePct} percent.`);

  const tooltip = document.createElement("div");
  tooltip.className = "tlx-tooltip";
  tooltip.textContent = `Confidence: ${confidencePct}%\n${explanation}`;
  badge.appendChild(tooltip);

  badge.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });

  element.appendChild(badge);
}

function clearAllOverlays() {
  tlxState.overlayEnabled = false;
  stopMutationObserver();

  const nodes = document.querySelectorAll(".tlx-analyzed");
  nodes.forEach((node) => {
    if (!(node instanceof HTMLElement)) {
      return;
    }
    node.classList.remove("tlx-analyzed", "tlx-analyzed-real", "tlx-analyzed-fake");

    const badge = node.querySelector(":scope > .tlx-inline-badge");
    if (badge) {
      badge.remove();
    }

    const originalPosition = node.dataset.tlxOriginalPosition;
    if (originalPosition !== undefined) {
      node.style.position = originalPosition;
      delete node.dataset.tlxOriginalPosition;
    }

    delete node.dataset.tlxElementId;
  });

  tlxState.elementById.clear();
  tlxState.analyzedElementIds.clear();

  if (tlxState.selectionCard) {
    tlxState.selectionCard.remove();
    tlxState.selectionCard = null;
  }
}

function startMutationObserver() {
  if (tlxState.observer) {
    return;
  }
  tlxState.observer = new MutationObserver(() => {
    if (!tlxState.overlayEnabled) {
      return;
    }
    if (tlxState.observerTimer) {
      clearTimeout(tlxState.observerTimer);
    }
    tlxState.observerTimer = setTimeout(() => {
      analyzeVisiblePage().catch(() => {
        /* ignore transient dynamic-page failures */
      });
    }, 700);
  });
  tlxState.observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });
}

function stopMutationObserver() {
  if (tlxState.observer) {
    tlxState.observer.disconnect();
    tlxState.observer = null;
  }
  if (tlxState.observerTimer) {
    clearTimeout(tlxState.observerTimer);
    tlxState.observerTimer = null;
  }
}

function isElementVisible(element) {
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return false;
  }
  const style = getComputedStyle(element);
  if (style.visibility === "hidden" || style.display === "none") {
    return false;
  }
  return true;
}

function normalizeText(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .trim();
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

