from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

CANON = ["false", "mixed", "true", "nei"]
CANON2ID = {"false": 0, "mixed": 1, "true": 2, "nei": 3}


def clean_text(s: object) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(frozen=True)
class VerifierOut:
    label_id: int
    label: str
    conf: float
    score: float
    probs: Dict[str, float]


class VerifierPipeline:
    def __init__(self, model_path: str, *, device: int = -1):
        try:
            from transformers import pipeline
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency: transformers. Install with `pip install transformers` "
                "(and ideally `pip install torch`)."
            ) from e

        self.pipe = pipeline(
            task="text-classification",
            model=model_path,
            tokenizer=model_path,
            return_all_scores=True,
            truncation=True,
            device=device,
        )

    def _to_canon_label(self, raw_label: str) -> str:
        """
        Map model label -> canonical label.
        Adjust here if your model outputs LABEL_0, etc.
        """
        lab = (raw_label or "").strip().casefold()

        # common patterns
        if lab in CANON:
            return lab
        if lab in {"label_0", "0"}:
            return "false"
        if lab in {"label_1", "1"}:
            return "mixed"
        if lab in {"label_2", "2"}:
            return "true"
        if lab in {"label_3", "3"}:
            return "nei"

        # fallback: if unknown, treat as NEI
        return "nei"

    def predict(self, claim: str) -> VerifierOut:
        claim = clean_text(claim)

        # HF pipeline returns: [[{"label": "...", "score": ...}, ...]]
        out = self.pipe(claim)
        rows = out[0] if out and isinstance(out, list) else []

        probs = {k: 0.0 for k in CANON}
        for r in rows:
            canon = self._to_canon_label(str(r.get("label", "")))
            probs[canon] = float(r.get("score", 0.0))

        # normalize defensively (some models may not sum to 1)
        s = sum(probs.values())
        if s > 0:
            probs = {k: v / s for k, v in probs.items()}

        label = max(probs, key=probs.get)
        label_id = CANON2ID[label]
        conf = float(probs[label])

        # score is a convenience margin; keep it stable
        score = float(probs["true"] - probs["false"])

        return VerifierOut(
            label_id=label_id,
            label=label,
            conf=conf,
            score=score,
            probs=probs,
        )
