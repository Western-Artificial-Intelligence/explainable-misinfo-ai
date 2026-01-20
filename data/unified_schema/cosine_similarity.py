from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Missing dependency: sentence-transformers. Install with:\n"
        "  pip install sentence-transformers"
    ) from e

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=4)
def _get_model(model_name: str = _DEFAULT_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def encode_texts(
    texts: Sequence[str],
    model_name: str = _DEFAULT_MODEL_NAME,
    *,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts into L2-normalized embeddings.
    Returns: np.ndarray of shape (N, D), dtype float32/float64 depending on backend.
    """
    model = _get_model(model_name)
    if device:
        model = model.to(device)

    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

def similarity_score(
    a: str,
    b: str,
    model_name: str = _DEFAULT_MODEL_NAME,
    *,
    device: Optional[str] = None,
) -> float:
    """
    Semantic cosine similarity between two strings.
    Returns 0.0 if either input is empty/whitespace.
    """
    if not a or not a.strip() or not b or not b.strip():
        return 0.0

    v = encode_texts([a, b], model_name=model_name, device=device, batch_size=2)
    return float(v[0] @ v[1])

def similarity_many(
    query: str,
    candidates: Sequence[str],
    model_name: str = _DEFAULT_MODEL_NAME,
    *,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute cosine similarities between one query and many candidates.
    Returns: scores shape (N,).
    """
    if not query or not query.strip():
        return np.zeros((len(candidates),), dtype=np.float32)

    # Note: keep empty candidates aligned as 0.0
    cand_list = list(candidates)
    mask = np.array([bool(c and str(c).strip()) for c in cand_list], dtype=bool)
    scores = np.zeros((len(cand_list),), dtype=np.float32)
    if not mask.any():
        return scores

    qv = encode_texts([query], model_name=model_name, device=device, batch_size=1)[0]
    cv = encode_texts(
        [cand_list[i] for i in np.where(mask)[0]],
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    scores[mask] = (cv @ qv).astype(np.float32)
    return scores

def similarity_matrix(
    A: Sequence[str],
    B: Sequence[str],
    model_name: str = _DEFAULT_MODEL_NAME,
    *,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> np.ndarray:
    """
    All-pairs cosine similarity between two lists.
    Returns: matrix shape (len(A), len(B)).
    """
    if not A or not B:
        return np.zeros((len(A), len(B)), dtype=np.float32)

    Av = encode_texts(A, model_name=model_name, device=device, batch_size=batch_size)
    Bv = encode_texts(B, model_name=model_name, device=device, batch_size=batch_size)
    return (Av @ Bv.T).astype(np.float32)
