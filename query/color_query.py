# query/colour_query.py
"""
Stage-1: colour-based retrieval using the existing ChromaDB vector store.

Returns the top-N stone family candidates with their colour scores.
These are then passed to the DINOv2 reranker for Stage-2.
"""

import numpy as np
import cv2
from collections import Counter
from pathlib import Path


# ── Colour feature extraction (unchanged from your original pipeline) ─────────

import numpy as np
from sklearn.cluster import KMeans

N_CLUSTERS = 8
HIST_BINS  = 64


def _extract_lab_histogram(lab: np.ndarray) -> np.ndarray:
    hists = []
    for ch in range(3):
        h, _ = np.histogram(lab[:, :, ch].ravel(), bins=HIST_BINS, density=True)
        hists.append(h)
    return np.concatenate(hists).astype(np.float32)


def _extract_kmeans_clusters(lab: np.ndarray, k: int = N_CLUSTERS) -> np.ndarray:
    pixels = lab.reshape(-1, 3).astype(np.float32)
    idx    = np.random.choice(len(pixels), min(5000, len(pixels)), replace=False)
    km     = KMeans(n_clusters=k, random_state=42, n_init=5).fit(pixels[idx])
    labels = km.predict(pixels)

    vec = []
    for i in range(k):
        mask   = labels == i
        weight = mask.mean()
        spread = pixels[mask].std() if mask.sum() > 1 else 0.0
        vec.extend([*km.cluster_centers_[i].tolist(), weight, spread])

    order = np.argsort([vec[i * 5 + 3] for i in range(k)])[::-1]
    out   = np.array([vec[i * 5:(i + 1) * 5] for i in order]).ravel()
    return out.astype(np.float32)


def _gabor_vein_mask(gray: np.ndarray) -> np.ndarray:
    responses = []
    for theta in np.linspace(0, np.pi, 8):
        kern = cv2.getGaborKernel((21, 21), 4.0, theta, 8.0, 0.5)
        responses.append(cv2.filter2D(gray, cv2.CV_32F, kern))
    energy = np.max(np.abs(responses), axis=0)
    _, mask = cv2.threshold(energy, energy.mean() + energy.std(), 1, cv2.THRESH_BINARY)
    return mask.astype(bool)


def _extract_vein_base_colors(lab: np.ndarray) -> np.ndarray:
    gray   = lab[:, :, 0]
    vmask  = _gabor_vein_mask(gray)
    vein_mean = lab[vmask].mean(axis=0)  if vmask.any()   else np.zeros(3)
    base_mean = lab[~vmask].mean(axis=0) if (~vmask).any() else np.zeros(3)
    vein_std  = lab[vmask].std(axis=0)   if vmask.any()   else np.zeros(3)
    base_std  = lab[~vmask].std(axis=0)  if (~vmask).any() else np.zeros(3)
    return np.concatenate([vein_mean, vein_std, base_mean, base_std]).astype(np.float32)


def extract_stone_vector(image_path: str) -> np.ndarray | None:
    """248-dim colour feature vector for a stone image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img  = cv2.resize(img, (512, 512))
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hist = _extract_lab_histogram(lab)
    km   = _extract_kmeans_clusters(lab)
    vein = _extract_vein_base_colors(lab)
    return np.concatenate([hist, km, vein])   # 248-dim


# ── ChromaDB query ─────────────────────────────────────────────────────────────

def colour_query(
    image_path: str,
    top_k: int = 40,
) -> list[tuple[str, float]]:
    """
    Query ChromaDB with the colour vector of `image_path`.

    Returns
    -------
    [(family_name, colour_score), ...]  sorted best-first, length = top_k.
    colour_score is the normalised hit count (0–1).
    """
    # Import here so the module can be imported even if chroma isn't set up yet
    from db.chroma_client import get_chroma_collection

    collection = get_chroma_collection()
    query_vec  = extract_stone_vector(image_path)

    if query_vec is None:
        return []

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k * 5, 200),   # fetch more, then aggregate
    )

    families = [meta["family"] for meta in results["metadatas"][0]]
    counter  = Counter(families)

    # Normalise counts → 0–1 score
    max_count = max(counter.values()) if counter else 1
    ranked = [
        (fam, round(count / max_count, 4))
        for fam, count in counter.most_common(top_k)
    ]
    return ranked