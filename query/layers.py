# query/layers.py
"""
Three independent query layers. Each returns:
  families → [(family_name, score), ...]   sorted best-first
  images   → [(path, score), ...]          top matching DB images

Scores are normalised to [0, 1].
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch

from config.settings import CKPT_DIR
from db.chroma_client import get_color_collection, get_embedding_collection
from features.stone_vector import extract_stone_vector


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — COLOUR
# ═══════════════════════════════════════════════════════════════════════════════

def color_layer(
    image_path: str,
    top_k_families: int = 40,
    top_k_images: int = 20,
    fetch_multiplier: int = 5,
) -> tuple[list, list]:
    """
    Query the stone_colors ChromaDB collection.

    Returns
    -------
    families : [(family_name, normalised_hit_count), ...]
    images   : [(path, similarity_score), ...]
    """
    collection = get_color_collection()
    query_vec  = extract_stone_vector(image_path)
    if query_vec is None:
        return [], []

    n_fetch = min(top_k_families * fetch_multiplier, collection.count() or 1)
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=n_fetch,
        include=["metadatas", "distances"],
    )

    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # ── family aggregation ────────────────────────────────────────────────────
    families_counter: Counter = Counter()
    for meta in metas:
        families_counter[meta["family"]] += 1
    max_count = max(families_counter.values()) if families_counter else 1
    families = [
        (fam, round(cnt / max_count, 4))
        for fam, cnt in families_counter.most_common(top_k_families)
    ]

    # ── top images (lowest L2 distance) ──────────────────────────────────────
    max_dist = max(distances) if distances else 1.0
    img_scores = [
        (meta["path"], round(1.0 - dist / (max_dist + 1e-9), 4))
        for meta, dist in zip(metas, distances)
    ]
    img_scores.sort(key=lambda x: x[1], reverse=True)
    images = img_scores[:top_k_images]

    return families, images


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — EMBEDDING (DINOv2 via ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════════

def embedding_layer(
    image_path: str,
    top_k_families: int = 40,
    top_k_images: int = 20,
    fetch_multiplier: int = 5,
) -> tuple[list, list]:
    """
    Query the stone_embeddings ChromaDB collection (cosine similarity).

    Returns
    -------
    families : [(family_name, normalised_hit_count), ...]
    images   : [(path, similarity_score), ...]
    """
    from features.dino_embedder import embed_image

    collection = get_embedding_collection()
    vec        = embed_image(image_path)
    if vec is None:
        return [], []

    n_fetch = min(top_k_families * fetch_multiplier, collection.count() or 1)
    results = collection.query(
        query_embeddings=[vec.tolist()],
        n_results=n_fetch,
        include=["metadatas", "distances"],
    )

    metas     = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distance ∈ [0, 2]

    # ── family aggregation ────────────────────────────────────────────────────
    families_counter: Counter = Counter()
    for meta in metas:
        families_counter[meta["family"]] += 1
    max_count = max(families_counter.values()) if families_counter else 1
    families = [
        (fam, round(cnt / max_count, 4))
        for fam, cnt in families_counter.most_common(top_k_families)
    ]

    # ── top images (lowest cosine distance → highest similarity) ─────────────
    img_scores = [
        (meta["path"], round(1.0 - dist / 2.0, 4))   # cosine dist ∈ [0,2]
        for meta, dist in zip(metas, distances)
    ]
    img_scores.sort(key=lambda x: x[1], reverse=True)
    images = img_scores[:top_k_images]

    return families, images


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — MODEL (DINOv2 + FAISS reranker)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_reranker_artifacts():
    """Load FAISS index + meta once."""
    meta_path = Path(CKPT_DIR) / "stone_index_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"[Model] stone_index_meta.pkl not found in '{CKPT_DIR}'")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    faiss_path = Path(CKPT_DIR) / "stone_index.faiss"
    if not faiss_path.exists():
        raise FileNotFoundError(f"[Model] stone_index.faiss not found in '{CKPT_DIR}'")
    index = faiss.read_index(str(faiss_path))

    from query.name_utils import build_alias_map
    alias_map = build_alias_map(meta["family_names"])

    return index, meta, alias_map


_reranker_cache = None


def _get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        _reranker_cache = _load_reranker_artifacts()
    return _reranker_cache


def model_layer(
    image_path: str,
    top_k_families: int = 10,
    top_k_images: int = 20,
    per_fam_vectors: int = 20,
    search_k: int = 800,
) -> tuple[list, list]:
    """
    Score families using DINOv2 embedding + FAISS nearest-neighbour search.

    Returns
    -------
    families : [(family_name, dino_score), ...]
    images   : [(path, similarity_score), ...]   ← paths from FAISS metadata
    """
    from features.dino_embedder import embed_image

    vec = embed_image(image_path)
    if vec is None:
        return [], []

    index, meta, alias_map = _get_reranker()
    family_names = meta["family_names"]
    labels_arr   = np.array(meta["labels"])
    paths_arr    = np.array(meta.get("paths", [""]*len(labels_arr)))

    vec_2d = vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec_2d)

    k = min(search_k, index.ntotal)
    sims, idxs = index.search(vec_2d, k)
    sims, idxs = sims[0], idxs[0]

    # ── family scoring (mean of top-3 nearest per family) ────────────────────
    family_sims: dict[int, list] = defaultdict(list)
    for sim, di in zip(sims, idxs):
        lbl = int(labels_arr[di])
        if len(family_sims[lbl]) < per_fam_vectors:
            family_sims[lbl].append(float(sim))

    ranked_families = []
    for lbl, sim_list in family_sims.items():
        top3_mean = float(np.mean(sorted(sim_list, reverse=True)[:3]))
        ranked_families.append((family_names[lbl], top3_mean))

    ranked_families.sort(key=lambda x: x[1], reverse=True)
    families = ranked_families[:top_k_families]

    # ── top images from FAISS hits ────────────────────────────────────────────
    img_scores = []
    for sim, di in zip(sims[:top_k_images * 3], idxs[:top_k_images * 3]):
        if di < len(paths_arr):
            p = paths_arr[di]
            if p:
                img_scores.append((p, round(float(sim), 4)))

    img_scores.sort(key=lambda x: x[1], reverse=True)
    images = img_scores[:top_k_images]

    return families, images