# query/query_image.py
"""
Two-stage stone family search.

Stage 1  — colour pipeline  (ChromaDB + LAB histogram + KMeans + Gabor veins)
            Returns top-N candidate families based on colour similarity.

Stage 2  — DINOv2 reranker  (FAISS + fine-tuned ViT-small)
            Takes Stage-1 candidates and reranks them using deep visual features.

Public API
----------
    from query.query_image import query_image

    # Returns [(family_name, dino_score), ...] sorted best-first
    results = query_image(image_path, colour_top_n=70, final_top_k=10)
"""

from collections import Counter
from features.stone_vector import extract_stone_vector
from db.chroma_client import get_chroma_collection
from query.reranker import rerank


# ── Stage 1: Colour retrieval ─────────────────────────────────────────────────
def _colour_query(image_path: str, top_n: int = 100) -> list:
    """
    Query ChromaDB with the colour vector of the image.
    Returns [(family_name, normalised_hit_count), ...] length = top_n.
    """
    collection = get_chroma_collection()
    query_vec  = extract_stone_vector(image_path)

    if query_vec is None:
        return []

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_n * 5, 2000),   # fetch more raw hits, then aggregate
    )

    families = [meta["family"] for meta in results["metadatas"][0]]
    counter  = Counter(families)

    # Normalise hit count to 0–1 score
    max_count = max(counter.values()) if counter else 1
    return [
        (fam, round(count / max_count, 4))
        for fam, count in counter.most_common(top_n)
    ]


# ── Public entry point ────────────────────────────────────────────────────────
def query_image(
    image_path: str,
    colour_top_n: int = 80,    # how many candidates Stage-1 shortlists
    final_top_k: int  = 10,    # how many results Stage-2 returns
) -> list:
    """
    Run the full two-stage pipeline.

    Returns
    -------
    [(family_name, dino_score), ...]  sorted best-first, length = final_top_k
    """
    # Stage 1 — colour candidates
    candidates = _colour_query(image_path, top_n=colour_top_n)
    if not candidates:
        return []

    # Stage 2 — DINOv2 rerank
    return rerank(image_path, candidates, top_k=final_top_k)