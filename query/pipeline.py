# query/pipeline.py
"""
Flexible multi-layer search pipeline.

Supports any combination and ordering of three layers:
  "color"     — LAB colour histogram + KMeans + Gabor vein features (ChromaDB)
  "embedding" — DINOv2 embeddings (ChromaDB cosine similarity)
  "model"     — DINOv2 + FAISS reranker (fine-tuned classifier backbone)

A "layer" does two things:
  1. Retrieve candidate families (with scores).
  2. Return the matched DB images for that layer.

When multiple layers are chained:
  - The first layer retrieves broadly (top_k_families candidates).
  - Each subsequent layer RERANKS the candidates from the previous layer,
    filtering to only those families.
  - Family scores are the score from the LAST layer in the chain.

The pipeline always returns:
  {
    "families"      : [(name, score), ...],      # final merged family ranking
    "images"        : {                           # per-layer image results
        "color"     : [(path, score), ...],
        "embedding" : [(path, score), ...],
        "model"     : [(path, score), ...],
    },
    "layer_families": {                           # per-layer family results
        "color"     : [(name, score), ...],
        "embedding" : [(name, score), ...],
        "model"     : [(name, score), ...],
    },
  }
"""

from __future__ import annotations

from query.layers import color_layer, embedding_layer, model_layer

LAYER_FN = {
    "color":     color_layer,
    "embedding": embedding_layer,
    "model":     model_layer,
}


def _merge_families(
    prev_families: list[tuple[str, float]],
    curr_families: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """
    Intersect curr_families with prev_families candidates.
    Score = curr_score (DINOv2 / colour / embedding of the current layer).
    If prev_families is empty, return curr_families as-is.
    """
    if not prev_families:
        return curr_families

    prev_names = {name for name, _ in prev_families}
    merged = [(name, score) for name, score in curr_families if name in prev_names]

    # Append prev-only families at the bottom with score 0.0
    curr_names = {name for name, _ in merged}
    for name, _ in prev_families:
        if name not in curr_names:
            merged.append((name, 0.0))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


def run_pipeline(
    image_path: str,
    layer_order: list[str],         # e.g. ["color", "model", "embedding"]
    top_k_families: int = 10,
    top_k_images: int = 10,
    first_layer_fetch: int = 40,    # broad fetch for the first layer
) -> dict:
    """
    Run the pipeline with the given layer order.

    Parameters
    ----------
    image_path      : path to the query image
    layer_order     : ordered list of layer names to run
    top_k_families  : number of final families to return
    top_k_images    : number of images to return per layer
    first_layer_fetch : how many candidates the first layer fetches

    Returns
    -------
    dict with keys: families, images, layer_families
    """
    if not layer_order:
        raise ValueError("layer_order must have at least one layer.")

    valid = set(LAYER_FN.keys())
    for name in layer_order:
        if name not in valid:
            raise ValueError(f"Unknown layer '{name}'. Choose from {valid}.")

    layer_families: dict[str, list] = {}
    layer_images:   dict[str, list] = {}

    prev_families: list[tuple[str, float]] = []

    for i, layer_name in enumerate(layer_order):
        fn = LAYER_FN[layer_name]

        # First layer fetches broadly; subsequent layers restrict to candidates
        k_fam = first_layer_fetch if i == 0 else max(top_k_families * 2, first_layer_fetch)

        fams, imgs = fn(
            image_path,
            top_k_families=k_fam,
            top_k_images=top_k_images,
        )

        layer_images[layer_name]   = imgs
        layer_families[layer_name] = fams

        # Chain: rerank using current-layer scores, restrict to prev candidates
        prev_families = _merge_families(prev_families, fams)

    final_families = prev_families[:top_k_families]

    return {
        "families":       final_families,
        "images":         layer_images,
        "layer_families": layer_families,
    }