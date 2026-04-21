# ingestion/ingest_dataset.py
"""
Ingest a stone image dataset into TWO ChromaDB collections:

  stone_colors      — 248-dim colour feature vectors (fast, CPU-only)
  stone_embeddings  — 256-dim DINOv2 embedding vectors (GPU-accelerated if available)

Both collections store metadata:
  { "family": <folder_name>, "path": <absolute_path> }

Run this once (or incrementally — already-indexed images are skipped).

Usage:
    python -m ingestion.ingest_dataset /path/to/dataset
"""

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from config.settings import BATCH_SIZE, IMAGE_EXTS
from db.chroma_client import get_color_collection, get_embedding_collection
from features.stone_vector import extract_stone_vector
from utils.file_utils import get_all_images


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def _existing_ids(collection) -> set:
    try:
        return set(collection.get(include=[])["ids"])
    except Exception:
        return set()


# ── Colour ingestion (multiprocess, CPU) ──────────────────────────────────────

def _process_color(data: tuple) -> dict | None:
    img_path, family = data
    try:
        vec = extract_stone_vector(str(img_path))
        if vec is None:
            return None
        return {
            "id":        _make_id(str(img_path)),
            "embedding": vec.tolist(),
            "metadata":  {"family": family, "path": str(img_path)},
        }
    except Exception:
        return None


def ingest_colors(parent_folder: str, num_workers: int | None = None):
    """Ingest colour vectors into stone_colors collection."""
    col  = get_color_collection()
    data = get_all_images(parent_folder)
    print(f"[Colour] Total images: {len(data)}")

    existing = _existing_ids(col)
    print(f"[Colour] Already indexed: {len(existing)}")

    nw = num_workers or max(1, multiprocessing.cpu_count() - 2)
    ids, embs, metas = [], [], []

    with ProcessPoolExecutor(max_workers=nw) as ex:
        futures = [ex.submit(_process_color, item) for item in data]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Colour"):
            res = fut.result()
            if res is None or res["id"] in existing:
                continue
            ids.append(res["id"])
            embs.append(res["embedding"])
            metas.append(res["metadata"])
            if len(ids) >= BATCH_SIZE:
                col.add(ids=ids, embeddings=embs, metadatas=metas)
                ids, embs, metas = [], [], []

    if ids:
        col.add(ids=ids, embeddings=embs, metadatas=metas)

    print(f"[Colour] ✅ Done. Collection size: {col.count()}")


# ── Embedding ingestion (batched, GPU-friendly) ───────────────────────────────

def ingest_embeddings(parent_folder: str, batch_size: int = 32):
    """Ingest DINOv2 embeddings into stone_embeddings collection."""
    from features.dino_embedder import embed_batch

    col  = get_embedding_collection()
    data = get_all_images(parent_folder)
    print(f"[Embedding] Total images: {len(data)}")

    existing = _existing_ids(col)
    print(f"[Embedding] Already indexed: {len(existing)}")

    # Filter to un-indexed only
    todo = [(p, f) for p, f in data if _make_id(str(p)) not in existing]
    print(f"[Embedding] To ingest: {len(todo)}")

    paths   = [str(p) for p, _ in todo]
    families = [f for _, f in todo]

    # Process in chunks so we can show progress
    chunk = batch_size * 8
    for start in tqdm(range(0, len(paths), chunk), desc="Embedding"):
        chunk_paths    = paths[start:start + chunk]
        chunk_families = families[start:start + chunk]

        valid_paths, embs = embed_batch(chunk_paths, batch_size=batch_size)
        if not valid_paths:
            continue

        # Map back to family names
        path_to_family = dict(zip(chunk_paths, chunk_families))

        ids, vecs, metas = [], [], []
        for vp, emb in zip(valid_paths, embs):
            doc_id = _make_id(vp)
            if doc_id in existing:
                continue
            ids.append(doc_id)
            vecs.append(emb.tolist())
            metas.append({"family": path_to_family.get(vp, "unknown"), "path": vp})

        if ids:
            col.add(ids=ids, embeddings=vecs, metadatas=metas)

    print(f"[Embedding] ✅ Done. Collection size: {col.count()}")


# ── Unified entry point ───────────────────────────────────────────────────────

def ingest_dataset(
    parent_folder: str,
    do_color: bool = True,
    do_embedding: bool = True,
    num_workers: int | None = None,
):
    if do_color:
        print("\n── Stage: Colour ingestion ──────────────────────────────")
        ingest_colors(parent_folder, num_workers=num_workers)

    if do_embedding:
        print("\n── Stage: Embedding ingestion ───────────────────────────")
        ingest_embeddings(parent_folder)

    print("\n✅ Full ingestion complete.")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    ingest_dataset(folder)