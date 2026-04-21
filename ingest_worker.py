# ingest_worker.py
"""
Parallel lot ingestion worker.

Workflow per slab:
  1. Check Cosmos (by id + stone_family partition key) — skip if already
     ingested with the same image URL; re-ingest if URL changed.
  2. Download image in parallel (no lock needed).
  3. SAM crop (serialized via _sam_lock — SAM is not thread-safe).
     Crop NEVER fails — falls back through 4 strategies.
  4. Embed cropped image via DINOv2.
  5. Extract color signature.
  6. Upsert document into Cosmos DB.

Resume behaviour
────────────────
  • Same URL already in Cosmos → skip immediately (no download, no crop).
  • URL changed since last ingest → re-process so embedding stays current.

Parallelism model
─────────────────
  • Downloads, Cosmos reads, embedding, color extraction → parallel across threads.
  • SAM inference → serialized via _sam_lock (SAM global model is not thread-safe).

Crop fallback chain (never returns None)
─────────────────────────────────────────
  1. SAM mask covering image centre (conf=0.4)        ← ideal
  2. Largest SAM mask regardless of centre (conf=0.4) ← stone not at exact centre
  3. SAM with relaxed confidence (conf=0.1)           ← low-contrast slabs
  4. Centre-weighted geometric crop (no SAM)          ← guaranteed, always works

Document schema stored per slab:
  {
    "id":                  str(slab_id),
    "slab_id":             int,
    "img_lot_no":          str,
    "img_slab_no":         int | str,
    "lot_id":              int,
    "lot_status":          str,
    "stone_family":        str,   ← Cosmos partition key (/stone_family)
    "img_blob_path":       str,   ← framed_stand_img_1 URL (source of truth)
    "embedding":           list[float],
    "img_color_signature": list[dict],
    "ingested_at":         str,
    "crop_strategy":       str,   ← which fallback was used
  }
"""

import logging
import threading
import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from PIL import Image

import features.dino_embedder as dino_embedder
from features.color_extractor import extract_color_signature
from crop_utils import model as sam_model   # SAM singleton loaded at import time

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

log = logging.getLogger("ingest_worker")

if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s | %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

# ─────────────────────────────────────────────
# SAM LOCK
# SAM is NOT thread-safe. Serialize all .predict() calls.
# Downloads / Cosmos reads / embedding / color still run in parallel.
# ─────────────────────────────────────────────

_sam_lock = threading.Lock()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_existing_doc(cosmos_container, doc_id: str, stone_family: str) -> dict | None:
    """
    Point-read a Cosmos document by id.

    IMPORTANT: This container is partitioned by /stone_family, so the
    partition key must be the stone_family value — NOT the doc id.
    Passing the wrong partition key always returns 404 even if the doc exists.
    """
    try:
        return cosmos_container.read_item(item=doc_id, partition_key=stone_family)
    except Exception as e:
        err_str = str(e)
        # 404 = doc genuinely doesn't exist yet — expected, not an error
        if "404" not in err_str and "NotFound" not in err_str:
            log.warning("COSMOS READ ERROR | doc_id=%s | family=%s | err=%s", doc_id, stone_family, e)
        return None


def _download_image(img_url: str, prefix: str):
    """
    Download and decode an image from a URL.

    Returns:
        numpy ndarray (BGR) on success, or None on failure.
    """
    try:
        response = requests.get(img_url, timeout=15)
    except requests.exceptions.RequestException as exc:
        log.warning("DOWNLOAD EXCEPTION | %s | err=%s", prefix, exc)
        return None

    if response.status_code != 200:
        log.warning("DOWNLOAD FAILED | %s | http=%d", prefix, response.status_code)
        return None

    img_array = np.frombuffer(response.content, np.uint8)
    img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        log.warning("DECODE FAILED | %s | url=%s", prefix, img_url)
        return None

    h, w = img_bgr.shape[:2]
    log.info("DOWNLOAD SUCCESS | %s | size=%dx%d", prefix, w, h)
    return img_bgr


def _crop_from_mask(img_bgr: np.ndarray, mask: np.ndarray, strategy: str, prefix: str) -> Image.Image | None:
    """
    Given a boolean mask, extract the bounding-box crop with a small inset.

    Returns:
        PIL Image (RGB) on success, or None if the crop is empty/degenerate.
    """
    h, w = img_bgr.shape[:2]
    y_coords, x_coords = np.where(mask)

    if len(y_coords) == 0:
        return None

    inset  = 5
    top    = max(0, int(np.min(y_coords)) + inset)
    bottom = min(h, int(np.max(y_coords)) - inset)
    left   = max(0, int(np.min(x_coords)) + inset)
    right  = min(w, int(np.max(x_coords)) - inset)

    if bottom <= top or right <= left:
        return None

    cropped_bgr = img_bgr[top:bottom, left:right]
    if cropped_bgr.size == 0:
        return None

    crop_h, crop_w = cropped_bgr.shape[:2]
    log.info(
        "SAM | mask selected | chosen_px=%d | crop=%dx%d | %s",
        int(np.sum(mask)), crop_w, crop_h, prefix,
    )
    return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))


def _run_sam(img_bgr: np.ndarray, center_x: int, center_y: int, conf: float):
    """
    Run SAM predict seeded at image centre.

    Returns:
        numpy bool array of shape (N, H, W), or None if no masks produced.
    """
    results = sam_model.predict(
        img_bgr,
        points=[[center_x, center_y]],
        labels=[1],
        conf=conf,
        verbose=False,
    )
    if results[0].masks is None:
        return None
    return results[0].masks.data.cpu().numpy().astype(bool)


def _sam_crop(img_bgr: np.ndarray, prefix: str) -> tuple[Image.Image, str]:
    """
    Run SAM segmentation and return a cropped PIL image + the strategy used.

    NEVER returns None — guaranteed to always return a valid crop via one of:
      1. SAM mask covering image centre         (conf=0.4) — ideal
      2. Largest SAM mask ignoring centre       (conf=0.4) — stone off-centre
      3. SAM with relaxed confidence            (conf=0.1) — low-contrast slabs
      4. Centre-weighted geometric crop (no SAM)           — always succeeds

    Args:
        img_bgr: BGR numpy array of the downloaded image.
        prefix:  Log prefix string (slab/lot/family context).

    Returns:
        (PIL.Image, strategy_name) tuple — always a valid image.
    """
    h, w = img_bgr.shape[:2]
    center_y, center_x = h // 2, w // 2

    # ── Strategy 1: Centre-covering mask at conf=0.4 ──────────────────────────
    masks = _run_sam(img_bgr, center_x, center_y, conf=0.4)

    if masks is not None and len(masks) > 0:
        centre_candidates = [i for i, m in enumerate(masks) if m[center_y, center_x]]

        if centre_candidates:
            best_idx = centre_candidates[np.argmax([np.sum(masks[i]) for i in centre_candidates])]
            result = _crop_from_mask(img_bgr, masks[best_idx], "centre_mask_conf0.4", prefix)
            if result is not None:
                return result, "centre_mask_conf0.4"

        # ── Strategy 2: Largest mask regardless of centre ─────────────────────
        log.warning(
            "SAM | no centre-covering mask | total_masks=%d | trying largest mask | %s",
            len(masks), prefix,
        )
        best_idx = int(np.argmax([np.sum(m) for m in masks]))
        result = _crop_from_mask(img_bgr, masks[best_idx], "largest_mask_conf0.4", prefix)
        if result is not None:
            return result, "largest_mask_conf0.4"
    else:
        log.warning("SAM | no masks returned at conf=0.4 | %s", prefix)

    # ── Strategy 3: Relaxed confidence conf=0.1 ───────────────────────────────
    log.warning("SAM | retrying with conf=0.1 | %s", prefix)
    masks_relaxed = _run_sam(img_bgr, center_x, center_y, conf=0.1)

    if masks_relaxed is not None and len(masks_relaxed) > 0:
        # Prefer centre-covering first, then largest
        centre_candidates = [i for i, m in enumerate(masks_relaxed) if m[center_y, center_x]]
        if centre_candidates:
            best_idx = centre_candidates[np.argmax([np.sum(masks_relaxed[i]) for i in centre_candidates])]
        else:
            best_idx = int(np.argmax([np.sum(m) for m in masks_relaxed]))

        result = _crop_from_mask(img_bgr, masks_relaxed[best_idx], "largest_mask_conf0.1", prefix)
        if result is not None:
            return result, "largest_mask_conf0.1"
    else:
        log.warning("SAM | no masks returned at conf=0.1 | %s", prefix)

    # ── Strategy 4: Centre-weighted geometric crop (guaranteed) ───────────────
    # Takes the inner 60% of the image. Stone slabs are always centred in frame
    # so this always captures the relevant material without needing SAM at all.
    log.warning("SAM | all SAM strategies exhausted — using centre geometric crop | %s", prefix)

    margin_y = int(h * 0.20)
    margin_x = int(w * 0.20)
    cropped_bgr = img_bgr[margin_y: h - margin_y, margin_x: w - margin_x]

    if cropped_bgr.size == 0:
        # Absolute last resort: return full image (cannot fail)
        log.warning("SAM | centre crop empty — returning full image | %s", prefix)
        cropped_bgr = img_bgr

    crop_h, crop_w = cropped_bgr.shape[:2]
    log.info("SAM | strategy=centre_geometric_fallback | crop=%dx%d | %s", crop_w, crop_h, prefix)
    return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)), "centre_geometric_fallback"


# ─────────────────────────────────────────────
# SINGLE SLAB PROCESSOR
# ─────────────────────────────────────────────

def process_slab(slab: dict, lot: dict, cosmos_container) -> dict:
    """
    Process one slab image end-to-end.

    Args:
        slab:             Single slab dict from inventory API.
                          Expected keys: "id", "framed_stand_img_1", "img_slab_no"
        lot:              Parent lot dict.
                          Expected keys: "id", "lotno", "lot_status", "family"
        cosmos_container: Azure Cosmos DB container client.

    Returns:
        dict with keys:
          "status"        → "success" | "skip" | "error"
          "reason"        → human-readable string (on skip/error)
          "slab_id"       → slab identifier
          "lot_no"        → lot number string
          "crop_strategy" → which crop strategy was used (on success)
    """
    slab_id  = slab.get("id")
    img_url  = slab.get("framed_stand_img_1")
    slab_no  = slab.get("img_slab_no") or slab.get("id")

    lot_id     = lot.get("id")
    lot_no     = lot.get("lotno", "unknown")
    lot_status = lot.get("lot_status")
    family     = lot.get("family")

    prefix = f"slab={slab_id} lot={lot_no} family={family}"

    # ── Basic validation ──────────────────────────────────────────────────────
    if not slab_id:
        log.warning("SKIP | missing slab id | lot=%s", lot_no)
        return {"status": "skip", "reason": "missing slab id", "lot_no": lot_no}

    if not img_url:
        log.warning("SKIP | no framed_stand_img_1 | %s", prefix)
        return {"status": "skip", "reason": "no framed_stand_img_1 url", "slab_id": slab_id, "lot_no": lot_no}

    if not family:
        log.warning("SKIP | missing family | %s", prefix)
        return {"status": "skip", "reason": "missing stone_family", "slab_id": slab_id, "lot_no": lot_no}

    doc_id = str(slab_id)

    # ── Resume check ──────────────────────────────────────────────────────────
    # Partition key is /stone_family — must pass family, NOT doc_id.
    existing = _get_existing_doc(cosmos_container, doc_id, family)
    if existing is not None:
        stored_url = existing.get("img_blob_path", "")
        if stored_url == img_url:
            log.info("SKIP (already ingested, URL unchanged) | %s", prefix)
            return {
                "status":  "skip",
                "reason":  "already ingested — url unchanged",
                "slab_id": slab_id,
                "lot_no":  lot_no,
            }
        log.info("RE-INGEST (URL changed) | %s | old=%s | new=%s", prefix, stored_url, img_url)

    # ── Step 1: Download (runs in parallel across threads) ────────────────────
    log.info("DOWNLOAD START | %s | url=%s", prefix, img_url)
    img_bgr = _download_image(img_url, prefix)
    if img_bgr is None:
        return {"status": "error", "reason": "download/decode failed", "slab_id": slab_id, "lot_no": lot_no}

    # ── Step 2: SAM crop (serialized — SAM is not thread-safe) ───────────────
    # _sam_crop NEVER returns None — it always falls back to a geometric crop.
    log.info("CROP START | %s", prefix)
    with _sam_lock:
        cropped_pil, crop_strategy = _sam_crop(img_bgr, prefix)

    crop_w, crop_h = cropped_pil.size
    log.info("CROP SUCCESS | %s | crop_size=%dx%d | strategy=%s", prefix, crop_w, crop_h, crop_strategy)

    # ── Step 3: DINOv2 embedding ──────────────────────────────────────────────
    embedding = dino_embedder.embed_image(cropped_pil)
    if embedding is None:
        log.error("EMBED FAILED | %s", prefix)
        return {"status": "error", "reason": "embedding failed", "slab_id": slab_id, "lot_no": lot_no}
    log.info("EMBED SUCCESS | %s | dim=%d", prefix, len(embedding))

    # ── Step 4: Color signature ───────────────────────────────────────────────
    try:
        color_signature = extract_color_signature(cropped_pil)
        log.info("COLOR SUCCESS | %s | clusters=%d", prefix, len(color_signature))
    except Exception as exc:
        log.warning("COLOR FAILED (non-fatal, storing empty) | %s | err=%s", prefix, exc)
        color_signature = []

    # ── Step 5: Build and upsert Cosmos document ──────────────────────────────
    document = {
        "id":                  doc_id,
        "slab_id":             slab_id,
        "img_lot_no":          lot_no,
        "img_slab_no":         slab_no,
        "lot_id":              lot_id,
        "lot_status":          lot_status,
        "stone_family":        family,          # ← Cosmos partition key
        "img_blob_path":       img_url,         # ← source of truth for skip check
        "embedding":           embedding.tolist(),
        "img_color_signature": color_signature,
        "crop_strategy":       crop_strategy,   # ← which fallback was used
        "ingested_at":         _utc_now(),
    }

    try:
        cosmos_container.upsert_item(document)
        log.info("INSERTED | %s | crop_strategy=%s", prefix, crop_strategy)
        return {"status": "success", "slab_id": slab_id, "lot_no": lot_no, "crop_strategy": crop_strategy}
    except Exception as exc:
        log.error("COSMOS UPSERT FAILED | %s | err=%s", prefix, exc)
        return {"status": "error", "reason": f"cosmos upsert: {exc}", "slab_id": slab_id, "lot_no": lot_no}


# ─────────────────────────────────────────────
# LOT-LEVEL PARALLEL PROCESSOR
# ─────────────────────────────────────────────

def process_lot_parallel(lot: dict, cosmos_container, max_workers: int = 8) -> list:
    """
    Process all slabs in a single lot in parallel.

    Args:
        lot:              Lot dict from inventory API.
        cosmos_container: Azure Cosmos DB container client.
        max_workers:      Thread pool size.
                          Downloads/Cosmos/embedding run in parallel.
                          SAM is serialized internally via _sam_lock.

    Returns:
        List of result dicts, one per slab.
    """
    slabs  = lot.get("slabs", [])
    lot_no = lot.get("lotno", "unknown")

    if not slabs:
        log.warning("No slabs found | lot=%s", lot_no)
        return [{"status": "skip", "reason": "no slabs", "lot_no": lot_no}]

    valid_slabs      = [s for s in slabs if isinstance(s, dict)]
    skipped_non_dict = len(slabs) - len(valid_slabs)

    if skipped_non_dict:
        log.warning("Skipped %d non-dict slab entries | lot=%s", skipped_non_dict, lot_no)

    log.info("LOT START | lot=%s | slabs=%d | workers=%d", lot_no, len(valid_slabs), max_workers)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_slab = {
            executor.submit(process_slab, slab, lot, cosmos_container): slab
            for slab in valid_slabs
        }

        for future in as_completed(future_to_slab):
            slab    = future_to_slab[future]
            slab_id = slab.get("id")
            try:
                results.append(future.result())
            except Exception as exc:
                log.error("UNHANDLED EXCEPTION | slab=%s lot=%s | err=%s", slab_id, lot_no, exc)
                results.append({
                    "status":  "error",
                    "reason":  f"unhandled exception: {exc}",
                    "slab_id": slab_id,
                    "lot_no":  lot_no,
                })

    success = sum(1 for r in results if r.get("status") == "success")
    skipped = sum(1 for r in results if r.get("status") == "skip")
    failed  = sum(1 for r in results if r.get("status") == "error")

    log.info("LOT DONE | lot=%s | success=%d skip=%d failed=%d", lot_no, success, skipped, failed)
    return results