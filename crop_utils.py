# crop_utils.py
"""
SAM-based image cropper.

Loads the SAM2 model once at module level.
Exposes:
  crop_from_url(image_url: str) → (PIL.Image, str) — always returns a valid image + strategy name

Crop fallback chain (never returns None)
─────────────────────────────────────────
  1. SAM mask covering image centre         (conf=0.4) — ideal
  2. Largest SAM mask ignoring centre       (conf=0.4) — stone not at exact centre pixel
  3. SAM with relaxed confidence            (conf=0.1) — low-contrast / difficult slabs
  4. Centre-weighted geometric crop (no SAM)           — guaranteed, always works
"""

import logging
import time

import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import SAM

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

log = logging.getLogger("crop_utils")

if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(name)s] %(levelname)s | %(message)s")
    )
    log.addHandler(_handler)
    log.setLevel(logging.INFO)


# ─────────────────────────────────────────────
# MODEL (loaded once at import time)
# ─────────────────────────────────────────────

log.info("Loading SAM2 model …")
model = SAM("sam2_s.pt")
log.info("SAM2 model ready.")


# ─────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────

def _run_sam(img_bgr: np.ndarray, center_x: int, center_y: int, conf: float):
    """
    Run SAM predict seeded at the image centre with the given confidence.

    Returns:
        numpy bool array of shape (N, H, W), or None if no masks produced.
    """
    results = model.predict(
        img_bgr,
        points=[[center_x, center_y]],
        labels=[1],
        conf=conf,
        verbose=False,
    )
    if results[0].masks is None:
        return None
    return results[0].masks.data.cpu().numpy().astype(bool)


def _crop_from_mask(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    strategy: str,
    url: str,
    elapsed_fn,
) -> Image.Image | None:
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
        "CROP_SUCCESS | strategy=%s | url=%s | original=%dx%d | crop=%dx%d"
        " | chosen_mask_px=%d | elapsed=%s",
        strategy, url, w, h, crop_w, crop_h,
        int(np.sum(mask)), elapsed_fn(),
    )
    return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def crop_from_url(image_url: str) -> tuple[Image.Image | None, str | None]:
    """
    Download an image from `image_url`, run the crop fallback chain, and
    return the best possible crop.

    Crop strategy chain (tried in order until one succeeds):
      1. SAM mask covering image centre         (conf=0.4)
      2. Largest SAM mask regardless of centre  (conf=0.4)
      3. SAM with relaxed confidence            (conf=0.1)
      4. Centre-weighted geometric crop (no SAM — always succeeds)

    Logging contract
    ────────────────
    Every exit path emits exactly one structured log line:

      INFO  CROP_SUCCESS  strategy=… url=… size=… elapsed=…
      WARN  CROP_SKIP     url=… reason=… elapsed=…   (only for download failures)
      ERROR CROP_ERROR    url=… reason=… elapsed=…   (only for unhandled exceptions)

    Returns:
        (PIL.Image, strategy_name) on success.
        (None, None) only if the image cannot be downloaded or decoded.
    """
    t0 = time.perf_counter()

    def _elapsed() -> str:
        return f"{time.perf_counter() - t0:.2f}s"

    try:
        # ── 1. Download ───────────────────────────────────────────────────────
        try:
            response = requests.get(image_url, timeout=10)
        except requests.exceptions.RequestException as exc:
            log.warning(
                "CROP_SKIP | url=%s | reason=download_exception | detail=%s | elapsed=%s",
                image_url, exc, _elapsed(),
            )
            return None, None

        if response.status_code != 200:
            log.warning(
                "CROP_SKIP | url=%s | reason=http_%d | elapsed=%s",
                image_url, response.status_code, _elapsed(),
            )
            return None, None

        # ── 2. Decode ─────────────────────────────────────────────────────────
        img_array = np.frombuffer(response.content, np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            log.warning(
                "CROP_SKIP | url=%s | reason=decode_failed | elapsed=%s",
                image_url, _elapsed(),
            )
            return None, None

        h, w, _ = img_bgr.shape
        log.info(
            "DOWNLOADED | url=%s | original_size=%dx%d | elapsed=%s",
            image_url, w, h, _elapsed(),
        )

        center_y, center_x = h // 2, w // 2

        # ── Strategy 1: Centre-covering mask at conf=0.4 ──────────────────────
        masks = _run_sam(img_bgr, center_x, center_y, conf=0.4)

        if masks is not None and len(masks) > 0:
            centre_candidates = [i for i, m in enumerate(masks) if m[center_y, center_x]]

            if centre_candidates:
                best_idx = centre_candidates[
                    np.argmax([np.sum(masks[i]) for i in centre_candidates])
                ]
                result = _crop_from_mask(
                    img_bgr, masks[best_idx],
                    "centre_mask_conf0.4", image_url, _elapsed,
                )
                if result is not None:
                    return result, "centre_mask_conf0.4"

            # ── Strategy 2: Largest mask regardless of centre ─────────────────
            log.warning(
                "CROP | no centre-covering mask | total_masks=%d"
                " | trying largest mask | url=%s | elapsed=%s",
                len(masks), image_url, _elapsed(),
            )
            best_idx = int(np.argmax([np.sum(m) for m in masks]))
            result = _crop_from_mask(
                img_bgr, masks[best_idx],
                "largest_mask_conf0.4", image_url, _elapsed,
            )
            if result is not None:
                return result, "largest_mask_conf0.4"
        else:
            log.warning(
                "CROP | no masks returned at conf=0.4 | url=%s | elapsed=%s",
                image_url, _elapsed(),
            )

        # ── Strategy 3: Relaxed confidence conf=0.1 ───────────────────────────
        log.warning(
            "CROP | retrying SAM with conf=0.1 | url=%s | elapsed=%s",
            image_url, _elapsed(),
        )
        masks_relaxed = _run_sam(img_bgr, center_x, center_y, conf=0.1)

        if masks_relaxed is not None and len(masks_relaxed) > 0:
            centre_candidates = [
                i for i, m in enumerate(masks_relaxed) if m[center_y, center_x]
            ]
            if centre_candidates:
                best_idx = centre_candidates[
                    np.argmax([np.sum(masks_relaxed[i]) for i in centre_candidates])
                ]
            else:
                best_idx = int(np.argmax([np.sum(m) for m in masks_relaxed]))

            result = _crop_from_mask(
                img_bgr, masks_relaxed[best_idx],
                "largest_mask_conf0.1", image_url, _elapsed,
            )
            if result is not None:
                return result, "largest_mask_conf0.1"
        else:
            log.warning(
                "CROP | no masks returned at conf=0.1 | url=%s | elapsed=%s",
                image_url, _elapsed(),
            )

        # ── Strategy 4: Centre-weighted geometric crop (guaranteed) ───────────
        # Takes the inner 60% of the image. Stone slabs are always centred in
        # frame so this always captures the relevant material without SAM.
        log.warning(
            "CROP | all SAM strategies exhausted — using centre geometric crop"
            " | url=%s | elapsed=%s",
            image_url, _elapsed(),
        )

        margin_y = int(h * 0.20)
        margin_x = int(w * 0.20)
        cropped_bgr = img_bgr[margin_y: h - margin_y, margin_x: w - margin_x]

        if cropped_bgr.size == 0:
            # Absolute last resort: return full image — cannot fail
            log.warning(
                "CROP | centre crop empty — returning full image"
                " | url=%s | elapsed=%s",
                image_url, _elapsed(),
            )
            cropped_bgr = img_bgr

        crop_h, crop_w = cropped_bgr.shape[:2]
        log.info(
            "CROP_SUCCESS | strategy=centre_geometric_fallback | url=%s"
            " | original=%dx%d | crop=%dx%d | elapsed=%s",
            image_url, w, h, crop_w, crop_h, _elapsed(),
        )
        return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)), "centre_geometric_fallback"

    except Exception as exc:
        log.error(
            "CROP_ERROR | url=%s | reason=unhandled_exception | detail=%s | elapsed=%s",
            image_url, exc, _elapsed(),
        )
        return None, None