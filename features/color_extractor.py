# features/color_extractor.py
"""
Color signature extractor.

Uses SLIC superpixels + Gaussian Mixture Model to produce a compact
colour signature in LAB space for a stone slab image.

Public API:
  extract_color_signature(image) → list[{"lab": [...], "weight": float}]

  Accepts: PIL.Image (primary) or file path str (legacy).
"""

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from skimage.segmentation import slic
from sklearn.mixture import GaussianMixture

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

N_SEGMENTS    = 200
COMPACTNESS   = 10
SIGMA         = 1
MIN_AREA_RATIO = 0.005
N_CLUSTERS    = 3


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def pil_to_rgb_np(img: Image.Image, target_size: int = 512) -> np.ndarray:
    """Convert a PIL Image to a resized RGB numpy array."""
    try:
        img = img.convert("RGB")
        image = np.array(img)

        h, w, _ = image.shape
        scale = target_size / max(h, w)

        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        return image

    except Exception:
        logger.exception("❌ Failed converting PIL image")
        raise


def load_image_rgb_from_path(path: str, target_size: int = 512) -> np.ndarray:
    """Legacy support: load from local file path."""
    try:
        logger.debug(f"📥 Loading image | path={path}")

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image not found: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        scale = target_size / max(h, w)

        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        return img

    except Exception:
        logger.exception("❌ Failed to load image from path")
        raise


# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def get_foreground_mask(image_rgb: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Return a boolean mask for non-black (foreground) pixels."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return gray > threshold


def extract_superpixels(image_rgb: np.ndarray) -> list:
    """
    Segment image into SLIC superpixels and return list of
    (mean_LAB_colour, area_ratio) for foreground segments above MIN_AREA_RATIO.
    """
    logger.info("🧠 Extracting superpixels")

    lab      = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    fg_mask  = get_foreground_mask(image_rgb)
    segments = slic(
        image_rgb,
        n_segments=N_SEGMENTS,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=0,
    )

    total_pixels = fg_mask.sum()
    if total_pixels == 0:
        logger.warning("⚠️ No foreground pixels detected")
        return []

    sp_data = []

    for seg_id in np.unique(segments):
        mask  = (segments == seg_id) & fg_mask
        area  = mask.sum()

        if area == 0:
            continue

        ratio = area / total_pixels
        if ratio < MIN_AREA_RATIO:
            continue

        mean_lab = lab[mask].mean(axis=0)
        sp_data.append((mean_lab, ratio))

    logger.info(f"✅ Superpixels extracted | valid_segments={len(sp_data)}")
    return sp_data


def _build_color_signature(sp_data: list) -> list:
    """
    Fit a GMM over superpixel LAB colours (weighted by area) and return
    N_CLUSTERS cluster centres with their relative weights, sorted descending.
    """
    if not sp_data:
        return []

    colors  = np.array([c for c, _ in sp_data])
    weights = np.array([w for _, w in sp_data])

    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42)

    try:
        gmm.fit(colors, sample_weight=weights)
    except TypeError:
        # Older sklearn versions don't support sample_weight → expand manually
        expanded = np.repeat(
            colors,
            (weights * 100).astype(int) + 1,
            axis=0,
        )
        gmm.fit(expanded)

    labels  = gmm.predict(colors)
    centers = gmm.means_

    cluster_weights = np.zeros(N_CLUSTERS)
    for i, (_, w) in enumerate(sp_data):
        cluster_weights[labels[i]] += w

    cluster_weights /= cluster_weights.sum()

    signature = [
        {"lab": lab.tolist(), "weight": float(weight)}
        for lab, weight in zip(centers, cluster_weights)
    ]

    signature.sort(key=lambda x: x["weight"], reverse=True)
    return signature


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def extract_color_signature(image) -> list:
    """
    Extract a compact colour signature from a slab image.

    Args:
        image: PIL.Image (preferred — used during blob ingestion)
               or str file path (legacy).

    Returns:
        List of N_CLUSTERS dicts: [{"lab": [L, A, B], "weight": float}, ...]
        Sorted by weight descending (dominant colour first).
    """
    try:
        logger.info("🎨 Extracting color signature")

        if isinstance(image, Image.Image):
            image_rgb = pil_to_rgb_np(image)
        elif isinstance(image, str):
            image_rgb = load_image_rgb_from_path(image)
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        sp_data   = extract_superpixels(image_rgb)
        signature = _build_color_signature(sp_data)

        logger.info("✅ Color signature extracted successfully")
        return signature

    except Exception:
        logger.exception("❌ Failed to extract color signature")
        raise