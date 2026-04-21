# features/stone_vector.py
"""
248-dim colour feature vector for a stone image.
  - 192 dims : LAB histogram  (3 channels × 64 bins)
  -  40 dims : KMeans cluster centres + weight + spread (8 clusters × 5)
  -  24 dims : Gabor vein/base colour means+stds (2 regions × 3 channels × 2 stats)
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from config.settings import N_CLUSTERS, HIST_BINS


def extract_lab_histogram(lab: np.ndarray) -> np.ndarray:
    hists = []
    for ch in range(3):
        h, _ = np.histogram(lab[:, :, ch].ravel(), bins=HIST_BINS, density=True)
        hists.append(h)
    return np.concatenate(hists).astype(np.float32)


def extract_kmeans_clusters(lab: np.ndarray, k: int = N_CLUSTERS) -> np.ndarray:
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


def gabor_vein_mask(gray: np.ndarray) -> np.ndarray:
    responses = []
    for theta in np.linspace(0, np.pi, 8):
        kern = cv2.getGaborKernel((21, 21), 4.0, theta, 8.0, 0.5)
        responses.append(cv2.filter2D(gray, cv2.CV_32F, kern))
    energy = np.max(np.abs(responses), axis=0)
    _, mask = cv2.threshold(energy, energy.mean() + energy.std(), 1, cv2.THRESH_BINARY)
    return mask.astype(bool)


def extract_vein_base_colors(lab: np.ndarray) -> np.ndarray:
    gray  = lab[:, :, 0]
    vmask = gabor_vein_mask(gray)
    vein_mean = lab[vmask].mean(axis=0)  if vmask.any()   else np.zeros(3)
    base_mean = lab[~vmask].mean(axis=0) if (~vmask).any() else np.zeros(3)
    vein_std  = lab[vmask].std(axis=0)   if vmask.any()   else np.zeros(3)
    base_std  = lab[~vmask].std(axis=0)  if (~vmask).any() else np.zeros(3)
    return np.concatenate([vein_mean, vein_std, base_mean, base_std]).astype(np.float32)


def extract_stone_vector(image_path: str) -> np.ndarray | None:
    """Return 248-dim colour feature vector, or None on failure."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (512, 512))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.concatenate([
        extract_lab_histogram(lab),
        extract_kmeans_clusters(lab),
        extract_vein_base_colors(lab),
    ])