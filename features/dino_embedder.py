# features/dino_embedder.py
"""
DINOv2 embedding extractor.

Loads the fine-tuned StoneEmbedder once, exposes:
  - get_model()          → StoneEmbedder singleton
  - embed_image(image)   → 256-dim L2-normalised numpy vector
                           Accepts: file path (str | Path) OR PIL.Image OR np.ndarray (BGR)
  - embed_batch(paths)   → (valid_paths, (N, 256) array)  [file paths only]
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from config.settings import CKPT_DIR, EMBED_DIM, PROJ_DIM, IMG_SIZE

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_TF = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────

class StoneEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=IMG_SIZE,
        )
        bdim = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(bdim, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(PROJ_DIM, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        feat = self.backbone(x)
        emb  = F.normalize(self.projector(feat), dim=-1)
        if return_embedding:
            return emb
        return emb, self.classifier(emb)


# ─────────────────────────────────────────────
# SINGLETON LOADER
# ─────────────────────────────────────────────

_model       = None
_num_classes = None


def _load_model(num_classes: int) -> StoneEmbedder:
    global _model, _num_classes
    if _model is not None and _num_classes == num_classes:
        return _model

    ckpt_dir  = Path(CKPT_DIR)
    ckpt_file = ckpt_dir / "best_stone_model_stage2.pt"
    if not ckpt_file.exists():
        ckpt_file = ckpt_dir / "best_stone_model.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"[DINOEmbedder] No model weights in '{ckpt_dir}'. "
            "Expected best_stone_model_stage2.pt or best_stone_model.pt"
        )

    ck    = torch.load(ckpt_file, map_location=_DEVICE)
    model = StoneEmbedder(num_classes).to(_DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    _model       = model
    _num_classes = num_classes
    print(f"[DINOEmbedder] ✅ Loaded {ckpt_file.name} | device={_DEVICE}")
    return model


def _get_num_classes() -> int:
    """Read num_classes from stone_index_meta.pkl."""
    import pickle
    meta_path = Path(CKPT_DIR) / "stone_index_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return len(meta["family_names"])


def get_model() -> StoneEmbedder:
    """Return the loaded StoneEmbedder singleton."""
    return _load_model(_get_num_classes())


# ─────────────────────────────────────────────
# INTERNAL: any input → RGB numpy array
# ─────────────────────────────────────────────

def _to_rgb_numpy(image) -> np.ndarray | None:
    """
    Accepts:
      - str / Path   → reads via cv2
      - PIL.Image    → converts directly
      - np.ndarray   → assumed BGR (cv2 convention), converts to RGB

    Returns HxWx3 uint8 RGB numpy array, or None on failure.
    """
    try:
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                print(f"[DINOEmbedder] ❌ Could not read file: {image}")
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                # Assume BGR (cv2 convention) → RGB
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("[DINOEmbedder] ❌ Unsupported ndarray shape")
            return None

        else:
            print(f"[DINOEmbedder] ❌ Unsupported image type: {type(image)}")
            return None

    except Exception as e:
        print(f"[DINOEmbedder] ❌ _to_rgb_numpy error: {e}")
        return None


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def embed_image(image) -> np.ndarray | None:
    """
    Return a 256-dim L2-normalised embedding vector, or None on failure.

    Accepts:
      - file path (str or Path)
      - PIL.Image
      - numpy ndarray (BGR, cv2 convention)
    """
    img_rgb = _to_rgb_numpy(image)
    if img_rgb is None:
        return None

    tensor = VAL_TF(image=img_rgb)["image"].unsqueeze(0).to(_DEVICE)
    model  = get_model()

    with torch.no_grad():
        vec = model(tensor, return_embedding=True).cpu().numpy().astype(np.float32)

    return vec[0]   # (256,) — L2-normalised by model.forward()


def embed_batch(image_paths: list[str], batch_size: int = 32) -> tuple[list[str], np.ndarray]:
    """
    Embed a list of image file paths in batches.

    Returns:
        (valid_paths, embeddings_array)  where embeddings_array is (N, 256).
    """
    model       = get_model()
    valid_paths = []
    all_embs    = []
    imgs_buf    = []
    path_buf    = []

    def _flush():
        if not imgs_buf:
            return
        batch = torch.stack(imgs_buf).to(_DEVICE)
        with torch.no_grad():
            embs = model(batch, return_embedding=True).cpu().numpy().astype(np.float32)
        all_embs.append(embs)
        valid_paths.extend(path_buf)
        imgs_buf.clear()
        path_buf.clear()

    for p in image_paths:
        img_rgb = _to_rgb_numpy(p)
        if img_rgb is None:
            continue
        t = VAL_TF(image=img_rgb)["image"]
        imgs_buf.append(t)
        path_buf.append(str(p))
        if len(imgs_buf) >= batch_size:
            _flush()

    _flush()

    if not all_embs:
        return [], np.zeros((0, EMBED_DIM), dtype=np.float32)

    return valid_paths, np.concatenate(all_embs, axis=0)