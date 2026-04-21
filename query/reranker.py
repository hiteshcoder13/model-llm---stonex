# query/reranker.py
"""
Stage-2: DINOv2-small + FAISS reranker.

Takes the colour-pipeline candidates and reranks them using
the fine-tuned visual model stored in stonex_checkpoints/.

Usage:
    from query.reranker import rerank

    # candidates = [(family_name, colour_score), ...]
    reranked = rerank(image_path, candidates, top_k=10)
    # returns  [(family_name, dino_score), ...]
"""

import cv2
import faiss
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from pathlib import Path

from query.name_utils import build_alias_map, resolve_name

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 256
PROJ_DIM  = 512
IMG_SIZE  = 224

# ✅ Matches your actual folder name in the project
CKPT_DIR  = Path("stonex_checkpoints")

VAL_TF = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ── Model architecture (must match Kaggle notebook exactly) ───────────────────
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


# ── Load all artifacts once at import time ────────────────────────────────────
def _load():
    meta_path = CKPT_DIR / "stone_index_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"[Reranker] stone_index_meta.pkl not found in '{CKPT_DIR}'.\n"
            "Make sure stonex_checkpoints/ is in your project root."
        )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    family_names = meta["family_names"]   # list of strings
    fam2idx      = meta["fam2idx"]        # dict  name → int index
    labels_arr   = np.array(meta["labels"])
    alias_map    = build_alias_map(family_names)

    faiss_path = CKPT_DIR / "stone_index.faiss"
    if not faiss_path.exists():
        raise FileNotFoundError(f"[Reranker] stone_index.faiss not found in '{CKPT_DIR}'.")
    index = faiss.read_index(str(faiss_path))

    # Prefer Stage-2 weights; fall back to Stage-1
    ckpt_file = CKPT_DIR / "best_stone_model_stage2.pt"
    if not ckpt_file.exists():
        ckpt_file = CKPT_DIR / "best_stone_model.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"[Reranker] No model weights found in '{CKPT_DIR}'.\n"
            "Expected best_stone_model_stage2.pt or best_stone_model.pt"
        )

    ck    = torch.load(ckpt_file, map_location=DEVICE)
    model = StoneEmbedder(len(family_names)).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    print(
        f"[Reranker] ✅  {len(family_names)} families | "
        f"{index.ntotal} vectors | {ckpt_file.name} | device={DEVICE}"
    )
    return model, index, family_names, fam2idx, labels_arr, alias_map


_model, _index, _family_names, _fam2idx, _labels_arr, _alias_map = _load()


# ── Public API ────────────────────────────────────────────────────────────────
def rerank(
    image_path: str,
    candidates: list,
    top_k: int = 10,
    per_fam_vectors: int = 20,
) -> list:
    """
    Rerank colour-pipeline candidates using DINOv2 embeddings.

    Parameters
    ----------
    image_path      : path to the query image (temp file from Streamlit)
    candidates      : [(family_name, colour_score), ...]
                      family_name can have spaces OR underscores — both handled
    top_k           : number of results to return
    per_fam_vectors : max FAISS neighbours per family used for scoring

    Returns
    -------
    [(canonical_family_name, dino_score), ...]  sorted best-first
    """

    # 1. Embed the query image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"[Reranker] Cannot read image: {image_path}")
    img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = VAL_TF(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        vec = _model(tensor, return_embedding=True).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(vec)

    # 2. Resolve candidate names → model canonical names
    resolved   = []
    unresolved = []
    for name, _score in candidates:
        canonical = resolve_name(name, _alias_map)
        if canonical:
            resolved.append(canonical)
        else:
            unresolved.append(name)

    if unresolved:
        print(f"[Reranker] ⚠️  {len(unresolved)} names not matched in model index: {unresolved[:5]}")

    if not resolved:
        print("[Reranker] ⚠️  No candidates could be resolved — returning empty.")
        return []

    # 3. FAISS search — broad search, filter to candidates only
    search_k = min(800, _index.ntotal)
    sims, idxs = _index.search(vec, search_k)
    sims, idxs = sims[0], idxs[0]

    candidate_lbl_set = {_fam2idx[n] for n in resolved if n in _fam2idx}

    # 4. Aggregate per-family similarity scores
    family_sims = defaultdict(list)
    for sim, di in zip(sims, idxs):
        lbl = int(_labels_arr[di])
        if lbl in candidate_lbl_set and len(family_sims[lbl]) < per_fam_vectors:
            family_sims[lbl].append(float(sim))

    # Score = mean of top-3 nearest neighbours for that family
    ranked = []
    for lbl, sim_list in family_sims.items():
        top3_mean = float(np.mean(sorted(sim_list, reverse=True)[:3]))
        ranked.append((_family_names[lbl], top3_mean))

    # Families with 0 FAISS hits still appear at the bottom
    for name in resolved:
        lbl = _fam2idx.get(name)
        if lbl is not None and lbl not in family_sims:
            ranked.append((name, 0.0))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]