# config/settings.py

# ── ChromaDB Collections ───────────────────────────────────────────────────────
CHROMA_DIR             = "./chroma_db"
CHROMA_COLOR_COLLECTION    = "stone_colors_original_fold"       # colour feature vectors (248-dim)
CHROMA_EMBEDDING_COLLECTION = "stone_embeddings_original_fold"  # DINOv2 embeddings (256-dim)

# ── Model / FAISS ─────────────────────────────────────────────────────────────
CKPT_DIR   = "./stonex_checkpoints"
EMBED_DIM  = 256
PROJ_DIM   = 512
IMG_SIZE   = 224
DEVICE     = "cuda"   # overridden at runtime if cuda not available

# ── Ingestion ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 256
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}

# ── Colour feature ─────────────────────────────────────────────────────────────
N_CLUSTERS = 8
HIST_BINS  = 64