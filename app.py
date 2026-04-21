import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tempfile
import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

# ── CMD Office mapping ────────────────────────────────────────────────────
from cmd_mapping import resolve_family_name, is_cmd_class

st.set_page_config(
    page_title="StoneX — Family Search",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #c9a96e 0%, #f0d5a0 50%, #c9a96e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #666;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
}
.layer-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    margin-right: 4px;
    letter-spacing: 0.05em;
}
.badge-model      { background: #2d1500; color: #ffa86f; border: 1px solid #8a3e00; }
.badge-rgb        { background: #0d2200; color: #7fff6f; border: 1px solid #2e6600; }
.badge-embedding  { background: #001a2d; color: #6fbfff; border: 1px solid #003e8a; }
.badge-top5       { background: #1a1500; color: #ffd700; border: 1px solid #665500; }
.badge-nonaug     { background: #1a0a2d; color: #c87fff; border: 1px solid #5a008a; }
.badge-fixed      { background: #0a1a0a; color: #7fff6f; border: 1px solid #2e6600; }
.badge-supplement { background: #1a0a1a; color: #ff88ff; border: 1px solid #8a008a; }
.badge-cmd        { background: #1a1000; color: #ffc94d; border: 1px solid #7a5000; }

.query-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
}
.query-card-header {
    background: #1a1a1a;
    padding: 8px 12px;
    border-bottom: 1px solid #2a2a2a;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #666;
    display: flex;
    align-items: center;
    gap: 8px;
}
.query-index-badge {
    background: #c9a96e22;
    border: 1px solid #c9a96e55;
    color: #c9a96e;
    border-radius: 6px;
    padding: 2px 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
}

.multi-summary-bar {
    background: linear-gradient(135deg, #0f0f0f 0%, #141008 100%);
    border: 1px solid #2a2200;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}

.family-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.rank-num    { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #555; min-width: 28px; }
.family-name { font-weight: 600; font-size: 0.95rem; flex: 1; color: #e0e0e0; }
.score-num   { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #888; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 1.4rem 0 0.7rem 0;
    color: #e0e0e0;
}
.stage-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin: 1rem 0 0.5rem 0;
    color: #bbb;
    display: flex;
    align-items: center;
    gap: 8px;
}
.img-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #777;
    padding: 4px 6px;
    line-height: 1.5;
}
.family-col-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #c9a96e;
    margin-bottom: 6px;
    text-align: center;
    line-height: 1.3;
    min-height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.info-box {
    background: #111827;
    border-left: 3px solid #c9a96e;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 12px;
    line-height: 1.6;
}
.info-box-gold {
    background: #141000;
    border-left: 3px solid #ffd700;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 12px;
    line-height: 1.6;
}
.info-box-green {
    background: #0a1a0a;
    border-left: 3px solid #4caf50;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 12px;
    line-height: 1.6;
}
.info-box-blue {
    background: #0a0f1a;
    border-left: 3px solid #2196f3;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 12px;
    line-height: 1.6;
}
.info-box-purple {
    background: #0f0a1a;
    border-left: 3px solid #9c27b0;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 12px;
    line-height: 1.6;
}
.info-box-teal {
    background: #0a1a1a;
    border-left: 3px solid #00bcd4;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #9ca3af;
    margin-bottom: 12px;
    line-height: 1.6;
}
.pipeline-flow {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 10px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #aaa;
    letter-spacing: 0.03em;
}
.section-divider {
    border: none;
    border-top: 1px solid #222;
    margin: 28px 0;
}
.not-found-box {
    background: #1a0a0a;
    border: 1px dashed #5a2020;
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 4px;
}
.source-chip {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    margin-left: 4px;
    vertical-align: middle;
}
.chip-top1    { background: #1a0a2d; color: #c87fff; border: 1px solid #5a008a; }
.chip-top2    { background: #0a1a2d; color: #7fc8ff; border: 1px solid #003e8a; }
.chip-fixed   { background: #0a1a0a; color: #7fff6f; border: 1px solid #2e6600; }
.chip-supplement { background: #1a0a1a; color: #ff88ff; border: 1px solid #8a008a; }
.chip-cmd     { background: #1a1000; color: #ffc94d; border: 1px solid #7a5000; }

/* Multi-image result separator */
.image-result-header {
    background: linear-gradient(90deg, #1a1200 0%, #111 100%);
    border: 1px solid #3a2800;
    border-radius: 10px;
    padding: 12px 18px;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.image-result-index {
    background: #c9a96e;
    color: #000;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 6px;
    min-width: 32px;
    text-align: center;
}
.image-result-filename {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #c9a96e;
    flex: 1;
}
.image-result-dims {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #444;
}

/* Overview grid for multi-image top families */
.overview-grid-item {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}

div[data-testid="stSidebar"] { background: #0c0c0c; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_ROOT       = "/home/Unthinkable/Downloads/onedrive(mayank) + ssd with reflection/159folder"
LOCAL_PATH_PREFIX  = "/home/Unthinkable/Downloads/onedrive(mayank) + ssd with reflection/Minimal-AUG-balanced-renamed/Minimal-AUG-balanced"
KAGGLE_PATH_PREFIX = "/kaggle/input/datasets/hunny2006/stonex/Minimal-AUG-balanced"

IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
layer_order        = ["model"]


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPER
# Centralises all name → display-name resolution so every place in the UI
# uses the same logic.
# ══════════════════════════════════════════════════════════════════════════════

def display_name(raw_family: str) -> str:
    """
    Return the human-readable stone name for a raw model class name.

    - CMD Office classes  → mapped stone name  (e.g. "Golden Spider")
    - Everything else     → underscores → spaces  (unchanged behaviour)
    """
    return resolve_family_name(raw_family)


def cmd_badge_html(raw_family: str) -> str:
    """Return an HTML badge if the family is a CMD Office class, else ''."""
    if is_cmd_class(raw_family):
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def remap_path(path: str) -> str:
    p = str(path)
    if p.startswith(KAGGLE_PATH_PREFIX):
        return LOCAL_PATH_PREFIX + p[len(KAGGLE_PATH_PREFIX):]
    return p


def normalize_name(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


def find_folder_for_family(family_name: str, root: str) -> str | None:
    target = normalize_name(family_name)
    try:
        entries = os.listdir(root)
    except (FileNotFoundError, PermissionError):
        return None

    for entry in entries:
        full = os.path.join(root, entry)
        if os.path.isdir(full) and normalize_name(entry) == target:
            return full

    for entry in entries:
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            n = normalize_name(entry)
            if target in n or n in target:
                return full

    return None


def get_images_from_folder(folder_path: str) -> list[str]:
    if not folder_path or not os.path.isdir(folder_path):
        return []
    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_nonaug_images_from_folder(folder_path: str) -> list[str]:
    return [
        p for p in get_images_from_folder(folder_path)
        if not os.path.basename(p).lower().startswith("aug")
    ]


def safe_open_image(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedder():
    import features.dino_embedder as _de
    _de.get_model()
    return _de


def get_query_embedding(embedder_mod, img_path: str) -> np.ndarray:
    vec = embedder_mod.embed_image(img_path)
    if vec is None:
        raise ValueError(f"Could not embed query image: {img_path}")
    return vec


# ══════════════════════════════════════════════════════════════════════════════
# RGB HISTOGRAM SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def _rgb_hist(img: Image.Image, bins: int = 32) -> np.ndarray:
    arr = np.array(img.resize((64, 64)).convert("RGB"), dtype=np.float32)
    parts = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256))
        parts.append(h.astype(np.float32))
    vec = np.concatenate(parts)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def compute_rgb_scores(
    query_img: Image.Image,
    candidate_paths: list[str],
    progress_placeholder=None,
    progress_label: str = "RGB scoring…",
) -> list[tuple[str, float]]:
    q_hist  = _rgb_hist(query_img)
    results = []
    total   = len(candidate_paths)

    for idx, p in enumerate(candidate_paths):
        img = safe_open_image(p)
        score = float(np.dot(q_hist, _rgb_hist(img))) if img is not None else 0.0
        results.append((p, score))

        if progress_placeholder is not None and idx % max(1, total // 50) == 0:
            progress_placeholder.progress(
                (idx + 1) / total,
                text=f"{progress_label} {idx + 1}/{total}",
            )

    if progress_placeholder is not None:
        progress_placeholder.empty()

    return sorted(results, key=lambda x: x[1], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE GRID RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_image_grid(
    items: list[tuple[str, ...]],
    n_cols: int,
    label_fn,
):
    cols = st.columns(n_cols)
    for i, item in enumerate(items):
        path = item[0]
        col  = cols[i % n_cols]
        img  = safe_open_image(path)
        if img is not None:
            col.image(img, use_container_width=True)
        else:
            fname = os.path.basename(path)
            col.markdown(
                f'<div class="not-found-box">'
                f'<span style="font-size:1.4rem">🚫</span>'
                f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;color:#c06060;">'
                f'Not found</span>'
                f'<span style="font-family:DM Mono,monospace;font-size:0.56rem;color:#555;'
                f'word-break:break-all">{fname}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        col.markdown(
            f'<div class="img-label">{label_fn(item)}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="main-title">🪨 StoneX</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Stone family visual search</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline")
    st.markdown(
        '<div class="pipeline-flow">'
        '<span class="layer-badge badge-model">🧠 Model</span> → '
        '<span class="layer-badge badge-rgb">🎨 RGB Filter</span> → '
        '<span class="layer-badge badge-embedding">🔬 Embedding Rerank</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🎛️ Model controls")
    top_k_families    = st.slider("Top-K families",            1,  30, 10, 1)
    top_k_images      = st.slider("Top-K images (model layer)", 4,  60, 12, 4)
    first_layer_fetch = st.slider("First-layer candidate pool", 10, 100, 60, 5)

    st.markdown("---")
    st.markdown("### 🎨 RGB filter (All Images tab)")
    rgb_keep = st.slider(
        "Keep top-N after RGB filter", 5, 50, 20, 1,
        help="From the top-1 family folder, keep only this many closest-colour images.",
    )

    st.markdown("---")
    st.markdown("### 🔬 Embedding rerank (All Images tab)")
    final_top_k = st.slider(
        "Final top-K to display", 5, 20, 10, 1,
        help="After embedding rerank, show this many best-matching images.",
    )

    st.markdown("---")
    st.markdown("### 🌿 Non-Aug tab settings")

    st.markdown(
        '<div style="font-size:0.78rem;color:#666;line-height:1.5;margin-bottom:8px;">'
        '🔒 <b style="color:#7fff6f">Fixed family</b>: If top-1 non-aug count is below '
        'this threshold, ALL top-1 non-aug images are always shown (just re-ranked by embedding).<br>'
        '➕ <b style="color:#ff88ff">Supplement</b> images from top-2 family are added alongside.'
        '</div>',
        unsafe_allow_html=True,
    )

    fixed_family_threshold = st.slider(
        "Fixed family threshold (top-1 always shown if non-aug < N)",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )

    st.markdown(
        '<div style="font-size:0.73rem;color:#555;margin:4px 0 10px 0;">'
        '↳ Top-2 supplement pipeline (when fixed mode is active):'
        '</div>',
        unsafe_allow_html=True,
    )

    nonaug_top2_rgb_keep = st.slider(
        "Top-2 supplement: keep top-N after RGB filter",
        min_value=5, max_value=60, value=20, step=1,
    )
    nonaug_top2_final_top_k = st.slider(
        "Top-2 supplement: final top-K after embedding",
        min_value=1, max_value=20, value=5, step=1,
    )

    st.markdown(
        '<div style="font-size:0.73rem;color:#555;margin:10px 0 4px 0;">'
        '↳ Standard mode (top-1 non-aug ≥ threshold):'
        '</div>',
        unsafe_allow_html=True,
    )

    nonaug_std_rgb_keep = st.slider(
        "Standard: keep top-N after RGB filter",
        min_value=5, max_value=60, value=20, step=1,
    )
    nonaug_std_final_top_k = st.slider(
        "Standard: final top-K after embedding",
        min_value=3, max_value=20, value=10, step=1,
    )

    st.markdown("---")
    st.markdown("### 👁️ Display")
    show_model_families  = st.checkbox("Show model family scores",   value=True)
    show_model_images    = st.checkbox("Show model matched images",  value=True)
    show_rgb_stage       = st.checkbox("Show RGB filter stage",      value=True)
    img_cols             = st.slider("Grid columns", 2, 6, 4, 1)

    st.markdown("---")
    st.markdown("### 🗄️ Ingest dataset")
    dataset_path_input = st.text_input("Dataset folder", value=DATASET_ROOT)
    do_embed_ingest    = st.checkbox("Ingest DINOv2 embeddings", value=True, key="ing_embed")

    if st.button("▶ Run ingestion", use_container_width=True):
        if not dataset_path_input:
            st.error("Please provide a dataset folder path.")
        else:
            from ingestion.ingest_dataset import ingest_dataset
            with st.spinner("Ingesting…"):
                try:
                    ingest_dataset(dataset_path_input, do_color=False, do_embedding=do_embed_ingest)
                    st.success("✅ Ingestion complete!")
                except Exception as exc:
                    st.error(f"Ingestion error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — header + upload
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🪨 StoneX</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    '<span class="layer-badge badge-model">🧠 Model</span>'
    '<span style="color:#444"> → </span>'
    '<span class="layer-badge badge-rgb">🎨 RGB Filter</span>'
    '<span style="color:#444"> → </span>'
    '<span class="layer-badge badge-embedding">🔬 Embedding Rerank</span>'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload stone images (multiple supported)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown("""
    <div style="background:#0f0f0f;border:2px dashed #2a2a2a;border-radius:16px;
                padding:70px 40px;text-align:center;margin-top:28px;">
        <div style="font-size:3.5rem">🪨</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;
                    color:#444;margin-top:14px;">
            Drop one or more stone images to begin search
        </div>
        <div style="font-size:0.82rem;color:#2e2e2e;margin-top:8px;">
            Supports JPG · PNG · WEBP &nbsp;·&nbsp; Multiple files allowed
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-IMAGE OVERVIEW (shown when >1 image uploaded)
# ══════════════════════════════════════════════════════════════════════════════

n_images = len(uploaded_files)

if n_images > 1:
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#141008,#0f0f0f);'
        f'border:1px solid #3a2800;border-radius:12px;padding:14px 20px;'
        f'margin-bottom:20px;display:flex;align-items:center;gap:16px;">'
        f'<span style="font-size:1.8rem">🪨</span>'
        f'<div>'
        f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;'
        f'color:#c9a96e;">{n_images} Images Queued</div>'
        f'<div style="font-size:0.8rem;color:#555;font-family:DM Mono,monospace;">'
        f'Results will be shown below, one per image · expand each section to view details'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )

    # Thumbnail strip
    thumb_cols = st.columns(min(n_images, 8))
    for i, uf in enumerate(uploaded_files):
        with thumb_cols[i % 8]:
            img_thumb = Image.open(uf).convert("RGB")
            uf.seek(0)
            st.image(img_thumb, use_container_width=True)
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.6rem;'
                f'color:#555;text-align:center;margin-top:2px;">#{i+1}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PER-IMAGE PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def process_single_image(uploaded_file, img_index: int, n_total: int):
    """Run the full StoneX pipeline for one uploaded image and render results."""

    # ── Load & save to temp ───────────────────────────────────────────────
    query_image = Image.open(uploaded_file).convert("RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        query_image.save(tmp.name, format="JPEG")
        temp_path = tmp.name

    # ── Image result header ───────────────────────────────────────────────
    filename = getattr(uploaded_file, "name", f"image_{img_index+1}")

    if n_total > 1:
        st.markdown(
            f'<div class="image-result-header">'
            f'<span class="image-result-index">#{img_index + 1}</span>'
            f'<span class="image-result-filename">📄 {filename}</span>'
            f'<span class="image-result-dims">'
            f'{query_image.size[0]} × {query_image.size[1]} px</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Layout: query image | results ─────────────────────────────────────
    col_img, col_results = st.columns([1, 2], gap="large")

    with col_img:
        st.image(query_image, caption=f"Query #{img_index+1}: {filename}", use_container_width=True)
        st.markdown(
            f'<div style="font-size:0.78rem;color:#444;font-family:DM Mono,monospace;">'
            f'{query_image.size[0]} × {query_image.size[1]} px</div>',
            unsafe_allow_html=True,
        )

    with col_results:
        from query.pipeline import run_pipeline

        with st.spinner(f"Running 🧠 Model for image #{img_index+1}…"):
            try:
                results = run_pipeline(
                    temp_path,
                    layer_order=layer_order,
                    top_k_families=top_k_families,
                    top_k_images=top_k_images,
                    first_layer_fetch=first_layer_fetch,
                )
                pipeline_error = None
            except Exception as exc:
                results        = None
                pipeline_error = exc

        if pipeline_error:
            st.error(f"Pipeline error for image #{img_index+1}: {pipeline_error}")
            try: os.remove(temp_path)
            except Exception: pass
            return None

        if not results or not results.get("families"):
            st.warning(f"No results for image #{img_index+1}. Make sure the dataset is ingested.")
            try: os.remove(temp_path)
            except Exception: pass
            return None

        # ── Top-K families ────────────────────────────────────────────
        st.markdown(
            f'<div class="section-header">🏆 Top {top_k_families} Stone Families</div>',
            unsafe_allow_html=True,
        )
        for rank, (raw_fam, score) in enumerate(results["families"], 1):
            bar          = max(0.0, min(1.0, float(score)))
            disp         = display_name(raw_fam)
            cmd_badge    = cmd_badge_html(raw_fam)
            st.markdown(
                f'<div class="family-card">'
                f'<span class="rank-num">#{rank}</span>'
                f'<span class="family-name">{disp} {cmd_badge}</span>'
                f'<span class="score-num">{score:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(bar)

        # ── Top-5 families representative images ──────────────────────
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">'
            '🌟 Top 5 Families — Best Representative Image'
            ' <span class="layer-badge badge-top5" style="vertical-align:middle;">'
            '✨ non-aug</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="info-box-gold">'
            'For each of the top-5 predicted families, the <b style="color:#ffd700">highest-scoring '
            'non-augmented image</b> returned by the model is shown. '
            'If the model returned no non-aug image for a family, the first non-aug file '
            'found in that family\'s folder on disk is used as fallback.'
            '</div>',
            unsafe_allow_html=True,
        )

        top5_families  = results["families"][:5]
        raw_imgs_all   = results["images"].get("model", [])

        top5_cols = st.columns(5)
        for col_idx, (raw_fam, fam_score) in enumerate(top5_families):
            col      = top5_cols[col_idx]
            fam_norm = normalize_name(raw_fam)
            disp     = display_name(raw_fam)

            col.markdown(
                f'<div class="family-col-header">'
                f'#{col_idx + 1}<br>{disp}'
                f'</div>',
                unsafe_allow_html=True,
            )

            best_path, best_score = None, -1.0

            for p, s in raw_imgs_all:
                lp    = remap_path(p)
                parts = str(lp).replace("\\", "/").split("/")
                img_family = normalize_name(parts[-2]) if len(parts) >= 2 else ""
                fname      = parts[-1]
                if img_family == fam_norm and not fname.lower().startswith("aug") and s > best_score:
                    best_path, best_score = lp, s

            if best_path is None:
                fam_folder = find_folder_for_family(raw_fam, LOCAL_PATH_PREFIX)
                if fam_folder is None:
                    fam_folder = find_folder_for_family(raw_fam, DATASET_ROOT)
                if fam_folder:
                    candidates = get_images_from_folder(fam_folder)
                    non_aug    = [
                        p for p in candidates
                        if not os.path.basename(p).lower().startswith("aug")
                    ]
                    if non_aug:
                        best_path  = non_aug[0]
                        best_score = None

            if best_path:
                img   = safe_open_image(best_path)
                fname = os.path.basename(best_path)
                if img:
                    col.image(img, use_container_width=True)
                else:
                    col.markdown(
                        '<div class="not-found-box">'
                        '<span style="font-size:1.2rem">🚫</span>'
                        '<span style="font-size:0.6rem;color:#c06060;">Cannot open</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                if best_score is not None:
                    score_html = f'<span style="color:#ffa86f">score: {best_score:.4f}</span>'
                else:
                    score_html = '<span style="color:#888">📁 folder fallback</span>'
                col.markdown(
                    f'<div class="img-label">📄 {fname}<br>{score_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                col.markdown(
                    '<div class="not-found-box">'
                    '<span style="font-size:1.4rem">🪨</span>'
                    '<span style="font-size:0.65rem;color:#555;">No non-aug image found</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # ── Per-layer family scores ────────────────────────────────────
        if show_model_families:
            st.markdown("---")
            st.markdown(
                '<div class="section-header">📊 Per-layer family scores'
                ' <span class="layer-badge badge-model">model</span></div>',
                unsafe_allow_html=True,
            )
            layer_fams = results["layer_families"].get("model", [])
            if not layer_fams:
                st.caption("No per-layer data available.")
            else:
                for rank, (raw_fam, sc) in enumerate(layer_fams[:top_k_families], 1):
                    bar  = max(0.0, min(1.0, float(sc)))
                    disp = display_name(raw_fam)
                    st.progress(bar, text=f"#{rank}  {disp}  ({sc:.4f})")

    # ── Model matched images ───────────────────────────────────────────────
    if show_model_images:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">🖼️ Model Matched Images'
            ' <span class="layer-badge badge-model">🧠 model layer</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        raw_imgs      = results["images"].get("model", [])
        remapped_imgs = [(remap_path(p), s) for p, s in raw_imgs]

        if not remapped_imgs:
            st.caption("No matched images returned by the model layer.")
        else:
            st.markdown(
                f'<div class="info-box">'
                f'Showing all <b style="color:#ffa86f">{len(remapped_imgs)}</b> images '
                f'returned by the model layer.'
                f'</div>',
                unsafe_allow_html=True,
            )

            def model_label(item):
                path, score = item
                parts      = str(path).replace("\\", "/").split("/")
                raw_fam    = parts[-2] if len(parts) >= 2 else "unknown"
                disp       = display_name(raw_fam)
                badge      = cmd_badge_html(raw_fam)
                fname      = parts[-1]
                return (
                    f'🏷 <b style="color:#ffa86f">{disp}</b> {badge}<br>'
                    f'📄 {fname}<br>'
                    f'<span style="color:#ffa86f">score: {score:.4f}</span>'
                )

            render_image_grid(remapped_imgs, img_cols, model_label)

    # ── Two-stage refinement ───────────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">🎯 Two-Stage Refinement</div>',
        unsafe_allow_html=True,
    )

    families_ranked = results.get("families", [])

    if not families_ranked:
        st.warning("No families available for refinement.")
        try: os.remove(temp_path)
        except Exception: pass
        return results

    top1_raw, top1_score = families_ranked[0]
    top1_disp  = display_name(top1_raw)
    top1_folder = find_folder_for_family(top1_raw, LOCAL_PATH_PREFIX)
    if top1_folder is None:
        top1_folder = find_folder_for_family(top1_raw, DATASET_ROOT)

    if top1_folder is None:
        st.warning(
            f"⚠️  Could not locate a folder matching "
            f"**{top1_disp}**."
        )
        try: os.remove(temp_path)
        except Exception: pass
        return results

    all_candidates = get_images_from_folder(top1_folder)

    tab_nonaug, tab_all = st.tabs([
        "🌿 Non-Aug Only  (default)",
        "📁 All Images  (top-1 family)",
    ])

    # ── TAB 1: Non-Aug Only ────────────────────────────────────────────────
    with tab_nonaug:
        st.markdown(
            '<div class="section-header">'
            '🌿 Non-Augmented Images Only'
            ' <span class="layer-badge badge-nonaug">✨ non-aug pipeline</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        top1_nonaug   = get_nonaug_images_from_folder(top1_folder)
        top1_count    = len(top1_nonaug)
        is_fixed_mode = top1_count < fixed_family_threshold

        if is_fixed_mode:
            st.markdown(
                f'<div class="info-box-teal">'
                f'🔒 <b style="color:#00e5ff">Fixed Family Mode</b> — '
                f'<b style="color:#00e5ff">{top1_disp}</b> '
                f'has only <b style="color:#00e5ff">{top1_count}</b> non-aug images '
                f'(threshold: {fixed_family_threshold}).<br>'
                f'→ All <b style="color:#7fff6f">{top1_count} top-1</b> images shown, '
                f're-ranked by DINOv2.<br>'
                f'→ <b style="color:#ff88ff">Top-2 supplement</b>: RGB → embedding rerank added.'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="info-box">'
                f'<span class="source-chip chip-fixed">🔒 Fixed</span>'
                f' <b style="color:#7fff6f">{top1_disp}</b>'
                f' &nbsp;·&nbsp; <b style="color:#7fff6f">{top1_count}</b> non-aug images<br>'
                f'<span style="font-size:0.74rem;color:#555;">{top1_folder}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="stage-header">'
                '<span class="layer-badge badge-embedding">🔬 Fixed Family</span>'
                f' All {top1_count} top-1 non-aug images — embedding re-ranked'
                '</div>',
                unsafe_allow_html=True,
            )

            fixed_results = []
            if top1_nonaug:
                with st.spinner(f"Embedding {top1_count} fixed top-1 non-aug images…"):
                    try:
                        embedder_mod = load_embedder()
                        query_vec    = get_query_embedding(embedder_mod, temp_path)
                        valid_fixed, vecs_fixed = embedder_mod.embed_batch(top1_nonaug)
                        skipped_fixed = top1_count - len(valid_fixed)
                        if skipped_fixed > 0:
                            st.caption(f"⚠️  {skipped_fixed} image(s) skipped.")
                        if valid_fixed:
                            emb_scores_fixed = (vecs_fixed @ query_vec).tolist()
                            fixed_results = sorted(
                                zip(valid_fixed, emb_scores_fixed),
                                key=lambda x: x[1], reverse=True,
                            )
                            st.markdown(
                                f'<div class="info-box-green">'
                                f'🔒 Showing all <b style="color:#7fff6f">{len(fixed_results)}</b> '
                                f'top-1 non-aug images, sorted by DINOv2 cosine similarity.'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            _top1_disp_copy = top1_disp
                            _top1_raw_copy  = top1_raw
                            def fixed_label(item, _d=_top1_disp_copy, _r=_top1_raw_copy):
                                path, emb_score = item
                                fname = os.path.basename(path)
                                badge = cmd_badge_html(_r)
                                return (
                                    f'<span class="source-chip chip-fixed">🔒 Fixed</span> '
                                    f'<b style="color:#7fff6f">{_d}</b> {badge}<br>'
                                    f'📄 {fname}<br>'
                                    f'<span style="color:#6fbfff">emb: {emb_score:.4f}</span>'
                                )
                            render_image_grid(fixed_results, img_cols, fixed_label)
                        else:
                            st.warning("No valid embeddings for top-1 non-aug images.")
                    except Exception as exc:
                        st.error(f"Embedding error (fixed family): {exc}")
                        st.exception(exc)
            else:
                st.info("No non-aug images found in the top-1 family folder.")

            # Top-2 supplement
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown(
                '<div class="stage-header">'
                '<span class="layer-badge badge-supplement">➕ Supplement</span>'
                ' Top-2 family — RGB filter → Embedding rerank'
                '</div>',
                unsafe_allow_html=True,
            )

            if len(families_ranked) < 2:
                st.info("No top-2 family available for supplement.")
            else:
                top2_raw, _  = families_ranked[1]
                top2_disp    = display_name(top2_raw)
                top2_folder  = find_folder_for_family(top2_raw, LOCAL_PATH_PREFIX)
                if top2_folder is None:
                    top2_folder = find_folder_for_family(top2_raw, DATASET_ROOT)

                if top2_folder is None:
                    st.warning(f"⚠️  Could not locate folder for top-2 family **{top2_disp}**.")
                else:
                    top2_nonaug = get_nonaug_images_from_folder(top2_folder)
                    top2_count  = len(top2_nonaug)

                    st.markdown(
                        f'<div class="info-box">'
                        f'<span class="source-chip chip-supplement">➕ Supplement</span>'
                        f' <b style="color:#ff88ff">{top2_disp}</b>'
                        f' &nbsp;·&nbsp; <b style="color:#ff88ff">{top2_count}</b> non-aug images'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if not top2_nonaug:
                        st.info("No non-aug images found in the top-2 family folder.")
                    else:
                        if show_rgb_stage:
                            st.markdown(
                                f'<div class="stage-header">'
                                f'<span class="layer-badge badge-rgb">🎨 Stage 1</span>'
                                f' RGB Colour Filter — keeping top {nonaug_top2_rgb_keep} of {top2_count}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        rgb_prog_t2 = st.progress(0.0, text="RGB scoring top-2 non-aug images…")
                        with st.spinner(""):
                            rgb_ranked_t2 = compute_rgb_scores(
                                query_image, top2_nonaug, rgb_prog_t2, "RGB scoring top-2 non-aug…"
                            )
                        rgb_top_t2 = rgb_ranked_t2[:nonaug_top2_rgb_keep]

                        if show_rgb_stage:
                            st.markdown(
                                f'<div class="info-box-green">'
                                f'✅ Kept <b style="color:#7fff6f">{len(rgb_top_t2)}</b> '
                                f'closest-colour top-2 non-aug images'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            _top2_disp_copy = top2_disp
                            _top2_raw_copy  = top2_raw
                            def rgb_label_t2(item, _d=_top2_disp_copy, _r=_top2_raw_copy):
                                path, score = item
                                fname = os.path.basename(path)
                                badge = cmd_badge_html(_r)
                                return (
                                    f'<span class="source-chip chip-supplement">➕</span> '
                                    f'<b style="color:#ff88ff">{_d}</b> {badge}<br>'
                                    f'📄 {fname}<br>'
                                    f'<span style="color:#7fff6f">rgb: {score:.4f}</span>'
                                )
                            render_image_grid(rgb_top_t2, img_cols, rgb_label_t2)

                        st.markdown(
                            f'<div class="stage-header">'
                            f'<span class="layer-badge badge-embedding">🔬 Stage 2</span>'
                            f' Embedding Rerank — top {nonaug_top2_final_top_k} of {len(rgb_top_t2)}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        rgb_survivor_t2 = [p for p, _ in rgb_top_t2]
                        rgb_lookup_t2   = {p: s for p, s in rgb_top_t2}
                        with st.spinner(f"Embedding {len(rgb_survivor_t2)} top-2 RGB-filtered images…"):
                            try:
                                embedder_mod = load_embedder()
                                query_vec    = get_query_embedding(embedder_mod, temp_path)
                                valid_t2, vecs_t2 = embedder_mod.embed_batch(rgb_survivor_t2)
                                if valid_t2:
                                    emb_scores_t2 = (vecs_t2 @ query_vec).tolist()
                                    emb_ranked_t2 = sorted(
                                        zip(valid_t2, emb_scores_t2),
                                        key=lambda x: x[1], reverse=True,
                                    )
                                    final_t2 = emb_ranked_t2[:nonaug_top2_final_top_k]
                                    st.markdown(
                                        f'<div class="info-box-blue">'
                                        f'🔬 Showing top <b style="color:#6fbfff">{len(final_t2)}</b> supplement images'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                                    _top2_disp_copy2 = top2_disp
                                    _top2_raw_copy2  = top2_raw
                                    def emb_label_t2(item, _d=_top2_disp_copy2, _r=_top2_raw_copy2, _lu=rgb_lookup_t2):
                                        path, emb_score = item
                                        fname = os.path.basename(path)
                                        r_s   = _lu.get(path, 0.0)
                                        badge = cmd_badge_html(_r)
                                        return (
                                            f'<span class="source-chip chip-supplement">➕</span> '
                                            f'<b style="color:#ff88ff">{_d}</b> {badge}<br>'
                                            f'📄 {fname}<br>'
                                            f'<span style="color:#6fbfff">emb: {emb_score:.4f}</span>'
                                            f'&nbsp;&nbsp;<span style="color:#7fff6f">rgb: {r_s:.4f}</span>'
                                        )
                                    render_image_grid(final_t2, img_cols, emb_label_t2)
                                else:
                                    st.warning("No valid embeddings for top-2 RGB-filtered images.")
                            except Exception as exc:
                                st.error(f"Embedding rerank error (top-2 supplement): {exc}")
                                st.exception(exc)
        else:
            # Standard mode
            st.markdown(
                f'<div class="info-box-purple">'
                f'📊 <b style="color:#c87fff">Standard Mode</b> — '
                f'<b style="color:#c87fff">{top1_disp}</b> '
                f'has <b style="color:#c87fff">{top1_count}</b> non-aug images '
                f'(≥ threshold of {fixed_family_threshold}).<br>'
                f'Pipeline: RGB colour filter (keep {nonaug_std_rgb_keep}) → '
                f'DINOv2 embedding rerank (show top {nonaug_std_final_top_k}).'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="info-box">'
                f'<span class="source-chip chip-top1">Top-1</span>'
                f' <b style="color:#c87fff">{top1_disp}</b>'
                f' &nbsp;·&nbsp; <b style="color:#c87fff">{top1_count}</b> non-aug images<br>'
                f'<span style="font-size:0.74rem;color:#555;">{top1_folder}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if show_rgb_stage:
                st.markdown(
                    f'<div class="stage-header">'
                    f'<span class="layer-badge badge-rgb">🎨 Stage 1</span>'
                    f' RGB Colour Filter — keeping top {nonaug_std_rgb_keep} of {top1_count}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            rgb_prog_std = st.progress(0.0, text="RGB scoring non-aug images…")
            with st.spinner(""):
                rgb_ranked_std = compute_rgb_scores(
                    query_image, top1_nonaug, rgb_prog_std, "RGB scoring non-aug…"
                )
            rgb_top_std = rgb_ranked_std[:nonaug_std_rgb_keep]

            if show_rgb_stage:
                st.markdown(
                    f'<div class="info-box-green">'
                    f'✅ Kept <b style="color:#7fff6f">{len(rgb_top_std)}</b> '
                    f'closest-colour non-aug images · '
                    f'excluded <b style="color:#ff6f6f">{top1_count - len(rgb_top_std)}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _top1_std_disp = top1_disp
                _top1_std_raw  = top1_raw
                def nonaug_rgb_label_std(item, _d=_top1_std_disp, _r=_top1_std_raw):
                    path, score = item
                    fname = os.path.basename(path)
                    badge = cmd_badge_html(_r)
                    return (
                        f'<span class="source-chip chip-top1">Top-1</span> '
                        f'<b style="color:#c87fff">{_d}</b> {badge}<br>'
                        f'📄 {fname}<br>'
                        f'<span style="color:#7fff6f">rgb: {score:.4f}</span>'
                    )
                render_image_grid(rgb_top_std, img_cols, nonaug_rgb_label_std)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="stage-header">'
                f'<span class="layer-badge badge-embedding">🔬 Stage 2</span>'
                f' Embedding Rerank — top {nonaug_std_final_top_k} of {len(rgb_top_std)} RGB survivors'
                f'</div>',
                unsafe_allow_html=True,
            )

            rgb_survivor_std = [p for p, _ in rgb_top_std]
            rgb_lookup_std   = {p: s for p, s in rgb_top_std}

            with st.spinner(f"Embedding {len(rgb_survivor_std)} non-aug RGB-filtered images…"):
                try:
                    embedder_mod = load_embedder()
                    query_vec    = get_query_embedding(embedder_mod, temp_path)
                    valid_std, vecs_std = embedder_mod.embed_batch(rgb_survivor_std)
                    skipped_std = len(rgb_survivor_std) - len(valid_std)
                    if skipped_std > 0:
                        st.caption(f"⚠️  {skipped_std} image(s) skipped.")
                    if not valid_std:
                        st.warning("No valid embeddings computed for non-aug RGB-filtered images.")
                    else:
                        emb_scores_std = (vecs_std @ query_vec).tolist()
                        emb_ranked_std = sorted(
                            zip(valid_std, emb_scores_std),
                            key=lambda x: x[1], reverse=True,
                        )
                        final_std = emb_ranked_std[:nonaug_std_final_top_k]
                        st.markdown(
                            f'<div class="info-box-blue">'
                            f'🔬 Re-ranked <b style="color:#6fbfff">{len(valid_std)}</b> images · '
                            f'showing top <b style="color:#6fbfff">{len(final_std)}</b>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        _top1_std_disp2 = top1_disp
                        _top1_std_raw2  = top1_raw
                        def nonaug_emb_label_std(item, _d=_top1_std_disp2, _r=_top1_std_raw2, _lu=rgb_lookup_std):
                            path, emb_score = item
                            fname = os.path.basename(path)
                            r_s   = _lu.get(path, 0.0)
                            badge = cmd_badge_html(_r)
                            return (
                                f'<span class="source-chip chip-top1">Top-1</span> '
                                f'<b style="color:#c87fff">{_d}</b> {badge}<br>'
                                f'📄 {fname}<br>'
                                f'<span style="color:#6fbfff">emb: {emb_score:.4f}</span>'
                                f'&nbsp;&nbsp;<span style="color:#7fff6f">rgb: {r_s:.4f}</span>'
                            )
                        render_image_grid(final_std, img_cols, nonaug_emb_label_std)
                except Exception as exc:
                    st.error(f"Embedding rerank error (standard non-aug): {exc}")
                    st.exception(exc)

    # ── TAB 2: All Images ─────────────────────────────────────────────────
    with tab_all:
        st.markdown(
            f'<div class="info-box">'
            f'📁 <b style="color:#c9a96e">{top1_disp}</b>'
            f' &nbsp;·&nbsp; {len(all_candidates)} images in folder<br>'
            f'<span style="font-size:0.76rem;color:#555;">{top1_folder}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not all_candidates:
            st.warning("No images found in the top-1 family folder.")
        else:
            if show_rgb_stage:
                st.markdown(
                    '<div class="stage-header">'
                    '<span class="layer-badge badge-rgb">🎨 Stage 1</span>'
                    f' RGB Colour Filter — keeping top {rgb_keep} of {len(all_candidates)}'
                    '</div>',
                    unsafe_allow_html=True,
                )
            rgb_prog_all = st.progress(0.0, text="RGB scoring…")
            with st.spinner(""):
                rgb_ranked_all = compute_rgb_scores(
                    query_image, all_candidates, rgb_prog_all, "RGB scoring…"
                )
            rgb_top_all = rgb_ranked_all[:rgb_keep]

            if show_rgb_stage:
                st.markdown(
                    f'<div class="info-box-green">'
                    f'✅ Kept <b style="color:#7fff6f">{rgb_keep}</b> closest-colour images · '
                    f'excluded <b style="color:#ff6f6f">{len(all_candidates) - rgb_keep}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                def rgb_label_all(item):
                    path, score = item
                    fname = os.path.basename(path)
                    return (
                        f'📄 {fname}<br>'
                        f'<span style="color:#7fff6f">rgb: {score:.4f}</span>'
                    )
                render_image_grid(rgb_top_all, img_cols, rgb_label_all)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="stage-header">'
                f'<span class="layer-badge badge-embedding">🔬 Stage 2</span>'
                f' Embedding Rerank — top {final_top_k} of {rgb_keep} RGB survivors'
                f'</div>',
                unsafe_allow_html=True,
            )
            rgb_survivor_paths_all = [p for p, _ in rgb_top_all]
            rgb_lookup_all         = {p: s for p, s in rgb_top_all}

            with st.spinner(f"Embedding {len(rgb_survivor_paths_all)} RGB-filtered images…"):
                try:
                    embedder_mod = load_embedder()
                    query_vec    = get_query_embedding(embedder_mod, temp_path)
                    valid_paths_all, db_vecs_all = embedder_mod.embed_batch(rgb_survivor_paths_all)
                    skipped_all = len(rgb_survivor_paths_all) - len(valid_paths_all)
                    if skipped_all > 0:
                        st.caption(f"⚠️  {skipped_all} image(s) skipped.")
                    if not valid_paths_all:
                        st.warning("No valid embeddings computed for RGB-filtered images.")
                    else:
                        emb_scores_all = (db_vecs_all @ query_vec).tolist()
                        emb_ranked_all = sorted(
                            zip(valid_paths_all, emb_scores_all),
                            key=lambda x: x[1], reverse=True,
                        )
                        final_results_all = emb_ranked_all[:final_top_k]
                        st.markdown(
                            f'<div class="info-box-blue">'
                            f'🔬 Re-ranked <b style="color:#6fbfff">{len(valid_paths_all)}</b> images · '
                            f'showing top <b style="color:#6fbfff">{len(final_results_all)}</b>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        def emb_label_all(item, _lu=rgb_lookup_all):
                            path, emb_score = item
                            fname = os.path.basename(path)
                            r_s   = _lu.get(path, 0.0)
                            return (
                                f'📄 {fname}<br>'
                                f'<span style="color:#6fbfff">emb: {emb_score:.4f}</span>'
                                f'&nbsp;&nbsp;<span style="color:#7fff6f">rgb: {r_s:.4f}</span>'
                            )
                        render_image_grid(final_results_all, img_cols, emb_label_all)
                except Exception as exc:
                    st.error(f"Embedding rerank error: {exc}")
                    st.exception(exc)

    # ── Cleanup ───────────────────────────────────────────────────────────
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return results


# ══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE FOR EACH UPLOADED IMAGE
# ══════════════════════════════════════════════════════════════════════════════

all_results = {}

for img_idx, uploaded_file in enumerate(uploaded_files):
    filename = getattr(uploaded_file, "name", f"image_{img_idx+1}")

    if n_images > 1:
        label = f"Image #{img_idx+1} — {filename}"
        with st.expander(label, expanded=(img_idx == 0)):
            result = process_single_image(uploaded_file, img_idx, n_images)
            if result:
                all_results[filename] = result
    else:
        result = process_single_image(uploaded_file, img_idx, n_images)
        if result:
            all_results[filename] = result

# ── Multi-image summary footer ─────────────────────────────────────────────
if n_images > 1 and all_results:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">📋 Batch Summary</div>',
        unsafe_allow_html=True,
    )

    summary_data = []
    for fname, res in all_results.items():
        families = res.get("families", [])
        # Use display_name for summary table too
        top1_raw = families[0][0] if families else "—"
        top1_sc  = families[0][1] if families else 0.0
        top2_raw = families[1][0] if len(families) > 1 else "—"
        top2_sc  = families[1][1] if len(families) > 1 else 0.0

        summary_data.append({
            "File":          fname,
            "Top-1 Family":  display_name(top1_raw) if top1_raw != "—" else "—",
            "Score":         f"{top1_sc:.4f}",
            "Top-2 Family":  display_name(top2_raw) if top2_raw != "—" else "—",
            "Score (2)":     f"{top2_sc:.4f}",
        })

    import pandas as pd
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)