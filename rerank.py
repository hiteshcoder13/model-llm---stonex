"""
Standalone Streamlit app for StoneX Visual Rerank Pipeline.
Uses local model (DINOv2) for prediction and Gemini for visual reranking.
No external API server required.
"""

import json
import os
import tempfile
import asyncio
import re
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Local imports – same as FastAPI uses
from query.pipeline import run_pipeline
import features.dino_embedder as dino_embedder
from cmd_mapping import resolve_family_name, is_cmd_class
from stone_reranker import (
    rerank_stone_families, 
    set_gemini_api_key, 
    get_gemini_api_key
)

load_dotenv()

st.set_page_config(page_title="StoneX Rerank Demo", layout="wide")
st.title("🪨 StoneX Visual Rerank Pipeline (Standalone)")
st.markdown(
    "Upload a slab image → model predicts top‑5 families → "
    "Gemini re‑ranks them visually. You can edit candidates before reranking."
)

# ------------------------------------------------------------------
# Cached model loading
# ------------------------------------------------------------------
@st.cache_resource
def load_dino_model():
    """Load DINOv2 model once and cache it."""
    dino_embedder.get_model()
    return True

with st.spinner("Loading DINOv2 model..."):
    load_dino_model()
st.success("✅ DINOv2 model loaded")

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def normalize_stone_family(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"^tile[_\s]*", "", name)
    name = name.replace("_", " ")
    name = re.sub(r"\b(all\s*)?variations?\b", "", name)
    name = re.sub(r"\d+", "", name)
    return " ".join(name.split()).upper()

def get_clean_family_name(raw_family: str) -> str:
    if is_cmd_class(raw_family):
        return resolve_family_name(raw_family).upper()
    else:
        return normalize_stone_family(raw_family)

def predict_families(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Run DINOv2 prediction and return top-5 families with scores."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        results = run_pipeline(
            temp_path,
            layer_order=["model"],
            top_k_families=5,
            top_k_images=800,
            first_layer_fetch=100,
        )

        families = results.get("families", [])[:5]
        output = []
        for raw_family, fam_score in families:
            clean_family = get_clean_family_name(raw_family)
            if clean_family:
                output.append({
                    "family": clean_family,
                    "family_score": round(float(fam_score), 4)
                })
        return output

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

async def run_rerank(image_bytes: bytes, mime_type: str, candidates: list) -> Dict[str, Any]:
    """Call Gemini reranker."""
    try:
        gemini_candidates = [{"family_name": c["family_name"]} for c in candidates]
        result = await rerank_stone_families(
            image_bytes=image_bytes,
            image_mime=mime_type,
            candidates=gemini_candidates,
        )
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def call_rerank_sync(image_bytes: bytes, mime_type: str, candidates: list) -> Dict[str, Any]:
    return asyncio.run(run_rerank(image_bytes, mime_type, candidates))

# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
def init_session_state():
    defaults = {
        "predictions": None,
        "candidates_json": "",
        "rerank_result": None,
        "last_uploaded_filename": None,
        "gemini_api_key": "",
        "api_key_validated": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ------------------------------------------------------------------
# Callback when new file uploaded: clear old results
# ------------------------------------------------------------------
def on_file_upload():
    if st.session_state.get("image_uploader") is not None:
        new_filename = st.session_state.image_uploader.name
        if st.session_state.last_uploaded_filename != new_filename:
            st.session_state.predictions = None
            st.session_state.candidates_json = ""
            st.session_state.rerank_result = None
            st.session_state.last_uploaded_filename = new_filename

# ------------------------------------------------------------------
# API Key validation
# ------------------------------------------------------------------
def validate_api_key(api_key: str) -> bool:
    """Quick validation of API key format."""
    if not api_key or not api_key.strip():
        return False
    # Basic format check: Gemini API keys are typically ~39 chars and start with "AIza"
    api_key = api_key.strip()
    if not api_key.startswith("AIza"):
        st.warning("⚠️ Gemini API keys typically start with 'AIza'")
    if len(api_key) < 30:
        st.warning("⚠️ API key seems too short")
    return True

# ------------------------------------------------------------------
# Sidebar for API Key input
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Gemini API Configuration")
    st.markdown("""
    To use the Gemini reranking feature, you need a **Gemini API key**.
    
    [Get a free API key from Google AI Studio](https://aistudio.google.com/apikey)
    """)
    
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter your API key (starts with AIza...)",
        value=st.session_state.gemini_api_key,
        key="api_key_input",
        help="Your API key is stored only in this session and never saved to disk."
    )
    
    if st.button("Set API Key", type="primary", use_container_width=True):
        if validate_api_key(api_key_input):
            st.session_state.gemini_api_key = api_key_input.strip()
            set_gemini_api_key(api_key_input.strip())
            st.session_state.api_key_validated = True
            st.success("✅ API key set successfully!")
        else:
            st.error("Please enter a valid API key")
    
    if st.session_state.api_key_validated:
        st.success("🔓 API key is set and ready")
        
        # Option to clear the key
        if st.button("Clear API Key", use_container_width=True):
            st.session_state.gemini_api_key = ""
            set_gemini_api_key("")
            st.session_state.api_key_validated = False
            st.session_state.rerank_result = None
            st.warning("API key cleared")
            st.rerun()
    else:
        st.warning("⚠️ API key not set. Gemini reranking will not work.")
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.markdown("""
    - The API key is **never saved** to disk
    - It's stored only in your current browser session
    - Free tier includes 60 requests per minute
    - Reranking uses ~1-2 requests per image
    """)

# ------------------------------------------------------------------
# UI Layout
# ------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📤 Step 1: Upload Slab Image")
    uploaded_file = st.file_uploader(
        "Choose a slab image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, PNG, WEBP",
        key="image_uploader",
        on_change=on_file_upload,
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded slab", use_container_width=True)

        # Predict button
        if st.button("🔮 Predict Top‑5 Families", type="primary", use_container_width=True):
            with st.spinner("Running prediction model (DINOv2)..."):
                preds = predict_families(uploaded_file.getvalue())
                if preds:
                    st.session_state.predictions = preds
                    # Auto‑populate candidate JSON with fresh predictions
                    candidates = [{"family_name": p["family"]} for p in preds]
                    st.session_state.candidates_json = json.dumps(candidates, indent=2)
                    st.success(f"✅ Got {len(preds)} predictions")
                else:
                    st.session_state.predictions = None

        # Show predictions if available
        if st.session_state.predictions:
            st.subheader("📋 Step 2: Predicted Families")
            st.markdown("**Top 5 (DINOv2 model)**")
            pred_df = []
            for p in st.session_state.predictions:
                pred_df.append({"Family": p["family"], "Score": p["family_score"]})
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            st.subheader("✏️ Step 3: Edit Candidates (optional)")
            st.caption("Modify the JSON array if needed. Only 'family_name' is required.")
            edited_json = st.text_area(
                "Candidate families (JSON)",
                value=st.session_state.candidates_json,
                height=200,
                key="candidates_editor",
            )
            # Keep session state in sync with editor
            st.session_state.candidates_json = edited_json

            # Rerank button - show warning if API key not set
            if not st.session_state.api_key_validated:
                st.warning("⚠️ Please set your Gemini API key in the sidebar before reranking.")
            
            if st.button("🚀 Run Gemini Rerank", type="primary", use_container_width=True, disabled=not st.session_state.api_key_validated):
                try:
                    candidates = json.loads(edited_json)
                    if not isinstance(candidates, list) or not all(
                        isinstance(c, dict) and "family_name" in c for c in candidates
                    ):
                        st.error("Invalid format. Must be a JSON array of objects with 'family_name'.")
                    else:
                        mime = uploaded_file.type or ""
                        if mime not in {"image/jpeg", "image/png", "image/webp"}:
                            ext = (uploaded_file.name or "").rsplit(".", 1)[-1].lower()
                            mime = {
                                "jpg": "image/jpeg",
                                "jpeg": "image/jpeg",
                                "png": "image/png",
                                "webp": "image/webp",
                            }.get(ext, "image/jpeg")

                        with st.spinner("Calling Gemini Rerank... (may take 10-30 seconds)"):
                            result = call_rerank_sync(
                                uploaded_file.getvalue(),
                                mime,
                                candidates,
                            )
                            if result.get("status") == "success":
                                st.session_state.rerank_result = result
                                st.success("✅ Rerank completed")
                            else:
                                st.error(f"Rerank error: {result.get('message')}")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

with col_right:
    st.subheader("📊 Rerank Results")

    if st.session_state.rerank_result:
        result = st.session_state.rerank_result

        if summary := result.get("image_summary"):
            st.markdown("### 📝 Gemini's Image Summary")
            st.info(summary)

        ranked = result.get("ranked_families", [])
        if ranked:
            st.markdown("### 🏆 Re‑ranked Families")
            table_data = []
            for item in ranked:
                table_data.append({
                    "Rank": item.get("rank", "?"),
                    "Family": item.get("family_name", ""),
                    "Confidence": (
                        f"{item['confidence']:.3f}" if item.get("confidence") is not None else "N/A"
                    ),
                    "Match Reason": item.get("match_reason", ""),
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            with st.expander("🔍 Detailed Explanations", expanded=False):
                for item in ranked:
                    st.markdown(f"**{item.get('family_name')}** (Rank {item.get('rank')})")
                    st.markdown(f"*Match reason:* {item.get('match_reason', 'N/A')}")
                    if mismatch := item.get("mismatch_notes"):
                        st.markdown(f"*Mismatch notes:* {mismatch}")
                    st.divider()

        col_hits, col_misses = st.columns(2)
        with col_hits:
            st.metric("✅ KB Hits", len(result.get("kb_hits", [])))
            if hits := result.get("kb_hits"):
                st.write(", ".join(hits))
        with col_misses:
            st.metric("❌ KB Misses", len(result.get("kb_misses", [])))
            if misses := result.get("kb_misses"):
                st.write(", ".join(misses))

        with st.expander("📄 Raw Response", expanded=False):
            st.json(result)

    else:
        if not st.session_state.api_key_validated:
            st.info("🔑 **First: Set your Gemini API key in the sidebar** →\n\nThen upload an image, click **Predict Top‑5 Families**, then **Run Gemini Rerank** to see results.")
        else:
            st.info("Upload an image, click **Predict Top‑5 Families**, then **Run Gemini Rerank** to see results.")
