"""
stone_reranker.py
-----------------
Gemini-powered visual reranking for stone families using discriminative visual keys.

Model: gemini-2.0-flash-exp

Improvements over previous version:
- Candidate block now highlights only the most differentiating visual attributes
- Emphasizes pattern type, directional features, and unique markers
- Instructs Gemini to first describe the slab's visual pattern, then match
- Reduces false matches (e.g., Carrara vs Bianco Vogue) by focusing on banding
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

logger = logging.getLogger(__name__)

# ── Gemini configuration ───────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Global variable to store the API key (set by UI)
_GEMINI_API_KEY: str | None = None


def set_gemini_api_key(api_key: str) -> None:
    """Set the Gemini API key to be used for reranking."""
    global _GEMINI_API_KEY
    _GEMINI_API_KEY = api_key
    logger.info("Gemini API key set")


def get_gemini_api_key() -> str | None:
    """Get the current Gemini API key."""
    return _GEMINI_API_KEY


def clear_gemini_api_key() -> None:
    """Clear the stored Gemini API key."""
    global _GEMINI_API_KEY
    _GEMINI_API_KEY = None
    logger.info("Gemini API key cleared")


def _create_client() -> genai.Client:
    """Create a new Gemini client with the stored API key."""
    if _GEMINI_API_KEY is None:
        raise RuntimeError("Gemini API key not set. Please provide an API key.")
    return genai.Client(api_key=_GEMINI_API_KEY)


# ── Stone knowledge base ───────────────────────────────────────────────────────
def _get_kb_path() -> Path:
    """Get the path to stones_db.json, handling both local and Streamlit Cloud deployment."""
    # Try to find the file in the current directory
    current_dir = Path(__file__).parent
    kb_path = current_dir / "stones_db.json"
    
    if kb_path.exists():
        return kb_path
    
    # Try in the app's root directory (for Streamlit Cloud)
    try:
        import streamlit.web.bootstrap as bootstrap
        app_root = Path(bootstrap.__file__).parent.parent
        kb_path = app_root / "stones_db.json"
        
        if kb_path.exists():
            return kb_path
    except ImportError:
        pass
    
    # If not found, return the original path (will raise error later)
    return current_dir / "stones_db.json"

_STONES_DB: dict[str, dict] | None = None

def _load_kb() -> dict[str, dict]:
    global _STONES_DB
    if _STONES_DB is None:
        kb_path = _get_kb_path()
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Stone knowledge base not found at {kb_path}. "
                "Place stones_db.json in the same directory as stone_reranker.py "
                "or in the app root directory."
            )
        with open(kb_path, encoding="utf-8") as fh:
            _STONES_DB = json.load(fh)
        logger.info("Loaded %d stones from knowledge base.", len(_STONES_DB))
    return _STONES_DB


def get_stone_profile(family_name: str) -> dict | None:
    """Return the stone descriptor for `family_name` (case-insensitive lookup)."""
    db = _load_kb()
    key = family_name.strip().upper()
    return db.get(key)


def _extract_discriminators(profile: dict) -> dict:
    """
    Extract only the most visually distinguishing features from a stone profile.
    Returns a dict with keys used in the candidate block.
    """
    color = profile.get("color", {})
    surface = profile.get("surface", {})
    pattern = profile.get("pattern", {})

    # Base color and tone
    primary_color = color.get("primary", "unknown")
    tone = color.get("tone", "unknown")  # warm/cool/neutral

    # Pattern type (cloudy, veined, banded, crackle, speckled, breccia, etc.)
    pattern_type = pattern.get("type", "unknown")

    # Vein presence and direction
    veins_present = pattern.get("veins_present", False)
    grain_dir = surface.get("grain_direction", "non-directional")

    # Unique structural marker (extract from differentiator or visual_description)
    differentiator = profile.get("differentiator", "")
    visual_desc = profile.get("visual_description", "")

    # Short unique marker (first sentence of differentiator, or key phrase)
    unique_marker = ""
    if differentiator:
        # Take first sentence, max 100 chars
        unique_marker = differentiator.split(".")[0][:100]
    elif visual_desc:
        unique_marker = visual_desc.split(".")[0][:100]

    return {
        "primary_color": primary_color,
        "tone": tone,
        "pattern_type": pattern_type,
        "veins_present": veins_present,
        "grain_direction": grain_dir,
        "unique_marker": unique_marker,
    }


def build_candidate_block(candidates: list[dict]) -> str:
    """Build a concise, discriminative summary for each candidate."""
    lines: list[str] = []
    for idx, cand in enumerate(candidates, start=1):
        name = cand["family_name"]
        profile = get_stone_profile(name)

        lines.append("─" * 60)
        lines.append(f"Candidate {idx}: {name}")
        lines.append("─" * 60)

        if profile:
            disc = _extract_discriminators(profile)

            lines.append(f"• Base color: {disc['primary_color']} ({disc['tone']} tone)")
            lines.append(f"• Pattern type: {disc['pattern_type']}")
            lines.append(f"• Veins present: {disc['veins_present']}")
            lines.append(f"• Grain direction: {disc['grain_direction']}")
            if disc['unique_marker']:
                lines.append(f"• Key differentiator: {disc['unique_marker']}")
        else:
            lines.append("(No knowledge-base descriptor found — rank on visual alone)")

        lines.append("")
    return "\n".join(lines)


# ── Prompt templates (improved) ───────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a stone identification assistant specialized in visually discriminating natural stone slabs.

You will be given an image of a stone slab and a short list of candidate stone families with their **visual discriminators**.

Your task:
1. Carefully observe the slab image focusing on pattern structure, vein direction, and unique visual features.
2. Compare the image ONLY against the given candidate families and their discriminators.
3. Rank all candidates from best match (rank 1) to worst match (rank N) based strictly on visual similarity.

Important rules:
- DO NOT describe or summarize the image.
- DO NOT generate any explanation outside ranking logic.
- Focus only on visual matching between the image and candidate characteristics.
- Pay special attention to directional features and pattern structure.

Return ONLY valid JSON — no markdown fences, no extra text.
Each entry in "ranked_families" must have exactly:
    rank           : integer starting at 1
    family_name    : string — exactly as given in the input
    confidence     : float 0.0–1.0
    match_reason   : string <= 40 words explaining the visual match
    mismatch_notes : string <= 30 words or "" if confident
"""

_USER_PROMPT_TEMPLATE = """\
Rank the given stone families based on how well they visually match the query image.

Candidates:
{candidate_block}

Return ONLY this JSON object:
{{
  "ranked_families": [
    {{
      "rank": 1,
      "family_name": "<exact candidate name>",
      "confidence": 0.00,
      "match_reason": "<how slab matches this candidate's discriminators>",
      "mismatch_notes": "<key feature missing or differing, or ''>"
    }}
  ]
}}
"""

# ── Gemini call ──────────────────────────────────────────────────────────────
def _call_gemini(
    image_bytes: bytes,
    image_mime: str,
    candidate_block: str,
) -> dict:
    """
    Call Gemini using the API key set via set_gemini_api_key().
    """
    if _GEMINI_API_KEY is None:
        raise RuntimeError("Gemini API key not set. Please provide an API key.")
    
    try:
        client = _create_client()
        
        user_text = _USER_PROMPT_TEMPLATE.format(
            candidate_block=candidate_block,
        )

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
            types.Part.from_text(text=user_text),
        ]

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text
        if not raw_text or not raw_text.strip():
            raise RuntimeError("Gemini returned an empty response.")

        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()

        # Success – return parsed JSON
        return json.loads(clean)

    except Exception as e:
        error_msg = str(e).lower()
        # Check if this is an API key error
        if "api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
            raise RuntimeError("Invalid or unauthorized Gemini API key. Please check your API key.") from e
        raise


# ── Public async entry point ───────────────────────────────────────────────────
async def rerank_stone_families(
    image_bytes: bytes,
    image_mime: str,
    candidates: list[dict],
) -> dict:
    """
    Re-rank stone families using Gemini's visual understanding with discriminative keys.

    Parameters
    ----------
    image_bytes : raw image bytes — JPG, PNG, or WEBP
    image_mime  : MIME type string, e.g. "image/jpeg"
    candidates  : list of dicts, each with:
                    family_name : str   — exact stone family name
                    score       : float — optional (ignored, kept for compatibility)

    Returns
    -------
    dict with keys:
        image_summary   : str
        ranked_families : list of dicts
        kb_hits         : list[str] — families found in knowledge base
        kb_misses       : list[str] — families NOT found
    """
    if not candidates:
        raise ValueError("candidates list must not be empty.")
    
    if _GEMINI_API_KEY is None:
        raise RuntimeError("Gemini API key not set. Please provide an API key.")

    kb_hits = []
    kb_misses = []
    for c in candidates:
        if get_stone_profile(c["family_name"]):
            kb_hits.append(c["family_name"])
        else:
            kb_misses.append(c["family_name"])

    if kb_misses:
        logger.warning("KB misses (no descriptor found): %s", kb_misses)

    candidate_block = build_candidate_block(candidates)

    # Run the Gemini call in a thread pool to avoid blocking the event loop
    gemini_result = await asyncio.to_thread(
        _call_gemini,
        image_bytes,
        image_mime,
        candidate_block,
    )

    return {
        "image_summary": gemini_result.get("image_summary", ""),
        "ranked_families": gemini_result.get("ranked_families", []),
        "kb_hits": kb_hits,
        "kb_misses": kb_misses,
    }
