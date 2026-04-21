"""
stone_reranker.py
-----------------
Gemini-powered visual reranking for stone families using discriminative visual keys.

Model: gemini-3.1-flash-lite-preview

Improvements over previous version:
- Candidate block now highlights only the most differentiating visual attributes
- Emphasizes pattern type, directional features, and unique markers
- Instructs Gemini to first describe the slab's visual pattern, then match
- Reduces false matches (e.g., Carrara vs Bianco Vogue) by focusing on banding

Multi‑API‑key fallback:
- Supports GEMINI_API_KEY (legacy) and GEMINI_API_KEY1..6
- On quota/resource exhausted errors, automatically switches to the next available key
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

# ── Gemini configuration (multiple keys) ──────────────────────────────────────
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

def _load_api_keys() -> list[str]:
    """Load all available API keys from environment variables.
    
    Priority order:
        1. GEMINI_API_KEY (legacy)
        2. GEMINI_API_KEY1, GEMINI_API_KEY2, ... GEMINI_API_KEY6
    Empty or None keys are filtered out.
    """
    keys = []
    
    # Legacy single key
    legacy_key = os.getenv("GEMINI_API_KEY")
    if legacy_key and legacy_key.strip():
        keys.append(legacy_key.strip())
    
    # Numbered keys
    for i in range(1, 7):
        key = os.getenv(f"GEMINI_API_KEY{i}")
        if key and key.strip():
            keys.append(key.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
    
    return unique_keys

_API_KEYS = _load_api_keys()
if not _API_KEYS:
    raise RuntimeError(
        "No Gemini API keys found. Please set GEMINI_API_KEY or GEMINI_API_KEY1..6 "
        "in your .env file. Get free keys at https://aistudio.google.com"
    )

logger.info("Loaded %d Gemini API key(s)", len(_API_KEYS))


def _create_client(api_key: str) -> genai.Client:
    """Create a new Gemini client with the given API key."""
    return genai.Client(api_key=api_key)


# ── Stone knowledge base ───────────────────────────────────────────────────────
_KB_PATH = Path(__file__).parent / "stones_db.json"
_STONES_DB: dict[str, dict] | None = None


def _load_kb() -> dict[str, dict]:
    global _STONES_DB
    if _STONES_DB is None:
        if not _KB_PATH.exists():
            raise FileNotFoundError(
                f"Stone knowledge base not found at {_KB_PATH}. "
                "Place stones_db.json in the same directory as stone_reranker.py."
            )
        with open(_KB_PATH, encoding="utf-8") as fh:
            _STONES_DB = json.load(fh)
        logger.info("Loaded %d stones from knowledge base.", len(_STONES_DB))
    return _STONES_DB


def _get_stone_profile(family_name: str) -> dict | None:
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


def _build_candidate_block(candidates: list[dict]) -> str:
    """Build a concise, discriminative summary for each candidate."""
    lines: list[str] = []
    for idx, cand in enumerate(candidates, start=1):
        name = cand["family_name"]
        profile = _get_stone_profile(name)

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

# ── Gemini call with multi‑key fallback ────────────────────────────────────────
def _call_gemini_with_fallback(
    image_bytes: bytes,
    image_mime: str,
    candidate_block: str,
) -> dict:
    """
    Attempt to call Gemini using available API keys.
    On quota/resource exhausted errors, automatically retry with the next key.
    """
    last_exception = None

    for idx, api_key in enumerate(_API_KEYS):
        try:
            client = _create_client(api_key)
            
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
            # Check if this is a quota / resource exhausted error
            is_quota_error = (
                "resource exhausted" in error_msg
                or "quota" in error_msg
                or "429" in error_msg
                or (isinstance(e, genai_errors.ClientError) and e.status_code == 429)
            )
            
            if is_quota_error:
                logger.warning(
                    "API key %d (prefix %s) failed with quota/resource exhausted error. "
                    "Switching to next key if available.",
                    idx + 1,
                    api_key[:8] + "...",
                )
                last_exception = e
                continue  # try next key
            else:
                # Non‑quota error – re‑raise immediately
                logger.error("Non‑retryable error with API key %d: %s", idx + 1, e)
                raise

    # If we exhausted all keys, raise the last quota error
    raise RuntimeError(
        "All available Gemini API keys have exhausted their quota or are invalid. "
        "Please add fresh keys or wait for quota reset."
    ) from last_exception


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

    kb_hits = []
    kb_misses = []
    for c in candidates:
        if _get_stone_profile(c["family_name"]):
            kb_hits.append(c["family_name"])
        else:
            kb_misses.append(c["family_name"])

    if kb_misses:
        logger.warning("KB misses (no descriptor found): %s", kb_misses)

    candidate_block = _build_candidate_block(candidates)

    # Run the Gemini call (with fallback) in a thread pool to avoid blocking the event loop
    gemini_result = await asyncio.to_thread(
        _call_gemini_with_fallback,
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