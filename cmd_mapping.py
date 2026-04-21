"""
cmd_mapping.py
==============
Mapping from CMD Office internal class/folder names → display Stone Names.

If a predicted family name matches any key in CMD_OFFICE_MAPPING (case-insensitive,
normalised), it should be REPLACED by the corresponding stone name before display.

Usage
-----
    from cmd_mapping import resolve_family_name

    display_name = resolve_family_name("tile_CMD_Office_Cabin_2")
    # → "Golden Spider"

    display_name = resolve_family_name("tile_Crema_Elysee_All_Variations")
    # → "tile_Crema_Elysee_All_Variations"   (unchanged – not a CMD class)
"""

from __future__ import annotations

# ── Mapping table (raw folder/class name  →  stone display name) ──────────
# Keys are normalised (lowercase, spaces instead of underscores/hyphens).
# Do NOT edit the keys manually; edit RAW_MAPPING below and the module
# derives the normalised lookup automatically.

RAW_MAPPING: dict[str, str] = {
    # Sr. No. | Area of Application              | Stone Name
    # -------   --------------------------------   -------------------
    "tile_CMD_office_Flooring_Storage":           "Grigio Bronze Amani",   # 1  (also stored as tile_CMD_office_Storage_Flooring)
    "tile_CMD_office_Storage_Flooring":           "Grigio Bronze Amani",   # 1
    "tile_CMD_Office_Storage_Flooring":           "Grigio Bronze Amani",   # 1  (capitalisation variant)
    "tile_CMD_office_Flooring_Storage_2":         "Breccia",               # 2
    "tile_CMD_office_Storage_Flooring_2":         "Breccia",               # 2  (variant)
    "tile_CMD_Office_Flooring_Storage_2":         "Breccia",               # 2
    "tile_CMD_Cabin_1":                           "Griccia Onyx",          # 3
    "tile_CMD_Office_Cabin_1":                    "Griccia Onyx",          # 3  (variant)
    "tile_CMD_Office_Cabin_2":                    "Golden Spider",         # 4
    "tile_CMD_Cabin_2":                           "Golden Spider",         # 4  (variant)
    "tile_CMD_Office_Reception_Flooring":         "Soda Lite Blue",        # 5
    "tile_CMD_Office_Conference_Table":           "Nozet Florry",          # 6
    "tile_CMD_Office_Stairs":                     "Nozet Florry",          # 7
    "tile_CMD_office_Stairs":                     "Nozet Florry",          # 7  (capitalisation variant)
}

# ── Derived normalised lookup (built once at import time) ─────────────────

def _normalise(name: str) -> str:
    """Lowercase + collapse underscores/hyphens to spaces + strip."""
    return name.lower().replace("_", " ").replace("-", " ").strip()


_NORMALISED_LOOKUP: dict[str, str] = {
    _normalise(k): v for k, v in RAW_MAPPING.items()
}


# ── Public API ────────────────────────────────────────────────────────────

def is_cmd_class(family_name: str) -> bool:
    """Return True if the family name is a CMD Office internal class."""
    return _normalise(family_name) in _NORMALISED_LOOKUP


def resolve_family_name(family_name: str) -> str:
    """
    Return the display stone name for CMD Office classes, or the original
    family name (with underscores replaced by spaces) for all others.
    """
    key = _normalise(family_name)
    if key in _NORMALISED_LOOKUP:
        return _NORMALISED_LOOKUP[key]
    # Default: just humanise underscores
    return family_name.replace("_", " ")


def resolve_family_entry(
    family_name: str,
    score: float,
) -> tuple[str, float, bool]:
    """
    Resolve a (family_name, score) pair.

    Returns
    -------
    (display_name, score, is_cmd)
        display_name : human-readable stone name
        score        : unchanged score
        is_cmd       : True if this was a CMD Office class
    """
    return resolve_family_name(family_name), score, is_cmd_class(family_name)