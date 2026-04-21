# query/name_utils.py
"""
Handles name mismatches between ChromaDB family names and model family names.
"""

import re


def _clean(name: str) -> str:
    name = name.strip().strip("'\"").strip()
    name = re.sub(r"[\s_]+", "_", name)
    return name


def to_underscore(name: str) -> str:
    return _clean(name)


def to_space(name: str) -> str:
    return _clean(name).replace("_", " ")


def build_alias_map(family_names: list) -> dict:
    alias = {}
    for fname in family_names:
        cleaned = _clean(fname)
        for variant in (
            fname,
            fname.strip(),
            cleaned,
            cleaned.replace("_", " "),
            fname.strip("'\"").strip(),
        ):
            alias.setdefault(variant, fname)
    return alias


def resolve_name(name: str, alias_map: dict) -> str | None:
    for candidate in (
        name,
        name.strip(),
        name.strip("'\"").strip(),
        _clean(name),
        _clean(name).replace("_", " "),
    ):
        if candidate in alias_map:
            return alias_map[candidate]
    return None