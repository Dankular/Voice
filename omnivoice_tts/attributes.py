"""Voice-design attributes and instruct-string handling.

The vocabulary and validation rules are those of upstream k2-fsa/OmniVoice
(``omnivoice/utils/voice_design.py`` and ``_resolve_instruct`` in
``omnivoice/models/omnivoice.py``, Apache-2.0). ``resolve_instruct`` below is a
port of that function; the facet lists and ``build_instruct`` are new.

Upstream documents only these categories: gender, age, pitch, style (whisper),
English accent, Chinese dialect. There is no documented emotion/"tone" control,
so the user-facing "tone" filter maps to pitch (+ the whisper style).
"""
import difflib
import re
from typing import Optional

from .voice_design import (
    _INSTRUCT_ALL_VALID,
    _INSTRUCT_CATEGORIES,
    _INSTRUCT_EN_TO_ZH,
    _INSTRUCT_MUTUALLY_EXCLUSIVE,
    _INSTRUCT_VALID_EN,
    _INSTRUCT_VALID_ZH,
    _INSTRUCT_ZH_TO_EN,
)

ZH_RE = re.compile(r"[一-鿿]")

# Ordered facets exposed to CLI/API/UI (English values).
FACETS = {
    "gender": ["male", "female"],
    "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "style": ["whisper"],
    "accent": sorted(c for c in _INSTRUCT_CATEGORIES[4]),
    "dialect": sorted(c for c in _INSTRUCT_CATEGORIES[5]),
}


def build_instruct(gender=None, age=None, pitch=None, style=None, accent=None, dialect=None) -> Optional[str]:
    """Join the chosen facet values into an instruct string (None if nothing chosen)."""
    items = [v for v in (gender, age, pitch, style, accent, dialect) if v]
    return ", ".join(items) if items else None


def resolve_instruct(instruct: Optional[str], use_zh: bool = False) -> Optional[str]:
    """Validate and normalise an instruct string (port of upstream ``_resolve_instruct``).

    Raises ValueError for unsupported items, mixed accent+dialect, or two items
    from the same category.
    """
    if instruct is None:
        return None
    instruct_str = instruct.strip()
    if not instruct_str:
        return None

    raw_items = [x for x in re.split(r"\s*[,，]\s*", instruct_str) if x]

    unknown, normalised = [], []
    for raw in raw_items:
        n = raw.strip().lower()
        if n in _INSTRUCT_ALL_VALID:
            normalised.append(n)
        else:
            sug = difflib.get_close_matches(n, _INSTRUCT_ALL_VALID, n=1, cutoff=0.6)
            unknown.append((raw, n, sug[0] if sug else None))
    if unknown:
        lines = [
            f"  '{raw}' (unsupported; did you mean '{sug}'?)" if sug else f"  '{raw}' (unsupported)"
            for raw, _n, sug in unknown
        ]
        raise ValueError(
            f"Unsupported instruct items in {instruct_str!r}:\n" + "\n".join(lines)
            + "\nValid English items: " + ", ".join(sorted(_INSTRUCT_VALID_EN))
        )

    has_dialect = any(n.endswith("话") for n in normalised)
    has_accent = any(" accent" in n for n in normalised)
    if has_dialect and has_accent:
        raise ValueError("Cannot mix a Chinese dialect and an English accent in one instruct.")
    if has_dialect:
        use_zh = True
    elif has_accent:
        use_zh = False

    table = _INSTRUCT_EN_TO_ZH if use_zh else _INSTRUCT_ZH_TO_EN
    normalised = [table.get(n, n) for n in normalised]

    conflicts = []
    for cat in _INSTRUCT_MUTUALLY_EXCLUSIVE:
        hits = [n for n in normalised if n in cat]
        if len(hits) > 1:
            conflicts.append(" vs ".join(f"'{x}'" for x in hits))
    if conflicts:
        raise ValueError(
            "Conflicting instruct items within the same category: " + "; ".join(conflicts)
            + ". Each category (gender, age, pitch, style, accent, dialect) allows at most one item."
        )

    sep = "，" if any(ZH_RE.search(n) for n in normalised) else ", "
    return sep.join(normalised)
