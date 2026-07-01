"""YamlParser — concrete Parser implementation for slot "parser", version "qp_v0.4".

Loads surface-form synonym tables from qp_v0.4.yaml (same directory) and
delegates the heavy parsing logic to query_parser (dictionary + Gemma paths)
and plr_prompts (PLR response parsing).

Registration (side-effect on import):
    from registry import register
    register(slot="parser", version="qp_v0.4", provider_class=YamlParser)

Backward-compat shim:
    query_parser.py imports this module at the bottom and re-exports
    parse_query / parse_plr_response so existing ``import query_parser``
    callers are unaffected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML synonym tables — loaded once at import time
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).parent / "qp_v0.4.yaml"

# Populated by _load_synonyms() below.
_SYNONYMS: dict[str, dict[str, str]] = {}


def _load_synonyms() -> dict[str, dict[str, str]]:
    """Load qp_v0.4.yaml and return the synonym tables as plain dicts.

    Falls back to the in-module tables in query_parser if yaml is unavailable
    (e.g. test environments without PyYAML installed).
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        log.warning(
            "PyYAML not available — YamlParser will delegate to query_parser "
            "in-module synonym tables."
        )
        return {}

    try:
        with open(_YAML_PATH, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except OSError as exc:
        log.warning("Could not read %s (%s) — using query_parser fallback", _YAML_PATH, exc)
        return {}

    if not isinstance(raw, dict):
        log.warning("qp_v0.4.yaml root is not a mapping — using query_parser fallback")
        return {}

    # Coerce every value to str (YAML may parse numeric keys or values).
    result: dict[str, dict[str, str]] = {}
    for table_name, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        result[table_name] = {str(k): str(v) for k, v in entries.items()}
    return result


_SYNONYMS = _load_synonyms()


def get_synonym_table(name: str) -> dict[str, str]:
    """Return the named synonym table from the YAML, or {} if not present."""
    return _SYNONYMS.get(name, {})


# ---------------------------------------------------------------------------
# Patch query_parser module-level synonym dicts from YAML (if loaded)
# ---------------------------------------------------------------------------

def _patch_query_parser_synonyms() -> None:
    """Replace query_parser's in-module synonym tables with the YAML versions.

    This is the mechanism that makes the YAML the single source of truth:
    after patching, parse_with_dictionary() (and all helpers) transparently
    use the YAML-loaded dicts without any other code change required.

    The patch is skipped silently when:
      - YAML could not be loaded (empty _SYNONYMS), or
      - query_parser is not yet importable (circular-import guard).
    """
    if not _SYNONYMS:
        return
    try:
        import query_parser as _qp  # noqa: PLC0415
    except ImportError:
        return

    _MAP = {
        "color_synonyms":          "_COLOR_SYNONYMS",
        "color_group_keywords":    "_COLOR_GROUP_KEYWORDS",
        "upper_type_synonyms":     "_UPPER_TYPE_SYNONYMS",
        "sleeve_synonyms":         "_SLEEVE_SYNONYMS",
        "lower_type_synonyms":     "_LOWER_TYPE_SYNONYMS",
        "equipment_synonyms":      "_EQUIPMENT_SYNONYMS",
        "equipment_group_keywords":"_EQUIPMENT_GROUP_KEYWORDS",
        "action_synonyms":         "_ACTION_SYNONYMS",
        "gender_synonyms":         "_GENDER_SYNONYMS",
        "age_synonyms":            "_AGE_SYNONYMS",
        "outfit_type_synonyms":    "_OUTFIT_TYPE_SYNONYMS",
        "vehicle_type_synonyms":   "_VEHICLE_TYPE_SYNONYMS",
    }
    for yaml_key, qp_attr in _MAP.items():
        table = _SYNONYMS.get(yaml_key)
        if table and hasattr(_qp, qp_attr):
            existing = getattr(_qp, qp_attr)
            existing.clear()
            existing.update(table)
            log.debug("YamlParser: patched query_parser.%s from YAML (%d entries)",
                      qp_attr, len(table))


# ---------------------------------------------------------------------------
# Concrete Parser implementation
# ---------------------------------------------------------------------------

from providers import Parser  # noqa: E402


class YamlParser(Parser):
    """Parser provider backed by YAML synonym tables (qp_v0.4).

    Delegates all heavy logic to the existing query_parser and plr_prompts
    modules — the only difference from calling them directly is that the
    synonym tables are loaded from YAML rather than being hardcoded in Python.
    """

    version: str = "qp_v0.4"  # type: ignore[assignment]

    def __init__(self) -> None:
        # Patch the query_parser module-level dicts so that any code still
        # importing query_parser directly also uses the YAML tables.
        _patch_query_parser_synonyms()

    @property  # type: ignore[override]
    def version(self) -> str:  # type: ignore[override]
        return "qp_v0.4"

    def parse_plr_response(
        self,
        raw: str,
        hint: str = "person",
        *,
        fmt: str | None = None,
    ) -> dict[str, Any]:
        """Parse a raw PLR model response into a validated dict.

        Delegates to plr_prompts.parse_plr_response which dispatches
        between YAML and JSON paths based on IR_PLR_FORMAT / fmt override.

        Raises ValueError on unrecoverable parse failure (propagated from
        plr_prompts.parse_plr_yaml / parse_plr_json).
        """
        from plr_prompts import parse_plr_response as _parse  # noqa: PLC0415
        return _parse(raw, hint=hint, fmt=fmt)

    def parse_query(
        self,
        user_text: str,
        *,
        backend: Any = None,
        force_gemma: bool = False,
    ) -> Any:
        """Parse a user query string into a QueryJSON.

        Delegates to query_parser.parse_query (Gemma + dictionary paths).
        The synonym tables used by the dictionary path are the YAML-loaded
        ones (patched in __init__).
        """
        from query_parser import parse_query as _parse  # noqa: PLC0415
        return _parse(user_text, backend=backend, force_gemma=force_gemma)


# ---------------------------------------------------------------------------
# Registration — runs on import
# ---------------------------------------------------------------------------

from registry import register  # noqa: E402

register(slot="parser", version="qp_v0.4", provider_class=YamlParser)
log.debug("parser/yaml_parser: registered YamlParser as parser/qp_v0.4")
