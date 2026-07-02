"""FilePromptProvider — YAML-backed PromptProvider for the pluggable-modules layer.

Loads prompt templates from prompts/<version>.yaml and produces OpenAI-style
message lists identical to the module-level builder functions in plr_prompts.py.

Registration (called at import time so registry.get_provider("prompt") works):
    from registry import register
    register(slot="prompt", version="plr_v1.3_cot", provider_class=FilePromptProvider)

Three versions are pre-registered:
  - plr_v1.4_cot  (YAML-CoT, prompt-native military judgment + reconciled hints)
  - plr_v1.3_cot  (YAML-CoT, default per IR_PROMPT_VER)
  - plr_v0.4      (JSON legacy)
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml

from providers import PromptProvider

# Resolved once at import so every instance shares the same base path.
_PROMPTS_DIR = pathlib.Path(__file__).parent.parent / "prompts"


def _load_yaml(version: str) -> dict[str, Any]:
    """Load and parse prompts/<version>.yaml.  Raises FileNotFoundError if absent."""
    path = _PROMPTS_DIR / f"{version}.yaml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML {path} did not parse to a dict")
    return data


class FilePromptProvider(PromptProvider):
    """Reads prompt templates from prompts/<version>.yaml.

    The YAML schema mirrors the template text in plr_prompts.py exactly.
    Builder methods produce identical output to the module-level functions
    in plr_prompts.py so downstream callers see no difference.

    Parameters
    ----------
    version_override : optional explicit version string.  When omitted the
        class uses its own _VERSION class attribute (set by the two subclasses
        registered below).  This lets a single class serve both prompt families.
    """

    # Subclass-level version tag — overridden by _Plrv04Provider and
    # _Plrv13CotProvider.  Direct instantiation of FilePromptProvider uses
    # the config default.
    _VERSION: str = ""

    def __init__(self, version_override: str | None = None) -> None:
        ver = version_override or self._VERSION
        if not ver:
            # Resolve from config when not pinned by a subclass.
            import config as _cfg
            ver = getattr(_cfg, "IR_PROMPT_VER", "plr_v1.3_cot")
        self._ver = ver
        self._data = _load_yaml(ver)
        self._fmt: str = str(self._data.get("format", "yaml")).strip().lower()
        # plr_v1.5_cot forced-commit contract: when the version yaml declares
        # `commit_enums: true`, the `*unknown*` escape hatches are excluded
        # from the enum lists offered to the model (mirrors
        # plr_prompts._commit_enum). Per-version so older yamls (v1.3/v1.4)
        # keep reproducing their historical prompts byte-for-byte.
        self._commit_enums: bool = bool(self._data.get("commit_enums", False))
        # Variant enum overrides: the version yaml may pin its own enum lists
        # (`enums: {colors: [...], ...}`) — the injected list then comes from
        # the yaml verbatim instead of plr_schema. This makes the enum
        # vocabulary part of the versioned variant bundle; at promotion the
        # winning lists are baked into plr_schema itself.
        self._enum_overrides: dict[str, list] = dict(self._data.get("enums") or {})

    # ------------------------------------------------------------------
    # PromptProvider contract
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._ver

    def build_plr_messages(self, object_hint: str = "person") -> list[dict[str, Any]]:
        """Build chat messages for one PLR call.

        Mirrors plr_prompts.build_plr_messages() exactly.
        """
        from plr_schema import (
            COLOR_ENUM, UPPER_TYPE_ENUM, LOWER_TYPE_ENUM,
            EQUIPMENT_TYPE_ENUM, STATIC_ACTION_ENUM,
            GENDER_ENUM, AGE_GROUP_ENUM, OUTFIT_TYPE_ENUM,
            VEHICLE_TYPE_ENUM, MILITARY_ENUM,
        )
        plr = self._data["plr"]
        sys_text: str = plr["system"].rstrip("\n")

        if self._commit_enums:
            from plr_prompts import _commit_enum
            _base = _commit_enum
        else:
            _base = tuple

        def _e(values, _key=None):
            if _key and _key in self._enum_overrides:
                return tuple(str(v) for v in self._enum_overrides[_key])
            return _base(values)

        if object_hint == "vehicle":
            user_text = plr["vehicle_user"].rstrip("\n").format(
                colors=", ".join(_e(COLOR_ENUM, "colors")),
                vehicle_types=", ".join(_e(VEHICLE_TYPE_ENUM, "vehicle_types")),
                military_enum="|".join(_e(MILITARY_ENUM, "military")),
            )
        else:
            # person path — CoT (reason) vs plain depends on env + template key.
            with_reason = _plr_with_reason()
            if with_reason and "person_user" in plr:
                tmpl_key = "person_user"
            elif not with_reason and "person_user_no_reason" in plr:
                tmpl_key = "person_user_no_reason"
            else:
                tmpl_key = "person_user"

            fmt_kwargs: dict[str, str]
            if self._fmt == "json":
                fmt_kwargs = dict(
                    genders=", ".join(GENDER_ENUM),
                    ages=", ".join(AGE_GROUP_ENUM),
                    outfits=", ".join(OUTFIT_TYPE_ENUM),
                    colors=", ".join(COLOR_ENUM),
                    upper_types=", ".join(UPPER_TYPE_ENUM),
                    lower_types=", ".join(LOWER_TYPE_ENUM),
                    equips=", ".join(EQUIPMENT_TYPE_ENUM),
                    actions=", ".join(STATIC_ACTION_ENUM),
                )
            else:
                fmt_kwargs = dict(
                    colors=", ".join(_e(COLOR_ENUM, "colors")),
                    upper_types=", ".join(_e(UPPER_TYPE_ENUM, "upper_types")),
                    lower_types=", ".join(_e(LOWER_TYPE_ENUM, "lower_types")),
                    equips=", ".join(_e(EQUIPMENT_TYPE_ENUM, "equips")),
                    actions=", ".join(_e(STATIC_ACTION_ENUM, "actions")),
                    military_enum="|".join(_e(MILITARY_ENUM, "military")),
                )
            user_text = plr[tmpl_key].rstrip("\n").format(**fmt_kwargs)

        return [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def build_freeform_vqa_messages(
        self,
        residue: list[dict[str, Any]] | list[str] | str,
    ) -> list[dict[str, Any]]:
        """Build yes/no VQA messages.

        Mirrors plr_prompts.build_freeform_vqa_messages() exactly.
        Logic is self-contained (no YAML template needed — the prompt is
        constructed dynamically from residue content).
        """
        pos_clauses: list[str] = []
        neg_clauses: list[str] = []

        def _clause(subj: str, attr_en: str, attr_ko: str, negate: bool) -> str:
            anchor = f"the {subj}" if subj else "the subject"
            descr = attr_en or attr_ko or ""
            if attr_en and attr_ko and attr_en != attr_ko:
                descr = f'{attr_en} (Korean: "{attr_ko}")'
            elif attr_ko and not attr_en:
                descr = f'"{attr_ko}"'
            verb = "does NOT show" if negate else "clearly shows"
            return f"{anchor} {verb} {descr}"

        def _collect(item: Any) -> None:
            if isinstance(item, dict):
                subj = (item.get("subject") or "").strip()
                attr_en = (item.get("attribute_en") or "").strip()
                attr_ko = (item.get("attribute_ko") or "").strip()
                if not (attr_en or attr_ko):
                    attr_ko = (item.get("attribute") or "").strip()
                if not (attr_en or attr_ko):
                    return
                negate = bool(item.get("is_negative"))
            else:
                attr_ko = (str(item) or "").strip()
                if not attr_ko:
                    return
                subj, attr_en, negate = "", "", False
            bucket = neg_clauses if negate else pos_clauses
            bucket.append(_clause(subj, attr_en, attr_ko, negate))

        if isinstance(residue, str):
            _collect(residue)
        elif isinstance(residue, list):
            for r in residue:
                _collect(r)

        clauses = pos_clauses + neg_clauses
        if not clauses:
            clauses = ["the listed attribute is clearly visible"]

        sys_text = (
            "You are a visual matcher. The crop's subject (person/vehicle/"
            "bag/etc.) has already been verified by an upstream filter — "
            "your job is to confirm the listed attribute(s) on that "
            "subject. Be strict about which subject each attribute belongs "
            "to: if an attribute would normally apply to a different "
            "subject than the one named, answer 'no'. Pay close attention "
            "to clauses that say 'does NOT show' — for those, answer 'no' "
            "if the attribute IS visible. Answer with one lowercase word: "
            "yes or no. No punctuation, no explanation."
        )
        user_text = (
            "Does this image satisfy ALL of the following: "
            + " AND ".join(clauses)
            + "? Answer yes or no."
        )
        return [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def build_plr_retry_messages(
        self,
        object_hint: str,
        original_response: str,
        error_reason: str,
    ) -> list[dict[str, Any]]:
        """Build retry messages when first PLR response fails parse/schema.

        Mirrors plr_prompts.build_plr_retry_messages() exactly.
        """
        format_word = "YAML" if self._fmt == "yaml" else "JSON"
        retry_text = (
            f"Your previous response did not match the required schema:\n"
            f"  Error: {error_reason}\n\n"
            f"Your previous output was:\n{original_response[:500]}\n\n"
            f"Output ONLY a corrected {format_word} response that conforms to the "
            f"schema. No markdown fences, no leading prose."
        )
        base = self.build_plr_messages(object_hint)
        base.append({"role": "user", "content": [{"type": "text", "text": retry_text}]})
        return base

    def build_query_parser_messages(self, user_query: str) -> list[dict[str, Any]]:
        """Build chat messages for a query-parser call.

        Mirrors plr_prompts.build_query_parser_messages() exactly.
        """
        qp = self._data["query_parser"]
        sys_text: str = qp["system"].rstrip("\n")
        user_text: str = qp["user"].rstrip("\n").format(
            user_query=user_query.replace('"', '\\"'),
        )
        return [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_text},
        ]


# ---------------------------------------------------------------------------
# Thin version-pinned subclasses used for registry entries
# ---------------------------------------------------------------------------

class _Plrv04Provider(FilePromptProvider):
    _VERSION = "plr_v0.4"


class _Plrv13CotProvider(FilePromptProvider):
    _VERSION = "plr_v1.3_cot"


class _Plrv14CotProvider(FilePromptProvider):
    _VERSION = "plr_v1.4_cot"


class _Plrv15CotProvider(FilePromptProvider):
    _VERSION = "plr_v1.5_cot"


# ---------------------------------------------------------------------------
# Self-registration at import time
# ---------------------------------------------------------------------------

def _register_all() -> None:
    from registry import register
    register(slot="prompt", version="plr_v0.4", provider_class=_Plrv04Provider)
    register(slot="prompt", version="plr_v1.3_cot", provider_class=_Plrv13CotProvider)
    register(slot="prompt", version="plr_v1.4_cot", provider_class=_Plrv14CotProvider)
    register(slot="prompt", version="plr_v1.5_cot", provider_class=_Plrv15CotProvider)


_register_all()


# ---------------------------------------------------------------------------
# Internal helpers (mirrors plr_prompts private helpers)
# ---------------------------------------------------------------------------

def _plr_with_reason() -> bool:
    """Toggle via IR_PLR_REASON=on (off by default)."""
    v = os.environ.get("IR_PLR_REASON", "off").strip().lower()
    return v in {"on", "true", "1", "yes"}
