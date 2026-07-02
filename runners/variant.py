"""variant — an EXPERIMENT COMBINATION file: prompt version × input knobs.

`prompts/<V>.yaml` stays a pure PROMPT version (templates only).
`variants/<name>.yaml` COMPOSES a combination by reference, so the same
prompt version can be crossed with different enum sets / preprocessing /
sampling without copying template text (copy = drift):

    # variants/male_fix_a.yaml
    prompt: plr_v1.5_cot        # REQUIRED — references prompts/<V>.yaml
    enums:                      # optional — override the injected enum lists
      colors: [black, white, red]
    preprocess:
      marker: false             # optional — skip the yellow corner marker
    sampling:
      max_tokens: 256           # optional — forwarded to the model
      temperature: 0.2

`lab run --version <name>` accepts a variant name OR a plain prompt version;
the ledger stamps the variant name, and provenance covers the combination
(prompt_hash hashes variants/*.yaml alongside the prompt surface).

Promotion mapping (a winning knob goes back to its home file in core/ir):
  plr templates -> plr_prompts.py constants   | enums    -> plr_schema.py
  preprocess    -> indexing/_draw call site   | sampling -> gemma call sites
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Variant:
    """One experiment combination (loaded from variants/<name>.yaml)."""

    name: str                 # variant file stem — the ledger version tag
    prompt: str               # referenced prompt version (prompts/<prompt>.yaml)
    enums: dict[str, list[str]] = field(default_factory=dict)
    marker: bool = True
    max_tokens: int | None = None
    temperature: float | None = None


def load_variant(lab_root: str | Path, name: str | None) -> Variant | None:
    """Load variants/<name>.yaml. Returns None when no such variant exists
    (the caller then treats `name` as a plain prompt version). Fail-loud on a
    malformed variant: a missing/dangling `prompt:` reference is an error,
    not a silent constants fallback.
    """
    if not name:
        return None
    root = Path(lab_root)
    path = root / "variants" / f"{name}.yaml"
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"{path}: a variant must declare `prompt: <version>`")
    if not (root / "prompts" / f"{prompt}.yaml").exists():
        raise ValueError(
            f"{path}: prompt {prompt!r} not found in prompts/ — dangling reference"
        )
    pre = data.get("preprocess") or {}
    samp = data.get("sampling") or {}
    return Variant(
        name=name,
        prompt=prompt,
        enums=dict(data.get("enums") or {}),
        marker=bool(pre.get("marker", True)),
        max_tokens=int(samp["max_tokens"]) if samp.get("max_tokens") is not None else None,
        temperature=float(samp["temperature"]) if samp.get("temperature") is not None else None,
    )


def apply_sampling(model: Any, variant: Variant | None) -> None:
    """Forward the variant's sampling knobs to models that expose them.

    Duck-typed: LabGemmaModel has max_tokens/temperature attributes;
    MockModel has none and is silently unaffected (deterministic anyway).
    """
    if variant is None:
        return
    for key in ("max_tokens", "temperature"):
        val = getattr(variant, key)
        if val is not None and hasattr(model, key):
            setattr(model, key, val)
