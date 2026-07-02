"""variant — one prompts/<V>.yaml versions the WHOLE input combination.

The model's input is more than the template text: the enum lists injected
into it, the image pre-processing (target marker), and the sampling
parameters all shape the output. A variant yaml bundles every one of those
knobs under a single version name, so an experiment result is attributable
to exactly one recorded combination (`prompt_hash` hashes the yaml, so the
combination is provenance-stamped automatically).

Optional variant keys (all default to the production behaviour):

    enums:                 # override the injected enum lists (verbatim)
      colors: [black, white, red]
    preprocess:
      marker: false        # skip the yellow corner marker (default true)
    sampling:
      max_tokens: 256      # forwarded to models exposing the attribute
      temperature: 0.2

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
    """Effective input-combination knobs for one prompt version."""

    version: str
    enums: dict[str, list[str]] = field(default_factory=dict)
    marker: bool = True
    max_tokens: int | None = None
    temperature: float | None = None


def load_variant(lab_root: str | Path, version: str | None) -> Variant | None:
    """Parse the variant knobs from prompts/<version>.yaml.

    Returns None when the version has no yaml (constants / mock path) — the
    caller then keeps every production default.
    """
    if not version:
        return None
    path = Path(lab_root) / "prompts" / f"{version}.yaml"
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pre = data.get("preprocess") or {}
    samp = data.get("sampling") or {}
    return Variant(
        version=version,
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
