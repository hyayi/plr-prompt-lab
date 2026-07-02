"""exp_config — an EXPERIMENT PARAMETER CONFIG: prompt × input knobs.

`prompts/<V>.yaml` is a pure PROMPT version (templates only).
`configs/<name>.yaml` COMPOSES an experiment by referencing each component,
so the same prompt version can be crossed with different enum sets /
preprocessing / sampling without copying template text (copy = drift):

    # configs/male_fix_a.yaml
    prompt: prompts/plr_v1.5_cot.yaml   # REQUIRED — path (or bare version name)
    enums:                              # optional — inline mapping OR a yaml path
      colors: [black, white, red]
    preprocess:
      marker: false                     # optional — skip the yellow corner marker
    sampling:
      max_tokens: 256                   # optional — forwarded to the model
      temperature: 0.2

`lab run --version <name>` accepts a config name OR a plain prompt version;
the ledger stamps the config name, and provenance covers the combination
(prompt_hash hashes configs/*.yaml alongside the prompt surface).

Promotion mapping (a winning knob goes back to its home file in core/ir):
  plr templates -> plr_prompts.py constants   | enums    -> plr_schema.py
  preprocess    -> indexing/_draw call site   | sampling -> gemma call sites
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExpConfig:
    """One experiment parameter config (loaded from configs/<name>.yaml)."""

    name: str                 # config file stem — the ledger version tag
    prompt: str               # resolved prompt VERSION (prompts/<prompt>.yaml)
    enums: dict[str, list[str]] = field(default_factory=dict)
    marker: bool = True
    max_tokens: int | None = None
    temperature: float | None = None


def _resolve_prompt_version(root: Path, ref: str, cfg_path: Path) -> str:
    """`prompt:` accepts a path ("prompts/plr_v1.5_cot.yaml") or a bare
    version name ("plr_v1.5_cot"). Returns the version name; dangling
    references fail loud."""
    ref = ref.strip()
    version = Path(ref).stem if ref.endswith(".yaml") else ref
    if not (root / "prompts" / f"{version}.yaml").exists():
        raise ValueError(
            f"{cfg_path}: prompt {ref!r} not found under prompts/ — dangling reference"
        )
    return version


def load_config(lab_root: str | Path, name: str | None) -> ExpConfig | None:
    """Load configs/<name>.yaml. Returns None when no such config exists
    (the caller then treats `name` as a plain prompt version). Fail-loud on
    a malformed config: a missing/dangling `prompt:` reference is an error,
    not a silent constants fallback.
    """
    if not name:
        return None
    root = Path(lab_root)
    path = root / "configs" / f"{name}.yaml"
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompt_ref = str(data.get("prompt") or "").strip()
    if not prompt_ref:
        raise ValueError(f"{path}: an experiment config must declare `prompt:`")
    prompt = _resolve_prompt_version(root, prompt_ref, path)

    enums = data.get("enums") or {}
    if isinstance(enums, str):  # component by path: a yaml file of enum lists
        enums_path = root / enums
        if not enums_path.exists():
            raise ValueError(f"{path}: enums file {enums!r} not found — dangling reference")
        enums = yaml.safe_load(enums_path.read_text(encoding="utf-8")) or {}

    pre = data.get("preprocess") or {}
    samp = data.get("sampling") or {}
    return ExpConfig(
        name=name,
        prompt=prompt,
        enums=dict(enums),
        marker=bool(pre.get("marker", True)),
        max_tokens=int(samp["max_tokens"]) if samp.get("max_tokens") is not None else None,
        temperature=float(samp["temperature"]) if samp.get("temperature") is not None else None,
    )


def apply_sampling(model: Any, cfg: ExpConfig | None) -> None:
    """Forward the config's sampling knobs to models that expose them.

    Duck-typed: LabGemmaModel has max_tokens/temperature attributes;
    MockModel has none and is silently unaffected (deterministic anyway).
    """
    if cfg is None:
        return
    for key in ("max_tokens", "temperature"):
        val = getattr(cfg, key)
        if val is not None and hasattr(model, key):
            setattr(model, key, val)
