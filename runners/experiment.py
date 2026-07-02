"""Experiment matrix runner for the PLR prompt lab (P2-2).

Enumerates the cross-product of datasets × models × prompts × pipelines ×
attributes (× formats × reasons, both optional) from an experiment.yaml, then
for each cell:

  (a) run  — re_score.re_score with get_model(model)
  (b) eval — run_eval, writing a ledger record with
              dataset / model / pipeline / prompt_hash.

The lab is PLR-only (2026-07): the text-search pipeline was removed.

Fail-loud-but-continue: a cell that errors is caught, logged, recorded as
failed, and the runner moves to the next cell.  At the end a matrix summary is
printed.

Exit code:
  0  — at least one cell succeeded (or --strict is not set)
  1  — --strict flag is set and ANY cell failed
  2  — ALL cells failed
"""
from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_LAB_ROOT = Path(__file__).resolve().parent.parent  # lab root (runners/ is one below)


# =====================================================================
# Schema / validation
# =====================================================================

_REQUIRED_KEYS = {"datasets", "models", "prompts", "pipelines"}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning a dict.  Raises with a clear message."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for experiment.yaml parsing. "
            "Install it with: pip install pyyaml"
        ) from None
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"experiment.yaml must be a YAML mapping, got {type(data).__name__}"
        )
    return data


def _validate_schema(cfg: dict[str, Any], path: str | Path) -> None:
    """Validate experiment.yaml schema.  Raises ValueError with a clear message."""
    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise ValueError(
            f"experiment.yaml ({path}) is missing required keys: {sorted(missing)}\n"
            f"Required: {sorted(_REQUIRED_KEYS)}"
        )
    # Each required key must be a non-empty list
    for key in _REQUIRED_KEYS:
        val = cfg[key]
        if not isinstance(val, list) or not val:
            raise ValueError(
                f"experiment.yaml ({path}): '{key}' must be a non-empty list, "
                f"got {val!r}"
            )
    # 'attributes' is required for plr pipelines; validated per-cell
    # Ledger is optional
    for key in ("datasets", "models", "prompts", "pipelines"):
        for item in cfg[key]:
            if not isinstance(item, str):
                raise ValueError(
                    f"experiment.yaml ({path}): all items in '{key}' must be "
                    f"strings, got {item!r}"
                )
    if "attributes" in cfg:
        if not isinstance(cfg["attributes"], list) or not cfg["attributes"]:
            raise ValueError(
                f"experiment.yaml ({path}): 'attributes' must be a non-empty list "
                f"when present"
            )
        for item in cfg["attributes"]:
            if not isinstance(item, str):
                raise ValueError(
                    f"experiment.yaml ({path}): all items in 'attributes' must be "
                    f"strings, got {item!r}"
                )
    # Optional env axes: 'formats' (IR_PLR_FORMAT) and 'reasons' (IR_PLR_REASON).
    # Closed value sets — anything else is a typo we want to fail loudly on.
    _ENV_AXES = {"formats": {"yaml", "json"}, "reasons": {"on", "off"}}
    for key, allowed in _ENV_AXES.items():
        if key not in cfg:
            continue
        val = cfg[key]
        if not isinstance(val, list) or not val:
            raise ValueError(
                f"experiment.yaml ({path}): '{key}' must be a non-empty list "
                f"when present, got {val!r}"
            )
        bad = [x for x in val if not isinstance(x, str) or x not in allowed]
        if bad:
            raise ValueError(
                f"experiment.yaml ({path}): invalid value(s) in '{key}': {bad}. "
                f"Allowed: {sorted(allowed)}"
            )


def _validate_registry(cfg: dict[str, Any], path: str | Path) -> None:
    """Check model and pipeline names against the registry before running any cell."""
    from registry import MODELS, PIPELINES

    unknown_models = [m for m in cfg["models"] if m not in MODELS]
    if unknown_models:
        raise ValueError(
            f"experiment.yaml ({path}): unknown model(s): {unknown_models}. "
            f"Available: {sorted(MODELS)}"
        )

    unknown_pipelines = [p for p in cfg["pipelines"] if p not in PIPELINES]
    if unknown_pipelines:
        raise ValueError(
            f"experiment.yaml ({path}): unknown pipeline(s): {unknown_pipelines}. "
            f"Available: {sorted(PIPELINES)}"
        )


# =====================================================================
# Cell dataclass
# =====================================================================

@dataclass
class Cell:
    """One experiment cell — the smallest schedulable unit."""

    dataset: str
    model: str
    prompt: str      # version tag (e.g. plr_v1.4_cot)
    pipeline: str
    attribute: str   # empty string for search pipeline cells
    fmt: str = ""    # IR_PLR_FORMAT axis value ("yaml"|"json"); "" = env untouched
    reason: str = "" # IR_PLR_REASON axis value ("on"|"off"); "" = env untouched

    def label(self) -> str:
        parts = [
            f"dataset={self.dataset!r}",
            f"model={self.model!r}",
            f"prompt={self.prompt!r}",
            f"pipeline={self.pipeline!r}",
        ]
        if self.attribute:
            parts.append(f"attribute={self.attribute!r}")
        if self.fmt:
            parts.append(f"format={self.fmt!r}")
        if self.reason:
            parts.append(f"reason={self.reason!r}")
        return "{" + ", ".join(parts) + "}"

    def version_tag(self) -> str:
        """Ledger version stamp. The base prompt tag alone cannot distinguish
        two cells that differ only in the format/reason env axes (prompt_hash
        hashes files, not env), so the axis values are appended to the tag."""
        tag = self.prompt
        if self.fmt:
            tag += f"+{self.fmt}"
        if self.reason:
            tag += f"+reason-{self.reason}"
        return tag


@dataclass
class CellResult:
    cell: Cell
    status: str      # "ok" | "failed"
    error: str = ""


# =====================================================================
# Cell enumeration
# =====================================================================

def enumerate_cells(cfg: dict[str, Any]) -> list[Cell]:
    """Build the cross-product of axes into a flat list of Cells.

    Each (dataset, model, prompt, attribute[, format, reason]) tuple is a
    cell. The optional 'formats' / 'reasons' axes cross IR_PLR_FORMAT /
    IR_PLR_REASON per cell.
    """
    datasets: list[str] = cfg["datasets"]
    models: list[str] = cfg["models"]
    prompts: list[str] = cfg["prompts"]
    pipelines: list[str] = cfg["pipelines"]
    attributes: list[str] = cfg.get("attributes") or [""]
    formats: list[str] = cfg.get("formats") or [""]
    reasons: list[str] = cfg.get("reasons") or [""]

    cells: list[Cell] = []
    for pipeline, dataset, model, prompt in product(pipelines, datasets, models, prompts):
        # one cell per attribute × format × reason
        for attribute, fmt, reason in product(attributes, formats, reasons):
            cells.append(Cell(
                dataset=dataset,
                model=model,
                prompt=prompt,
                pipeline=pipeline,
                attribute=attribute,
                fmt=fmt,
                reason=reason,
            ))
    return cells


# =====================================================================
# Cell runner
# =====================================================================

def _set_prompt_env(prompt_version: str) -> None:
    """Set IR_PLR_FORMAT / IR_PLR_REASON from a prompt version tag.

    Mirrors what _cmd_run does: only propagate when the version tag IS a wire
    format name (yaml|json).  Named version tags like 'plr_v1.4_cot' are
    passed directly as the --version arg to run_eval.
    """
    if prompt_version.strip().lower() in {"yaml", "json"}:
        os.environ["IR_PLR_FORMAT"] = prompt_version.strip().lower()


def _apply_env_axes(cell: Cell) -> None:
    """Apply the cell's optional format/reason axes to the environment.

    Called after _set_prompt_env so an explicit 'formats' axis wins over a
    format-named prompt tag.  Guard: a yaml-backed prompt version pins its own
    wire format (FilePromptProvider reads it from the yaml's `format:` key),
    while the RESPONSE parser follows IR_PLR_FORMAT — a mismatch would make the
    model emit one format and the parser expect the other, failing every crop.
    Fail the cell loudly instead of producing garbage metrics.
    """
    if cell.fmt:
        yaml_path = _LAB_ROOT / "prompts" / f"{cell.prompt}.yaml"
        if yaml_path.exists():
            import yaml

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            pinned = str(data.get("format", "yaml")).strip().lower()
            if pinned != cell.fmt:
                raise RuntimeError(
                    f"format axis {cell.fmt!r} conflicts with prompts/{cell.prompt}.yaml "
                    f"which pins format={pinned!r}: the prompt would ask for {pinned} "
                    f"but the response parser (IR_PLR_FORMAT={cell.fmt}) would reject it. "
                    f"Drop this (prompt, format) combination or use a constants-backed "
                    f"version tag for the format axis."
                )
        os.environ["IR_PLR_FORMAT"] = cell.fmt
    if cell.reason:
        os.environ["IR_PLR_REASON"] = cell.reason


def _run_plr_cell(cell: Cell, ledger_path: str) -> None:
    """Run one PLR cell: re_score then run_eval."""
    from runners import re_score as rs
    from registry import get_model

    # (a) run — re_score
    ds_path = Path(cell.dataset)
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {cell.dataset}"
        )

    model = get_model(cell.model)
    _set_prompt_env(cell.prompt)
    _apply_env_axes(cell)

    meta = rs.re_score(
        attribute=cell.attribute,
        model=model,
        golden_dir=str(ds_path),
        prompt_version=cell.prompt,
    )
    log.debug("re_score meta: %s", meta)

    # (b) eval — run_eval.main() via importlib (matches lab.py pattern)
    spec = importlib.util.spec_from_file_location(
        "run_eval",
        str(_LAB_ROOT / "eval" / "run_eval.py"),
    )
    run_eval = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(run_eval)  # type: ignore[union-attr]

    orig_argv = sys.argv
    sys.argv = [
        "run_eval",
        "--attribute", cell.attribute,
        "--golden", str(ds_path),
        "--version", cell.version_tag(),
        "--ledger", ledger_path,
        "--model", cell.model,
        "--pipeline", cell.pipeline,
        "--dataset", str(ds_path),
    ]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            run_eval.main()
    except SystemExit as exc:
        if exc.code and int(exc.code) != 0:
            raise RuntimeError(
                f"run_eval exited with code {exc.code}"
            ) from exc
    finally:
        sys.argv = orig_argv

    log.debug("run_eval output: %s", buf.getvalue())


def run_cell(cell: Cell, ledger_path: str) -> CellResult:
    """Run one cell (run + eval).  Returns CellResult; never raises.

    IR_PLR_FORMAT / IR_PLR_REASON are snapshotted before and restored after the
    cell so the format/reason axes (and _set_prompt_env) never leak into the
    next cell — a cell without an axis value must see the pre-matrix env.
    """
    label = cell.label()
    print(f"[experiment] CELL {label}", flush=True)
    saved_env = {k: os.environ.get(k) for k in ("IR_PLR_FORMAT", "IR_PLR_REASON")}
    try:
        _run_plr_cell(cell, ledger_path)
        print(f"[experiment]   OK  {label}", flush=True)
        return CellResult(cell=cell, status="ok")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[experiment]   FAILED {label}\n             {msg}", flush=True)
        log.error("Cell failed: %s — %s", label, msg)
        return CellResult(cell=cell, status="failed", error=msg)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# =====================================================================
# Matrix runner entry point
# =====================================================================

def run_experiment(
    experiment_yaml: str | Path,
    strict: bool = False,
) -> int:
    """Parse experiment.yaml, enumerate cells, run each, print summary.

    Args:
      experiment_yaml: path to the experiment YAML file.
      strict:          if True, exit nonzero on ANY cell failure.

    Returns:
      exit code:
        0  all cells passed (or at least one passed and --strict not set)
        1  --strict set and at least one cell failed
        2  ALL cells failed
    """
    path = Path(experiment_yaml)
    cfg = _load_yaml(path)
    _validate_schema(cfg, path)
    _validate_registry(cfg, path)

    ledger_path = cfg.get("ledger") or str(_LAB_ROOT / "eval" / "ledger.jsonl")
    # Make ledger path absolute relative to the yaml's directory if not absolute
    if not os.path.isabs(ledger_path):
        ledger_path = str(path.parent / ledger_path)
    # Ensure the ledger directory exists
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)

    cells = enumerate_cells(cfg)
    n_total = len(cells)
    print(f"[experiment] {n_total} cells to run (from {path})", flush=True)

    results: list[CellResult] = []
    for cell in cells:
        result = run_cell(cell, ledger_path)
        results.append(result)

    # Summary
    n_ok = sum(1 for r in results if r.status == "ok")
    n_failed = sum(1 for r in results if r.status == "failed")

    print(f"\n[experiment] === MATRIX SUMMARY ===", flush=True)
    print(f"[experiment] total={n_total}  ok={n_ok}  failed={n_failed}", flush=True)
    if n_failed:
        print("[experiment] failed cells:", flush=True)
        for r in results:
            if r.status == "failed":
                print(f"[experiment]   {r.cell.label()}", flush=True)
                print(f"[experiment]     {r.error}", flush=True)

    if n_ok == 0:
        # All cells failed
        return 2
    if strict and n_failed > 0:
        return 1
    return 0
