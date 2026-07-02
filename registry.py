"""Registries for the lab.

Two registries live here:

1. A minimal provider REGISTER shim (`register`) — the byte-synced modules
   (providers/file_prompt_provider.py, gemma_backend.py) self-register their
   provider classes at import time, exactly as they do in core/ir. The lab
   never RESOLVES providers through a registry (it instantiates
   FilePromptProvider directly with version_override), so core/ir's
   get_provider/validation machinery was dropped in the 2026-07 slim-down —
   this shim only has to accept the registrations.

2. The MODELS / PIPELINES experiment registry (P2-1) — the axes the lab CLI
   and the experiment runner select over.
"""

from __future__ import annotations

import logging
from typing import Any, Type

log = logging.getLogger(__name__)

_SLOTS = ("model", "prompt", "parser", "scoring")

# slot -> {version_string -> provider_class}
_registry: dict[str, dict[str, Type[Any]]] = {s: {} for s in _SLOTS}


def register(slot: str, version: str, provider_class: Type[Any]) -> None:
    """Accept a provider registration (import-time hook of the synced modules)."""
    if slot not in _SLOTS:
        raise ValueError(f"Unknown slot {slot!r}. Valid slots: {list(_SLOTS)}")
    _registry[slot][version] = provider_class
    log.debug("registry: registered slot=%r version=%r class=%r",
              slot, version, provider_class.__name__)


# ===========================================================================
# Model / Pipeline registry (Phase-2, P2-1)
# ---------------------------------------------------------------------------
# A SEPARATE, higher-level registry from the provider slots above: it names the
# experiment axes the lab CLI and the P2-2 experiment runner select over —
# WHICH model (gemma vs mock) and WHICH pipeline (plr vs search).  Kept in this
# module so there is one obvious place to discover selectable parameters.
#
# `MODELS`    : name -> zero-arg factory returning a gemma_model.Model.
# `PIPELINES` : name -> Pipeline descriptor (the run/eval seam P2-2 builds on).
# ===========================================================================

from dataclasses import dataclass  # noqa: E402
from typing import Callable  # noqa: E402


def _make_gemma_model() -> Any:
    """Factory for the real (GPU) Gemma model.  Imported lazily so that merely
    listing/registering models never pulls the heavy backend."""
    from gemma_model import LabGemmaModel

    return LabGemmaModel()


def _make_mock_model() -> Any:
    """Factory for the deterministic, GPU-free MockModel (usable outside tests)."""
    from gemma_model import MockModel

    return MockModel()


# name -> factory().  Factories are zero-arg and construct a fresh Model.
MODELS: dict[str, Callable[[], Any]] = {
    "gemma": _make_gemma_model,
    "mock": _make_mock_model,
}


def get_model(name: str) -> Any:
    """Return a fresh model instance for a registered name.

    `get_model("mock")` is fully GPU-free; `get_model("gemma")` constructs the
    real LabGemmaModel (weights load lazily on first .generate).
    """
    try:
        factory = MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(MODELS)}"
        ) from None
    return factory()


def list_models() -> list[str]:
    """Registered model names, for help text."""
    return sorted(MODELS)


@dataclass(frozen=True)
class Pipeline:
    """Descriptor for one experiment pipeline (the run→eval pair).

    Fields:
      name        : registry key ("plr" | "search").
      description : one-line human summary for help text.
      eval_mode   : the value lab's ``eval --mode`` uses for this pipeline
                    ("attr" for plr, "search" for search) — this is how
                    ``--pipeline`` reconciles with the existing ``--mode`` flag:
                    ``--pipeline plr`` == ``--mode attr``, ``--pipeline search``
                    == ``--mode search``.
      run_fn      : callable that performs the "run" step (re-score / retrieve).
      eval_fn     : callable that performs the "eval" step (score + ledger).

    ``run_fn`` / ``eval_fn`` are thin, lazily-bound accessors (they import their
    target module on call) so importing ``registry`` stays GPU/DB-free.  The
    P2-2 experiment runner dispatches uniformly over these; lab run/eval use
    ``eval_mode`` + ``run_fn`` to route.
    """

    name: str
    description: str
    eval_mode: str
    run_fn: Callable[..., Any]
    eval_fn: Callable[..., Any]


def _plr_run(*args: Any, **kwargs: Any) -> Any:
    from runners.re_score import re_score

    return re_score(*args, **kwargs)


def _plr_eval(*args: Any, **kwargs: Any) -> Any:
    import importlib.util
    import os as _os

    spec = importlib.util.spec_from_file_location(
        "run_eval",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "eval", "run_eval.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.main(*args, **kwargs)


# The lab is PLR-only (2026-07): the text-search pipeline was removed — the
# lab optimizes the PLR prompt; search evaluation lives in core/ir (and the
# cctv-eval oracle skill), where the full stack (embedding + VQA) exists.
PIPELINES: dict[str, Pipeline] = {
    "plr": Pipeline(
        name="plr",
        description="attribute extraction: re_score → run_eval",
        eval_mode="attr",
        run_fn=_plr_run,
        eval_fn=_plr_eval,
    ),
}


def get_pipeline(name: str) -> Pipeline:
    """Return the Pipeline descriptor for a registered name ("plr" | "search")."""
    try:
        return PIPELINES[name]
    except KeyError:
        raise ValueError(
            f"Unknown pipeline {name!r}. Available: {sorted(PIPELINES)}"
        ) from None


def list_pipelines() -> list[str]:
    """Registered pipeline names, for help text."""
    return sorted(PIPELINES)
