"""Registries for the lab.

Two registries live here:

1. A minimal provider REGISTER shim (`register`) — the byte-synced modules
   (providers/file_prompt_provider.py, gemma_backend.py) self-register their
   provider classes at import time, exactly as they do in core/ir. The lab
   never RESOLVES providers through a registry (it instantiates
   FilePromptProvider directly with version_override), so core/ir's
   get_provider/validation machinery was dropped in the 2026-07 slim-down —
   this shim only has to accept the registrations.

2. The MODELS / PIPELINES registry — the axes the lab CLI selects over
   (which model, which pipeline).
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
# Model / Pipeline registry
# ---------------------------------------------------------------------------
# A SEPARATE, higher-level registry from the provider slots above: it names the
# axes the lab CLI selects over — WHICH model (gemma vs mock).  Kept in this
# module so there is one obvious place to discover selectable parameters.
#
# `MODELS`    : name -> zero-arg factory returning a gemma_model.Model.
# `PIPELINES` : name -> Pipeline descriptor (the "run" seam lab run routes on).
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
    """Descriptor for one pipeline's "run" step.

    Fields:
      name        : registry key ("plr").
      description : one-line human summary for help text.
      run_fn      : callable that performs the "run" step (re-score).

    Scoring moved to the eval server (RE-004, 2026-07): the lab no longer has a
    local eval step, so ``Pipeline`` carries no ``eval_fn``.  ``run_fn`` is a
    thin, lazily-bound accessor (it imports its target module on call) so
    importing ``registry`` stays GPU/DB-free.  lab run uses ``run_fn`` to route.
    """

    name: str
    description: str
    run_fn: Callable[..., Any]


def _plr_run(*args: Any, **kwargs: Any) -> Any:
    from runners.re_score import re_score

    return re_score(*args, **kwargs)


# The lab is PLR-only (2026-07): the text-search pipeline was removed — the
# lab optimizes the PLR prompt; search evaluation lives in core/ir (and the
# cctv-eval oracle skill), where the full stack (embedding + VQA) exists.
PIPELINES: dict[str, Pipeline] = {
    "plr": Pipeline(
        name="plr",
        description="attribute extraction: re_score (scoring on eval server)",
        run_fn=_plr_run,
    ),
}


def get_pipeline(name: str) -> Pipeline:
    """Return the Pipeline descriptor for a registered name ("plr")."""
    try:
        return PIPELINES[name]
    except KeyError:
        raise ValueError(
            f"Unknown pipeline {name!r}. Available: {sorted(PIPELINES)}"
        ) from None


def list_pipelines() -> list[str]:
    """Registered pipeline names, for help text."""
    return sorted(PIPELINES)
