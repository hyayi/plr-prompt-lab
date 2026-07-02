"""Provider registry for the IR pluggable-modules layer.

Usage pattern (called by A2-A5 at module load time):

    from registry import register, get_provider
    from providers import ModelProvider

    class GemmaModelProvider(ModelProvider):
        ...

    register("model", gemma_model_version(), GemmaModelProvider)

Then anywhere that needs the active provider:

    model = get_provider("model")
    result = model.generate(pil, messages)

The active implementation for each slot is selected by the IR_*_VER config
vars (see config.py):
  - IR_MODEL_VER   -> ModelProvider
  - IR_PROMPT_VER  -> PromptProvider
  - IR_PARSER_VER  -> Parser
  - IR_SCORING_VER -> ScoringStrategy

If the requested version is not registered, the registry falls back to the
first registered implementation for that slot and logs a warning.

Validation
----------
On first resolution of a slot the registry calls _validate_provider() on the
chosen provider instance.  If validation raises, the registry falls back to
the previously-resolved provider (or the first available), logs a warning, and
does NOT propagate the exception so the caller degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from providers import ModelProvider, PromptProvider, Parser, ScoringStrategy

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_SLOT_TYPES: dict[str, type] = {
    "model": ModelProvider,
    "prompt": PromptProvider,
    "parser": Parser,
    "scoring": ScoringStrategy,
}

# slot -> {version_string -> provider_class}
_registry: dict[str, dict[str, Type[Any]]] = {
    "model": {},
    "prompt": {},
    "parser": {},
    "scoring": {},
}

# Resolved singletons: slot -> provider_instance
_singletons: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Registration API (called by A2-A5)
# ---------------------------------------------------------------------------


def register(slot: str, version: str, provider_class: Type[Any]) -> None:
    """Register a provider implementation for a given slot and version tag.

    Called by concrete provider modules (A2-A5) at import time so the
    registry knows which classes are available.

    Parameters
    ----------
    slot           : one of "model" | "prompt" | "parser" | "scoring"
    version        : the version string this class identifies itself with
                     (must match provider_class().version)
    provider_class : a concrete subclass of the matching ABC from providers/

    Raises ValueError for unknown slots or wrong base class.
    """
    if slot not in _SLOT_TYPES:
        raise ValueError(
            f"Unknown slot {slot!r}. Valid slots: {list(_SLOT_TYPES)}"
        )
    base = _SLOT_TYPES[slot]
    if not (isinstance(provider_class, type) and issubclass(provider_class, base)):
        raise ValueError(
            f"{provider_class!r} is not a subclass of {base.__name__}"
        )
    if version in _registry[slot]:
        log.debug(
            "registry: slot=%r version=%r already registered, overwriting",
            slot, version,
        )
    _registry[slot][version] = provider_class
    # Invalidate cached singleton so next get_provider() re-resolves.
    _singletons.pop(slot, None)
    log.debug("registry: registered slot=%r version=%r class=%r",
              slot, version, provider_class.__name__)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _config_ver(slot: str) -> str:
    """Read the active version for a slot from config / environment."""
    import config as _cfg
    mapping = {
        "model":   getattr(_cfg, "IR_MODEL_VER", ""),
        "prompt":  getattr(_cfg, "IR_PROMPT_VER", ""),
        "parser":  getattr(_cfg, "IR_PARSER_VER", ""),
        "scoring": getattr(_cfg, "IR_SCORING_VER", ""),
    }
    return (mapping.get(slot) or "").strip()


def _instantiate(slot: str, version: str) -> Any:
    """Instantiate the provider class for slot+version.  No-arg constructor."""
    cls = _registry[slot][version]
    instance = cls()
    log.info("registry: slot=%r resolved version=%r class=%r",
             slot, version, cls.__name__)
    return instance


def _validate_provider(instance: Any, slot: str) -> None:
    """Lightweight validation — confirm the instance's .version is declared.

    A2-A5 may extend this by registering a validate_<slot>() hook.
    Currently performs a minimal structural check so missing ABC methods
    surface at registry-load time rather than deep inside a search call.
    """
    ver = getattr(instance, "version", None)
    if ver is None:
        raise ValueError(
            f"Provider for slot={slot!r} has no .version property"
        )
    # TODO(A2): ModelProvider — validate that the model file is accessible.
    # TODO(A3): PromptProvider — validate prompt output is non-empty.
    # TODO(A4): Parser — validate parse_plr_response on a known fixture.
    # TODO(A5): ScoringStrategy — validate scoring weights sum to 1.0.


def get_provider(slot: str) -> Any:
    """Return the singleton provider for the given slot.

    Resolution order:
      1. Return cached singleton if already resolved.
      2. Look up IR_*_VER config var for the target version.
      3. If target version is registered, instantiate and validate it.
         On validation failure: log warning, fall back (see step 4).
      4. Fall back to the first registered version for the slot.
      5. If no implementations are registered, raise RuntimeError with a
         clear message pointing to the TODO anchor for that slot.

    Parameters
    ----------
    slot : "model" | "prompt" | "parser" | "scoring"
    """
    if slot not in _SLOT_TYPES:
        raise ValueError(
            f"Unknown slot {slot!r}. Valid slots: {list(_SLOT_TYPES)}"
        )

    if slot in _singletons:
        return _singletons[slot]

    target_ver = _config_ver(slot)
    available = _registry[slot]

    if not available:
        # No concrete providers registered yet.  A2-A5 haven't been imported.
        # Provide an actionable error message.
        _TODO_HINTS = {
            "model":   "A2: import gemma_model_provider to register",
            "prompt":  "A3: import yaml_cot_prompt_provider to register",
            "parser":  "A4: import yaml_parser_provider to register",
            "scoring": "A5: import v05_scoring_provider to register",
        }
        raise RuntimeError(
            f"No provider registered for slot={slot!r}. "
            f"({_TODO_HINTS.get(slot, 'import the concrete provider module')})"
        )

    # Try the configured version first.
    instance: Any = None
    if target_ver and target_ver in available:
        try:
            instance = _instantiate(slot, target_ver)
            _validate_provider(instance, slot)
        except Exception as exc:
            log.warning(
                "registry: validation failed for slot=%r version=%r (%s); "
                "falling back to first available",
                slot, target_ver, exc,
            )
            instance = None

    if instance is None:
        # First registered version (insertion order, Python 3.7+).
        active_ver = next(iter(available))
        if target_ver:
            # A specific version was pinned but is missing/invalid — warn.
            log.warning(
                "registry: slot=%r configured version=%r not usable; "
                "using version=%r",
                slot, target_ver, active_ver,
            )
        else:
            # No version pinned (empty IR_*_VER) — using the active provider
            # is the intended path, not a degraded fallback.
            log.info(
                "registry: slot=%r no version pinned; using active version=%r",
                slot, active_ver,
            )
        instance = _instantiate(slot, active_ver)

    _singletons[slot] = instance
    return instance


def reset_singletons() -> None:
    """Clear all resolved singletons.  Used by tests and reload scenarios."""
    _singletons.clear()


def registered_versions(slot: str) -> list[str]:
    """Return the list of registered version strings for a slot."""
    return list(_registry.get(slot, {}).keys())


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
    from re_score import re_score

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


def _search_run(*args: Any, **kwargs: Any) -> Any:
    from re_score import run_search_over_golden

    return run_search_over_golden(*args, **kwargs)


def _search_eval(*args: Any, **kwargs: Any) -> Any:
    from run_search_eval import main as _main

    return _main(*args, **kwargs)


PIPELINES: dict[str, Pipeline] = {
    "plr": Pipeline(
        name="plr",
        description="attribute extraction: re_score → run_eval (eval --mode attr)",
        eval_mode="attr",
        run_fn=_plr_run,
        eval_fn=_plr_eval,
    ),
    "search": Pipeline(
        name="search",
        description="text retrieval: run_search_over_golden → run_search_eval (eval --mode search)",
        eval_mode="search",
        run_fn=_search_run,
        eval_fn=_search_eval,
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
