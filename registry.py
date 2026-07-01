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
