"""Model interface for PLR inference — decouples the pure PLR core from the
concrete Gemma backend + scheduler wiring.

This module is intentionally dependency-light: it imports ONLY `scheduler`
(lazily, inside the method) so that `plr_core` / `search_core` can depend on
the `Model` Protocol without transitively pulling storage / psycopg2 / redis.

`SchedulerGemmaModel` wraps an existing GemmaBackend and routes `.generate`
through the SAME `scheduler.scheduled_generate` path the live code uses —
same priority/current_priority ContextVar, same positional args, same
max_tokens / temperature — so wrapping the backend changes nothing at runtime.
It is retained (unused) in the lab for parity; the lab has no scheduler.

`LabGemmaModel` is the lab's DIRECT model: `.generate(messages, image)` calls
`gemma_backend.load_backend().generate(...)` with no scheduler layer, using the
same positional order / max_tokens / temperature the scheduled path used, and
returns `GenResult.raw`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """Thin single-view inference interface used by the pure PLR core.

    A Model turns a (messages, image) pair into the raw model output text.
    Implementations decide how the generation is scheduled / batched; the PLR
    core only needs the raw string back so it can parse it.
    """

    def generate(self, messages: list[dict[str, Any]], image: Any) -> str:
        ...


class SchedulerGemmaModel:
    """Model adapter that preserves the live scheduled-generate path exactly.

    `.generate(messages, image)` funnels through `scheduler.scheduled_generate(
    backend, image, messages, max_tokens=512, temperature=0.0)` — identical to
    the existing `_call_plr_once` call — and returns the `GenResult.raw` string.

    Because `scheduled_generate` reads the `current_priority` ContextVar, the
    priority (PRIORITY_SEARCH default vs PRIORITY_EAGER during indexing) is
    inherited from the caller's context exactly as before; this wrapper adds no
    priority argument of its own.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def generate(self, messages: list[dict[str, Any]], image: Any) -> str:
        # Import lazily so `import gemma_model` never transitively pulls the
        # scheduler's dependencies until an actual generate is requested.
        from scheduler import scheduled_generate

        gen = scheduled_generate(
            self._backend, image, messages, max_tokens=512, temperature=0.0
        )
        return gen.raw


class LabGemmaModel:
    """Direct (scheduler-free) Model for the offline lab.

    `.generate(messages, image)` calls
    `gemma_backend.load_backend().generate(image, messages, max_tokens=512,
    temperature=0.0)` DIRECTLY — same positional order and generation params the
    live scheduled path used, minus the scheduler (the lab has none) — and
    returns the raw model output string (`GenResult.raw`).

    The backend is loaded lazily on first `.generate` via the `load_backend()`
    singleton, so constructing the model (and `import gemma_model`) never loads
    the GGUF weights.
    """

    def generate(self, messages: list[dict[str, Any]], image: Any) -> str:
        # Lazy import: keep `import gemma_model` free of the heavy GPU backend.
        from gemma_backend import load_backend

        gen = load_backend().generate(
            image, messages, max_tokens=512, temperature=0.0
        )
        return gen.raw
