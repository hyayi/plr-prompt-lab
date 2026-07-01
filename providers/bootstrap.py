"""Provider bootstrap for the pluggable-modules layer.

Importing this module triggers the self-registration side-effects of all four
concrete provider modules so that registry.get_provider() works for every slot
before any orchestration code (main.py, indexing.py, image_retrieval.py) calls
it.

The four concrete modules each call registry.register() at import time:

    gemma_backend           → slot "model",   version derived from IR_GEMMA_REPO
    providers.file_prompt_provider → slot "prompt",  version "plr_v1.4_cot" + "plr_v1.3_cot" + "plr_v0.4"
    parser.yaml_parser      → slot "parser",  version "qp_v0.4"
    scoring_provider        → slot "scoring", version "score_v0.5"

Calling bootstrap_providers() once at startup is sufficient.  The function is
idempotent: repeated calls re-import modules (Python caches them) so no extra
registration occurs, only debug-level log messages (registry.py line ~99).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def bootstrap_providers() -> None:
    """Import all four concrete provider modules to trigger self-registration.

    Must be called before the first registry.get_provider() call.  Designed to
    be called from main.py at process startup (before ImageRetrievalService is
    constructed).
    """
    import gemma_backend  # noqa: F401  — registers slot "model"
    import providers.file_prompt_provider  # noqa: F401  — registers slot "prompt"
    import parser.yaml_parser  # noqa: F401  — registers slot "parser"
    import scoring_provider  # noqa: F401  — registers slot "scoring"

    log.info(
        "providers.bootstrap: all four provider slots bootstrapped "
        "(model / prompt / parser / scoring)"
    )
