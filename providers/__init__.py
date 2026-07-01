"""Provider interfaces for the IR pluggable-modules layer.

Four ABCs define the stable contracts that concrete providers (A2-A5) must
implement.  The registry (registry.py) resolves the active provider per slot
from config and hands callers a singleton that matches these interfaces.

Versioning contract
-------------------
Every provider exposes a ``version`` property so storage.py can record which
implementation produced each indexed row.  Version strings follow the existing
conventions found in the codebase:
  - ModelProvider   : e.g. "gemma4_e4b_q4_0"
  - PromptProvider  : e.g. "plr_v1.3_cot", "plr_v0.4"
  - Parser          : e.g. "qp_v0.4"
  - ScoringStrategy : e.g. "score_v0.5"

These strings are written to ir_plr_index.{model_ver, prompt_ver,
parser_ver, scoring_ver} — see storage.py for the column names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# GenResult re-export (lazy, cycle-free)
# ---------------------------------------------------------------------------
# A2 (ModelProvider) returns the same GenResult that gemma_backend produces,
# and we re-export it here so callers can `from providers import GenResult`.
#
# gemma_backend imports ModelProvider from this package, so importing
# gemma_backend at module top-level would create a circular import: whichever
# module loaded first would observe the other half-initialised, and (because
# `from __future__ import annotations` makes all GenResult annotations lazy
# strings) the only fallout was a silently-swallowed ImportError that skipped
# model registration depending on import order.
#
# The cycle is removed structurally: GenResult is only referenced at runtime
# when something actually accesses `providers.GenResult`, resolved on demand
# via module __getattr__ (PEP 562).  For type checkers, the import lives in a
# TYPE_CHECKING block.  Import order no longer affects registration.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from gemma_backend import GenResult


def __getattr__(name: str) -> Any:
    """Lazily resolve re-exported names without an import-time cycle."""
    if name == "GenResult":
        try:
            from gemma_backend import GenResult as _GenResult

            return _GenResult
        except ImportError:
            # Graceful fallback for envs without the backend installed
            # (unit-test / lint contexts).
            from dataclasses import dataclass

            @dataclass
            class GenResult:  # type: ignore[no-redef]
                """Minimal stub matching gemma_backend.GenResult."""

                raw: str
                input_tokens: int = 0
                output_tokens: int = 0

            return GenResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ===========================================================================
# 1. ModelProvider
# ===========================================================================


class ModelProvider(ABC):
    """Wraps a generative model backend (VLM / LLM).

    The concrete Gemma GGUF implementation lives in gemma_backend.py
    (GemmaModelProvider) and self-registers via registry.register().
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Short identifier string for this model variant.

        Written to ir_plr_index.model_ver on every upsert.
        Example: "gemma4_e4b_q4_0", "gemma4_e4b_qat_q4_0".
        """

    @abstractmethod
    def generate(
        self,
        pil_image: Any,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenResult:
        """Run a single chat completion.

        Parameters match GemmaBackend.generate() exactly:
          pil_image  : PIL.Image or None (text-only calls pass None)
          messages   : OpenAI-style message list built by a PromptProvider
          max_tokens : upper token budget for the response
          temperature: sampling temperature (0.0 = greedy)
          stop       : stop sequences; provider may set its own defaults

        Returns GenResult(raw, input_tokens, output_tokens).
        """

    @abstractmethod
    def unload(self) -> None:
        """Release GPU / memory resources. Called on graceful shutdown."""

    @abstractmethod
    def info(self) -> str:
        """Human-readable description of the loaded model (for logs/health)."""


# ===========================================================================
# 2. PromptProvider
# ===========================================================================


class PromptProvider(ABC):
    """Builds chat message lists for PLR extraction and query parsing.

    Concrete implementations (FilePromptProvider) load templates from
    prompts/<version>.yaml and self-register the YAML-CoT and JSON v0.4
    variants. Method signatures mirror the original plr_prompts.py builders.
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Prompt version string written to ir_plr_index.prompt_ver.

        Examples: "plr_v1.3_cot", "plr_v0.5_yaml", "plr_v0.4".
        Must be <=16 chars to fit the varchar(16) column.
        """

    @abstractmethod
    def build_plr_messages(
        self,
        object_hint: str = "person",
    ) -> list[dict[str, Any]]:
        """Build the VLM chat messages for one PLR extraction call.

        Parameters:
          object_hint : 'person' or 'vehicle' — selects the user template.

        Returns an OpenAI-style message list.  The caller (indexing.py)
        passes this list to ModelProvider.generate(pil, messages).
        """

    @abstractmethod
    def build_freeform_vqa_messages(
        self,
        residue: list[dict[str, Any]] | list[str] | str,
    ) -> list[dict[str, Any]]:
        """Build yes/no VQA messages for free-form residue verification.

        Parameters:
          residue : list of {subject, attribute_ko, attribute_en, is_negative}
                    dicts (qp_v0.6+), or legacy list[str]/str.

        Returns an OpenAI-style message list with max_tokens=4 semantics
        (caller should pass max_tokens=4 to ModelProvider.generate).
        """

    @abstractmethod
    def build_plr_retry_messages(
        self,
        object_hint: str,
        original_response: str,
        error_reason: str,
    ) -> list[dict[str, Any]]:
        """Build retry messages when first PLR response fails parse/schema.

        Parameters:
          object_hint       : 'person' or 'vehicle'
          original_response : the raw model output that failed
          error_reason      : short description of the failure

        Returns a message list that includes the failed output + correction
        instruction.
        """

    @abstractmethod
    def build_query_parser_messages(
        self,
        user_query: str,
    ) -> list[dict[str, Any]]:
        """Build chat messages for a query-parser call.

        Parameters:
          user_query : raw Korean (or mixed) user query string.

        Returns an OpenAI-style message list.  The caller (query_parser.py)
        passes this to ModelProvider.generate(None, messages).
        """


# ===========================================================================
# 3. Parser
# ===========================================================================


class Parser(ABC):
    """Converts raw model text output into structured Python dicts.

    The concrete YamlParser wraps parse_plr_response() / parse_query()
    from plr_prompts.py and query_parser.py and loads its surface-form
    synonym tables from parser/<version>.yaml.
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Parser version string written to ir_plr_index.parser_ver.

        Example: "qp_v0.4".
        """

    @abstractmethod
    def parse_plr_response(
        self,
        raw: str,
        hint: str = "person",
        *,
        fmt: str | None = None,
    ) -> dict[str, Any]:
        """Parse a raw PLR model response into a validated dict.

        Parameters:
          raw  : raw text from ModelProvider.generate()
          hint : 'person' or 'vehicle' (object type hint for YAML path)
          fmt  : format override ('yaml' | 'json'); None = use env default

        Returns a PLR dict conforming to plr_schema.PERSON_SCHEMA or
        plr_schema.VEHICLE_SCHEMA.
        Raises ValueError on unrecoverable parse failure.
        """

    @abstractmethod
    def parse_query(
        self,
        user_text: str,
        *,
        backend: Any = None,
        force_gemma: bool = False,
    ) -> Any:
        """Parse a user query string into a QueryJSON.

        Parameters:
          user_text   : raw Korean (or mixed) user query
          backend     : optional ModelProvider instance (or legacy
                        GemmaBackend); if None, uses dictionary-only path
          force_gemma : skip dictionary fallback (used in tests)

        Returns a query_parser.QueryJSON instance.
        """


# ===========================================================================
# 4. ScoringStrategy
# ===========================================================================


class ScoringStrategy(ABC):
    """Computes candidate scores and applies hard / strict filters.

    The concrete DefaultScoringStrategy (scoring_provider.py) delegates
    1-for-1 to the scoring.py functions, preserving the scoring formula.
    """

    @property
    @abstractmethod
    def version(self) -> str:
        """Scoring version string written to ir_plr_index.scoring_ver.

        Example: "score_v0.5".
        """

    @abstractmethod
    def final_score(
        self,
        *,
        query_required: dict[str, list[str]],
        query_excluded: dict[str, list[str]] | None,
        candidate_plr: dict[str, Any],
        candidate_embedding: Any,
        candidate_quality: float,
        q_template_emb: Any,
        q_raw_emb: Any,
        is_vehicle: bool,
    ) -> Any:
        """Compute the full ScoreBreakdown for one candidate.

        Parameters and semantics match scoring.final_score() exactly.
        Returns a scoring.ScoreBreakdown instance.
        """

    @abstractmethod
    def passes_hard_filter(
        self,
        query_required: dict[str, list[str]],
        candidate_plr: dict[str, Any],
        query_excluded: dict[str, list[str]] | None = None,
    ) -> bool:
        """Two-stage cascade gate.  Returns False to reject a candidate.

        Parameters and semantics match scoring.passes_hard_filter() exactly.
        """

    @abstractmethod
    def passes_strict_filter(
        self,
        query_required: dict[str, list[str]],
        candidate_plr: dict[str, Any],
        is_vehicle: bool,
    ) -> bool:
        """Strict mode: every required slot must score > 0.

        Parameters and semantics match scoring.passes_strict_filter() exactly.
        """

    @abstractmethod
    def attribute_match(
        self,
        query_required: dict[str, list[str]],
        query_excluded: dict[str, list[str]] | None,
        candidate_plr: dict[str, Any],
        is_vehicle: bool,
    ) -> tuple[float, dict[str, float]]:
        """Slot-by-slot match score for diagnostics.

        Returns (score_in_0_1, per_slot_contribution_dict).
        Parameters and semantics match scoring.attribute_match() exactly.
        """


__all__ = [
    "GenResult",
    "ModelProvider",
    "PromptProvider",
    "Parser",
    "ScoringStrategy",
]
