"""Gemma 4 GGUF backend for ir service (llama-cpp-python + optional speculative decoding).

Ported from gemma4_poc/src/backends/llama_cpp_backend.py with these changes:
  - Single mode: Gemma 4 E4B Q4_0 GGUF (no e2b fallback, no other quantizations)
  - Optional speculative decoding with Gemma 3 1B Q4_0 as draft model
  - Configuration via environment variables (no constructor args needed for normal use)
  - Singleton load via load_backend() so the ir worker keeps one instance

Environment variables:
  IR_GEMMA_REPO          default: unsloth/gemma-4-E4B-it-GGUF
  IR_GEMMA_MAIN_FILE     specific file in repo (autodetect Q4_0 if unset)
  IR_GEMMA_MMPROJ_FILE   specific mmproj file (autodetect if unset)
  IR_GEMMA_N_CTX         context window, default 4096
  IR_GEMMA_N_GPU_LAYERS  GPU offload layers, default -1 (all)
  IR_SPEC_DECODE         "on" enables speculative decoding (default "off")
  IR_DRAFT_REPO          draft model repo, default unsloth/gemma-3-1b-it-GGUF
  IR_DRAFT_MAIN_FILE     specific draft file (autodetect Q4_0 if unset)
  IR_DRAFT_N_GPU_LAYERS  draft GPU layers, default -1
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)


GEMMA_REPO_DEFAULT = "unsloth/gemma-4-E4B-it-GGUF"
DRAFT_REPO_DEFAULT = "unsloth/gemma-3-1b-it-GGUF"


@dataclass
class GenResult:
    """Result of one generate() call. raw is the model output text as-is."""

    raw: str
    input_tokens: int = 0
    output_tokens: int = 0


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _download_gguf(
    repo: str,
    main_hint: str | None,
    mmproj_hint: str | None,
) -> tuple[str, str | None]:
    """Resolve (main_gguf_path, mmproj_gguf_path) for a vision-capable repo.

    Picks the first Q4_0 main file and any mmproj-* file, unless explicit
    hints are provided.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo)
    ggufs = [f for f in files if f.endswith(".gguf")]
    if not ggufs:
        raise RuntimeError(f"No .gguf files in {repo}")

    def _pick(hint: str | None, pattern_fn) -> str | None:
        if hint:
            if hint in ggufs:
                return hint
            raise RuntimeError(f"Requested file {hint!r} not found in {repo}")
        matches = [f for f in ggufs if pattern_fn(f.lower())]
        return matches[0] if matches else None

    main_file = _pick(main_hint, lambda s: "q4_0" in s and "mmproj" not in s)
    if main_file is None:
        raise RuntimeError(
            f"No Q4_0 main file found in {repo}. Set IR_GEMMA_MAIN_FILE."
        )
    mmproj_file = _pick(mmproj_hint, lambda s: "mmproj" in s)
    log.info("GGUF main:   %s :: %s", repo, main_file)
    log.info("GGUF mmproj: %s", mmproj_file or "(none — text-only)")

    main_path = hf_hub_download(repo, main_file)
    mmproj_path = hf_hub_download(repo, mmproj_file) if mmproj_file else None
    return main_path, mmproj_path


def _download_draft(repo: str, main_hint: str | None) -> str:
    """Resolve draft model GGUF path. Text-only (no mmproj for the draft)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo)
    ggufs = [f for f in files if f.endswith(".gguf")]
    if not ggufs:
        raise RuntimeError(f"No .gguf files in draft repo {repo}")
    if main_hint:
        if main_hint not in ggufs:
            raise RuntimeError(f"Requested draft file {main_hint!r} not in {repo}")
        chosen = main_hint
    else:
        q4 = [f for f in ggufs if "q4_0" in f.lower() and "mmproj" not in f.lower()]
        if not q4:
            raise RuntimeError(
                f"No Q4_0 draft file found in {repo}. Set IR_DRAFT_MAIN_FILE."
            )
        chosen = q4[0]
    log.info("Draft GGUF: %s :: %s", repo, chosen)
    return hf_hub_download(repo, chosen)


def _pick_chat_handler(mmproj_path: str | None):
    """Pick the multimodal chat handler bundled with llama-cpp-python."""
    if mmproj_path is None:
        return None
    from llama_cpp import llama_chat_format as fmt

    for name in (
        "Gemma3ChatHandler",
        "Gemma2ChatHandler",
        "Llava16ChatHandler",
        "Llava15ChatHandler",
    ):
        cls = getattr(fmt, name, None)
        if cls is not None:
            log.info("Chat handler: %s", name)
            return cls(clip_model_path=mmproj_path)
    raise RuntimeError(
        "llama-cpp-python has no known VLM chat handler. "
        "Upgrade llama-cpp-python."
    )


def _pil_to_data_url(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _messages_to_oai(
    messages: list[dict[str, Any]], image_url: str | None
) -> list[dict[str, Any]]:
    """Translate our internal message shape to OpenAI-style for llama-cpp-python.

    Image inputs become `image_url` parts when a vision handler is wired up;
    otherwise images are dropped.
    """
    oai = []
    for m in messages:
        role = m["role"]
        content = m.get("content", [])
        if isinstance(content, str):
            oai.append({"role": role, "content": content})
            continue
        parts = []
        for c in content:
            t = c.get("type")
            if t == "text":
                parts.append({"type": "text", "text": c["text"]})
            elif t == "image" and image_url is not None:
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
        oai.append({"role": role, "content": parts})
    return oai


class GemmaBackend:
    """Wraps llama-cpp-python Llama with vision + optional spec decoding.

    Public API:
        backend = GemmaBackend()                  # loads from env
        result  = backend.generate(pil, messages) # PLR/VQA call
        info    = backend.info()
        backend.unload()
    """

    def __init__(
        self,
        repo: str | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        spec_decode: bool | None = None,
        draft_repo: str | None = None,
    ) -> None:
        self.repo = repo or _env("IR_GEMMA_REPO", GEMMA_REPO_DEFAULT)
        self.n_ctx = n_ctx if n_ctx is not None else _env_int("IR_GEMMA_N_CTX", 4096)
        self.n_gpu_layers = (
            n_gpu_layers
            if n_gpu_layers is not None
            else _env_int("IR_GEMMA_N_GPU_LAYERS", -1)
        )
        self.spec_decode = (
            spec_decode if spec_decode is not None else _env_bool("IR_SPEC_DECODE", False)
        )
        self.draft_repo = draft_repo or _env("IR_DRAFT_REPO", DRAFT_REPO_DEFAULT)
        self.draft_n_gpu_layers = _env_int("IR_DRAFT_N_GPU_LAYERS", -1)

        self.main_path, self.mmproj_path = _download_gguf(
            self.repo,
            main_hint=_env("IR_GEMMA_MAIN_FILE"),
            mmproj_hint=_env("IR_GEMMA_MMPROJ_FILE"),
        )
        self.draft_path: Optional[str] = None
        if self.spec_decode:
            self.draft_path = _download_draft(
                self.draft_repo, _env("IR_DRAFT_MAIN_FILE")
            )

        self._load()

    def _load(self) -> None:
        from llama_cpp import Llama

        handler = _pick_chat_handler(self.mmproj_path)
        if handler is None:
            log.warning(
                "%s has no mmproj — running text-only. Image inputs will be dropped.",
                self.repo,
            )

        # When spec decoding is enabled, both models must use the same logits
        # buffer layout — and the multimodal chat handler in llama-cpp-python
        # 0.3.20 sizes scores[] based on logits_all. Empirically logits_all=True
        # plus identical n_batch on both Llama instances avoids the shape
        # mismatch (could not broadcast input array from shape (X,) into (Y,)).
        logits_all = bool(self.spec_decode)
        n_batch = 512

        kwargs: dict[str, Any] = dict(
            model_path=self.main_path,
            chat_handler=handler,
            n_ctx=self.n_ctx,
            n_batch=n_batch,
            n_gpu_layers=self.n_gpu_layers,
            logits_all=logits_all,
            verbose=False,
        )

        if self.spec_decode and self.draft_path:
            # Try to use LlamaPromptLookupDecoding-style speculation via the
            # draft_model argument. Same vocab family is required (e.g. Gemma 4
            # E2B as draft for Gemma 4 E4B).
            draft_llm = Llama(
                model_path=self.draft_path,
                n_ctx=self.n_ctx,
                n_batch=n_batch,
                n_gpu_layers=self.draft_n_gpu_layers,
                logits_all=logits_all,
                verbose=False,
            )
            kwargs["draft_model"] = draft_llm
            log.info("Speculative decoding ENABLED with draft %s (logits_all=True)",
                     self.draft_repo)

        log.info("Loading Gemma backend: %s (n_gpu_layers=%d, n_ctx=%d)",
                 self.main_path, self.n_gpu_layers, self.n_ctx)
        self.llm = Llama(**kwargs)

    def generate(
        self,
        pil_image,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenResult:
        """Run a single chat completion. Returns raw text + token counts.

        `pil_image` may be None — used for text-only calls like the query
        parser. We still go through the multimodal model (so the same
        backend instance serves both query parsing and image VQA), we
        just skip the image attachment.
        """
        image_url = (
            _pil_to_data_url(pil_image)
            if pil_image is not None and self.mmproj_path
            else None
        )
        oai = _messages_to_oai(messages, image_url)
        # Default stop: cut off generation as soon as the JSON object ends.
        # The PLR/Query JSON envelopes always close with "}}", and any prose
        # after that is unwanted padding that costs tokens.
        if stop is None:
            stop = ["\n\n", "\nOutput", "\n```", "\n\nNote", "\n\nThe "]
        resp = self.llm.create_chat_completion(
            messages=oai,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        choice = resp["choices"][0]
        raw = choice["message"]["content"] or ""
        usage = resp.get("usage", {}) or {}
        return GenResult(
            raw=raw,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    def unload(self) -> None:
        try:
            del self.llm
        except AttributeError:
            pass

    def info(self) -> str:
        vision = "vision" if self.mmproj_path else "text-only"
        spec = f"+spec({self.draft_repo})" if self.spec_decode else ""
        return (
            f"{self.repo}{spec} · "
            f"main={os.path.basename(self.main_path)} · {vision}"
        )


_singleton: GemmaBackend | None = None


def load_backend() -> GemmaBackend:
    """Return the process-wide backend, loading on first call."""
    global _singleton
    if _singleton is None:
        _singleton = GemmaBackend()
        log.info("GemmaBackend loaded: %s", _singleton.info())
    return _singleton


def unload_backend() -> None:
    global _singleton
    if _singleton is not None:
        _singleton.unload()
        _singleton = None


# ---------------------------------------------------------------------------
# ModelProvider concrete implementation
# ---------------------------------------------------------------------------


def gemma_model_version(repo: str | None = None) -> str:
    """Stable identifier for the active GGUF model, stamped to
    ir_plr_index.model_ver.

    Derives from IR_GEMMA_REPO so a repo/quant swap (e.g. unsloth PTQ vs
    Google QAT Q4_0) is reflected automatically.  A previously hardcoded
    constant silently mislabelled model_ver whenever the repo pin changed.
    """
    repo = repo or _env("IR_GEMMA_REPO", GEMMA_REPO_DEFAULT)
    return (repo or GEMMA_REPO_DEFAULT).replace("/", "__")


# Captured once at import so registration key and the .version property agree
# (IR_GEMMA_REPO is fixed for the process lifetime).
_ACTIVE_MODEL_VERSION = gemma_model_version()

try:
    from providers import ModelProvider as _ModelProvider

    class GemmaModelProvider(_ModelProvider):
        """ModelProvider that delegates to the GemmaBackend singleton.

        Preserves exact inference semantics: GGUF, IR_GEMMA_REPO,
        IR_GEMMA_N_GPU_LAYERS, and optional speculative decoding are all
        configured via environment variables consumed by GemmaBackend.__init__.
        """

        _VERSION = _ACTIVE_MODEL_VERSION

        @property
        def version(self) -> str:
            return self._VERSION

        def generate(
            self,
            pil_image: Any,
            messages: list[dict[str, Any]],
            max_tokens: int = 512,
            temperature: float = 0.0,
            stop: list[str] | None = None,
        ) -> GenResult:
            return load_backend().generate(
                pil_image,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )

        def unload(self) -> None:
            unload_backend()

        def info(self) -> str:
            backend = _singleton
            if backend is None:
                return f"{self._VERSION} (not yet loaded)"
            return backend.info()

    from registry import register as _register
    _register(slot="model", version=GemmaModelProvider._VERSION, provider_class=GemmaModelProvider)
    log.debug("gemma_backend: registered GemmaModelProvider version=%r", GemmaModelProvider._VERSION)

except ImportError:
    # Heavy deps (providers, registry) not available in lightweight test envs.
    # The provider simply won't be registered; callers that need the registry
    # must ensure the full runtime environment is present.
    log.debug("gemma_backend: skipping ModelProvider registration (import unavailable)")
