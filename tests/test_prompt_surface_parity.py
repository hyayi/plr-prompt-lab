"""Lab-side prompt source-parity gate.

Mirror of core/ir's tests/test_prompt_source_parity.py so the lab enforces the
same invariant WITHOUT needing `lab port` to be run manually: the prompt lives
in TWO sources that must stay byte-identical mirrors —
  (a) plr_prompts.py string constants (the default / live-parity path), and
  (b) prompts/<version>.yaml loaded by providers.file_prompt_provider (what
      `lab run --version <V>` actually sends).

Drift between them is exactly the bug class this repo exists to prevent: an
edit to only one source makes `--version` comparisons silently measure the
wrong prompt.

Coverage beyond the core/ir original: the query_parser block (search prompt) is
also parity-checked — the lab wires it through `--version` (search pipeline),
so its yaml copy must mirror the constants too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

plr_prompts = pytest.importorskip("plr_prompts")
fpp = pytest.importorskip("providers.file_prompt_provider")


@pytest.fixture(autouse=True)
def _restore_plr_env():
    """These tests mutate IR_PLR_FORMAT / IR_PLR_REASON to drive the builder.
    Restore them afterwards so we don't leak env into sibling tests."""
    saved = {k: os.environ.get(k) for k in ("IR_PLR_FORMAT", "IR_PLR_REASON")}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _build_constants(fmt: str, reason: str, hint: str):
    os.environ["IR_PLR_FORMAT"] = fmt
    os.environ["IR_PLR_REASON"] = reason
    return plr_prompts.build_plr_messages(hint)


def _build_provider(version: str, fmt: str, reason: str, hint: str):
    os.environ["IR_PLR_FORMAT"] = fmt
    os.environ["IR_PLR_REASON"] = reason
    return fpp.FilePromptProvider(version_override=version).build_plr_messages(hint)


# (version, fmt, reason) tuples with a matching yaml in prompts/. Same rows as
# core/ir: plr_v1.5_cot (forced-commit) is the live tuple and MUST pass for
# both reason modes; older rows document historical drift (strict xfail).
_COMPARABLE = [
    ("plr_v1.5_cot", "yaml", "on"),   # the LIVE production tuple — forced-commit
    ("plr_v1.5_cot", "yaml", "off"),  # plain (no-reason) template, same contract
    pytest.param(
        "plr_v1.4_cot", "yaml", "on",
        marks=pytest.mark.xfail(reason="superseded by plr_v1.5_cot forced-commit; yaml keeps the historical prompt", strict=True),
    ),
    pytest.param(
        "plr_v1.3_cot", "yaml", "on",
        marks=pytest.mark.xfail(reason="superseded; historical drift", strict=True),
    ),
    pytest.param(
        "plr_v0.4", "json", "off",
        marks=pytest.mark.xfail(reason="legacy JSON constants/yaml diverged; v1.5 is the live target", strict=True),
    ),
]


# plr_v1.5_cot forced-commit gate (mirror of core/ir): `unknown` must never be
# OFFERED as an answer option. The word may still appear inside the "never
# answer unknown" instruction, so we check option-shaped patterns only.
_OFFERED_UNKNOWN_PATTERNS = ("|unknown", "unknown|", "_unknown,", ", unknown", "unknown>", "pick the \"_unknown\"")


@pytest.mark.parametrize("reason", ["on", "off"])
@pytest.mark.parametrize("hint", ["person", "vehicle"])
def test_live_prompt_never_offers_unknown(reason, hint):
    msgs = _build_constants("yaml", reason, hint)
    text = "\n".join(
        chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        for m in msgs
        for chunk in (m["content"] if isinstance(m["content"], list) else [{"text": m["content"]}])
    )
    hits = [p for p in _OFFERED_UNKNOWN_PATTERNS if p in text]
    assert not hits, (
        f"forced-commit violated: unknown offered as an option in the live "
        f"prompt (hint={hint} reason={reason}): {hits}"
    )


@pytest.mark.parametrize("version,fmt,reason", _COMPARABLE)
@pytest.mark.parametrize("hint", ["person", "vehicle"])
def test_constants_equal_yaml(version, fmt, reason, hint):
    const_msgs = _build_constants(fmt, reason, hint)
    yaml_msgs = _build_provider(version, fmt, reason, hint)
    assert const_msgs == yaml_msgs, (
        f"DRIFT: plr_prompts constants != prompts/{version}.yaml "
        f"for hint={hint} fmt={fmt} reason={reason}"
    )


# All three yaml versions currently carry the SAME query_parser block as the
# constants (qp_v0.4 — verified empirically at wiring time). If a future
# version legitimately forks the search prompt, drop that version from this
# list and record the fork in the version yaml's comment header.
_QP_VERSIONS = ["plr_v0.4", "plr_v1.3_cot", "plr_v1.4_cot", "plr_v1.5_cot"]


@pytest.mark.parametrize("version", _QP_VERSIONS)
def test_query_parser_constants_equal_yaml(version):
    query = "빨간 옷 입은 남자"
    const_msgs = plr_prompts.build_query_parser_messages(query)
    yaml_msgs = fpp.FilePromptProvider(
        version_override=version
    ).build_query_parser_messages(query)
    assert const_msgs == yaml_msgs, (
        f"DRIFT: query_parser constants != prompts/{version}.yaml query_parser block"
    )
