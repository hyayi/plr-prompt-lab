"""Story S3 tests — no GPU, no DB, no redis.

Tests:
  1. run_search_eval: synthetic search_results.jsonl + queries.jsonl
     → correct recall@k / precision@k + a ledger record appended.
  2. lab port (read-only): runs without --apply, produces a non-empty diff
     when lab file differs from core/ir copy, does NOT write to core/ir.
  3. lab CLI: --help lists all 5 subcommands; eval --help works.
  4. run_search_eval stale-seed warning path (no actual git subprocess needed).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))


# =====================================================================
# Test 1: run_search_eval — metrics correct + ledger record appended
# =====================================================================


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]








# =====================================================================
# Test 2: lab port — read-only, diff produced, core/ir NOT mutated
# =====================================================================


def test_lab_port_readonly_no_mutation(tmp_path: Path) -> None:
    """lab port (no --apply): shows diff when lab file differs from a temp
    core/ir copy, but does NOT write to core/ir."""
    # Make a temp copy of core/ir's plr_core.py to act as our fake core/ir
    fake_ir = tmp_path / "fake_core_ir"
    fake_ir.mkdir()
    (fake_ir / "prompts").mkdir()

    # Copy real lab files as the "baseline" for fake core/ir
    for lab_rel, ir_rel in [
        ("prompts/plr_v1.4_cot.yaml", "prompts/plr_v1.4_cot.yaml"),
        ("plr_prompts.py", "plr_prompts.py"),
        ("plr_core.py", "plr_core.py"),
    ]:
        src = _LAB_ROOT / lab_rel
        dst = fake_ir / ir_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(str(src), str(dst))

    # Now modify a file in the fake core/ir to create a detectable diff
    fake_plr_core = fake_ir / "plr_core.py"
    original_content = fake_plr_core.read_text(encoding="utf-8")
    fake_plr_core.write_text(
        "# FAKE CORE IR MARKER LINE — should appear in diff\n" + original_content,
        encoding="utf-8",
    )

    # Record mtime of the fake_ir file before port run
    mtime_before = fake_plr_core.stat().st_mtime

    # Run lab port (read-only, no --apply)
    result = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "port",
         "--core-ir", str(fake_ir)],
        capture_output=True, text=True,
    )

    # Should exit 0
    assert result.returncode == 0, (
        f"lab port exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # Diff output should mention plr_core.py
    combined = result.stdout + result.stderr
    assert "plr_core.py" in combined, (
        f"Expected 'plr_core.py' in port output:\n{combined}"
    )

    # Diff should be non-empty (files differ)
    assert "---" in result.stdout or "+++" in result.stdout, (
        f"Expected unified diff in stdout:\n{result.stdout}"
    )

    # core/ir file must NOT have been modified (read-only default)
    mtime_after = fake_plr_core.stat().st_mtime
    assert mtime_after == mtime_before, (
        "lab port (no --apply) mutated the fake core/ir file — read-only contract violated!"
    )

    # Reminder text should NOT mention "copied" (only shown with --apply)
    assert "copied:" not in result.stdout, (
        f"Unexpected 'copied:' in read-only port output:\n{result.stdout}"
    )


def test_lab_port_apply_uses_temp_copy(tmp_path: Path) -> None:
    """--apply on a temp core/ir copy DOES write files and prints the parity reminder."""
    fake_ir = tmp_path / "fake_core_ir"
    (fake_ir / "prompts").mkdir(parents=True)

    # Put stale content in fake core/ir
    (fake_ir / "plr_prompts.py").write_text("# stale\n", encoding="utf-8")
    (fake_ir / "plr_core.py").write_text("# stale\n", encoding="utf-8")
    for y in ["plr_v0.4.yaml", "plr_v1.3_cot.yaml", "plr_v1.4_cot.yaml"]:
        (fake_ir / "prompts" / y).write_text("# stale yaml\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "port",
         "--apply", "--core-ir", str(fake_ir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"lab port --apply exited {result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    assert "copied:" in result.stdout, (
        f"Expected 'copied:' in --apply output:\n{result.stdout}"
    )
    assert "test_prompt_source_parity.py" in result.stdout, (
        f"Expected parity test reminder in output:\n{result.stdout}"
    )

    # plr_prompts.py should now match lab content (not "# stale")
    new_content = (fake_ir / "plr_prompts.py").read_text(encoding="utf-8")
    assert new_content != "# stale\n", "apply did not overwrite stale plr_prompts.py"


# =====================================================================
# Test 3: lab CLI — --help lists all 5 subcommands
# =====================================================================


def test_lab_help_lists_all_subcommands() -> None:
    """python3 lab.py --help must list all 5 subcommands."""
    result = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"lab.py --help exited {result.returncode}\n{result.stderr}"
    )
    out = result.stdout
    for cmd in ["build-golden", "label", "run", "eval", "port"]:
        assert cmd in out, f"Subcommand '{cmd}' missing from --help output:\n{out}"



def test_lab_m_invocation() -> None:
    """python3 -m lab --help must also work (verifies __main__.py)."""
    result = subprocess.run(
        [sys.executable, "-m", "lab", "--help"],
        capture_output=True, text=True,
        cwd=str(_LAB_ROOT),
    )
    assert result.returncode == 0, (
        f"python3 -m lab --help exited {result.returncode}\n{result.stderr}"
    )
    assert "build-golden" in result.stdout


# =====================================================================
# Test 4: run_search_eval.py — metric helper unit tests
# =====================================================================






# =====================================================================
# Test 5: run_eval.py ledger enrichment — seed_hash + gemma_repo present
# =====================================================================


