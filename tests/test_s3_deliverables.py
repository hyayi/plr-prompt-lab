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


def test_search_eval_metrics_and_ledger(tmp_path: Path) -> None:
    """Synthetic 2-query eval:
      query A: relevant=[obj1], ranked=[obj1, obj2, obj3] → hit in top-5
      query B: relevant=[obj3], ranked=[obj1, obj2]       → miss (obj3 not in ranked)

    At k=5:
      query A: recall@5 = 1/1 = 1.0,  precision@5 = 1/5 = 0.2
      query B: recall@5 = 0/1 = 0.0,  precision@5 = 0/5 = 0.0
      mean recall@5 = 0.5,  mean precision@5 = 0.1
    """
    import run_search_eval as rse

    queries = [
        {"query": "query_A", "relevant": ["obj1"]},
        {"query": "query_B", "relevant": ["obj3"]},
    ]
    results = [
        {"query": "query_A", "ranked": ["obj1", "obj2", "obj3", "obj4", "obj5"]},
        {"query": "query_B", "ranked": ["obj1", "obj2"]},
    ]

    q_path = tmp_path / "queries.jsonl"
    r_path = tmp_path / "search_results.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"

    _write_jsonl(q_path, queries)
    _write_jsonl(r_path, results)

    record = rse.main(
        results_path=str(r_path),
        queries_path=str(q_path),
        ledger_path=str(ledger_path),
        version="test_v0.1",
        k=5,
        date="2026-07-01T00:00:00",
        seed_hash="abc123",
        gemma_repo="test/repo",
        core_ir_path=None,  # skip stale-seed subprocess
    )

    # --- metric correctness ---
    assert record["recall_at_k"] == pytest.approx(0.5, abs=1e-4), (
        f"Expected recall@5=0.5, got {record['recall_at_k']}"
    )
    assert record["precision_at_k"] == pytest.approx(0.1, abs=1e-4), (
        f"Expected precision@5=0.1, got {record['precision_at_k']}"
    )
    assert record["n_queries"] == 2
    assert record["k"] == 5
    assert record["attribute"] == "search"
    assert record["version"] == "test_v0.1"

    # --- ledger record appended ---
    assert ledger_path.exists(), "ledger.jsonl was not created"
    ledger_records = _read_jsonl(ledger_path)
    assert len(ledger_records) == 1
    lr = ledger_records[0]
    assert lr["attribute"] == "search"
    assert lr["version"] == "test_v0.1"
    assert lr["seed_hash"] == "abc123"
    assert lr["gemma_repo"] == "test/repo"
    assert lr["recall_at_k"] == pytest.approx(0.5, abs=1e-4)
    assert lr["precision_at_k"] == pytest.approx(0.1, abs=1e-4)


def test_search_eval_perfect_recall(tmp_path: Path) -> None:
    """Single query: all relevant items in top-k → recall=1.0, precision=relevant/k."""
    import run_search_eval as rse

    queries = [{"query": "q1", "relevant": ["a", "b"]}]
    results = [{"query": "q1", "ranked": ["a", "b", "c"]}]

    q_path = tmp_path / "queries.jsonl"
    r_path = tmp_path / "results.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    _write_jsonl(q_path, queries)
    _write_jsonl(r_path, results)

    record = rse.main(
        results_path=str(r_path),
        queries_path=str(q_path),
        ledger_path=str(ledger_path),
        version="v_test",
        k=3,
        date="2026-07-01T00:00:00",
        seed_hash=None,
        gemma_repo=None,
        core_ir_path=None,
    )
    assert record["recall_at_k"] == pytest.approx(1.0, abs=1e-4)
    # 2 hits in top-3 → 2/3
    assert record["precision_at_k"] == pytest.approx(2 / 3, abs=1e-4)


def test_search_eval_ledger_diff(tmp_path: Path) -> None:
    """Two runs: second run shows a Δ against the first in stdout."""
    import run_search_eval as rse
    import io
    from contextlib import redirect_stdout

    q_path = tmp_path / "queries.jsonl"
    r_path = tmp_path / "results.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"

    queries = [{"query": "q1", "relevant": ["obj1"]}]
    _write_jsonl(q_path, queries)

    # First run: ranked list misses
    _write_jsonl(r_path, [{"query": "q1", "ranked": []}])
    rse.main(
        results_path=str(r_path), queries_path=str(q_path),
        ledger_path=str(ledger_path), version="v0",
        k=5, date="2026-07-01T00:00:00", seed_hash=None, gemma_repo=None,
        core_ir_path=None,
    )

    # Second run: ranked list hits
    _write_jsonl(r_path, [{"query": "q1", "ranked": ["obj1"]}])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rse.main(
            results_path=str(r_path), queries_path=str(q_path),
            ledger_path=str(ledger_path), version="v1",
            k=5, date="2026-07-01T00:00:01", seed_hash=None, gemma_repo=None,
            core_ir_path=None,
        )
    output = buf.getvalue()
    # Should show Δ in the output
    assert "Δ vs v0" in output, f"Expected diff line in output:\n{output}"

    # Two records in ledger
    records = _read_jsonl(ledger_path)
    assert len(records) == 2
    assert records[0]["version"] == "v0"
    assert records[1]["version"] == "v1"


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


def test_lab_eval_subhelp() -> None:
    """python3 lab.py eval --help must exit 0 and show mode choices."""
    result = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "eval", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"lab.py eval --help exited {result.returncode}\n{result.stderr}"
    )
    assert "--mode" in result.stdout, (
        f"Expected '--mode' in eval --help:\n{result.stdout}"
    )
    assert "attr" in result.stdout, (
        f"Expected 'attr' in eval --help:\n{result.stdout}"
    )
    assert "search" in result.stdout, (
        f"Expected 'search' in eval --help:\n{result.stdout}"
    )


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


def test_recall_at_k_basic() -> None:
    import run_search_eval as rse

    assert rse._recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == pytest.approx(1.0)
    assert rse._recall_at_k(["a", "b", "c"], {"a", "b"}, k=1) == pytest.approx(0.5)
    assert rse._recall_at_k(["x", "y"], {"a", "b"}, k=5) == pytest.approx(0.0)
    assert rse._recall_at_k([], set(), k=5) == pytest.approx(0.0)


def test_precision_at_k_basic() -> None:
    import run_search_eval as rse

    assert rse._precision_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(2 / 3)
    assert rse._precision_at_k(["a", "b", "c"], {"a", "b"}, k=2) == pytest.approx(1.0)
    assert rse._precision_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)
    assert rse._precision_at_k([], {"a"}, k=0) == pytest.approx(0.0)


# =====================================================================
# Test 5: run_eval.py ledger enrichment — seed_hash + gemma_repo present
# =====================================================================


def test_run_eval_ledger_has_seed_and_gemma(tmp_path: Path) -> None:
    """run_eval.main() appends a ledger record that includes seed_hash and gemma_repo."""
    # Build minimal golden dir
    gdir = tmp_path / "golden" / "gender"
    gdir.mkdir(parents=True)

    preds = [{"obj_id": "p1", "pred": "male", "reason": ""}]
    labels = [{"obj_id": "p1", "true": "male"}]
    _write_jsonl(gdir / "predictions.jsonl", preds)
    _write_jsonl(gdir / "labels.jsonl", labels)

    ledger_path = tmp_path / "ledger.jsonl"

    import importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_eval_mod",
        str(_LAB_ROOT / "eval" / "run_eval.py"),
    )
    run_eval = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(run_eval)  # type: ignore[union-attr]

    orig_argv = sys.argv
    sys.argv = [
        "run_eval",
        "--attribute", "gender",
        "--golden", str(gdir),
        "--version", "test_v",
        "--ledger", str(ledger_path),
    ]
    os.environ["IR_GEMMA_REPO"] = "test/gemma"
    try:
        run_eval.main()
    except SystemExit:
        pass
    finally:
        sys.argv = orig_argv
        del os.environ["IR_GEMMA_REPO"]

    assert ledger_path.exists(), "ledger.jsonl not written by run_eval"
    records = _read_jsonl(ledger_path)
    assert len(records) == 1
    rec = records[0]
    assert "seed_hash" in rec, f"seed_hash missing from ledger record: {rec}"
    assert "gemma_repo" in rec, f"gemma_repo missing from ledger record: {rec}"
    assert rec["gemma_repo"] == "test/gemma"
