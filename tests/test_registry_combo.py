"""P2-1 tests — model/pipeline registry + ledger combination keys.

GPU-free.  Covers:
  1. registry.get_model('mock') / list_models / list_pipelines / get_pipeline.
  2. A mock run→eval writes a ledger record carrying the new combination keys
     (dataset / model / pipeline / prompt_hash) with sensible values, while
     preserving the historical fields.
  3. `lab run --model mock --pipeline plr` exits 0 GPU-free (smoke).
  4. Back-compat: an old ledger record lacking the new keys is still readable
     by run_eval (its _last_ledger diff path tolerates missing keys).
  5. prompt_hash is stable and 12 hex chars.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tiny_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 150), (120, 110, 100)).save(str(path), format="JPEG")


def _make_gender_dataset(tmp_path: Path, oids: list[str]) -> Path:
    ds = tmp_path / "synth"
    (ds / "crops").mkdir(parents=True)
    _write_jsonl(ds / "predictions.jsonl",
                 [{"obj_id": o, "pred": "male", "reason": ""} for o in oids])
    _write_jsonl(ds / "labels.jsonl",
                 [{"obj_id": o, "label": "female"} for o in oids])
    for o in oids:
        _tiny_jpg(ds / "crops" / f"{o}.jpg")
    return ds


# =====================================================================
# 1. Registry API — GPU-free
# =====================================================================


def test_registry_get_model_mock_and_listings() -> None:
    import registry

    m = registry.get_model("mock")
    # Deterministic: same canned YAML regardless of args
    out = m.generate([], None)
    assert "gender:" in out and "target: person" in out

    assert registry.list_models() == ["gemma", "mock"]
    assert registry.list_pipelines() == ["plr"]  # PLR-only lab (search removed 2026-07)

    plr = registry.get_pipeline("plr")
    assert plr.eval_mode == "attr"

    with pytest.raises(ValueError):
        registry.get_model("nope")
    with pytest.raises(ValueError):
        registry.get_pipeline("nope")
    with pytest.raises(ValueError):
        registry.get_pipeline("search")  # removed with the PLR-only refit


def test_get_model_mock_is_gpu_free() -> None:
    """Importing/using the mock model must not pull GPU/DB modules."""
    import registry

    registry.get_model("mock").generate([], None)
    forbidden = {"storage", "psycopg2", "redis"}
    imported = {mod.split(".")[0] for mod in sys.modules}
    assert not (forbidden & imported)


# =====================================================================
# 2. prompt_hash stability
# =====================================================================


def test_prompt_hash_stable_and_short() -> None:
    from evalkit.provenance import prompt_hash

    h1 = prompt_hash()
    h2 = prompt_hash()
    assert h1 == h2
    assert len(h1) == 12
    assert all(c in "0123456789abcdef" for c in h1)


# =====================================================================
# 3. Mock run → eval writes the combination keys into the ledger
# =====================================================================


def test_mock_run_eval_ledger_has_combination_keys(tmp_path: Path) -> None:
    from runners import re_score as rs
    from registry import get_model
    from evalkit.provenance import prompt_hash

    ds = _make_gender_dataset(tmp_path, ["c1", "c2", "c3"])

    # run step (GPU-free mock)
    rs.re_score("gender", get_model("mock"), golden_dir=str(ds))
    preds = _read_jsonl(ds / "predictions.jsonl")
    assert all(r["pred"] == "female" for r in preds)

    # eval step via run_eval.main (importlib, mirrors lab eval attr path)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_eval_combo", str(_LAB_ROOT / "eval" / "run_eval.py"))
    run_eval = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(run_eval)  # type: ignore[union-attr]

    ledger = tmp_path / "ledger.jsonl"
    orig_argv = sys.argv
    sys.argv = [
        "run_eval", "--attribute", "gender", "--golden", str(ds),
        "--version", "mock_v1", "--ledger", str(ledger),
        "--model", "mock", "--pipeline", "plr", "--dataset", str(ds),
        "--date", "2026-07-02T00:00:00",
    ]
    try:
        run_eval.main()
    except SystemExit:
        pass
    finally:
        sys.argv = orig_argv

    rec = _read_jsonl(ledger)[-1]
    # New combination keys
    assert rec["dataset"] == str(ds)
    assert rec["model"] == "mock"
    assert rec["pipeline"] == "plr"
    assert rec["prompt_hash"] == prompt_hash()
    # Preserved historical fields
    for key in ("version", "attribute", "accuracy", "recall", "bias",
                "confusion", "seed_hash", "gemma_repo", "n"):
        assert key in rec, f"historical field {key!r} missing from ledger record"
    assert rec["accuracy"] == pytest.approx(1.0, abs=1e-4)




# =====================================================================
# 4. Back-compat: old ledger records lacking new keys don't break readers
# =====================================================================


def test_run_eval_tolerates_old_ledger_records(tmp_path: Path) -> None:
    """A prior ledger record WITHOUT the new keys must not break the diff path."""
    ds = _make_gender_dataset(tmp_path, ["b1", "b2"])

    ledger = tmp_path / "ledger.jsonl"
    # Old-style record (no dataset/model/pipeline/prompt_hash)
    _write_jsonl(ledger, [{
        "attribute": "gender", "version": "old_v", "n": 2,
        "accuracy": 0.5, "recall": {}, "bias": None, "confusion": {},
        "seed_hash": "", "gemma_repo": "",
    }])

    from runners import re_score as rs
    from registry import get_model
    rs.re_score("gender", get_model("mock"), golden_dir=str(ds))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_eval_bc", str(_LAB_ROOT / "eval" / "run_eval.py"))
    run_eval = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(run_eval)  # type: ignore[union-attr]

    orig_argv = sys.argv
    sys.argv = [
        "run_eval", "--attribute", "gender", "--golden", str(ds),
        "--version", "new_v", "--ledger", str(ledger), "--dataset", str(ds),
    ]
    try:
        run_eval.main()  # must not raise on the old record's missing keys
    except SystemExit:
        pass
    finally:
        sys.argv = orig_argv

    records = _read_jsonl(ledger)
    assert len(records) == 2
    assert "prompt_hash" in records[-1]  # new record enriched


# =====================================================================
# 5. Smoke: lab run --model mock --pipeline plr exits 0 GPU-free
# =====================================================================


def test_lab_run_mock_plr_smoke(tmp_path: Path) -> None:
    ds = _make_gender_dataset(tmp_path, ["m1", "m2"])
    result = subprocess.run(
        [sys.executable, str(_LAB_ROOT / "lab.py"), "run",
         "--model", "mock",
         "--attribute", "gender", "--version", "mock_v1",
         "--dataset", str(ds)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"lab run mock/plr exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    preds = _read_jsonl(ds / "predictions.jsonl")
    assert all(r["pred"] == "female" for r in preds)
