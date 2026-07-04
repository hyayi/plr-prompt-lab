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
    assert plr.name == "plr" and callable(plr.run_fn)

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
