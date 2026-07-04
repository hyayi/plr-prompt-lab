"""Regression: `lab label` must forward correction flags to make_labels.py.

The bug (architect P1 review): argparse.REMAINDER on the `label` subcommand
dropped option-looking tokens that followed the consumed `--dataset` optional,
so `lab label --dataset ds --female-in-male M2` (and the `-- --female-in-male`
form documented in SKILL.md) errored with "unrecognized arguments" and no
correction was recorded. Fixed via parse_known_args routing in main(). These
tests exercise the REAL CLI via subprocess (the defect was in arg parsing).
"""
import json
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent


def _make_dataset(tmp_path):
    ds = tmp_path / "ds"
    (ds / "crops").mkdir(parents=True)
    # two male-predicted tiles; M2's obj is actually female
    (ds / "index_map.json").write_text(json.dumps({"M1": "1001", "M2": "1002"}))
    (ds / "predictions.jsonl").write_text(
        '{"obj_id": "1001", "pred": "male"}\n'
        '{"obj_id": "1002", "pred": "male"}\n'
    )
    return ds


def _run_label(ds, *extra):
    return subprocess.run(
        [sys.executable, "lab.py", "label", "--dataset", str(ds), *extra],
        cwd=LAB, capture_output=True, text=True,
    )


def _labels(ds):
    return {j["obj_id"]: j["label"] for j in (
        json.loads(l) for l in (ds / "labels.jsonl").read_text().splitlines() if l.strip()
    )}


def test_correction_flags_reach_make_labels(tmp_path):
    ds = _make_dataset(tmp_path)
    r = _run_label(ds, "--female-in-male", "M2")
    assert r.returncode == 0, f"lab label failed: {r.stderr}"
    labels = _labels(ds)
    assert labels["1002"] == "female", "M2 correction (female) not recorded"
    assert labels["1001"] == "male", "M1 model prediction not kept"


def test_flag_order_dataset_first_and_last(tmp_path):
    # --dataset before the correction flag (the form that used to fail)
    ds = _make_dataset(tmp_path)
    r = subprocess.run(
        [sys.executable, "lab.py", "label", "--female-in-male", "M2", "--dataset", str(ds)],
        cwd=LAB, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"flags-first form failed: {r.stderr}"
    assert _labels(ds)["1002"] == "female"


def test_double_dash_separator_form_is_tolerated(tmp_path):
    # the exact `-- --female-in-male` form SKILL.md used to document
    ds = _make_dataset(tmp_path)
    r = _run_label(ds, "--", "--female-in-male", "M2")
    assert r.returncode == 0, f"`--` separator form failed: {r.stderr}"
    assert _labels(ds)["1002"] == "female"


def test_no_corrections_keeps_all_predictions(tmp_path):
    ds = _make_dataset(tmp_path)
    r = _run_label(ds)
    assert r.returncode == 0, r.stderr
    labels = _labels(ds)
    assert labels == {"1001": "male", "1002": "male"}


def test_non_label_command_rejects_extras(tmp_path):
    # main()'s parse_known_args routing must NOT silently swallow stray flags
    # for other subcommands.
    r = subprocess.run(
        [sys.executable, "lab.py", "run", "--version", "v", "--attribute", "gender", "--bogus-flag"],
        cwd=LAB, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "unrecognized arguments" in (r.stderr + r.stdout)
