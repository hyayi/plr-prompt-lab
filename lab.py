#!/usr/bin/env python3
"""lab — single argparse entrypoint for the PLR prompt lab.

Subcommands:
  build-golden  Build golden eval set (wraps eval/build_golden.py) and
                copy crops to the authoritative eval/golden/<A>/crops/ path.
  label         Turn human misclassification notes into labels.jsonl
                (wraps eval/make_labels.py).
  run           Re-score golden set with Gemma (wraps re_score.re_score).
  eval          Score predictions vs golden labels (PLR attribute eval).
  port          Diff / apply lab prompt surface against core/ir (read-only by
                default; --apply copies lab files into core/ir).
  demo          GPU-free onboarding: run a full mock cycle on a synthetic
                dataset so a new user can see the loop end-to-end immediately.
  report        Turn the eval ledger into ONE self-contained HTML report
                (trends, model×prompt matrix, prompt-change→metric-delta).

Usage:
    python3 lab.py <cmd> [options]
    python3 -m lab  <cmd> [options]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add lab root to path so sibling modules are importable.
_LAB_ROOT = os.path.dirname(os.path.abspath(__file__))
_LAB_ROOT_PATH = Path(_LAB_ROOT)
if _LAB_ROOT not in sys.path:
    sys.path.insert(0, _LAB_ROOT)

def _resolve_core_ir(override: str | None = None) -> str | None:
    """Resolve the core/ir repo path without any ziovision-specific default.

    Precedence:
      1. explicit override (e.g. --core-ir)
      2. CORE_IR_PATH env var
      3. a generic relative guess: ../ziomilitary/core/ir next to this repo
         (only if it exists)
      4. None — the caller decides whether that is fatal.
    """
    if override:
        return override
    env = os.environ.get("CORE_IR_PATH")
    if env:
        return env
    guess = os.path.abspath(os.path.join(_LAB_ROOT, "..", "ziomilitary", "core", "ir"))
    if os.path.isdir(guess):
        return guess
    return None


def _require_core_ir(override: str | None = None) -> str:
    """Like _resolve_core_ir but raise a clear error when it cannot be found.

    Used by commands that genuinely need core/ir (port)."""
    path = _resolve_core_ir(override)
    if not path:
        raise SystemExit(
            "core/ir path not set. Set the CORE_IR_PATH environment variable "
            "or pass --core-ir /path/to/core/ir."
        )
    return path

# =====================================================================
# Subcommand: build-golden
# =====================================================================


def _cmd_build_golden(args: argparse.Namespace) -> int:
    """Call eval/build_golden.py's builder then copy crops to the authoritative
    eval/golden/<attribute>/crops/<obj_id>.jpg location."""
    import importlib.util
    import shutil

    from evalkit.dataset import resolve_dataset_dir

    # Load build_golden as a module (avoids sys.argv interference).
    spec = importlib.util.spec_from_file_location(
        "build_golden",
        os.path.join(_LAB_ROOT, "eval", "build_golden.py"),
    )
    bg = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(bg)  # type: ignore[union-attr]

    gdir = str(resolve_dataset_dir(_LAB_ROOT, args.attribute, getattr(args, "dataset", None)))

    # Inject sys.argv so build_golden.main() parses the right args.
    orig_argv = sys.argv
    sys.argv = [
        "build_golden",
        "--video", args.video,
        "--attribute", args.attribute,
        "--out", gdir,
    ]
    if args.per_class:
        sys.argv += ["--per-class", str(args.per_class)]
    if args.review_dir:
        sys.argv += ["--review-dir", args.review_dir]

    try:
        bg.main()
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    finally:
        sys.argv = orig_argv

    # ---- Ensure crops land at the authoritative bare path ----
    # build_golden copies crops to review_dir with decorated names
    # (e.g. M1__male__obj_id.jpg).  We also copy to the bare path
    # <dataset>/crops/<obj_id>.jpg so re_score can find them.
    crops_dst = os.path.join(gdir, "crops")
    os.makedirs(crops_dst, exist_ok=True)

    review = args.review_dir or os.path.join(
        os.path.expanduser("~"), f"{args.attribute}_eval"
    )

    import json
    index_map_path = os.path.join(gdir, "index_map.json")
    if os.path.exists(index_map_path):
        index_map: dict[str, str] = json.load(open(index_map_path))
        # index_map: {tile -> obj_id}
        result_path = os.environ.get("RESULT_PATH", "./results")
        video_crops = os.path.join(result_path, args.video, "objects")
        for tile, obj_id in index_map.items():
            dst = os.path.join(crops_dst, f"{obj_id}.jpg")
            if os.path.exists(dst):
                continue  # already there
            # Try source: video objects dir first, then review dir
            src = os.path.join(video_crops, f"{obj_id}.jpg")
            if not os.path.exists(src):
                # Fall back: find the decorated review copy
                for fname in os.listdir(review) if os.path.isdir(review) else []:
                    if f"__{obj_id}.jpg" in fname:
                        src = os.path.join(review, fname)
                        break
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  crops/{obj_id}.jpg -> {dst}")
            else:
                print(
                    f"  WARNING: no source crop found for obj_id={obj_id!r}",
                    file=sys.stderr,
                )

    print(f"  authoritative crops dir: {crops_dst}")
    return 0


# =====================================================================
# Subcommand: label
# =====================================================================


def _cmd_label(args: argparse.Namespace) -> int:
    """Thin wrapper around eval/make_labels.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_labels",
        os.path.join(_LAB_ROOT, "eval", "make_labels.py"),
    )
    ml = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(ml)  # type: ignore[union-attr]

    orig_argv = sys.argv
    sys.argv = ["make_labels"]
    # When --dataset is given, point make_labels.py at the dataset's files
    # unless the user already passed the corresponding flag verbatim.
    forwarded = list(getattr(args, "label_args", []))
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]  # drop an optional argparse separator
    if getattr(args, "dataset", None):
        ds = Path(args.dataset)
        if "--index-map" not in forwarded:
            sys.argv += ["--index-map", str(ds / "index_map.json")]
        if "--pred" not in forwarded:
            sys.argv += ["--pred", str(ds / "predictions.jsonl")]
        if "--out" not in forwarded:
            sys.argv += ["--out", str(ds / "labels.jsonl")]
    # Forward any remaining args (argparse remainder or explicit flags).
    sys.argv += forwarded
    try:
        ml.main()
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    finally:
        sys.argv = orig_argv
    return 0


# =====================================================================
# Subcommand: run
# =====================================================================


def _cmd_run(args: argparse.Namespace) -> int:
    """Set env for version, then re_score the golden set (PLR pipeline).

    This is the GPU step at real runtime — the CLI wires it but does not
    import Gemma at module level.

    --model selects the model via the registry (default 'gemma'; 'mock' is
    GPU-free). The lab is PLR-only: the text-search pipeline was removed
    (2026-07 — the lab optimizes the PLR prompt; search eval lives in
    core/ir / cctv-eval).
    """
    from runners import re_score as rs
    from evalkit.dataset import resolve_dataset_dir
    from registry import get_model

    model_name = getattr(args, "model", "gemma")
    attribute = args.attribute
    ds_dir = resolve_dataset_dir(_LAB_ROOT, attribute, getattr(args, "dataset", None))

    # get_model('mock') is GPU-free; 'gemma' constructs LabGemmaModel (weights
    # load lazily on first .generate).
    model = get_model(model_name)
    print(f"[run] re_score attribute={attribute!r} version={args.version!r} "
          f"model={model_name!r} dataset={str(ds_dir)!r}")
    meta = rs.re_score(
        attribute, model, golden_dir=str(ds_dir), prompt_version=args.version
    )
    print(f"[run] re_score done: {meta}")
    return 0


# =====================================================================
# Subcommand: eval
# =====================================================================


def _cmd_eval(args: argparse.Namespace) -> int:
    """Score predictions vs golden labels (PLR attribute eval → eval/run_eval.py)."""
    import importlib.util

    from evalkit.dataset import declared_attributes, resolve_dataset_dir

    model_name = getattr(args, "model", "gemma")

    spec = importlib.util.spec_from_file_location(
        "run_eval",
        os.path.join(_LAB_ROOT, "eval", "run_eval.py"),
    )
    re_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(re_mod)  # type: ignore[union-attr]

    # --attribute all → manifest 선언 속성 전부 순회, "a,b" → 나열 순회.
    # 모델 재실행 없음: 각 속성 예측은 attributes.jsonl(전체 plr_json 캐시)에서
    # run_eval이 재추출한다 — 라벨 1회 + 모델 1회로 전 속성 평가.
    requested = str(args.attribute)
    if requested == "all":
        if not getattr(args, "dataset", None):
            print("[eval] --attribute all requires --dataset", file=sys.stderr)
            return 2
        attributes = declared_attributes(args.dataset)
        if not attributes:
            print(f"[eval] {args.dataset}: manifest.yaml에 attributes:/attribute "
                  f"선언이 없어 'all'을 해석할 수 없습니다", file=sys.stderr)
            return 2
    else:
        attributes = [a.strip() for a in requested.split(",") if a.strip()]

    failures: list[str] = []
    for attribute in attributes:
        ds_dir = resolve_dataset_dir(_LAB_ROOT, attribute, getattr(args, "dataset", None))
        orig_argv = sys.argv
        sys.argv = ["run_eval", "--attribute", attribute,
                    "--golden", str(ds_dir),
                    "--model", model_name, "--pipeline", "plr",
                    "--dataset", str(ds_dir)]
        if args.version:
            sys.argv += ["--version", args.version]
        if args.ledger:
            sys.argv += ["--ledger", args.ledger]
        if getattr(args, "core_ir", None):
            sys.argv += ["--core-ir", args.core_ir]
        try:
            re_mod.main()
        except SystemExit as e:
            code = int(e.code) if isinstance(e.code, int) else 1
            if code:
                failures.append(f"{attribute}: {e.code}")
        except Exception as exc:  # noqa: BLE001 — 한 속성 실패가 나머지를 막지 않게
            failures.append(f"{attribute}: {type(exc).__name__}: {exc}")
        finally:
            sys.argv = orig_argv
        if len(attributes) > 1:
            print()

    if failures:
        print(f"[eval] {len(failures)}/{len(attributes)} attribute(s) failed: "
              + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


# =====================================================================
# Subcommand: port
# =====================================================================

# True prompt surface: files to compare between lab and core/ir. Built from the
# single source of truth (provenance.surface_relpaths) so `lab port` and the
# ledger prompt_hash always cover the SAME files. Same relative path both sides;
# globs prompts/*.yaml so new prompt versions are picked up automatically.
def _port_files() -> list[tuple[str, str]]:
    from evalkit import provenance
    return [(rel, rel) for rel in provenance.surface_relpaths(_LAB_ROOT)]


def _cmd_port(args: argparse.Namespace) -> int:
    """Diff (or apply) the lab prompt surface against core/ir.

    Read-only by default — print a unified diff between lab copy and core/ir
    of: prompts/*.yaml, plr_prompts.py, plr_core.py.
    --apply copies the lab versions into core/ir and prints a reminder to
    run core/ir/tests/test_prompt_source_parity.py.
    """
    import difflib
    import shutil

    core_ir = _require_core_ir(getattr(args, "core_ir", None))

    # Warn if stale seed
    from evalkit import provenance
    seed_hash = provenance.read_seed_hash(_LAB_ROOT)
    provenance.warn_stale_seed(_LAB_ROOT, seed_hash, core_ir)

    attribute_filter = getattr(args, "attribute", None)  # optional, currently unused
    apply_mode = getattr(args, "apply", False)

    all_diffs: list[str] = []

    for lab_rel, ir_rel in _port_files():
        lab_path = os.path.join(_LAB_ROOT, lab_rel)
        ir_path = os.path.join(core_ir, ir_rel)

        lab_lines = _read_lines(lab_path)
        ir_lines = _read_lines(ir_path)

        diff = list(difflib.unified_diff(
            ir_lines, lab_lines,
            fromfile=f"core/ir/{ir_rel}",
            tofile=f"lab/{lab_rel}",
        ))

        if diff:
            all_diffs.extend(diff)
            print(f"--- diff: {lab_rel} ---")
            print("".join(diff), end="")
        else:
            print(f"--- identical: {lab_rel} ---")

    if not all_diffs:
        print("\n[port] lab surface is identical to core/ir — nothing to apply.")
        return 0

    if not apply_mode:
        print("\n[port] (read-only) Re-run with --apply to copy lab files into core/ir.")
        return 0

    # --apply: copy lab files into core/ir
    print("\n[port] --apply: copying lab files into core/ir …")
    for lab_rel, ir_rel in _port_files():
        lab_path = os.path.join(_LAB_ROOT, lab_rel)
        ir_path = os.path.join(core_ir, ir_rel)
        if os.path.exists(lab_path):
            os.makedirs(os.path.dirname(ir_path), exist_ok=True)
            shutil.copy2(lab_path, ir_path)
            print(f"  copied: lab/{lab_rel} -> core/ir/{ir_rel}")
        else:
            print(f"  WARNING: lab file not found: {lab_path}", file=sys.stderr)

    print(
        "\n[port] Files applied. Reminder: run the parity test in core/ir:\n"
        "  python3 -m pytest core/ir/tests/test_prompt_source_parity.py -q\n"
        "  (path: {core_ir}/tests/test_prompt_source_parity.py)".format(
            core_ir=core_ir
        )
    )
    return 0


def _read_lines(path: str) -> list[str]:
    """Read file lines for diffing; return empty list if file does not exist."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


# =====================================================================
# Argument parser
# =====================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lab",
        description="PLR prompt lab CLI — build, label, run, eval, port.",
    )
    sub = p.add_subparsers(dest="cmd", metavar="<cmd>")
    sub.required = True

    # -- build-golden --
    bg = sub.add_parser(
        "build-golden",
        help="Build golden eval set for an attribute from an indexed video.",
    )
    bg.add_argument("--video", "-V", required=True, help="video ID (vd_..._xxxx)")
    bg.add_argument("--attribute", "-A", required=True,
                    choices=["gender", "vehicle_type", "military"],
                    help="PLR attribute to evaluate")
    bg.add_argument("--per-class", type=int, default=None,
                    help="cap per predicted class (default: 50)")
    bg.add_argument("--review-dir", default=None,
                    help="browsable crop dir (default: ~/<attr>_eval)")
    bg.add_argument("--dataset", default=None,
                    help="dataset dir to write into "
                         "(default: eval/golden/<attribute>)")

    # -- label --
    la = sub.add_parser(
        "label",
        help="Turn human misclassification notes into labels.jsonl.",
    )
    la.add_argument("--dataset", default=None,
                    help="dataset dir (default: eval/golden/gender). Supplies "
                         "--index-map/--pred/--out to make_labels.py.")
    # Correction flags (--female-in-male M3,M7 / --male-in-female F2 / --unknown M40)
    # are NOT declared here — they are captured as extras by parse_known_args in
    # main() and forwarded verbatim to make_labels.py, so any flag order works.

    # -- run --
    ru = sub.add_parser(
        "run",
        help="Re-score golden set with Gemma (GPU step).",
    )
    ru.add_argument("--version", "-X", required=True,
                    help="PLR version string (e.g. plr_v1.4_cot)")
    ru.add_argument("--attribute", "-A", required=True,
                    help="PLR attribute to re-score")
    ru.add_argument("--dataset", default=None,
                    help="dataset dir (default: eval/golden/<attribute>)")
    ru.add_argument("--model", default="gemma",
                    help="registry model name (default: gemma; 'mock' is GPU-free)")

    # -- eval --
    ev = sub.add_parser(
        "eval",
        help="Score predictions vs golden labels (PLR attribute eval).",
    )
    ev.add_argument("--attribute", "-A", required=True,
                    help="PLR attribute (gender | vehicle_type | military | "
                         "custom), comma list, or 'all' (= manifest-declared "
                         "attributes; requires --dataset)")
    ev.add_argument("--model", default="gemma",
                    help="registry model name recorded in the ledger (default: gemma)")
    ev.add_argument("--version", default="plr_v1.5_cot",
                    help="PLR version tag")
    ev.add_argument("--ledger", default=None, help="ledger.jsonl path override")
    ev.add_argument("--core-ir", default=None, dest="core_ir",
                    help="path to core/ir repo (for stale-seed warning)")
    ev.add_argument("--dataset", default=None,
                    help="dataset dir (default: eval/golden/<attribute>)")

    # -- port --
    po = sub.add_parser(
        "port",
        help="Diff (or apply) lab prompt surface vs core/ir. Read-only by default.",
    )
    po.add_argument("--attribute", "-A", default=None,
                    help="(optional) filter to a specific attribute (informational)")
    po.add_argument("--apply", action="store_true",
                    help="Copy lab files into core/ir (default: read-only diff)")
    po.add_argument("--core-ir", default=None, dest="core_ir",
                    help="path to core/ir repo (default: CORE_IR_PATH env, "
                         "else ../ziomilitary/core/ir if present)")

    # -- validate-dataset --
    vd = sub.add_parser(
        "validate-dataset",
        help="Validate a dataset directory against the PLR dataset spec.",
    )
    vd.add_argument("--dataset", "-D", required=True,
                    help="path to the dataset directory to validate")

    # -- demo --
    dm = sub.add_parser(
        "demo",
        help="GPU-free onboarding: run a full mock cycle on a synthetic dataset.",
    )
    dm.add_argument("--keep", action="store_true",
                    help="Keep demo_dataset/ directory after the run (default: removed).")

    # -- experiment --
    exp = sub.add_parser(
        "experiment",
        help="Run an experiment matrix defined by an experiment.yaml.",
    )
    exp_sub = exp.add_subparsers(dest="experiment_cmd", metavar="<experiment_cmd>")
    exp_sub.required = True

    exp_run = exp_sub.add_parser(
        "run",
        help="Run the cross-product matrix from an experiment.yaml.",
    )
    exp_run.add_argument(
        "experiment_yaml",
        help="Path to the experiment.yaml file.",
    )
    exp_run.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if ANY cell fails (default: only exit nonzero when ALL cells fail).",
    )

    # -- report --
    rp = sub.add_parser(
        "report",
        help="Generate a self-contained HTML report from the eval ledger.",
    )
    rp.add_argument("--out", default=os.path.join(_LAB_ROOT, "report.html"),
                    help="output HTML path (default: report.html)")
    rp.add_argument("--ledger", default=os.path.join(_LAB_ROOT, "eval", "ledger.jsonl"),
                    help="ledger.jsonl path (default: eval/ledger.jsonl)")
    rp.add_argument("--compare", default=None, metavar="LEDGER_B",
                    help="second ledger to compare side-by-side (experiment-set vs experiment-set)")

    # -- gallery --
    ga = sub.add_parser(
        "gallery",
        help="Crops-vs-labels visual HTML for one dataset (wrong-first).",
    )
    ga.add_argument("--dataset", "-D", required=True,
                    help="dataset directory (crops/ + labels.jsonl [+ predictions.jsonl])")
    ga.add_argument("--attribute", "-A", default=None,
                    help="which attribute's labels to render (required for "
                         "multi-attribute labels.jsonl)")
    ga.add_argument("--out", default=None,
                    help="output HTML path (default: <dataset>/gallery.html)")

    return p


# =====================================================================
# Subcommand: demo
# =====================================================================


def _cmd_demo(args: argparse.Namespace) -> int:
    """GPU-free onboarding: full mock cycle on a synthetic dataset.

    Delegates to demo.run_demo() which lives in demo.py alongside this file.
    No GPU, no DB, no Redis required.
    """
    from runners.demo import run_demo
    keep = getattr(args, "keep", False)
    return run_demo(lab_root=_LAB_ROOT_PATH, keep_dir=keep)


# =====================================================================
# Subcommand: experiment
# =====================================================================


def _cmd_experiment_run(args: argparse.Namespace) -> int:
    """Run the cross-product experiment matrix defined in an experiment.yaml.

    Delegates to experiment.run_experiment().
    """
    from runners.experiment import run_experiment

    return run_experiment(
        experiment_yaml=args.experiment_yaml,
        strict=getattr(args, "strict", False),
    )


# =====================================================================
# Subcommand: report
# =====================================================================


def _cmd_report(args: argparse.Namespace) -> int:
    """Generate a self-contained HTML report from the eval ledger.

    Delegates to report.build_report(). GPU-free, network-free.
    """
    from evalkit.report import build_report

    build_report(ledger_path=args.ledger, out_path=args.out,
                 compare_ledger=getattr(args, "compare", None))
    return 0


# =====================================================================
# Subcommand: gallery
# =====================================================================


def _cmd_gallery(args: argparse.Namespace) -> int:
    """Self-contained crops-vs-labels HTML (wrong-first). GPU-free."""
    from evalkit.gallery import build_gallery

    out = build_gallery(args.dataset, out_path=getattr(args, "out", None),
                        attribute=getattr(args, "attribute", None))
    print(f"[gallery] written: {out}")
    return 0


# =====================================================================
# Subcommand: validate-dataset
# =====================================================================


def _cmd_validate_dataset(args: argparse.Namespace) -> int:
    """Validate a dataset directory against the PLR dataset spec."""
    from evalkit.validate import validate_dataset

    ok = validate_dataset(args.dataset)
    return 0 if ok else 1


# =====================================================================
# Dispatch
# =====================================================================

_DISPATCH = {
    "build-golden": _cmd_build_golden,
    "label": _cmd_label,
    "run": _cmd_run,
    "eval": _cmd_eval,
    "port": _cmd_port,
    "validate-dataset": _cmd_validate_dataset,
    "demo": _cmd_demo,
    "report": _cmd_report,
    "gallery": _cmd_gallery,
    "experiment": None,  # nested — dispatched via _EXPERIMENT_DISPATCH below
}

_EXPERIMENT_DISPATCH = {
    "run": _cmd_experiment_run,
}


def main() -> None:
    parser = _build_parser()
    # `label` forwards correction flags (--female-in-male, ...) verbatim to
    # make_labels.py. argparse.REMAINDER can't capture option-looking tokens
    # that follow a consumed optional (--dataset), so use parse_known_args and
    # route the unknowns to `label`; any other command with extras is an error.
    args, extras = parser.parse_known_args()
    if args.cmd == "label":
        args.label_args = extras
    elif extras:
        parser.error("unrecognized arguments: " + " ".join(extras))

    if args.cmd == "experiment":
        handler = _EXPERIMENT_DISPATCH.get(args.experiment_cmd)
        if handler is None:
            parser.error(f"Unknown experiment subcommand: {args.experiment_cmd!r}")
        sys.exit(handler(args))

    handler = _DISPATCH[args.cmd]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
