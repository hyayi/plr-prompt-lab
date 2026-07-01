#!/usr/bin/env python3
"""lab — single argparse entrypoint for the PLR prompt lab.

Subcommands:
  build-golden  Build golden eval set (wraps eval/build_golden.py) and
                copy crops to the authoritative eval/golden/<A>/crops/ path.
  label         Turn human misclassification notes into labels.jsonl
                (wraps eval/make_labels.py).
  run           Re-score golden set with Gemma + (optionally) run search
                (wraps re_score.re_score + re_score.run_search_over_golden).
  eval          Score predictions vs golden labels (attr or search mode).
  port          Diff / apply lab prompt surface against core/ir (read-only by
                default; --apply copies lab files into core/ir).

Usage:
    python3 lab.py <cmd> [options]
    python3 -m lab  <cmd> [options]
"""
from __future__ import annotations

import argparse
import os
import sys

# Add lab root to path so sibling modules are importable.
_LAB_ROOT = os.path.dirname(os.path.abspath(__file__))
if _LAB_ROOT not in sys.path:
    sys.path.insert(0, _LAB_ROOT)

_DEFAULT_CORE_IR = "/home/ziovision/ziomilitary/core/ir"

# =====================================================================
# Subcommand: build-golden
# =====================================================================


def _cmd_build_golden(args: argparse.Namespace) -> int:
    """Call eval/build_golden.py's builder then copy crops to the authoritative
    eval/golden/<attribute>/crops/<obj_id>.jpg location."""
    import importlib.util
    import shutil

    # Load build_golden as a module (avoids sys.argv interference).
    spec = importlib.util.spec_from_file_location(
        "build_golden",
        os.path.join(_LAB_ROOT, "eval", "build_golden.py"),
    )
    bg = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(bg)  # type: ignore[union-attr]

    # Inject sys.argv so build_golden.main() parses the right args.
    orig_argv = sys.argv
    sys.argv = [
        "build_golden",
        "--video", args.video,
        "--attribute", args.attribute,
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
    # eval/golden/<attribute>/crops/<obj_id>.jpg so re_score can find them.
    gdir = os.path.join(_LAB_ROOT, "eval", "golden", args.attribute)
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
        result_path = os.environ.get(
            "RESULT_PATH",
            "/home/ziovision/data/ziosummary/results",
        )
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
    # Forward any remaining args (argparse remainder or explicit flags).
    sys.argv += args.label_args
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
    """Set env for version, call re_score then optionally run_search_over_golden.

    This is the GPU step at real runtime — the CLI wires it but does not
    import Gemma at module level.
    """
    import re_score as rs

    # Set version env vars so re_score.re_score() picks them up.
    if args.version:
        os.environ["IR_PLR_FORMAT"] = args.version

    # LabGemmaModel is imported lazily here so the CLI is importable without GPU.
    from gemma_model import LabGemmaModel

    attribute = args.attribute
    print(f"[run] re_score attribute={attribute!r} version={args.version!r}")
    meta = rs.re_score(attribute, LabGemmaModel())
    print(f"[run] re_score done: {meta}")

    # Run search over golden if queries.jsonl exists.
    queries_path = os.path.join(
        _LAB_ROOT, "eval", "golden", "search", "queries.jsonl"
    )
    if os.path.exists(queries_path):
        print("[run] running search over golden …")
        rs.run_search_over_golden(model=None)  # dictionary path, no GPU needed
        print("[run] search done")
    else:
        print(f"[run] {queries_path} not found — skipping search runner")

    return 0


# =====================================================================
# Subcommand: eval
# =====================================================================


def _cmd_eval(args: argparse.Namespace) -> int:
    """Score predictions vs golden labels.

    Mode 'attr'   → eval/run_eval.py
    Mode 'search' → run_search_eval.py
    """
    core_ir = getattr(args, "core_ir", None) or _DEFAULT_CORE_IR

    if args.mode == "attr":
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_eval",
            os.path.join(_LAB_ROOT, "eval", "run_eval.py"),
        )
        re_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(re_mod)  # type: ignore[union-attr]

        orig_argv = sys.argv
        sys.argv = ["run_eval", "--attribute", args.attribute]
        if args.version:
            sys.argv += ["--version", args.version]
        if args.ledger:
            sys.argv += ["--ledger", args.ledger]
        try:
            re_mod.main()
        except SystemExit as e:
            return int(e.code) if e.code is not None else 0
        finally:
            sys.argv = orig_argv

    elif args.mode == "search":
        import run_search_eval as rse

        ledger_path = args.ledger or os.path.join(
            _LAB_ROOT, "eval", "ledger.jsonl"
        )
        rse.main(
            version=args.version or "plr_v1.4_cot",
            ledger_path=ledger_path,
            k=args.k if hasattr(args, "k") and args.k else 5,
            core_ir_path=core_ir,
        )
    else:
        print(f"Unknown mode: {args.mode!r}", file=sys.stderr)
        return 1

    return 0


# =====================================================================
# Subcommand: port
# =====================================================================

# True prompt surface: files to compare between lab and core/ir.
_PORT_FILES = [
    # (lab relative path,  core/ir relative path)
    ("prompts/plr_v0.4.yaml",    "prompts/plr_v0.4.yaml"),
    ("prompts/plr_v1.3_cot.yaml","prompts/plr_v1.3_cot.yaml"),
    ("prompts/plr_v1.4_cot.yaml","prompts/plr_v1.4_cot.yaml"),
    ("plr_prompts.py",           "plr_prompts.py"),
    ("plr_core.py",              "plr_core.py"),
]


def _cmd_port(args: argparse.Namespace) -> int:
    """Diff (or apply) the lab prompt surface against core/ir.

    Read-only by default — print a unified diff between lab copy and core/ir
    of: prompts/*.yaml, plr_prompts.py, plr_core.py.
    --apply copies the lab versions into core/ir and prints a reminder to
    run core/ir/tests/test_prompt_source_parity.py.
    """
    import difflib
    import shutil

    core_ir = getattr(args, "core_ir", None) or _DEFAULT_CORE_IR

    # Warn if stale seed
    import run_search_eval as rse
    seed_hash = rse._read_seed_hash(_LAB_ROOT)
    rse._warn_stale_seed(_LAB_ROOT, seed_hash, core_ir)

    attribute_filter = getattr(args, "attribute", None)  # optional, currently unused
    apply_mode = getattr(args, "apply", False)

    all_diffs: list[str] = []

    for lab_rel, ir_rel in _PORT_FILES:
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
    for lab_rel, ir_rel in _PORT_FILES:
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

    # -- label --
    la = sub.add_parser(
        "label",
        help="Turn human misclassification notes into labels.jsonl.",
    )
    la.add_argument("label_args", nargs=argparse.REMAINDER,
                    help="Arguments forwarded verbatim to make_labels.py "
                         "(e.g. --female-in-male M3,M7 --male-in-female F2)")

    # -- run --
    ru = sub.add_parser(
        "run",
        help="Re-score golden set with Gemma (GPU step).",
    )
    ru.add_argument("--version", "-X", required=True,
                    help="PLR version string (e.g. plr_v1.4_cot)")
    ru.add_argument("--attribute", "-A", required=True,
                    help="PLR attribute to re-score")

    # -- eval --
    ev = sub.add_parser(
        "eval",
        help="Score predictions vs golden labels (attr or search mode).",
    )
    ev.add_argument("--attribute", "-A", required=True,
                    help="PLR attribute (for attr mode) or 'search'")
    ev.add_argument("--mode", choices=["attr", "search"], default="attr",
                    help="'attr' (default) = run_eval.py; 'search' = run_search_eval.py")
    ev.add_argument("--version", default="plr_v1.4_cot",
                    help="PLR version tag")
    ev.add_argument("--ledger", default=None, help="ledger.jsonl path override")
    ev.add_argument("--k", type=int, default=5,
                    help="rank cutoff for search mode (default: 5)")
    ev.add_argument("--core-ir", default=None, dest="core_ir",
                    help="path to core/ir repo (for stale-seed warning)")

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
                    help="path to core/ir repo (default: /home/ziovision/ziomilitary/core/ir)")

    return p


# =====================================================================
# Dispatch
# =====================================================================

_DISPATCH = {
    "build-golden": _cmd_build_golden,
    "label": _cmd_label,
    "run": _cmd_run,
    "eval": _cmd_eval,
    "port": _cmd_port,
}


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    handler = _DISPATCH[args.cmd]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
