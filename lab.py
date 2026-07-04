#!/usr/bin/env python3
"""lab — single argparse entrypoint for the PLR prompt lab.

Scoring/report/gallery now live on the eval server: run `lab run` to produce
attributes.jsonl, then `lab submit --pull` to score on the server and fetch the
rendered report/gallery. The lab no longer scores locally.

Subcommands:
  build-golden  Build golden eval set (wraps eval/build_golden.py) and
                copy crops to the authoritative eval/golden/<A>/crops/ path.
  label         Turn human misclassification notes into labels.jsonl
                (wraps eval/make_labels.py).
  run           Re-score golden set with Gemma (wraps re_score.re_score).
  port          Diff / apply lab prompt surface against core/ir (read-only by
                default; --apply copies lab files into core/ir).
  demo          GPU-free onboarding: run a mock re-score on a synthetic
                dataset so a new user can see the loop end-to-end immediately.

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
    from evalkit.dataset import declared_attributes, resolve_dataset_dir
    from registry import get_model

    model_name = getattr(args, "model", "gemma")
    # -A는 선택: 모델 호출은 속성과 무관(전체 plr_json이 attributes.jsonl에
    # 저장)하고, 크롭별 프롬프트는 labels.jsonl의 object_type이 정한다.
    # 생략 시 manifest 선언 속성의 첫 번째가 predictions.jsonl 기본 추출 뷰.
    attribute = getattr(args, "attribute", None)
    if not attribute:
        if not getattr(args, "dataset", None):
            print("[run] --attribute 생략 시 --dataset이 필요합니다 "
                  "(manifest에서 기본 속성을 읽음)", file=sys.stderr)
            return 2
        declared = declared_attributes(args.dataset)
        if not declared:
            print(f"[run] {args.dataset}: manifest에 attributes:/attribute 선언이 "
                  f"없습니다 — -A로 속성을 지정하세요", file=sys.stderr)
            return 2
        attribute = declared[0]
        print(f"[run] --attribute 생략 → manifest 첫 속성 '{attribute}' 사용 "
              f"(추출 뷰만 결정; 전체 속성은 attributes.jsonl에 저장됨)")
    ds_dir = resolve_dataset_dir(_LAB_ROOT, attribute, getattr(args, "dataset", None))

    # get_model('mock') is GPU-free; 'gemma' constructs LabGemmaModel (weights
    # load lazily on first .generate).
    model = get_model(model_name)
    print(f"[run] re_score attribute={attribute!r} version={args.version!r} "
          f"model={model_name!r} dataset={str(ds_dir)!r}")
    meta = rs.re_score(
        attribute, model, golden_dir=str(ds_dir),
        prompt_version=args.version, model_name=model_name or "gemma"
    )
    print(f"[run] re_score done: {meta}")

    # run 시점 표면 지문 + 실행 파라미터 기록 — 서버 제출 무결성 대조의 기준점.
    from runners.client import write_run_provenance
    prov_path = write_run_provenance(
        Path(str(ds_dir)), _LAB_ROOT_PATH,
        model=model_name or "gemma", version=args.version,
        max_tokens=getattr(model, "max_tokens", None),
        temperature=getattr(model, "temperature", None),
    )
    print(f"[run] provenance -> {prov_path}")
    return 0


# =====================================================================
# Subcommand: port
# =====================================================================

# True prompt surface: files to compare between lab and core/ir. Built from the
# single source of truth (provenance.surface_relpaths) so `lab port` and the
# run_provenance surface_hash always cover the SAME files. Same relative path both sides;
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
        description="PLR prompt lab CLI — build, label, run, submit, port.",
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
    ru.add_argument("--attribute", "-A", default=None,
                    help="predictions.jsonl extraction view (optional — "
                         "defaults to the dataset manifest's first declared "
                         "attribute; the model call itself is attribute-"
                         "independent and stores the full plr_json)")
    ru.add_argument("--dataset", default=None,
                    help="dataset dir (default: eval/golden/<attribute>)")
    ru.add_argument("--model", default="gemma",
                    help="registry model name (default: gemma; 'mock' is GPU-free)")

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
        help="GPU-free onboarding: mock re-score on a synthetic dataset.",
    )
    dm.add_argument("--keep", action="store_true",
                    help="Keep datasets/demo/ directory after the run (default: removed).")

    # -- dataset-push / submit (평가 서버 클라이언트) --
    dp = sub.add_parser("dataset-push",
                        help="Register a dataset directory on the eval server.")
    dp.add_argument("--dataset", "-D", required=True, help="dataset directory")
    dp.add_argument("--name", default=None, help="server-side name (default: dir name)")
    dp.add_argument("--server", "-S", default=os.environ.get("EVAL_SERVER_URL", ""),
                    help="eval server URL (default: env EVAL_SERVER_URL)")
    dp.add_argument("--token", default=os.environ.get("EVAL_SERVER_TOKEN", ""))
    dp.add_argument("--by", default="", help="registrant display name")

    sm = sub.add_parser("submit",
                        help="Submit a run (attributes.jsonl + surface bundle) to the eval server.")
    sm.add_argument("--dataset", required=True, help="server-side dataset name")
    sm.add_argument("--run-dir", required=True, dest="run_dir",
                    help="local dataset dir holding attributes.jsonl (+run_provenance.json)")
    sm.add_argument("--version", "-X", required=True, help="prompt version label")
    sm.add_argument("--server", "-S", default=os.environ.get("EVAL_SERVER_URL", ""),
                    help="eval server URL (default: env EVAL_SERVER_URL)")
    sm.add_argument("--token", default=os.environ.get("EVAL_SERVER_TOKEN", ""))
    sm.add_argument("--by", default="", help="submitter display name")
    sm.add_argument("--pull", action="store_true",
                    help="제출 후 서버가 렌더한 metrics/report/gallery를 로컬로 받아온다")
    sm.add_argument("--out", default=None,
                    help="--pull 저장 위치 (기본: <run-dir>/pulled/)")

    return p


# =====================================================================
# Subcommand: demo
# =====================================================================


def _cmd_demo(args: argparse.Namespace) -> int:
    """GPU-free onboarding: mock re-score on a synthetic dataset.

    Delegates to demo.run_demo() which lives in demo.py alongside this file.
    No GPU, no DB, no Redis required.
    """
    from runners.demo import run_demo
    keep = getattr(args, "keep", False)
    return run_demo(lab_root=_LAB_ROOT_PATH, keep_dir=keep)


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

def _cmd_dataset_push(args: argparse.Namespace) -> int:
    """데이터셋 디렉터리를 평가 서버에 등록 (tar.gz 업로드)."""
    if not args.server:
        print("[dataset-push] --server 또는 EVAL_SERVER_URL이 필요합니다", file=sys.stderr)
        return 2
    from runners.client import dataset_push
    ds = Path(args.dataset)
    name = args.name or ds.name
    res = dataset_push(args.server, ds, name, args.token, created_by=args.by)
    print(f"[dataset-push] registered {res['name']!r} (crops={res['n_crops']})")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    """run 산출물 + 표면 번들을 평가 서버에 제출하고 지표 요약을 출력."""
    if not args.server:
        print("[submit] --server 또는 EVAL_SERVER_URL이 필요합니다", file=sys.stderr)
        return 2
    from runners.client import submit_run
    res = submit_run(args.server, args.dataset, Path(args.run_dir), args.version,
                     args.token, submitted_by=args.by, lab_root=_LAB_ROOT_PATH)
    badge = "" if res.get("hash_verified") else "  ⚠ hash unverified"
    if res.get("git_dirty"):
        badge += "  ⚠ dirty"
    print(f"[submit] run {res['run_id']}{badge}")
    agg = res.get("aggregate") or {}
    print(f"[submit] aggregate: macro_f1={agg.get('macro_f1')} "
          f"macro_acc={agg.get('macro_acc')} micro_acc={agg.get('micro_acc')}")
    for a, m in (res.get("attributes") or {}).items():
        print(f"[submit]   {a}: acc={m['accuracy']} macro_f1={m['macro_f1']} n={m['n']}")
    if res.get("skipped"):
        print(f"[submit] skipped(라벨 없음): {', '.join(res['skipped'])}")
    if getattr(args, "pull", False):
        from runners.client import pull_artifacts
        out_dir = Path(args.out) if args.out else Path(args.run_dir) / "pulled"
        got = pull_artifacts(args.server, res["run_id"], out_dir, args.token)
        print(f"[submit] pulled {', '.join(got)} -> {out_dir}")
    return 0


_DISPATCH = {
    "build-golden": _cmd_build_golden,
    "label": _cmd_label,
    "run": _cmd_run,
    "port": _cmd_port,
    "validate-dataset": _cmd_validate_dataset,
    "demo": _cmd_demo,
    "dataset-push": _cmd_dataset_push,
    "submit": _cmd_submit,
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

    handler = _DISPATCH[args.cmd]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
