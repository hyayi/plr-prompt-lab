"""AC4 파리티 게이트: lab `eval` 경로와 서버 채점 경로의 지표 완전 일치.

두 경로 모두 evalkit.scoring.score()에서 종결되지만, 이 테스트는 그 사실을
**믿지 않고 측정한다** — 래퍼 어느 쪽이 해석/전처리를 몰래 더하면 빨간불.

계약 (계획 Step 2):
  - 비교는 지표 필드만: prompt_hash/seed_hash/gemma_repo/model/dataset/
    version/date/pipeline은 체크아웃·제출 문맥이므로 제외.
  - 수치는 근사가 아닌 정확 일치 (소스가 round(...,4)로 고정).
  - fixture는 ① predictions.jsonl 부재→attributes.jsonl 폴백 분기
    ② predictions 스탬프+model 해석 분기를 모두 실통과.
  - margin 유·무 행 혼합 → margin_stats 값·None 일치, pred_unknown 포함.

No GPU, no DB, no network.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from evalkit.scoring import score  # noqa: E402
from server.scoring import score_run  # noqa: E402

# 지표 필드 — 이 목록 전체가 정확 일치해야 한다 (pred_unknown 포함).
METRIC_FIELDS = [
    "n", "accuracy", "recall", "precision", "f1", "macro_f1", "bias",
    "confusion", "pred_unknown", "n_label_unknown",
    "margin_stats", "quality_stats",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _plr(gender: str, margin: float | None, equip: str | None) -> dict:
    gs: dict = {"male": 0.0, "female": 0.0, "selected": gender}
    if margin is not None:
        gs["decision_margin"] = margin
    return {"object_type": "person", "attributes": {
        "gender_scores": gs,
        "equipment": ([{"type": equip, "score": 0.9}] if equip else []),
    }}


def _make_dataset(base: Path) -> Path:
    """gender(프리셋, margin 有/無 혼합) + helmet(커스텀) 다속성 fixture.
    predictions.jsonl 없음 → 모든 채점이 attributes.jsonl 폴백 분기를 지난다."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "manifest.yaml").write_text(
        "n: 4\ncreated: '2026-07-04'\nsource_note: parity fixture\n"
        "attributes:\n  gender: {}\n  helmet:\n"
        "    labels: [helmet, no_helmet]\n"
        "    pred_path: attributes.equipment[0].type\n",
        encoding="utf-8")
    _write_jsonl(base / "labels.jsonl", [
        {"obj_id": "a", "labels": {"gender": "male", "helmet": "helmet"}},
        {"obj_id": "b", "labels": {"gender": "female", "helmet": "no_helmet"}},
        {"obj_id": "c", "labels": {"gender": "unknown", "helmet": "helmet"}},
        {"obj_id": "d", "labels": {"gender": "male", "helmet": "helmet"}},
    ])
    _write_jsonl(base / "attributes.jsonl", [
        {"obj_id": "a", "plr_json": _plr("male", 0.9, "helmet")},
        {"obj_id": "b", "plr_json": _plr("male", 0.2, None)},      # gender 오답·저margin, helmet 오답(unknown)
        {"obj_id": "c", "plr_json": _plr("female", None, "helmet")},  # margin 無 행
        {"obj_id": "d", "plr_json": _plr("male", 0.8, "helmet")},
    ])
    return base


def _lab_eval_record(golden: Path, ledger: Path, attribute: str) -> dict:
    """lab CLI 경로 그대로: run_eval.main()을 argv 주입으로 실행 (lab.py 방식)."""
    spec = importlib.util.spec_from_file_location(
        "run_eval_parity", str(_LAB_ROOT / "eval" / "run_eval.py"))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    orig = sys.argv
    sys.argv = ["run_eval", "--attribute", attribute, "--golden", str(golden),
                "--version", "parity_v1", "--ledger", str(ledger)]
    try:
        mod.main()
    except SystemExit:
        pass
    finally:
        sys.argv = orig
    recs = [json.loads(l) for l in open(ledger, encoding="utf-8")]
    return next(r for r in reversed(recs) if r["attribute"] == attribute)


def test_metric_parity_lab_vs_server(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path / "ds")
    ledger = tmp_path / "ledger.jsonl"

    # 서버 경로: score_run (run의 attributes.jsonl = 데이터셋의 것과 동일 원본)
    server_res = score_run(ds, ds / "attributes.jsonl")

    for attribute in ("gender", "helmet"):
        lab_rec = _lab_eval_record(ds, ledger, attribute)
        srv = server_res["attributes"][attribute]
        for field in METRIC_FIELDS:
            assert srv.get(field) == lab_rec.get(field), (
                f"PARITY BREAK {attribute}.{field}: "
                f"server={srv.get(field)!r} lab={lab_rec.get(field)!r}"
            )

    # margin 캘리브레이션이 실제로 계산됐는지(전부 None이면 파리티가 공허)
    g = server_res["attributes"]["gender"]
    assert g["margin_stats"] is not None and g["margin_stats"]["n"] == 3, \
        "margin 有 행 = a,b,d (c는 라벨 unknown으로 채점 제외)"
    # helmet은 margin_path가 없으므로 양쪽 다 None이어야 (None-일치)
    assert server_res["attributes"]["helmet"]["margin_stats"] is None


def test_model_stamp_branch_parity(tmp_path: Path) -> None:
    """predictions.jsonl 스탬프 경로: 양쪽이 같은 파일을 읽어 같은 지표 +
    model 스탬프 해석까지 일치."""
    ds = _make_dataset(tmp_path / "ds2")
    _write_jsonl(ds / "predictions.jsonl", [
        {"obj_id": "a", "attribute": "gender", "pred": "male", "margin": 0.9,
         "quality": 0.8, "model": "mock"},
        {"obj_id": "b", "attribute": "gender", "pred": "male", "margin": 0.2,
         "quality": 0.3, "model": "mock"},
        {"obj_id": "c", "attribute": "gender", "pred": "female", "model": "mock"},
        {"obj_id": "d", "attribute": "gender", "pred": "male", "margin": 0.8,
         "quality": 0.7, "model": "mock"},
    ])
    ledger = tmp_path / "ledger2.jsonl"
    lab_rec = _lab_eval_record(ds, ledger, "gender")
    assert lab_rec["model"] == "mock", "CLI가 예측 행 스탬프에서 모델을 해석해야"

    direct = score(ds, "gender")  # 같은 score() — 스탬프 일치 경로
    assert direct["resolved_model"] == "mock"
    for field in METRIC_FIELDS:
        assert direct.get(field) == lab_rec.get(field), (
            f"PARITY BREAK (stamped) gender.{field}")
    # quality_stats가 스탬프 경로에서만 존재(폴백엔 quality 無) — 값 검증
    assert direct["quality_stats"] is not None
