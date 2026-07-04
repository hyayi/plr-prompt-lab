"""report.py가 소비하는 ledger 레코드 형식의 계약 고정 (RE-000).

evalkit/report.py는 `load_ledger()`로 flat 레코드 리스트를 읽고 top-level 필드를
소비한다(attribute/version/date/model/dataset/pipeline/prompt_hash + 지표들).
server/render.py 어댑터의 출력이 이 계약과 필드-대-필드 일치해야 report.py를
손대지 않고 서버에서 렌더할 수 있다.

- REQUIRED_FIELDS: report.py가 실제로 `r.get(...)`으로 읽는 키 (evalkit/report.py
  grep로 도출). 어댑터 레코드는 이 키를 전부 가져야 한다.
- provenance 매핑(서버 meta.json → ledger 필드):
    version   ← meta.version_label
    date      ← meta.submitted_at
    prompt_hash← meta.surface_hash
    model     ← meta.model
    dataset   ← meta.dataset
  나머지 지표(accuracy/recall/precision/f1/macro_f1/bias/confusion/
  pred_unknown/n_label_unknown/margin_stats/quality_stats/n)는 score() 결과.
"""
from __future__ import annotations

# report.py가 top-level로 읽는 필드 (grep evalkit/report.py: r.get(...))
REQUIRED_FIELDS: frozenset[str] = frozenset({
    "attribute", "version", "date", "n", "accuracy",
    "recall", "precision", "f1", "macro_f1", "bias", "confusion",
    "pred_unknown", "dataset", "model", "pipeline", "prompt_hash",
})

# score() 결과에서 그대로 오는 지표 키
SCORE_METRIC_FIELDS: frozenset[str] = frozenset({
    "n", "accuracy", "recall", "precision", "f1", "macro_f1",
    "bias", "confusion", "pred_unknown", "n_label_unknown",
    "margin_stats", "quality_stats",
})

# 서버 meta.json → ledger 레코드 필드 매핑 (provenance)
META_TO_LEDGER: dict[str, str] = {
    "version_label": "version",
    "submitted_at": "date",
    "surface_hash": "prompt_hash",
    "model": "model",
    "dataset": "dataset",
}


def assert_ledger_record(rec: dict) -> None:
    """어댑터 출력 레코드가 report.py 계약을 만족하는지 단언 (테스트용)."""
    missing = REQUIRED_FIELDS - set(rec)
    assert not missing, f"ledger record missing report.py fields: {sorted(missing)}"
