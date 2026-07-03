"""autorender — eval/experiment 종료 후 gallery.html·report.html 자동 생성.

측정(ledger 적재)이 끝났는데 시각화가 낡아 있는 상태를 없앤다:
  - 평가에 참여한 데이터셋마다 <dataset>/gallery.html (다속성 태그 뷰)
  - 사용한 ledger 옆에 report.html (전체 비교표)

렌더 실패는 경고만 하고 측정 결과(ledger)는 건드리지 않는다 — 시각화는
파생물이라 언제든 `lab gallery`/`lab report`로 재생성할 수 있기 때문.
"""
from __future__ import annotations

import sys
from pathlib import Path


def auto_render(datasets: list[str | Path], ledger_path: str | Path | None) -> list[str]:
    """gallery(데이터셋별) + report(ledger 옆) 생성. 생성된 경로 목록 반환."""
    outs: list[str] = []
    for ds in dict.fromkeys(str(d) for d in datasets):
        try:
            from evalkit.gallery import build_gallery

            outs.append(build_gallery(ds))
        except Exception as exc:  # noqa: BLE001 — 렌더 실패가 측정을 무효화하면 안 됨
            print(f"[render] WARNING: gallery({ds}) 실패: {exc}", file=sys.stderr)
    if ledger_path:
        try:
            from evalkit.report import build_report

            out = str(Path(ledger_path).resolve().with_name("report.html"))
            outs.append(build_report(str(ledger_path), out))
        except Exception as exc:  # noqa: BLE001
            print(f"[render] WARNING: report({ledger_path}) 실패: {exc}", file=sys.stderr)
    for o in outs:
        print(f"[render] {o}")
    return outs
