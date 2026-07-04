#!/usr/bin/env python3
"""골든셋 채점 코어 재노출 shim — 채점은 evalkit/scoring.py::score()가 단일 원천.

RE-004(2026-07) 이후 lab은 로컬 채점을 하지 않는다: 지표/리포트/갤러리는
평가 서버가 렌더한다(`lab run` → attributes.jsonl → `lab submit --pull`).
따라서 이 파일의 CLI(main/argparse/__main__)는 제거되었다.

호환을 위해 evalkit.scoring의 score()/ScoringError만 재노출한다.
"""
from __future__ import annotations

import os
import sys

# Lab root (one level above eval/) must be importable for the shared helpers
# whether this runs standalone or loaded via importlib.
_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LAB_ROOT not in sys.path:
    sys.path.insert(0, _LAB_ROOT)

from evalkit.scoring import ScoringError, score  # noqa: E402,F401

__all__ = ["ScoringError", "score"]
