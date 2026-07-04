"""lab 클라이언트 계약 (US-007 / AC10): run_provenance 기록·표면 번들·multipart.

네트워크 없음 — 서버 왕복은 e2e가 담당, 여기선 산출물 형식을 고정한다.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from runners.client import (  # noqa: E402
    build_surface_bundle,
    multipart_body,
    write_run_provenance,
)


def test_run_provenance_fields(tmp_path: Path) -> None:
    out = write_run_provenance(tmp_path, _LAB_ROOT, model="mock",
                               version="tv1", max_tokens=512, temperature=0.0)
    prov = json.loads(out.read_text(encoding="utf-8"))
    for key in ("surface_hash", "lab_sha", "git_dirty", "model", "version",
                "max_tokens", "temperature", "reason"):
        assert key in prov, f"missing {key}"
    assert len(prov["surface_hash"]) == 12  # prompt_hash 기본 길이
    assert prov["model"] == "mock" and prov["max_tokens"] == 512


def test_surface_bundle_matches_surface_relpaths_and_hash(tmp_path: Path) -> None:
    """번들 = surface_relpaths 집합 그대로 → 해제 후 lab 해시 재계산이
    로컬 계산과 일치해야 한다 (서버 대조가 성립하는 근거)."""
    from evalkit.provenance import prompt_hash, surface_relpaths

    data = build_surface_bundle(_LAB_ROOT)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        names = {m.name for m in tf.getmembers() if m.isfile()}
        tf.extractall(tmp_path)
    expected = set(surface_relpaths(_LAB_ROOT, include_exp_configs=True))
    assert names == expected
    assert prompt_hash(tmp_path) == prompt_hash(_LAB_ROOT)


def test_multipart_body_parseable(tmp_path: Path) -> None:
    body, ctype = multipart_body({"a": "1"}, {"f": ("x.bin", b"BYTES", "application/octet-stream")})
    boundary = ctype.split("boundary=")[1]
    assert f"--{boundary}--".encode() in body
    assert b'name="a"' in body and b"BYTES" in body
