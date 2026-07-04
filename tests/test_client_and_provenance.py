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


def test_surface_hash_scoped_to_used_version(tmp_path: Path) -> None:
    """S4: surface_hash/번들이 run이 쓴 프롬프트 버전만 커버 — 안 쓴 버전 편집은
    지문을 안 바꾸고, 쓴 버전 편집은 바꾼다. version 미지정(lab port 경로)은 전 버전."""
    import io
    import tarfile

    from evalkit.provenance import prompt_hash
    from runners.client import build_surface_bundle

    root = tmp_path
    (root / "prompts" / "v1").mkdir(parents=True)
    (root / "prompts" / "v2").mkdir(parents=True)
    (root / "prompts" / "v1" / "person.yaml").write_text("system: v1\n", encoding="utf-8")
    (root / "prompts" / "v2" / "person.yaml").write_text("system: v2\n", encoding="utf-8")

    h1 = prompt_hash(root, version="v1")
    # ① 안 쓴 v2 편집 → v1 해시 불변
    (root / "prompts" / "v2" / "person.yaml").write_text("system: v2-EDIT\n", encoding="utf-8")
    assert prompt_hash(root, version="v1") == h1, "다른 버전 편집이 지문을 바꾸면 안 됨"
    # ② 쓴 v1 편집 → 해시 변함
    (root / "prompts" / "v1" / "person.yaml").write_text("system: v1-EDIT\n", encoding="utf-8")
    assert prompt_hash(root, version="v1") != h1, "사용 버전 편집은 지문을 바꿔야 함"

    # ③ version 미지정(전 버전 = lab port 경로)은 어느 버전 변경에도 반응 — 동작 불변
    hall = prompt_hash(root)
    (root / "prompts" / "v2" / "person.yaml").write_text("system: v2-AGAIN\n", encoding="utf-8")
    assert prompt_hash(root) != hall, "version 미지정은 전 버전을 해싱(lab port 불변)"

    # ④ 번들도 사용 버전만 담는다
    data = build_surface_bundle(root, version="v1")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        names = {m.name for m in tf.getmembers() if m.isfile()}
    assert "prompts/v1/person.yaml" in names
    assert not any(n.startswith("prompts/v2/") for n in names), "번들에 안 쓴 버전 포함 금지"

    # ⑤ 불변식: 서버가 version 미지정으로 번들을 재해싱한 값 == client의 버전스코프 해시
    #    (서버 코드 변경 없이 hash_verified가 성립하는 근거)
    bdir = tmp_path / "extracted"
    bdir.mkdir()
    with tarfile.open(fileobj=io.BytesIO(build_surface_bundle(root, version="v1")),
                      mode="r:gz") as tf:
        tf.extractall(bdir)
    assert prompt_hash(bdir) == prompt_hash(root, version="v1"), \
        "서버 재해싱(version 미지정)이 client 버전스코프 해시와 일치해야 함"


def test_connection_refused_gives_friendly_error(monkeypatch) -> None:
    """서버가 안 떠 있을 때(URLError) raw 트레이스백 대신 원인 안내 SystemExit."""
    import urllib.error
    import urllib.request

    import pytest

    from runners import client as C

    def _boom(*a, **k):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(SystemExit, match="서버 연결 실패"):
        C._post("http://127.0.0.1:9/api/datasets", b"", "text/plain", "")
    with pytest.raises(SystemExit, match="서버 연결 실패"):
        C._get("http://127.0.0.1:9/api/runs/x", "")


def test_multipart_body_parseable(tmp_path: Path) -> None:
    body, ctype = multipart_body({"a": "1"}, {"f": ("x.bin", b"BYTES", "application/octet-stream")})
    boundary = ctype.split("boundary=")[1]
    assert f"--{boundary}--".encode() in body
    assert b'name="a"' in body and b"BYTES" in body


def test_re_score_validates_plr_and_fails_loud(tmp_path, monkeypatch) -> None:
    """re_score가 산출된 각 plr_json을 validate_plr로 검증하고 실패 시 fail-loud
    (AC3b — 서버가 클라이언트 검증을 신뢰하는 계약의 근거)."""
    import pytest
    from PIL import Image

    import plr_schema
    from runners import re_score as rs

    ds = tmp_path / "ds"
    (ds / "crops").mkdir(parents=True)
    Image.new("RGB", (60, 90), (100, 100, 100)).save(str(ds / "crops" / "x.jpg"))
    (ds / "manifest.yaml").write_text(
        "n: 1\ncreated: '2026-07-04'\nsource_note: t\nattributes:\n  gender: {}\n")
    (ds / "labels.jsonl").write_text(
        '{"obj_id": "x", "labels": {"gender": "male"}}\n')

    class _MockModel:
        def generate(self, messages, image):
            return ("target: person\ngender: male\ngender_reason: t\nage: adult\n"
                    "outfit: two_piece\nupper:\n  color: black\n  type: jacket\n"
                    "lower:\n  color: black\n  type: pants\naction: standing\n"
                    "military: civilian\nmargins: {gender: 0.9}")

    # validate_plr이 실제로 호출됨을 증명: raise로 바꾸면 re_score가 fail-loud.
    def _boom(data):
        raise ValueError("SCHEMA VIOLATION (injected)")
    monkeypatch.setattr(plr_schema, "validate_plr", _boom)

    with pytest.raises(ValueError, match="SCHEMA VIOLATION"):
        rs.re_score("gender", _MockModel(), golden_dir=str(ds), model_name="mock")

    # 정상 경로(진짜 validate_plr)는 통과해 파일을 쓴다.
    monkeypatch.undo()
    rs.re_score("gender", _MockModel(), golden_dir=str(ds), model_name="mock")
    assert (ds / "attributes.jsonl").exists()
