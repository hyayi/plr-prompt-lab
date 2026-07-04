"""서버 render 라우트 게이트 (RE-000 실패-설계 → RE-001 green 전환).

계획 Step 0: 삭제 전에 대체재(서버 report/gallery 렌더)를 먼저 확보한다.
이 두 게이트는 RE-000 시점엔 xfail(미구현), RE-001에서 라우트가 생기면
xfail 마킹을 제거하고 green이 되어야 Step 3(lab eval 삭제)이 허용된다.

No GPU, no network (TestClient).
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from plr_parse import parse_plr_response  # noqa: E402

TOKEN = {"X-Auth-Token": "sekrit"}

_YAML = ("target: person\ngender: {g}\ngender_reason: t\nage: adult\n"
         "outfit: two_piece\nupper:\n  color: black\n  type: jacket\n"
         "lower:\n  color: black\n  type: pants\naction: standing\n"
         "military: civilian\nmargins: {{gender: 0.9}}")


def _targz_dir(src: Path, arcname: str | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        if arcname is None:
            for p in sorted(src.rglob("*")):
                if p.is_file():
                    tf.add(p, arcname=str(p.relative_to(src)))
        else:
            tf.add(src, arcname=arcname)
    return buf.getvalue()


def _surface_bundle(base: Path) -> bytes:
    (base / "prompts" / "v").mkdir(parents=True, exist_ok=True)
    (base / "prompts" / "v" / "person.yaml").write_text("system: t\n")
    (base / "schema").mkdir(exist_ok=True)
    (base / "schema" / "vocab.yaml").write_text("enums:\n  gender: [male, female]\n")
    return _targz_dir(base)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_SERVER_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_SERVER_TOKEN", "sekrit")
    from server.app import app
    with TestClient(app) as c:
        yield c


def _seed_dataset(client, tmp_path) -> str:
    from PIL import Image

    src = tmp_path / "src" / "rnd_ds"
    (src / "crops").mkdir(parents=True)
    for oid in ("a", "b"):
        Image.new("RGB", (60, 90), (90, 90, 90)).save(str(src / "crops" / f"{oid}.jpg"))
    (src / "manifest.yaml").write_text(
        "n: 2\ncreated: '2026-07-04'\nsource_note: t\nattributes:\n  gender: {}\n")
    (src / "labels.jsonl").write_text(
        '{"obj_id": "a", "labels": {"gender": "male"}}\n'
        '{"obj_id": "b", "labels": {"gender": "female"}}\n')
    assert client.post("/api/datasets", headers=TOKEN, data={"name": "rnd_ds"},
                       files={"archive": ("d.tgz", _targz_dir(src, "rnd_ds"),
                                          "application/gzip")}).status_code == 201
    return "rnd_ds"


def _submit(client, tmp_path, version: str) -> str:
    rows = [{"obj_id": "a", "plr_json": parse_plr_response(_YAML.format(g="male"), hint="person")},
            {"obj_id": "b", "plr_json": parse_plr_response(_YAML.format(g="male"), hint="person")}]
    attrs = "\n".join(json.dumps(r) for r in rows).encode()
    bundle = _surface_bundle(tmp_path / f"bundle-{version}")
    r = client.post("/api/runs", headers=TOKEN,
                    data={"dataset": "rnd_ds", "version_label": version},
                    files={"attributes": ("a.jsonl", attrs, "application/json"),
                           "surface": ("s.tgz", bundle, "application/gzip")})
    assert r.status_code == 201, r.text
    return r.json()["run_id"]


def test_gate_gallery_for_run(client, tmp_path) -> None:
    """게이트 1: 서버가 run에 대해 gallery.html을 렌더(two-root)하고 base64 크롭 내장."""
    _seed_dataset(client, tmp_path)
    run_id = _submit(client, tmp_path, "gv1")
    r = client.get(f"/api/runs/{run_id}/gallery.html")
    assert r.status_code == 200, r.text
    assert "data:image/jpeg;base64," in r.text, "gallery에 크롭 이미지 내장"
    assert "WRONG" in r.text or "CORRECT" in r.text, "정오 배지"


def test_gate_report_trend_across_runs(client, tmp_path) -> None:
    """게이트 2: 같은 데이터셋에 버전 다른 run 2회 → 데이터셋 report가 시퀀스 렌더."""
    _seed_dataset(client, tmp_path)
    _submit(client, tmp_path, "tv1")
    _submit(client, tmp_path, "tv2")
    r = client.get("/api/datasets/rnd_ds/report.html")
    assert r.status_code == 200, r.text
    assert "tv1" in r.text and "tv2" in r.text, "두 버전이 리포트에 나열"


def test_adapter_record_matches_report_contract(client, tmp_path) -> None:
    """server/render.py의 ledger 레코드가 report.py 계약(fixture)과 필드 일치."""
    from server.render import run_ledger_records
    from tests.fixtures.ledger_record_schema import assert_ledger_record

    _seed_dataset(client, tmp_path)
    run_id = _submit(client, tmp_path, "cv1")
    root = tmp_path / "data"
    recs = run_ledger_records(root / "runs" / run_id)
    assert recs, "run has at least one attribute record"
    for rec in recs:
        assert_ledger_record(rec)
    assert recs[0]["version"] == "cv1" and recs[0]["pipeline"] == "plr"
