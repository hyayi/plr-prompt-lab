"""데이터셋 API 계약 (US-005 / AC1·AC12 전반부): 등록·중복·거부·라벨 정정.

run 재채점을 포함한 전체 사이클은 test_server_e2e.py가 담당.
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

TOKEN = {"X-Auth-Token": "sekrit"}


def _make_dataset_dir(base: Path) -> Path:
    from PIL import Image

    base.mkdir(parents=True)
    (base / "crops").mkdir()
    for oid in ("a", "b"):
        Image.new("RGB", (60, 90), (90, 90, 90)).save(str(base / "crops" / f"{oid}.jpg"))
    (base / "manifest.yaml").write_text(
        "n: 2\ncreated: '2026-07-04'\nsource_note: api test\n"
        "attributes:\n  gender: {}\n", encoding="utf-8")
    (base / "labels.jsonl").write_text(
        '{"obj_id": "a", "labels": {"gender": "male"}}\n'
        '{"obj_id": "b", "labels": {"gender": "female"}}\n', encoding="utf-8")
    return base


def _targz(src: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src, arcname=src.name)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EVAL_SERVER_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_SERVER_TOKEN", "sekrit")
    # app 모듈 재로드 없이 lifespan이 env를 읽도록 — STATE는 기동마다 재구성
    from server.app import app
    with TestClient(app) as c:
        yield c


def test_register_then_duplicate_409(client, tmp_path: Path) -> None:
    tar = _targz(_make_dataset_dir(tmp_path / "src" / "my_ds"))
    r = client.post("/api/datasets", headers=TOKEN,
                    data={"name": "my_ds", "created_by": "tester"},
                    files={"archive": ("my_ds.tgz", tar, "application/gzip")})
    assert r.status_code == 201, r.text
    assert r.json()["n_crops"] == 2 and "PASS" in r.json()["report"]

    listed = client.get("/api/datasets").json()["datasets"]
    assert listed[0]["name"] == "my_ds" and listed[0]["attributes"] == ["gender"]

    r2 = client.post("/api/datasets", headers=TOKEN,
                     data={"name": "my_ds"},
                     files={"archive": ("x.tgz", tar, "application/gzip")})
    assert r2.status_code == 409 and "새 이름" in r2.json()["detail"]


def test_register_rejects_invalid_dataset(client, tmp_path: Path) -> None:
    bad = tmp_path / "src" / "bad_ds"
    _make_dataset_dir(bad)
    (bad / "labels.jsonl").write_text(
        '{"obj_id": "a", "labels": {"gendre": "male"}}\n', encoding="utf-8")  # 오타 속성
    r = client.post("/api/datasets", headers=TOKEN,
                    data={"name": "bad_ds"},
                    files={"archive": ("b.tgz", _targz(bad), "application/gzip")})
    assert r.status_code == 422 and "FAIL" in r.json()["detail"]


def test_label_correction_swaps_and_audits(client, tmp_path: Path) -> None:
    tar = _targz(_make_dataset_dir(tmp_path / "src" / "fix_ds"))
    assert client.post("/api/datasets", headers=TOKEN, data={"name": "fix_ds"},
                       files={"archive": ("d.tgz", tar, "application/gzip")}).status_code == 201

    new_labels = ('{"obj_id": "a", "labels": {"gender": "female"}}\n'
                  '{"obj_id": "b", "labels": {"gender": "female"}}\n')
    r = client.patch("/api/datasets/fix_ds/labels", headers=TOKEN,
                     data={"changed_by": "tester"},
                     files={"labels": ("labels.jsonl", new_labels.encode(), "application/json")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rescored_runs"] == []  # 아직 run 없음
    assert "-1 +1" in body["diff_summary"]

    # 무효 정정은 원복 + 422
    r2 = client.patch("/api/datasets/fix_ds/labels", headers=TOKEN,
                      files={"labels": ("labels.jsonl", b'{"obj_id":"a","labels":{"gender":"purple"}}\n', "application/json")})
    assert r2.status_code == 422 and "원복" in r2.json()["detail"]
