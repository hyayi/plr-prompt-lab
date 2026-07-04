"""run 제출 API 계약 (US-006 / AC2·AC3·AC7): 제출→채점→append, 무결성 대조.

No GPU (예측은 파서로 생성한 valid plr_json), no network.
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

_YAML = ("target: person\ngender: {g}\ngender_reason: test\nage: adult\n"
         "outfit: two_piece\nupper:\n  color: black\n  type: jacket\n"
         "lower:\n  color: black\n  type: pants\naction: standing\n"
         "military: civilian\nmargins: {{gender: {m}}}")


def _plr(gender: str, margin: float) -> dict:
    return parse_plr_response(_YAML.format(g=gender, m=margin), hint="person")


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


def _make_bundle(base: Path) -> tuple[bytes, str]:
    """표면 번들 (prompts+vocab) + lab 알고리즘으로 계산한 지문."""
    from evalkit.provenance import prompt_hash

    (base / "prompts" / "tv1").mkdir(parents=True, exist_ok=True)
    (base / "prompts" / "tv1" / "person.yaml").write_text("system: test-prompt\n")
    (base / "schema").mkdir(exist_ok=True)
    (base / "schema" / "vocab.yaml").write_text("enums:\n  gender: [male, female]\n")
    return _targz_dir(base), prompt_hash(base)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EVAL_SERVER_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_SERVER_TOKEN", "sekrit")
    from server.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def dataset(client, tmp_path: Path) -> str:
    from PIL import Image

    src = tmp_path / "src" / "runs_ds"
    src.mkdir(parents=True)
    (src / "crops").mkdir()
    for oid in ("a", "b"):
        Image.new("RGB", (60, 90), (90, 90, 90)).save(str(src / "crops" / f"{oid}.jpg"))
    (src / "manifest.yaml").write_text(
        "n: 2\ncreated: '2026-07-04'\nsource_note: t\nattributes:\n  gender: {}\n")
    (src / "labels.jsonl").write_text(
        '{"obj_id": "a", "labels": {"gender": "male"}}\n'
        '{"obj_id": "b", "labels": {"gender": "female"}}\n')
    r = client.post("/api/datasets", headers=TOKEN, data={"name": "runs_ds"},
                    files={"archive": ("d.tgz", _targz_dir(src, "runs_ds"), "application/gzip")})
    assert r.status_code == 201, r.text
    return "runs_ds"


def _attrs_bytes() -> bytes:
    rows = [{"obj_id": "a", "plr_json": _plr("male", 0.9)},
            {"obj_id": "b", "plr_json": _plr("male", 0.3)}]  # b는 오답
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode()


def _submit(client, tmp_path: Path, version: str, *, claimed: str | None = "good"):
    bundle, true_hash = _make_bundle(tmp_path / f"bundle-{version}-{claimed}")
    prov = {"surface_hash": true_hash if claimed == "good" else "deadbeef0000",
            "lab_sha": "abc1234", "git_dirty": False, "model": "mock",
            "max_tokens": 512, "temperature": 0.0, "reason": "on"}
    files = {"attributes": ("attributes.jsonl", _attrs_bytes(), "application/json"),
             "surface": ("surface.tgz", bundle, "application/gzip")}
    if claimed is not None:
        files["provenance"] = ("run_provenance.json",
                               json.dumps(prov).encode(), "application/json")
    return client.post("/api/runs", headers=TOKEN,
                       data={"dataset": "runs_ds", "version_label": version,
                             "submitted_by": "tester"}, files=files)


def test_submit_scores_and_appends(client, dataset, tmp_path: Path) -> None:
    r = _submit(client, tmp_path, "tv1")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["hash_verified"] is True
    assert body["attributes"]["gender"]["accuracy"] == 0.5  # a 정답, b 오답
    assert body["aggregate"]["micro_acc"] == 0.5

    # 재제출 = 새 run_id append (AC3), 기존 불변
    r2 = _submit(client, tmp_path, "tv1")
    assert r2.status_code == 201 and r2.json()["run_id"] != body["run_id"]

    default = client.get("/api/runs", params={"dataset": "runs_ds"}).json()["runs"]
    assert len(default) == 1, "기본 뷰 = version_label당 최신 run"
    full = client.get("/api/runs", params={"dataset": "runs_ds",
                                           "all_history": True}).json()["runs"]
    assert len(full) == 2

    detail = client.get(f"/api/runs/{body['run_id']}").json()
    assert detail["meta"]["model"] == "mock"
    assert any(f.startswith("prompts/") for f in detail["surface_files"])


def test_hash_mismatch_flagged_not_rejected(client, dataset, tmp_path: Path) -> None:
    r = _submit(client, tmp_path, "tv2", claimed="bad")
    assert r.status_code == 201, "불일치는 거부가 아니라 배지 (사용자 확정 정책)"
    assert r.json()["hash_verified"] is False


def test_structurally_broken_attributes_rejected(client, dataset, tmp_path: Path) -> None:
    """P2-1: 서버는 plr_json *시맨틱*은 재검증하지 않지만(클라이언트 신뢰), 채점에
    필요한 *구조*(obj_id+plr_json 키)는 지킨다. plr_json 키가 없는 행은 422."""
    bundle, _ = _make_bundle(tmp_path / "bundle-bad")
    bad = json.dumps({"obj_id": "a", "object_type": "person"}).encode()  # plr_json 키 누락
    r = client.post("/api/runs", headers=TOKEN,
                    data={"dataset": "runs_ds", "version_label": "bad"},
                    files={"attributes": ("attributes.jsonl", bad, "application/json"),
                           "surface": ("s.tgz", bundle, "application/gzip")})
    assert r.status_code == 422 and "line 1" in r.json()["detail"]
    # 불완전 run 디렉터리가 남지 않아야
    assert client.get("/api/runs", params={"dataset": "runs_ds",
                                           "all_history": True}).json()["runs"] == []


def test_uploaded_python_is_never_executed(client, dataset, tmp_path: Path) -> None:
    """US-006 하드 규칙의 회귀 가드: 표면 번들의 py는 저장·열람·diff 어디서도
    실행되지 않는다 — import 부수효과(센티널 파일 생성)가 절대 일어나면 안 됨."""
    sentinel = tmp_path / "PWNED"
    evil_dir = tmp_path / "bundle-evil"
    (evil_dir / "prompts" / "v").mkdir(parents=True)
    (evil_dir / "prompts" / "v" / "person.yaml").write_text("system: x\n")
    (evil_dir / "plr_parse.py").write_text(
        f"open({str(sentinel)!r}, 'w').write('pwned')\n")  # import 시 즉시 실행되는 코드

    r = client.post("/api/runs", headers=TOKEN,
                    data={"dataset": "runs_ds", "version_label": "evil"},
                    files={"attributes": ("attributes.jsonl", _attrs_bytes(), "application/json"),
                           "surface": ("s.tgz", _targz_dir(evil_dir), "application/gzip")})
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    # 열람·diff 경로까지 통과시켜도 실행 없음
    assert "pwned" in client.get(f"/r/{run_id}/surface/plr_parse.py").text
    client.get(f"/diff?a={run_id}&b={run_id}")
    assert not sentinel.exists(), "업로드된 py가 실행됨 — RCE 방지 규칙 위반!"


def test_label_correction_rescores_runs(client, dataset, tmp_path: Path) -> None:
    run_id = _submit(client, tmp_path, "tv3").json()["run_id"]
    # b의 정답을 male로 정정 → 두 예측 모두 정답이 됨
    new_labels = ('{"obj_id": "a", "labels": {"gender": "male"}}\n'
                  '{"obj_id": "b", "labels": {"gender": "male"}}\n')
    r = client.patch("/api/datasets/runs_ds/labels", headers=TOKEN,
                     files={"labels": ("labels.jsonl", new_labels.encode(), "application/json")})
    assert r.status_code == 200 and run_id in r.json()["rescored_runs"]
    runs = client.get("/api/runs", params={"dataset": "runs_ds"}).json()["runs"]
    assert runs[0]["metrics"]["gender"]["accuracy"] == 1.0, "재채점 반영"
