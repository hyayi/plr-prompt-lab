"""AC9 e2e: 등록 → run 산출물 → 제출 → 리더보드 → 재제출 → 라벨 정정 → 재기동.

lab 클라이언트의 실제 multipart 인코더와 실제 표면 번들(surface_relpaths
전체)을 사용 — hash_verified=True는 "run 시점 지문 == 서버 재계산"이
end-to-end로 성립함을 증명한다. No GPU, no network (TestClient).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from plr_parse import parse_plr_response  # noqa: E402
from runners.client import (  # noqa: E402
    build_surface_bundle,
    multipart_body,
    targz_dir,
    write_run_provenance,
)

TOKEN = {"X-Auth-Token": "sekrit"}

_YAML = ("target: person\ngender: {g}\ngender_reason: t\nage: adult\n"
         "outfit: two_piece\nupper:\n  color: black\n  type: jacket\n"
         "lower:\n  color: black\n  type: pants\naction: standing\n"
         "military: civilian\nmargins: {{gender: {m}}}")


def _make_local_run(base: Path) -> Path:
    """lab run 산출물 시뮬레이션: 데이터셋 + attributes.jsonl + run_provenance."""
    from PIL import Image

    base.mkdir(parents=True)
    (base / "crops").mkdir()
    for oid in ("a", "b"):
        Image.new("RGB", (60, 90), (90, 90, 90)).save(str(base / "crops" / f"{oid}.jpg"))
    (base / "manifest.yaml").write_text(
        "n: 2\ncreated: '2026-07-04'\nsource_note: e2e\nattributes:\n  gender: {}\n")
    (base / "labels.jsonl").write_text(
        '{"obj_id": "a", "labels": {"gender": "male"}}\n'
        '{"obj_id": "b", "labels": {"gender": "female"}}\n')
    rows = [{"obj_id": "a", "plr_json": parse_plr_response(_YAML.format(g="male", m=0.9), hint="person")},
            {"obj_id": "b", "plr_json": parse_plr_response(_YAML.format(g="male", m=0.3), hint="person")}]
    (base / "attributes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    write_run_provenance(base, _LAB_ROOT, model="mock", version="e2e_v1",
                         max_tokens=512, temperature=0.0)
    return base


def _post_multipart(client, url: str, fields: dict, files: dict) -> dict:
    """runners.client의 인코더를 실제 서버 파서에 통과시킨다."""
    body, ctype = multipart_body(fields, files)
    r = client.post(url, content=body,
                    headers=TOKEN | {"Content-Type": ctype})
    assert r.status_code in (200, 201), f"{r.status_code}: {r.text}"
    return r.json()


def test_full_acceptance_scenario(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_SERVER_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_SERVER_TOKEN", "sekrit")
    from server.app import app

    local = _make_local_run(tmp_path / "local" / "e2e_ds")
    bundle = build_surface_bundle(_LAB_ROOT)

    with TestClient(app) as client:
        # ① 데이터셋 등록 (lab dataset-push와 동일 페이로드)
        _post_multipart(client, "/api/datasets", {"name": "e2e_ds"},
                        {"archive": ("d.tgz", targz_dir(local, "e2e_ds"),
                                     "application/gzip")})

        # ② run 제출 — 실제 lab 번들 + run 시점 지문 → 검증 성공해야 함
        files = {
            "attributes": ("attributes.jsonl",
                           (local / "attributes.jsonl").read_bytes(), "application/json"),
            "surface": ("surface.tgz", bundle, "application/gzip"),
            "provenance": ("run_provenance.json",
                           (local / "run_provenance.json").read_bytes(), "application/json"),
        }
        res = _post_multipart(client, "/api/runs",
                              {"dataset": "e2e_ds", "version_label": "e2e_v1",
                               "submitted_by": "e2e"}, files)
        run1 = res["run_id"]
        assert res["hash_verified"] is True, \
            "실번들·실지문의 end-to-end 무결성 검증이 성립해야 한다"
        assert res["attributes"]["gender"]["accuracy"] == 0.5

        # ③ 재제출 = append (AC3)
        run2 = _post_multipart(client, "/api/runs",
                               {"dataset": "e2e_ds", "version_label": "e2e_v1"},
                               files)["run_id"]
        assert run2 != run1

        # ④ 리더보드: 기본=최신 1행, 전체=2행, 웹 페이지 렌더
        assert len(client.get("/api/runs", params={"dataset": "e2e_ds"}).json()["runs"]) == 1
        assert len(client.get("/api/runs", params={"dataset": "e2e_ds",
                                                   "all_history": True}).json()["runs"]) == 2
        assert run2 in client.get("/d/e2e_ds").text

        # ⑤ 라벨 정정 → 전 run 자동 재채점 (AC12)
        new_labels = ('{"obj_id": "a", "labels": {"gender": "male"}}\n'
                      '{"obj_id": "b", "labels": {"gender": "male"}}\n')
        body, ctype = multipart_body({}, {"labels": ("labels.jsonl",
                                                     new_labels.encode(), "application/json")})
        r = client.patch("/api/datasets/e2e_ds/labels", content=body,
                         headers=TOKEN | {"Content-Type": ctype})
        assert r.status_code == 200, r.text
        assert set(r.json()["rescored_runs"]) == {run1, run2}
        runs = client.get("/api/runs", params={"dataset": "e2e_ds"}).json()["runs"]
        assert runs[0]["metrics"]["gender"]["accuracy"] == 1.0

    # ⑥ 재기동: DB는 파생 캐시 — 파일 리플레이로 동일 상태 복원
    with TestClient(app) as client2:
        runs = client2.get("/api/runs", params={"dataset": "e2e_ds",
                                                "all_history": True}).json()["runs"]
        assert {r["run_id"] for r in runs} == {run1, run2}
        assert client2.get("/health").json()["rebuild"]["quarantined"] == []


def test_submit_pull_all_or_nothing(tmp_path: Path, monkeypatch) -> None:
    """pull은 metrics+report+gallery 3종을 로컬로; 존재하지 않는 run은 부분 dir 안 남김."""
    monkeypatch.setenv("EVAL_SERVER_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_SERVER_TOKEN", "sekrit")
    from server.app import app
    from runners.client import build_surface_bundle, multipart_body, targz_dir

    local = _make_local_run(tmp_path / "local" / "pull_ds")
    bundle = build_surface_bundle(_LAB_ROOT)
    with TestClient(app) as client:
        _post_multipart(client, "/api/datasets", {"name": "pull_ds"},
                        {"archive": ("d.tgz", targz_dir(local, "pull_ds"), "application/gzip")})
        run_id = _post_multipart(client, "/api/runs",
            {"dataset": "pull_ds", "version_label": "pv1"},
            {"attributes": ("attributes.jsonl", (local / "attributes.jsonl").read_bytes(), "application/json"),
             "surface": ("surface.tgz", bundle, "application/gzip"),
             "provenance": ("run_provenance.json", (local / "run_provenance.json").read_bytes(), "application/json"),
            })["run_id"]

        # pull 3종을 TestClient로 직접 검증(실 네트워크 대신)
        for path in (f"/api/runs/{run_id}", f"/api/runs/{run_id}/report.html",
                     f"/api/runs/{run_id}/gallery.html"):
            assert client.get(path).status_code == 200, path
        # 없는 run은 404 → all-or-nothing으로 저장 안 됨
        assert client.get("/api/runs/nope/report.html").status_code == 404


def test_pull_artifacts_roundtrip_and_partial_cleanup(tmp_path: Path, monkeypatch) -> None:
    """pull_artifacts 자체의 all-or-nothing 계약(라우트가 아니라 클라이언트 로직):
    ① 3종 전부 성공 → out_dir에 착지  ② 중간 실패 → SystemExit + temp/out_dir 잔재 0."""
    import io
    import urllib.error

    from runners import client as C

    out = tmp_path / "pulled"

    # ① happy round-trip — _get를 스텁해 3종 모두 바이트 반환
    def _ok(url: str, token: str) -> bytes:
        return b"{}" if url.endswith(f"/api/runs/r1") else b"<html></html>"
    monkeypatch.setattr(C, "_get", _ok)
    got = C.pull_artifacts("http://x", "r1", out, "")
    assert got == ["metrics.json", "report.html", "gallery.html"]
    assert {p.name for p in out.iterdir()} == set(got)
    assert not any(p.name.startswith(".pull-") for p in tmp_path.iterdir()), "temp dir 잔재"

    # ② partial failure — 2번째(report.html)에서 404 → 아무것도 저장 안 됨, temp 정리
    out2 = tmp_path / "pulled2"

    def _fail_on_report(url: str, token: str) -> bytes:
        if url.endswith("/report.html"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None,
                                         io.BytesIO(b"no such run"))
        return b"{}"
    monkeypatch.setattr(C, "_get", _fail_on_report)
    with pytest.raises(SystemExit, match="report.html"):
        C.pull_artifacts("http://x", "r1", out2, "")
    assert not out2.exists(), "부분 실패 시 out_dir 생성 안 됨"
    assert not any(p.name.startswith(".pull-") for p in tmp_path.iterdir()), "temp dir 잔재"
