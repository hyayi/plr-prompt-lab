"""웹 UI 스모크 (US-008 / AC5·6·8·11): 페이지 렌더·배지·diff·업로드 폼.

No GPU, no network.
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


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EVAL_SERVER_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_SERVER_TOKEN", "sekrit")
    from server.app import app
    with TestClient(app) as c:
        yield c


def _seed(client, tmp_path: Path) -> tuple[str, str]:
    """데이터셋 1 + run 2(내용 다른 프롬프트, 하나는 해시 미검증) 심기."""
    from PIL import Image

    src = tmp_path / "src" / "web_ds"
    (src / "crops").mkdir(parents=True)
    for oid in ("a", "b"):
        Image.new("RGB", (60, 90), (90, 90, 90)).save(str(src / "crops" / f"{oid}.jpg"))
    (src / "manifest.yaml").write_text(
        "n: 2\ncreated: '2026-07-04'\nsource_note: t\nattributes:\n  gender: {}\n")
    (src / "labels.jsonl").write_text(
        '{"obj_id": "a", "labels": {"gender": "male"}}\n'
        '{"obj_id": "b", "labels": {"gender": "female"}}\n')
    assert client.post("/api/datasets", headers=TOKEN, data={"name": "web_ds"},
                       files={"archive": ("d.tgz", _targz_dir(src, "web_ds"),
                                          "application/gzip")}).status_code == 201

    def submit(version: str, prompt_text: str, good_hash: bool) -> str:
        from evalkit.provenance import prompt_hash
        b = tmp_path / f"bundle-{version}"
        (b / "prompts" / "v").mkdir(parents=True, exist_ok=True)
        (b / "prompts" / "v" / "person.yaml").write_text(prompt_text)
        claimed = prompt_hash(b) if good_hash else "deadbeef0000"
        rows = [{"obj_id": "a", "plr_json": parse_plr_response(_YAML.format(g="male"), hint="person")},
                {"obj_id": "b", "plr_json": parse_plr_response(_YAML.format(g="male"), hint="person")}]
        attrs = "\n".join(json.dumps(r) for r in rows).encode()
        prov = json.dumps({"surface_hash": claimed, "lab_sha": "abc", "git_dirty": False,
                           "model": "mock", "max_tokens": 512, "temperature": 0.0,
                           "reason": "on"}).encode()
        r = client.post("/api/runs", headers=TOKEN,
                        data={"dataset": "web_ds", "version_label": version},
                        files={"attributes": ("a.jsonl", attrs, "application/json"),
                               "surface": ("s.tgz", _targz_dir(b), "application/gzip"),
                               "provenance": ("p.json", prov, "application/json")})
        assert r.status_code == 201, r.text
        return r.json()["run_id"]

    r1 = submit("wv1", "system: prompt ONE\n", True)
    r2 = submit("wv2", "system: prompt TWO\n", False)  # 해시 미검증 배지 대상
    return r1, r2


def test_pages_render(client, tmp_path: Path) -> None:
    r1, r2 = _seed(client, tmp_path)

    home = client.get("/").text
    assert "web_ds" in home and "/d/web_ds" in home

    lb = client.get("/d/web_ds").text
    assert "gender acc" in lb and "sortable" in lb
    assert "미검증" in lb, "해시 불일치 run에 ⚠ 배지"

    run = client.get(f"/r/{r1}").text
    assert "confusion" in run and "prompts/v/person.yaml" in run

    raw = client.get(f"/r/{r1}/surface/prompts/v/person.yaml").text
    assert "prompt ONE" in raw
    assert client.get(f"/r/{r1}/surface/../../meta.json").status_code == 404, "경로 탈출 차단"

    diff = client.get(f"/diff?a={r1}&b={r2}").text
    assert "person.yaml" in diff and "prompt ONE" in diff and "prompt TWO" in diff

    up = client.get("/upload").text
    assert "attributes.jsonl" in up and "FormData" in up
