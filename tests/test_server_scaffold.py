"""서버 스캐폴드 계약 (US-003): 저장 계층·DB·보안 해제·기동 복구.

No GPU, no network. fastapi가 없으면 app 테스트만 skip.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
import sys
import tarfile
from pathlib import Path

import pytest

_LAB_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_LAB_ROOT))

from server import db as dbm  # noqa: E402
from server.storage import (  # noqa: E402
    ExtractError,
    new_run_id,
    reconcile_and_rebuild,
    safe_extract_targz,
)


def _tar_bytes(entries: dict[str, bytes], *, evil_name: str | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if evil_name:
            info = tarfile.TarInfo(name=evil_name)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    return buf.getvalue()


def test_run_id_is_sortable() -> None:
    rid = new_run_id()
    assert re.fullmatch(r"r\d{8}-\d{6}-[0-9a-f]{6}", rid), rid


def test_safe_extract_rejects_traversal_and_absolute(tmp_path: Path) -> None:
    for evil in ("../escape.txt", "/etc/pwned"):
        p = tmp_path / "a.tgz"
        p.write_bytes(_tar_bytes({"ok.txt": b"hi"}, evil_name=evil))
        with pytest.raises(ExtractError):
            safe_extract_targz(p, tmp_path / "out")


def test_safe_extract_enforces_size_cap(tmp_path: Path) -> None:
    p = tmp_path / "big.tgz"
    p.write_bytes(_tar_bytes({"big.bin": b"A" * 1000}))
    with pytest.raises(ExtractError, match="cap"):
        safe_extract_targz(p, tmp_path / "out", max_bytes=100)


def test_db_schema_and_wal(tmp_path: Path) -> None:
    conn = dbm.connect(tmp_path / "db.sqlite")
    dbm.init_schema(conn)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"datasets", "runs", "metrics", "label_audit"} <= tables


def test_reconcile_quarantines_orphans_and_rebuilds(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    for sub in ("datasets", "runs", "quarantine"):
        (root / sub).mkdir()
    # 정상 run: meta+metrics 존재 → DB 재구축
    good = root / "runs" / "r20260704-000000-aaaaaa"
    good.mkdir()
    (good / "meta.json").write_text(json.dumps({
        "run_id": good.name, "dataset": "ds1", "version_label": "v1",
        "submitted_at": "2026-07-04T00:00:00"}))
    (good / "metrics.json").write_text(json.dumps({
        "gender": {"accuracy": 0.9, "macro_f1": 0.88, "n": 10, "correct": 9,
                    "pred_unknown": {"rate": 0.0}, "bias": None}}))
    # 고아 run: metrics 없음(크래시 잔재) → 격리
    orphan = root / "runs" / "r20260704-000001-bbbbbb"
    orphan.mkdir()
    (orphan / "meta.json").write_text("{}")

    conn = dbm.connect(root / "db.sqlite")
    dbm.init_schema(conn)
    result = reconcile_and_rebuild(root, conn)

    assert good.name in result["runs"]
    assert orphan.name in result["quarantined"]
    assert (root / "quarantine" / orphan.name).exists()
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (good.name,)).fetchone()
    assert row["dataset"] == "ds1"
    acc = conn.execute(
        "SELECT value FROM metrics WHERE run_id=? AND attribute='gender' AND metric='accuracy'",
        (good.name,)).fetchone()[0]
    assert acc == 0.9


def test_app_boots_with_token_guard(tmp_path: Path, monkeypatch) -> None:
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    monkeypatch.setenv("EVAL_SERVER_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_SERVER_TOKEN", "sekrit")
    from server.app import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["ok"] is True
        # 변이 메서드는 토큰 없이 401 (등록 라우트가 아직 없어도 미들웨어가 선행)
        r = client.post("/api/datasets")
        assert r.status_code == 401
