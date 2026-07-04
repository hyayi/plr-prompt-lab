"""plr-eval-server — FastAPI 엔트리.

배포 계약: uvicorn --workers 1 **고정** (쓰기 직렬화가 프로세스 내
asyncio.Lock이므로 다중 워커 금지). 기동 복구(reconcile_and_rebuild)는
startup에서 요청 수락 전에 완료된다.

실행: uvicorn server.app:app --host 0.0.0.0 --port 8890 --workers 1
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# lab 루트를 import 경로에 (evalkit/plr_schema 재사용 — 서버는 lab의 채점 코드를
# 직접 import한다; 업로드된 코드는 절대 import하지 않는다)
_LAB_ROOT = Path(__file__).resolve().parent.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from server import db as dbm
from server.storage import data_root, reconcile_and_rebuild

STATE: dict = {}  # {"root": Path, "conn": sqlite3.Connection, "rebuild": dict}


@asynccontextmanager
async def lifespan(app: FastAPI):
    root = data_root()
    conn = dbm.connect(root / "db.sqlite")
    dbm.init_schema(conn)
    # 요청 수락 전 정합 복구 — DB는 파일 리플레이로 재구축되는 파생 캐시.
    STATE["rebuild"] = reconcile_and_rebuild(root, conn)
    STATE["root"], STATE["conn"] = root, conn
    yield
    conn.close()
    STATE.clear()


app = FastAPI(title="plr-eval-server", lifespan=lifespan)


@app.middleware("http")
async def token_guard(request: Request, call_next):
    """변이 메서드는 X-Auth-Token 요구 (EVAL_SERVER_TOKEN 설정 시).
    조회(GET/HEAD)는 면제 — 사내 LAN 열람용."""
    token = os.environ.get("EVAL_SERVER_TOKEN", "")
    if token and request.method not in ("GET", "HEAD"):
        if request.headers.get("X-Auth-Token") != token:
            return JSONResponse({"error": "invalid or missing X-Auth-Token"},
                                status_code=401)
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "data_root": str(STATE.get("root", "")),
        "rebuild": STATE.get("rebuild", {}),
    }


# =====================================================================
# Datasets API (AC1, AC12)
# =====================================================================

import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from datetime import datetime  # noqa: E402

from fastapi import File, Form, HTTPException, UploadFile  # noqa: E402

from server.storage import (  # noqa: E402
    ExtractError,
    find_dataset_root,
    safe_extract_targz,
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\.]{0,63}$")


def _validate_report(ds_dir) -> tuple[bool, str]:
    """validate_dataset의 PASS/WARN/FAIL 리포트를 구조화해 반환
    (verbose=True여야 줄이 출력됨 — redirect_stdout으로 포집)."""
    from evalkit.validate import validate_dataset

    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = validate_dataset(ds_dir, verbose=True)
    return ok, buf.getvalue()


async def _save_upload(upload: UploadFile, dest) -> None:
    """multipart를 디스크로 스트리밍 (메모리 버퍼링 금지)."""
    with open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 1024):
            f.write(chunk)


@app.get("/api/datasets")
def list_datasets() -> dict:
    rows = STATE["conn"].execute(
        "SELECT name, created_at, created_by, n_crops, attrs_json FROM datasets "
        "ORDER BY name").fetchall()
    return {"datasets": [dict(r) | {"attributes": json.loads(r["attrs_json"])}
                         for r in rows]}


@app.post("/api/datasets", status_code=201)
async def register_dataset(
    name: str = Form(...),
    created_by: str = Form(""),
    archive: UploadFile = File(...),
) -> dict:
    root, conn = STATE["root"], STATE["conn"]
    if not _NAME_RE.fullmatch(name):
        raise HTTPException(400, "invalid dataset name (영숫자/._- 만, 64자 이내)")
    dest = root / "datasets" / name
    if dest.exists():
        raise HTTPException(
            409, f"dataset {name!r} already exists — 크롭/구성이 바뀌었으면 "
                 f"새 이름(예: {name}_v2)으로 등록하세요 (등록 후 크롭은 불변, "
                 f"라벨만 PATCH로 정정 가능)")

    tmp = Path(tempfile.mkdtemp(dir=root, prefix=".ingest-"))
    try:
        tar_path = tmp / "upload.tgz"
        await _save_upload(archive, tar_path)
        try:
            safe_extract_targz(tar_path, tmp / "x")
        except ExtractError as exc:
            raise HTTPException(400, f"archive rejected: {exc}")
        ds_root = find_dataset_root(tmp / "x")
        if ds_root is None:
            raise HTTPException(400, "manifest.yaml not found in archive "
                                     "(최상위 또는 단일 하위 디렉터리에 있어야 함)")
        ok, report = _validate_report(ds_root)
        if not ok:
            raise HTTPException(422, f"validate-dataset FAILED:\n{report}")

        created_at = datetime.now().isoformat(timespec="seconds")
        (ds_root / "registered.json").write_text(json.dumps(
            {"created_at": created_at, "created_by": created_by},
            ensure_ascii=False), encoding="utf-8")

        from evalkit.dataset import declared_attributes
        async with dbm.WRITE_LOCK:
            if dest.exists():
                raise HTTPException(409, f"dataset {name!r} already exists")
            shutil.move(str(ds_root), str(dest))  # 원자 이동 (같은 볼륨)
            n_crops = len(list((dest / "crops").glob("*.jpg")))
            dbm.upsert_dataset(conn, name, created_at, created_by,
                               n_crops, declared_attributes(dest))
        return {"name": name, "n_crops": n_crops, "report": report}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _rescore_dataset_runs(root, conn, dataset: str) -> list[str]:
    """등록 데이터셋의 전 run 재채점 — 라벨 정정 후 호출 (WRITE_LOCK 안에서).
    예측 원본(attributes.jsonl) 재사용이라 GPU 불필요."""
    from server.scoring import score_run
    from server.storage import read_json

    ds_dir = root / "datasets" / dataset
    rescored: list[str] = []
    rows = conn.execute("SELECT run_id FROM runs WHERE dataset=?", (dataset,)).fetchall()
    for row in rows:
        run_dir = root / "runs" / row["run_id"]
        attrs = run_dir / "attributes.jsonl"
        meta = read_json(run_dir / "meta.json")
        if not meta or not attrs.exists():
            continue
        res = score_run(ds_dir, attrs)
        (run_dir / "metrics.json").write_text(
            json.dumps(res, ensure_ascii=False), encoding="utf-8")
        dbm.upsert_run(conn, meta, res["attributes"])
        rescored.append(row["run_id"])
    return rescored


@app.patch("/api/datasets/{name}/labels")
async def correct_labels(
    name: str,
    labels: UploadFile = File(...),
    changed_by: str = Form(""),
) -> dict:
    root, conn = STATE["root"], STATE["conn"]
    ds_dir = root / "datasets" / name
    if not ds_dir.is_dir():
        raise HTTPException(404, f"dataset {name!r} not found")

    changed_at = datetime.now().isoformat(timespec="seconds")
    labels_path = ds_dir / "labels.jsonl"
    backup = ds_dir / f"labels.jsonl.bak-{changed_at.replace(':', '')}"

    async with dbm.WRITE_LOCK:
        shutil.copy2(labels_path, backup)
        await _save_upload(labels, labels_path)
        ok, report = _validate_report(ds_dir)
        if not ok:
            shutil.copy2(backup, labels_path)  # 원복
            backup.unlink(missing_ok=True)
            raise HTTPException(422, f"correction rejected (원복됨):\n{report}")

        old = set(backup.read_text(encoding="utf-8").splitlines())
        new = set(labels_path.read_text(encoding="utf-8").splitlines())
        diff_summary = f"-{len(old - new)} +{len(new - old)} lines"

        rescored = _rescore_dataset_runs(root, conn, name)

        dbm.add_label_audit(conn, name, changed_at, changed_by, diff_summary, rescored)
        with open(ds_dir / "label_audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "changed_at": changed_at, "changed_by": changed_by,
                "diff_summary": diff_summary, "rescored_runs": rescored,
                "backup": backup.name,
            }, ensure_ascii=False) + "\n")

    return {"dataset": name, "diff_summary": diff_summary,
            "rescored_runs": rescored, "report": report}
