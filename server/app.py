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
