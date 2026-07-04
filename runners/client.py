"""client — 평가 서버 제출 클라이언트 (stdlib만 사용, 의존성 0).

lab dataset-push : 데이터셋 디렉터리를 tar.gz로 서버에 등록
lab submit       : run 산출물(attributes.jsonl + run_provenance.json)과
                   표면 번들(surface_relpaths 전체)을 서버에 제출

표면 번들 = lab `provenance.surface_relpaths()`가 정의하는 파일 집합 그대로 —
서버가 같은 알고리즘(prompt_hash)으로 재계산해 run 시점 지문과 대조한다.
"""
from __future__ import annotations

import io
import json
import os
import secrets
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


def multipart_body(fields: dict[str, str],
                   files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """(body, content_type) — files: {필드명: (파일명, bytes, MIME)}."""
    boundary = "----plrlab" + secrets.token_hex(8)
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        out.write(str(value).encode() + b"\r\n")
    for name, (fname, data, mime) in files.items():
        out.write(f"--{boundary}\r\n".encode())
        out.write(f'Content-Disposition: form-data; name="{name}"; '
                  f'filename="{fname}"\r\n'.encode())
        out.write(f"Content-Type: {mime}\r\n\r\n".encode())
        out.write(data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def _post(url: str, body: bytes, content_type: str, token: str,
          method: str = "POST") -> dict:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", content_type)
    if token:
        req.add_header("X-Auth-Token", token)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"server {e.code}: {detail}")


def targz_dir(src: Path, arcname: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src, arcname=arcname)
    return buf.getvalue()


def build_surface_bundle(lab_root: Path) -> bytes:
    """surface_relpaths 전체(prompts/vocab/configs/파서·코어·스키마·전처리 py)를
    상대경로 그대로 tar.gz — 서버 보관·diff·해시 재계산용 (서버는 절대 실행 안 함)."""
    from evalkit.provenance import surface_relpaths

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in surface_relpaths(lab_root, include_exp_configs=True):
            tf.add(lab_root / rel, arcname=rel)
    return buf.getvalue()


def dataset_push(server: str, dataset_dir: Path, name: str,
                 token: str, created_by: str = "") -> dict:
    body, ctype = multipart_body(
        {"name": name, "created_by": created_by},
        {"archive": (f"{name}.tgz", targz_dir(dataset_dir, name), "application/gzip")},
    )
    return _post(server.rstrip("/") + "/api/datasets", body, ctype, token)


def submit_run(server: str, dataset_name: str, run_dir: Path, version: str,
               token: str, submitted_by: str = "",
               lab_root: Path | None = None) -> dict:
    lab_root = lab_root or Path(__file__).resolve().parent.parent
    attrs = run_dir / "attributes.jsonl"
    if not attrs.exists():
        raise SystemExit(f"{attrs} 없음 — 먼저 `lab run`으로 산출물을 만드세요")
    files = {
        "attributes": ("attributes.jsonl", attrs.read_bytes(), "application/json"),
        "surface": ("surface.tgz", build_surface_bundle(lab_root), "application/gzip"),
    }
    prov = run_dir / "run_provenance.json"
    if prov.exists():
        files["provenance"] = ("run_provenance.json", prov.read_bytes(), "application/json")
    else:
        print("WARNING: run_provenance.json 없음 — 무결성 대조 없이 제출됩니다 "
              "(서버에서 unverified 표시)")
    body, ctype = multipart_body(
        {"dataset": dataset_name, "version_label": version,
         "submitted_by": submitted_by},
        files,
    )
    return _post(server.rstrip("/") + "/api/runs", body, ctype, token)


def write_run_provenance(dataset_dir: Path, lab_root: Path, *,
                         model: str, version: str,
                         max_tokens: int | None = None,
                         temperature: float | None = None) -> Path:
    """run 종료 시점의 표면 지문 + 실행 파라미터 기록 — 제출 무결성 대조의 기준.
    (제출 시점 계산은 번들과 항상 일치해 무의미 — 반드시 run 시점에 찍는다.)"""
    import subprocess

    from evalkit.provenance import prompt_hash

    lab_sha, dirty = "", False
    try:
        lab_sha = subprocess.run(
            ["git", "-C", str(lab_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(lab_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:  # noqa: BLE001 — git 없는 배포 환경 허용
        pass

    prov = {
        "surface_hash": prompt_hash(lab_root),
        "lab_sha": lab_sha,
        "git_dirty": dirty,
        "model": model,
        "version": version,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reason": os.environ.get("IR_PLR_REASON", "off"),
    }
    out = dataset_dir / "run_provenance.json"
    out.write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
