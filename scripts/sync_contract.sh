#!/usr/bin/env bash
# 공유 계약 파일을 lab repo(원천) → server repo 로 복사하고 manifest 를 동기화한다.
# 사용: scripts/sync_contract.sh /path/to/plr-eval-server
#
# lab repo 에서 공유 파일(evalkit/dataset·provenance·validate, plr_schema.py,
# schema/vocab.yaml)을 고친 뒤 실행하면, 서버 레포의 복본과 manifest.json 이
# lab 과 byte-identical 이 된다. 이후 두 레포에서 test_contract_parity 가 초록.
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ROOT="${1:-}"

if [[ -z "$SERVER_ROOT" ]]; then
  echo "usage: $0 <server-repo-root>" >&2
  exit 2
fi
if [[ ! -d "$SERVER_ROOT" ]]; then
  echo "error: server repo root not found: $SERVER_ROOT" >&2
  exit 1
fi

# 1) lab 에서 manifest 재생성 (원천 기준 최신 해시)
python3 "$LAB_ROOT/contract/gen_manifest.py" >/dev/null

# 2) SHARED_FILES 를 파이썬으로 열거해 파일별 복사 (경로에 공백 없다고 가정)
mapfile -t SHARED < <(python3 - "$LAB_ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/contract")
from shared_files import SHARED_FILES
print("\n".join(SHARED_FILES))
PY
)

for rel in "${SHARED[@]}"; do
  src="$LAB_ROOT/$rel"
  dst="$SERVER_ROOT/$rel"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  echo "  synced  $rel"
done

# 3) contract/ 툴킷(manifest+목록+테스트)도 서버로 전파 → 서버가 독립적으로 파리티 검증
mkdir -p "$SERVER_ROOT/contract" "$SERVER_ROOT/tests"
cp "$LAB_ROOT/contract/manifest.json"   "$SERVER_ROOT/contract/manifest.json"
cp "$LAB_ROOT/contract/shared_files.py" "$SERVER_ROOT/contract/shared_files.py"
cp "$LAB_ROOT/contract/gen_manifest.py" "$SERVER_ROOT/contract/gen_manifest.py"
cp "$LAB_ROOT/contract/CONTRACT.md"     "$SERVER_ROOT/contract/CONTRACT.md"
cp "$LAB_ROOT/tests/test_contract_parity.py" "$SERVER_ROOT/tests/test_contract_parity.py"
echo "  synced  contract/ + tests/test_contract_parity.py"

echo "[sync_contract] done → $SERVER_ROOT"
echo "  다음: (cd '$SERVER_ROOT' && python3 -m pytest tests/test_contract_parity.py -q)"
