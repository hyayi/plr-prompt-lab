#!/usr/bin/env bash
# GitHub 게시 스크립트 — 스냅샷 1커밋만 push한다.
#
# 왜 스냅샷인가: 로컬 master의 과거 이력에 시크릿(구 DB 비밀번호)이 남아 있어
# `git push origin master`(전체 이력)는 영구 금지. 이 스크립트는 현재 트리를
# 이력 없는 단일 커밋으로 만들어 원격을 덮어쓴다.
#
# 사용법:
#   scripts/publish.sh ["커밋 메시지"] [원격브랜치=master]
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) 작업 트리 시크릿 스캔 (push 전 마지막 방어선)
#    PGPASSWORD=<리터럴>만 잡는다 — f-string {var}·셸 $VAR 참조는 정상 코드.
#    이 스크립트 자신은 패턴 문자열을 담고 있으므로 스캔에서 제외.
PATTERN='OLD_SECRET_PATTERN|PGPASSWORD=[A-Za-z0-9]'
if git grep -qiE "${PATTERN}" -- . ':!scripts/publish.sh' 2>/dev/null; then
  echo "ABORT: 시크릿 패턴이 작업 트리에 있습니다 — push 중단" >&2
  git grep -niE "${PATTERN}" -- . ':!scripts/publish.sh' | head -5 >&2
  exit 1
fi

# 2) 추적된 이미지가 합성 템플릿 2장뿐인지 (실 크롭 커밋 방지)
IMGS=$(git ls-files | grep -icE '\.(jpg|jpeg|png)$' || true)
if [ "${IMGS}" -gt 2 ]; then
  echo "ABORT: 추적된 이미지가 ${IMGS}장 — 실 크롭이 커밋됐는지 확인하세요" >&2
  git ls-files | grep -iE '\.(jpg|jpeg|png)$' >&2
  exit 1
fi

# 3) 스냅샷 1커밋 생성 → 원격 덮어쓰기
MSG="${1:-plr-prompt-lab snapshot ($(git rev-parse --short master))}"
REMOTE_BRANCH="${2:-master}"
SNAP=$(git commit-tree "master^{tree}" -m "${MSG}")
git branch -f public-snapshot "${SNAP}"
git push -f -u origin "public-snapshot:${REMOTE_BRANCH}"
echo "published: ${SNAP} → origin/${REMOTE_BRANCH} (이력 1커밋)"
