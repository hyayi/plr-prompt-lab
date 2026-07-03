#!/usr/bin/env bash
# GitHub 게시 스크립트 — 가드 통과 시 master 전체 이력을 push한다.
# (이력은 2026-07-03 filter-branch로 시크릿 치환 완료 — 스냅샷 불필요.
#  가드 패턴은 리터럴 노출을 피하려고 조각 결합으로 만든다.)
#
# 사용법: scripts/publish.sh [원격브랜치=master]
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) 시크릿 스캔 — 구 DB 비밀번호 리터럴 / 하드코딩 PGPASSWORD / 내부 IP
P1='ziovision@'"1234"
P2='PGPASSWORD=[A-Za-z0-9]'
P3='192\.168\.[0-9]+\.[0-9]+'
if git grep -qiE "${P1}|${P2}|${P3}" -- . ':!scripts/publish.sh' 2>/dev/null; then
  echo "ABORT: 시크릿/내부IP 패턴이 작업 트리에 있습니다 — push 중단" >&2
  git grep -niE "${P1}|${P2}|${P3}" -- . ':!scripts/publish.sh' | head -5 >&2
  exit 1
fi

# 2) 추적 이미지 검사 — 합성 템플릿 2장 초과면 실 크롭 커밋 의심
IMGS=$(git ls-files | grep -icE '\.(jpg|jpeg|png)$' || true)
if [ "${IMGS}" -gt 2 ]; then
  echo "ABORT: 추적된 이미지 ${IMGS}장 — 실 크롭이 커밋됐는지 확인하세요" >&2
  git ls-files | grep -iE '\.(jpg|jpeg|png)$' >&2
  exit 1
fi

# 3) push
REMOTE_BRANCH="${1:-master}"
git push -u origin "master:${REMOTE_BRANCH}"
echo "published: master → origin/${REMOTE_BRANCH}"
