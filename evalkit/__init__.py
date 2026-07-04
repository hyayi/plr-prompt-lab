"""Eval-support layer.

- dataset·provenance = **공유 계약**: 클라이언트 run/submit + 서버 채점이 함께 씀
  (두 레포에 byte-identical vendored — contract/CONTRACT.md).
- validate = **lab 전용**: 라벨 어휘(enum) 검증(`lab validate-dataset`). 서버는 이
  검증을 신뢰(SPEC:41)하므로 vendoring 안 함 — plr_schema/vocab.yaml 도 lab 전용.
- scoring·report·gallery = 서버 전용이라 별도 레포(~/plr-eval-server)로 이관됨."""
