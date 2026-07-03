# Skill: co-change (동행 수정 지침)

## Purpose / When to Use

PLR 표면(프롬프트·어휘·파서·스키마·전처리)의 **무엇이든 수정하기 전에** 이
스킬을 열어, 해당 변경 유형의 행을 찾아 **동행 수정 목록 전체**를 적용한다.
이 스킬이 존재하는 이유는 아래 "사고 사례" 3건 — 전부 "같이 고쳐야 하는
것을 사람 기억에 맡겨서" 생긴 조용한 드리프트였다.

## 0. 공통 절차 — 두 단계로 나뉜다

**개발/실험 단계 (lab만 있으면 됨 — core/ir 불필요):**

1. lab 안에서 자유롭게 수정·실험한다 (새 프롬프트는 새 버전 디렉터리,
   knob 조합은 `configs/<exp>.yaml`).
2. lab 테스트만 돌린다: `python3 -m pytest tests/ -q`.
3. lab에 커밋. **core/ir 동기화는 이 단계에서 하지 않는다** — 실험 중 표면은
   의도적으로 core/ir과 달라진 상태이고, 그게 정상이다 (`lab port`의 diff와
   SEED stale 경고는 "달라져 있음"의 가시화이지 오류가 아니다).

**승격/최종 반영 단계 (core/ir 보유자만, 실험 승자가 확정된 뒤 1회):**

1. **lab → core/ir 바이트 동기화** — 승격 대상 표면 파일(`prompts/**`,
   `schema/vocab.yaml`, `plr_prompts.py`, `plr_parse.py`,
   `plr_core.py`(build_messages 1곳 의도적 분기), `plr_schema.py`,
   `preprocess.py`)을 복사하고 `diff`로 byte-equal 확인.
2. **양쪽 테스트 스위트** — lab + core/ir
   (`/home/ziovision/deploy_2026/core/ir/.venv/bin/python -m pytest tests/ -q`).
3. **양쪽 커밋** (push는 명시 요청 시에만).
4. 운영 반영은 컨테이너 재시작(=배포)이며 별도 결정 사항 — 파일 수정만으로는
   실행 중인 프로세스에 영향 없음.

아래 매트릭스의 동행 수정 항목 중 core/ir 쪽 파일(query_parser·scoring·
indexing 등)을 지목하는 것들도 마찬가지로 **승격 단계의 체크리스트**다 —
개발 단계에서는 "승격 시 필요한 목록"으로 기록만 해 둔다.

## 변경 유형 → 동행 수정 매트릭스

### ① 어휘 추가/삭제 (`schema/vocab.yaml`)

자동 동행(손댈 것 없음): enum 상수 → 프롬프트 주입(`{colors}` 등 placeholder)
· 파서 enum 강제(`_coerce_topk_labels`) · 스키마 검증(enum 파생).
**수동으로 따라가야 하는 것:**
- 프롬프트 yaml에 그 값들이 **literal로 박힌 답 슬롯** (예: `sleeve: <long|short>`
  — 주입이 아니라 손글씨). 선택지를 바꾸려면 yaml 텍스트 수정 = 프롬프트
  바이트 변경 → **새 버전 디렉터리 + PROMPT_VERSION bump + 재인덱싱**.
- 주입 placeholder를 쓰는 enum이라도, 주입 결과가 프롬프트 바이트를 바꾸므로
  운영 반영에는 **PROMPT_VERSION bump**가 필요하다 (bump 없으면 기존 인덱스와
  섞임).
- enum 내용을 **박제한 테스트** (예: `tests/test_plr_schema.py`의
  `set(...) == {...}` 단언) — 의도된 확장이면 테스트를 갱신.
- **검색 쿼리 매핑** (core/ir `query_parser.py`/`image_retrieval.py`의
  한국어→enum 매핑, 예: 나시/민소매→sleeveless) — 검색어로 쓰일 값이면 배선.
- 색 enum이면 그룹(`groups:`)에도 소속시킬 것 — 그룹 없는 색은 하드필터에서
  고아가 된다.

### ② 새 답 슬롯 추가 (프롬프트 출력 구조 변경)

1. 프롬프트 yaml — **새 버전 디렉터리**로 (`skills/author-prompt` 계약 준수),
   margins 블록에 슬롯 margin도 추가.
2. `plr_parse.py` — 슬롯 추출 + 정규화(enum 강제) + default.
3. `plr_schema.py` — PERSON/VEHICLE_SCHEMA에 선언
   (**잊으면 `tests/test_schema_declares_parser_keys.py`가 빨간불** — 이
   테스트의 `_MAX_*_YAML`에도 새 슬롯을 채워 넣어야 사각지대가 안 생긴다).
4. 평가 대상 속성이면 `evalkit/dataset.py`의 `PRESET_SPECS`/`attribute_spec`.
5. 검색 축이면 ①의 쿼리 매핑 + 게이트(scoring) 배선, 구행 wildcard-pass
   (버전 게이트) 처리.
6. PROMPT_VERSION bump = 재인덱싱 트리거.

### ③ 프롬프트 문구만 개선 (출력 구조 불변)

- 새 버전 디렉터리 (`prompts/<new>/`) — 기존 버전 yaml은 역사 보존, 절대 덮어쓰지
  않는다.
- lab에서 `configs/<exp>.yaml`로 A/B (`lab experiment run`).
- 승격 시: `plr_prompts.PROMPT_VERSION_YAML_COT` bump + query_parser 블록 포함
  확인 (`skills/author-prompt`의 promotion 절 참조).

### ④ 파서 로직 변경 (`plr_parse.py`)

- emit 키가 늘면 ②-3 (스키마 선언 — parity test가 강제).
- 하드코딩 금지: 허용값 집합은 enum 상수에서 파생시킬 것
  (`_norm_sleeve`가 교훈 — 하드코딩 `{"long","short"}`이 vocab과 따로 놀았다).
- 관용 처리(펜스 제거·평면 폴백·enum 강제)는 구행 호환이 아니라 **상시 방어**
  — 정리 대상으로 오인해 제거하지 말 것.

### ⑤ 스키마(구조) 변경 (`plr_schema.py`의 *_SCHEMA)

- 필드 **추가**는 additive라 안전. 필드 **제거**는 구행 호환 문제 — 운영 DB의
  전 행이 신버전으로 재인덱싱된 후에만 (`SESSION_HANDOFF.md` §7.5 백로그 절차).
- required에 추가하려면 파서가 그 필드를 **항상** 채우는지 먼저 확인
  (validate 실패 = 행이 DLQ행).

### ⑥ 전처리/샘플링 변경 (`preprocess.py`, max_tokens/temperature)

- 모델 인풋 바이트가 바뀌므로 결과 비교 불가 → 실험은 config로, 운영 반영은
  PROMPT_VERSION bump와 함께.

## 이미 있는 자동 가드 (믿되, 우회하지 말 것)

| 가드 | 잡는 드리프트 |
| --- | --- |
| `schema/vocab.yaml` 단일 원천 | 주입·파싱·검증 어휘 불일치 (구조적으로 불가능화) |
| `tests/test_schema_declares_parser_keys.py` | 파서 emit ↔ 스키마 선언 누락 |
| core/ir `tests/test_prompt_source_parity.py` | 프롬프트 원천 이원화 드리프트 |
| `test_live_prompt_never_offers_unknown` | forced-commit 위반 (unknown 선택지) |
| `lab port` + `prompt_hash` + SEED stale 경고 | lab ↔ core/ir 표면 불일치 |

## 사고 사례 (이 스킬의 존재 이유)

1. **ca1a922** — military_olive 힌트를 yaml에만 넣고 상수에 안 넣어 LIVE
   프롬프트가 힌트를 영영 못 받음 → source-parity 테스트 탄생.
2. **sleeveless 이관 유실 (2026-07)** — deploy_2026→ziomilitary 이관에서
   enum의 sleeveless만 소실(쿼리 매핑은 생존), 기존 테스트가 유실 상태를
   정답으로 박제 → vocab 복원 + `_norm_sleeve` vocab-driven화.
3. **sleeve/reason 미선언 (2026-07)** — 파서는 emit하는데 스키마 미명시,
   JSON Schema의 "미명시 허용" 기본값 탓에 조용한 검증 사각지대 →
   emit-parity 테스트 탄생.
