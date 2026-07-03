# plr-prompt-lab — 세션 핸드오프 (2026-07-02)

> 새 세션에서 이 프로젝트를 이어서 할 때 **먼저 이 파일을 읽으세요.**
> 주제는 `plr-prompt-lab`(프롬프트/검색 성능 측정·개선 독립 도구)이고, 운영 `core/ir`과 연동돼 있어
> **어느 폴더를 비교해야 하는지**를 아래 "저장소·폴더 지도"에 정리했습니다.

---

## 0. TL;DR — 지금 상태

- **무엇**: **PLR(객체 속성 추출) 프롬프트를 측정하며 개선**하는 독립 실험 도구 (2026-07: 텍스트검색 파이프라인 제거 — PLR 전용). 데이터·모델·프롬프트·format/reason을 파라미터로 조합 → 자동 eval → ledger 추이 → HTML 리포트.
- **어디(lab)**: `/home/ziovision/plr-prompt-lab` — 별도 git repo, `master`, **72 tests green + 6 xfail** (GPU-free). PLR 전용(검색 실행 표면 제거).
- **운영 원본(ir)**: `/home/ziovision/ziomilitary/core/ir` — branch **`feat/plr-single-call-commit`**, HEAD **`92d2665`** (미배포). ⚠️ 운영 컨테이너 `ziosummary-ir`이 이 트리를 바인드마운트 — **재시작 시 v1.5 강제커밋 + PROMPT_VERSION bump가 배포되고 lazy per-video reindex가 발동**함(사용자 결정 사항). 롤백 = `git checkout fix/ir-classmap`.
- **진행 완료**: v1(사이클 코어) → v2 Phase 1(패키징+데이터준비) → v2 Phase 2(매트릭스+리포트) → GUIDE.html(구조/사용법 원페이지) → `--version` 실-프롬프트-로드 픽스 → **(2026-07-02) search `--version` 배선 + format/reason 매트릭스 축 + lab-side parity 테스트** (§8 갭 2·3·4 해소).
- **미완/다음**: 실측 baseline 미실행(GPU + gender 라벨 필요) — 남은 유일한 실행 갭.

---

## 1. 저장소·폴더 지도 (⭐ 어디를 비교하나)

| 역할 | 경로 |
|---|---|
| **lab** (주제, 개발 대상) | `/home/ziovision/plr-prompt-lab` (별도 repo) |
| **core/ir** (운영, lab의 lean-추출 원본) | `/home/ziovision/ziomilitary/core/ir` (submodule, branch `feat/plr-single-call-commit`) |
| **deploy repo** (plan/spec/progress) | `/home/ziovision/ziomilitary` → `.omc/plans/`, `.omc/specs/` |
| **GUIDE 배포 사본** | `/home/ziovision/plr-prompt-lab/_serve/index.html` (gitignore) |

### lab ↔ core/ir **parity 표면** (이 파일들이 서로 같아야 함 — 개선을 이식할 때 비교 대상)
`lab port` 명령이 자동으로 비교하는 "진짜 프롬프트 표면":
- `prompts/*.yaml` (4개: `plr_v0.4`, `plr_v1.3_cot`, `plr_v1.4_cot`, `plr_v1.5_cot`) — lab == core/ir (확인됨, 동일). **query_parser 블록 포함** — lab이 search를 안 돌려도 미러 무결성 때문에 유지.
- `plr_prompts.py` — 프롬프트 **상수** (lab == core/ir, 바이트 동일)
- `plr_core.py` — ⚠️ **의도적 divergence 1곳**: lab에는 `run_plr(..., build_messages=None)` 파라미터가 추가됨(`--version` 픽스, 하위호환). core/ir엔 없음. 그 외는 동일.
- ~~(시노님) parser/qp_v0.4.yaml / query_parser.py~~ — **2026-07-02 lab에서 삭제됨**(PLR 전용 슬림화). core/ir에는 그대로 존재.

**비교 방법**:
```bash
cd /home/ziovision/plr-prompt-lab
python3 lab.py port --core-ir /home/ziovision/ziomilitary/core/ir   # read-only diff
```

---

## 2. lab ↔ core/ir 관계 (어떻게 연동되나)

- lab은 core/ir **SEED `c2fc1cf`** 에서 개발 표면만 **lean 추출**한 것. (`SEED.md`에 원본 해시 기록.)
- 이후 core/ir이 **S0/S1 리팩터**로 `1690f25`가 됨 — 공유 pure 코어 `run_plr`/`run_search` + `Model` 인터페이스를 core/ir에서 추출(lab과 같은 심볼). 이 커밋은 **동작 보존**(ir 테스트 200 passed/4 xfailed) 확인 완료, **미배포**.
- **개선 흐름**: lab에서 프롬프트/파이프라인 개선 → 골든셋 eval 검증 → `lab port`로 diff 확인 → **사람이** core/ir에 적용(`lab port --apply`) → core/ir parity 테스트(`tests/test_prompt_source_parity.py`) 실행 → 배포.
- lab은 **DB/redis/서비스에 절대 접근 안 함** (import 격리 확인됨).

---

## 3. ⭐ 핵심 구조: 프롬프트 (여러 번 헷갈렸던 부분 — 정확히 이해할 것)

**(2026-07-02 단일 소스화 — 과거의 "두 갈래" 혼란 종결)** 런타임 프롬프트 소스는 이제 **하나**:
1. **기본** (`--version` 없음) = `plr_prompts.py`가 **`prompts/<PROMPT_VERSION_YAML_COT>.yaml`을 import 시 로드** (= 현재 plr_v1.5_cot). 하드코딩 상수는 삭제됨(바이트 동등성 스냅샷 검증 후). `IR_PLR_REASON`(on/off)이 CoT/plain 템플릿 선택, `IR_PLR_FORMAT=json`은 레거시 JSON 상수 경로(A/B용, 상수 유지).
2. **`lab run --version <V>`** = 같은 yaml 패밀리를 `FilePromptProvider`로 로드. **이제 yaml 편집이 곧 프롬프트 변경** — "yaml만 고치면 됨"이 (과거엔 틀렸지만) 지금은 맞음. 승격 = 새 버전 yaml 작성 + `PROMPT_VERSION_YAML_COT` bump가 전부.

**프롬프트 "종류"** (한 버전 yaml 안 = `plr_prompts.py` 상수):
- `system` (고정 역할지시)
- 템플릿: `person_user`(CoT) / `person_user_no_reason` / `vehicle_user` — plr용
- `query_parser`: `system` + `user` — **parity 미러로만 존재** (lab은 search 미실행; core/ir 쪽 실사용)

**파이프라인**: `plr`(속성: accuracy/bias/pred_unknown) 단일 — search는 2026-07 제거

**편집 대상**:
- 특정 **버전** 개선 → `prompts/<V>.yaml` 편집 (`lab run --version V`가 로드)
- **기본/현재(v1.5)** 및 **core/ir 이식** → `plr_prompts.py` 상수 편집 + `prompts/plr_v1.5_cot.yaml` parity 유지 (`tests/test_prompt_surface_parity.py`가 강제)

**역사**: 초기 HANDOFF가 "yaml만 고치면 됨"이라 했다가 틀림이 실증돼 정정(7634a0a) → 2026-07-02 프롬프트 선언화로 그 직관이 드디어 **참**이 됨(87ab817).

---

## 4. 무엇이 만들어졌나 (연대기 + 커밋)

### v1 — 사이클 코어 (plan: `.omc/plans/plr-prompt-lab-cycle.md`)
- core/ir S0/S1 공유코어 추출(`run_plr`/`run_search`/`Model`) + lab lean 씨딩
- CLI 사이클: build-golden → label → run → eval → port
- 관련 core/ir 커밋: `1690f25`

### v2 Phase 1 — 패키징 + 데이터준비 (전달 가능) — plan `.omc/plans/plr-prompt-lab-v2.md`
- `requirements.txt`/`.env.example`/`LICENSE`/`INSTALL.md`; 하드코딩 경로 제거(env: `CORE_IR_PATH`/`RESULT_PATH`/`DATASET_DIR`)
- `--dataset <path>` 파라미터 + `Dataset` 추상화 (`dataset.py`)
- `DATASET_SPEC.md` + `lab validate-dataset` (`validate.py`)
- `skills/prepare-dataset/SKILL.md` (Claude 스킬: 데이터셋 구성 가이드)
- `HANDOFF.md`(외부 엔지니어용) + `lab demo`(GPU-free 온보딩) + README v2
- 커밋: `44ee850`(패키징+dataset) · `0cf5504`(spec+validate) · `64406b1`(skill) · `c6878f5`(demo) · `456385d`(label 필드 통일) · `99a451e`(lab label CLI 픽스)

### v2 Phase 2 — 매트릭스 + 리포트
- `registry.py` (모델 `gemma`/`mock`, 파이프라인 `plr`/`search`) + `--model`/`--pipeline`
- ledger 조합키(`dataset`/`model`/`pipeline`/`prompt_hash`) — `provenance.py`
- `experiment.py` + `lab experiment run <yaml>` (교차곱 + fail-loud-but-continue) + `EXPERIMENT_SPEC.md`
- `report.py` + `lab report` → **자체완결 HTML**(inline SVG: 추이·히트맵·프롬프트변화→Δ)
- 커밋: `51d1a5d`(registry) · `095c501`(experiment) · `28187b8`(re_score obj_id 폴백) · `2ec70b3`(report) · `98983a2`(prompt 표면 단일출처)

### GUIDE + 프롬프트-축 픽스
- `GUIDE.html` (구조+사용법 원페이지, 배포용): `654ca3a` · `d7285e3` · `7634a0a`
- **`--version` 실-프롬프트-로드 픽스**: `51022f6`(feat) · `2553b13`(docs) — experiment의 프롬프트 축이 이제 진짜로 다른 프롬프트를 비교

### 프롬프트 기능별 분리 + 파서 분리 + JSON 경로 제거 (2026-07-02, 사용자 구조안)
- **prompts/<버전>/ 디렉터리 = 버전, 기능별 파일**: person/vehicle/query_parser/vqa/retry.yaml — **py에 프롬프트 텍스트 0줄** (VQA system·retry 템플릿도 yaml로 이동). `plr_prompts.py` 910→280줄, 조합(로드+enum주입+메시지 조립) 전용.
- **`plr_parse.py` 분리**(~500줄): 응답 파서+정규화 = 아웃풋 parity 표면. plr_prompts가 re-export(호환).
- **레거시 JSON 경로 삭제**: v0.4 프롬프트 상수/빌더, `IR_PLR_FORMAT` 스위치, indexing json 버전 분기, experiment `formats:` 축(명시적 거부 에러). `parse_plr_json`은 유지(검색 쿼리파서 응답용).
- provider 듀얼모드(디렉터리 버전 + 레거시 단일파일 아카이브 v0.4/1.3/1.4), exp_config/re_score가 디렉터리 버전 해석. port/hash 표면: prompts rglob + plr_parse.py 추가.
- **바이트 동등성 스냅샷 검증**(person CoT/plain·vehicle·retry×2·qp·VQA + provider 디렉터리모드). core/ir `17199df`(206), lab 89 passed, port 13 identical+plr_core divergence만.

### 구조 슬림화 (2026-07-02) — 죽은 기계장치 제거
- **registry.py 372→170줄**: get_provider/검증/폴백 등 슬롯 해석 기계장치 제거(lab은 provider를 registry로 해석하지 않음 — FilePromptProvider 직접 생성). 동기 모듈들의 import-시 self-register용 `register()` shim + MODELS/PIPELINES만 유지.
- **providers/__init__.py 352→223줄**: Parser/ScoringStrategy ABC 제거(PLR 전용 — 소비자 0). PromptProvider(+gemma_backend용 ModelProvider) 유지.
- **seed.sh 버그 수정**: DIRS에서 삭제된 `parser`, lab-소유 `eval`(재씨딩 시 run_eval 개조 소실 위험!), `providers`(lab이 더 슬림) 제거 → DIRS=(prompts schema), provider는 file_prompt_provider.py만 FILES로.
- **파일 필요성 판정**: config.py(24줄 — 동기 모듈의 lazy import 대상, 유지) · SEED.md(provenance/stale 경고, 유지) · seed.sh(core/ir→lab 역방향 동기화 도구, 유지).

### 어휘 선언화 + 전처리 분리 (2026-07-02, 재설계 3단계 — core/ir 먼저·양쪽 동시)
- **`schema/vocab.yaml` = 도메인 어휘의 단일 원천** (enum 13 + group 6 + map 1). `plr_schema.py`는 로더+파생물 생성기(모듈 상수·그룹 함수·JSON 스키마) — 프롬프트 주입/파서 정규화/게이트/저장 계약 4소비자가 같은 로드 결과를 봄. **"파일 하나 = 어휘 버전"** 성립: 어휘 확장은 이제 vocab.yaml 편집(주입·파싱이 자동으로 함께 움직임).
- **`preprocess.py`**: 마커 전처리를 plr_core에서 분리(이미지 쪽 인풋 표면의 명명된 컴포넌트). plr_core가 옛 이름 re-export — indexing 등 무수정.
- 20개 상수 스냅샷 동등성 검증(순서 포함) 후 치환 — 동작 불변. core/ir `d65ee2e`(206 passed) → lab 동기화(90 passed).
- **port/hash 표면 확장**: schema/*.yaml + plr_schema.py + preprocess.py 편입 → `lab port`가 어휘·전처리 parity까지 diff (확인: 전부 identical, plr_core divergence만). seed.sh FILES/DIRS 갱신, SEED.md `d65ee2e`.
- exp_config의 subset 가드는 유지(런타임 축소용); **어휘 확장의 정식 경로는 vocab.yaml**로 승격됨.

### 실험 config — 인풋 조합의 독립 버저닝 (2026-07-02)
- **원리(사용자 확정)**: 모델 인풋 = 템플릿 텍스트 + 주입 enum + 이미지 전처리(마커) + 샘플링 파라미터 — 조합이 중요하므로 **조합 자체를 버저닝**. 단, 프롬프트 복사를 피하려고 **분리 설계**: `prompts/<V>.yaml`=템플릿만, **`configs/<name>.yaml`=실험 파라미터 config**(`prompt: prompts/<V>.yaml` 경로 참조 + knob; enums는 inline 또는 yaml 경로). 같은 프롬프트 × N knob 조합 = config 파일 N개, 템플릿 복사 0. (명명: 사용자 지정 — 통용어 experiment config; 런타임 설정 config.py와는 별개)
- `lab run --version <이름>`이 config명/프롬프트명 모두 해석(config 우선, dangling 참조는 fail-loud). ledger version=config명, `prompt_hash`가 configs/*.yaml 포함(port 표면에는 미포함 — lab 전용). 로더: `runners/exp_config.py`.
- knob: `enums:`(provider `enum_overrides` 생성자 파라미터 — **lab·core/ir provider 동일 패치, 바이트 parity 유지**), `preprocess.marker:`(run_plr `_pre_marked=True`로 생략), `sampling.*`(LabGemmaModel 속성 — 하드코딩을 `__init__` 파라미터화).
- 승격 매핑: 템플릿→plr_prompts 상수 · enums→plr_schema · marker→indexing 호출부 · sampling→gemma 호출부 (author-prompt 스킬 §Promotion에 기록 — "lab에서 이겼는데 안 가져가면 운영에서 조용히 원복"됨).
- seed.sh에서 gemma_model.py 제외(LabGemmaModel/MockModel로 lab-분기). `runners/variant.py` 신설. 86 tests green.

### 필수 기능 완성 (2026-07-02, 사용자 요구 6종)
- **generic 데이터셋**: manifest.yaml이 `labels`/`pred_path`/`margin_path`/`bias_pair`/`object_type_hint` 선언 → validate/re_score/run_eval이 스펙 기반 동작 (`evalkit.dataset.attribute_spec`). gender/vehicle_type/military는 내장 프리셋(예시). 템플릿: `examples/dataset_template/` (validate PASS).
- **지표 완성**: precision·F1(클래스별+macro) ledger 추가; report.html에 **전체 실험 비교표** + **confusion 매트릭스**(recall/precision/F1 컬럼) 렌더링.
- **`lab gallery --dataset D`**: 크롭-라벨 시각화 자체완결 HTML — base64 썸네일, pred vs label, CORRECT/WRONG 배지, margin/quality, **오답 우선·저margin 우선** 정렬, 클래스 필터.
- **`lab report --compare LEDGER_B`**: 실험군 요약표 나란히 비교.
- **`skills/improve-prompt/SKILL.md`**: 분석자→제안자→비판자→수정자→(반복)→판단자 6역할 루프(최대 3라운드, 판단자 합격 기준 명시). 근거 규칙: 모든 제안은 실제로 본 크롭 obj_id + 이미지 관찰 + 수치 인용 필수. 산출물 = 분석 리포트 + prompts/<신버전>.yaml 초안 + 측정가능한 예상효과 + 검증 experiment yaml.
- **`skills/co-change/SKILL.md`**: 동행 수정 지침 — 표면(어휘/프롬프트/파서/스키마/전처리) 중 하나를 바꿀 때 같이 바꿔야 하는 것들의 매트릭스 + 공통 절차(byte-sync·양쪽 테스트·커밋) + 자동 가드 목록. **모든 표면 수정 전에 먼저 열 것** (sleeveless 유실·ca1a922가 존재 이유).
- 테스트 83+α passed.

### 신뢰·품질 스코어 평가 활용 (2026-07-02)
- **re_score**: 크롭마다 `margin`(모델 decision_margin — gender만 프롬프트가 emit, 그 외 None)과 `quality`(quality_gate 점수 — **측정 전용, 게이팅 아님**)를 predictions.jsonl에 기록.
- **run_eval**: `margin_stats`/`quality_stats` — 임계값(기본 0.7/0.4, `--margin-threshold`/`--quality-threshold`) 기준 high/low 구간별 accuracy + 정답·오답 평균. **v1.5 강제커밋의 캘리브레이션 검증 지표**: 오답이 저margin/저품질에 몰리면 신호 유효(런타임 필터 활용 가능), 아니면 margin은 노이즈.
- 하위호환: margin/quality 없는 구 predictions도 평가됨(stats=None). 75 tests green.

### 폴더 재구성 (2026-07-02, 기능별 분류)
- 루트 과밀 해소(사용자 요청). **parity/공유 표면은 core/ir과 같은 경로여야 해서 루트 고정**: `plr_core.py`·`plr_prompts.py`·`prompts/`·`providers/`·`plr_schema.py`·`quality_gate.py`·`config.py`·`registry.py`·`gemma_model.py`·`gemma_backend.py`.
- 이동: **`runners/`**(re_score·experiment·demo) · **`evalkit/`**(dataset·validate·provenance·report) · **`docs/`**(GUIDE.html·STRUCTURE.html·HANDOFF·INSTALL·DATASET_SPEC·EXPERIMENT_SPEC). 루트 잔류 문서: README(.ko)·SESSION_HANDOFF·SEED.md(provenance가 루트에서 읽음)·LICENSE·requirements.txt·seed.sh.
- import 경로: `from runners import re_score` / `from evalkit.dataset import …` 형태로 전환(테스트 포함). 각 이동 모듈의 `_LAB_ROOT`/`here` 앵커는 `.parent.parent`로 보정. seed.sh 파일 목록에서 삭제된 search 모듈 제거.

### PLR 전용 슬림화 (2026-07-02, PLR 집중 재설계 2단계)
- **search 실행 표면 제거**(사용자 결정): `search_core.py`·`query_parser.py`·`query_normalizer.py`·`run_search_eval.py`·`scoring.py`·`parser/`·`providers/bootstrap.py` 삭제, CLI(run/eval)의 search 분기·`--mode`/`--pipeline search`/`--k` 제거, registry PIPELINES=plr만, experiment search 셀 제거. seed 헬퍼는 `provenance.read_seed_hash`/`warn_stale_seed`로 이동.
- ⚠️ **parity 미러는 유지**: `plr_prompts.py`의 query_parser 상수와 `prompts/*.yaml`의 query_parser 블록은 core/ir 바이트-동일 미러라서 남김(§1 참조). provider의 build_query_parser_messages도 미러로 유지.
- **라벨 정책 확정**: `label=unknown`(사람도 판별 불가)은 accuracy/recall/bias에서 **제외**(`n_label_unknown`으로 별도 보고) — 강제 커밋 모델을 채점 불가 크롭으로 벌점 주지 않기 위함. **`pred_unknown`**(모델 unknown율 = 강제 커밋 준수도) ledger 메트릭 신설.
- DATASET_SPEC(queries 삭제·라벨 정책), EXPERIMENT_SPEC, README(en/ko), HANDOFF 갱신. SEED.md를 core/ir `92d2665`로 재동기화(stale 경고 해소). 테스트 **72 passed, 6 xfailed**.
- 검색 평가의 거처: core/ir 재평가(임베딩+VQA 풀스택) + `/cctv-eval` 오라클.

### plr_v1.5_cot — 강제커밋 + single-view (2026-07-02, PLR 집중 재설계 1단계)
- **설계 확정(사용자)**: ① quality_gate 제거(모든 크롭이 모델로) ② 크롭당 모델 호출 정확히 1회(SR 이중뷰 제거; transient/schema retry는 오류 처리라 유지) ③ **unknown 제거** — 프롬프트가 항상 커밋 강제, 저신뢰는 margins로.
- **core/ir 먼저 수정 → lab 동기화** (사용자 지시로 개선 흐름 역방향; 표면은 바이트 동일 유지).
- core/ir 커밋 `92d2665`: 프롬프트 unknown 제거 + `_commit_enum` 필터 + `PROMPT_VERSION_YAML_COT=plr_v1.5_cot`(reindex 트리거) + indexing single-view + provider `commit_enums:` yaml 플래그(구버전 yaml은 역사적 프롬프트 보존) + no-unknown 게이트 테스트. **206 passed, 6 xfailed**.
- lab 동기화: plr_prompts.py/provider/v1.5 yaml 복사(바이트 동일), re_score quality_gate 제거, parity 테스트 v1.5 이동. **86 passed, 6 xfailed**, port diff = plr_core divergence만.
- **유지된 것**: plr_schema enum의 unknown 멤버(구 인덱스 행 판독 + 방어적 정규화), 게이트의 unknown wildcard-pass(재인덱싱 완료 전까지), rider_vehicle N/A 센티널(shape 계약).
- **다음**: gender 골든셋 라벨 + GPU로 v1.4 vs v1.5 A/B (`lab experiment run`, accuracy/bias 나란히) → 배포 결정.

### 갭 마감 (2026-07-02, §8의 2·3·4)
- **search `--version` 배선**: `run_search_over_golden(prompt_version=…)` → `parse_query(build_messages=…)` → `parse_with_gemma` 주입구. gemma 백엔드가 있을 때 `prompts/<V>.yaml`의 `query_parser` 블록을 실제 전송(dictionary 경로는 무프롬프트라 영향 없음). lab.py / experiment.py 양쪽 배선.
- **format/reason 매트릭스 축**: experiment.yaml 옵션 키 `formats:`(yaml|json) / `reasons:`("on"|"off" — 따옴표 필수). plr 셀에만 교차, 셀 단위 env 적용+복원, ledger version 태그에 `+json`/`+reason-off` 접미사로 구분. yaml-고정 버전 × 불일치 format 은 셀 단위 fail-loud.
- **lab-side parity 테스트**: `tests/test_prompt_surface_parity.py` — core/ir의 `test_prompt_source_parity.py` 미러 + query_parser 블록 parity(3버전 모두) 추가.
- 신규 테스트: `tests/test_search_version_and_axes.py` (배선/축/가드/env-복원 검증). 총 **79 passed, 4 xfailed**.

---

## 5. CLI (9 서브명령)

```
lab.py build-golden --video V --attribute A [--dataset D]   # 골든셋 크롭 생성
lab.py label --dataset D --female-in-male M3,M7 --unknown M9 # 사람 라벨(오분류/판별불가)
lab.py validate-dataset --dataset D                         # 형식 검증(fail-loud)
lab.py run --model gemma|mock --version V --attribute A --dataset D   # PLR 재채점
lab.py eval --attribute A --dataset D                       # 채점 + ledger Δ (+unknown율)
lab.py experiment run experiment.yaml                       # 교차곱 매트릭스
lab.py report --out report.html                             # ledger → HTML
lab.py port [--apply] [--core-ir PATH]                      # lab↔core/ir diff
lab.py demo                                                 # GPU-free 온보딩
```
(2026-07: PLR 전용 — search 관련 옵션 `--pipeline search`/`--mode`/`--k`는 제거됨)

---

## 6. 실행 / 검증

```bash
cd /home/ziovision/plr-prompt-lab
python3 -m pytest tests/ -q          # → 67 passed (GPU/DB/네트워크 없음)
python3 lab.py demo                  # GPU 없이 전체 사이클 시연
```
- **실측**(실제 Gemma)은 GPU + Gemma-4-E4B GGUF 필요 → `INSTALL.md`. 운영 ir과 GPU 경합 주의(전용/오프피크).

---

## 7. 주요 파일 역할

| 파일 | 역할 |
|---|---|
| `lab.py` | 단일 CLI 진입점 |
| `prompts/*.yaml` | 버전별 프롬프트(`--version`이 로드) — plr + query_parser 블록 |
| `plr_prompts.py` | **프롬프트 로더**(prompts/<현재버전>.yaml → 라이브 템플릿) + `parse_plr_response` (레거시 JSON 상수만 잔존) |
| `re_score.py` | plr 실행 러너 (재채점 → predictions/attributes.jsonl) |
| `eval/run_eval.py` | 채점(accuracy/bias/pred_unknown) + ledger |
| `plr_core.py` | `run_plr`(속성 추론 코어) — `build_messages` 주입구 |
| `schema/vocab.yaml` | 선언적 도메인 어휘 — 단일 원천 (plr_schema가 로드·파생) |
| `preprocess.py` | 이미지 전처리(마커) — 인풋 표면의 명명된 컴포넌트 |
| `configs/*.yaml` | 실험 파라미터 config — prompt 경로 참조 + enum축소/마커/샘플링 knob |
| `registry.py` | 모델/파이프라인 레지스트리 + `MockModel` |
| `provenance.py` | `prompt_hash` + `surface_relpaths` + seed 헬퍼(read_seed_hash/warn_stale_seed) |
| `experiment.py` | 매트릭스 러너 |
| `report.py` | HTML 리포트 |
| `dataset.py` · `validate.py` | 데이터셋 추상화 · 검증 |
| `providers/file_prompt_provider.py` | 버전 yaml → 메시지(`--version` 픽스가 사용) |
| `eval/ledger.jsonl` | 실험 추이 저장소(append-only) |
| `SEED.md` | 원본 core/ir 해시 |

문서: 루트 `README.md`(한글)/`SESSION_HANDOFF.md`/`SEED.md` · **`docs/`** (GUIDE.html · STRUCTURE.html · INSTALL.md · HANDOFF.md · DATASET_SPEC.md · EXPERIMENT_SPEC.md) · `skills/prepare-dataset/SKILL.md`

---

## 7.5 재인덱싱-후 정리 백로그 (v1.5 배포 → reindex 완료가 선행 조건)

구세대 행 호환 때문에 남긴 잔재들 — **모든 비디오가 v1.5로 재인덱싱된 뒤** 일괄 제거:
1. PERSON/VEHICLE_SCHEMA의 `evidence`/`caution` 선택 필드 (v0.4 근거-목록 시절)
2. `visibility` 빈 블록 (quality_gate 텔레메트리 자리 — shape 호환용으로만 잔존)
3. 검색 게이트(scoring)의 unknown wildcard-pass (구행 보호 장치)
4. `maps.lower_type_to_shape` 읽기시점 파생 + `upper_outer_of` (전용 추출 필드가 재인덱싱으로 채워지면)
5. 파서 `_UNKNOWN_FALLBACKS` 축소 검토
6. `_normalize_plr_json`의 vehicle 행 의복 플레이스홀더 제거(person 한정으로
   — template_caption 등 upper_clothing 무조건 접근 소비자 확인 후;
   tests/test_schema_declares_parser_keys.py의 whitelist를 비워 회귀 가드)
7. `ziosummary_engine` DB의 빈 `ir_plr_index`/`ir_indexing_failures` 테이블 drop
   (실사용은 ziosummary_management 쪽 — IR_PG_DB 확인 완료)

**제거 금지(잔재 아님·현역)**: 파서 관용 처리(모델은 계속 일탈 — 상시 방어) ·
분포 플레이스홀더 male/female 1.0/0.0 (core/ir scoring이 인터페이스로 소비 — 제거는 scoring 개편과 한 몸) ·
rider_vehicle unknown 센티널(N/A shape 계약).

**지금 가능(누락 보완, additive)**: PERSON_SCHEMA에 현행 필드 `sleeve`·`reason` 명시.

## 8. 알려진 갭 / 다음 할 일

1. **실측 baseline 미실행** — gender 골든셋(63)에 **사람 라벨**이 아직 없고 GPU 실행 필요. (`lab build-golden`으로 크롭 재생성 → `lab label` → `lab run --model gemma` → `lab eval`.)
2. ~~search `--version` 미배선~~ — ✅ **완료 (2026-07-02)**: `run_search_over_golden(prompt_version=…)` → `parse_query(build_messages=…)` 배선. 단, 효과는 gemma 백엔드 사용 시에만(현재 lab run search는 `model=None` dictionary 경로 — 검색 프롬프트 A/B는 GPU 백엔드 연결이 선행 조건).
3. ~~format/reason 매트릭스 축 승격~~ — ✅ **완료 (2026-07-02)**: experiment.yaml `formats:`/`reasons:` 옵션 축 (EXPERIMENT_SPEC.md 참고).
4. ~~lab parity 테스트 없음~~ — ✅ **완료 (2026-07-02)**: `tests/test_prompt_surface_parity.py`.
5. **core/ir `1690f25` 미배포** — ir 재시작 시 반영됨. 배포 여부는 사용자 결정(architect가 "재시작 안전" 확인).
6. **골든 크롭 gitignore** — 실측 데이터는 repo에 없음(`~/gender_eval` 등). 프라이버시상 배포 금지 → 받는 사람은 자기 데이터로.
7. ~~search의 gemma 백엔드 스위치~~ — **무효 (2026-07-02)**: search 파이프라인 자체가 제거됨. (구)  — `lab run --pipeline search`가 `--model`을 무시하고 항상 dictionary 경로. 검색 프롬프트를 실제로 A/B 하려면 query-parser용 backend(.generate(pil,msgs,…)→.raw 프로토콜, lab Model 프로토콜과 다름) 어댑터가 필요.

---

## 9. GUIDE 배포 (외부 서버)

- **URL(LAN)**: `http://<서버IP>:8899/` — 공인 IP/도메인이면 그 호스트 + `:8899`.
- 서빙 대상: `_serve/index.html` (= `GUIDE.html` 사본). 소스코드는 노출 안 됨.
- **세션 종료 시 서버 죽음** → 재기동:
  ```bash
  python3 -m http.server 8899 --bind 0.0.0.0 --directory /home/ziovision/plr-prompt-lab/_serve
  ```
- GUIDE 수정 후 재배포: `cp docs/GUIDE.html _serve/index.html` (서버는 파일을 매 요청 읽음).

---

## 10. deploy repo 산출물 (참고)

- `.omc/plans/plr-prompt-lab-cycle.md` — v1 consensus plan (ADR 포함)
- `.omc/plans/plr-prompt-lab-v2.md` — v2 plan (Phase 1/2)
- `.omc/specs/deep-interview-prompt-lab-cycle.md` — 최초 deep-interview 스펙
- `.omc/state/sessions/<id>/progress.txt` — 이번 세션 진행 로그(상세)
