# plr-prompt-lab — 세션 핸드오프 (2026-07-02)

> 새 세션에서 이 프로젝트를 이어서 할 때 **먼저 이 파일을 읽으세요.**
> 주제는 `plr-prompt-lab`(프롬프트/검색 성능 측정·개선 독립 도구)이고, 운영 `core/ir`과 연동돼 있어
> **어느 폴더를 비교해야 하는지**를 아래 "저장소·폴더 지도"에 정리했습니다.

---

## 0. TL;DR — 지금 상태

- **무엇**: PLR(객체 속성 추출) + 텍스트검색의 **프롬프트·파이프라인을 측정하며 개선**하는 독립 실험 도구. 데이터·모델·프롬프트·파이프라인을 파라미터로 조합 → 자동 eval → ledger 추이 → HTML 리포트.
- **어디(lab)**: `/home/ziovision/plr-prompt-lab` — 별도 git repo, `master`, HEAD **`2553b13`**, 71 tracked files, **67 tests green** (GPU-free).
- **운영 원본(ir)**: `/home/ziovision/ziomilitary/core/ir` — branch **`feat/plr-single-call-commit`**, HEAD **`92d2665`** (미배포). ⚠️ 운영 컨테이너 `ziosummary-ir`이 이 트리를 바인드마운트 — **재시작 시 v1.5 강제커밋 + PROMPT_VERSION bump가 배포되고 lazy per-video reindex가 발동**함(사용자 결정 사항). 롤백 = `git checkout fix/ir-classmap`.
- **진행 완료**: v1(사이클 코어) → v2 Phase 1(패키징+데이터준비) → v2 Phase 2(매트릭스+리포트) → GUIDE.html(구조/사용법 원페이지) → `--version` 실-프롬프트-로드 픽스 → **(2026-07-02) search `--version` 배선 + format/reason 매트릭스 축 + lab-side parity 테스트** (§8 갭 2·3·4 해소).
- **미완/다음**: 실측 baseline 미실행(GPU + gender 라벨 필요) — 남은 유일한 실행 갭.

---

## 1. 저장소·폴더 지도 (⭐ 어디를 비교하나)

| 역할 | 경로 |
|---|---|
| **lab** (주제, 개발 대상) | `/home/ziovision/plr-prompt-lab` (별도 repo) |
| **core/ir** (운영, lab의 lean-추출 원본) | `/home/ziovision/ziomilitary/core/ir` (submodule, branch `fix/ir-classmap`) |
| **deploy repo** (plan/spec/progress) | `/home/ziovision/ziomilitary` → `.omc/plans/`, `.omc/specs/` |
| **GUIDE 배포 사본** | `/home/ziovision/plr-prompt-lab/_serve/index.html` (gitignore) |

### lab ↔ core/ir **parity 표면** (이 파일들이 서로 같아야 함 — 개선을 이식할 때 비교 대상)
`lab port` 명령이 자동으로 비교하는 "진짜 프롬프트 표면":
- `prompts/*.yaml` (3개: `plr_v0.4`, `plr_v1.3_cot`, `plr_v1.4_cot`) — lab == core/ir (확인됨, 동일)
- `plr_prompts.py` — 프롬프트 **상수** (lab == core/ir, 바이트 동일)
- `plr_core.py` — ⚠️ **의도적 divergence 1곳**: lab에는 `run_plr(..., build_messages=None)` 파라미터가 추가됨(`--version` 픽스, 하위호환). core/ir엔 없음. 그 외는 동일.
- (시노님) `parser/qp_v0.4.yaml` — search 쪽, 동일
- (표면 밖) `query_parser.py` — ⚠️ lab-only divergence: `parse_query`/`parse_with_gemma`에 `build_messages=None` 주입구 추가(search `--version` 배선, 하위호환 — plr_core와 같은 패턴). port 표면(provenance)에는 포함되지 않으므로 `lab port`에는 안 나타남; core/ir 이식 시 참고.

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

**런타임 프롬프트 소스는 두 갈래:**
1. **기본** (`--version` 없음) = `plr_prompts.py`의 **하드코딩 상수** (= 현재 버전 v1.4). `IR_PLR_FORMAT`(yaml/json) × `IR_PLR_REASON`(on/off) **env**가 어느 상수를 쓸지 선택. → core/ir과 바이트 동일(이식 대상).
2. **`lab run --version <V>`** = `prompts/<V>.yaml`을 `FilePromptProvider`로 **로드** (커밋 `51022f6`에서 배선). 없는 버전/mock은 상수로 폴백.

**증거**: `prompts/*.yaml` 편집만으론 기본 프롬프트 안 바뀜(상수가 authoritative). 하지만 `--version plr_v1.3_cot` vs `plr_v1.4_cot`는 실제로 다른 프롬프트를 보냄(해시 `34008…` ≠ `85bf4…`).

**프롬프트 "종류"** (한 버전 yaml 안 = `plr_prompts.py` 상수):
- `system` (고정 역할지시)
- 템플릿: `person_user`(CoT) / `person_user_no_reason` / `vehicle_user` — plr용
- `query_parser`: `system` + `user` — search용 (+ `parser/qp_v0.4.yaml` 시노님 사전)

**파이프라인 2개**: `plr`(속성: accuracy/bias) / `search`(검색: recall@k)

**편집 대상**:
- 특정 **버전** 개선 → `prompts/<V>.yaml` 편집 (`lab run --version V`가 로드)
- **기본/현재(v1.4)** 및 **core/ir 이식** → `plr_prompts.py` 상수 편집 + `prompts/plr_v1.4_cot.yaml` parity 유지

**주의**: HANDOFF.md 초기 버전이 "yaml만 고치면 됨"이라 **틀렸었고**, 실증 후 정정함(커밋 `7634a0a`, `2553b13`).

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
lab.py label --dataset D --female-in-male M3,M7 ...          # 사람 라벨(오분류만)
lab.py validate-dataset --dataset D                         # 형식 검증(fail-loud)
lab.py run --model gemma|mock --pipeline plr|search --version V --attribute A --dataset D
lab.py eval --attribute A --mode attr|search --dataset D    # 채점 + ledger Δ
lab.py experiment run experiment.yaml                       # 교차곱 매트릭스
lab.py report --out report.html                             # ledger → HTML
lab.py port [--apply] [--core-ir PATH]                      # lab↔core/ir diff
lab.py demo                                                 # GPU-free 온보딩
```

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
| `plr_prompts.py` | 프롬프트 **상수**(기본/런타임 실사용) + `parse_plr_response` |
| `plr_core.py` | `run_plr`(속성 추론 코어) — `build_messages` 주입구 |
| `search_core.py` | `run_search`(하드필터+랭킹) |
| `registry.py` | 모델/파이프라인 레지스트리 + `MockModel` |
| `provenance.py` | `prompt_hash` + `surface_relpaths`(port/hash 단일출처) |
| `re_score.py` | plr 실행(A) + `run_search_over_golden`(B) |
| `eval/run_eval.py` · `run_search_eval.py` | 채점(accuracy/bias · recall@k) + ledger |
| `experiment.py` | 매트릭스 러너 |
| `report.py` | HTML 리포트 |
| `dataset.py` · `validate.py` | 데이터셋 추상화 · 검증 |
| `providers/file_prompt_provider.py` | 버전 yaml → 메시지(`--version` 픽스가 사용) |
| `eval/ledger.jsonl` | 실험 추이 저장소(append-only) |
| `SEED.md` | 원본 core/ir 해시 |

문서: `README.md` · `GUIDE.html` · `INSTALL.md` · `HANDOFF.md` · `DATASET_SPEC.md` · `EXPERIMENT_SPEC.md` · `skills/prepare-dataset/SKILL.md`

---

## 8. 알려진 갭 / 다음 할 일

1. **실측 baseline 미실행** — gender 골든셋(63)에 **사람 라벨**이 아직 없고 GPU 실행 필요. (`lab build-golden`으로 크롭 재생성 → `lab label` → `lab run --model gemma` → `lab eval`.)
2. ~~search `--version` 미배선~~ — ✅ **완료 (2026-07-02)**: `run_search_over_golden(prompt_version=…)` → `parse_query(build_messages=…)` 배선. 단, 효과는 gemma 백엔드 사용 시에만(현재 lab run search는 `model=None` dictionary 경로 — 검색 프롬프트 A/B는 GPU 백엔드 연결이 선행 조건).
3. ~~format/reason 매트릭스 축 승격~~ — ✅ **완료 (2026-07-02)**: experiment.yaml `formats:`/`reasons:` 옵션 축 (EXPERIMENT_SPEC.md 참고).
4. ~~lab parity 테스트 없음~~ — ✅ **완료 (2026-07-02)**: `tests/test_prompt_surface_parity.py`.
5. **core/ir `1690f25` 미배포** — ir 재시작 시 반영됨. 배포 여부는 사용자 결정(architect가 "재시작 안전" 확인).
6. **골든 크롭 gitignore** — 실측 데이터는 repo에 없음(`~/gender_eval` 등). 프라이버시상 배포 금지 → 받는 사람은 자기 데이터로.
7. **(신규, 옵션) search의 gemma 백엔드 스위치** — `lab run --pipeline search`가 `--model`을 무시하고 항상 dictionary 경로. 검색 프롬프트를 실제로 A/B 하려면 query-parser용 backend(.generate(pil,msgs,…)→.raw 프로토콜, lab Model 프로토콜과 다름) 어댑터가 필요.

---

## 9. GUIDE 배포 (외부 서버)

- **URL(LAN)**: `http://<서버IP>:8899/` — 공인 IP/도메인이면 그 호스트 + `:8899`.
- 서빙 대상: `_serve/index.html` (= `GUIDE.html` 사본). 소스코드는 노출 안 됨.
- **세션 종료 시 서버 죽음** → 재기동:
  ```bash
  python3 -m http.server 8899 --bind 0.0.0.0 --directory /home/ziovision/plr-prompt-lab/_serve
  ```
- GUIDE 수정 후 재배포: `cp GUIDE.html _serve/index.html` (서버는 파일을 매 요청 읽음).

---

## 10. deploy repo 산출물 (참고)

- `.omc/plans/plr-prompt-lab-cycle.md` — v1 consensus plan (ADR 포함)
- `.omc/plans/plr-prompt-lab-v2.md` — v2 plan (Phase 1/2)
- `.omc/specs/deep-interview-prompt-lab-cycle.md` — 최초 deep-interview 스펙
- `.omc/state/sessions/<id>/progress.txt` — 이번 세션 진행 로그(상세)
