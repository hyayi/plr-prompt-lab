# plr-prompt-lab (v2) — 한글판

> 📖 **처음이라면 [`docs/GUIDE.html`](docs/GUIDE.html)을 브라우저로 여세요** — 구조와 사용법을
> 한 장(자체완결 HTML·오프라인)으로 그림과 함께 설명합니다.
> 아키텍처 시각화는 [`docs/STRUCTURE.html`](docs/STRUCTURE.html), 영문판은 [`README.md`](README.md).

PLR(Person-Level Recognition) **속성 채점** 전용 프롬프트 평가 사이클
도구입니다 — lab의 목적은 PLR 프롬프트 최적화입니다. (텍스트 검색 파이프라인은
2026-07에 제거: 검색 recall은 lab이 담을 수 없는 표면 — 임베딩·VQA — 이 지배하므로
검색 평가는 core/ir와 cctv-eval이 담당합니다.) `core/ir`에서 lean 추출했습니다
([SEED.md](SEED.md) 참고).

mock/합성 경로에서는 **데이터베이스·Redis·GPU 없이** 프롬프트, 채점 로직,
평가 하네스를 반복 개선할 수 있습니다. 실제 Gemma 추론(크롭 재채점)은 전용 GPU와
사람이 라벨링한 크롭이 필요합니다 — 아래 [실측 실행 전제조건](#실측-실행-전제조건) 참고.

v2에서 추가된 것: `--dataset` 파라미터, `validate-dataset` 서브명령, GPU-free
온보딩 `lab demo`, [DATASET_SPEC.md](docs/DATASET_SPEC.md) 형식 스펙,
[HANDOFF.md](docs/HANDOFF.md) 외부 프롬프트 엔지니어 가이드. v2 Phase 2에서
모델/파이프라인 **레지스트리**, **experiment 매트릭스 러너**, **HTML 리포트**가
추가로 구현됐습니다([EXPERIMENT_SPEC.md](docs/EXPERIMENT_SPEC.md) 참고).

---

## lab이란 무엇인가

lab은 `core/ir`의 PLR 개발 표면을 **lean 스냅샷**으로 담고 있습니다:

- `plr_core.py`, `plr_prompts.py`, `plr_schema.py` — 순수 PLR 추론 코어
- `gemma_model.py` — `Model` 프로토콜 + `LabGemmaModel`(직접 호출, 스케줄러 없음)
- `gemma_backend.py` — GPU GGUF 로더 (가드됨; `lab run` 전에는 import 안 됨)
- `prompts/` — PLR 프롬프트 YAML (`plr_v0.4` … `plr_v1.5_cot`)
- `eval/` — 골든셋, 러너 스크립트, ledger
- `demo.py` — `lab demo`용 자체완결 MockModel + 합성 데이터셋
- `registry.py`, `experiment.py`, `report.py` — 모델/파이프라인 레지스트리,
  매트릭스 러너, HTML 리포트 (Phase 2)

**복사하지 않은 것** (서비스/DB/redis/임베딩 계층): `storage.py`,
`redis_handler.py`, `indexing.py`, `main.py`, `scheduler.py`, `text_embed.py`,
`backfill.py` 등.

### import 순수성 계약

```bash
python3 -c "import plr_core, gemma_model, quality_gate, plr_prompts, \
    plr_schema; print('lab imports OK')"
```

위 import 후 `sys.modules`에 `storage`, `psycopg2`, `redis`가 나타나면 안 됩니다.

---

## 파라미터 모델

실험에서 선택 가능한 축:

| 축 | 선택 방법 | 상태 |
|---|---|---|
| **데이터셋** | `--dataset /path/to/dir` (기본: `eval/golden/<attribute>/`) | 구현됨 |
| **프롬프트** | `prompts/*.yaml` 편집 후 `lab run`/`lab eval`에 `--version <이름>` | 구현됨 |
| **모델** | `--model gemma\|mock` (레지스트리, `registry.py`) | 구현됨 (P2-1) |
| **파이프라인** | `plr` 전용 (레지스트리) — 검색은 2026-07 제거 | 구현됨 |
| **format/reason** | experiment.yaml의 `formats:`/`reasons:` 축 (`IR_PLR_FORMAT`/`IR_PLR_REASON`) | 구현됨 |

여러 축의 **교차곱을 한 번에** 돌리려면 `lab experiment run <yaml>`을 쓰세요
([EXPERIMENT_SPEC.md](docs/EXPERIMENT_SPEC.md)). 결과 추이는 `lab report`가
자체완결 HTML로 만들어 줍니다.

---

## 사이클

```
build-golden  ──►  label  ──►  run  ──►  eval  ──►  port
    │                               │          │
    │  (실측 데이터, 운영자 단계)      │          └──► ledger.jsonl Δ
    │                               └──► 크롭에 Gemma 재실행 (GPU)
    └── <dataset>/crops/<obj_id>.jpg
```

### A — PLR 속성 평가 (gender, vehicle_type, military)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. build-golden  (운영자 단계 — 실제 비디오 + DB 필요)                │
│     lab build-golden --video <vd_id> --attribute gender             │
│     → <dataset>/crops/<obj_id>.jpg                                  │
│     → <dataset>/index_map.json                                      │
│     → <dataset>/predictions.jsonl  (부트스트랩)                      │
│                                                                     │
│  2. label  (사람 단계)                                               │
│     lab label --dataset <dataset> --female-in-male M3,M7           │
│     → <dataset>/labels.jsonl                                        │
│                                                                     │
│  3. run  (GPU 단계)                                                  │
│     lab run --attribute gender --version plr_v1.4_cot               │
│              --dataset <dataset>                                    │
│     → <dataset>/predictions.jsonl  (덮어씀)                          │
│     → <dataset>/attributes.jsonl                                    │
│                                                                     │
│  4. eval                                                            │
│     lab eval --attribute gender --version plr_v1.4_cot              │
│              --dataset <dataset>                                    │
│     → accuracy/confusion/bias + 이전 버전 대비 Δ 출력                 │
│     → eval/ledger.jsonl에 레코드 추가                                │
│                                                                     │
│  5. port  (읽기전용 diff / --apply로 core/ir에 반영)                  │
│     lab port [--apply] [--core-ir /path/to/core/ir]                 │
└─────────────────────────────────────────────────────────────────────┘
```

### B — 텍스트 검색 평가 — 제거됨 (2026-07)

lab은 더 이상 검색 파이프라인을 돌리지 않습니다. PLR 라벨이 운영 검색의
입력이므로 PLR accuracy/bias(+ `pred_unknown` 비율)가 lab의 신호이고,
end-to-end 검색 품질은 core/ir(임베딩+VQA 풀스택)과 cctv-eval 오라클이
측정합니다.

---

## 명령어

```bash
# GPU-free 온보딩 — 데이터 없이 전체 루프를 즉시 체험
python3 lab.py demo

# 골든셋 생성 (실측 데이터, 운영자 단계 — 전제조건 참고)
python3 lab.py build-golden --video <vd_...> --attribute gender [--dataset <dir>]

# 라벨링 (사람 단계)
python3 lab.py label [--dataset <dir>] --female-in-male M3,M7 --male-in-female F2

# Gemma로 재채점 (GPU 단계)
python3 lab.py run --attribute gender --version plr_v1.5_cot [--dataset <dir>] [--model gemma|mock]

# PLR 속성 평가
python3 lab.py eval --attribute gender --version plr_v1.4_cot [--dataset <dir>]

# 실험 매트릭스 실행 (교차곱 — EXPERIMENT_SPEC.md 참고)
python3 lab.py experiment run examples/experiment.example.yaml [--strict]

# ledger → 자체완결 HTML 리포트 (추이·히트맵·프롬프트변화→Δ)
python3 lab.py report --out report.html

# 데이터셋 디렉터리 검증
python3 lab.py validate-dataset --dataset <dir>

# lab 프롬프트 표면 ↔ core/ir diff (또는 --apply로 반영)
python3 lab.py port [--apply] [--core-ir /path/to/core/ir]

# 전체 테스트 (GPU·DB 불필요)
python3 -m pytest tests/ -q
```

---

## `--dataset` 파라미터

골든셋을 읽거나 쓰는 모든 명령이 `--dataset <dir>`을 받습니다. 생략하면
`eval/golden/<attribute>/`로 폴백합니다 — v2 이전과 같은 레이아웃이므로 기존
워크플로는 그대로 동작합니다.

데이터셋 디렉터리는 [DATASET_SPEC.md](docs/DATASET_SPEC.md)를 따라야 합니다.
새 데이터셋은 `lab run` 전에 검증하세요:

```bash
python3 lab.py validate-dataset --dataset /path/to/my_dataset/
```

`prepare-dataset` 스킬(`skills/` 참고)은 build-golden + label 단계를 하나의
가이드 워크플로로 자동화합니다.

---

## `lab demo` — GPU-free 온보딩

```bash
python3 lab.py demo
```

**GPU·데이터베이스·모델 다운로드 없이** 전체 mock 평가 사이클을 실행합니다:

1. `demo_dataset/`에 5-크롭 합성 데이터셋 생성 (작은 JPEG + 라벨).
2. 내장 `MockModel`로 `re_score()`를 두 번 호출 — v1은 female 예측(accuracy 1.0),
   v2는 male 예측(accuracy 0.0).
3. 버전별로 `run_eval()`을 돌리고 accuracy + Δ 출력.
4. 무슨 일이 일어났는지와 다음 단계 안내 출력.
5. 종료 시 `demo_dataset/` 정리 (`--keep`으로 유지 가능).

새로 설치한 환경이 제대로 배선됐는지 확인하거나, GPU + 라벨 준비 전에 팀원에게
루프를 시연할 때 사용하세요.

---

## `HANDOFF.md` — 외부 프롬프트 엔지니어 가이드

[HANDOFF.md](docs/HANDOFF.md)는 운영 서비스를 건드리지 않고 PLR/검색 프롬프트를
개선하는 프롬프트 엔지니어를 위한 가이드입니다:

- 무엇을 편집하나 (`prompts/*.yaml`, 그리고 언제 `plr_prompts.py`도 함께).
- 무엇을 건드리면 안 되나 (추론 코어, 스토리지 — 둘 다 lab에 없음).
- 전체 반복 루프: 데이터셋 준비 → `lab run` → `lab eval` → Δ 확인 → 반복 →
  `lab port` → diff + 우승 YAML을 ZioVision에 반납.
- 실측 실행 전제조건 (GPU + 모델) vs GPU-free `lab demo`.
- 반납 절차: `lab port`가 읽기전용 diff를 생성; 외부 엔지니어는 diff와 우승
  프롬프트 YAML을 ZioVision에 전달하고, ZioVision이 `core/ir` 내부 재평가를
  게이트로 적용합니다.

---

## 콜드스타트 전제조건

깨끗한 체크아웃에서는 다음 세 조건이 모두 충족되기 전까지 `eval`을 실행할 수 없습니다:

1. **크롭 씨딩** — `predictions.jsonl`의 모든 `obj_id`에 대해
   `<dataset>/crops/<obj_id>.jpg`가 있어야 합니다. `lab build-golden`이 생성.
   `eval/golden/*/crops/` 디렉터리는 gitignore 대상 — 절대 커밋하지 않습니다.

2. **`labels.jsonl` 생성** — `<dataset>/labels.jsonl`이 있어야 합니다.
   사람이 컨택트시트를 검토한 뒤 `lab label`이 생성. 사람이 라벨링하기 전까지
   `run_eval.py`는 `FileNotFoundError`를 냅니다.

3. **`ledger.jsonl`은 첫 eval에서 자동 생성** — `eval/ledger.jsonl`은 첫
   `lab eval` 실행 시 자동으로 만들어집니다. 미리 있을 필요 없습니다.

---

## 실측 실행 전제조건

`lab run`은 `LabGemmaModel`을 로드하고, 이는 `gemma_backend.load_backend()`를
호출해 4B GGUF 모델을 VRAM에 내려받아 적재합니다.

**실측 측정 전에 두 가지 명시적 블로커가 있습니다:**

1. **전용 GPU 또는 오프피크 시간대 필요.**
   `engine/`에서 돌고 있는 운영 `ir` 서비스가 이미 GPU와 VRAM 대부분을 점유
   중입니다(인덱싱/검색용 `GemmaBackend` 상주). `lab run`을 동시에 돌리면
   OOM이 나거나 GPU를 경합합니다. 운영 `ir` 컨테이너를 중지하거나 오프피크
   유지보수 시간대에 실행하세요:

   ```bash
   # lab 실행 전 ir 중지
   cd /home/ziovision/ziomilitary/engine && docker-compose stop imageretrieval
   python3 lab.py run --attribute gender --version plr_v1.4_cot
   # 끝나면 ir 재시작
   docker-compose start imageretrieval
   ```

2. **사람이 라벨링한 골든셋 필요.**
   `lab eval`이 의미 있는 accuracy를 내려면 `<dataset>/labels.jsonl`에
   사람이 검증한 정답이 있어야 합니다. 컨택트시트 검토 후 교정 타일 ID로
   `lab label`을 실행하세요.

전체 셋업(Python 환경, CUDA 빌드, 모델 다운로드)은 [INSTALL.md](docs/INSTALL.md) 참고.

---

## 메트릭과 ledger

### A — PLR 속성 메트릭 (`eval/ledger.jsonl` 레코드 필드)

| 필드 | 설명 |
|---|---|
| `attribute` | `"gender"` / `"vehicle_type"` / `"military"` |
| `version` | PLR 프롬프트 버전 문자열 (예: `"plr_v1.4_cot"`) |
| `date` | ISO-8601 타임스탬프 |
| `n` | 채점된 라벨 객체 수 |
| `accuracy` | 전체 정확도 (correct / n) |
| `recall` | 클래스별 recall dict |
| `bias` | 속성별 bias 지표 (예: `female→male` 오분류율) |
| `confusion` | 전체 confusion matrix (행=정답, 열=예측) |
| `pred_unknown` | 모델 unknown율 `{rate, count}` — 강제 커밋 준수도 (전체 매칭 id 기준) |
| `n_label_unknown` | 사람도 판별 불가(label=unknown)로 정확도에서 제외된 크롭 수 |
| `seed_hash` | 씨딩 시점의 `core/ir HEAD` (`SEED.md` 기준) |
| `gemma_repo` | 실행 시점의 `IR_GEMMA_REPO` env |
| `dataset` / `model` / `pipeline` / `prompt_hash` | 실험 조합키 (P2-1) |

**라벨 정책**: 사람이 `unknown`으로 라벨한 크롭(판별 불가)은
accuracy/recall/bias에서 **제외**됩니다 — 강제 커밋 프롬프트(plr_v1.5_cot)
하에서 모델은 반드시 답하지만, 그 답을 채점할 정답이 없기 때문입니다.

`run_eval.py`는 ledger의 **가장 최근 이전 버전**과 비교해 `Δ accuracy / Δ bias`를
출력합니다.

---|---|
| `attribute` | `"search"` |
| `version` | PLR 프롬프트 버전 문자열 |
| `k` | 랭킹 컷오프 |
| `recall_at_k` | 전체 쿼리 평균 recall@k |
| `precision_at_k` | 전체 쿼리 평균 precision@k |
| `n_queries` | 평가된 쿼리 수 |
| `seed_hash` | 씨딩 시점의 `core/ir HEAD` |
| `gemma_repo` | 실행 시점의 `IR_GEMMA_REPO` env |
| `dataset` / `model` / `pipeline` / `prompt_hash` | 실험 조합키 (P2-1) |

`run_search_eval.py`는 ledger의 가장 최근 이전 버전과 비교해
`Δ recall@k / Δ precision@k`를 출력합니다.

experiment의 `formats:`/`reasons:` 축을 쓰면 version 태그에
`+json`/`+reason-off` 접미사가 붙어 셀이 구분됩니다.

---

## 문서 지도

| 문서 | 내용 |
|---|---|
| [README.ko.md](README.ko.md) / [README.md](README.md) | 이 문서 (한/영) |
| [GUIDE.html](docs/GUIDE.html) | 구조+사용법 원페이지 (배포용) |
| [STRUCTURE.html](docs/STRUCTURE.html) | 아키텍처 시각화 (계층·파이프라인·프롬프트 2-소스) |
| [HANDOFF.md](docs/HANDOFF.md) | 외부 프롬프트 엔지니어 워크플로 |
| [DATASET_SPEC.md](docs/DATASET_SPEC.md) | 데이터셋 디렉터리 형식 스펙 |
| [EXPERIMENT_SPEC.md](docs/EXPERIMENT_SPEC.md) | experiment 매트릭스 yaml 스펙 |
| [INSTALL.md](docs/INSTALL.md) | GPU/Gemma 실측 환경 셋업 |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | 세션 간 인수인계 (진행 상태·갭) |
| [SEED.md](SEED.md) | 원본 core/ir 해시 기록 |

---

## core/ir에서 재씨딩

```bash
./seed.sh /path/to/ziomilitary/core/ir
```

`core/ir` HEAD에서 lab 소스 파일을 다시 복사하고 `SEED.md`를 갱신합니다.
업스트림 PLR/검색 표면에 큰 변경이 있으면 실행해 lab을 동기화하세요.
