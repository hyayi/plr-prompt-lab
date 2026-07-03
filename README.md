# plr-prompt-lab

> 📖 **처음이라면 [`docs/GUIDE.html`](docs/GUIDE.html)을 브라우저로 여세요** —
> 설치부터 전체 개선 루프까지 복붙으로 따라 하는 실습 가이드(자체완결 HTML·오프라인)입니다.
> 아키텍처 시각화는 [`docs/STRUCTURE.html`](docs/STRUCTURE.html).

PLR(객체 속성 추출) **프롬프트를 측정하며 개선**하는 독립 실험 도구입니다.
운영 추론 서비스(`core/ir`)의 PLR 인풋/아웃풋 표면만 lean 추출했고
([SEED.md](SEED.md)), mock 경로에서는 **DB·Redis·GPU 없이** 전체 사이클이
돌아갑니다. 실측(크롭 재채점)만 전용 GPU + 사람이 라벨한 골든셋이 필요합니다.

```
데이터셋 준비 → run(모델 크롭당 1회) → eval -A all(라벨된 전 속성 채점)
      ↑              └→ gallery/report 자동 렌더 → 오답 분석 → 프롬프트 새 버전
      └────────────── experiment A/B → 이기면 승격(port) ──────────────┘
```

---

## 빠른 시작

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -q     # 기대: 99 passed, 4 xfailed (GPU·DB 불필요)
python3 lab.py demo --keep      # mock 전체 사이클 — datasets/demo/ 생성
```

여기까지로 GPU 없이 모든 기능(mock)을 체험할 수 있습니다. 실측 준비(CUDA
빌드 + Gemma GGUF)는 [INSTALL.md](docs/INSTALL.md).

---

## 무엇이 들어있나

- `plr_core.py` / `plr_prompts.py` / `plr_parse.py` / `plr_schema.py` /
  `preprocess.py` / `schema/vocab.yaml` — **core/ir과 byte-parity인 PLR 표면**
  (프롬프트 조합 · 응답 파싱 · 선언적 어휘 · 스키마 검증 · 마커 전처리)
- `prompts/<버전>/` — 버전당 디렉터리, 기능당 yaml 1개(person/vehicle/
  query_parser/vqa/retry). **프롬프트 텍스트는 전부 여기** — py에는 0줄
- `configs/` — 실험 파라미터 config(프롬프트 참조 + enum 축소·marker·sampling knob)
- `runners/` — re_score(크롭 루프)·experiment(매트릭스)·demo /
  `evalkit/` — dataset·validate·gallery·report·provenance·autorender
- `eval/` — 채점기(run_eval)·골든셋·ledger / `datasets/` — 사용자 데이터셋(gitignore)
- `gemma_model.py`(Model 프로토콜 + LabGemmaModel + MockModel) ·
  `gemma_backend.py`(GPU GGUF 로더 — `lab run` 전에는 import되지 않음)
- `skills/` — author-prompt(작성 계약) · improve-prompt(6역할 개선 루프) ·
  prepare-dataset · **co-change(동행 수정 매트릭스 — 표면 수정 전 필독)**

복사하지 않은 것: 서비스/DB/Redis/임베딩 계층 (`storage.py`, `indexing.py`,
`scheduler.py`, `text_embed.py` 등 — 검색 평가는 core/ir·cctv-eval 담당).

---

## 명령어

```bash
python3 lab.py demo [--keep]                       # GPU-free 온보딩 (mock 전체 사이클)
python3 lab.py validate-dataset --dataset D        # 데이터셋 형식 검증 (fail-loud)
python3 lab.py run -X plr_v1.5_cot --dataset D     # 재채점 — 크롭당 모델 1회, 전 속성 저장
                                                   #   (-A는 추출 뷰만; 생략=manifest 첫 속성)
python3 lab.py eval -A all --dataset D             # 라벨된 전 속성 채점 → ledger
                                                   #   (-A gender / -A a,b 도 가능; 끝나면
                                                   #    gallery.html/report.html 자동 렌더)
python3 lab.py gallery --dataset D [-A a,b]        # 크롭 시각화 — 속성별 태그 + AND/OR 오답 필터
python3 lab.py report [--ledger L] [--compare L2]  # ledger → 비교표·추이·confusion HTML
python3 lab.py experiment run my_ab.yaml [--strict]# 교차곱 매트릭스 (EXPERIMENT_SPEC.md)
python3 lab.py build-golden --video V -A gender    # 운영 비디오→골든셋 (운영자 단계)
python3 lab.py label --dataset D --female-in-male M3,M7 --unknown M9   # 사람 라벨
python3 lab.py port [--apply]                      # lab ↔ core/ir 표면 diff (승격 시에만)
python3 -m pytest tests/ -q                        # 전체 테스트
```

---

## 데이터셋 — 다속성 라벨 (핵심 개념)

한 크롭셋에 **속성별 정답을 함께** 라벨합니다. 모델 호출은 크롭당 1회
(`attributes.jsonl`에 plr_json 전체 저장)이고, `eval -A all`이 속성별 예측을
재추출해 전부 채점합니다 — 속성을 늘려도 GPU 비용은 그대로:

```jsonl
{"obj_id": "p1", "object_type": "person",  "labels": {"gender": "male", "helmet": "helmet"}}
{"obj_id": "v1", "object_type": "vehicle", "labels": {"vehicle_type": "sedan"}}
```

- manifest의 `attributes:` 맵으로 선언 — 프리셋(gender/vehicle_type/military)은
  빈 dict, 커스텀은 `labels:` + `pred_path:` (경로로 PLR JSON에서 추출)
- `object_type`이 크롭별 person/vehicle **프롬프트를 라우팅** (혼합 데이터셋)
- 라벨 정책: `unknown` = 사람도 판별 불가(채점 제외·별도 집계), 키 생략 = 미라벨(조인 제외)
- 템플릿: `cp -r examples/dataset_template datasets/내이름` → 상세는
  [DATASET_SPEC.md](docs/DATASET_SPEC.md)

---

## 실측 전제조건 (GPU)

1. **전용 GPU** — 운영 `ir` 서비스가 GPU/VRAM 점유 중이면 경합/OOM.
   중지·재기동은 관리자와 협의 (운영 재시작은 배포 행위 — 재인덱싱 유발).
2. **모델** — Gemma GGUF 다운로드 + env ([INSTALL.md](docs/INSTALL.md)).
3. **사람 라벨** — `labels.jsonl` 없이는 eval이 무의미.

크롭·라벨은 사적 CCTV 데이터 — `datasets/`, `eval/golden/*/crops/`는
gitignore이며 **절대 커밋 금지**.

---

## ledger 레코드 (지표)

| 필드 | 설명 |
|---|---|
| `attribute` / `version` / `date` / `n` | 속성 · 프롬프트 버전 태그 · 시각 · 채점 수 |
| `accuracy` / `recall` / `precision` / `f1` / `macro_f1` | 정확도 + 클래스별 성능 |
| `bias` | 헤드라인 오분류율 (예: female→male; manifest `bias_pair`로 선언) |
| `confusion` | confusion matrix (행=정답, 열=예측) |
| `pred_unknown` | 모델 unknown율 — 강제커밋(plr_v1.5_cot) 준수도 |
| `n_label_unknown` | 사람도 판별 불가로 제외된 크롭 수 |
| `margin_stats` / `quality_stats` | 신뢰/품질 구간별 accuracy — 캘리브레이션 |
| `dataset` / `model` / `pipeline` / `prompt_hash` / `seed_hash` | 실험 조합키 + 표면/씨드 해시 (provenance) |

`model`은 re_score가 predictions.jsonl 행에 남긴 스탬프에서 자동 해석되고,
`prompt_hash`는 프롬프트 표면 전체(prompts/**·vocab·파서·코어·스키마·전처리·
configs)의 해시라 knob 하나만 바꿔도 다른 값이 찍힙니다.

---

## 문서 지도

| 문서 | 내용 |
|---|---|
| [docs/GUIDE.html](docs/GUIDE.html) | **실습 가이드** — 설치·전체 루프·속성 레시피 (여기부터) |
| [docs/STRUCTURE.html](docs/STRUCTURE.html) | 아키텍처 시각화 |
| [docs/HANDOFF.md](docs/HANDOFF.md) | 외부 프롬프트 엔지니어 워크플로·규칙·반납 절차 |
| [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md) | 데이터셋 형식·파일 스키마 |
| [docs/EXPERIMENT_SPEC.md](docs/EXPERIMENT_SPEC.md) | experiment yaml 스펙 |
| [docs/INSTALL.md](docs/INSTALL.md) | 환경 셋업 (GPU 빌드·모델 다운로드) |
| [SESSION_HANDOFF.md](SESSION_HANDOFF.md) | 진행 상태·결정 기록 |
| [SEED.md](SEED.md) | 원본 core/ir 해시 기록 |

---

## core/ir에서 재씨딩

```bash
./seed.sh /path/to/ziomilitary/core/ir
```

core/ir HEAD에서 parity 표면 파일을 다시 복사하고 `SEED.md`를 갱신합니다.
반대 방향(승격)은 `skills/co-change/SKILL.md`의 승격 체크리스트를 따르세요.
