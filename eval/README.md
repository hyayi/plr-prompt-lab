# PLR 프롬프트 eval 루프 (loop-engineering)

프롬프트를 **개선 → 측정 → 채택/롤백**으로 반복 정교화하는 체계.
설계 근거: `.omc/specs/deep-interview-prompt-eval-loop.md`.

핵심 원칙: **측정 없이 프롬프트를 바꾸지 않는다.** (성별 프롬프트는 v0.5=여성편향 ↔
v0.7=여성편향 사이를 진동한 이력이 있음 — 골드셋으로 재기 전엔 어느 방향인지 모름.)

## 3 구성요소

| | 무엇 | 파일 |
|---|---|---|
| **C1 프롬프트 파일** | 버전별 yaml (수정대상=`person_user`/`vehicle_user`, 고정=`system`) | `core/ir/prompts/plr_v*.yaml` |
| **C2 골드셋 + 측정** | 라벨된 크롭 → per-attribute accuracy | `eval/build_golden.py`, `eval/run_eval.py` |
| **C3 버전 ledger + diff** | 버전별 점수 기록 + 직전 대비 diff, 버전-bump 시 자동 | `eval/ledger.jsonl`, `hooks/post-commit-prompt-eval` |

## 루프 (한 사이클)

```
1. 프롬프트 수정 (prompts/plr_vX.yaml + plr_prompts.py 동기, S0 parity gate로 검증)
2. version bump (PROMPT_VERSION ≤16자)            ← git commit 시 훅이 감지
3. 골드셋에 점수 매기기
     - 현재 버전(배포됨): predictions = DB 스냅샷 → 바로 측정
     - 새 버전(미배포): re_score.py 로 골드 크롭에만 Gemma 재실행(전체 재색인 X, GPU 필요)
4. run_eval.py --attribute <attr>  → accuracy + 혼동행렬 + 직전 버전 대비 Δ
5. 판단:  개선 → 채택(배포 + 재색인)   /   악화 → 롤백(이전 version yaml)
```

## 사용법

### 골드셋 구축 (속성별, 1회 + 필요시 확장)

```bash
cd core/ir/eval
# 영상의 색인 데이터에서 크롭 샘플 + 예측 + 컨택트시트 + 라벨템플릿 생성
python3 build_golden.py --video <video_id> --attribute gender --per-class 50
#   → golden/gender/{predictions.jsonl,index_map.json,labels_template.csv}
#   → ~/gender_eval/*.jpg (파일명에 예측), ~/gender_<class>.png (번호 붙은 시트)
```

지원 속성: `gender` (남/여), `vehicle_type` (자가용/버스/트럭…), `military` (군용/민간).

### 라벨링 (사람 ground-truth — 유일한 수작업)

컨택트시트(`~/gender_male.png` 등)를 보고 **오분류된 타일 번호만** 지정:

```bash
python3 make_labels.py \
  --female-in-male M3,M7,M12 \   # 남성예측인데 실제 여성
  --male-in-female F2 \          # 여성예측인데 실제 남성
  --unknown M40                  # 저해상도로 판단불가
#   → golden/gender/labels.jsonl  (지정 안 한 타일은 모델 예측을 정답으로 유지)
```

### 측정

```bash
python3 run_eval.py --attribute gender --version plr_v1.4_cot
#   accuracy, per-class recall, 혼동행렬, bias(여→남 오분류율),
#   직전 ledger 버전 대비 Δ, ledger.jsonl 에 append
```

## C3 자동화 — 버전-bump 감지

`hooks/post-commit-prompt-eval` 를 git 훅으로 설치하면, `prompts/*.yaml` 이 바뀐
커밋에서 **현재 라벨된 골드셋으로 자동 eval** 후 ledger diff 를 출력한다.

```bash
ln -sf ../../core/ir/eval/hooks/post-commit-prompt-eval \
   $(git rev-parse --git-path hooks/post-commit)   # core/ir 저장소 기준
```

## ledger.jsonl

버전×속성 키의 append-only 기록:
```json
{"attribute":"gender","version":"plr_v1.4_cot","date":"...","n":63,
 "accuracy":0.90,"recall":{...},"bias":{"pair":"female->male","rate":0.31},"confusion":{...}}
```

## 파일

- `build_golden.py` — 영상+속성 → 골드셋 아티팩트(크롭·예측·시트·라벨템플릿)
- `make_labels.py` — 시트 타일 번호 → labels.jsonl (오분류만 지정)
- `run_eval.py` — predictions+labels → accuracy/혼동/bias/diff, ledger append
- `re_score.py` — (예정) 변경된 프롬프트를 골드 크롭에만 Gemma 재실행 → 새 predictions (재색인 없이 새 버전 측정, GPU 필요)
- `golden/<attr>/` — predictions.jsonl · index_map.json · labels.jsonl · labels_template.csv
- `ledger.jsonl` — 버전별 점수 기록
