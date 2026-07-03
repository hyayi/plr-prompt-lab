# EXPERIMENT_SPEC.md — 실험 매트릭스 러너

`lab experiment run <experiment.yaml>`은
`datasets × models × prompts × pipelines × attributes`의 **교차곱**을
셀 단위로 열거하고, 각 셀마다:

1. **run** — registry를 통해 해당 파이프라인의 러너 실행 (re_score)
2. **eval** — 예측을 채점하고 ledger에 레코드 1줄 append

전 셀 종료 후 참여 데이터셋마다 gallery.html, ledger 옆 report.html이
**자동 생성**됩니다 (렌더 실패는 경고만 — 측정 결과는 무효화되지 않음).

## 스키마

```yaml
# 필수 축
datasets:   [./datasets/gender_v1]        # 데이터셋 디렉터리 경로 (1개 이상)
models:     [mock]                         # registry 모델명 (mock | gemma)
prompts:    [plr_v1.4_cot, plr_v1.5_exp]  # 버전 태그 — run_eval의 --version으로 전달
pipelines:  [plr]                          # plr (PLR 전용 lab — 검색은 2026-07 제거)
attributes: [gender]                       # PLR 속성 (plr 파이프라인 전용)

# 옵션
ledger:     ./eval/ledger.jsonl            # ledger 경로 (기본: eval/ledger.jsonl)
reasons:    ["on", "off"]                  # IR_PLR_REASON 축 (on | off), plr 셀 전용
```

### 필드 레퍼런스

| 필드 | 타입 | 필수 | 설명 |
|-------------|----------------|----------|-------------|
| `datasets`  | list[str]      | 예       | 데이터셋 디렉터리 경로. 각각 `crops/`, `labels.jsonl`, `predictions.jsonl`을 포함해야 한다. 상대경로는 **이 experiment yaml 파일 기준**으로 해석 (`ledger`와 동일). |
| `models`    | list[str]      | 예       | registry 모델명. `mock`은 GPU-free, `gemma`는 모델 가중치 필요. 미등록 이름은 **셀 실행 전에** 오류. |
| `prompts`   | list[str]      | 예       | 프롬프트 버전 태그. `run_eval`에 `--version`으로 전달된다. |
| `pipelines` | list[str]      | 예       | `plr` (속성 추출). 검색 파이프라인은 2026-07 제거. 미등록 이름은 셀 실행 전에 오류. |
| `attributes`| list[str]      | 아니오   | PLR 속성명 (예: `gender`, `vehicle_type`, `military`, 커스텀). 기본: `[""]`. |
| `ledger`    | str            | 아니오   | ledger JSONL 경로. 상대경로는 yaml 파일 기준. 기본: lab 루트의 `eval/ledger.jsonl`. |
| `reasons`   | list[str]      | 아니오   | `IR_PLR_REASON` env 축. 허용값 `on`, `off` (**따옴표 필수** — YAML에서 맨 on/off는 불리언으로 파싱됨). 기본: env 미변경. |

### reason 축의 의미

- `IR_PLR_REASON`은 person 템플릿의 CoT(`user_cot`) vs plain(`user_plain`)을
  선택한다. (`formats` 축은 레거시 JSON 프롬프트 경로와 함께 2026-07 제거 —
  YAML이 유일한 wire format.)
- **ledger 구분**: reason 축만 다른 셀은 버전 태그가 구분되어 찍힌다
  (`plr_v1.5_cot+reason-off`).

## 셀 열거

교차곱은:
`pipelines × datasets × models × prompts × attributes × reasons`
(뒤 세 축은 옵션 — 생략된 축은 셀 1개로 계산).

## registry를 통한 디스패치

- **plr 셀** → `registry.get_model(model)` +
  `re_score.re_score(attribute, model, golden_dir=dataset)` →
  `eval/run_eval.py main()`

각 셀의 ledger 레코드에는 `dataset`, `model`, `pipeline`,
`prompt_hash`(`provenance.prompt_hash`)가 함께 찍힌다.

## 검증

미등록 `models`/`pipelines` 값은 **어떤 셀도 실행되기 전에** 가용 이름
목록을 담은 `ValueError`를 낸다.

## fail-loud-but-continue

예외를 낸 셀은 잡아서 셀별 오류 줄로 기록하고(`status=failed`), 러너는
다음 셀로 계속 간다. 전 셀 종료 후 매트릭스 요약이 출력된다:

```
[experiment] === MATRIX SUMMARY ===
[experiment] total=4  ok=3  failed=1
[experiment] failed cells:
[experiment]   {dataset='./datasets/missing', model='mock', ...}
[experiment]     FileNotFoundError: Dataset directory not found: ./datasets/missing
```

## 종료 코드

| 코드 | 의미 |
|------|---------|
| 0    | 전 셀 성공 (또는 `--strict` 없이 1개 이상 성공) |
| 1    | `--strict` 지정 + 1개 이상 실패 |
| 2    | **전** 셀 실패 |

## CLI 사용법

```bash
# 매트릭스 실행
python3 lab.py experiment run examples/experiment.example.yaml

# 첫 실패에서 즉시 실패 처리 (CI용)
python3 lab.py experiment run examples/experiment.example.yaml --strict

# mock 모델로 GPU-free 스모크
python3 lab.py experiment run tests/fixtures/mock_experiment.yaml
```

## ledger 레코드 형태

각 셀이 ledger에 append하는 레코드:

```json
{
  "attribute": "gender",
  "version": "plr_v1.5_cot",
  "date": "2026-07-02T12:00:00",
  "n": 5,
  "accuracy": 1.0,
  "recall": {"female": 1.0},
  "bias": null,
  "confusion": {"female": {"female": 5}},
  "pred_unknown": {"rate": 0.0, "count": "0/5"},
  "n_label_unknown": 0,
  "seed_hash": "",
  "gemma_repo": "",
  "dataset": "./datasets/gender_v1",
  "model": "mock",
  "pipeline": "plr",
  "prompt_hash": "abc123"
}
```

## 예제 파일

전체 주석이 달린 예제는 `examples/experiment.example.yaml` 참고.
