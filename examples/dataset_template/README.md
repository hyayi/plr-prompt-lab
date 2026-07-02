# 데이터셋 템플릿

구조는 고정, **라벨 집합은 데이터셋 작성자가 manifest.yaml에서 선언**합니다.
(gender / vehicle_type / military는 내장 프리셋 — 선언 없이 동작하는 예시입니다.)

```
my_dataset/
    manifest.yaml       # 속성·라벨 선언 (이 폴더의 예시 참고)
    labels.jsonl        # {"obj_id", "label"} — 사람 정답
    crops/
        <obj_id>.jpg    # obj_id = 파일명 stem
```

## 생성 절차

1. **크롭 준비** — `crops/<obj_id>.jpg`. (운영 비디오에서 뽑으려면
   `lab build-golden`, 임의 수집이면 그냥 파일을 넣으면 됨)
2. **manifest.yaml 작성** — 이 폴더의 예시를 복사해 attribute/labels/
   pred_path 를 본인 속성에 맞게 수정
3. **라벨링** — `labels.jsonl` 직접 작성 또는 `lab label --dataset D ...`
   (사람도 판별 불가한 크롭은 `unknown` — 채점에서 제외되고 별도 집계됨)
4. **검증** — `python3 lab.py validate-dataset --dataset my_dataset/`
5. **실행** — `lab run --attribute <속성> --version <V> --dataset my_dataset/`
   → `lab eval ...` → `lab gallery --dataset my_dataset/` (시각 확인)

전체 스키마: `docs/DATASET_SPEC.md`. 크롭 준비 상세: `skills/prepare-dataset/`.
