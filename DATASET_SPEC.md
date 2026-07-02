# PLR Prompt Lab — Dataset Directory Specification

This document defines the exact directory layout and file schemas that a
dataset directory must satisfy for use with `lab validate-dataset`,
`lab run`, and `lab eval`.

The existing `eval/golden/<attribute>/` directories are themselves valid
datasets — they were the reference layout from which this spec was derived.

---

## 1. Directory Layout

```
<dataset>/
    crops/
        <obj_id>.jpg          # one crop image per object (JPEG)
    labels.jsonl              # human ground-truth labels (required for eval)
    predictions.jsonl         # model output (written by `lab run`; seed for obj_id set)
    attributes.jsonl          # full PLR JSON per obj_id (written by `lab run`)
    queries.jsonl             # text-search queries with relevant obj_ids (optional)
    manifest.yaml             # dataset metadata (required)
```

Only `crops/`, `labels.jsonl`, and `manifest.yaml` are required for
`lab validate-dataset` to pass. `predictions.jsonl` and `attributes.jsonl`
are written by `lab run` and read by `lab eval`. `queries.jsonl` is optional
and only needed for search-mode evaluation.

---

## 2. obj_id Rules

- The **stem** of each crop filename is the `obj_id`:
  `crops/abc123.jpg` → `obj_id = "abc123"`.
- The `obj_id` field in every `labels.jsonl` line must equal the crop stem
  exactly (case-sensitive, no extension).
- `labels.jsonl` and `predictions.jsonl` use the same `obj_id` namespace.
- There must be a one-to-one correspondence between labeled obj_ids and crop
  files; extra crop files (unlabeled) produce a **warning**, missing crop
  files produce an **error**.

---

## 3. `manifest.yaml` Schema

```yaml
attribute: gender          # required — PLR attribute this dataset covers
n: 150                     # required — expected number of labeled objects
created: "2026-07-01"      # required — ISO date of dataset creation
source_note: "video vd_001_0032, frames 0–3600"  # required — provenance note

# Optional fields (may be absent):
model: "plr_v1.4_cot"      # PLR prompt version used for initial predictions
prompt: "prompts/plr_v1.4_cot.yaml"  # prompt file path
```

**Required fields**: `attribute`, `n`, `created`, `source_note`.

The `attribute` value governs which label vocabulary is enforced by
`validate-dataset` (see section 5).

---

## 4. `labels.jsonl` Schema

One JSON object per line (UTF-8, no trailing comma). Blank lines are ignored.

```json
{"obj_id": "1003", "label": "female"}
{"obj_id": "1013", "label": "male", "notes": "ambiguous — borderline case"}
```

**Required fields per line**:

| Field    | Type   | Description                                      |
|----------|--------|--------------------------------------------------|
| `obj_id` | string | Object identifier; must match a crop stem        |
| `label`  | string | Human ground-truth value (see vocabulary below)  |

**Optional fields per line**: `notes` (free text), any additional attributes.

---

## 5. Label Vocabularies

The allowed values for `label` depend on the `attribute` declared in
`manifest.yaml`. Labels outside the vocabulary for the declared attribute
cause a **validation error** (not a warning) because they will silently
corrupt scoring.

### `attribute: gender`

| Value     | Meaning                                          |
|-----------|--------------------------------------------------|
| `male`    | Person presenting male gender expression         |
| `female`  | Person presenting female gender expression       |
| `unknown` | Indeterminate from the crop (occlusion, quality) |

### `attribute: vehicle_type`

Allowed values are the `type_topk` labels from `plr_schema.VEHICLE_TYPE_ENUM`:

```
sedan, suv, hatchback, light_car, van, minivan,
pickup_truck, truck, bus, taxi,
ambulance, police_car, fire_truck, emergency_vehicle,
motorcycle, scooter, bicycle, kickboard,
construction_vehicle, vehicle_unknown
```

### `attribute: military`

| Value      | Meaning                                  |
|------------|------------------------------------------|
| `military` | Military person/vehicle                  |
| `civilian` | Non-military                             |
| `unknown`  | Cannot be determined from the crop       |

---

## 6. `queries.jsonl` Schema (optional)

Present only for search-mode datasets. One JSON object per line.

```json
{"query": "검은색 차", "relevant": ["1003", "1013"]}
{"query": "빨간 오토바이", "relevant": ["2045"]}
```

**Required fields per line**:

| Field      | Type            | Description                                         |
|------------|-----------------|-----------------------------------------------------|
| `query`    | string          | Korean (or English) natural-language search query   |
| `relevant` | array of string | obj_ids that are ground-truth relevant for this query |

Every `obj_id` in `relevant` must exist as either a labeled obj_id or a crop
in the dataset. Dangling references cause a **validation error**.

---

## 7. `predictions.jsonl` Schema (written by `lab run`)

```json
{"obj_id": "1003", "pred": "male", "reason": "broad shoulders"}
```

This file is the seed for the obj_id set that `re_score` processes. It is
overwritten by `lab run` and read by `lab eval --mode attr`.

---

## 8. `attributes.jsonl` Schema (written by `lab run`)

```json
{"obj_id": "1003", "plr_json": { ...full PLR JSON... }}
```

Used by `lab run` search mode and `lab eval --mode search` to build
full-attribute candidate rows for `search_core.run_search`. The `plr_json`
value must conform to `plr_schema.PERSON_SCHEMA` or `plr_schema.VEHICLE_SCHEMA`.

---

## 9. Minimal Complete Example

```
my_gender_dataset/
    manifest.yaml
    labels.jsonl
    crops/
        obj_001.jpg
        obj_002.jpg
        obj_003.jpg
```

**`manifest.yaml`**:
```yaml
attribute: gender
n: 3
created: "2026-07-01"
source_note: "synthetic example — three CCTV person crops"
```

**`labels.jsonl`**:
```json
{"obj_id": "obj_001", "label": "female"}
{"obj_id": "obj_002", "label": "male"}
{"obj_id": "obj_003", "label": "unknown", "notes": "heavy occlusion"}
```

**`crops/`**: three JPEG images named `obj_001.jpg`, `obj_002.jpg`, `obj_003.jpg`.

Validate with:
```bash
python3 lab.py validate-dataset --dataset my_gender_dataset/
```

---

## 10. `validate-dataset` Check List and Exit-Code Contract

`lab validate-dataset --dataset <path>` runs the following checks in order.
Each produces a `PASS`, `WARN`, or `FAIL` line.

| # | Check | Level on failure |
|---|-------|-----------------|
| 1 | `manifest.yaml` present and parseable as YAML | **error** |
| 2 | `manifest.yaml` contains all required fields (`attribute`, `n`, `created`, `source_note`) | **error** |
| 3 | `labels.jsonl` present | **error** |
| 4 | Every line in `labels.jsonl` is valid JSON with `obj_id` and `label` | **error** |
| 5 | Every `label` value is in the allowed vocabulary for the manifest's `attribute` | **error** |
| 6 | `crops/` directory present | **error** |
| 7 | Every labeled `obj_id` has a matching `<obj_id>.jpg` in `crops/` | **error** |
| 8 | Crop files without a matching label entry | **warning** |
| 9 | `queries.jsonl` (if present): every line is valid JSON with `query` and `relevant` | **error** |
| 10 | `queries.jsonl` (if present): every `relevant` obj_id exists in the dataset | **error** |

**Exit code contract**:
- Exit **0**: no errors (warnings are allowed).
- Exit **non-zero**: one or more errors found.

Output always ends with a one-line summary:
```
Summary: N crops, N labels, N error(s), N warning(s)
Result: PASS   # or FAIL
```

The programmatic API `validate_dataset(path) -> bool` returns `True` on
pass, `False` on any error. It prints the same lines to stdout.
