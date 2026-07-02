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
    manifest.yaml             # dataset metadata (required)
```

Only `crops/`, `labels.jsonl`, and `manifest.yaml` are required for
`lab validate-dataset` to pass. A ready-to-copy skeleton lives at
`examples/dataset_template/` (validates PASS as-is).

**The structure is fixed; the label set is yours.** Besides the required
manifest fields, a dataset may declare its own attribute (generic datasets):

```yaml
attribute: helmet                        # any name — presets: gender | vehicle_type | military
labels: [helmet, no_helmet]              # allowed label values (validate enforces)
pred_path: attributes.equipment[0].type  # where the pred lives in the PLR JSON (dots + [idx])
margin_path: ...                         # optional — model confidence path
bias_pair: [no_helmet, helmet]           # optional — headline bias [true, mistaken-as]
object_type_hint: person                 # optional — person | vehicle (default person)
```

The three PLR attributes (gender / vehicle_type / military) are built-in
presets — they need no declaration and double as reference examples of the
scheme (`evalkit/dataset.py` `PRESET_SPECS`).

### Creation procedure

1. crops — `crops/<obj_id>.jpg` (`lab build-golden` for production videos,
   or drop in arbitrary collected crops)
2. manifest — copy `examples/dataset_template/manifest.yaml`, adjust
3. labels — write `labels.jsonl` or use `lab label --dataset D ...`
   (human-undecidable crops → `unknown`: excluded from scoring, reported
   separately)
4. `lab validate-dataset --dataset D` → 5. `lab run` → `lab eval` →
   `lab gallery --dataset D` (visual check) `predictions.jsonl` and `attributes.jsonl`
are written by `lab run` and read by `lab eval`. (The `queries.jsonl`
search-dataset kind was removed 2026-07 — the lab is PLR-only.)

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

**`label: unknown` policy (forced-commit, plr_v1.5_cot)**: label a crop
`unknown` ONLY when a human cannot decide the attribute from the crop
(occlusion / extreme quality). Such crops are **excluded from
accuracy/recall/bias/confusion** by `lab eval` — under the forced-commit
prompt the model must still answer, and there is no ground truth to score
that answer against. They are reported separately as `n_label_unknown`,
and the model's own refusal rate is tracked as `pred_unknown` in the
ledger. Set them via `lab label --unknown <tile ids>`.

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

## 6. `predictions.jsonl` Schema (written by `lab run`)

```json
{"obj_id": "1003", "pred": "male", "reason": "broad shoulders", "margin": 0.8, "quality": 0.71}
```

- `margin` — the model's decision confidence for the evaluated attribute
  (from the prompt's `margins` block; `null` for attributes whose prompt
  emits none, e.g. vehicle_type/military). Under the forced-commit prompt
  this replaces the removed `unknown` escape hatch.
- `quality` — crop quality score in [0,1] (quality_gate, **measurement
  only** — it never gates the model call).

`lab eval` splits accuracy by both signals (`margin_stats` /
`quality_stats` in the ledger) to check that errors concentrate in
low-margin / low-quality crops. Both fields are optional — old files
without them still evaluate.

This file is the seed for the obj_id set that `re_score` processes. It is
overwritten by `lab run` and read by `lab eval`.

---

## 7. `attributes.jsonl` Schema (written by `lab run`)

```json
{"obj_id": "1003", "plr_json": { ...full PLR JSON... }}
```

The full PLR output per crop — raw material for per-slot analysis
(unknown-rate, margin distributions) beyond the single evaluated attribute.
The `plr_json` value must conform to `plr_schema.PERSON_SCHEMA` or
`plr_schema.VEHICLE_SCHEMA`.

---

## 8. Minimal Complete Example

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
