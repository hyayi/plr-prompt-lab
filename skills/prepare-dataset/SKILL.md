# Skill: prepare-dataset

## Purpose / When to Use

Use this skill when you have raw object crops — either extracted from a video by
the DeepStream pipeline or collected independently — and need to turn them into a
`plr-prompt-lab`-compliant dataset so you can evaluate a PLR prompt version.

The dataset you build here is the direct input to `lab validate-dataset`,
`lab run`, and `lab eval`. Nothing in this workflow touches `lab.py` code or
any Python module — it is purely an operator data-preparation step.

**Do not use this skill** to evaluate an already-built dataset; for that, run
`lab run` and then `lab eval` directly (see `README.md` cycle diagram).

---

## Dataset Contract (reference first)

Before doing anything, read `docs/DATASET_SPEC.md`. That file is
authoritative. This skill summarises the layout and points to the parts you
will fill in:

```
<dataset>/
    crops/
        <obj_id>.jpg        # one JPEG per object (stem = obj_id)
    labels.jsonl            # human ground-truth (required)
    manifest.yaml           # dataset metadata (required)
    queries.jsonl           # optional — only for text-search eval
    predictions.jsonl       # written later by lab run — do not create manually
    attributes.jsonl        # written later by lab run — do not create manually
```

The three files you must produce: `crops/`, `labels.jsonl`, `manifest.yaml`.
Everything else (`predictions.jsonl`, `attributes.jsonl`) is written
automatically when you later run `lab run`.

---

## Step-by-step Workflow

### Step 1 — Organise crops

Each crop must be a JPEG named `<obj_id>.jpg` inside a `crops/` subdirectory
of your dataset directory. The filename stem is the object identifier used in
every other file.

```
my_dataset/
    crops/
        1003.jpg
        1013.jpg
        2045.jpg
```

Rules (from `docs/DATASET_SPEC.md` section 2):
- Stem is case-sensitive and must contain no extension (`.jpg` is stripped).
- You can use any alphanumeric string as `obj_id` (e.g. `1003`, `obj_001`,
  `cam2_frame450_id7`).
- Every `obj_id` in `labels.jsonl` **must** have a matching `.jpg`; missing
  crops cause a validation **error**. Extra crops with no label produce a
  **warning** (allowed by the spec).

**If you have crops from a video's detection pipeline**, they are already at
`$RESULT_PATH/<video_id>/objects/<obj_id>.jpg`. Copy the ones you want:

```bash
mkdir -p my_dataset/crops
cp $RESULT_PATH/vd_..._xxxx/objects/*.jpg my_dataset/crops/
```

**If you have arbitrary crops** (from your own source), rename them to match
your chosen `obj_id` scheme before copying:

```bash
# example: rename to zero-padded integers
ls my_raw_crops/*.jpg | nl -nrz -w4 | \
  awk '{printf "cp %s my_dataset/crops/%s.jpg\n", $2, $1}' | bash
```

Balance tip: aim for roughly equal counts per label class. The model's error
patterns are much harder to interpret when one class has 5× more samples than
another. 30–100 crops per expected class is a practical minimum; 150+ gives
stable accuracy numbers.

---

### Step 2 — Choose attribute and vocabulary

Pick one of the three supported attributes. The `attribute` value also
determines which label vocabulary `validate-dataset` will enforce (see
`docs/DATASET_SPEC.md` section 5):

| `attribute`    | Valid `label` values |
|---------------|----------------------|
| `gender`      | `male`, `female`, `unknown` |
| `vehicle_type`| `sedan`, `suv`, `hatchback`, `light_car`, `van`, `minivan`, `pickup_truck`, `truck`, `bus`, `taxi`, `ambulance`, `police_car`, `fire_truck`, `emergency_vehicle`, `motorcycle`, `scooter`, `bicycle`, `kickboard`, `construction_vehicle`, `vehicle_unknown` |
| `military`    | `military`, `civilian`, `unknown` |

You cannot mix attributes in one dataset directory.

---

### Step 3 — Produce `labels.jsonl`

This is the human ground-truth step. There are two paths.

#### Path A: crops come from a video indexed by the pipeline (preferred)

This path uses `lab build-golden` to extract stratified samples from the DB,
then `lab label` to record your corrections. It only works when:
- The video has been indexed (`ir_plr_index` rows exist for it in the DB).
- You have `$RESULT_PATH/<video_id>/objects/` crops on disk.

**Step A-1 — Build the golden set and contact sheets:**

```bash
python3 lab.py build-golden \
    --video vd_..._xxxx \
    --attribute gender \
    --dataset my_dataset/
```

`--video` is the video ID (e.g. `vd_001_0032`); `--attribute` must be one of
`gender`, `vehicle_type`, `military`; `--dataset` points to your target
directory (default: `eval/golden/<attribute>`).

This writes to `my_dataset/`:
- `predictions.jsonl` — the model's bootstrap predictions (one per obj_id)
- `index_map.json` — tile ID (e.g. `M1`, `F3`) → `obj_id` mapping
- `labels_template.csv` — spreadsheet skeleton for reference
- `crops/<obj_id>.jpg` — crop images copied from the objects dir

It also writes contact-sheet PNGs to `~/<attribute>_MALE.png` etc. (one per
predicted class). Each tile in the sheet is labelled with its tile ID.

Optional flags:
- `--per-class N` — cap samples per predicted class (default: 50)
- `--review-dir /path/` — override the browsable crop dir (default: `~/<attr>_eval`)

**Step A-2 — Review the contact sheets:**

Open `~/<attribute>_male.png` and `~/<attribute>_female.png` (or the
equivalent for your attribute). Identify the tile IDs that are wrong. You
only need to note the misclassified tiles; everything else keeps the model's
prediction as the ground truth.

**Step A-3 — Run `lab label` to write `labels.jsonl`:**

For `gender` (the `--female-in-male` / `--male-in-female` / `--unknown`
flags map directly to the contact-sheet tile IDs):

```bash
python3 lab.py label \
    --dataset my_dataset/ \
    --female-in-male M3,M7 \
    --male-in-female F2 \
    --unknown M40
```

Correction flags work in any order; no `--` separator is needed.

Tile IDs that are not mentioned keep the model's prediction. Tiles in
`--unknown` get label `unknown` regardless of prediction. Run `lab label
--help` (or inspect `eval/make_labels.py`) for the full flag list.

When `--dataset` is given, `lab label` automatically resolves
`my_dataset/index_map.json`, `my_dataset/predictions.jsonl`, and writes to
`my_dataset/labels.jsonl` — you do not need to pass those paths separately.

`make_labels.py` writes the `"label"` key that `validate-dataset` and `lab eval`
expect in `labels.jsonl`. After running `lab label`, verify the output schema:

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('my_dataset/labels.jsonl') if l.strip()]
assert all('obj_id' in r and 'label' in r for r in rows), 'schema mismatch'
print(f'OK — {len(rows)} rows')
"
```

#### Path B: arbitrary crops (no DB / no indexed video)

If you collected crops independently (from your own camera, downloaded frames,
or a non-indexed video), write `labels.jsonl` by hand. One JSON object per
line (UTF-8, no trailing commas):

```json
{"obj_id": "obj_001", "label": "female"}
{"obj_id": "obj_002", "label": "male"}
{"obj_id": "obj_003", "label": "unknown", "notes": "heavy occlusion"}
```

Required fields per line: `obj_id` (must match the crop filename stem exactly)
and `label` (must be in the vocabulary for your attribute — see Step 2).
Optional: `notes` (free text, any string). Do not add any other top-level keys
beyond these unless you understand they will be ignored by the evaluator.

Labeling honestly:
- Use `unknown` whenever you cannot determine the true class from the image
  alone (occlusion, low resolution, partial crop). Do not guess.
- `unknown` does not penalise accuracy the same way an incorrect label does —
  the evaluator treats it separately.
- Label from the crop, not from other context you may have about the video.
  The model only sees the crop.

---

### Step 4 — (Optional) Add `queries.jsonl` for text-search eval

Skip this step if you only want attribute-classification (`lab eval --mode attr`).

`queries.jsonl` enables `lab eval --mode search` (recall@k). Each line is a
search query with a list of `obj_id`s that are ground-truth relevant:

```json
{"query": "검은색 세단", "relevant": ["1003", "1013"]}
{"query": "빨간 오토바이", "relevant": ["2045"]}
```

Required fields: `query` (natural-language string, Korean or English) and
`relevant` (array of `obj_id` strings). Every `obj_id` in `relevant` must
exist either as a labeled entry or as a crop in `crops/`. Dangling references
are a validation error.

Write one query per real information need. 10–30 queries is typically enough
for a meaningful recall@k signal. Make sure at least one relevant crop exists
per query, and that the relevant set is complete (don't omit matching crops you
know about).

---

### Step 5 — Write `manifest.yaml`

```yaml
attribute: gender          # required — must match one of: gender, vehicle_type, military
n: 150                     # required — expected number of labeled objects
created: "2026-07-02"      # required — ISO date (today)
source_note: "vd_001_0032, frames 0-3600, stratified sample 50/class"
```

Required fields: `attribute`, `n`, `created`, `source_note`. The `n` field
should equal the number of lines in `labels.jsonl` (it is checked against the
actual count during validation). The `source_note` is free text — record
enough to reconstruct where these crops came from.

Optional fields (add if you know them):
```yaml
model: "plr_v1.4_cot"
prompt: "prompts/plr_v1.4_cot.yaml"
```

---

### Step 6 — Validate

Run `lab validate-dataset` until it exits 0:

```bash
python3 lab.py validate-dataset --dataset my_dataset/
```

The command runs ten checks (full list in `docs/DATASET_SPEC.md` section 10) and
prints a `PASS`, `WARN`, or `FAIL` line for each. It always ends with:

```
Summary: N crops, N labels, N error(s), N warning(s)
Result: PASS     # or FAIL
```

Exit code contract:
- **Exit 0** (PASS): no errors; warnings are allowed and do not block `lab run`.
- **Exit non-zero** (FAIL): one or more errors found; fix them before proceeding.

Common errors and fixes:

| Error message | Fix |
|---|---|
| `labels.jsonl not found` | Create the file per Step 3 |
| `invalid label 'X' for attribute gender` | Check vocabulary in Step 2; rename the value |
| `crop missing for obj_id 'X'` | Add `crops/X.jpg` or remove the label line |
| `manifest.yaml missing required field: n` | Add the field to `manifest.yaml` |
| `queries.jsonl: dangling obj_id 'X'` | Fix the `relevant` list in `queries.jsonl` |

Warnings (do not block):
- `crops without label entry` — crop files exist but have no corresponding
  label line; safe to leave if intentional (e.g. held-out images).

The dataset is ready for `lab run` when `validate-dataset` exits 0.

---

## What Comes Next

Once `validate-dataset` passes (exit 0), you can proceed to the GPU step:

```bash
# Stop the live ir service first (it holds the GPU)
python3 lab.py run --version plr_v1.4_cot --dataset my_dataset/   # -A 생략 = manifest 첫 속성
```

Then evaluate:

```bash
python3 lab.py eval --attribute gender --version plr_v1.4_cot --dataset my_dataset/
```

For search mode:

```bash
python3 lab.py eval --attribute search --mode search --version plr_v1.4_cot --dataset my_dataset/
```

See `README.md` for the full cycle diagram and GPU preconditions.

---

## Privacy Note

These crops contain images of real people or vehicles from CCTV footage.
Before sharing a dataset directory outside your organisation:
- Confirm you have authority to share the source footage.
- Redact or anonymise `source_note` if the video ID is sensitive.
- The `crops/` directory is gitignored in this repo (`eval/golden/*/crops/`);
  keep it that way — do not commit crop images to version control.

---

## Done Criterion

This skill's work is complete when:

```bash
python3 lab.py validate-dataset --dataset <path>
# → Result: PASS   (exit 0)
```

If `validate-dataset` exits non-zero, fix the reported errors and re-run.
Do not proceed to `lab run` or `lab eval` until validation passes.
