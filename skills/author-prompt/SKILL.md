# Skill: author-prompt

## Purpose / When to Use

Use this skill when creating a NEW PLR prompt version (`prompts/<version>.yaml`)
— whether from scratch, from an improve-prompt proposal, or as a manual
experiment variant. It encodes the contracts a version must respect and the
exact authoring procedure, so a new version is runnable with
`lab run --version <V>` the moment the yaml lands.

## Non-negotiable contracts (violating any of these breaks the lab or prod)

1. **Forced-commit (v1.5+)** — never offer `unknown` as an answer option
   (no `<a|b|unknown>` literals, no "if uncertain pick _unknown"). Low
   confidence is expressed through the `margins` block. The parity test
   `test_live_prompt_never_offers_unknown` enforces this for the constants;
   keep experiment versions to the same rule unless the experiment is
   explicitly about re-introducing an escape hatch.
2. **Version name ≤ 16 chars** — it is stamped into a `varchar(16)` column
   (`tb_cache_function_response.plr_prompt_version`). Pattern:
   `plr_v<major>.<minor>_<tag>` (e.g. `plr_v1.6_cot`). Never reuse a name:
   the version string is the reindex trigger and the ledger key.
3. **Enums are INJECTED, not hand-written** — keep the `{colors}`,
   `{upper_types}`, `{lower_types}`, `{equips}`, `{actions}`,
   `{military_enum}`, `{vehicle_types}` placeholders. Set
   `commit_enums: true` so the provider filters `*unknown*` variants out of
   the injected lists. Hand-writing enum values silently drifts from
   plr_schema and breaks the search-side enum contract.
4. **Output schema = parser contract** — the YAML shape the prompt asks for
   must stay parseable by `plr_prompts.parse_plr_response`. Adding /
   renaming an output field requires a parser change in the same proposal
   (and that is a constants-level change, not an experiment-yaml change).
5. **`query_parser:` block — optional in the lab, required at promotion** —
   the lab never sends it (search removed), so an EXPERIMENT version may
   omit it entirely. But in core/ir one version yaml serves BOTH the PLR and
   the search prompt (the provider's build_query_parser_messages KeyErrors
   without it), so when a version is ported to core/ir the block must be
   added back (copy verbatim from the current production version).
6. **Marker instructions stay** — the yellow corner-mark paragraphs exist
   because multi-figure crops mislabel without them; do not remove them to
   "shorten" the prompt.

## Authoring procedure

```bash
# 1. Copy the current version as the base
cp prompts/plr_v1.5_cot.yaml prompts/plr_v1.6_cot.yaml
```

2. Edit the header comment: version name, date, and a per-change rationale
   (this is the version's changelog — future sessions rely on it).
3. Apply the changes — one hypothesis per version where possible.
4. There is NO registration step: `lab run --version plr_v1.6_cot` loads the
   yaml directly via FilePromptProvider.
5. Verify:

```bash
python3 -c "
from providers.file_prompt_provider import FilePromptProvider
for h in ('person','vehicle'):
    m = FilePromptProvider(version_override='plr_v1.6_cot').build_plr_messages(h)
    print(h, 'OK', len(m[1]['content'][1]['text']))"
python3 lab.py experiment run my_ab.yaml   # prompts: [plr_v1.5_cot, plr_v1.6_cot]
```

## YAML structure template

```yaml
# plr_v1.6_cot — <one-line intent> (<date>)
# Changes vs plr_v1.5_cot:
#   - <change 1>: <rationale, cite obj_ids/metrics if from improve-prompt>
format: yaml            # wire format the templates emit (parser follows IR_PLR_FORMAT)
commit_enums: true      # provider injects *unknown*-filtered enum lists
plr:
  system: |-
    <system prompt — role + output discipline + commit rule>
  person_user: |-
    <CoT person template — used when IR_PLR_REASON=on (production default)>
  person_user_no_reason: |-
    <plain person template — IR_PLR_REASON=off>
  vehicle_user: |-
    <vehicle template>
# query_parser:         # OPTIONAL for lab experiments — REQUIRED when the
#   system: |-           # version is ported to core/ir (one yaml serves both
#     ...                # prompts there). Copy verbatim from the production
#   user: |-             # version at promotion time.
#     ...
```

## Experiment configs — parameter combinations (configs/<name>.yaml)

A prompt yaml holds templates ONLY. To cross a prompt with other input
knobs (enum lists, preprocessing, sampling) WITHOUT copying template text,
write an experiment config that references each component by path (see
`configs/example.yaml`):

```yaml
prompt: prompts/plr_v1.5_cot.yaml    # component by path (bare name also ok)
enums: { colors: [black, white, red] }   # inline, or a yaml path
preprocess: { marker: false }
sampling: { max_tokens: 256, temperature: 0.2 }
```

`lab run --version <config-name>` resolves the combination; the ledger
stamps the config name; prompt_hash covers configs/*.yaml. Same prompt ×
N knob-sets = N config files, zero template copies.

**Enum overrides may only NARROW the vocabulary** (subset — enforced,
fail-loud). The parser coerces every slot back onto the schema vocabulary
(measured: color `crimson`→`gray`, action `crawling`→`posture_unknown`), so
offering the model a word the schema doesn't know is a half-experiment —
the answer gets thrown away at parse time.

**EXTENDING a vocabulary = edit `schema/vocab.yaml`** (the declarative
single source — plr_schema derives constants, parser normalisation, group
functions and JSON schemas from it, so injection AND parsing move together
automatically). vocab.yaml is part of the port/hash surface: the change is
provenance-stamped, `lab port` diffs it against core/ir, and promotion
copies it home like any other surface file.

## Domain lessons (encoded history — do not relearn these the hard way)

- **Measure before you change** (`eval/README.md`): the gender prompt
  oscillated female-biased ↔ male-biased between v0.5 and v0.7; without the
  golden set you cannot tell which direction you are pushing.
- **Example bias is real**: v0.6.1 changed only the gender_reason examples
  (hair-first) and accuracy dropped — examples steer more than instructions.
- **Over-correction is real**: v0.7's hard "shoulders alone unreliable"
  directive pulled correct answers back to the opposite bias (9/14).
- **YAML beat JSON for a reason**: v0.4 JSON burned ~80% of output tokens on
  structure and broke mid-string on ~5% of objects. Stay with bare
  `key: value` YAML lines.
- **Reason-before-label is the CoT win**: `gender_reason` BEFORE `gender`
  (commit to evidence first) took a 14-object review from 1/14 to 13/14.
- **Hints must be precision-aware**: the military hint pairs every positive
  cue list with an explicit civilian fallback ("without corroboration answer
  civilian") — a bare cue list inflates false positives.

## Promotion (experiment version → production default)

Winning a lab A/B does NOT deploy anything. Promotion is a separate,
human-gated step — and since the 2026-07 declarative-prompt refactor it is
two lines, not a parity dance:
1. Ensure the winning version yaml has the `query_parser:` block (copy from
   the previous production version) — core/ir serves search from it.
2. Bump `PROMPT_VERSION_YAML_COT` in plr_prompts.py to the new version name
   (plr_prompts LOADS that yaml — this is also the lazy-reindex trigger).
3. `lab port` → apply to core/ir → core/ir tests → deploy decision.
4. Experiment knobs go back to their HOME files, not to the config yaml:
   `enums:` narrowing → schema/vocab.yaml · vocabulary extension →
   schema/vocab.yaml (already port surface) · `preprocess.marker` →
   indexing's marker call site (preprocess.py) · `sampling:` → the gemma
   generate call sites. A knob that wins in the lab but is not carried home
   silently reverts in production.
