# Skill: improve-prompt

## Purpose / When to Use

Use this skill AFTER an experiment run (`lab experiment run` or
`lab run` + `lab eval`) to turn the results into a **grounded prompt
improvement proposal**. The skill is an agent workflow: it reads the ledger /
report, LOOKS AT the actual failure crops (gallery + raw images), and drives
a multi-role debate loop until a judge accepts the proposal.

Do NOT use this skill to "polish wording" without measurements — every claim
in the proposal must be backed by (a) a metric from the ledger and/or (b) a
named sample (`obj_id`) whose crop was actually viewed.

## Inputs (all produced by the lab)

| Source | What it gives you |
|---|---|
| `eval/ledger.jsonl` (or the experiment's ledger) | accuracy / recall / precision / F1 / confusion / bias / `pred_unknown` / `margin_stats` / `quality_stats` per version |
| `lab report --out report.html` | cross-version comparison table, trends, confusion rendering |
| `lab gallery --dataset D` | crops-vs-labels HTML, **wrong-first** — the visual evidence base |
| `<dataset>/predictions.jsonl` | per-crop pred + margin + quality |
| `<dataset>/attributes.jsonl` | full PLR JSON per crop (per-slot analysis) |
| `<dataset>/crops/<obj_id>.jpg` | raw crops — **Read these directly** when analysing errors |
| `prompts/<version>.yaml` | the prompt under improvement |

## The improvement loop (6 roles)

Run the loop with subagents (Agent tool) or sequentially in one context if
the error set is small. **Hard cap: 3 full rounds** — if the judge still
rejects after round 3, STOP and output the proposal marked
`status: unresolved` with the judge's outstanding objections listed. Never
loop indefinitely.

```
1. 분석자 ──► 2. 제안자 ──► 3. 비판자 ──► 4. 수정자 ──► (3↔4 반복) ──► 5/6. 판단자
   (error         (prompt        (attack        (revise)                 (accept /
    patterns)      changes)       evidence)                               reject+why)
```

### 1. 분석자 (Analyst)
- Read the ledger record(s) and confusion matrix; identify WHERE the errors
  are (which class pairs, which direction — e.g. `female→male` 0.31).
- Open the gallery / Read the wrong crops **as images**. Group failures into
  named patterns with member obj_ids, e.g.
  `P1 "야간 후면 실루엣" — {M3, M7, M12}: 어깨 실루엣만 보임, 모델이 전부 male`.
- Cross-check `margin_stats`: are the errors low-margin (model knows it's
  guessing) or high-margin (confidently wrong — the dangerous kind)?
  Same for `quality_stats`.
- Output: pattern list, each with obj_ids, image observations, and metrics.

### 2. 제안자 (Proposer)
- For each pattern, propose a CONCRETE change with the causal story: *why*
  this change should fix *this* pattern. Cite the pattern's obj_ids and
  observations as grounds.
- The levers are ALL the input knobs, not just wording:
  ① prompt template text (a new prompts/<V>.yaml)
  ② `enums:` ③ `preprocess.marker` ④ `sampling:` — knobs ②-④ go into a
  variants/<name>.yaml that REFERENCES the prompt version (no template
  copies; see skills/author-prompt §Variants). Pick the lever the evidence
  points at: a vocabulary-confusion pattern wants an enum change, not more
  instructions.
- Anti-patterns to avoid: piling on generic instructions ("be more careful"),
  changes that contradict the forced-commit contract (never re-introduce
  unknown), enum/vocabulary edits that break the query-side enum contract.

### 3. 비판자 (Critic)
- Attack each proposal: Would it regress other classes (check the confusion
  for the reverse direction)? Does history contradict it (prompt version
  comments record past failures — e.g. v0.6.1/v0.7 show example-bias and
  over-correction regressions)? Is the evidence actually in the images, or
  inferred? Is the change measurable with the current golden set?
- Every objection must itself cite evidence (a metric, a sample, or a
  documented precedent).

### 4. 수정자 (Reviser)
- Revise the proposals to answer the objections; drop proposals that cannot
  be defended. Return to 3 until the critic has no NEW objections
  (or the round cap hits).

### 5/6. 판단자 (Judge) — acceptance criteria
Accept only if ALL hold:
- [ ] Every surviving proposal cites ≥1 named sample (obj_id) whose crop was
      actually viewed, plus the supporting metric.
- [ ] Every critic objection is either resolved or explicitly accepted as a
      known risk with a monitoring plan (which metric would catch it).
- [ ] The expected effect is stated as a measurable prediction
      (e.g. "bias female→male 0.31 → ≤0.15, accuracy no worse than −0.02").
- [ ] The change respects the contracts: forced-commit (no unknown), enum
      vocabulary unchanged unless the query side is updated in the same
      proposal, constants↔yaml parity plan included.
If any fail → back to 2 (next round). After round 3 → emit `unresolved`.

## Output (deliverable)

1. **분석 리포트** — error patterns with obj_ids + image observations +
   metrics (including margin/quality calibration reading).
2. **수정안** — a ready draft: a new `prompts/<new_version>.yaml` (template
   changes) and/or a `variants/<name>.yaml` (knob changes referencing an
   existing prompt), with a header comment documenting the rationale per
   change.
3. **예상 효과** — the judge-approved measurable predictions.
4. **검증 계획** — the exact `lab experiment run` yaml comparing
   current vs new version on the same dataset.
5. (if capped) **미해결 쟁점** — the judge's outstanding objections.

## Ground rules

- Evidence beats plausibility: a proposal without a viewed sample is dropped.
- One variable at a time where possible — if two changes land in one version,
  say why they can't be separated.
- Never edit `plr_prompts.py` constants in this skill — new versions go to
  `prompts/<new>.yaml` only (the `--version` path). Promotion to constants /
  core/ir is a separate human-gated step (`lab port`).
