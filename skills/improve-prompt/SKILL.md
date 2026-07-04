# Skill: improve-prompt

## Purpose / When to Use

Use this skill AFTER a submit (`lab run` → `lab submit --pull`, 여러 버전이면
각 버전 반복) to turn the pulled server results into a **grounded prompt
improvement proposal**. The skill is an agent workflow: it reads the ledger /
report, LOOKS AT the actual failure crops (gallery + raw images), and drives
a multi-role debate loop until a judge accepts the proposal.

Do NOT use this skill to "polish wording" without measurements — every claim
in the proposal must be backed by (a) a metric from the ledger and/or (b) a
named sample (`obj_id`) whose crop was actually viewed.

## Inputs (all produced by the lab)

**서버에서 `lab submit --pull`로 받는 파일 (채점은 서버 단일 — lab 로컬 eval 제거됨)**

| Source (pulled) | What it gives you |
|---|---|
| `<pulled>/metrics.json` | accuracy / recall / precision / F1 / confusion / bias / `pred_unknown` / `margin_stats` / `quality_stats` — 서버 채점 결과 |
| `<pulled>/report.html` | cross-version comparison table, trends, confusion (서버 렌더) |
| `<pulled>/gallery.html` | crops-vs-labels HTML, **wrong-first** — 오답 크롭 base64 내장 (서버 렌더) |

**로컬 run 산출물 (lab이 만들어 그대로 보유 — 서버로 안 감)**

| Source (local) | What it gives you |
|---|---|
| `<dataset>/predictions.jsonl` | per-crop pred + margin + quality |
| `<dataset>/attributes.jsonl` | full PLR JSON per crop (per-slot analysis) |
| `<dataset>/raw_responses.jsonl` | VERBATIM model text per crop + input/output token counts — check what the model actually said before parsing/normalisation touched it (e.g. was `gray` really answered, or coerced from an off-enum word?) |
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
- For suspicious labels, read the crop's row in `raw_responses.jsonl`: the
  stored label may be a NORMALISATION artifact (off-enum answer coerced to a
  fallback), which points at a vocabulary fix, not a prompt-wording fix.
  Token counts also reveal cost regressions between versions.
- Output: pattern list, each with obj_ids, image observations, and metrics.

### 2. 제안자 (Proposer)
- For each pattern, propose a CONCRETE change with the causal story: *why*
  this change should fix *this* pattern. Cite the pattern's obj_ids and
  observations as grounds.
- The levers are ALL the input knobs, not just wording:
  ① prompt template text (a new prompts/<V>.yaml)
  ② `enums:` ③ `preprocess.marker` ④ `sampling:` — knobs ②-④ go into a
  configs/<name>.yaml experiment config that REFERENCES the prompt by path
  (no template copies; see skills/author-prompt). Pick the lever the evidence
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
   changes) and/or a `configs/<name>.yaml` experiment config (knob changes
   referencing an existing prompt), with a header comment documenting the
   rationale per change.
3. **예상 효과** — the judge-approved measurable predictions.
4. **검증 계획** — 신·구 버전을 각각 `lab run` → `lab submit`(같은 서버 데이터셋)한 뒤
   서버 리더보드(`/d/<dataset>`)에서 지표 Δ를 확인하는 절차. (로컬 experiment 스윕은
   제거됨 — 비교는 서버 리더보드가 담당.)
5. (if capped) **미해결 쟁점** — the judge's outstanding objections.

## Ground rules

- Evidence beats plausibility: a proposal without a viewed sample is dropped.
- One variable at a time where possible — if two changes land in one version,
  say why they can't be separated.
- Never edit `plr_prompts.py` constants in this skill — new versions go to
  `prompts/<new>.yaml` only (the `--version` path). Promotion to constants /
  core/ir is a separate human-gated step (`lab port`).
