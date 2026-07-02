# plr-prompt-lab Skills

Claude Code skills for the lab's prompt-improvement workflow.

| Skill | When to use |
|---|---|
| `prepare-dataset/` | Turn raw crops into a validated lab dataset (structure fixed, labels yours — see docs/DATASET_SPEC.md) |
| `author-prompt/` | Create a NEW prompt version (contracts, function-per-file layout, domain lessons, promotion steps) |
| `improve-prompt/` | Turn experiment results + failure crops into a grounded improvement proposal (analyst → proposer → critic ⇄ reviser → judge, max 3 rounds) |

Each skill is self-contained in its `SKILL.md`. Typical loop:
prepare-dataset → `lab run`/`lab eval` → improve-prompt → author-prompt →
`lab experiment run` (A/B) → promotion (`lab port`).
