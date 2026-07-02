# plr-prompt-lab Skills

This directory contains Claude Code skills for operators working with the
`plr-prompt-lab` dataset and evaluation workflow.

## Available Skills

### `prepare-dataset`

**File**: `skills/prepare-dataset/SKILL.md`

**When to use**: You have raw object crops (from a video's detection pipeline
or collected independently) and need to build a `plr-prompt-lab`-compliant
dataset — `crops/`, `labels.jsonl`, `manifest.yaml` — so you can run
`lab validate-dataset`, `lab run`, and `lab eval`.

**Done criterion**: `python3 lab.py validate-dataset --dataset <path>` exits 0
(`Result: PASS`).

---

## How to Use a Skill

### Option A — Load into Claude Code (interactive)

Copy `skills/prepare-dataset/SKILL.md` into your Claude skills directory
(typically `~/.claude/skills/` or the project-local `.claude/skills/`), then
invoke it from a Claude Code session:

```
/prepare-dataset
```

Claude Code will read the skill file and guide you through the steps.

### Option B — Follow manually

Open `skills/prepare-dataset/SKILL.md` in any text editor and follow the
step-by-step workflow directly. Every command references `lab.py` subcommands
that can be run without Claude Code.

---

## Skill Authoring Notes

Skills in this directory are docs-only: they describe operator workflows and
reference real `lab.py` subcommands. They do not add Python logic. If you add
a new skill, update this README with a one-line summary and its done criterion.
