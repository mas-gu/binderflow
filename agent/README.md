# Agent guide — ProteaFlow binder design pipeline

You are an AI agent driving a protein binder design pipeline. **Read this file first, then the
one page that matches your task.** Every path and number here was verified on disk on 2026-09-28.

## Where to run these commands

Command examples are written as `python binders_pipeline_env/generate_binders.py ...`, i.e. from the
**parent** of the repo directory. If you cloned the repo (which `git clone` names `binderflow/`, not
`binders_pipeline_env/`), either:

- `cd` into the clone and drop the prefix: `python generate_binders.py ...`, or
- substitute your own directory name for `binders_pipeline_env/`.

Nothing in the code depends on the directory being called `binders_pipeline_env` — the scripts
resolve their own location. Only these docs' example paths assume it.

```
agent/
├── README.md          ← you are here: orientation + the rules that prevent expensive mistakes
├── 00_CAPABILITIES.md   what this machine can actually do right now (tools, envs, GPU, limits)
├── 01_INSTALL.md        installing the pipeline and its tools on a fresh machine
├── 02_RUN_BINDERS.md    running a binder design campaign end to end
├── 03_RERANK.md         re-scoring / re-filtering an existing run without re-designing
├── 04_OUTPUTS.md        what every output file and column means, incl. status.json schema
├── 05_TROUBLESHOOT.md   failure taxonomy: what each exit code and failure_kind means, and what to do
├── 06_AGENT_WORKFLOW.md ← THE ONE TO READ if you are driving this from a user request
└── 07_VISUALIZE.md      rendering designs to PNG, reading them, and what to report
```

**Driving the whole thing from natural language? Read `06_AGENT_WORKFLOW.md`.** It is the
discover → resolve → plan → validate → confirm → execute → interpret procedure, designed so
everything is checked before any GPU time is spent.

## The one-paragraph version

`generate_binders.py` designs protein binders against a target's binding site using up to 8 design
tools, validates them with Boltz-2 and Rosetta, scores and ranks them, and writes structures plus a
`rankings.csv`. `--render_designs` then turns the top hits into PNGs you can read and show. `rerank_binders.py` re-scores an existing run without redesigning. Both live in
`binders_pipeline_env/`. Runs take **hours to days** on one GPU.

## Rules that prevent expensive mistakes

**1. This is slow, GPU-bound, real science. A wrong argument costs hours, not seconds.**
Always `--dry_run` first. It validates arguments, prints the exact commands, and exits 0 without
using the GPU. **It does still create `--out_dir` and write placeholder outputs there**, so dry-run
into a throwaway directory. It deliberately writes no `status.json`, so that file's presence always
means a real run finished. A `test` mode run is ~4 hours; `production` is ~5 days.

**2. Check the exit code, and then check `status.json`. Not the log text.**

| Exit | Meaning | What to do |
|---|---|---|
| 0 | Ran, something was ranked, no tool failed | Read `status.json` + `rankings.csv` |
| 1 | Rejected before compute, or nothing usable | Read stderr — the message is specific and actionable |
| 2 | Malformed command line (argparse) | Fix the flags |
| 3 | Ran, but **no design could be ranked** | Read `status.json`; usually all tools failed or every design was filtered |
| 4 | **Partial success** — ranked some, but ≥1 tool failed | Usable results, but read `status.json:failure_kinds` |

`out_dir/status.json` is the machine-readable outcome. `run_params.txt` records *intent* at start,
not outcome — do not use it to decide whether a run worked.

**3. stdout is progress, stderr is problems.** Warnings, tool failures and errors go to stderr.
Do not scrape stdout looking for trouble.

**4. Never guess a residue number — preview the site.**
`--preview_site --target T.pdb --site "A:54,56"` costs seconds, needs no GPU, and exits 1 if the
residues are buried or scattered. It also renders `site_preview.png`, which you can look at
directly.
 `--site` residues that do not exist in the target chain are
warned about up front, and the run refuses to start if *none* of them exist. But a *partially*
wrong site still runs and silently drops the missing residues. Confirm the target's numbering
first (see `02_RUN_BINDERS.md` §"Choosing the site").

**5. Do not invent flags or tool names.** `--tools` accepts exactly:
`rfdiffusion, rfdiffusion3, boltzgen, bindcraft, bindcraft2, pxdesign, proteina, proteina_complexa`.
Anything else exits 1 with the valid list. `generate_binders.py --help` is authoritative; this
folder is a guide, not a substitute.

**6. Ask before launching anything longer than `--mode test`.** A `standard` or `production` run
commits the machine's GPU for a day or more. Confirm with the human first.

**7. Do not edit `binderflow/`.** It is a stale 2026-03-31 copy. The live tree is
`binders_pipeline_env/`.

## Known documentation hazard

Earlier project docs named scripts that **do not exist**
(`generate_binders3_newbeta_cpumulti.py`, its `_fix`/`_fixgpu` variants,
`generate_peptides5_new.py`). If you see those anywhere, they are stale. The real entry points are:

```
binders_pipeline_env/generate_binders.py     binder design
binders_pipeline_env/rerank_binders.py       re-score an existing run
binders_pipeline_env/generate_molecules.py   small-molecule SBDD
binders_pipeline_env/rerank_molecules.py     re-score molecules
predflow/prediction_flow/compare_predictions3.py   cross-tool structure comparison
```

**There is no peptide pipeline.** Do not attempt peptide design; no script implements it.

## Before you run anything

```bash
cd <repo>            # the binderflow checkout
conda activate boltz          # the host env for both pipeline scripts

# Start here. Needs no other arguments, touches no GPU, exits 0, and tells you
# exactly which tools this machine can run and what fits in its VRAM.
python binders_pipeline_env/generate_binders.py --capabilities

# And on an unfamiliar box, before trusting any of it (~1 min, no GPU):
python binders_pipeline_env/generate_binders.py --self_test
```

If `conda activate` is unavailable, call the interpreter directly:
`python`.
