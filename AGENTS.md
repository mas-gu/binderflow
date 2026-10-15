# binderflow — agent instructions

> **This file is the project's own instructions and is self-contained.**
> If you are working on a machine that also has its own `CLAUDE.md` / `AGENTS.md` in a *parent*
> directory, that belongs to the operator of that machine — it describes their campaigns, paths and
> hardware. **For anything about this project, this file and `agent/` are authoritative.** Where a
> parent file and this one disagree about binderflow, follow this one, and verify against the code:
>
> ```bash
> python generate_binders.py --capabilities --deep
> ```

Multi-tool protein binder design. Eight design tools, Boltz-2 + Rosetta validation, scoring,
ranking and visualisation.

> **AI agents: read `agent/README.md` first**, then `agent/06_AGENT_WORKFLOW.md` if you are driving
> this from a user request. Those docs are task-oriented and maintained against the running code.

## Never trust documentation about this machine — ask it

```bash
python generate_binders.py --capabilities          # what is configured (fast)
python generate_binders.py --capabilities --deep    # what actually imports (~1 min)
python generate_binders.py --self_test              # full health check; exit 0 = healthy
```

**If the docs and the code disagree, the code wins.** A tool whose directory exists can still be
broken — `--self_test` is what tells you.

## First run on a new machine

1. `cp config.example.yaml config.yaml`, then set `software_dir` and `weights_dir` — every tool
   path resolves through these (env vars `BINDER_SOFTWARE_DIR` / `BINDER_WEIGHTS_DIR` override).
2. Install tools: `agent/01_INSTALL.md`. The `boltz` host env first, then the `bindcraft` v1 env
   (it carries the PyRosetta stack used by **every** tool's scoring), then the rest.
3. `python generate_binders.py --self_test` until it exits 0.

## The rules that prevent expensive mistakes

- **Runs take hours to days on a GPU.** Always `--dry_run` first; it validates everything and costs
  seconds. Confirm with the user before anything above `--mode test`.
- **Verify the binding site before spending GPU**: `--preview_site --target T.pdb --site "A:54,56"`
  exits 1 if the residues are buried or scattered, and renders a PNG you can look at. Residue
  numbers are meaningless without the file they refer to — see `targets/`.
- **Read `out_dir/status.json`, not the exit code alone.** Exit 0 = clean, 3 = nothing ranked,
  4 = partial (some tools failed). `failure_kinds` says *why* — `oom` vs `missing_env` vs `bad_args`.
- **stdout is progress, stderr is problems.** A clean run emits no stderr.
- **Look at the results**: `--render_designs --results_dir <run> --site "<site>"` writes PNGs.
  A high score on a binder docked to the wrong face is still wrong.

## Entry points

| Script | Use |
|---|---|
| `generate_binders.py` | design campaign |
| `rerank_binders.py` | re-score an existing run; `--rank_only` needs no GPU |
| `generate_molecules.py` / `rerank_molecules.py` | small-molecule SBDD |

Full detail in `agent/`.
