# The complete agent workflow — request to result, one pass

This is the decision procedure for driving the pipeline from a natural-language request without
trial and error. **Everything is verified before anything expensive starts.** A binder run is
4 hours to 5 days; there is no cheap retry, so all validation happens up front.

```
user request
   │
   ├─ 1. DISCOVER      --capabilities                      (seconds, no GPU)
   ├─ 2. RESOLVE       target, chain, site  + --preview_site  (seconds, no GPU)
   ├─ 3. PLAN          tools, length, mode, filters         (no I/O)
   ├─ 4. VALIDATE      --dry_run                            (seconds, no GPU)
   ├─ 5. CONFIRM       show the human the plan + the cost   ← required above --mode test
   ├─ 6. EXECUTE       detached, poll                       (hours→days, GPU)
   ├─ 7. INTERPRET     status.json → rankings.csv           (seconds)
   └─ 8. VISUALISE     --render_designs, then LOOK at it    (seconds, no GPU)
```

If any step fails, **stop and report** — do not improvise past a failed check.

---

## 1. Discover — never assume, ask the machine

```bash
cd <repo>            # the binderflow checkout
python binders_pipeline_env/generate_binders.py --capabilities
```

Needs no other arguments, touches no GPU, exits 0. Returns JSON:

| Key | Use |
|---|---|
| `tools_available` / `tools_unavailable` | The **only** valid `--tools` values. Never offer an unavailable tool. |
| `tools[t].kind` | `iptm_native` (cheap without `--reprediction`) vs `backbone_only` (always needs Boltz-2) |
| `tools[t].accepts_cif_target` | `false` for `rfdiffusion` and `bindcraft` — they are PDB-only |
| `vram_gb`, `limits.*` | How big a target/binder fits. See step 3. |
| `modes[m].total_all_tools` | How many designs, hence how long |
| `bindcraft2.modalities` / `.filter_presets` | Valid BC2 values |
| `exit_codes`, `failure_kinds` | How to read the outcome |

This is read from live code and the live filesystem, so it cannot go stale. Prefer it over anything
written in prose, **including this folder**.

## 2. Resolve the target — the step most likely to be silently wrong

The user says "target PD-L1 at the PD-1 binding site". You need: a file, a chain, and residue
numbers **in that file's numbering**.

```bash
python -c "
import sys; sys.path.insert(0,'binders_pipeline_env')
from generate_binders import get_chain_info
for c, i in get_chain_info('TARGET.pdb').items():
    print(c, i['length'], 'residues', min(i['residues']), '-', max(i['residues']))"
```

Then **preview the site** — this is the check that catches a wrong epitope, not just a wrong number:

```bash
python binders_pipeline_env/generate_binders.py --preview_site \
    --target TARGET.pdb --site "A:54,56,66,115" --out_dir /tmp/sitecheck
echo "exit=$?"        # 0 = plausible, 1 = implausible
```

It needs only `--target` and `--site`, uses no GPU, and prints a per-residue exposure map, the
patch spread, and a verdict. Add `--json` for a parseable dict, `--no_render` to skip the image.

It writes three things to `--out_dir`:

| File | For |
|---|---|
| `site_preview.png` | **a ray-traced image — if you have vision, LOOK at it.** A real epitope is one contiguous orange patch on the grey surface; a wrong one is scattered specks, or nothing visible because the residues are buried |
| `site_preview.pml` | PyMOL scene for the human |
| `site_preview.cxc` | ChimeraX scene for the human |

Three failure modes it catches, each a real class of mistake:

- **buried residues** — solvent-inaccessible, so no binder can reach them. Almost always means the
  numbering belongs to a different file. Exit 1.
- **scattered residues** — Rg far larger than a compact patch of that size would have. These are
  several faces of the protein, not one epitope; a binder cannot touch them all. Exit 1.
- **missing residues** — not in the chain at all; they get silently dropped downstream. Warning.

**Numbering from a paper, UniProt or a different PDB entry is frequently offset from the file you
have.** Exit 1 here means stop and ask — do not "try it and see", because seeing takes hours. If
the preview passes but you are still unsure the epitope is the one the user meant, show them the
PNG and ask.

If the user gave no site at all, do not invent one. Ask.

## 3. Plan — map the request onto flags

**Tools.** Default to 2-3 diverse, available, iPTM-native tools: `bindcraft2,boltzgen,pxdesign`.
Add `rfdiffusion`/`proteina` for backbone diversity at the cost of mandatory Boltz-2 validation.
If the target is `.cif`, drop `rfdiffusion` and `bindcraft`.

**Length.** `60-80` unless the user asked otherwise. Valid range 10-500. Scaffolded BindCraft2
modalities (VHH/scFv/Fab/ARP) ignore it.

**Size check — do this arithmetic before running.**
`total_N = target_length + max_binder_length`, compared against `limits` from step 1:

| Condition | Consequence |
|---|---|
| `total_N > 340` | Complexa and RFdiffusion3 drop to batch_size 1 (slower, automatic) |
| `total_N > limits.bindcraft2_max_total_n` (from `--capabilities`; ~416 on a 20 GB card) | BindCraft2 will be very slow or OOM |
| `target_length > 300` | PXDesign auto-crops (automatic, fine) |
| `total_N > 700` | Boltz-2 complex validation may OOM |

If total_N exceeds the BC2 limit, say so and propose a smaller binder or a cropped target rather
than launching something that will OOM.

**Mode.** `test` unless the user explicitly asked for a large campaign. See step 5.

**Filters.** Quality filters are **rerank-only**. In the design run use at most one geometric
filter, and set it **loose**: `--min_site_interface_fraction 0.25` regardless of site size.

This deliberately contradicts the "SIF by site size" table in `02_RUN_BINDERS.md` (which says 0.5
for an 8+ residue site). Both are right for different stages, and the asymmetry is the reason:

| Stage | SIF | Why |
|---|---|---|
| **Design run** | **0.25**, always | Over-filtering here is unrecoverable — the designs are discarded and getting them back costs another full run |
| **Rerank** | the size-scaled value (0.5 for 8+, 0.25-0.30 for 4-5, 0.20-0.25 for 3) | Free, repeatable, and you can try several thresholds in minutes |

**Filter loosely when it costs GPU, tightly when it costs seconds.** Everything else belongs in a
free `--rank_only` rerank afterwards.

**Reprediction.** Leave `--reprediction` off unless the user wants uniform cross-tool scoring. It
multiplies runtime, and with it on only designs that get `boltz_iptm` are ranked.

## 4. Validate — the free full check

```bash
python binders_pipeline_env/generate_binders.py <all your flags> --dry_run
echo "exit=$?"
```

`--dry_run` now validates **everything cheap**: file exists, chain exists, site residues exist,
length sane, tool names valid, tool envs present, `n_designs` keys valid. It prints the exact
commands and exits 0. It does not touch the GPU.

**exit 0 here is the gate.** Anything non-zero: read stderr, fix, re-dry-run. Never skip to step 6
on a failed dry run, and never "try it and see" — seeing takes hours.

## 5. Confirm with the human

Above `--mode test`, **always confirm before launching.** Show:

- the exact command
- the tools and why
- the site, in the target's numbering, and that you verified it
- the cost: `test` ≈ 4 h, `standard` ≈ 37 h, `production` ≈ 5 days, on the single GPU
- anything the dry run warned about

This machine has **one GPU**. A production run monopolises it for days. That is the human's call,
not yours.

## 6. Execute and poll

```bash
cd <repo>            # the binderflow checkout
nohup python binders_pipeline_env/generate_binders.py <flags> \
      > run.out 2> run.err &
echo "pid=$!"
```

Poll **every few minutes at most** — nothing changes faster:

```bash
tail -5 run.out                     # progress
tail -20 run.err                    # problems only; empty is good
ls {out_dir}/status.json 2>/dev/null && echo "FINISHED"
```

`status.json` appears once, atomically, at the end. Its presence means the run is over.

To drop a stalled tool without losing the run: `touch {out_dir}/.skip_{tool}` (noticed within 15 s,
partial output collected).

## 7. Interpret

```bash
cat {out_dir}/status.json
```

| `ok` | `exit` | Meaning | Say what |
|---|---|---|---|
| `true` | 0 | Clean | Report top designs |
| `false` | 4 | Partial — some ranked, ≥1 tool failed | Report results **and** which tool failed and why (`failure_kinds`) |
| `false` | 3 | Nothing ranked | Do not report designs. Diagnose via `05_TROUBLESHOOT.md`. |

Then read `rankings.csv` (see `04_OUTPUTS.md`). Report `combined_score`, `boltz_iptm` or the native
iPTM, `rosetta_dG`, and the structure paths under `top_designs/`.

**Always check `status.json:site_residues_not_in_target` before reporting.** If it is non-empty,
part of the requested epitope was silently dropped and the results answer a different question than
the user asked — say so.

## 8. Visualise — and check your own work

Do not stop at the numbers. Render the top designs and look at them:

```bash
python binders_pipeline_env/generate_binders.py --render_designs \
    --results_dir {out_dir} --site "A:54,56,66,115" --top 6
```

Seconds, no GPU. Writes `{out_dir}/renders/rank{NNN}_{design_id}.png` — target grey, epitope
orange sticks, binder coloured per tool. **Read the PNGs.**

This catches what scores cannot:

- binder packed against the **wrong face** → the panel docked off-site; tighten
  `--min_site_interface_fraction` and re-rank (free)
- binder with **no real fold** → a bad design that scored well
- binder touching the epitope by **one residue on an edge** → not an interface

If the top designs look wrong, diagnose rather than reporting their scores. Then give the user the
images, the numbers, the PLIP contacts, and the `.pml`/`.cxc` paths so they can look themselves.

Full detail: `07_VISUALIZE.md`.

### Reporting honestly

These are **predictions, not measurements**. A high iPTM means the model is confident about a pose,
not that the protein binds. Never call a design "a binder". Say "predicted binder, requires
experimental validation", and give the numbers.

## Post-processing is free — use it

Anything that is re-scoring rather than re-designing costs nothing:

```bash
python binders_pipeline_env/rerank_binders.py --target ... --site ... \
    --results_dir {out_dir} --out_dir {new_dir} --rank_only \
    --max_refolding_rmsd 2.5 --no_cys --min_site_interface_fraction 0.3
```

So do not re-run a design campaign to change weights, apply quality filters, tighten the site
filter, or regenerate visualisations. Rerank instead. It also writes `status.json`.

## Installation requests

"Set up the pipeline on this machine" → `01_INSTALL.md`. Order is load-bearing: `boltz` host env
first, then the `bindcraft` v1 env (PyRosetta, needed for **every** tool's Rosetta scoring), then
design tools. Verify with `--capabilities` — a tool is installed when it appears in
`tools_available`, not when the download finished.

## What to refuse

- Launching above `test` mode without explicit confirmation
- Inventing site residues, or proceeding when numbering cannot be verified
- Peptide design — **no peptide pipeline exists** in this tree
- Any tool in `tools_unavailable`
- Reporting a design as validated, or as "a binder", on the strength of a score
