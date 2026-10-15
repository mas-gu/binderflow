# Reading the outputs

## What a finished run contains

```
{out_dir}/
├── status.json          ← READ THIS FIRST. Machine-readable outcome.
├── rankings.csv           every design, every score. rank is blank for unranked designs.
├── dashboard.png          summary plots
├── run_params.txt         the command as INTENT at start — not outcome, and incomplete
├── {tool}/                raw per-tool output (rfdiffusion/, bindcraft2/, ...)
├── validation/
│   ├── boltz_monomer/     Stage 1: refolding pre-filter
│   ├── boltz/             Stage 2: complex prediction
│   └── rosetta/           Stage 3: interface energetics (+ errors.log)
├── top_designs/
│   ├── rank001..rankNNN_{design_id}.{cif|pdb}   top --top_n overall (default 50)
│   ├── view_by_chain.pml / .cxc                 PyMOL + ChimeraX scenes
│   ├── view_by_iptm.pml  / .cxc
│   └── {tool}/            top 20 per tool + its own view files
└── plip_analysis/
    └── rank{NN}_{design_id}/   H-bonds, salt bridges, hydrophobic, pi-stacking + .pse
```

## status.json — the file to trust

Written once, atomically, at the end of the run.

```json
{
  "schema": 1,
  "script": "generate_binders.py",
  "finished_utc": "2026-09-28T14:25:51Z",
  "ok": true,
  "target": "/.../PDL1_clean.pdb",
  "target_chain": "A",
  "site": "A:54,56,500",
  "site_residues_not_in_target": [500],
  "tools_requested": ["bindcraft2", "rfdiffusion"],
  "tools_failed": [],
  "failure_kinds": {},
  "tool_results": {"bindcraft2": "OK (1 designs)", "rfdiffusion": "OK (1 designs)"},
  "per_tool": {"bindcraft2": {"designs": 1, "ranked": 0}},
  "designs_total": 2,
  "designs_ranked": 1,
  "reprediction": false,
  "mode": "test",
  "top_score": 0.58,
  "top_design_id": "rfdiffusion_0000",
  "artifacts": {"rankings_csv": "rankings.csv", "top_designs": "top_designs", "dashboard": "dashboard.png"}
}
```

| Field | How to use it |
|---|---|
| `ok` | `true` only if every requested tool succeeded **and** something was ranked |
| `tools_failed` / `failure_kinds` | Which tools died and **why**, classified — see `05_TROUBLESHOOT.md` |
| `designs_ranked` vs `designs_total` | A large gap means filters or missing validation removed most designs |
| `per_tool[tool].ranked` | `0` for a tool that produced designs means none survived scoring/filters |
| `site_residues_not_in_target` | **Non-empty = your `--site` was partly wrong** and those residues were dropped |
| `top_score` / `top_design_id` | The best design, matching `rank 1` in `rankings.csv` |

`run_params.txt` is written at *start* and records intent. It omits ~14 flags. Do not use it to
judge whether a run succeeded.

## rankings.csv

One row per design. Key columns:

| Column | Meaning |
|---|---|
| `rank` | Position. **Blank = not ranked** (filtered out, or no interface score) |
| `design_id` | `{tool}_{NNNN}` — the join key for everything else, incl. cached validation |
| `tool` | Which tool made it. **Empty in every run written before 2026-09-28** — it was declared but never populated. Recover it from the `design_id` prefix. |
| `combined_score` | The ranking score. Blank when unranked |
| `binder_sequence` | The designed sequence |
| `iptm_source` | Where the interface score came from: `boltz`, or `native_{tool}` |
| `boltz_iptm` | Interface pTM from Boltz-2 re-prediction |
| `boltz_binder_plddt` | Binder confidence, **0-100** |
| `rosetta_dG` | Interface binding energy — more negative is better |
| `site_interface_fraction` | Fraction of the binder's interface that is on your `--site` |
| `site_geometry_status` | `measured` / `no_structure` / `parse_failed` / `dry_run`. **Read this before trusting any `site_*` value** — anything other than `measured` means the design was never checked against the site. |
| `binder_rg_ratio` | Rg / (2.2·N^0.38). ~1.0 globular; >1.25 extended or single-helix. |
| `binder_rg` | Radius of gyration of the binder CA trace (Å) |
| `refolding_rmsd` | Does the binder fold the same alone as in complex? Lower is better |
| `{bc2_,bg_,px_,pc_,bc_}*` | Per-tool native metrics |

### The scoring formula

```
combined_score = w_plddt × (pLDDT/100) + w_iptm × iPTM + w_dg × dG_norm − site_pae_penalty
```

with `--score_weights w_plddt,w_iptm,w_dg` (default `0.4,0.5,0.1`; recommended `0.3,0.6,0.1`).
`dG_norm` maps Rosetta dG from [-40, 0] onto [0, 1]. If Rosetta produced no dG, the remaining terms
are **renormalised** so the design is not handicapped (column `_score_renormalized`).

`iPTM` comes from `boltz_iptm` when `--reprediction` is on, otherwise from the tool's native score
(`bc2_i_ptm`, `bg_design_to_target_iptm`, `px_iptm`, `pc_iptm`, `bc_i_ptm`).

### Scale traps

- `boltz_binder_plddt` is **0-100**. BindCraft2's `bc2_plddt` is **0-1**. Do not compare directly.
- `bc2_binder_helix_fraction` is a **fraction (0-1)**; v1's `bc_binder_helix_pct` is a **percentage**.
- `boltz_iplddt` / `boltz_complex_plddt` are 0-100 — **but in rerank-written CSVs from before
  2026-09-28 they are 0-1.** Check the magnitude before comparing across old runs.

### Numbering traps — read before feeding an output structure back into anything

**Residue numbers are not preserved through the pipeline.** A `--site` that was correct for the
target file will match *nothing* in the run's own output structures, and most tools degrade
silently when that happens rather than failing.

| Frame | Numbering | Where it appears |
|---|---|---|
| Target file | whatever the PDB says | `--site`, `--preview_site`, `crossreactivity_filter.py` |
| Cropped target | target numbering, preserved | a domain crop you made yourself |
| **Boltz-2 output** | **renumbered from 1** | `validation/boltz/**/*.cif`, merged `top_designs/*.cif` |
| Boltz pocket constraint | 1-based **token index**, not residue number | `constraints: pocket: contacts` |

So for a target cropped to start at residue N, a site residue `R` becomes `R - (N-1)` in every
predicted complex and in every pocket constraint. Example: a domain III crop starting at 311 turns
site residue 384 into 74.

Two checks worth making every time:

```bash
# what numbering does this structure actually use?
python -c "import gemmi,sys; st=gemmi.read_structure(sys.argv[1]); st.setup_entities()
print([(c.name, c[0].seqid.num, c[len(c)-1].seqid.num) for c in st[0]])" <file>
```

- `mpnn_redesign.py` now **fails hard** when `--site` matches zero residues, and takes
  `--site_offset` to convert (for a crop starting at N, pass `-(N-1)`). It used to widen to the
  whole interface and merely warn — which changed what got designed while every line still read
  like success.
- `ortholog_cofold.py` takes `--target_start` and refuses to run if a site residue maps outside
  the sequence.

**Related: file extensions.** `top_designs/` is `.cif` for most tools. A `*.pdb` glob over it
matches only the copied target and reports success with zero designs processed. Glob `rank*.cif`,
or handle both.

## Structures

`top_designs/` holds the top `--top_n` (default 50) by `combined_score`, plus the top 20 per tool in
subdirectories. Files are `.cif` or `.pdb` depending on the producing tool.

Viewing — **as an agent, render them and look**:

```bash
python binders_pipeline_env/generate_binders.py --render_designs \
    --results_dir <run> --site "<your site>" --top 6      # -> renders/*.png
```

For the human, the interactive scenes:

```bash
pymol    top_designs/view_by_chain.pml     # target grey, epitope orange, binders by tool colour
chimerax top_designs/view_by_chain.cxc     # same scene
```

See `07_VISUALIZE.md`.

`view_by_iptm` colours binders on a confidence ramp instead of by tool. Both viewers get the same
scene. The `.cxc` files are command scripts, not baked sessions — that is deliberate, this box has
no OpenGL.

## PLIP interaction analysis

`plip_analysis/rank{NN}_{design_id}/` contains a text report, an XML report with fully resolved
interaction records (atom serials and coordinates), and a PyMOL `.pse` session. Categories:
hydrophobic contacts, hydrogen bonds, salt bridges, pi-stacking, water bridges, halogen bonds.

Absent or empty means `--plip_top 0` was set, or PLIP failed — check stderr.

## Judging whether designs are actually good

Confidence metrics measure **prediction confidence, not affinity**. A high iPTM is not evidence of
binding. Rough orientation:

| Signal | Encouraging |
|---|---|
| `boltz_iptm` | > 0.7 |
| `boltz_binder_plddt` | > 80 |
| `rosetta_dG` | < -30 |
| `refolding_rmsd` | < 2.5 Å |
| `site_interface_fraction` | ≥ your filter threshold |

Report these as computational predictions requiring experimental validation. Do not describe a
design as "a binder" on the strength of a score.
