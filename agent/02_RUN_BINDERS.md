# Running a binder design campaign

Entry point: `binders_pipeline_env/generate_binders.py`. Host env: `boltz`.

## The shortest correct invocation

```bash
cd <repo>            # the binderflow checkout
conda activate boltz

python binders_pipeline_env/generate_binders.py \
    --target /path/to/target.pdb \
    --site "A:54,56,66,115" \
    --length 60-80 \
    --tools bindcraft2,boltzgen \
    --mode test \
    --out_dir ./runs/my_campaign/ \
    --dry_run
```

`--target --site --length --tools --out_dir` are all **required**. Run with `--dry_run` first,
read what it prints, then re-run without it.

## The five required flags

### `--target` — the protein you want to bind
A `.pdb` file. Some tools accept `.cif` (bindcraft2, boltzgen, pxdesign) but **`rfdiffusion` and
`bindcraft` are PDB-only** and the run exits 1 up front if you mix them with a CIF target.

The target is automatically sanitised into `{out_dir}/{stem}_clean.pdb` — HETATM, CONECT, ANISOU
stripped and insertion codes cleared. Design tools see the cleaned copy.

### `--site` — the epitope you want the binder to hit
Preferred form: `"A:8-30,120-123"` (chain, then residues and ranges).
Legacy form `"A10 A15 A22"` also parses. **Single-letter chain IDs only** — multi-character mmCIF
chain names cannot be expressed.

> **The site's numbering is only valid for the target file.** Predicted complexes produced by
> the run are renumbered from 1, so the same `--site` will match nothing downstream. See
> `04_OUTPUTS.md` → "Numbering traps" before feeding outputs into another tool.

**Choosing the site — do this before running.** The numbering must match the target file, not
UniProt, not a paper. Check it:

```bash
python -c "
import sys; sys.path.insert(0,'binders_pipeline_env')
from generate_binders import get_chain_info
ci = get_chain_info('target.pdb')
for c, i in ci.items():
    print(c, i['length'], 'residues', min(i['residues']), '-', max(i['residues']))"
```

The pipeline warns before any GPU is spent if site residues are absent, and **exits 1 if none
exist**. But a site that exists can still be the wrong site, so confirm it visually:

```bash
python binders_pipeline_env/generate_binders.py --preview_site \
    --target TARGET.pdb --site "A:54,56,66,115" --out_dir /tmp/sitecheck
```

Prints a per-residue exposure map and a verdict; exit 1 if the residues are buried or scattered
across several faces. Writes `site_preview.png` (ray-traced, look at it), plus `.pml` and `.cxc`
scenes. See `06_AGENT_WORKFLOW.md` §2.

### `--length` — binder length range
`60-80` for a range, or a single number like `70`. Typical miniprotein binders are 55-100 aa.
Scaffolded BindCraft2 modalities (VHH, scFv, Fab, ARP) carry their own length and ignore this.

### `--tools` — which design tools to run
Comma-separated, from exactly this list:
`rfdiffusion, rfdiffusion3, boltzgen, bindcraft, bindcraft2, pxdesign, proteina, proteina_complexa`

A sensible default for a first pass is `bindcraft2,boltzgen,pxdesign` — diverse methods, all
iPTM-native so they are cheaper without `--reprediction`. Adding all 8 maximises diversity but
multiplies runtime.

### `--out_dir` — where results go
Created if absent. **Re-using a directory silently merges with what is already there** and
overwrites `rankings.csv`. Use a fresh directory per campaign unless you intend a resume.

## Flags that change results, and when to use them

| Flag | Default | Use it when |
|---|---|---|
| `--mode test\|standard\|production` | `standard` | Always start `test`. See 00_CAPABILITIES for cost. |
| `--reprediction` | off | You want uniform cross-tool scoring: Boltz-2 re-predicts every design, including the iPTM-native ones. Costs a lot of compute. **With it on, only designs that get `boltz_iptm` are ranked.** |
| `--score_weights w_plddt,w_iptm,w_dg` | `0.4,0.5,0.1` | `0.3,0.6,0.1` is the project's recommendation — weights interface confidence higher. |
| `--ss_bias balanced\|helix\|beta` | `balanced` | `balanced` is ~80% helical in practice. Use `beta` for sheet-rich binders; expect lower scores (~0.82-0.89 vs ~0.92-0.94). |
| `--n_designs` | from `--mode` | Per-tool override, e.g. `"boltzgen=200,bindcraft2=10"`. A bare number applies to all. **Unknown tool names are silently ignored** — spell them exactly. |
| `--plip_top N` | 10 | Interaction analysis on the top N. `0` disables. |

## Filters — and which script accepts which

**The two scripts do not share a flag set.** This trips people up constantly, and older project
docs list rerank-only flags as if `generate_binders.py` accepted them. It does not — it exits 2.

**Quality filters are `rerank_binders.py`-ONLY:**
`--max_refolding_rmsd`, `--no_cys`, `--max_aa_fraction`, `--min_sc`, `--max_interface_ke`,
`--min_site_cos`

Design first, then apply them with a `--rank_only` rerank (free, no GPU — see `03_RERANK.md`).
The underlying measurements (`refolding_rmsd`, `netsolp_solubility`) are computed during the design
run as of 2026-09-29 — before that both columns were emitted but never populated by
`generate_binders.py`, so the first rerank had nothing to filter on:

| Flag | Recommended |
|---|---|
| `--max_refolding_rmsd` | 2.5 Å |
| `--no_cys` | on if expressing in E. coli |
| `--max_aa_fraction` | 0.3 |
| `--min_sc` | 0.45 for helical designs; **skip for beta** |
| `--max_interface_ke` | 0.25 |

**Also rerank-only:** `--rank_only`, `--results_dir`, `--rerun_rosetta`, `--skip_esmfold`,
`--skip_boltz`, `--skip_rosetta`.

**generate-only:** `--length`, `--mode`, `--n_designs`, `--device`, `--boltz_devices`,
`--force_site`, `--site_max_distance`, `--holo_ligands`, `--bindcraft_filters`,
`--bindcraft_advanced`, `--bindcraft2_*`.

Geometric site filters — accepted by **both** scripts. **Pick one, do not stack them**:

| Flag | Recommended |
|---|---|
| `--min_site_interface_fraction` (SIF) | **0.25-0.5, scaled to site size** — best standalone |
| `--max_site_centroid_dist` + `--centroid_atoms CA\|heavy` | 10-15 Å |
| `--max_site_dist` + `--min_site_fraction` | 7-8 Å; skip if the site has < 5 residues |
| `--require_site_geometry` | Drop designs whose geometry could not be measured at all |

**BoltzGen enforces your `--site` by default** (since 2026-09-29). Its own `filter_bindingsite`
ships as `false`, so before that it routinely spent its whole budget off-epitope — in one 50-design
run the six requested residues were contacted by 15/12/11/1/0/0 of 50. Expect **fewer but
better-targeted** BoltzGen designs now; `--no_boltzgen_site_filter` restores the old behaviour. If
BoltzGen yield collapses on a target, that is a signal it cannot find your epitope, not a bug.

Shape filter — independent of the site, safe to combine with one of the above:

| Flag | Recommended |
|---|---|
| `--max_rg_ratio` | **1.25** — rejects extended / single-helix binders |

`binder_rg_ratio` = Rg / (2.2·N^0.38), the globular expectation for that length. ~1.0 is compact.
It is **always recorded**, filtering is opt-in. This exists because refolding RMSD structurally
cannot catch a lone helix — a lone helix refolds to a lone helix and scores a near-perfect RMSD.
In `pdia6_validation_001` the #2 design by score was exactly that (ratio 1.40; six true single
helices sat at 2.49-2.81).

**SIF by site size** — these are **rerank** values, where re-filtering is free:
8+ residues → 0.5; 4-5 → 0.25-0.30; 3 → 0.20-0.25.

**In the design run use 0.25 regardless of site size.** Over-filtering during design throws away
structures you cannot get back without another full run; over-filtering during rerank costs
seconds. See `06_AGENT_WORKFLOW.md` §3.

**All filters pass NaN through** — a design with no measurement is not penalised. Every filter logs
`removed N/M`, and says explicitly when the metric was missing, so "matched nothing" is
distinguishable from "had no data". Read those lines.

**Site geometry is measured whenever you pass `--site`**, with or without a filter flag. Filtering
is separate and opt-in. (Before 2026-09-29 the measurement itself was gated on a filter flag, so
the `site_*` columns were blank by default and "never measured" looked identical to "never asked
for".)

**Check `site_geometry_status` before you trust a SIF number.** It is `measured` when the geometry
was really computed, and `no_structure` / `parse_failed` / `dry_run` otherwise. Designs that are
not `measured` carry `site_geometric_pass=None` — they are **not** known to engage the site, and a
NaN SIF for them means "never checked", not "checked and fine". The run log prints a
`WARNING: N/M designs were NOT geometrically measured` line whenever this happens.

> **Historical note:** before 2026-09-28, `--min_site_interface_fraction` and
> `--max_site_centroid_dist` were accepted by `generate_binders.py` and **filtered nothing** — the
> column was written but no design was dropped. Runs from before that date that used these flags
> have unfiltered panels. Re-rank them to apply the filter retroactively.
>
> **Before 2026-09-29** the site geometry itself was wrong for any tool whose mmCIF column order
> was not Boltz-2's. BoltzGen, PXDesign and BindCraft2 designs were never measured yet were marked
> as *passing*, and PDB-based tools were matched against the wrong residue numbering. In
> `pdia6_validation_001` that was 58% of the ranked panel, and it discarded the single
> best-engaging tool (`proteina_complexa`) while promoting largely off-site BoltzGen designs into
> the top 50. **Re-rank any earlier run before drawing conclusions from its SIF column.**

## BindCraft2 flags

```bash
--bindcraft2_modality binder        # see 00_CAPABILITIES for the full list
--bindcraft2_filters no_filters     # no_filters|relaxed|default|kras_relaxed|peptide|peptide_relaxed
--bindcraft2_max_trajectories 500   # attempt ceiling; default is 50 × requested designs
```

BC2 leaves `max_trajectories` unset by default, meaning it designs until it reaches the target count
with **no upper bound**. The wrapper always sets a ceiling; raise it if BC2 is struggling on a hard
target rather than letting it run unbounded.

## GPU pinning

```bash
CUDA_VISIBLE_DEVICES=0 python binders_pipeline_env/generate_binders.py ...
```

**Do not also pass `--device cuda:0`** — it overrides the environment variable. And do not combine
`--device` with `--boltz_devices N`: the latter writes `CUDA_VISIBLE_DEVICES` into the Boltz
subprocess at highest precedence and will escape your pin.

## A realistic full run

```bash
cd <repo>            # the binderflow checkout
conda activate boltz

CUDA_VISIBLE_DEVICES=0 python binders_pipeline_env/generate_binders.py \
    --target targets/PDL1.pdb \
    --site "A:54,56,66,115" \
    --length 60-80 \
    --tools bindcraft2,boltzgen,pxdesign \
    --mode test \
    --score_weights 0.3,0.6,0.1 \
    --min_site_interface_fraction 0.25 \
    --plip_top 10 \
    --out_dir ./runs/pdl1_test/ \   # ./runs/ is created for you
    > ./runs/pdl1_test_stdout.log 2> ./runs/pdl1_test_stderr.log
echo "exit=$?"
```

Then:

```bash
cat ./runs/pdl1_test/status.json      # did it work, and what failed
cat ./runs/pdl1_test_stderr.log       # every warning and failure, nothing else
head -3 ./runs/pdl1_test/rankings.csv
```

## Long runs

A `test` run is ~4 hours and `production` ~5 days, so run detached and poll:

```bash
nohup python binders_pipeline_env/generate_binders.py ... > run.out 2> run.err &
echo $!   # keep the PID
```

Poll `status.json` for completion (it is written once, atomically, at the end) and `run.err` for
trouble. **Do not poll more often than every few minutes** — nothing changes faster than that.

To skip a tool that is stalling, without killing the run:
`touch {out_dir}/.skip_{toolname}` — the pipeline notices within 15 s, terminates that tool, and
collects whatever partial output exists.

## After it finishes

Read `04_OUTPUTS.md`. If it failed, read `05_TROUBLESHOOT.md`.
