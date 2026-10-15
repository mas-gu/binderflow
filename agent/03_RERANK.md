# Re-ranking an existing run

`binders_pipeline_env/rerank_binders.py` re-scores designs that already exist. Use it instead of
re-designing whenever the structures are fine and only the **scoring or filtering** should change.

## Two modes, very different cost

| Mode | Flag | GPU | Time | What it does |
|---|---|---|---|---|
| **Rank-only** | `--rank_only` | none | seconds–minutes | Loads cached scores, re-applies filters and weights |
| **Re-validation** | `--reprediction` | yes | hours | Ignores cache; fresh Boltz-2 monomer + complex + Rosetta |

**Default to `--rank_only`.** It answers most questions for free. Reach for `--reprediction` only
when you need new structural predictions, not new arithmetic.

## Rank-only: change weights or filters

```bash
cd <repo>            # the binderflow checkout
conda activate boltz

python binders_pipeline_env/rerank_binders.py \
    --target ./targets/PDL1.pdb \
    --site "A:54,56,66,115" \
    --results_dir ./runs/pdl1_test/ \
    --out_dir ./runs/pdl1_reranked/ \
    --rank_only \
    --score_weights 0.3,0.6,0.1 \
    --min_site_interface_fraction 0.25 \
    --plip_top 10
```

Drop `--results_dir` and point `--out_dir` at the original run to re-rank **in place** (it
overwrites `rankings.csv`, so keep a copy if you want the old ordering).

### Merging several runs

```bash
--results_dir ./out_helix,./out_balanced,./out_beta
```

Designs are deduplicated by sequence, and ids are prefixed with the source directory
(`out_helix__rfdiffusion3_0006`). This is how you compare `--ss_bias` variants on one ranking.

Each design's cached Boltz/Rosetta scores and structures are read **only from its own source
directory**. That matters because independent runs reuse the same bare design ids (every run has a
`rfdiffusion_0000`).

> **Before 2026-09-29 this was broken.** Every cache loader ran over the full merged list once per
> directory and matched on the *unprefixed* id, so the second and later directories loaded the
> first directory's scores. Scores were first-wins while `complex_cif` was last-wins, so a merged
> design could carry one run's iPTM and a different run's structure — and the same applied to the
> monomer CIF that drives SS composition and refolding RMSD. **Any merged ranking produced before
> that date should be re-run.** Single-directory reranks were never affected.

## Re-validation: fresh predictions

```bash
CUDA_VISIBLE_DEVICES=0 python binders_pipeline_env/rerank_binders.py \
    --target ./targets/PDL1.pdb \
    --site "A:54,56,66,115" \
    --results_dir ./runs/pdl1_test/ \
    --out_dir ./runs/pdl1_revalidated/ \
    --reprediction \
    --plip_top 10
```

Re-running into the **same** `--out_dir` reuses that directory's own cache, so a re-validation is
resumable in a way a design run is not.

## Filters available here that `generate_binders.py` does not have

`rerank_binders.py` is the richer filtering tool:

| Flag | Use |
|---|---|
| `--min_site_cos` | Surface-normal alignment, 0.3-0.5 — is the binder *facing* the site |
| `--skip_esmfold` / `--skip_boltz` / `--skip_rosetta` | Skip individual validation stages |
| `--rerun_rosetta` | Redo Rosetta only, reusing Boltz results |
| `--filter_interface_pae` / `--filter_site_pae` | Hard PAE cutoffs |

Everything in `02_RUN_BINDERS.md` §Filters applies here too, with the same advice: quality filters
stack, geometric filters do not — pick one.

## Retroactively fixing an unfiltered panel

If a run before 2026-09-28 passed `--min_site_interface_fraction` to `generate_binders.py`, that
filter **did nothing**. Apply it now, for free:

```bash
python binders_pipeline_env/rerank_binders.py \
    --target <same target> --site "<same site>" \
    --results_dir <the old run> --out_dir <new dir> \
    --rank_only --min_site_interface_fraction 0.25
```

Compare `designs_ranked` before and after to see how many were docking off-site all along.

## What it regenerates

The full output set — `rankings.csv`, `top_designs/` with both PyMOL and ChimeraX scenes,
`dashboard.png`, `plip_analysis/`. So `--rank_only` is also the cheapest way to regenerate
visualisations after changing `--top_n` or `--plip_top`.

`rerank_binders.py` writes `status.json` too, same schema as `generate_binders.py` plus
`rank_only`, `results_dirs` and `designs_collected`. Its exit codes are **0** (clean) and **3**
(nothing could be ranked) — there is no 4, because rerank runs no design tools.

## Cautions

- **`--rank_only --reprediction` together loads almost nothing.** They are contradictory; pick one.
- **Re-ranking with a different `--site`** compares new geometry against old docking. Legitimate for
  asking "did these designs happen to hit a different epitope", misleading otherwise.
- Designs lacking `boltz_iptm` get a blank rank and stay in `rankings.csv`. If a lot vanish from the
  ranking, check how many were validated rather than assuming a filter removed them.
