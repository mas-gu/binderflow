# Visualising results — the last step, and the one an agent can actually do

A finished run writes structures and two viewer scenes. Those assume a human at a GUI. On this
headless box an agent that stops at the numbers can neither check its own work nor show the user
anything.

**Render the designs. Then look at them.**

```bash
cd <repo>            # the binderflow checkout
python binders_pipeline_env/generate_binders.py --render_designs \
    --results_dir ./runs/pdl1_test/ \
    --site "A:54,56,66,115" \
    --top 6
```

No GPU, a few seconds per design. Writes `{results_dir}/renders/rank{NNN}_{design_id}.png` and
prints a table of what it made. Exit 0 = rendered, 1 = nothing to render (almost always: no design
was ranked — check `status.json`).

Flags: `--top N` (default 6), `--out_dir` (default `{results_dir}/renders/`), `--json` for a
parseable dict, `--site` to highlight the epitope (omit it and you get no orange).

## How the binder chain is identified

A render is only right if the correct chain was coloured as the binder. Three tiers, best first —
`--json` reports which one was used in `binder_chain_by`:

| Tier | Method | When it is used |
|---|---|---|
| 1 | **Exact sequence match** against `binder_sequence` in `rankings.csv` | Normal case — definitive |
| 2 | **Least identical to the target** (`pick_binder_chain_by_identity`) | No designed sequence recorded. Length-agnostic. |
| 3 | Shortest chain | Last resort. **Logs a warning on stderr** — it is wrong whenever a binder is longer than its target. |

Tier 2 matters: tier 3 alone is the same heuristic that once made `collect_boltzgen` score the
target against itself for any binder longer than its target. If you see
`shortest chain (UNRELIABLE)` in `binder_chain_by`, or a "guessed by length" warning on stderr,
treat the colours as unverified.

## What you are looking at

Same convention in every artefact this project produces:

| Element | Colour |
|---|---|
| Target | grey cartoon |
| Epitope (your `--site`) | **orange**, drawn as sticks |
| Binder | per-tool colour (bindcraft2 pink, boltzgen pale green, rfdiffusion salmon, …) |

**Read the PNG and judge it.** This is a real check, not decoration:

- **Is the binder touching the orange?** If it is packed against a different face, the design docked
  off-site — and if that is true of most of the top designs, the ranking is not telling you what you
  think. Tighten `--min_site_interface_fraction` and re-rank (free).
- **Does the binder look like a protein?** A compact helical bundle or a sheet is plausible. Loose
  spaghetti with no secondary structure is a bad design that scored well.
- **Is the interface a real surface contact,** or is the binder perched on an edge with one or two
  residues touching?

If the top-ranked designs all look wrong, say so and diagnose — do not report scores from a panel
you can see is bad.

## The scene files — for the human, not for you

```bash
pymol    runs/pdl1_test/top_designs/view_by_chain.pml
chimerax runs/pdl1_test/top_designs/view_by_chain.cxc
```

| File | Shows |
|---|---|
| `view_by_chain.{pml,cxc}` | all top designs, binders coloured **by tool** |
| `view_by_iptm.{pml,cxc}` | same scene, binders on a **confidence ramp** (red→green) |
| `{tool}/view_{tool}.{pml,cxc}` | that tool's top 20 only |

Both viewers get the same scene. The `.cxc` are command scripts, never baked `.cxs` sessions —
this box has no OpenGL, so a saved session would be unopenable here.

Give the human these paths. They are the interactive version of what you just looked at.

## The dashboard

`dashboard.png` — score distributions, per-tool comparison, pLDDT vs iPTM. One image, read it
directly. Good for "how did the run go overall" as opposed to "is design 1 any good".

## PLIP — what the interface is actually made of

With `--plip_top N`, each of the top N gets `plip_analysis/rank{NN}_{design_id}/`:

| File | Use |
|---|---|
| `*_report.txt` | **read this** — hydrogen bonds, salt bridges, hydrophobic contacts, π-stacking, by residue |
| `*_report.xml` | same, parseable, with atom serials and coordinates |
| `*.pse` | PyMOL session for the human |

A design with a handful of hydrogen bonds and a salt bridge across the epitope is more convincing
than one held on by a single hydrophobic patch. This is the closest thing to a mechanistic answer
the pipeline produces — quote it when you report.

Empty or missing means `--plip_top 0`, or PLIP failed — check stderr.

## Regenerating visuals is free

Changed `--top_n`, or want PLIP on more designs? Do **not** re-run the design campaign:

```bash
python binders_pipeline_env/rerank_binders.py --target <t> --site <s> \
    --results_dir <run> --out_dir <new> --rank_only --top_n 20 --plip_top 20
```

`--rank_only` regenerates the full visual set — `top_designs/`, both scene pairs, `dashboard.png`,
`plip_analysis/` — with no GPU. Then `--render_designs` on the new directory.

## Reporting

Give the user, in this order:

1. **The images** — what the top designs look like, and your own read of whether they engage the site
2. The numbers: `combined_score`, iPTM, `rosetta_dG`, `refolding_rmsd` (see `04_OUTPUTS.md`)
3. The PLIP contacts for the best one or two
4. Paths to `top_designs/` and the `.pml`/`.cxc` scenes so they can look themselves

And state plainly that these are **computational predictions requiring experimental validation**.
A confident-looking render is a model's opinion about a pose, not evidence of binding.
