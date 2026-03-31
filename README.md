# binderflow

**Author:** Guillaume Mas

A unified pipeline for de novo protein binder design using six complementary generative tools, with automated validation, geometric site filtering, and interface analysis.

<img width="3999" height="1158" alt="pipeline_overview" src="https://github.com/user-attachments/assets/0db27013-e165-425d-94dc-5e74c1b94dca" />

## Overview

Given a target protein PDB and a binding site, the pipeline:

1. **Generates** binder candidates using up to 6 design tools in parallel
2. **Validates** all designs with ESMFold (fast fold quality filter) and Boltz-2 (uniform cross-tool scoring with site pocket constraint)
3. **Scores** interfaces with Rosetta
4. **Ranks** all designs by combined score (pLDDT + iPTM + dG)
5. **Filters** off-site binders and designs predicted to poorly perform
6. **Outputs** ranked structures, PyMOL scripts, dashboard plots, and PLIP reports

## Design Tools

| Tool | Type | What it generates | Reference |
|------|------|-------------------|-----------|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Backbone diffusion | Backbone + LigandMPNN sequences | Watson et al., Nature 2023 |
| [BoltzGen](https://github.com/HannesStark/boltzgen) | Full-atom diffusion | Full-atom binder structures | Stark et al., 2025 |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | AF2-guided optimization | Iterative sequence design | Pacesa et al., 2024 |
| [PXDesign](https://github.com/bytedance/PXDesign) | DiT diffusion + AF2-IG | Backbone + sequence + validation | ByteDance, 2024 |
| [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina/) | Flow-based backbone | Unconditional backbones + ProteinMPNN | NVIDIA, 2024 |
| [Proteina Complexa](https://github.com/NVIDIA-Digital-Bio/proteina-complexa) | Flow-based full-atom | Target-conditioned binder design | NVIDIA, ICLR 2026 |

## Quick Start

```bash
# Activate host environment
conda activate boltz

# Test run (6 tools, ~4h, single GPU)
# Use CUDA_VISIBLE_DEVICES to pin GPU (do NOT use --device with CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=0 python generate_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --length 60-80 \
    --tools rfdiffusion,boltzgen,bindcraft,pxdesign,proteina,proteina_complexa \
    --mode test \
    --ss_bias balanced \
    --max_site_dist 8.0 \
    --reprediction \
    --plip_top 10 \
    --out_dir ./output/
```

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04)
- Python >= 3.7
- NVIDIA GPU with >= 20 GB VRAM (RTX 3090, A100, etc.)
- Conda (Miniconda or Anaconda)
- ~200 GB disk space (conda envs + weights)

### 1. Host Environment (boltz)

The pipeline runs from the `boltz` conda environment, which also handles Boltz-2 validation.

```bash
conda create -n boltz python=3.12
conda activate boltz
pip install boltz numpy matplotlib gemmi plip
```

### 2. Design Tools

Each tool runs in its own isolated conda environment. Install each tool following its official instructions:

| Tool | Conda env name | Install guide |
|------|---------------|---------------|
| RFdiffusion | `rfdiffusion` | [GitHub](https://github.com/RosettaCommons/RFdiffusion) |
| LigandMPNN | `mpnn` | [GitHub](https://github.com/dauparas/LigandMPNN) |
| BoltzGen | `boltzgen` (prefix env) | [GitHub](https://github.com/HannesStark/boltzgen) |
| BindCraft | `BindCraft` (capital B, C) | [GitHub](https://github.com/martinpacesa/BindCraft) |
| PXDesign | `pxdesign` | [GitHub](https://github.com/bytedance/PXDesign) |
| Proteina | `proteina_env` (prefix env) | [GitHub](https://github.com/NVIDIA-Digital-Bio/proteina/) |
| Proteina Complexa | UV venv (not conda) | [GitHub](https://github.com/NVIDIA-Digital-Bio/proteina-complexa) |
| ESMFold | `esmfold` | [HuggingFace](https://huggingface.co/facebook/esmfold_v1) |

### 3. Optional Dependencies

| Tool | Conda env | Purpose |
|------|-----------|---------|
| PyRosetta | (in boltz env) | Interface energy scoring (dG) + shape complementarity (Sc). Optional but recommended |
| PLIP | (in boltz env) | Interface contact analysis. `pip install plip` |

### 4. Path Configuration

Set environment variables to point to your tool installations:

```bash
# Required: base directories
export BINDER_SOFTWARE_DIR="$HOME/data/software"
export BINDER_WEIGHTS_DIR="$HOME/data/weights"

# Optional: override individual tool paths
export RFDIFFUSION_DIR="$HOME/data/software/RFdiffusion"
export BINDCRAFT_DIR="$HOME/data/software/BindCraft"
export BOLTZGEN_BIN="$HOME/data/software/envs/boltzgen/bin/boltzgen"
# ... etc (see generate_binders.py constants section for all variables)
```

Default layout expected under `BINDER_SOFTWARE_DIR`:
```
$BINDER_SOFTWARE_DIR/
├── RFdiffusion/
├── LigandMPNN/
├── BindCraft/
├── PXDesign/
├── Proteina/
├── Proteina-Complexa/
│   ├── .venv/           # UV virtual environment
│   └── ckpts/           # Proteina Complexa weights
├── NetSolP-1.0/         # Solubility prediction (ONNX models in PredictionServer/models/)
└── envs/
    ├── boltzgen/        # BoltzGen prefix conda env
    └── proteina_env/    # Proteina prefix conda env
```

Default layout expected under `BINDER_WEIGHTS_DIR`:
```
$BINDER_WEIGHTS_DIR/
├── rfdiffusion/
│   └── Complex_base_ckpt.pt
├── ligandmpnn/
│   └── proteinmpnn_v_48_020.pt
└── proteina/
    └── proteina_v1.1_DFS_200M_tri.ckpt
```

## Usage

### Design Modes

| Mode | Total designs | Full pipeline (1 GPU) | Revalidation only |
|------|--------------|----------------------|-------------------|
| test | 135 | ~4h | ~30 min |
| standard | 1,310 | ~37h | ~10.5h |
| production | 12,020 | ~5 days | ~36h |

Boltz-2 MSA is pre-computed once (~60s). Revalidation skips design generation + ESMFold.

Per-tool breakdown:

| Tool | test | standard | production |
|------|------|----------|------------|
| RFdiffusion | 20 | 200 | 500 |
| BoltzGen | 50 | 500 | 10,000 |
| BindCraft | 5 | 10 | 20 |
| PXDesign | 20 | 200 | 500 |
| Proteina | 20 | 200 | 500 |
| Proteina Complexa | 20 | 200 | 500 |

### Key Flags

| Flag | Description | Recommended |
|------|-------------|-------------|
| `--site "A:11-17,119-124"` | Binding site (chain:residues) | Required |
| `--ss_bias balanced\|beta\|helix` | Secondary structure bias | `balanced` or `beta` |
| `--reprediction` | Boltz-2 validates ALL designs | Yes for quality |
| `--score_weights 0.4,0.5,0.1` | pLDDT, iPTM, dG weights (default). Recommended: `0.3,0.6,0.1` for higher iPTM weight | Higher iPTM = binding quality |
| `--plip_top 10` | PLIP analysis on top N designs | 10 |
| `--boltz_devices 3` | Multi-GPU for Boltz-2 batch validation | All free GPUs |

### Re-ranking Existing Runs

```bash
# Re-rank with quality + geometric filters (no re-validation, fast)
python rerank_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --results_dir ./output/ \
    --out_dir ./output_reranked/ \
    --rank_only \
    --min_site_interface_fraction 0.5 \
    --no_cys \
    --max_aa_fraction 0.3 \
    --min_sc 0.45 \
    --ss_bias helix \
    --plip_top 10

# Revalidate: batch Boltz-2 on ALL designs (finds hidden good designs)
CUDA_VISIBLE_DEVICES=0 python rerank_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --results_dir ./output/ \
    --out_dir ./output_revalidated/ \
    --reprediction \
    --min_site_interface_fraction 0.5 \
    --plip_top 10

# Merge multiple runs
python rerank_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --results_dir ./out_balanced,./out_beta \
    --out_dir ./merged/ \
    --rank_only

# Re-run Rosetta scoring only (adds SAP without re-running ESMFold/Boltz-2)
python rerank_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --results_dir ./output/ \
    --out_dir ./output_reranked/ \
    --rerun_rosetta \
    --min_site_interface_fraction 0.5
```

### Running Specific Tools

```bash
# Only RFdiffusion + BoltzGen (fastest combination)
CUDA_VISIBLE_DEVICES=0 python generate_binders.py \
    --target target.pdb \
    --site "A:11-17" \
    --tools rfdiffusion,boltzgen \
    --mode test \
    --out_dir ./output/
```

## Filters

Quality filters and geometric filters listed below are available in `rerank_binders.py`. The `generate_binders.py` pipeline applies ESMFold pre-filter, Boltz-2 validation, geometric site metrics, Rosetta scoring, and SS/KE composition automatically.

Filters are applied in this order during ranking. Quality filters run first, geometric site filters last.

### Quality Filters

| Flag | Default | Description |
|------|---------|-------------|
| `--max_refolding_rmsd` | disabled | Max CA RMSD between ESMFold (binder alone) and Boltz-2 (binder in complex). Removes target-dependent binders that don't fold independently. Default: disabled. Typical: 2.5 A |
| `--no_cys` | off | Exclude designs containing cysteine. Unpaired Cys cause aggregation in E.coli expression |
| `--max_aa_fraction` | disabled | Max fraction of any single amino acid. Catches poly-Ala/poly-Glu hallucinations. Typical: 0.3 |
| `--min_sc` | disabled | Min Rosetta shape complementarity (0-1). Natural interfaces ~0.65, designed aim >0.55. **Skip for beta-biased runs** (beta sheets have intrinsically lower Sc on predicted structures) |
| `--max_interface_ke` | disabled | Max K+E fraction at binder-target interface. Typical: 0.25 |
| `--ss_bias` | balanced | SS composition filter. `helix` removes sheet >0.3, `beta` removes helix >0.4 |

### Confidence Filters

| Flag | Default | Description |
|------|---------|-------------|
| `--filter_site_pae` | disabled | Max Boltz-2 predicted aligned error at site residues. Typical: 10-15 |
| `--filter_interface_pae` | disabled | Max Boltz-2 PAE at interface. Typical: 5-10 |

### Geometric Site Filters (applied last)

These filter for binders that sit on top of the specified binding site. Computed in parallel on multiple CPUs.

| Flag | Default | Description |
|------|---------|-------------|
| `--min_site_interface_fraction` | disabled | **SIF: best standalone filter.** Fraction of binder interface residues that contact site residues (vs non-site target residues). 0.5 = at least half the binder's contact surface is on the site. Typical: 0.5-0.7 |
| `--max_site_centroid_dist` | disabled | Distance (A) from binder interface centroid to site CA centroid. Measures how close the binder center is to the site center. Typical: 10-15 A. Note: doesn't distinguish sides — pair with `--min_site_cos` |
| `--centroid_atoms` | CA | Atoms for site centroid: `CA` (1 per residue, equal weight) or `heavy` (all heavy atoms, biased by sidechain size) |
| `--min_site_cos` | disabled | Cosine of angle between target surface normal and binder approach direction. 0.3 = within 72 degrees, 0.5 = within 60 degrees. When used, adds cone visualization to PyMOL scripts |
| `--max_site_dist` | generate_binders: 15.0; rerank: 0 (disabled) | Contact distance cutoff (A) for counting site residues as "contacted" |
| `--min_site_fraction` | 0 | Min fraction of site residues contacted within max_site_dist. Caution: with <5 site residues, 0.4 is too aggressive |
| `--interface_dist` | 5.0 | Distance cutoff defining binder interface residues (for SIF and centroid). 5.0 = direct contacts, 7.0 = broader footprint |

**Recommended for reranking:**
```bash
# Helix designs (with Sc)
--min_site_interface_fraction 0.5 --no_cys --max_aa_fraction 0.3 --min_sc 0.45 --max_interface_ke 0.25

# Beta designs (without Sc — beta sheets have poor Sc on predicted structures)
--min_site_interface_fraction 0.5 --no_cys --max_aa_fraction 0.3 --max_interface_ke 0.25
```

## Output Structure

```
{out_dir}/
├── rfdiffusion/           Backbone PDBs + FASTA sequences
├── boltzgen/              BoltzGen raw outputs
├── bindcraft/             BindCraft accepted designs
├── pxdesign/              PXDesign outputs (merged with full target)
├── proteina/              Proteina backbones + ProteinMPNN sequences
├── proteina_complexa/     Proteina Complexa outputs
├── validation/
│   ├── esmfold/           ESMFold PDBs + pLDDT scores
│   └── boltz/             Boltz-2 CIFs + confidence + PAE
├── top_designs/           Top N ranked complex structures
│   ├── view_by_chain.pml  PyMOL: colored by tool, site in red sticks (+ cone if --min_site_cos)
│   ├── view_by_iptm.pml   PyMOL: colored by iPTM score
│   ├── ke_analysis.csv    K+E composition per design (rank, scores, KE%, sequence)
│   └── {tool}/            Top 20 per tool + per-tool PML scripts
├── plip_analysis/         PLIP interaction reports
│   ├── rank01_{id}/       H-bonds, salt bridges, hydrophobic, .pse
│   └── PLIP_SUMMARY.txt   Per-design interaction counts
├── rankings.csv           All designs with scores + all filter metrics
└── dashboard.png          6-panel summary plot
```

### rankings.csv Columns

Core scoring columns plus per-filter metrics:

| Column | Description |
|--------|-------------|
| `combined_score` | Weighted score (pLDDT + iPTM + dG - penalties) |
| `boltz_iptm` | Boltz-2 interface predicted TM-score |
| `esmfold_plddt` | ESMFold binder pLDDT (0-100) |
| `rosetta_dG` | Rosetta interface binding energy |
| `rosetta_sc` | Rosetta shape complementarity (0-1) |
| `rosetta_sap` | Surface aggregation propensity (lower = less aggregation-prone) |
| `refolding_rmsd` | CA RMSD: ESMFold binder vs Boltz-2 binder |
| `netsolp_solubility` | NetSolP predicted E.coli solubility (0-1, higher = more soluble) |
| `pDockQ` | Docking quality: sigmoid of iPTM × iPLDDT (pDockQ2 formula) (browser-only, computed on load) |
| `site_interface_fraction` | SIF: binder interface at site / total interface |
| `site_contact_fraction` | Site residues contacted / total site residues |
| `site_centroid_dist_CA` | Binder interface centroid to site CA centroid (A) |
| `site_centroid_dist_heavy` | Same, using all heavy atoms for site centroid |
| `site_cos_angle` | Cosine of binder approach angle vs surface normal |
| `interface_KE_fraction` | K+E fraction at binder-target interface |
| `tool` | Design tool (rfdiffusion, boltzgen, etc.) |
| `binder_length` | Binder sequence length (aa) |
| `tier` | Quality tier (1=top 7.5%, 2=top 25%, 3=rest) (browser-only, computed on load) |

## Scoring

```
combined_score = w_plddt * (pLDDT/100) + w_iptm * iPTM + w_dg * dG_norm - site_pae_penalty
```

Default weights: `0.4, 0.5, 0.1` (pLDDT, iPTM, dG). Recommended: `0.3, 0.6, 0.1` to prioritize binding quality over fold confidence.

With `--reprediction`, all tools are scored by Boltz-2 iPTM for uniform cross-tool comparison. Without it, iPTM-native tools (BindCraft, BoltzGen, PXDesign, Proteina Complexa) use their own iPTM scores.

## Validation Pipeline

```
Design (6 tools) → GPU cleanup between tools (auto-retry on failure)
→ ESMFold filter (pLDDT >= 80) → Boltz-2 batch prediction (pre-computed MSA, fast mode)
→ Geometric site metrics (parallel, 12 CPUs) → Rosetta scoring (dG, Sc, SAP)
→ SS + K/E composition → Ranking → PLIP
```

**Note:** Refolding RMSD and NetSolP solubility predictions run during reranking only (`rerank_binders.py`), not during the initial `generate_binders.py` pipeline.

**GPU cleanup:** Between each tool, orphaned GPU processes are detected and killed. On tool failure, GPU is cleaned and the tool is retried once automatically.

**Batch Boltz-2:** All ESMFold-passing designs are validated in a single Boltz-2 invocation (model loads once). With `--boltz_devices N`, inference is distributed across N free GPUs.

**MSA optimization:** The target sequence MSA is computed once via the ColabFold API (~60s), then reused for all designs. Binder sequences use `msa: "empty"` (de novo sequences have no homologs). This eliminates the MSA bottleneck — preprocessing drops from hours to seconds at any scale.

| Scale | MSA time (API, old) | MSA time (pre-computed, new) | Inference time |
|-------|--------------------|-----------------------------|---------------|
| 40 designs (test) | ~20 min | ~60s | ~20 min |
| 1100 designs (standard) | **~34h** | **~60s** | ~10h |

**Reranking ESMFold skip:** When revalidating existing runs, ESMFold results from the original run are reused. Designs that previously failed ESMFold (pLDDT below threshold) are automatically detected and skipped — no redundant re-testing.

**RFdiffusion hotspot subsampling:** When >6 site residues, auto-subsamples to 5 evenly spaced residues prioritizing charged/aromatic (K, R, D, E, F, W, Y). Only affects RFdiffusion.

**Refolding RMSD:** After Boltz-2 validation, the pipeline computes CA RMSD between ESMFold (binder folded alone) and Boltz-2 (binder in complex) using Kabsch superposition. High RMSD (>2.5 A) indicates the binder only adopts its structure when bound to the target — likely disordered on its own.

## K+E Composition Analysis

High lysine (K) + glutamate (E) content in designed binders correlates with poor binding in experimental validation. The pipeline computes K+E metrics for all designs automatically.

**Metrics added to `rankings.csv`:**
- `binder_KE_fraction` — fraction of K + E residues (0-1)
- `binder_K_count` — number of Lys residues
- `binder_E_count` — number of Glu residues

**Reranking log output** includes per-tool K+E summary:
```
K+E composition by tool (high K+E = poor binding risk):
  boltzgen                mean=18%  max=24%  (n=50)
  bindcraft               mean=42%  max=47%  (n=5)   ** 5 above 25%
  proteina                mean=40%  max=62%  (n=20)  ** 15 above 25%
  rfdiffusion             mean=22%  max=72%  (n=20)  ** 3 above 25%
```

**Guidelines:**
- < 20% K+E: good for binding
- 20-25% K+E: acceptable
- \> 25% K+E: poor binding risk — flag for review

**Per-tool observations:**
- **BoltzGen**: cleanest (16-20% typical)
- **PXDesign**: moderate (12-31%)
- **BindCraft**: high (36-47%) — AF2 optimization favors charged interfaces
- **Proteina + ProteinMPNN**: high (37-62%) — ProteinMPNN over-charges on some targets
- **RFdiffusion + LigandMPNN**: variable (1-72%) — some extreme outliers

A separate `ke_analysis.csv` is generated in `top_designs/` with rank, scores, K/E counts, and full sequences for easy filtering in spreadsheet software.

### Interface vs Surface K+E

Total K+E alone is misleading. What matters is **where** the charged residues are:
- **High surface K+E, low interface K+E** — good (soluble + specific binding)
- **High interface K+E** — bad (non-specific electrostatic binding, false positive iPTM)

The pipeline computes interface composition from Boltz-2 complex structures (5A contact threshold):

| Column | Description |
|--------|-------------|
| `interface_KE_fraction` | K+E at interface residues only |
| `interface_K_count`, `interface_E_count` | Counts at interface |
| `interface_n_residues` | Total binder residues at interface |
| `surface_KE_fraction` | K+E on non-interface (surface) residues |

Reranking log shows per-tool breakdown:
```
K+E composition by tool:
  Tool                      Total KE    Interface KE    Surface KE
  pxdesign                   mean=21%        mean=16%      mean=25%
  boltzgen                   mean=18%        mean=10%      mean=22%
```

## SS Bias

| Mode | Avg helix | Avg sheet | Avg loop | Top tools |
|------|-----------|-----------|----------|-----------|
| balanced | ~80% | ~3% | ~17% | PXDesign, RFdiffusion |
| beta | ~26-35% | ~24-28% | ~40-46% | BoltzGen, Proteina Complexa |
| helix | ~85%+ | ~2% | ~13% | PXDesign, RFdiffusion |

Run both `balanced` and `beta` for structural diversity. Beta mode produces lower scores but different binding modes.

**Note:** Most tools produce helical binders regardless of beta bias. Only BoltzGen (~25% sheet) and Proteina Complexa (~16% sheet) respond meaningfully to beta conditioning.

## GPU Notes

- Use `CUDA_VISIBLE_DEVICES=N` to pin GPU. Do **not** combine with `--device cuda:0` — the `--device` flag overrides `CUDA_VISIBLE_DEVICES` and can cause all runs to land on GPU 0.
- The pipeline automatically cleans up orphaned GPU processes between tools and retries on failure.
- For multi-GPU Boltz-2 validation, use `--boltz_devices N`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BINDER_SOFTWARE_DIR` | `~/data/software` | Base directory for all tool installations |
| `BINDER_WEIGHTS_DIR` | `~/data/weights` | Base directory for model weights |
| `RFDIFFUSION_DIR` | `{SOFTWARE}/RFdiffusion` | RFdiffusion installation |
| `RFDIFFUSION_CKPT` | `{WEIGHTS}/rfdiffusion/Complex_base_ckpt.pt` | RFdiffusion checkpoint |
| `LIGANDMPNN_DIR` | `{SOFTWARE}/LigandMPNN` | LigandMPNN installation |
| `LIGANDMPNN_CKPT` | `{WEIGHTS}/ligandmpnn/proteinmpnn_v_48_020.pt` | LigandMPNN checkpoint |
| `BINDCRAFT_DIR` | `{SOFTWARE}/BindCraft` | BindCraft installation |
| `BOLTZGEN_BIN` | `{SOFTWARE}/envs/boltzgen/bin/boltzgen` | BoltzGen binary |
| `PXDESIGN_DIR` | `{SOFTWARE}/PXDesign` | PXDesign installation |
| `PROTEINA_DIR` | `{SOFTWARE}/Proteina` | Proteina installation |
| `PROTEINA_CKPT` | `{WEIGHTS}/proteina/proteina_v1.1_DFS_200M_tri.ckpt` | Proteina checkpoint |
| `PROTEINA_COMPLEXA_DIR` | `{SOFTWARE}/Proteina-Complexa` | Proteina Complexa installation |
| `PROTEINA_COMPLEXA_VENV` | `{SOFTWARE}/Proteina-Complexa/.venv` | Proteina Complexa UV venv |
| `PROTEINA_PYTHON` | `{SOFTWARE}/envs/proteina_env/bin/python` | Proteina Python binary |

## Conda Environment Summary

| Env name | Tool(s) | Key packages |
|----------|---------|-------------|
| `boltz` | Host + Boltz-2 validation | boltz, numpy, matplotlib, gemmi, plip |
| `rfdiffusion` | RFdiffusion | PyTorch, SE3-Transformers |
| `mpnn` | LigandMPNN + ProteinMPNN | PyTorch |
| `boltzgen` (prefix) | BoltzGen | PyTorch (needs `MKL_THREADING_LAYER=GNU`) |
| `BindCraft` | BindCraft | JAX, AlphaFold2 |
| `pxdesign` | PXDesign | PyTorch (needs `LAYERNORM_TYPE=torch`) |
| `proteina_env` (prefix) | Proteina | PyTorch |
| `esmfold` | ESMFold | PyTorch, transformers, esm |
| UV venv | Proteina Complexa | PyTorch, AF2, ESM2 (self-contained) |

## Binder Browser

Interactive PyQt6 desktop application for exploring, comparing, and filtering binder design results.

```bash
# Launch from the protein_folding directory
python -m binders_pipeline.binder_browser --results_dir outputs/my_run/reranks/my_rerank/
```

### Features

- **Rankings table** — sortable, filterable, click to select design
- **Scatter plots** — 15 presets (iPTM vs SIF, Score vs Length, Structure × Docking, etc.) + custom axes
- **Tool comparison** — per-tool box plots, overlaid histograms, score vs length by tool
- **Radar chart** — multi-axis developability profile with 8 axes:

| Axis | Metric | Good value |
|------|--------|-----------|
| Binding | iPTM | > 0.8 |
| Structure | Shape complementarity (Sc) | > 0.55 |
| Stability | Refolding RMSD | < 2.0 Å |
| Docking | pDockQ | > 0.8 |
| Low KE | Interface K+E fraction | < 20% |
| Low Aggregation | SAP score | < 80 |
| Solubility | NetSolP (E.coli) | > 0.7 |
| Site Focus | SIF | > 0.5 |

- **Design detail** — rank navigation dropdown, all score cards, sequence, SS composition
- **Tier classification** — Top 7.5% = Tier 1, next 17.5% = Tier 2, rest = Tier 3
- **pDockQ** — sigmoid of iPTM × iPLDDT (pDockQ2 formula, computed on load)
- **Dark theme** — modern dark UI with matching matplotlib plots
- **Scores guide** — "? Scores Guide" button explains each radar axis

### Radar modes

| Mode | Shows |
|------|-------|
| None (design only) | Empty grid + selected design overlay |
| Tiers | Tier 1/2/3 median profiles |
| Tools | Per-tool median profiles |
| Top 10 vs All | Top 10 ranked vs all designs |

Select individual designs via rank dropdown, text search (with autocomplete), or clicking in the rankings table / scatter plot.

### Dependencies

```bash
pip install PyQt6 matplotlib pandas numpy
```

### Developability predictions

| Prediction | Tool | When computed |
|-----------|------|--------------|
| SAP (aggregation) | Rosetta InterfaceAnalyzerMover | During Rosetta scoring step |
| Solubility | [NetSolP](https://github.com/tvinet/NetSolP-1.0) (ESM-based, ONNX) | During reranking (runs in esmfold env) |
| pDockQ | Sigmoid of iPTM × iPLDDT (pDockQ2 formula) | On load in browser |

NetSolP requires ONNX models downloaded from [DTU](https://services.healthtech.dtu.dk/services/NetSolP-1.0/) and `onnxruntime` + `fair-esm` in the esmfold conda env.

## License

The **binderflow** pipeline code is released under the MIT License. See [LICENSE](LICENSE).

Each design tool has its own license — you are responsible for complying with them:

| Tool | License | Notes |
|------|---------|-------|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | BSD 3-clause | Attribution required |
| [LigandMPNN](https://github.com/dauparas/LigandMPNN) | MIT | Attribution required |
| [BoltzGen](https://github.com/HannesStark/boltzgen) | See repo | Check upstream license |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | MIT | Attribution required |
| [PXDesign](https://github.com/bytedance/PXDesign) | Apache 2.0 | Attribution required |
| [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina/) | NVIDIA Custom | **Non-commercial / research use only** |
| [Proteina Complexa](https://github.com/NVIDIA-Digital-Bio/proteina-complexa) | NVIDIA Custom | **Non-commercial / research use only** |
| [Boltz-2](https://github.com/jwohlwend/boltz) | MIT | Attribution required |
| [ESMFold](https://github.com/facebookresearch/esmfold) | MIT | Attribution required |

> **Note:** Proteina and Proteina Complexa (NVIDIA) are restricted to non-commercial and research/evaluation use. If you use these tools, ensure your use case complies with NVIDIA's license terms.

## Citation

If you use this pipeline, please cite the individual design tools used in your study.
