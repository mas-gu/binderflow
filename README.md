# binderflow

**Author:** Guillaume Mas

A unified pipeline for de novo protein binder design using six complementary generative tools, with automated validation, geometric site filtering, and interface analysis.

## Overview

Given a target protein PDB and a binding site, the pipeline:

1. **Generates** binder candidates using up to 6 design tools in parallel
2. **Validates** all designs with ESMFold (fast fold quality filter) and Boltz-2 (uniform cross-tool scoring with site pocket constraint)
3. **Filters** off-site binders using geometric site contact fraction
4. **Scores** interfaces with Rosetta and analyzes contacts with PLIP
5. **Ranks** all designs by combined score (pLDDT + iPTM + dG)
6. **Outputs** ranked structures, PyMOL scripts, dashboard plots, and PLIP reports

## Design Tools

| Tool | Type | What it generates | Reference |
|------|------|-------------------|-----------|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Backbone diffusion | Backbone + LigandMPNN sequences | Watson et al., Nature 2023 |
| [BoltzGen](https://github.com/jwohlwend/boltzgen) | Full-atom diffusion | Full-atom binder structures | Jing et al., 2024 |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | AF2-guided optimization | Iterative sequence design | Pacesa et al., 2024 |
| [PXDesign](https://github.com/NVIDIA/PXDesign) | DiT diffusion + AF2-IG | Backbone + sequence + validation | NVIDIA, 2024 |
| [Proteina](https://github.com/NVIDIA/Proteina) | Flow-based backbone | Unconditional backbones + ProteinMPNN | NVIDIA, 2024 |
| [Proteina Complexa](https://github.com/NVIDIA/Proteina) | Flow-based full-atom | Target-conditioned binder design | NVIDIA, ICLR 2026 |

## Quick Start

```bash
# Activate host environment
conda activate boltz

# Test run (6 tools, ~4h, single GPU)
python generate_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --length 60-80 \
    --tools rfdiffusion,boltzgen,bindcraft,pxdesign,proteina,proteina_complexa \
    --mode test \
    --device cuda:0 \
    --score_weights 0.3,0.6,0.1 \
    --ss_bias balanced \
    --max_site_dist 8.0 \
    --filter_site_pae 15.0 \
    --reprediction \
    --plip_top 10 \
    --out_dir ./output/
```

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04)
- Python >= 3.7
- NVIDIA GPU with >= 24 GB VRAM (RTX 3090, A100, etc.)
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
| BoltzGen | `boltzgen` (prefix env) | [GitHub](https://github.com/jwohlwend/boltzgen) |
| BindCraft | `BindCraft` (capital B, C) | [GitHub](https://github.com/martinpacesa/BindCraft) |
| PXDesign | `pxdesign` | [GitHub](https://github.com/NVIDIA/PXDesign) |
| Proteina | `proteina_env` (prefix env) | [GitHub](https://github.com/NVIDIA/Proteina) |
| Proteina Complexa | UV venv (not conda) | [GitHub](https://github.com/NVIDIA/Proteina) |
| ESMFold | `esmfold` | [HuggingFace](https://huggingface.co/facebook/esmfold_v1) |

### 3. Optional Dependencies

| Tool | Conda env | Purpose |
|------|-----------|---------|
| PyRosetta | (in boltz env) | Interface energy scoring (dG). Optional but recommended |
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
| production | 4,520 | ~5 days | ~36h |

Boltz-2 MSA is pre-computed once (~60s). Revalidation skips design generation + ESMFold.

Per-tool breakdown:

| Tool | test | standard | production |
|------|------|----------|------------|
| RFdiffusion | 20 | 200 | 500 |
| BoltzGen | 50 | 500 | 2,500 |
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
| `--max_site_dist 8.0` | Contact distance threshold (A) | 6.0-8.0 |
| `--min_site_fraction 0.4` | Min fraction of site residues contacted | 0.3-0.6 |
| `--filter_site_pae 15.0` | Max site PAE (removes off-site binders) | 10.0-15.0 |
| `--score_weights 0.3,0.6,0.1` | pLDDT, iPTM, dG weights | Higher iPTM = binding quality |
| `--plip_top 10` | PLIP analysis on top N designs | 10 |
| `--boltz_devices 3` | Multi-GPU for Boltz-2 batch validation | All free GPUs |

### Re-ranking Existing Runs

```bash
# Re-rank with different filters (no re-validation, fast)
python rerank_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --results_dir ./output/ \
    --out_dir ./output_reranked/ \
    --rank_only \
    --ss_bias beta \
    --max_site_dist 6.0 \
    --min_site_fraction 0.5 \
    --plip_top 10

# Revalidate: batch Boltz-2 on ALL designs (finds hidden good designs)
CUDA_VISIBLE_DEVICES=0 python rerank_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --results_dir ./output/ \
    --out_dir ./output_revalidated/ \
    --reprediction \
    --max_site_dist 8.0 \
    --min_site_fraction 0.4 \
    --plip_top 10

# Merge multiple runs
python rerank_binders.py \
    --target target.pdb \
    --site "A:11-17,119-124" \
    --results_dir ./out_balanced,./out_beta \
    --out_dir ./merged/ \
    --rank_only
```

### Running Specific Tools

```bash
# Only RFdiffusion + BoltzGen (fastest combination)
python generate_binders.py \
    --target target.pdb \
    --site "A:11-17" \
    --tools rfdiffusion,boltzgen \
    --mode test \
    --device cuda:0 \
    --out_dir ./output/
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
│   ├── view_by_chain.pml  PyMOL: colored by tool, site in red sticks
│   ├── view_by_iptm.pml   PyMOL: colored by iPTM score
│   ├── ke_analysis.csv    K+E composition per design (rank, scores, KE%, sequence)
│   └── {tool}/            Top 20 per tool + per-tool PML scripts
├── plip_analysis/         PLIP interaction reports
│   ├── rank01_{id}/       H-bonds, salt bridges, hydrophobic, .pse
│   └── PLIP_SUMMARY.txt   Per-design interaction counts
├── rankings.csv           All designs with scores
└── dashboard.png          6-panel summary plot
```

## Scoring

```
combined_score = w_plddt * (pLDDT/100) + w_iptm * iPTM + w_dg * dG_norm - site_pae_penalty
```

Default weights: `0.3, 0.6, 0.1` (pLDDT, iPTM, dG).

With `--reprediction`, all tools are scored by Boltz-2 iPTM for uniform cross-tool comparison. Without it, iPTM-native tools (BindCraft, BoltzGen, PXDesign, Proteina Complexa) use their own iPTM scores.

## Validation Pipeline

```
Design (6 tools) → ESMFold filter (pLDDT ≥ 70) → Boltz-2 batch prediction
→ Geometric site filter → Rosetta scoring → SS + K/E composition → Ranking → PLIP
```

**Batch Boltz-2:** All ESMFold-passing designs are validated in a single Boltz-2 invocation (model loads once). With `--boltz_devices N`, inference is distributed across N free GPUs.

**MSA optimization:** The target sequence MSA is computed once via the ColabFold API (~60s), then reused for all designs. Binder sequences use `msa: "empty"` (de novo sequences have no homologs). This eliminates the MSA bottleneck — preprocessing drops from hours to seconds at any scale.

| Scale | MSA time (API, old) | MSA time (pre-computed, new) | Inference time |
|-------|--------------------|-----------------------------|---------------|
| 40 designs (test) | ~20 min | ~60s | ~20 min |
| 1100 designs (standard) | **~34h** | **~60s** | ~10h |

**Reranking ESMFold skip:** When revalidating existing runs, ESMFold results from the original run are reused. Designs that previously failed ESMFold (pLDDT below threshold) are automatically detected and skipped — no redundant re-testing.

**Geometric site filter:** Counts site residues with binder heavy atoms within `--max_site_dist`. Designs with `site_contact_fraction < --min_site_fraction` are excluded.

**RFdiffusion hotspot subsampling:** When >6 site residues, auto-subsamples to 5 evenly spaced residues prioritizing charged/aromatic (K, R, D, E, F, W, Y). Only affects RFdiffusion.

## K+E Composition Analysis

High lysine (K) + glutamate (E) content in designed binders correlates with poor expression in experimental validation. The pipeline computes K+E metrics for all designs automatically.

**Metrics added to `rankings.csv`:**
- `binder_KE_fraction` — fraction of K + E residues (0-1)
- `binder_K_count` — number of Lys residues
- `binder_E_count` — number of Glu residues

**Reranking log output** includes per-tool K+E summary:
```
K+E composition by tool (high K+E = expression risk):
  boltzgen                mean=18%  max=24%  (n=50)
  bindcraft               mean=42%  max=47%  (n=5)   ** 5 above 25%
  proteina                mean=40%  max=62%  (n=20)  ** 15 above 25%
  rfdiffusion             mean=22%  max=72%  (n=20)  ** 3 above 25%
```

**Guidelines:**
- < 20% K+E: good for expression
- 20-25% K+E: acceptable
- \> 25% K+E: expression risk — flag for review

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

## License

The **binderflow** pipeline code is released under the MIT License. See [LICENSE](LICENSE).

Each design tool has its own license — you are responsible for complying with them:

| Tool | License | Notes |
|------|---------|-------|
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | BSD 3-clause | Attribution required |
| [LigandMPNN](https://github.com/dauparas/LigandMPNN) | MIT | Attribution required |
| [BoltzGen](https://github.com/jwohlwend/boltzgen) | See repo | Check upstream license |
| [BindCraft](https://github.com/martinpacesa/BindCraft) | MIT | Attribution required |
| [PXDesign](https://github.com/bytedance/PXDesign) | Apache 2.0 | Attribution required |
| [Proteina](https://github.com/NVIDIA-Digital-Bio/proteina) | NVIDIA Custom | **Non-commercial / research use only** |
| [Proteina Complexa](https://github.com/NVIDIA-Digital-Bio/proteina) | NVIDIA Custom | **Non-commercial / research use only** |
| [Boltz-2](https://github.com/jwohlwend/boltz) | MIT | Attribution required |
| [ESMFold](https://github.com/facebookresearch/esm) | MIT | Attribution required |

> **Note:** Proteina and Proteina Complexa (NVIDIA) are restricted to non-commercial and research/evaluation use. If you use these tools, ensure your use case complies with NVIDIA's license terms.

## Citation

If you use this pipeline, please cite the individual design tools used in your study.
