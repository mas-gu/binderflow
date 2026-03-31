#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume Mas
# SPDX-License-Identifier: MIT
"""
generate_binders.py — Multi-tool binder design pipeline.

Given a target protein and a binding site, runs up to six complementary design
tools, validates all outputs with ESMFold (fast pre-filter) and Boltz-2 (uniform
cross-tool scoring with site pocket constraint), then ranks and outputs the best
binder candidates.

═══════════════════════════════════════════════════════════════════════════════
QUICK START
═══════════════════════════════════════════════════════════════════════════════
    conda activate boltz   # host env — needs numpy + matplotlib

    python /data/scripts/protein_folding/generate_binders.py \\
        --target  target.pdb \\
        --site    "A:8-30,120-123" \\
        --length  60-80 \\
        --tools   rfdiffusion,boltzgen,bindcraft \\
        --out_dir ./binder_run/

    # Dry run (prints all commands, no execution):
    python generate_binders.py --target t.pdb --site "A:10-30" --length 60-80 \\
        --tools rfdiffusion --mode test --out_dir ./test --dry_run

═══════════════════════════════════════════════════════════════════════════════
TOOLS
═══════════════════════════════════════════════════════════════════════════════
rfdiffusion
    RFdiffusion backbone generation (Complex_base_ckpt.pt, T=50 steps,
    noise_scale 0.5 for CA and frame) followed by LigandMPNN sequence
    design (4 sequences per backbone, T=0.1, pick best by MPNN confidence).
    Conda env: rfdiffusion + mpnn.

boltzgen
    BoltzGen end-to-end diffusion design (protein-anything protocol,
    200 sampling steps, batch_size=5 for 20GB GPU, 2 inverse-fold sequences
    per backbone, budget=min(n,500) diverse final designs, alpha=0.1 for
    quality+diversity balance). Merges 28 native quality columns (bg_*).
    Conda env: base.

bindcraft
    BindCraft iterative JAX/AF2 design (default_4stage_multimer preset:
    75/45/5/15 iterations, 3 AF2 validation recycles, default filters).
    NOTE: weights_helicity=-0.3 in preset biases toward β-sheets. For targets
    where helical binders are preferred (e.g. KRAS Switch I/II), swap to a
    custom preset with weights_helicity=+0.3 or use settings_advanced/
    default_4stage_multimer.json edited accordingly.
    Merges 39 native quality columns (bc_*).
    Conda env: BindCraft.

pxdesign
    PXDesign full binder design pipeline (ByteDance/Protenix team).
    DiT diffusion backbone generation → ProteinMPNN sequence design →
    AF2-IG filter → Protenix filter → Foldseek clustering → Ranked output.
    Achieves 20-73% experimental nanomolar hit rates (2-6x RFdiffusion).
    PXDesign does its own internal multi-model filtering; designs reaching
    summary.csv are already pre-validated. Our ESMFold + Boltz-2 stages
    add independent cross-model validation.
    Merges 7 native quality columns (px_*).
    Conda env: pxdesign.

proteina
    Proteina flow-based backbone generation (NVIDIA, arxiv:2503.00710).
    Flow matching backbone generator with classifier-free guidance for fold
    conditioning, handles up to 800 residues. Same pipeline as RFdiffusion:
    backbone PDBs → LigandMPNN sequence design (4 seqs/backbone, T=0.1,
    pick best by MPNN confidence).
    Conda env: proteina + mpnn.

═══════════════════════════════════════════════════════════════════════════════
DESIGN MODES
═══════════════════════════════════════════════════════════════════════════════
    test:       rfdiffusion=20,   boltzgen=50,    bindcraft=5,   pxdesign=20,   proteina=20    (~1–2 h)
    standard:   rfdiffusion=200,  boltzgen=1000,  bindcraft=10,  pxdesign=200,  proteina=200   (~12–24 h)  [default]
    production: rfdiffusion=500,  boltzgen=10000, bindcraft=20,  pxdesign=500,  proteina=500   (~2–4 days)

    # Override per-tool count (takes precedence over mode):
    python generate_binders.py ... --mode standard --n_designs boltzgen=10000

═══════════════════════════════════════════════════════════════════════════════
KEY ARGUMENTS
═══════════════════════════════════════════════════════════════════════════════
    --site "CHAIN:RES[,RES...]"
        Binding site residues (1-based, chain-prefixed).
        Ranges and comma lists supported: "A:8-30,120-123"
        Used for:
          • RFdiffusion hotspot_res  (3–6 residues optimal per README)
          • BoltzGen binding_types YAML
          • BindCraft target_hotspot_residues
          • Boltz pocket constraint (forces docking at site; avoids off-target)

    --score_weights W_PLDDT,W_IPTM[,W_DG]
        Weights for combined_score = w_plddt*(pLDDT/100) + w_iptm*ipTM + w_dg*dG_norm
        Default: 0.4,0.5,0.1  (iPTM-weighted + Rosetta dG physics bonus)
        Two values also accepted for backward compat (dG weight=0).

    --filter_interface_pae THRESHOLD
        Drop Boltz-validated designs with min interface pAE above threshold.
        High pAE = uncertain contacts = likely not a real binder.
        Example: --filter_interface_pae 12.0

    --max_validate N
        Number of designs per tool sent to Boltz validation (default: 20).
        Selected by highest ESMFold pLDDT.

    --top_n N
        Number of top designs copied to top_designs/ (default: 50).

═══════════════════════════════════════════════════════════════════════════════
VALIDATION PIPELINE
═══════════════════════════════════════════════════════════════════════════════
Stage 1 — ESMFold pre-filter
    Each binder sequence folded independently. Threshold: pLDDT ≥ 70.
    Filters unstructured/disordered sequences before expensive Boltz runs.

Stage 2 — Boltz-2 uniform scoring (top N per tool by ESMFold pLDDT)
    Binder + target complex predicted with:
      • recycling_steps=2, sampling_steps=200, num_subsampled_msa=128
      • Pocket constraint steering to --site residues (prevents off-target docking)
      • Per-design seed for reproducibility
    Metrics extracted per design:
      boltz_iptm               — interface pTM (primary ranking signal)
      boltz_confidence         — composite confidence score
      boltz_protein_iptm       — protein-protein interface pTM
      boltz_ptm                — full complex pTM
      boltz_complex_plddt      — mean pLDDT over complex (0–100)
      boltz_iplddt             — interface pLDDT (0–100)
      boltz_binder_plddt       — mean pLDDT of binder chain only (0–100)
      boltz_complex_pde        — complex predicted distance error (0–1)
      boltz_complex_ipde       — interface predicted distance error (Å)
      boltz_min_interface_pae  — min PAE in target→binder block
      boltz_mean_interface_pae — mean PAE in target→binder block
      boltz_min_interface_pde  — min PDE in target→binder block
      boltz_mean_interface_pde — mean PDE in target→binder block

═══════════════════════════════════════════════════════════════════════════════
RANKINGS & OUTPUT
═══════════════════════════════════════════════════════════════════════════════
    combined_score = 0.4*(pLDDT/100) + 0.5*ipTM + 0.1*dG_norm   [default weights]

    rankings.csv columns (50+):
      Core:    rank, design_id, tool, binder_length, binder_sequence
      ESMFold: esmfold_plddt
      Boltz:   boltz_iptm, boltz_confidence, boltz_protein_iptm, boltz_ptm,
               boltz_complex_plddt, boltz_iplddt, boltz_binder_plddt,
               boltz_complex_pde, boltz_complex_ipde,
               boltz_min/mean_interface_pae, boltz_min/mean_interface_pde,
               combined_score
      Native:  native_score, native_score_name
               ligandmpnn_seq_rec  (RFdiffusion only)
      BoltzGen (bg_*): design_to_target_iptm, min_design_to_target_pae,
               design_ptm, complex_plddt, complex_iplddt, filter_rmsd,
               delta_sasa_refolded, plip_hbonds_refolded, plip_saltbridge_refolded,
               design_hydrophobicity, largest_hydrophobic_patch_refolded,
               liability_score, liability_num_violations, liability_violations_summary,
               quality_score, helix/loop/sheet, bindsite_under_3/4rmsd,
               native_rmsd_refolded, design_sasa_bound/unbound_refolded,
               pass_filters, rank_design_to_target_iptm, rank_design_ptm
      BindCraft (bc_*): pLDDT, pTM, i_pTM, pAE, i_pAE, i_pLDDT, ss_pLDDT,
               Binder_Energy_Score, Surface_Hydrophobicity, ShapeComplementarity,
               PackStat, dG, dSASA, dG_per_dSASA, Interface_Hydrophobicity,
               Interface_SASA_pct, n_InterfaceResidues, n_InterfaceHbonds,
               InterfaceHbondsPercentage, n_InterfaceUnsatHbonds,
               Binder/Interface_Helix/BetaSheet/Loop_pct,
               Hotspot_RMSD, Target_RMSD, Binder_RMSD,
               Binder_pLDDT, Binder_pTM, Binder_pAE,
               Unrelaxed/Relaxed_Clashes, MPNN_score, MPNN_seq_recovery
      PXDesign (px_*): iptm, ipae, plddt, ptm, rmsd, cluster_id, filter_level
      Proteina: ligandmpnn_seq_rec (same as RFdiffusion — backbone generator)

    {out_dir}/
    ├── rfdiffusion/   backbones/ (PDB) + sequences/ (FASTA + per-residue stats)
    ├── boltzgen/      raw BoltzGen outputs + final_ranked_designs/
    ├── pxdesign/      PXDesign outputs + summary.csv
    ├── proteina/      backbones/ (PDB) + sequences/ (FASTA + per-residue stats)
    ├── bindcraft/     bindcraft_output/Accepted/ + final_design_stats.csv
    ├── validation/    esmfold/ + boltz/ per-design structures & scores
    ├── top_designs/   top N binder+KRAS complex CIFs + binder FASTA
    ├── rankings.csv   all designs, all scores, sortable
    └── dashboard.png  6-panel summary plot

Host env: boltz conda env (numpy, matplotlib).
Each tool runs via `conda run -n <env>` subprocess.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Constants ──────────────────────────────────────────────────────────────────
# Paths are configurable via environment variables.
# Set BINDER_SOFTWARE_DIR and BINDER_WEIGHTS_DIR, or override individual paths.

_SOFTWARE_DIR = os.environ.get("BINDER_SOFTWARE_DIR", os.path.expanduser("~/data/software"))
_WEIGHTS_DIR  = os.environ.get("BINDER_WEIGHTS_DIR", os.path.expanduser("~/data/weights"))

RFDIFFUSION_DIR      = os.environ.get("RFDIFFUSION_DIR",      f"{_SOFTWARE_DIR}/RFdiffusion")
RFDIFFUSION_CKPT     = os.environ.get("RFDIFFUSION_CKPT",     f"{_WEIGHTS_DIR}/rfdiffusion/Complex_base_ckpt.pt")
LIGANDMPNN_DIR       = os.environ.get("LIGANDMPNN_DIR",       f"{_SOFTWARE_DIR}/LigandMPNN")
LIGANDMPNN_CKPT      = os.environ.get("LIGANDMPNN_CKPT",      f"{_WEIGHTS_DIR}/ligandmpnn/proteinmpnn_v_48_020.pt")
BINDCRAFT_DIR        = os.environ.get("BINDCRAFT_DIR",         f"{_SOFTWARE_DIR}/BindCraft")
BOLTZGEN_BIN         = os.environ.get("BOLTZGEN_BIN",          f"{_SOFTWARE_DIR}/envs/boltzgen/bin/boltzgen")
PXDESIGN_DIR         = os.environ.get("PXDESIGN_DIR",          f"{_SOFTWARE_DIR}/PXDesign")
PROTEINA_DIR         = os.environ.get("PROTEINA_DIR",          f"{_SOFTWARE_DIR}/Proteina")
PROTEINA_CKPT        = os.environ.get("PROTEINA_CKPT",        f"{_WEIGHTS_DIR}/proteina/proteina_v1.1_DFS_200M_tri.ckpt")
PROTEINA_COMPLEXA_DIR  = os.environ.get("PROTEINA_COMPLEXA_DIR",  f"{_SOFTWARE_DIR}/Proteina-Complexa")
PROTEINA_COMPLEXA_VENV = os.environ.get("PROTEINA_COMPLEXA_VENV", f"{_SOFTWARE_DIR}/Proteina-Complexa/.venv")

TOOL_COLORS = {
    "rfdiffusion":        "#E53935",
    "boltzgen":           "#1E88E5",
    "bindcraft":          "#43A047",
    "pxdesign":           "#FF9800",
    "proteina":           "#76FF03",
    "proteina_complexa":  "#AB47BC",
}
ALL_TOOLS = list(TOOL_COLORS)

# Tools that produce their own iPTM scores natively (no Boltz-2 re-prediction needed)
IPTM_NATIVE_TOOLS = {"bindcraft", "boltzgen", "pxdesign", "proteina_complexa"}
# Backbone-only generators that need Boltz-2 for iPTM scoring
BACKBONE_ONLY_TOOLS = {"rfdiffusion", "proteina"}

# ── Secondary structure bias parameters ───────────────────────────────────────
SS_BIAS_PARAMS = {
    "beta": {
        "bc_helicity": -0.6,
        "bg_helix_max": 0.4,
        "pc_max_helix": 0.4,          # hard-filter Complexa designs with helix > 0.4
        "penalty_helix_above": 0.35,
        "penalty_sheet_below": 0.15,
    },
    "helix": {
        "bc_helicity": 0.5,
        "bg_sheet_max": 0.2,
        "pc_max_sheet": 0.3,           # hard-filter Complexa designs with sheet > 0.3
        "penalty_sheet_above": 0.30,
        "penalty_helix_below": 0.25,
    },
    "balanced": {},  # no modifications
}

# ── Design mode presets ────────────────────────────────────────────────────────
# n_designs per tool for each mode. BoltzGen needs many more because most raw
# designs are filtered internally before reaching final_ranked_designs.
DESIGN_MODES = {
    "test":       {"rfdiffusion":  20, "boltzgen":    50, "bindcraft":   5, "pxdesign":   20, "proteina":   20, "proteina_complexa":   20},
    "standard":   {"rfdiffusion": 200, "boltzgen":   500, "bindcraft":  10, "pxdesign":  200, "proteina":  200, "proteina_complexa":  200},
    "production": {"rfdiffusion": 500, "boltzgen": 10000, "bindcraft":  20, "pxdesign":  500, "proteina":  500, "proteina_complexa":  500},
}
DEFAULT_MODE = "standard"

# Standard 3-letter → 1-letter amino acid codes
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine
}


# ── Utility ────────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# GPU pinning env — set by --device flag, merged into every run_cmd call
GPU_ENV = {}

def run_cmd(cmd, timeout=86400, extra_env=None, cwd=None, dry_run=False):
    """Run a command. On dry_run, print without executing. Raises RuntimeError on failure."""
    if dry_run:
        print(f"  [DRY RUN] {' '.join(str(c) for c in cmd)}")
        return "", ""
    env = {**os.environ, **GPU_ENV, **(extra_env or {})}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=env, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:] or r.stdout[-3000:])
    return r.stdout, r.stderr


def _cleanup_gpu(gpu_id=None):
    """Kill orphaned GPU processes on the target GPU after a tool finishes.

    When a tool subprocess crashes, child processes may survive and hold GPU
    memory (zombie CUDA contexts). This function finds and kills any python
    processes on our GPU that are NOT the main pipeline process.
    """
    if gpu_id is None:
        gpu_id = GPU_ENV.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if gpu_id is None:
        return
    my_pid = os.getpid()
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            pid = int(parts[0].strip())
            if pid == my_pid:
                continue
            # Check if this process is a child of our pipeline — don't kill unrelated work
            try:
                pr = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                                    capture_output=True, text=True, timeout=5)
                ppid = int(pr.stdout.strip())
            except Exception:
                continue
            # Kill orphans whose parent is init (1) or our own PID
            # (our subprocesses should have exited; lingering ones are zombies)
            if ppid == my_pid:
                mem_mb = int(parts[1].strip())
                log(f"  GPU cleanup: killing orphan PID {pid} ({mem_mb} MiB on GPU {gpu_id})")
                os.kill(pid, 9)
                import time; time.sleep(2)
    except Exception:
        pass  # non-critical — best effort cleanup


def _safe_float(val, default=float("nan")):
    """Parse val as float, returning default on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_tool_from_design_id(design_id):
    """Extract tool name from design_id like 'rfdiffusion_0003' or 'proteina_complexa_0005'.

    Matches against ALL_TOOLS longest-first so that 'proteina_complexa_0005' returns
    'proteina_complexa' rather than 'proteina'.

    Also handles prefixed IDs from rerank_binders.py merge mode, e.g.
    'out_helix__rfdiffusion_0006' — strips everything before '__'.
    """
    # Strip merge prefix (e.g. "dirname__rfdiffusion_0006" → "rfdiffusion_0006")
    if "__" in design_id:
        design_id = design_id.split("__", 1)[1]
    for tool in sorted(ALL_TOOLS, key=len, reverse=True):
        if design_id.startswith(tool + "_"):
            return tool
    return design_id.split("_")[0]  # fallback


# ── Structure parsers ──────────────────────────────────────────────────────────

def extract_pdb_chain_info(pdb_path):
    """
    Parse PDB ATOM records.
    Returns {chain_id: {"length": N, "sequence": str, "residues": [resnum, ...]}}
    """
    chains = {}  # chain_id → {resnum: resname}
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        altloc = line[16]
        if altloc not in (" ", "A", "1"):
            continue
        chain = line[21]
        try:
            resnum  = int(line[22:26].strip())
            resname = line[17:20].strip()
        except ValueError:
            continue
        chains.setdefault(chain, {})
        if resnum not in chains[chain]:
            chains[chain][resnum] = resname

    result = {}
    for chain, rmap in chains.items():
        resnums = sorted(rmap)
        result[chain] = {
            "length":   len(resnums),
            "residues": resnums,
            "sequence": "".join(AA3.get(rmap[r], "X") for r in resnums),
        }
    return result


def extract_cif_chain_info(cif_path):
    """
    Parse mmCIF _atom_site records.
    Returns {chain_id: {"length": N, "sequence": str}}
    """
    lines = Path(cif_path).read_text().splitlines()
    col_map, col_count, in_atom_site, data_start = {}, 0, False, None

    for i, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            in_atom_site = False; col_map = {}; col_count = 0
        elif s.startswith("_atom_site."):
            in_atom_site = True; col_map[s] = col_count; col_count += 1
        elif in_atom_site and s and not s.startswith(("_", "#", "loop_", "data_")):
            data_start = i; break

    if not col_map or data_start is None:
        return {}

    def gcol(name):
        return col_map.get(name, col_map.get(name.replace("label_", "auth_"), -1))

    ci_atom  = gcol("_atom_site.label_atom_id")
    ci_chain = gcol("_atom_site.label_asym_id")
    ci_res   = gcol("_atom_site.label_comp_id")
    ci_seqid = gcol("_atom_site.label_seq_id")

    if any(c < 0 for c in [ci_atom, ci_chain, ci_res, ci_seqid]):
        return {}

    max_ci = max(ci_atom, ci_chain, ci_res, ci_seqid)
    chains = {}  # chain → {seqid: resname}
    for line in lines[data_start:]:
        s = line.strip()
        if not s or s.startswith(("#", "_", "loop_", "data_")):
            break
        parts = s.split()
        if len(parts) <= max_ci or parts[ci_atom] != "CA":
            continue
        chain   = parts[ci_chain]
        resname = parts[ci_res]
        try:
            seqid = int(parts[ci_seqid])
        except (ValueError, IndexError):
            continue
        chains.setdefault(chain, {})
        if seqid not in chains[chain]:
            chains[chain][seqid] = resname

    result = {}
    for chain, rmap in chains.items():
        resnums = sorted(rmap)
        result[chain] = {
            "length":   len(resnums),
            "sequence": "".join(AA3.get(rmap[r], "X") for r in resnums),
        }
    return result


def get_chain_info(struct_path):
    """Dispatch to PDB or CIF parser based on extension."""
    ext = Path(struct_path).suffix.lower()
    if ext == ".cif":
        return extract_cif_chain_info(struct_path)
    return extract_pdb_chain_info(struct_path)


# ── Input parsing ──────────────────────────────────────────────────────────────

def parse_site(site_str):
    """
    Parse binding site specification into (chain_id, [resnum, ...]).

    Accepted formats:
        "A:8-30,120-123"     — colon-separated chain + comma/range list  [preferred]
        "A:8,15,22"          — colon-separated chain + individual residues
        "A:8"                — single residue
        "A10 A15 A22"        — legacy space-separated tokens
        "A10,A15,A22"        — legacy comma-separated tokens

    Raises ValueError if residues span multiple chains or format unrecognised.
    """
    site_str = site_str.strip()
    resnums  = []
    chain    = None

    # ── New format: "CHAIN:spec,spec" where spec is "N" or "N-M" ──────────────
    m_colon = re.match(r"^([A-Za-z]):(.+)$", site_str)
    if m_colon:
        chain = m_colon.group(1).upper()
        for part in re.split(r",", m_colon.group(2)):
            part = part.strip()
            m_range = re.match(r"^(\d+)-(\d+)$", part)
            m_single = re.match(r"^(\d+)$", part)
            if m_range:
                resnums.extend(range(int(m_range.group(1)), int(m_range.group(2)) + 1))
            elif m_single:
                resnums.append(int(m_single.group(1)))
            else:
                raise ValueError(
                    f"Cannot parse residue spec '{part}' in '{site_str}'. "
                    f"Expected N or N-M (e.g. '8-30' or '120').")
    else:
        # ── Legacy format: "A10 A15 A22" or "A10,A15,A22" ────────────────────
        tokens = re.split(r"[\s,]+", site_str)
        for tok in tokens:
            if not tok:
                continue
            m = re.match(r"^([A-Za-z])(\d+)$", tok)
            if not m:
                raise ValueError(
                    f"Cannot parse '{tok}'. "
                    f"Use 'A:8-30,120-123' or legacy 'A10 A15 A22'.")
            c, r = m.group(1).upper(), int(m.group(2))
            if chain is None:
                chain = c
            elif c != chain:
                raise ValueError(
                    f"All hotspot residues must be on the same chain. "
                    f"Got '{c}' and '{chain}'.")
            resnums.append(r)

    if not chain or not resnums:
        raise ValueError("No residues parsed from site specification.")
    return chain, sorted(set(resnums))


def parse_length(length_str):
    """Parse "80-120" or "100" → (min_len, max_len)."""
    m = re.match(r"^(\d+)(?:-(\d+))?$", length_str.strip())
    if not m:
        raise ValueError(f"Cannot parse '{length_str}'. Expected: 80-120 or 100")
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return lo, hi


def parse_n_designs(n_str):
    """
    Parse per-tool design counts.
    "rfdiffusion=200,boltzgen=5000,bindcraft=100" → dict
    "100" → {tool: 100 for all tools}
    """
    if "=" not in n_str:
        n = int(n_str)
        return {t: n for t in ALL_TOOLS}
    result = {}
    for part in n_str.split(","):
        if "=" in part:
            tool, n = part.split("=", 1)
            result[tool.strip()] = int(n.strip())
    return result


def parse_score_weights(weights_str):
    """
    Parse "0.4,0.5,0.1" → (0.4, 0.5, 0.1)  or  "0.4,0.6" → (0.4, 0.6, 0.0).
    Three weights: pLDDT, iPTM, Rosetta dG (normalized).
    Backward compatible: 2 weights treated as (pLDDT, iPTM, 0.0).
    """
    parts = weights_str.split(",")
    if len(parts) == 2:
        return float(parts[0]), float(parts[1]), 0.0
    elif len(parts) == 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    else:
        raise ValueError(f"--score_weights must be 2 or 3 comma-separated floats, got: {weights_str!r}")


# ── Config file generators ─────────────────────────────────────────────────────

def write_boltzgen_yaml(target_path, chain_id, site_resnums, length_min, length_max, out_path):
    """Write BoltzGen design YAML."""
    # BoltzGen expects 1-based sequential indices, not PDB residue numbers.
    # Convert PDB resnums → 1-based indices by mapping through the chain's residue list.
    chain_info = get_chain_info(target_path)
    tchain = chain_info.get(chain_id)
    if tchain:
        resnum_to_idx = {r: i + 1 for i, r in enumerate(tchain["residues"])}
        site_indices = [resnum_to_idx[r] for r in site_resnums if r in resnum_to_idx]
    else:
        site_indices = site_resnums  # fallback
    binding_str = ",".join(str(r) for r in site_indices)
    lines = [
        "entities:",
        f"  # Binder ({length_min}–{length_max} residues, sequence generated by boltzgen)",
        "  - protein:",
        "      id: B",
        f"      sequence: {length_min}..{length_max}",
        "",
        "  # Target protein",
        "  - file:",
        f"      path: {Path(target_path).resolve()}",
        "      include:",
        "        - chain:",
        f"            id: {chain_id}",
        "      binding_types:",
        "        - chain:",
        f"            id: {chain_id}",
        f"            binding: {binding_str}",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")


def write_bindcraft_settings(target_path, chain_id, site_resnums, length_min, length_max,
                              n_designs, design_path, out_path):
    """Write BindCraft settings.json."""
    settings = {
        "design_path":             str(Path(design_path).resolve()) + "/",
        "binder_name":             "binder",
        "starting_pdb":            str(Path(target_path).resolve()),
        "chains":                  chain_id,
        "target_hotspot_residues": ",".join(str(r) for r in site_resnums),
        "lengths":                 [length_min, length_max],
        "number_of_final_designs": n_designs,
    }
    Path(out_path).write_text(json.dumps(settings, indent=2))


# ── Tool runners ───────────────────────────────────────────────────────────────

def run_rfdiffusion(target_path, chain_id, site_resnums, length_min, length_max,
                    n_designs, out_dir, dry_run=False):
    """
    RFdiffusion backbone generation + LigandMPNN sequence design.

    LigandMPNN FASTA header format (observed):
        >design_0, T=0.1, seed=42, num_res=247, ..., batch_size=1, ...  (first entry — template)
        >design_0, id=1, T=0.1, seed=42, overall_confidence=0.38, ligand_confidence=0.38, seq_rec=0.34
        SEQUENCE

    We extract overall_confidence and seq_rec from each sample header and keep
    the sample with the highest overall_confidence.

    Returns list of design dicts, each with keys:
        design_id, binder_sequence, backbone_pdb, native_score,
        native_score_name, ligandmpnn_seq_rec
    """
    if Path(target_path).suffix.lower() == ".cif":
        raise ValueError(
            f"RFdiffusion requires a PDB file, not CIF: {target_path}\n"
            f"  Use --target with a .pdb file. For KRAS4B: "
            f"--target /data/scripts/protein_folding/outputs/KRASB/8ECR_monomer.pdb"
        )

    out_dir = Path(out_dir)
    bb_dir  = out_dir / "backbones"
    sq_dir  = out_dir / "sequences"
    bb_dir.mkdir(parents=True, exist_ok=True)
    sq_dir.mkdir(parents=True, exist_ok=True)

    # Get target chain residue range for contig spec
    chain_info = get_chain_info(target_path)
    tchain = chain_info.get(chain_id)
    if tchain is None and not dry_run:
        raise RuntimeError(
            f"Chain {chain_id} not found in {target_path} "
            f"(available: {list(chain_info)})")
    target_len = tchain["length"] if tchain else 200  # fallback for dry_run
    # Use actual PDB residue numbers (not 1-N) for contig spec
    residues = tchain["residues"] if tchain else list(range(1, 201))
    res_start, res_end = residues[0], residues[-1]

    # RFdiffusion works best with 3-6 hotspot residues. If the user specified
    # more, subsample to ~5 evenly spaced residues, preferring those with
    # strong chemical identity (charged, aromatic) over featureless ones (G, A, S).
    MAX_HOTSPOTS = 6
    rf_site = list(site_resnums)
    if len(rf_site) > MAX_HOTSPOTS:
        # Score each residue by chemical identity
        # Build resnum → amino acid mapping from chain info
        res_to_aa = {}
        if tchain:
            seq = tchain.get("sequence", "")
            res_list = tchain.get("residues", [])
            for i, rn in enumerate(res_list):
                if i < len(seq):
                    res_to_aa[rn] = seq[i]

        # Priority: charged/aromatic > polar > hydrophobic > Gly/Ala
        PRIORITY = {
            "K": 3, "R": 3, "D": 3, "E": 3, "H": 3,  # charged
            "F": 3, "W": 3, "Y": 3,                     # aromatic
            "N": 2, "Q": 2, "S": 1, "T": 1,             # polar
            "L": 1, "I": 1, "V": 1, "M": 1, "P": 1,    # hydrophobic
            "G": 0, "A": 0, "C": 1,                      # featureless
        }

        # Pick ~5 residues with even spacing, prioritizing high-scoring ones
        n_pick = 5
        scored = [(r, PRIORITY.get(res_to_aa.get(r, "A"), 1)) for r in rf_site]

        # Step 1: include first and last for spatial coverage, but swap
        # for a nearby higher-priority residue if available (±2 positions)
        def _best_near(idx, exclude):
            best_r, best_p = rf_site[idx], PRIORITY.get(res_to_aa.get(rf_site[idx], "A"), 0)
            for offset in range(-2, 3):
                j = idx + offset
                if 0 <= j < len(rf_site) and rf_site[j] not in exclude:
                    p = PRIORITY.get(res_to_aa.get(rf_site[j], "A"), 0)
                    if p > best_p:
                        best_r, best_p = rf_site[j], p
            return best_r
        first = _best_near(0, set())
        last = _best_near(len(rf_site) - 1, {first})
        picked = {first, last}

        # Step 2: pick remaining from evenly spaced candidates, prefer high priority
        n_remaining = n_pick - len(picked)
        # Generate evenly spaced indices across the site
        step = len(rf_site) / (n_remaining + 1)
        candidate_indices = [int(round(step * (i + 1))) for i in range(n_remaining)]
        # At each position, consider a window of ±2 and pick highest priority
        for idx in candidate_indices:
            window = []
            for offset in range(-2, 3):
                j = idx + offset
                if 0 <= j < len(rf_site) and rf_site[j] not in picked:
                    window.append((PRIORITY.get(res_to_aa.get(rf_site[j], "A"), 1), rf_site[j]))
            if window:
                window.sort(reverse=True)  # highest priority first
                picked.add(window[0][1])

        rf_site = sorted(picked)
        label = [str(r) + "(" + res_to_aa.get(r, "?") + ")" for r in rf_site]
        log(f"  RFdiffusion: subsampled {len(site_resnums)} hotspots → {len(rf_site)}: {label}")

    hotspots = ",".join(f"{chain_id}{r}" for r in rf_site)
    contig   = f"[{chain_id}{res_start}-{res_end}/0 {length_min}-{length_max}]"

    log(f"RFdiffusion: target chain {chain_id} ({target_len} aa), "
        f"hotspots=[{hotspots}], binder {length_min}–{length_max} aa, n={n_designs}")

    rf_cmd = [
        "conda", "run", "--no-capture-output", "-n", "rfdiffusion",
        "python", f"{RFDIFFUSION_DIR}/scripts/run_inference.py",
        f"inference.input_pdb={Path(target_path).resolve()}",
        f"contigmap.contigs={contig}",
        f"ppi.hotspot_res=[{hotspots}]",
        f"inference.num_designs={n_designs}",
        f"inference.output_prefix={bb_dir}/design",
        f"inference.ckpt_override_path={RFDIFFUSION_CKPT}",
        # Reduce denoiser noise for PPI (README-recommended; reduces off-target designs)
        "denoiser.noise_scale_ca=0.5",
        "denoiser.noise_scale_frame=0.5",
    ]
    log("RFdiffusion: generating backbones...")
    run_cmd(rf_cmd, timeout=None, dry_run=dry_run)

    backbones = sorted(bb_dir.glob("design_*.pdb"))
    if not backbones and not dry_run:
        raise FileNotFoundError(f"No backbone PDBs in {bb_dir}")

    # LigandMPNN: pass all backbones at once via --pdb_path_multi JSON
    multi_json = out_dir / "backbone_list.json"
    multi_json.write_text(json.dumps({str(p): "" for p in backbones}))

    mpnn_cmd = [
        "conda", "run", "--no-capture-output", "-n", "mpnn",
        "python", f"{LIGANDMPNN_DIR}/run.py",
        "--model_type",              "protein_mpnn",
        "--checkpoint_protein_mpnn", LIGANDMPNN_CKPT,
        "--pdb_path_multi",          str(multi_json),
        "--out_folder",              str(sq_dir),
        "--number_of_batches",       "4",    # 4 seqs/backbone (RFdiffusion paper used 2; pick best by confidence)
        "--batch_size",              "1",
        "--temperature",             "0.1",
        "--seed",                    "42",
        "--save_stats",              "1",   # per-residue confidence to seqs/stats/
    ]
    log(f"LigandMPNN: designing sequences for {len(backbones)} backbones...")
    run_cmd(mpnn_cmd, timeout=3600, cwd=LIGANDMPNN_DIR, dry_run=dry_run)

    designs = []
    if not dry_run:
        for i, bb_pdb in enumerate(backbones):
            fasta_path = sq_dir / "seqs" / f"{bb_pdb.stem}.fa"
            if not fasta_path.exists():
                fastas = sorted((sq_dir / "seqs").glob(f"{bb_pdb.stem}*.fa"))
                if not fastas:
                    log(f"  Warning: no FASTA for {bb_pdb.stem}, skipping")
                    continue
                fasta_path = fastas[0]

            best_seq, best_conf, best_seq_rec = None, -1.0, float("nan")
            current_header = None
            for line in fasta_path.read_text().splitlines():
                if line.startswith(">"):
                    current_header = line
                elif current_header and line.strip():
                    # Only sample entries have overall_confidence (not the template header)
                    m_conf = re.search(r"overall_confidence=([0-9.]+)", current_header)
                    m_rec  = re.search(r"seq_rec=([0-9.]+)", current_header)
                    conf   = float(m_conf.group(1)) if m_conf else 0.0
                    seq_rec = float(m_rec.group(1)) if m_rec else float("nan")
                    if conf > best_conf:
                        best_conf    = conf
                        best_seq     = line.strip()
                        best_seq_rec = seq_rec
                    current_header = None

            if best_seq:
                # LigandMPNN FASTA has full complex sequence (target:binder) — extract binder only
                binder_seq = best_seq.split(":")[-1] if ":" in best_seq else best_seq
                designs.append({
                    "design_id":           f"rfdiffusion_{i:04d}",
                    "binder_sequence":     binder_seq,
                    "backbone_pdb":        str(bb_pdb),
                    "native_score":        best_conf,
                    "native_score_name":   "mpnn_confidence",
                    "ligandmpnn_seq_rec":  best_seq_rec,
                })
    else:
        designs = [{"design_id": "rfdiffusion_0000", "binder_sequence": "MOCK" * 25,
                    "backbone_pdb": str(bb_dir / "design_0.pdb"),
                    "native_score": 0.45, "native_score_name": "mpnn_confidence",
                    "ligandmpnn_seq_rec": 0.35}]

    log(f"RFdiffusion+LigandMPNN: {len(designs)} designs collected")
    return designs


def _load_boltzgen_metrics(raw_dir):
    """
    Load BoltzGen design metrics from the final_ranked_designs CSV.
    Returns dict: binder_sequence → {metric_key: value, ...} with bg_ prefix.

    Columns extracted (bg_ prefix applied):
        bg_design_to_target_iptm, bg_min_design_to_target_pae, bg_design_ptm,
        bg_complex_plddt, bg_complex_iplddt, bg_filter_rmsd, bg_delta_sasa_refolded,
        bg_plip_hbonds_refolded, bg_plip_saltbridge_refolded, bg_design_hydrophobicity,
        bg_design_largest_hydrophobic_patch_refolded, bg_liability_score,
        bg_liability_num_violations, bg_liability_high_severity_violations,
        bg_liability_violations_summary, bg_quality_score, bg_helix, bg_loop, bg_sheet
    """
    raw_dir = Path(raw_dir)

    # Prefer final_designs_metrics_*.csv; fall back to all_designs_metrics.csv
    candidates = sorted(raw_dir.glob("final_ranked_designs/final_designs_metrics_*.csv"))
    if not candidates:
        candidates = list(raw_dir.glob("final_ranked_designs/all_designs_metrics.csv"))
    if not candidates:
        return {}

    metrics_csv = candidates[0]

    # Column mapping: CSV column → bg_key
    COL_MAP = {
        "design_to_target_iptm":                     "bg_design_to_target_iptm",
        "min_design_to_target_pae":                  "bg_min_design_to_target_pae",
        "design_ptm":                                "bg_design_ptm",
        "complex_plddt":                             "bg_complex_plddt",
        "complex_iplddt":                            "bg_complex_iplddt",
        "filter_rmsd":                               "bg_filter_rmsd",
        "delta_sasa_refolded":                       "bg_delta_sasa_refolded",
        "plip_hbonds_refolded":                      "bg_plip_hbonds_refolded",
        "plip_saltbridge_refolded":                  "bg_plip_saltbridge_refolded",
        "design_hydrophobicity":                     "bg_design_hydrophobicity",
        "design_largest_hydrophobic_patch_refolded": "bg_design_largest_hydrophobic_patch_refolded",
        "liability_score":                           "bg_liability_score",
        "liability_num_violations":                  "bg_liability_num_violations",
        "liability_high_severity_violations":        "bg_liability_high_severity_violations",
        "liability_violations_summary":              "bg_liability_violations_summary",
        "quality_score":                             "bg_quality_score",
        "helix":                                     "bg_helix",
        "loop":                                      "bg_loop",
        "sheet":                                     "bg_sheet",
        # Binding site precision metrics
        "bindsite_under_3rmsd":                      "bg_bindsite_under_3rmsd",
        "bindsite_under_4rmsd":                      "bg_bindsite_under_4rmsd",
        "native_rmsd_refolded":                      "bg_native_rmsd_refolded",
        # SASA metrics
        "design_sasa_bound_refolded":                "bg_design_sasa_bound_refolded",
        "design_sasa_unbound_refolded":              "bg_design_sasa_unbound_refolded",
        # Filter pass/fail
        "pass_filters":                              "bg_pass_filters",
        "pass_filter_rmsd_filter":                   "bg_pass_filter_rmsd_filter",
        # Ranking percentiles
        "rank_design_to_target_iptm":                "bg_rank_design_to_target_iptm",
        "rank_design_ptm":                           "bg_rank_design_ptm",
    }

    result = {}
    try:
        with open(metrics_csv) as f:
            for row in csv.DictReader(f):
                seq = row.get("designed_sequence") or row.get("sequence") or ""
                if not seq:
                    continue
                entry = {}
                for csv_col, bg_key in COL_MAP.items():
                    raw = row.get(csv_col, "")
                    if bg_key == "bg_liability_violations_summary":
                        entry[bg_key] = raw  # keep as string
                    else:
                        entry[bg_key] = _safe_float(raw)
                result[seq] = entry
    except Exception as e:
        log(f"  Warning: could not load BoltzGen metrics CSV {metrics_csv}: {e}")

    return result


def run_boltzgen(target_path, chain_id, site_resnums, length_min, length_max,
                 n_designs, out_dir, dry_run=False, ss_bias="balanced"):
    """
    BoltzGen binder design pipeline (runs in dedicated boltzgen conda env).

    After collecting CIFs from final_ranked_designs, loads the BoltzGen
    metrics CSV and merges bg_* quality columns into each design dict.

    Returns list of design dicts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "design.yaml"
    raw_dir   = out_dir / "boltzgen_raw"

    write_boltzgen_yaml(target_path, chain_id, site_resnums, length_min, length_max, yaml_path)
    log(f"BoltzGen: design YAML written → {yaml_path}")

    cmd = [
        BOLTZGEN_BIN, "run", str(yaml_path),
        "--output",                      str(raw_dir) + "/",
        "--protocol",                    "protein-anything",
        "--num_designs",                 str(n_designs),
        "--budget",                      str(min(n_designs, 500)),  # keep top-500 diverse designs (default=30 discards too many)
        "--alpha",                       "0.1",  # 10% diversity weight (default 0.001 is near-zero diversity)
        "--diffusion_batch_size",        "5",    # safe for 20GB GPU (RTX 4000 Ada)
        "--inverse_fold_num_sequences",  "2",    # 2 sequences per backbone for diversity
        "--reuse",                               # resume if interrupted mid-run
        "--config", "design", "sampling_steps=200",
        "--config", "design", "compile_pairformer=true", "compile_structure=true",
    ]
    # SS bias: filter by helix/sheet fraction
    if ss_bias == "beta":
        cmd.extend(["--additional_filters", "helix<0.4"])
    elif ss_bias == "helix":
        cmd.extend(["--additional_filters", "sheet<0.2"])
    log(f"BoltzGen: running {n_designs} designs (may take hours)...")
    run_cmd(cmd, timeout=None, dry_run=dry_run,
            extra_env={"MKL_THREADING_LAYER": "GNU", "PYTHONNOUSERSITE": "1"})

    designs = []
    if not dry_run:
        # Collect ranked CIFs from final_ranked_designs
        final_dirs = sorted(raw_dir.glob("final_ranked_designs/final_*_designs"))
        if not final_dirs:
            final_dirs = sorted(raw_dir.glob("intermediate_designs_inverse_folded/refold_cif"))
            if final_dirs:
                log("  WARNING: using unranked intermediate designs (filter step incomplete).")
                log("  Run is likely still in progress or was interrupted before final ranking.")

        cifs = []
        for d in final_dirs:
            cifs.extend(sorted(d.glob("*.cif")))

        # Load iPTM scores from all_designs_metrics.csv (for native_score field)
        iptm_scores = {}
        metrics_csv = next(raw_dir.glob("final_ranked_designs/all_designs_metrics.csv"), None)
        if metrics_csv:
            with open(metrics_csv) as f:
                for row in csv.DictReader(f):
                    name = row.get("name", "")
                    raw_iptm = row.get("iptm") or row.get("iPTM") or row.get("design_to_target_iptm") or ""
                    try:
                        iptm_scores[name] = float(raw_iptm)
                    except (ValueError, TypeError):
                        pass

        # Load full BoltzGen metrics keyed by binder sequence
        bg_metrics = _load_boltzgen_metrics(raw_dir)
        log(f"  BoltzGen: loaded metrics for {len(bg_metrics)} sequences from CSV")

        for i, cif in enumerate(cifs):
            chain_info = extract_cif_chain_info(cif)
            # BoltzGen CIFs place the binder in the shorter chain regardless of YAML entity IDs
            # (typically chain A=binder, chain B=target, opposite of the input YAML convention)
            binder_info = (min(chain_info.values(), key=lambda v: v["length"])
                           if chain_info else {})
            seq = binder_info.get("sequence", "")
            if not seq:
                continue
            d = {
                "design_id":         f"boltzgen_{i:04d}",
                "binder_sequence":   seq,
                "complex_cif":       str(cif),
                "native_score":      iptm_scores.get(cif.stem, 0.0),
                "native_score_name": "boltzgen_iptm",
            }
            # Merge bg_* metrics from CSV (matched by binder sequence)
            if seq in bg_metrics:
                d.update(bg_metrics[seq])
            designs.append(d)
    else:
        designs = [{"design_id": "boltzgen_0000", "binder_sequence": "MOCK" * 25,
                    "complex_cif": str(out_dir / "mock.cif"),
                    "native_score": 0.55, "native_score_name": "boltzgen_iptm"}]

    log(f"BoltzGen: {len(designs)} designs collected")
    return designs


def strip_pdb_insertion_codes(src_path, dst_path):
    """
    Write a copy of src_path with insertion codes (PDB col 27) removed.
    Replaces any non-space character in the iCode column with a space.
    Needed for BindCraft/ColabDesign which rejects PDBs with insertion codes.
    """
    lines = Path(src_path).read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 27 and line[26] != " ":
            line = line[:26] + " " + line[27:]
        out.append(line)
    Path(dst_path).write_text("\n".join(out) + "\n")


def _load_bindcraft_metrics(design_path):
    """
    Load BindCraft final_design_stats.csv and return dict:
        pdb_stem_substring → {bc_key: value, ...}

    BindCraft column → bc_ key mapping covers numeric AF2/Rosetta scores.
    All values are stored as floats; non-numeric fields are stored as strings.
    """
    final_csv = Path(design_path) / "final_design_stats.csv"
    if not final_csv.exists():
        return {}

    # Numeric columns to extract (use Average_ prefix versions where available)
    NUMERIC_COLS = [
        "Average_pLDDT", "Average_pTM", "Average_i_pTM", "Average_pAE",
        "Average_i_pAE", "Average_i_pLDDT", "Average_ss_pLDDT",
        "Average_Binder_Energy_Score", "Average_Surface_Hydrophobicity",
        "Average_ShapeComplementarity", "Average_PackStat",
        "Average_dG", "Average_dSASA", "Average_dG/dSASA",
        "Average_Interface_Hydrophobicity", "Average_Interface_SASA_%",
        "Average_n_InterfaceResidues", "Average_n_InterfaceHbonds",
        "Average_InterfaceHbondsPercentage", "Average_n_InterfaceUnsatHbonds",
        "Average_Binder_Helix%", "Average_Binder_BetaSheet%", "Average_Binder_Loop%",
        "Average_Interface_Helix%", "Average_Interface_BetaSheet%", "Average_Interface_Loop%",
        "Average_Hotspot_RMSD", "Average_Target_RMSD", "Average_Binder_RMSD",
        "Average_Binder_pLDDT", "Average_Binder_pTM", "Average_Binder_pAE",
        "Average_Unrelaxed_Clashes", "Average_Relaxed_Clashes",
        "MPNN_score", "MPNN_seq_recovery",
        "Length", "Helicity",
    ]
    # bc_ key = lowercase of column name after Average_ prefix stripped
    def _bc_key(col):
        stem = col[len("Average_"):] if col.startswith("Average_") else col
        return "bc_" + stem.lower().replace("%", "_pct").replace("/", "_per_")

    result = {}
    try:
        with open(final_csv) as f:
            for row in csv.DictReader(f):
                name = row.get("Design", "")
                if not name:
                    continue
                entry = {}
                for col in NUMERIC_COLS:
                    if col in row:
                        entry[_bc_key(col)] = _safe_float(row[col])
                result[name] = entry
    except Exception as e:
        log(f"  Warning: could not load BindCraft metrics CSV {final_csv}: {e}")

    return result


def run_bindcraft(target_path, chain_id, site_resnums, length_min, length_max,
                  n_designs, out_dir, dry_run=False, filters_path=None,
                  advanced_path=None, ss_bias="balanced"):
    """
    BindCraft binder design (JAX/AF2).

    After collecting accepted PDBs, parses final_design_stats.csv and merges
    bc_* quality columns (AF2 pLDDT, i_pTM, Rosetta dG, etc.) into each design.

    Returns list of design dicts.
    """
    if Path(target_path).suffix.lower() == ".cif":
        raise ValueError(
            f"BindCraft requires a PDB file, not CIF: {target_path}\n"
            f"  Use --target with a .pdb file. For KRAS4B: "
            f"--target /data/scripts/protein_folding/outputs/KRASB/8ECR_monomer.pdb"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure BindCraft bundled binaries are executable (git clone loses +x bits)
    for _bin in ("functions/dssp", "functions/DAlphaBall.gcc"):
        _bin_path = Path(BINDCRAFT_DIR) / _bin
        if _bin_path.exists() and not os.access(_bin_path, os.X_OK):
            _bin_path.chmod(_bin_path.stat().st_mode | 0o755)
            log(f"BindCraft: fixed execute permission on {_bin_path.name}")

    # BindCraft/ColabDesign rejects PDBs with insertion codes — strip them first
    clean_pdb = out_dir / f"{Path(target_path).stem}_clean.pdb"
    strip_pdb_insertion_codes(target_path, clean_pdb)
    target_path = clean_pdb
    log(f"BindCraft: stripped insertion codes → {clean_pdb}")

    design_path   = out_dir / "bindcraft_output"
    settings_path = out_dir / "settings.json"

    write_bindcraft_settings(target_path, chain_id, site_resnums, length_min, length_max,
                              n_designs, design_path, settings_path)
    log(f"BindCraft: settings.json written → {settings_path}")

    # Patch advanced settings for SS bias
    effective_advanced = advanced_path or f"{BINDCRAFT_DIR}/settings_advanced/default_4stage_multimer_flexible.json"
    if ss_bias != "balanced":
        patched = Path(out_dir) / "advanced_ss.json"
        with open(effective_advanced) as f:
            adv = json.load(f)
        adv["weights_helicity"] = SS_BIAS_PARAMS[ss_bias]["bc_helicity"]
        with open(patched, "w") as f:
            json.dump(adv, f, indent=2)
        effective_advanced = str(patched)
        log(f"BindCraft: patched weights_helicity={SS_BIAS_PARAMS[ss_bias]['bc_helicity']} → {patched}")

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "BindCraft",
        "python", f"{BINDCRAFT_DIR}/bindcraft.py",
        "--settings", str(settings_path),
        "--filters",  filters_path or f"{BINDCRAFT_DIR}/settings_filters/no_filters.json",
        "--advanced", str(effective_advanced),
    ]
    log(f"BindCraft: running until {n_designs} designs accepted...")
    run_cmd(cmd, timeout=None, cwd=BINDCRAFT_DIR, dry_run=dry_run)

    designs = []
    if not dry_run:
        ranked_dir   = design_path / "Accepted" / "Ranked"
        accepted_dir = design_path / "Accepted"
        pdbs = sorted(ranked_dir.glob("*.pdb")) if ranked_dir.exists() else []
        if not pdbs:
            pdbs = sorted(accepted_dir.glob("*.pdb")) if accepted_dir.exists() else []

        # Load iptm scores for native_score field
        iptm_scores = {}
        final_csv = design_path / "final_design_stats.csv"
        if final_csv.exists():
            with open(final_csv) as f:
                for row in csv.DictReader(f):
                    name = row.get("Design", "")
                    raw_iptm = row.get("Average_i_pTM") or ""
                    try:
                        iptm_scores[name] = float(raw_iptm)
                    except (ValueError, TypeError):
                        pass

        # Load full bc_* metrics keyed by design name
        bc_metrics = _load_bindcraft_metrics(design_path)
        log(f"  BindCraft: loaded metrics for {len(bc_metrics)} designs from CSV")

        for i, pdb in enumerate(pdbs):
            chain_info = extract_pdb_chain_info(pdb)
            # BindCraft hard-codes binder as chain B (bindcraft.py: binder_chain = "B")
            binder_info = chain_info.get("B") or (next(iter(chain_info.values())) if chain_info else {})
            seq = binder_info.get("sequence", "")
            if not seq:
                continue
            # Match iptm score: exact match first, then CSV-key-in-pdb-stem (never pdb.stem-in-key
            # which would let short names like "design_0" match "binder_design_0_model1" wrongly)
            native_score = (iptm_scores.get(pdb.stem)
                            or next((v for k, v in iptm_scores.items() if k in pdb.stem), 0.0))
            d = {
                "design_id":         f"bindcraft_{i:04d}",
                "binder_sequence":   seq,
                "binder_pdb":        str(pdb),
                "native_score":      native_score,
                "native_score_name": "bindcraft_iptm",
            }
            # Merge bc_* metrics from CSV: exact match first, then CSV-key-in-pdb-stem
            matched_key = (pdb.stem if pdb.stem in bc_metrics
                           else next((k for k in bc_metrics if k in pdb.stem), None))
            if matched_key:
                d.update(bc_metrics[matched_key])
            designs.append(d)
    else:
        designs = [{"design_id": "bindcraft_0000", "binder_sequence": "MOCK" * 25,
                    "binder_pdb": str(out_dir / "mock.pdb"),
                    "native_score": 0.50, "native_score_name": "bindcraft_iptm"}]

    log(f"BindCraft: {len(designs)} designs collected")
    return designs


# ── PXDesign ──────────────────────────────────────────────────────────────────

def _fix_cif_entity_ids(cif_path):
    """Fix gemmi-generated CIF to use numeric entity IDs (Protenix requirement).

    gemmi often writes entity_id as chain letter ('A') and label_asym_id
    as chain+'xp' ('Axp').  Protenix/PXDesign expects numeric entity IDs
    (1, 2, …) and simple label_asym_id matching the chain name.
    """
    import re
    with open(cif_path, "r") as f:
        lines = f.readlines()

    # Build chain→numeric mapping from _entity.id rows
    chain_to_num = {}
    next_id = 1
    new_lines = []
    for line in lines:
        # Fix _entity.id <chain> → _entity.id <num>
        m = re.match(r"^(_entity\.id\s+)(\S+)", line)
        if m:
            chain = m.group(2)
            if not chain.isdigit():
                chain_to_num[chain] = str(next_id)
                line = f"{m.group(1)}{next_id}\n"
                next_id += 1

        # Fix _entity_poly.entity_id, _struct_asym.entity_id
        for tag in ("_entity_poly.entity_id", "_struct_asym.entity_id"):
            m = re.match(rf"^({re.escape(tag)}\s+)(\S+)", line)
            if m and m.group(2) in chain_to_num:
                line = f"{m.group(1)}{chain_to_num[m.group(2)]}\n"

        # Fix _struct_asym.id  Axp → A  (strip 'xp' suffix)
        m = re.match(r"^(_struct_asym\.id\s+)(\S+)", line)
        if m and m.group(2).endswith("xp"):
            line = f"{m.group(1)}{m.group(2)[:-2]}\n"

        # Detect loop_ block for _entity or _struct_asym (multi-row)
        # Fix entity loop rows: <chain> polymer → <num> polymer
        if re.match(r"^[A-Z]\s+polymer", line):
            parts = line.split()
            if parts[0] in chain_to_num:
                parts[0] = chain_to_num[parts[0]]
                line = " ".join(parts) + "\n"

        # Fix struct_asym loop rows: Axp <chain> → A <num>
        m = re.match(r"^(\S+xp)\s+([A-Z])\s*$", line)
        if m and m.group(2) in chain_to_num:
            asym = m.group(1)[:-2]  # strip xp
            line = f"{asym} {chain_to_num[m.group(2)]}\n"

        # Fix ATOM/HETATM lines
        if line.startswith(("ATOM ", "HETATM ")):
            parts = line.split()
            # label_asym_id is column index 6 (0-based), label_entity_id is 7
            if len(parts) >= 8:
                if parts[6].endswith("xp"):
                    parts[6] = parts[6][:-2]
                if parts[7] in chain_to_num:
                    parts[7] = chain_to_num[parts[7]]
                line = " ".join(parts) + "\n"

        new_lines.append(line)

    with open(cif_path, "w") as f:
        f.writelines(new_lines)


def _convert_pdb_to_cif(pdb_path, cif_path):
    """Convert PDB to CIF using gemmi. Falls back to simple rename if gemmi unavailable."""
    try:
        run_cmd(["gemmi", "convert", str(pdb_path), str(cif_path)])
        _fix_cif_entity_ids(str(cif_path))
        return str(cif_path)
    except Exception:
        # Try with python gemmi module
        try:
            script = (
                f"import gemmi; "
                f"st = gemmi.read_structure('{pdb_path}'); "
                f"st.make_mmcif_document().write_file('{cif_path}')"
            )
            run_cmd(["python3", "-c", script])
            _fix_cif_entity_ids(str(cif_path))
            return str(cif_path)
        except Exception:
            log(f"  Warning: gemmi not available, using PDB directly (PXDesign will auto-convert)")
            return str(pdb_path)


def write_pxdesign_yaml(target_path, chain_id, site_resnums, length_min, length_max,
                        out_path):
    """
    Generate PXDesign input YAML from target and site specification.

    PXDesign expects:
      binder_length: int
      target:
        file: "path.cif"
        chains:
          A:
            crop: ["11-17", "119-124"]
            hotspots: [11, 12, 13, ...]

    The crop defines what the model sees of the target. For targets ≤300
    residues, the full target is used (no crop). For larger targets, we
    expand the hotspot by ±20 residues in sequence and include all residues
    with CA within 10 Å of the binding site to capture 3D context.
    """
    # Get all target residues
    chain_info = get_chain_info(target_path)
    tchain = chain_info.get(chain_id, {})
    all_residues = tchain.get("residues", [])
    target_len = tchain.get("length", 0)

    # For targets ≤300 residues, use full target (no crop needed)
    if target_len <= 300 and all_residues:
        crop_resnums = set(all_residues)
    else:
        # Expand crop: hotspot ±20 residues in sequence + 10 Å spatial shell
        crop_resnums = set(site_resnums)
        if all_residues:
            site_set = set(site_resnums)
            res_list = sorted(all_residues)
            for r in site_resnums:
                try:
                    idx = res_list.index(r)
                except ValueError:
                    continue
                for offset in range(-20, 21):
                    j = idx + offset
                    if 0 <= j < len(res_list):
                        crop_resnums.add(res_list[j])

            # Also add spatially nearby residues (CA within 10 Å of any site CA)
            try:
                site_cas = {}
                all_cas = {}
                for line in Path(target_path).read_text().splitlines():
                    if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == chain_id:
                        rn = int(line[22:26].strip())
                        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                        all_cas[rn] = xyz
                        if rn in site_set:
                            site_cas[rn] = xyz
                for rn, xyz in all_cas.items():
                    for s_xyz in site_cas.values():
                        dist = sum((a - b) ** 2 for a, b in zip(xyz, s_xyz)) ** 0.5
                        if dist < 10.0:
                            crop_resnums.add(rn)
                            break
            except Exception:
                pass  # fall back to sequence-only expansion

    # Convert crop resnums to ranges (consecutive residues grouped)
    crop_sorted = sorted(crop_resnums)
    crops = []
    if crop_sorted:
        start = crop_sorted[0]
        end = crop_sorted[0]
        for r in crop_sorted[1:]:
            if r == end + 1:
                end = r
            else:
                crops.append(f"{start}-{end}" if start != end else str(start))
                start = end = r
        crops.append(f"{start}-{end}" if start != end else str(start))

    # Use midpoint of length range
    binder_length = (length_min + length_max) // 2

    # Write YAML manually (avoids PyYAML dependency in host env)
    crop_str = ", ".join(f'"{c}"' for c in crops)
    hotspot_str = ", ".join(str(r) for r in site_resnums)

    yaml_content = (
        f"binder_length: {binder_length}\n"
        f"\n"
        f"target:\n"
        f"  file: \"{target_path}\"\n"
        f"  chains:\n"
        f"    {chain_id}:\n"
        f"      crop: [{crop_str}]\n"
        f"      hotspots: [{hotspot_str}]\n"
    )

    Path(out_path).write_text(yaml_content)
    return out_path


def run_pxdesign(target_path, chain_id, site_resnums, length_min, length_max,
                 n_designs, out_dir, dry_run=False, **kwargs):
    """
    PXDesign binder design (DiT diffusion + MPNN + AF2-IG + Protenix filtering).

    PXDesign runs its own internal multi-model filtering pipeline. Designs
    that reach summary.csv are already pre-validated by AF2 and Protenix.

    Returns list of design dicts with px_* native metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PXDesign handles PDB→CIF conversion internally with proper entity IDs.
    # External gemmi conversion produces CIFs missing _entity_poly_seq which
    # Protenix's parser requires. So always pass PDB directly.
    target_for_px = target_path

    # Write input YAML
    yaml_path = out_dir / "design.yaml"
    write_pxdesign_yaml(target_for_px, chain_id, site_resnums,
                        length_min, length_max, yaml_path)
    log(f"PXDesign: config written → {yaml_path}")

    # Choose preset based on design count
    preset = "preview" if n_designs <= 30 else "extended"

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "pxdesign",
        "pxdesign", "pipeline",
        "--preset", preset,
        "-i", str(yaml_path),
        "-o", str(out_dir / "pxdesign_output"),
        "--N_sample", str(n_designs),
        "--dtype", "bf16",
    ]
    log(f"PXDesign: running {n_designs} designs (preset={preset})...")
    px_env = {
        "LAYERNORM_TYPE": "torch",  # Protenix: use PyTorch LayerNorm (avoid CUDA JIT)
        "PYTHONNOUSERSITE": "1",    # isolate from ~/.local/ packages
    }
    run_cmd(cmd, timeout=86400, extra_env=px_env, dry_run=dry_run)

    designs = []
    if not dry_run:
        # Parse summary.csv for design IDs, sequences, and native scores
        px_output = out_dir / "pxdesign_output"
        summary_csv = None
        # PXDesign outputs to design_outputs/<task_name>/summary.csv
        for candidate in [
            px_output / "summary.csv",
            *px_output.glob("*/summary.csv"),
            *px_output.glob("design_outputs/*/summary.csv"),
        ]:
            if candidate.exists():
                summary_csv = candidate
                break

        if summary_csv:
            log(f"  PXDesign: parsing {summary_csv}")
            with open(summary_csv) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Extract sequence
                    seq = row.get("sequence", "")
                    if not seq:
                        continue

                    # PXDesign summary.csv columns: rank, af2_iptm, af2_ipAE,
                    # af2_plddt, af2_ptm, af2_complex_pred_design_rmsd,
                    # ptx_iptm_binder, ptx_plddt, ptx_ptm, ptx_pred_design_rmsd,
                    # AF2-IG-easy-success, Protenix-success, etc.
                    rank = row.get("rank", str(i + 1))
                    iptm = _safe_float(row.get("ptx_iptm_binder", "") or row.get("af2_iptm", ""))
                    ipae = _safe_float(row.get("af2_ipAE", ""))
                    plddt = _safe_float(row.get("ptx_plddt", "") or row.get("af2_plddt", ""))
                    ptm = _safe_float(row.get("ptx_ptm", "") or row.get("af2_ptm", ""))
                    rmsd = _safe_float(row.get("ptx_pred_design_rmsd", "") or
                                       row.get("af2_complex_pred_design_rmsd", ""))
                    cluster_id = row.get("cluster_id", "")

                    # Determine filter level from CSV boolean columns
                    filter_level = "none"
                    if row.get("Protenix-success", "").strip().lower() == "true":
                        filter_level = "Protenix"
                    elif row.get("Protenix-basic-success", "").strip().lower() == "true":
                        filter_level = "Protenix-basic"
                    elif row.get("AF2-IG-success", "").strip().lower() == "true":
                        filter_level = "AF2-IG"
                    elif row.get("AF2-IG-easy-success", "").strip().lower() == "true":
                        filter_level = "AF2-IG-easy"

                    # Find the structure file — prefer chosen_struct_path column,
                    # fall back to rank_N.cif in standard directories
                    struct_file = None
                    chosen_path = row.get("chosen_struct_path", "")
                    if chosen_path:
                        candidate = summary_csv.parent / chosen_path
                        if candidate.exists():
                            struct_file = str(candidate)
                    if not struct_file:
                        for search_dir in ["passing-Protenix", "passing-Protenix-basic",
                                           "passing-AF2-IG", "passing-AF2-IG-easy",
                                           "orig_designed"]:
                            search_path = summary_csv.parent / search_dir
                            if search_path.exists():
                                rank_file = search_path / f"rank_{rank}.cif"
                                if rank_file.exists():
                                    struct_file = str(rank_file)
                                    break

                    d = {
                        "design_id":         f"pxdesign_{i:04d}",
                        "binder_sequence":   seq,
                        "native_score":      iptm if iptm == iptm else 0.0,
                        "native_score_name": "pxdesign_iptm",
                        "px_iptm":           iptm,
                        "px_ipae":           ipae,
                        "px_plddt":          plddt,
                        "px_ptm":            ptm,
                        "px_rmsd":           rmsd,
                        "px_cluster_id":     cluster_id,
                        "px_filter_level":   filter_level,
                    }
                    if struct_file:
                        d["binder_pdb"] = struct_file
                    designs.append(d)
        else:
            log("  PXDesign: WARNING — no summary.csv found in output directory")
            # Try to find any CIF/PDB structures in output
            for cif in sorted(px_output.rglob("*.cif"))[:n_designs]:
                seq = ""
                # Try to extract sequence from CIF
                try:
                    ci = extract_cif_chain_info(cif)
                    # Get the binder chain (not the target)
                    for cid, info in ci.items():
                        if cid != chain_id:
                            seq = info.get("sequence", "")
                            break
                except Exception:
                    pass
                if seq:
                    designs.append({
                        "design_id":         f"pxdesign_{len(designs):04d}",
                        "binder_sequence":   seq,
                        "binder_pdb":        str(cif),
                        "native_score":      0.0,
                        "native_score_name": "pxdesign_iptm",
                    })
    else:
        designs = [{"design_id": "pxdesign_0000", "binder_sequence": "MOCK" * 20,
                    "binder_pdb": str(out_dir / "mock.cif"),
                    "native_score": 0.60, "native_score_name": "pxdesign_iptm"}]

    log(f"PXDesign: {len(designs)} designs collected")
    return designs


def run_proteina(target_path, chain_id, site_resnums, length_min, length_max,
                 n_designs, out_dir, dry_run=False, **kwargs):
    """
    Proteina flow-based backbone generation + LigandMPNN sequence design.

    Proteina (arxiv:2503.00710, NVIDIA) is a flow-based protein backbone
    generator that uses flow matching to produce highly designable backbones.
    Unlike RFdiffusion, Proteina generates standalone backbone structures
    (not target-bound complexes). The generated backbones are sequenced by
    LigandMPNN and validated downstream by ESMFold + Boltz-2 (with pocket
    constraint) to identify sequences that bind the target.

    Pipeline: Proteina backbones → LigandMPNN sequences → ESMFold/Boltz filter

    Returns list of design dicts with keys:
        design_id, binder_sequence, backbone_pdb, native_score,
        native_score_name, ligandmpnn_seq_rec
    """
    out_dir = Path(out_dir)
    bb_dir  = out_dir / "backbones"
    sq_dir  = out_dir / "sequences"
    bb_dir.mkdir(parents=True, exist_ok=True)
    sq_dir.mkdir(parents=True, exist_ok=True)

    log(f"Proteina: binder {length_min}–{length_max} aa, n={n_designs}")

    # When --device is set, CUDA_VISIBLE_DEVICES restricts to one GPU.
    # Inside the process that GPU always appears as cuda:0.
    # Only auto-detect when no --device flag was given.
    if GPU_ENV.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_device = "cuda:0"
    else:
        gpu_device = "cuda:0"
        try:
            import subprocess as _sp
            nvsmi = _sp.run(["nvidia-smi", "--query-gpu=index,memory.used",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5)
            if nvsmi.returncode == 0:
                gpu_usage = []
                for line in nvsmi.stdout.strip().split("\n"):
                    idx, mem = line.split(",")
                    gpu_usage.append((int(idx.strip()), int(mem.strip())))
                best = min(gpu_usage, key=lambda x: x[1])
                gpu_device = f"cuda:{best[0]}"
                log(f"  Proteina: using {gpu_device} ({best[1]} MiB used)")
        except Exception:
            pass

    # Proteina generate_backbones.py — our thin wrapper around Proteina's API
    # Generates unconditional backbone PDBs of specified lengths
    proteina_python = os.environ.get("PROTEINA_PYTHON", f"{_SOFTWARE_DIR}/envs/proteina_env/bin/python")
    proteina_cmd = [
        proteina_python, f"{PROTEINA_DIR}/generate_backbones.py",
        "--ckpt", PROTEINA_CKPT,
        "--output_dir", str(bb_dir),
        "--n_designs", str(n_designs),
        "--length_min", str(length_min),
        "--length_max", str(length_max),
        "--noise_scale", "0.45",
        "--seed", "42",
        "--device", gpu_device,
        "--max_batch", "5",
    ]
    proteina_env = {
        "DATA_PATH": os.path.dirname(PROTEINA_CKPT),
        "PYTHONNOUSERSITE": "1",  # isolate from ~/.local/ packages
    }
    log("Proteina: generating backbones...")
    run_cmd(proteina_cmd, timeout=None, extra_env=proteina_env, dry_run=dry_run)

    backbones = sorted(bb_dir.glob("design_*.pdb"))
    if not backbones and not dry_run:
        raise FileNotFoundError(f"No backbone PDBs in {bb_dir}")

    # Proteina outputs CA-only PDBs. Use Proteina's bundled ProteinMPNN CA model
    # (not LigandMPNN which requires full N/CA/C/O backbone).
    proteina_mpnn_script = f"{PROTEINA_DIR}/ProteinMPNN/protein_mpnn_run.py"
    for bb_pdb in backbones:
        mpnn_cmd = [
            "conda", "run", "--no-capture-output", "-n", "mpnn",
            "python", proteina_mpnn_script,
            "--ca_only",
            "--pdb_path",          str(bb_pdb),
            "--out_folder",        str(sq_dir),
            "--num_seq_per_target", "4",
            "--sampling_temp",     "0.1",
            "--seed",              "42",
            "--batch_size",        "1",
        ]
        if not dry_run:
            run_cmd(mpnn_cmd, timeout=300, cwd=PROTEINA_DIR, dry_run=dry_run)
    log(f"ProteinMPNN (CA): designed sequences for {len(backbones)} Proteina backbones")

    designs = []
    if not dry_run:
        for i, bb_pdb in enumerate(backbones):
            fasta_path = sq_dir / "seqs" / f"{bb_pdb.stem}.fa"
            if not fasta_path.exists():
                fastas = sorted((sq_dir / "seqs").glob(f"{bb_pdb.stem}*.fa"))
                if not fastas:
                    log(f"  Warning: no FASTA for {bb_pdb.stem}, skipping")
                    continue
                fasta_path = fastas[0]

            best_seq, best_conf, best_seq_rec = None, float("inf"), float("nan")
            current_header = None
            for line in fasta_path.read_text().splitlines():
                if line.startswith(">"):
                    current_header = line
                elif current_header and line.strip():
                    # Skip the native/wildtype sequence (first entry) — it has no
                    # T= prefix and is all-A for CA-only backbones
                    if "sample=" not in current_header:
                        current_header = None
                        continue
                    # CA-only ProteinMPNN uses score=/global_score= and seq_recovery=
                    # (LigandMPNN uses overall_confidence= and seq_rec=)
                    m_conf = re.search(r"(?:overall_confidence|score)=([0-9.]+)", current_header)
                    m_rec  = re.search(r"(?:seq_rec|seq_recovery)=([0-9.]+)", current_header)
                    # ProteinMPNN score = negative log-likelihood (lower = better)
                    conf   = float(m_conf.group(1)) if m_conf else float("inf")
                    seq_rec = float(m_rec.group(1)) if m_rec else float("nan")
                    if conf < best_conf:
                        best_conf    = conf
                        best_seq     = line.strip()
                        best_seq_rec = seq_rec
                    current_header = None

            if best_seq:
                binder_seq = best_seq.split(":")[-1] if ":" in best_seq else best_seq
                designs.append({
                    "design_id":           f"proteina_{i:04d}",
                    "binder_sequence":     binder_seq,
                    "backbone_pdb":        str(bb_pdb),
                    "native_score":        best_conf,
                    "native_score_name":   "mpnn_score",
                    "ligandmpnn_seq_rec":  best_seq_rec,
                })
    else:
        designs = [{"design_id": "proteina_0000", "binder_sequence": "MOCK" * 25,
                    "backbone_pdb": str(bb_dir / "design_0.pdb"),
                    "native_score": 0.45, "native_score_name": "mpnn_confidence",
                    "ligandmpnn_seq_rec": 0.35}]

    log(f"Proteina+LigandMPNN: {len(designs)} designs collected")
    return designs


# ── Proteina Complexa ─────────────────────────────────────────────────────────

def _inject_complexa_target(local_yaml, complexa_targets_yaml):
    """
    Inject our binder_target entry into Complexa's targets_dict.yaml.

    Reads the local YAML we generated, extracts the binder_target block,
    and appends/replaces it in the Complexa repo's config file.
    """
    local_content = Path(local_yaml).read_text()
    complexa_path = Path(complexa_targets_yaml)

    # Read existing Complexa targets
    existing = complexa_path.read_text() if complexa_path.exists() else "target_dict_cfg:\n"

    # Remove any previous binder_target entry
    lines = existing.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        if line.strip().startswith("binder_target:"):
            skip = True
            continue
        if skip and (line.startswith("    ") or line.startswith("\t\t")):
            continue  # skip indented content of binder_target
        skip = False
        cleaned.append(line)

    # Extract binder_target block from local YAML (everything after "target_dict_cfg:")
    local_lines = local_content.splitlines()
    target_block = []
    in_block = False
    for line in local_lines:
        if "binder_target:" in line:
            in_block = True
        if in_block:
            target_block.append(line)

    # Append to cleaned content
    result = "\n".join(cleaned).rstrip() + "\n\n" + "\n".join(target_block) + "\n"
    complexa_path.write_text(result)


def write_proteina_complexa_target(target_path, chain_id, site_resnums, length_min, length_max, out_dir):
    """Write a targets_dict.yaml for Proteina Complexa binder design.

    The YAML follows the Complexa target_dict_cfg format:
        target_dict_cfg:
          binder_target:
            target_path: /absolute/path/to/target.pdb
            target_input: A270-419
            hotspot_residues: ["A321", "A322", ...]
            binder_length: [60, 80]
            pdb_id: null

    Returns the path to the written YAML file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build target_input: ChainStart-End using PDB residue numbers
    chain_info = get_chain_info(target_path)
    tchain = chain_info.get(chain_id, {})
    residues = tchain.get("residues", [])
    if residues:
        target_input = f"{chain_id}{residues[0]}-{residues[-1]}"
    else:
        target_input = f"{chain_id}1-100"

    # Hotspot residues: PDB residue numbers with chain prefix
    hotspot_list = [f"{chain_id}{r}" for r in site_resnums]
    hotspot_str = ", ".join(f'"{h}"' for h in hotspot_list)

    # source + target_filename are required by OmegaConf interpolation in
    # Complexa configs even when target_path is set (oc.select resolves both
    # branches). Point source at the target's parent directory.
    target_abs = Path(target_path).resolve()
    source_dir = target_abs.parent.name
    target_fname = target_abs.stem  # without .pdb

    yaml_content = (
        f"target_dict_cfg:\n"
        f"  binder_target:\n"
        f"    source: {source_dir}\n"
        f"    target_filename: {target_fname}\n"
        f"    target_path: {target_abs}\n"
        f'    target_input: "{target_input}"\n'
        f"    hotspot_residues: [{hotspot_str}]\n"
        f"    binder_length: [{length_min}, {length_max}]\n"
        f"    pdb_id: null\n"
    )

    targets_yaml = out_dir / "targets_dict.yaml"
    targets_yaml.write_text(yaml_content)
    return targets_yaml


def run_proteina_complexa(target_path, chain_id, site_resnums, length_min, length_max,
                          n_designs, out_dir, dry_run=False, ss_bias="balanced", **kwargs):
    """
    Proteina Complexa binder design (NVIDIA Complexa, flow-based with AF2 reward).

    Complexa is an end-to-end binder design tool that generates protein
    backbones via flow matching, uses AF2-based reward models during search,
    and evaluates designs with refolding metrics (scRMSD, iPAE, pLDDT).

    Pipeline: Complexa flow generation + search → AF2 reward scoring → evaluation

    Returns list of design dicts with keys:
        design_id, binder_sequence, complex_cif or binder_pdb, native_score,
        native_score_name, pc_* metrics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write target entry into Complexa's targets_dict.yaml (Hydra reads from
    # configs/targets/ in the repo). Also save a local copy for reference.
    targets_yaml = write_proteina_complexa_target(
        target_path, chain_id, site_resnums, length_min, length_max, out_dir)
    log(f"Proteina Complexa: local targets_dict.yaml written → {targets_yaml}")

    # Inject our target into Complexa's config (Hydra requires this location)
    complexa_targets = Path(PROTEINA_COMPLEXA_DIR) / "configs" / "targets" / "targets_dict.yaml"
    _inject_complexa_target(targets_yaml, complexa_targets)

    # Calculate nsamples: best-of-n with replicas=2 means each "sample"
    # produces ~2 candidates after search; nsamples=n_designs//2 gives ~n_designs outputs
    nsamples = max(2, n_designs // 2)

    complexa_bin = f"{PROTEINA_COMPLEXA_VENV}/bin/complexa"

    # Run full design pipeline (generate → filter → evaluate → analyze).
    # Requires SC, DSSP, ESM2, foldseek, mmseqs installed (see CLAUDE.md).
    cmd = [
        complexa_bin, "design",
        f"{PROTEINA_COMPLEXA_DIR}/configs/search_binder_local_pipeline.yaml",
        f"++run_name={out_dir.parent.name}_{out_dir.name}",
        f"++generation.task_name=binder_target",
        f"++generation.dataloader.dataset.nres.nsamples={nsamples}",
        f"++generation.dataloader.dataset.nres.low={length_min}",
        f"++generation.dataloader.dataset.nres.high={length_max}",
        f"++generation.dataloader.batch_size=8",
        f"++generation.search.algorithm=best-of-n",
        f"++generation.search.best_of_n.replicas=2",
        f"++gen_njobs=1",
        f"++eval_njobs=1",
        f"++seed=42",
    ]

    log(f"Proteina Complexa: running {n_designs} designs (nsamples={nsamples})...")

    # Build environment: source .env paths + Complexa-specific vars
    extra_env = {
        "COMPLEXA_INIT": "uv",
        "PYTHONNOUSERSITE": "1",
        # Paths from .env needed by Complexa Hydra configs
        "LOCAL_CODE_PATH": PROTEINA_COMPLEXA_DIR,
        "LOCAL_DATA_PATH": _WEIGHTS_DIR,
        "DATA_PATH": _WEIGHTS_DIR,
        "CKPT_PATH": f"{PROTEINA_COMPLEXA_DIR}/ckpts",
        "ESM_DIR": f"{PROTEINA_COMPLEXA_DIR}/community_models/ckpts/ESM2",
        "AF2_DIR": f"{PROTEINA_COMPLEXA_DIR}/community_models/ckpts/AF2",
        "FOLDSEEK_EXEC": f"{PROTEINA_COMPLEXA_VENV}/bin/foldseek",
        "SC_EXEC": f"{PROTEINA_COMPLEXA_DIR}/env/docker/internal/sc",
        "DSSP_EXEC": f"{PROTEINA_COMPLEXA_DIR}/env/docker/internal/dssp",
        "MMSEQS_EXEC": f"{PROTEINA_COMPLEXA_VENV}/bin/mmseqs",
        "TMOL_PATH": f"{PROTEINA_COMPLEXA_VENV}/lib/python3.12/site-packages/tmol",
        "USE_V2_COMPLEXA_ARCH": "False",
    }

    try:
        run_cmd(cmd, timeout=None, cwd=PROTEINA_COMPLEXA_DIR, extra_env=extra_env, dry_run=dry_run)
    except RuntimeError as e:
        # Complexa may fail on filter/evaluate but still produce PDBs from generate.
        # Try to collect whatever was generated.
        log(f"  Proteina Complexa: command returned error (will attempt to collect outputs): {str(e)[:200]}")

    designs = []
    if not dry_run:
        # Collect output PDBs from inference/ directory
        # Output path: ./inference/{config_name}_{task_name}_{run_name}/
        inference_base = Path(PROTEINA_COMPLEXA_DIR) / "inference"
        # Find the matching run directory using the full run_name to avoid
        # cross-run contamination (e.g. "pae8_helix" matching "reprediction_pae8_helix")
        run_name = f"{out_dir.parent.name}_{out_dir.name}"
        run_dirs = sorted(inference_base.glob(f"*{run_name}"))
        if not run_dirs:
            # Fallback: try with binder_target prefix
            run_dirs = sorted(inference_base.glob(f"*binder_target*{run_name}*"))
        if not run_dirs:
            log(f"  WARNING: no inference directory found matching run_name={run_name}")
            run_dirs = []

        sample_pdbs = []
        for rd in run_dirs:
            # Complexa outputs to subdirectories like job_0_n_80_id_0_*/
            for sub in sorted(rd.iterdir()):
                if sub.is_dir() and sub.name.startswith("job_"):
                    pdbs = sorted(sub.glob("*.pdb"))
                    # Take the complex PDB (not *_binder.pdb intermediates)
                    for p in pdbs:
                        if "_binder" not in p.stem:
                            sample_pdbs.append(p)
            # Also check for PDBs directly in the run directory
            if not sample_pdbs:
                sample_pdbs = sorted(p for p in rd.glob("*.pdb") if "_binder" not in p.stem)

        log(f"  Proteina Complexa: found {len(sample_pdbs)} sample PDBs")

        # Try to load evaluation results CSV (binder_results from evaluate step,
        # or rewards CSV from generate step as fallback)
        eval_metrics = {}  # sample_name → {metric: value}

        # First try rewards CSV from generate step (always present)
        for rd in run_dirs:
            for rew_csv in sorted(rd.glob("rewards_*.csv")):
                try:
                    with open(rew_csv) as f:
                        for row in csv.DictReader(f):
                            pdb_path = row.get("pdb_path", "")
                            if not pdb_path:
                                continue
                            sample_id = Path(pdb_path).parent.name
                            entry = {}
                            # AF2 reward metrics from generate step
                            for csv_col, pc_key in [
                                ("af2folding_i_pae", "pc_ipae"),
                                ("af2folding_plddt", "pc_plddt"),
                                ("af2folding_rmsd", "pc_scrmsd"),
                                ("total_reward", "pc_total_reward"),
                            ]:
                                val = row.get(csv_col, "")
                                if val:
                                    entry[pc_key] = _safe_float(val)
                            # i_ptm not directly in rewards but ptm_log is
                            for csv_col, pc_key in [
                                ("af2folding_i_ptm_log", "pc_iptm"),
                                ("af2folding_ptm_log", "pc_ptm"),
                            ]:
                                val = row.get(csv_col, "")
                                if val:
                                    entry[pc_key] = _safe_float(val)
                            if entry:
                                eval_metrics[sample_id] = entry
                except Exception as e:
                    log(f"  Warning: could not parse rewards CSV {rew_csv}: {e}")

        # Then try binder_results CSV from evaluate step (overrides rewards if present)
        for rd in run_dirs:
            eval_base = Path(PROTEINA_COMPLEXA_DIR) / "evaluation_results"
            eval_dirs = sorted(eval_base.glob(f"*{rd.name}*")) if eval_base.exists() else []
            csv_dirs = list(eval_dirs) + [rd]
            for csv_dir in csv_dirs:
                for eval_csv in sorted(csv_dir.glob("*binder_results_*.csv")):
                    try:
                        with open(eval_csv) as f:
                            for row in csv.DictReader(f):
                                sample_id = row.get("id_gen", "")
                                if not sample_id:
                                    continue
                                entry = {}
                                for csv_col, pc_key in [
                                    ("self_complex_i_pTM", "pc_iptm"),
                                    ("self_complex_i_pAE", "pc_ipae"),
                                    ("self_complex_pLDDT", "pc_plddt"),
                                    ("self_complex_pTM", "pc_ptm"),
                                    ("self_binder_scRMSD_ca_colabdesign", "pc_scrmsd"),
                                    ("self_complex_sc_value", "pc_sc"),
                                    ("self_complex_n_interface_hbonds_tmol", "pc_hbonds"),
                                    ("self_complex_n_interface_unsatH_tmol", "pc_bunsats"),
                                ]:
                                    val = row.get(csv_col, "")
                                    if val:
                                        entry[pc_key] = _safe_float(val)
                                eval_metrics[sample_id] = entry
                    except Exception as e:
                        log(f"  Warning: could not parse eval CSV {eval_csv}: {e}")

        # Also try monomer results CSV for scRMSD
        for rd in run_dirs:
            for mono_csv in sorted(rd.glob("monomer_results_*.csv")):
                try:
                    with open(mono_csv) as f:
                        for row in csv.DictReader(f):
                            sample_id = row.get("id_gen", "")
                            if sample_id and sample_id in eval_metrics:
                                scrmsd = _safe_float(row.get("_res_co_scRMSD_ca_esmfold", ""))
                                if scrmsd == scrmsd:  # not NaN
                                    eval_metrics[sample_id]["pc_scrmsd_esmfold"] = scrmsd
                except Exception:
                    pass

        log(f"  Proteina Complexa: loaded eval metrics for {len(eval_metrics)} samples")

        for i, pdb in enumerate(sample_pdbs):
            # Extract binder sequence from PDB
            chain_info_pdb = extract_pdb_chain_info(pdb)
            # In Complexa output, binder is typically the shorter chain
            # or the chain that is not the target chain
            binder_info = None
            for cid, info in sorted(chain_info_pdb.items()):
                if cid != chain_id:
                    binder_info = info
                    break
            if not binder_info and chain_info_pdb:
                binder_info = min(chain_info_pdb.values(), key=lambda v: v["length"])
            if not binder_info:
                continue

            seq = binder_info.get("sequence", "")
            if not seq:
                continue

            # Match eval metrics by sample directory name (e.g., job_0_n_80_id_0_single_orig0)
            sample_name = pdb.parent.name if pdb.parent.name.startswith("job_") else pdb.stem
            metrics = eval_metrics.get(sample_name, {})

            # Use iPTM as native score
            native_iptm = metrics.get("pc_iptm", 0.0)
            if native_iptm != native_iptm:  # NaN
                native_iptm = 0.0

            d = {
                "design_id":         f"proteina_complexa_{i:04d}",
                "binder_sequence":   seq,
                "binder_pdb":        str(pdb),
                "native_score":      native_iptm,
                "native_score_name": "complexa_iptm",
            }
            d.update(metrics)
            designs.append(d)
    else:
        designs = [{"design_id": "proteina_complexa_0000", "binder_sequence": "MOCK" * 25,
                    "binder_pdb": str(out_dir / "mock.pdb"),
                    "native_score": 0.55, "native_score_name": "complexa_iptm"}]

    log(f"Proteina Complexa: {len(designs)} designs collected")

    # SS bias post-filter: compute DSSP on Complexa PDBs and filter by SS composition
    if ss_bias != "balanced" and designs and not dry_run:
        n_before = len(designs)
        for d in designs:
            compute_binder_ss(d)
        params = SS_BIAS_PARAMS[ss_bias]
        if ss_bias == "beta" and "pc_max_helix" in params:
            designs = [d for d in designs
                       if d.get("binder_helix_frac", 0) <= params["pc_max_helix"]
                       or d.get("binder_helix_frac") is None]
            log(f"  Proteina Complexa SS filter (beta): kept {len(designs)}/{n_before} "
                f"with helix <= {params['pc_max_helix']}")
        elif ss_bias == "helix" and "pc_max_sheet" in params:
            designs = [d for d in designs
                       if d.get("binder_sheet_frac", 0) <= params["pc_max_sheet"]
                       or d.get("binder_sheet_frac") is None]
            log(f"  Proteina Complexa SS filter (helix): kept {len(designs)}/{n_before} "
                f"with sheet <= {params['pc_max_sheet']}")

    return designs


# ── Secondary structure analysis ──────────────────────────────────────────────

def _dssp_fractions(struct_path, binder_seq_len=None):
    """
    Run DSSP on a structure file and return (helix_frac, sheet_frac, loop_frac).

    For complex files with multiple chains, picks the binder chain as:
    - The chain closest in length to binder_seq_len (if provided)
    - Otherwise the shortest chain

    Returns (helix, sheet, loop) tuple or None on failure.
    """
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb_io
        import biotite.structure.io.pdbx as pdbx_io
    except ImportError:
        return None

    struct_path = Path(struct_path)
    if not struct_path.exists():
        return None

    try:
        if struct_path.suffix.lower() == ".cif":
            # PXDesign CIFs use non-standard chain IDs (A0, B0) that biotite
            # can't parse. Convert CIF→PDB via gemmi first, sanitizing chains.
            try:
                pdbx_file = pdbx_io.PDBxFile.read(str(struct_path))
                atoms = pdbx_io.get_structure(pdbx_file, model=1)
            except Exception:
                # Fallback: use gemmi to convert CIF→PDB with single-char chains
                import tempfile
                try:
                    import gemmi
                    st = gemmi.read_structure(str(struct_path))
                    for model in st:
                        for chain in model:
                            if len(chain.name) > 1:
                                chain.name = chain.name[0]
                    fd, tmp_pdb = tempfile.mkstemp(suffix=".pdb")
                    os.close(fd)
                    st.write_pdb(tmp_pdb)
                    pdb_file = pdb_io.PDBFile.read(tmp_pdb)
                    atoms = pdb_file.get_structure(model=1)
                    os.remove(tmp_pdb)
                except Exception:
                    return None
        else:
            pdb_file = pdb_io.PDBFile.read(str(struct_path))
            atoms = pdb_file.get_structure(model=1)

        # For complex files, pick the binder chain
        chain_ids = struc.get_chains(atoms)
        if len(chain_ids) > 1:
            chain_lens = {}
            for cid in chain_ids:
                mask = atoms.chain_id == cid
                chain_atoms = atoms[mask]
                ca_mask = chain_atoms.atom_name == "CA"
                chain_lens[cid] = int(ca_mask.sum())
            if binder_seq_len:
                # Pick chain closest in length to binder sequence
                binder_chain = min(chain_lens, key=lambda c: abs(chain_lens[c] - binder_seq_len))
            else:
                binder_chain = min(chain_lens, key=chain_lens.get)
            atoms = atoms[atoms.chain_id == binder_chain]

        # Filter to peptide residues only
        atoms = atoms[struc.filter_amino_acids(atoms)]
        if len(atoms) == 0:
            return None

        # Run DSSP (biotite's annotate_sse: 'a'=helix, 'b'=sheet, 'c'=coil)
        sse = struc.annotate_sse(atoms)
        n_total = len(sse)
        if n_total == 0:
            return None

        n_helix = sum(1 for s in sse if s == "a")
        n_sheet = sum(1 for s in sse if s == "b")
        n_loop = n_total - n_helix - n_sheet

        return (n_helix / n_total, n_sheet / n_total, n_loop / n_total)
    except Exception:
        return None


def _pyrosetta_dssp_fractions(struct_path, binder_seq_len=None):
    """
    Fallback DSSP via PyRosetta (available in BindCraft env).
    Returns (helix_frac, sheet_frac, loop_frac) or None on failure.

    PyRosetta DSSP codes: H=helix, E=sheet, L=loop.
    """
    struct_path = Path(struct_path)
    if not struct_path.exists():
        return None

    try:
        # Convert CIF→PDB if needed (PyRosetta handles PDB more reliably)
        load_path = str(struct_path)
        tmp_pdb = None
        if struct_path.suffix.lower() == ".cif":
            import tempfile
            try:
                import gemmi
                st = gemmi.read_structure(str(struct_path))
                for model in st:
                    for chain in model:
                        if len(chain.name) > 1:
                            chain.name = chain.name[0]
                fd, tmp_pdb = tempfile.mkstemp(suffix=".pdb")
                os.close(fd)
                st.write_pdb(tmp_pdb)
                load_path = tmp_pdb
            except Exception:
                return None

        import pyrosetta as pr
        if not pr.rosetta.basic.was_init_called():
            pr.init("-mute all", silent=True)
        pose = pr.pose_from_file(load_path)

        if tmp_pdb:
            os.remove(tmp_pdb)

        # Identify binder chain (shortest, or closest to binder_seq_len)
        chains = set()
        for i in range(1, pose.total_residue() + 1):
            chains.add(pose.pdb_info().chain(i))
        chains = sorted(chains)

        if len(chains) > 1:
            chain_lens = {}
            for c in chains:
                chain_lens[c] = sum(1 for i in range(1, pose.total_residue() + 1)
                                    if pose.pdb_info().chain(i) == c)
            if binder_seq_len:
                binder_chain = min(chain_lens, key=lambda c: abs(chain_lens[c] - binder_seq_len))
            else:
                binder_chain = min(chain_lens, key=chain_lens.get)
        else:
            binder_chain = chains[0] if chains else "A"

        # Get DSSP
        dssp = pr.rosetta.core.scoring.dssp.Dssp(pose)
        dssp_str = dssp.get_dssp_secstruct()

        # Extract binder residues only
        binder_ss = ""
        for i in range(1, pose.total_residue() + 1):
            if pose.pdb_info().chain(i) == binder_chain:
                binder_ss += dssp_str[i - 1]

        n_total = len(binder_ss)
        if n_total == 0:
            return None

        n_helix = binder_ss.count("H")
        n_sheet = binder_ss.count("E")
        n_loop = n_total - n_helix - n_sheet

        return (n_helix / n_total, n_sheet / n_total, n_loop / n_total)
    except Exception:
        return None


def compute_binder_ss(design):
    """
    Compute helix/sheet/loop fractions for a binder.

    Tries biotite DSSP first, then PyRosetta DSSP as fallback.

    Structure file priority:
    1. Boltz-2 complex CIF
    2. ESMFold PDB
    3. Native binder/complex PDB (BindCraft, PXDesign, Proteina Complexa)
    4. Backbone PDB (RFdiffusion, Proteina)

    Sets binder_helix_frac, binder_sheet_frac, binder_loop_frac on the design dict.
    Returns True if SS was computed, False otherwise.
    """
    binder_len = len(design.get("binder_sequence", "")) or None

    # Collect available structure files in priority order
    struct_files = []
    for key in ("complex_cif", "esmfold_pdb", "binder_pdb", "backbone_pdb"):
        p = design.get(key)
        if p and Path(p).exists():
            struct_files.append(p)

    # Try biotite DSSP first (fast, no heavy imports)
    for p in struct_files:
        result = _dssp_fractions(p, binder_seq_len=binder_len)
        if result:
            design["binder_helix_frac"] = result[0]
            design["binder_sheet_frac"] = result[1]
            design["binder_loop_frac"] = result[2]
            return True

    # Fallback: PyRosetta DSSP (handles more formats, slower)
    for p in struct_files:
        result = _pyrosetta_dssp_fractions(p, binder_seq_len=binder_len)
        if result:
            design["binder_helix_frac"] = result[0]
            design["binder_sheet_frac"] = result[1]
            design["binder_loop_frac"] = result[2]
            return True

    return False


def populate_binder_composition(designs):
    """
    Populate amino acid composition metrics for all designs.
    Computes from binder_sequence (no structure needed).

    Adds to each design dict:
        binder_KE_fraction — fraction of K + E residues (high = poor expression risk)
        binder_K_count     — number of Lys residues
        binder_E_count     — number of Glu residues
    """
    for d in designs:
        seq = d.get("binder_sequence", "")
        if not seq:
            d["binder_KE_fraction"] = float("nan")
            d["binder_K_count"] = 0
            d["binder_E_count"] = 0
            continue
        seq_upper = seq.upper()
        k = seq_upper.count("K")
        e = seq_upper.count("E")
        d["binder_K_count"] = k
        d["binder_E_count"] = e
        d["binder_KE_fraction"] = (k + e) / len(seq_upper)


def populate_interface_composition(designs, interface_dist=5.0):
    """
    Compute K+E composition at the binder-target interface vs surface.
    Requires a complex structure (complex_cif or binder_pdb with target).

    For each design with a complex structure, identifies binder residues
    within interface_dist of any target residue, then counts K+E separately
    for interface and surface (non-interface) residues.

    Adds to each design dict:
        interface_n_residues     — number of binder residues at interface
        interface_KE_fraction    — K+E fraction among interface residues
        interface_K_count        — K at interface
        interface_E_count        — E at interface
        surface_KE_fraction      — K+E fraction among non-interface residues
    """
    # 3-letter to 1-letter mapping for standard amino acids
    AA3TO1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    n_computed = 0
    n_skipped = 0

    for d in designs:
        # Find complex structure
        struct_file = None
        if d.get("complex_cif") and Path(d["complex_cif"]).exists():
            struct_file = d["complex_cif"]
        elif d.get("binder_pdb") and Path(d["binder_pdb"]).exists():
            struct_file = d["binder_pdb"]

        if not struct_file:
            # Try to discover CIF from validation directory
            # (paths not stored in CSV, only available at runtime)
            design_id = d.get("design_id", "")
            for parent in [Path(d.get("_results_dir", ".")), Path(".")]:
                cif_glob = list(parent.glob(
                    f"validation/boltz/{design_id}/boltz_results_*/predictions/*/*.cif"))
                if not cif_glob:
                    cif_glob = list(parent.glob(
                        f"validation/boltz/boltz_results_batch_inputs/predictions/{design_id}/*.cif"))
                if cif_glob:
                    struct_file = str(cif_glob[0])
                    d["complex_cif"] = struct_file
                    break
        if not struct_file:
            n_skipped += 1
            continue

        # Parse atoms with residue names
        is_cif = str(struct_file).endswith(".cif")
        chain_atoms = {}      # chain_id → [(resnum, x, y, z)]
        chain_resnames = {}   # chain_id → {resnum: resname_3letter}
        chain_res_counts = {}
        try:
            with open(struct_file) as fh:
                for line in fh:
                    if not line.startswith("ATOM"):
                        continue
                    if is_cif:
                        parts = line.split()
                        if len(parts) < 13:
                            continue
                        if parts[2] == "H":
                            continue
                        chain = parts[9] if len(parts) > 9 else parts[6]
                        resname = parts[5] if len(parts) > 9 else parts[5]
                        try:
                            resnum = int(parts[7] if len(parts) > 9 else parts[8])
                            x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                        except (ValueError, IndexError):
                            continue
                    else:
                        if len(line) < 54:
                            continue
                        element = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[0]
                        if element == "H":
                            continue
                        chain = line[21]
                        resname = line[17:20].strip()
                        try:
                            resnum = int(line[22:26].strip())
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                        except ValueError:
                            continue
                    chain_atoms.setdefault(chain, []).append((resnum, x, y, z))
                    chain_resnames.setdefault(chain, {})[resnum] = resname
                    chain_res_counts.setdefault(chain, set()).add(resnum)
        except (OSError, IOError):
            n_skipped += 1
            continue

        if len(chain_res_counts) < 2:
            n_skipped += 1
            continue

        # Identify binder (fewer residues) and target (more)
        sorted_chains = sorted(chain_res_counts.items(), key=lambda x: len(x[1]))
        binder_chain = sorted_chains[0][0]
        target_chain = sorted_chains[-1][0]

        binder_atoms = chain_atoms[binder_chain]
        target_atoms = chain_atoms[target_chain]
        binder_resnames = chain_resnames[binder_chain]

        # Build target coordinate array
        target_coords = np.array([[a[1], a[2], a[3]] for a in target_atoms])

        # For each binder residue, find min distance to any target atom
        binder_residues = sorted(chain_res_counts[binder_chain])
        interface_resnums = set()

        for resnum in binder_residues:
            res_atoms = [(a[1], a[2], a[3]) for a in binder_atoms if a[0] == resnum]
            if not res_atoms:
                continue
            res_coords = np.array(res_atoms)
            # Min distance from any atom of this residue to any target atom
            diffs = res_coords[:, None, :] - target_coords[None, :, :]
            min_dist = float(np.sqrt((diffs ** 2).sum(axis=2)).min())
            if min_dist <= interface_dist:
                interface_resnums.add(resnum)

        # Count K+E at interface vs surface
        interface_k, interface_e = 0, 0
        surface_k, surface_e = 0, 0
        n_interface = len(interface_resnums)
        n_surface = len(binder_residues) - n_interface

        for resnum in binder_residues:
            resname = binder_resnames.get(resnum, "")
            aa = AA3TO1.get(resname, "")
            if resnum in interface_resnums:
                if aa == "K":
                    interface_k += 1
                elif aa == "E":
                    interface_e += 1
            else:
                if aa == "K":
                    surface_k += 1
                elif aa == "E":
                    surface_e += 1

        d["interface_n_residues"] = n_interface
        d["interface_K_count"] = interface_k
        d["interface_E_count"] = interface_e
        d["interface_KE_fraction"] = round((interface_k + interface_e) / n_interface, 3) if n_interface > 0 else float("nan")
        d["surface_KE_fraction"] = round((surface_k + surface_e) / n_surface, 3) if n_surface > 0 else float("nan")
        n_computed += 1

    log(f"  Interface composition: {n_computed} computed, {n_skipped} skipped (no structure)")


def populate_binder_ss(designs):
    """
    Populate binder_helix_frac, binder_sheet_frac, binder_loop_frac for all designs.

    Uses native tool SS values where available, falls back to DSSP computation.
    """
    n_native = 0
    n_computed = 0
    n_failed = 0
    for d in designs:
        if "binder_helix_frac" in d:
            continue  # already computed (e.g. by Complexa SS filter)

        tool = _get_tool_from_design_id(d.get("design_id", ""))

        # BoltzGen: use native bg_helix/sheet/loop — these are residue COUNTS,
        # not fractions. Normalize by total to get 0-1 fractions.
        if tool == "boltzgen" and "bg_helix" in d:
            h = _safe_float(d["bg_helix"])
            s = _safe_float(d["bg_sheet"])
            l = _safe_float(d["bg_loop"])
            total = h + s + l
            if total > 0 and h == h and s == s and l == l:  # all non-NaN
                d["binder_helix_frac"] = h / total
                d["binder_sheet_frac"] = s / total
                d["binder_loop_frac"] = l / total
                n_native += 1
            else:
                # Fallback to DSSP if native counts are invalid
                if compute_binder_ss(d):
                    n_computed += 1
                else:
                    n_failed += 1
            continue

        # BindCraft: use native bc_binder_helix_pct etc. (0-100, convert to 0-1)
        if tool == "bindcraft" and "bc_binder_helix_pct" in d:
            d["binder_helix_frac"] = _safe_float(d["bc_binder_helix_pct"]) / 100.0
            d["binder_sheet_frac"] = _safe_float(d["bc_binder_betasheet_pct"]) / 100.0
            d["binder_loop_frac"] = _safe_float(d["bc_binder_loop_pct"]) / 100.0
            n_native += 1
            continue

        # All others: compute via DSSP
        if compute_binder_ss(d):
            n_computed += 1
        else:
            n_failed += 1

    log(f"  SS fractions: {n_native} native, {n_computed} DSSP-computed, {n_failed} no structure")


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_esmfold(designs, val_dir, plddt_threshold=70.0, dry_run=False):
    """
    Stage 1: ESMFold pre-filter.
    Folds each binder sequence independently and filters by mean pLDDT.
    Adds 'esmfold_plddt' to each design dict.
    Returns list of designs passing the threshold.
    """
    val_dir = Path(val_dir) / "esmfold"
    val_dir.mkdir(parents=True, exist_ok=True)

    log(f"ESMFold validation: {len(designs)} designs, pLDDT threshold={plddt_threshold}")
    passing = []

    for d in designs:
        seq       = d["binder_sequence"]
        design_id = d["design_id"]
        pdb_path  = val_dir / f"{design_id}.pdb"

        if dry_run:
            d["esmfold_plddt"] = 80.0
            passing.append(d)
            continue

        script = (
            "import torch, os\n"
            "from transformers import AutoTokenizer, EsmForProteinFolding\n"
            "os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'\n"
            "tok=AutoTokenizer.from_pretrained('facebook/esmfold_v1')\n"
            "model=EsmForProteinFolding.from_pretrained('facebook/esmfold_v1',"
            "low_cpu_mem_usage=True)\n"
            "model=model.cuda(); model.esm=model.esm.half(); model.eval()\n"
            "model.trunk.set_chunk_size(64)\n"
            f"seq='{seq}'\n"
            "inp=tok([seq],return_tensors='pt',add_special_tokens=False).to('cuda')\n"
            "with torch.no_grad(): out=model(**inp)\n"
            "pdb_str=model.output_to_pdb(out)[0]\n"
            f"open(r'{pdb_path}','w').write(pdb_str)\n"
            "import numpy as np\n"
            "bfacs=[float(l[60:66]) for l in pdb_str.splitlines()\n"
            "       if l.startswith('ATOM') and l[12:16].strip()=='CA']\n"
            "plddt=np.mean(bfacs) if bfacs else 0.0\n"
            "plddt=plddt*100.0 if plddt <= 1.0 else plddt\n"
            "print(f'PLDDT:{plddt:.2f}')\n"
        )

        try:
            stdout, _ = run_cmd(
                ["conda", "run", "--no-capture-output", "-n", "esmfold",
                 "python", "-c", script],
                timeout=300)
            m = re.search(r"PLDDT:([0-9.]+)", stdout)
            plddt = float(m.group(1)) if m else 0.0
        except Exception as e:
            log(f"  ESMFold failed for {design_id}: {e}")
            plddt = 0.0

        d["esmfold_plddt"] = plddt
        d["esmfold_pdb"] = str(pdb_path)
        if plddt >= plddt_threshold:
            passing.append(d)
        else:
            log(f"  {design_id}: ESMFold pLDDT={plddt:.1f} < {plddt_threshold} — filtered")

    log(f"ESMFold: {len(passing)}/{len(designs)} passed")
    return passing


def validate_boltz(designs, target_chain_seq, val_dir,
                   max_per_tool=20, target_len=0, dry_run=False,
                   site_resnums=None, target_residues=None,
                   boltz_devices=1):
    """
    Stage 2: Boltz-2 uniform scoring of binder+target complex.
    BATCH MODE: all ESMFold-passing designs are predicted in a single Boltz-2
    invocation (model loaded once). max_per_tool is kept for API compatibility
    but ignored — all designs are validated.

    Adds to each validated design dict:
        boltz_iptm          — interface pTM (chain B → chain A)
        boltz_ptm           — full complex pTM
        boltz_complex_plddt — mean pLDDT over all complex tokens (0–1, stored ×100)
        boltz_iplddt        — interface pLDDT (0–1, stored ×100)
        boltz_binder_plddt  — mean pLDDT of binder tokens only (0–100)
        boltz_min_interface_pae  — min PAE value in target→binder off-diagonal block
        boltz_mean_interface_pae — mean PAE value in same block
        boltz_site_min_pae  — min PAE in site-residues→binder block
        boltz_site_mean_pae — mean PAE in site-residues→binder block
        complex_cif         — path to output CIF

    Designs not sent to Boltz get nan for all boltz_* fields.
    If site_resnums is provided, a pocket constraint steers docking to the site.

    Parameters
    ----------
    target_residues : list of int or None
        PDB residue numbers for the target chain (from extract_pdb_chain_info).
        Used to convert site_resnums (PDB numbering) to 1-based sequential
        indices for the Boltz pocket constraint and site-specific PAE.
    """
    val_dir = Path(val_dir) / "boltz"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Convert PDB residue numbers → 1-based sequential indices for Boltz.
    # Boltz pocket constraints use token positions (1-based) within the YAML
    # sequence, not PDB residue numbers.  Same conversion as write_boltzgen_yaml.
    site_indices = None  # 1-based positions in the target sequence
    if site_resnums and target_residues:
        resnum_to_idx = {r: i + 1 for i, r in enumerate(target_residues)}
        site_indices = [resnum_to_idx[r] for r in site_resnums if r in resnum_to_idx]
        if len(site_indices) != len(site_resnums):
            missing = [r for r in site_resnums if r not in resnum_to_idx]
            log(f"  WARNING: {len(missing)} site residues not found in target chain: {missing}")
        if site_indices:
            log(f"  Pocket constraint: PDB resnums {site_resnums[:3]}..{site_resnums[-1]} "
                f"→ Boltz indices {site_indices[:3]}..{site_indices[-1]}")
    elif site_resnums and not target_residues:
        # Fallback: assume PDB numbering starts at 1 (may be wrong)
        site_indices = list(site_resnums)
        log(f"  WARNING: target_residues not provided, using raw PDB resnums "
            f"for pocket constraint (may be incorrect if numbering ≠ 1-based)")

    # 0-based row indices into the PAE matrix for site-specific PAE
    site_row_indices = [idx - 1 for idx in site_indices] if site_indices else None

    # Select ALL ESMFold-passing designs for Boltz-2 batch validation.
    # No per-tool cap — every design with a valid ESMFold pLDDT is repredicted.
    to_validate = [d for d in designs
                   if d.get("esmfold_plddt", 0) == d.get("esmfold_plddt", 0)]  # not NaN

    log(f"Boltz validation: {len(to_validate)} designs (all ESMFold-passing)")

    # Auto memory-saving for large targets
    large      = target_len > 500
    recycling  = "2"   # 2 recycles; better interface refinement
    samp_steps = "200" # 200 diffusion steps; Boltz default; better quality
    msa_depth  = "64" if large else "128"

    tlen = len(target_chain_seq)  # number of target residues

    if dry_run:
        for d in to_validate:
            d["boltz_iptm"]               = 0.60
            d["boltz_ptm"]                = 0.70
            d["boltz_confidence"]         = 0.65
            d["boltz_protein_iptm"]       = 0.62
            d["boltz_complex_plddt"]      = 75.0
            d["boltz_iplddt"]             = 72.0
            d["boltz_binder_plddt"]       = 80.0
            d["boltz_complex_pde"]        = 1.8
            d["boltz_complex_ipde"]       = 2.5
            d["boltz_min_interface_pae"]  = 5.0
            d["boltz_mean_interface_pae"] = 8.0
            d["boltz_site_min_pae"]      = 4.0
            d["boltz_site_mean_pae"]     = 6.0
            d["boltz_min_interface_pde"]  = 0.8
            d["boltz_mean_interface_pde"] = 1.5
    else:
        # ── BATCH MODE: write all YAMLs to one directory, run Boltz-2 once ──
        batch_yaml_dir = val_dir / "batch_inputs"
        batch_yaml_dir.mkdir(parents=True, exist_ok=True)

        # Check which designs already have results (for --reuse)
        pending = []
        for d in to_validate:
            design_id = d["design_id"]
            result_dir = val_dir / "boltz_results_batch_inputs" / "predictions" / design_id
            if result_dir.exists() and list(result_dir.glob("confidence_*.json")):
                pending.append((d, False))  # already done, just parse
            else:
                pending.append((d, True))   # needs prediction

        needs_prediction = [(d, flag) for d, flag in pending if flag]
        already_done = [(d, flag) for d, flag in pending if not flag]

        if already_done:
            log(f"  Boltz batch: {len(already_done)} designs already predicted, "
                f"{len(needs_prediction)} new")

        # ── Pre-compute target MSA once ──────────────────────────────────
        # The target sequence is identical across all designs. Compute its
        # MSA once and reference it in every YAML to avoid redundant API
        # queries (saves hours at scale).
        target_msa_path = val_dir / "target_msa.csv"
        if not target_msa_path.exists() and needs_prediction:
            # Check if a previous run already computed the target MSA
            existing_msa = list(val_dir.glob("boltz_results_batch_inputs/msa/*_0.csv"))
            if existing_msa:
                # Reuse existing target MSA from a previous partial run
                import shutil
                shutil.copy2(existing_msa[0], target_msa_path)
                log(f"  Boltz batch: reusing target MSA from previous run")
            else:
                # Compute target MSA via single Boltz prediction
                log(f"  Boltz batch: computing target MSA (one-time)...")
                msa_yaml_dir = val_dir / "target_msa_input"
                msa_yaml_dir.mkdir(parents=True, exist_ok=True)
                msa_yaml = msa_yaml_dir / "target_msa.yaml"
                msa_yaml.write_text(
                    "version: 1\n"
                    "sequences:\n"
                    "  - protein:\n"
                    "      id: A\n"
                    f"      sequence: {target_chain_seq}\n"
                )
                msa_cmd = [
                    "conda", "run", "--no-capture-output", "-n", "boltz",
                    "boltz", "predict", str(msa_yaml),
                    "--out_dir", str(val_dir / "target_msa_run"),
                    "--use_msa_server",
                    "--recycling_steps", "1",
                    "--diffusion_samples", "1",
                    "--sampling_steps", "10",
                    "--no_kernels",
                ]
                try:
                    run_cmd(msa_cmd, timeout=600,
                            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
                    # Find the generated MSA CSV
                    msa_csvs = list((val_dir / "target_msa_run").rglob("*_0.csv"))
                    if msa_csvs:
                        import shutil
                        shutil.copy2(msa_csvs[0], target_msa_path)
                        log(f"  Boltz batch: target MSA computed ({target_msa_path})")
                    else:
                        log(f"  Boltz batch: WARNING — target MSA not found, falling back to per-design MSA")
                except Exception as e:
                    log(f"  Boltz batch: WARNING — target MSA computation failed: {e}")

        use_precomputed_msa = target_msa_path.exists()
        if use_precomputed_msa:
            log(f"  Boltz batch: using pre-computed target MSA + empty binder MSA (fast mode)")
        else:
            log(f"  Boltz batch: using API MSA for all designs (slow mode)")

        # Write YAML files for designs that need prediction
        for d, _ in needs_prediction:
            seq = d["binder_sequence"]
            design_id = d["design_id"]
            yaml_path = batch_yaml_dir / f"{design_id}.yaml"

            yaml_content = (
                "version: 1\n"
                "sequences:\n"
                "  - protein:\n"
                "      id: A\n"
                f"      sequence: {target_chain_seq}\n"
            )
            if use_precomputed_msa:
                yaml_content += f"      msa: {target_msa_path}\n"
            yaml_content += (
                "  - protein:\n"
                "      id: B\n"
                f"      sequence: {seq}\n"
            )
            if use_precomputed_msa:
                yaml_content += "      msa: empty\n"
            if site_indices:
                contacts = "\n".join(f"        - [A, {r}]" for r in site_indices)
                yaml_content += (
                    "constraints:\n"
                    "  - pocket:\n"
                    "      binder: B\n"
                    "      contacts:\n"
                    f"{contacts}\n"
                )
            yaml_path.write_text(yaml_content)

        # Run Boltz-2 batch prediction (single invocation, model loaded once)
        if needs_prediction:
            # Detect free GPUs for multi-GPU Boltz-2
            actual_devices = 1
            boltz_extra_env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            if boltz_devices > 1:
                try:
                    gpu_check = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10)
                    free_gpus = []
                    for line in gpu_check.stdout.strip().split("\n"):
                        idx, mem_used = line.split(",")
                        if int(mem_used.strip()) < 500:
                            free_gpus.append(idx.strip())
                    actual_devices = min(boltz_devices, len(free_gpus))
                    if actual_devices > 1:
                        boltz_extra_env["CUDA_VISIBLE_DEVICES"] = ",".join(free_gpus[:actual_devices])
                        log(f"  Boltz batch: {len(free_gpus)} GPUs free, "
                            f"using {actual_devices} (cuda:{','.join(free_gpus[:actual_devices])})")
                    else:
                        log(f"  Boltz batch: requested {boltz_devices} GPUs but "
                            f"only {len(free_gpus)} free, using 1")
                except Exception:
                    log(f"  Boltz batch: GPU detection failed, using 1 GPU")

            log(f"  Boltz batch: predicting {len(needs_prediction)} designs "
                f"on {actual_devices} GPU(s)...")
            cmd = [
                "conda", "run", "--no-capture-output", "-n", "boltz",
                "boltz", "predict", str(batch_yaml_dir),
                "--out_dir",            str(val_dir),
                "--use_msa_server",
                "--recycling_steps",    recycling,
                "--diffusion_samples",  "1",
                "--sampling_steps",     samp_steps,
                "--num_subsampled_msa", msa_depth,
                "--preprocessing-threads", "48",
                "--no_kernels",
                "--write_full_pae",
                "--write_full_pde",
            ]
            if actual_devices > 1:
                cmd += ["--devices", str(actual_devices)]

            try:
                # No timeout — batch mode handles preprocessing + inference
                # for all designs. Let it run to completion.
                run_cmd(cmd, timeout=None, extra_env=boltz_extra_env)
            except Exception as e:
                log(f"  Boltz batch prediction failed: {e}")
                log(f"  Falling back to per-design sequential prediction...")
                # Fallback: run remaining designs one by one
                for d, _ in needs_prediction:
                    design_id = d["design_id"]
                    result_dir = val_dir / "boltz_results_batch_inputs" / "predictions" / design_id
                    if result_dir.exists() and list(result_dir.glob("confidence_*.json")):
                        continue  # this one completed before the batch failed
                    yaml_path = batch_yaml_dir / f"{design_id}.yaml"
                    fallback_dir = val_dir / design_id
                    fallback_dir.mkdir(exist_ok=True)
                    fb_cmd = [
                        "conda", "run", "--no-capture-output", "-n", "boltz",
                        "boltz", "predict", str(yaml_path),
                        "--out_dir",            str(fallback_dir),
                        "--use_msa_server",
                        "--recycling_steps",    recycling,
                        "--diffusion_samples",  "1",
                        "--sampling_steps",     samp_steps,
                        "--num_subsampled_msa", msa_depth,
                        "--no_kernels",
                        "--write_full_pae",
                        "--write_full_pde",
                    ]
                    try:
                        run_cmd(fb_cmd, timeout=3600,
                                extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
                    except Exception as e2:
                        log(f"  Boltz fallback failed for {design_id}: {e2}")

        # ── Parse results for all designs ──
        nan_keys = ("boltz_iptm", "boltz_ptm", "boltz_confidence", "boltz_protein_iptm",
                    "boltz_complex_plddt", "boltz_iplddt", "boltz_binder_plddt",
                    "boltz_complex_pde", "boltz_complex_ipde",
                    "boltz_min_interface_pae", "boltz_mean_interface_pae",
                    "boltz_site_min_pae", "boltz_site_mean_pae",
                    "boltz_min_interface_pde", "boltz_mean_interface_pde")

    for d in to_validate:
        design_id = d["design_id"]
        if dry_run:
            continue

        # Find prediction output — batch mode or fallback per-design
        pred_dir = val_dir / "boltz_results_batch_inputs" / "predictions" / design_id
        if not pred_dir.exists():
            # Try fallback per-design directory
            pred_dir = val_dir / design_id / "boltz_results_input" / "predictions" / "input"
        if not pred_dir.exists():
            # Also check legacy per-design layout
            pred_dir = val_dir / design_id / f"boltz_results_{design_id}" / "predictions" / design_id
        if not pred_dir.exists():
            for k in nan_keys:
                d[k] = float("nan")
            continue

        # ── confidence JSON ─────────────────────────────────────────────────
        boltz_iptm = 0.0
        boltz_ptm  = float("nan")
        boltz_complex_plddt  = float("nan")
        boltz_iplddt         = float("nan")
        boltz_confidence     = float("nan")
        boltz_complex_pde    = float("nan")
        boltz_complex_ipde   = float("nan")
        boltz_protein_iptm   = float("nan")

        for conf_f in sorted(pred_dir.glob("confidence_*.json")):
            try:
                with open(conf_f) as f:
                    cdata = json.load(f)
                boltz_iptm        = cdata.get("iptm", 0.0)
                boltz_ptm         = cdata.get("ptm", float("nan"))
                boltz_confidence  = cdata.get("confidence_score", float("nan"))
                boltz_complex_pde = cdata.get("complex_pde", float("nan"))
                boltz_complex_ipde= cdata.get("complex_ipde", float("nan"))
                boltz_protein_iptm= cdata.get("protein_iptm", float("nan"))
                # complex_plddt and complex_iplddt are 0–1 in the JSON; convert to 0–100
                raw_cplddt = cdata.get("complex_plddt", float("nan"))
                raw_iplddt = cdata.get("complex_iplddt", float("nan"))
                boltz_complex_plddt = raw_cplddt * 100.0 if raw_cplddt == raw_cplddt else float("nan")
                boltz_iplddt        = raw_iplddt * 100.0 if raw_iplddt == raw_iplddt else float("nan")
            except Exception as e:
                log(f"  Warning: could not parse confidence JSON for {design_id}: {e}")
            break  # take first file

        # ── per-token pLDDT — extract binder mean ──────────────────────────
        boltz_binder_plddt = float("nan")
        plddt_files = sorted(pred_dir.glob("plddt_*.npz"))
        if plddt_files:
            try:
                plddt_arr = np.load(plddt_files[0])["plddt"] * 100.0
                if len(plddt_arr) > tlen:
                    boltz_binder_plddt = float(plddt_arr[tlen:].mean())
                else:
                    boltz_binder_plddt = float(plddt_arr.mean())
            except Exception as e:
                log(f"  Warning: could not parse plddt npz for {design_id}: {e}")

        # ── PAE matrix — interface block ────────────────────────────────────
        boltz_min_interface_pae  = float("nan")
        boltz_mean_interface_pae = float("nan")
        boltz_site_min_pae       = float("nan")
        boltz_site_mean_pae      = float("nan")
        pae_files = sorted(pred_dir.glob("pae_*.npz"))
        if pae_files:
            try:
                pae_mat = np.load(pae_files[0])["pae"]  # shape (N_tokens, N_tokens), float32
                # Off-diagonal block: rows=target (0:tlen), cols=binder (tlen:)
                if pae_mat.shape[0] > tlen and pae_mat.shape[1] > tlen:
                    interface_block = pae_mat[:tlen, tlen:]
                    boltz_min_interface_pae  = float(interface_block.min())
                    boltz_mean_interface_pae = float(interface_block.mean())

                    # Site-specific PAE: only rows corresponding to binding site residues
                    if site_row_indices:
                        valid_rows = [r for r in site_row_indices if r < tlen]
                        if valid_rows:
                            site_block = pae_mat[valid_rows][:, tlen:]
                            boltz_site_min_pae  = float(site_block.min())
                            boltz_site_mean_pae = float(site_block.mean())
            except Exception as e:
                log(f"  Warning: could not parse PAE npz for {design_id}: {e}")

        # ── PDE matrix — interface block ────────────────────────────────────
        boltz_min_interface_pde  = float("nan")
        boltz_mean_interface_pde = float("nan")
        pde_files = sorted(pred_dir.glob("pde_*.npz"))
        if pde_files:
            try:
                pde_data = np.load(pde_files[0])
                pde_mat  = pde_data[list(pde_data.keys())[0]]  # shape (N, N)
                if pde_mat.shape[0] > tlen and pde_mat.shape[1] > tlen:
                    pde_block = pde_mat[:tlen, tlen:]
                    boltz_min_interface_pde  = float(pde_block.min())
                    boltz_mean_interface_pde = float(pde_block.mean())
            except Exception as e:
                log(f"  Warning: could not parse PDE npz for {design_id}: {e}")

        d["boltz_iptm"]               = boltz_iptm
        d["boltz_ptm"]                = boltz_ptm
        d["boltz_confidence"]         = boltz_confidence
        d["boltz_protein_iptm"]       = boltz_protein_iptm
        d["boltz_complex_plddt"]      = boltz_complex_plddt
        d["boltz_iplddt"]             = boltz_iplddt
        d["boltz_binder_plddt"]       = boltz_binder_plddt
        d["boltz_complex_pde"]        = boltz_complex_pde
        d["boltz_complex_ipde"]       = boltz_complex_ipde
        d["boltz_min_interface_pae"]  = boltz_min_interface_pae
        d["boltz_mean_interface_pae"] = boltz_mean_interface_pae
        d["boltz_site_min_pae"]      = boltz_site_min_pae
        d["boltz_site_mean_pae"]     = boltz_site_mean_pae
        d["boltz_min_interface_pde"]  = boltz_min_interface_pde
        d["boltz_mean_interface_pde"] = boltz_mean_interface_pde

        # Save complex CIF path (may already be set for BoltzGen designs)
        cifs = sorted(pred_dir.glob("*.cif"))
        if cifs:
            d["complex_cif"] = str(cifs[0])

    # Mark designs that weren't sent to Boltz as nan
    validated_ids = {d["design_id"] for d in to_validate}
    nan_keys = ("boltz_iptm", "boltz_ptm", "boltz_confidence", "boltz_protein_iptm",
                "boltz_complex_plddt", "boltz_iplddt", "boltz_binder_plddt",
                "boltz_complex_pde", "boltz_complex_ipde",
                "boltz_min_interface_pae", "boltz_mean_interface_pae",
                "boltz_site_min_pae", "boltz_site_mean_pae",
                "boltz_min_interface_pde", "boltz_mean_interface_pde")
    for d in designs:
        if d["design_id"] not in validated_ids:
            for k in nan_keys:
                d[k] = float("nan")

    return designs



# ── Geometric site proximity filter ───────────────────────────────────────────

def _parse_heavy_atoms(filepath):
    """Parse all heavy atom coordinates by chain from PDB or CIF file.

    Returns (chains, chain_res_counts) where:
      chains: chain_id → [(resnum, x, y, z), ...] (one entry per heavy atom)
      chain_res_counts: chain_id → number of unique residues
    Returns (None, None) on failure.
    """
    chains = {}
    chain_res = {}
    path_str = str(filepath)
    is_cif = path_str.endswith(".cif")
    try:
        with open(filepath) as fh:
            for line in fh:
                if is_cif:
                    if not line.startswith("ATOM"):
                        continue
                    parts = line.split()
                    if len(parts) < 13:
                        continue
                    element = parts[2] if len(parts) > 2 else ""
                    if element == "H":
                        continue
                    chain = parts[9] if len(parts) > 9 else parts[6]
                    try:
                        resnum = int(parts[7] if len(parts) > 9 else parts[8])
                        x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                    except (ValueError, IndexError):
                        continue
                else:
                    if not line.startswith("ATOM"):
                        continue
                    element = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[0]
                    if element == "H":
                        continue
                    chain = line[21]
                    try:
                        resnum = int(line[22:26].strip())
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue
                chains.setdefault(chain, []).append((resnum, x, y, z))
                chain_res.setdefault(chain, set()).add(resnum)
    except (OSError, IOError):
        return None, None
    return chains, {c: len(r) for c, r in chain_res.items()}


def _compute_site_metrics(args):
    """Worker function for parallel site metric computation.

    Returns dict with site metrics, or None to skip.
    """
    import numpy as np
    struct_file, site_set, n_site_residues, max_dist, interface_dist = args

    chains, chain_res_counts = _parse_heavy_atoms(struct_file)
    if not chains or not chain_res_counts or len(chain_res_counts) < 2:
        return None

    sorted_chains = sorted(chain_res_counts.items(), key=lambda x: x[1])
    binder_chain_id = sorted_chains[0][0]
    target_chain_id = sorted_chains[-1][0]

    binder_atoms = chains[binder_chain_id]
    target_atoms = chains[target_chain_id]

    binder_coords = np.array([[a[1], a[2], a[3]] for a in binder_atoms])

    # Site contact fraction
    site_res_atoms = {}
    for resnum, x, y, z in target_atoms:
        if resnum in site_set:
            site_res_atoms.setdefault(resnum, []).append([x, y, z])

    n_contacted = 0
    for resnum, coords_list in site_res_atoms.items():
        res_coords = np.array(coords_list)
        diffs = binder_coords[:, None, :] - res_coords[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        if float(dists.min()) <= max_dist:
            n_contacted += 1

    contact_frac = n_contacted / n_site_residues if n_site_residues > 0 else 0.0

    # Site interface fraction
    # interface_dist passed via args
    target_all_atoms = {}
    for resnum, x, y, z in target_atoms:
        target_all_atoms.setdefault(resnum, []).append([x, y, z])

    binder_res_atoms = {}
    for resnum, x, y, z in binder_atoms:
        binder_res_atoms.setdefault(resnum, []).append([x, y, z])

    binder_at_site = 0
    binder_at_target = 0
    for b_resnum, b_coords_list in binder_res_atoms.items():
        b_arr = np.array(b_coords_list)
        contacts_any_target = False
        contacts_site = False
        for t_resnum, t_coords_list in target_all_atoms.items():
            t_arr = np.array(t_coords_list)
            diffs = b_arr[:, None, :] - t_arr[None, :, :]
            min_d = float(np.sqrt((diffs ** 2).sum(axis=2)).min())
            if min_d <= interface_dist:
                contacts_any_target = True
                if t_resnum in site_set:
                    contacts_site = True
        if contacts_any_target:
            binder_at_target += 1
        if contacts_site:
            binder_at_site += 1

    site_iface_frac = binder_at_site / binder_at_target if binder_at_target > 0 else 0.0

    # Centroid distance: center of binder INTERFACE residues vs center of site
    # Using only interface residues avoids bias from binder core atoms pulling centroid away
    iface_binder_coords = []
    for b_resnum, b_coords_list in binder_res_atoms.items():
        b_arr = np.array(b_coords_list)
        for t_resnum, t_coords_list in target_all_atoms.items():
            t_arr = np.array(t_coords_list)
            diffs = b_arr[:, None, :] - t_arr[None, :, :]
            if float(np.sqrt((diffs ** 2).sum(axis=2)).min()) <= interface_dist:
                iface_binder_coords.extend(b_coords_list)
                break
    # Compute both centroid variants:
    # 1) Heavy-atom centroid: all heavy atoms of site residues (already collected)
    site_heavy_coords = []
    for resnum, x, y, z in target_atoms:
        if resnum in site_set:
            site_heavy_coords.append([x, y, z])

    # 2) CA centroid: one CA per residue, equal weight — true geometric center
    site_ca_coords = []
    is_cif = str(struct_file).endswith(".cif")
    try:
        with open(struct_file) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                if is_cif:
                    parts = line.split()
                    if len(parts) < 13:
                        continue
                    atom_name = parts[3]
                    chain = parts[9] if len(parts) > 9 else parts[6]
                    resnum_val = int(parts[7] if len(parts) > 9 else parts[8])
                    x_val, y_val, z_val = float(parts[10]), float(parts[11]), float(parts[12])
                else:
                    atom_name = line[12:16].strip()
                    chain = line[21]
                    resnum_val = int(line[22:26].strip())
                    x_val, y_val, z_val = float(line[30:38]), float(line[38:46]), float(line[46:54])
                if chain == target_chain_id and resnum_val in site_set and atom_name == "CA":
                    site_ca_coords.append([x_val, y_val, z_val])
    except Exception:
        pass

    binder_iface_center = np.mean(iface_binder_coords, axis=0) if iface_binder_coords else None
    centroid_dist_heavy = float("nan")
    centroid_dist_CA = float("nan")
    site_cos_angle = float("nan")
    if binder_iface_center is not None:
        if site_heavy_coords:
            centroid_dist_heavy = float(np.linalg.norm(
                binder_iface_center - np.mean(site_heavy_coords, axis=0)))
        if site_ca_coords:
            site_center = np.mean(site_ca_coords, axis=0)
            centroid_dist_CA = float(np.linalg.norm(
                binder_iface_center - site_center))
            # Surface normal check: is the binder on the outward-facing side?
            # Outward normal = target_center → site_center (points away from protein core)
            # Binder vector = site_center → binder_interface_center
            # cos(angle) > 0 means binder approaches from solvent side
            target_center = np.mean([[x, y, z] for resnum, x, y, z in target_atoms], axis=0)
            outward = site_center - target_center
            outward_norm = np.linalg.norm(outward)
            if outward_norm > 0.1:
                outward = outward / outward_norm
                binder_vec = binder_iface_center - site_center
                binder_vec_norm = np.linalg.norm(binder_vec)
                if binder_vec_norm > 0.1:
                    site_cos_angle = float(np.dot(outward, binder_vec / binder_vec_norm))

    return {
        "site_contact_fraction": round(contact_frac, 3),
        "site_n_contacted": n_contacted,
        "site_n_total": n_site_residues,
        "site_interface_fraction": round(site_iface_frac, 3),
        "site_interface_binder_res": binder_at_site,
        "site_interface_total_res": binder_at_target,
        "site_centroid_dist_heavy": round(centroid_dist_heavy, 1),
        "site_centroid_dist_CA": round(centroid_dist_CA, 1),
        "site_cos_angle": round(site_cos_angle, 3),
    }


def geometric_site_filter(designs, site_resnums, target_residues,
                          max_dist=15.0, min_site_fraction=0.0, interface_dist=5.0,
                          dry_run=False):
    """
    Site contact filter: for each design, count how many site residues have
    at least one binder heavy atom within max_dist Å.

    Adds to each design dict:
      site_contact_fraction — fraction of site residues contacted (0-1)
      site_n_contacted      — number of site residues contacted
      site_n_total          — total number of site residues
      site_geometric_pass   — True if fraction >= min_site_fraction and > 0
      site_interface_fraction — fraction of binder interface at site (0-1)

    Uses multiprocessing for the expensive per-design distance computation.
    """
    import numpy as np
    from multiprocessing import Pool, cpu_count

    if not site_resnums or max_dist <= 0:
        return designs

    # Build renumbered site indices (for CIF files)
    renumbered_site = set()
    if target_residues:
        resnum_to_idx = {r: i + 1 for i, r in enumerate(target_residues)}
        renumbered_site = {resnum_to_idx[r] for r in site_resnums if r in resnum_to_idx}
    original_site = set(site_resnums)

    # Split site into two patches for dual-patch proximity check
    sorted_site = sorted(site_resnums)
    if len(sorted_site) < 2:
        return designs
    gaps = [(sorted_site[i+1] - sorted_site[i], i) for i in range(len(sorted_site) - 1)]
    max_gap_val, max_gap_idx = max(gaps)
    if max_gap_val <= 5:
        patches_orig = [original_site]
        patches_renum = [renumbered_site]
    else:
        patch1_orig = {r for r in sorted_site[:max_gap_idx + 1]}
        patch2_orig = {r for r in sorted_site[max_gap_idx + 1:]}
        patches_orig = [patch1_orig, patch2_orig]
        if target_residues:
            patch1_renum = {resnum_to_idx[r] for r in patch1_orig if r in resnum_to_idx}
            patch2_renum = {resnum_to_idx[r] for r in patch2_orig if r in resnum_to_idx}
            patches_renum = [patch1_renum, patch2_renum]
        else:
            patches_renum = patches_orig

    pass_counts = {}
    fail_counts = {}
    skip_counts = {}

    all_site_renum = renumbered_site
    all_site_orig = original_site
    n_site_residues = len(site_resnums)

    # Prepare worker args for parallelizable designs
    worker_indices = []
    worker_args = []
    for i, d in enumerate(designs):
        tool = d.get("tool") or _get_tool_from_design_id(d.get("design_id", "")) or "unknown"

        struct_file = None
        use_renumbered = True
        if d.get("complex_cif") and Path(d["complex_cif"]).exists():
            struct_file = d["complex_cif"]
            use_renumbered = True
        elif d.get("binder_pdb") and Path(d["binder_pdb"]).exists():
            if tool == "pxdesign":
                skip_counts[tool] = skip_counts.get(tool, 0) + 1
                d["site_geometric_pass"] = True
                continue
            struct_file = d["binder_pdb"]
            use_renumbered = False

        if not struct_file or dry_run:
            skip_counts[tool] = skip_counts.get(tool, 0) + 1
            d["site_geometric_pass"] = True
            continue

        site_set = all_site_renum if use_renumbered else all_site_orig
        worker_indices.append(i)
        worker_args.append((struct_file, site_set, n_site_residues, max_dist, interface_dist))

    # Run in parallel
    n_workers = min(max(1, cpu_count() - 2), 12, len(worker_args) or 1)
    if worker_args:
        log(f"  Geometric filter: processing {len(worker_args)} designs on {n_workers} CPUs...")
        with Pool(n_workers) as pool:
            results = pool.map(_compute_site_metrics, worker_args)
    else:
        results = []

    # Apply results back to designs
    for idx, result in zip(worker_indices, results):
        d = designs[idx]
        tool = d.get("tool") or _get_tool_from_design_id(d.get("design_id", "")) or "unknown"

        if result is None:
            skip_counts[tool] = skip_counts.get(tool, 0) + 1
            d["site_geometric_pass"] = True
            continue

        d.update(result)
        contact_frac = result["site_contact_fraction"]
        n_contacted = result["site_n_contacted"]

        if n_contacted == 0 or contact_frac < min_site_fraction:
            d["site_geometric_pass"] = False
            fail_counts[tool] = fail_counts.get(tool, 0) + 1
        else:
            d["site_geometric_pass"] = True
            pass_counts[tool] = pass_counts.get(tool, 0) + 1

    # Log summary
    all_tools = sorted(set(list(pass_counts) + list(fail_counts) + list(skip_counts)))
    for tool in all_tools:
        p = pass_counts.get(tool, 0)
        f = fail_counts.get(tool, 0)
        s = skip_counts.get(tool, 0)
        log(f"  {tool}: {p} pass, {f} fail (0/{n_site_residues} contacts ≤{max_dist}Å), {s} skip")

    fracs = [d.get("site_contact_fraction", 0) for d in designs
             if d.get("site_geometric_pass") is True and "site_contact_fraction" in d]
    if fracs:
        log(f"  Contact fractions: min={min(fracs):.2f}, median={sorted(fracs)[len(fracs)//2]:.2f}, "
            f"max={max(fracs):.2f} (n={len(fracs)})")

    total_f = sum(fail_counts.values())
    if total_f:
        log(f"  Geometric filter: {total_f} off-site designs (0 site contacts) will skip Rosetta")

    return designs


# ── Refolding RMSD ────────────────────────────────────────────────────────────

def compute_refolding_rmsd(designs, dry_run=False):
    """Compute CA RMSD between ESMFold (binder alone) and Boltz-2 (binder in complex).

    High RMSD (>3-4 Å) indicates the binder only folds correctly when bound
    to the target — likely disordered on its own.

    Adds to each design dict:
        refolding_rmsd — CA RMSD in Angstrom (nan if structures unavailable)
    """
    import numpy as np

    def _parse_ca_coords_pdb(path):
        """Extract CA coords from PDB, return dict chain -> [(resnum, x, y, z)]."""
        chains = {}
        with open(path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    ch = line[21]
                    rn = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    chains.setdefault(ch, []).append((rn, x, y, z))
        return chains

    def _parse_ca_coords_cif(path):
        """Extract CA coords from CIF, return dict chain -> [(resnum, x, y, z)]."""
        chains = {}
        with open(path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                parts = line.split()
                if len(parts) < 13 or parts[3] != "CA":
                    continue
                ch = parts[9] if len(parts) > 9 else parts[6]
                rn = int(parts[7] if len(parts) > 9 else parts[8])
                x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                chains.setdefault(ch, []).append((rn, x, y, z))
        return chains

    def _kabsch_rmsd(coords1, coords2):
        """Compute RMSD after optimal superposition (Kabsch algorithm)."""
        c1 = np.array(coords1)
        c2 = np.array(coords2)
        c1 -= c1.mean(axis=0)
        c2 -= c2.mean(axis=0)
        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, 1, d])
        R = Vt.T @ sign_matrix @ U.T
        c1_rot = c1 @ R.T
        return float(np.sqrt(np.mean(np.sum((c1_rot - c2) ** 2, axis=1))))

    n_computed = 0
    n_skipped = 0
    for d in designs:
        if dry_run:
            continue
        esm_pdb = d.get("esmfold_pdb")
        boltz_cif = d.get("complex_cif") or d.get("_orig_complex_cif")

        if not esm_pdb or not boltz_cif:
            n_skipped += 1
            continue
        if not Path(esm_pdb).exists() or not Path(boltz_cif).exists():
            n_skipped += 1
            continue

        try:
            # ESMFold PDB: single chain, binder only
            esm_chains = _parse_ca_coords_pdb(esm_pdb)
            esm_cas = list(esm_chains.values())[0] if esm_chains else []

            # Boltz CIF: multi-chain complex, binder is the shorter chain
            cif_chains = _parse_ca_coords_cif(boltz_cif)
            if len(cif_chains) < 2:
                n_skipped += 1
                continue
            binder_chain = min(cif_chains, key=lambda c: len(cif_chains[c]))
            boltz_cas = cif_chains[binder_chain]

            # Match by residue count (both should be 1-based sequential)
            n = min(len(esm_cas), len(boltz_cas))
            if n < 10:
                n_skipped += 1
                continue

            esm_xyz = [[x, y, z] for _, x, y, z in esm_cas[:n]]
            boltz_xyz = [[x, y, z] for _, x, y, z in boltz_cas[:n]]

            d["refolding_rmsd"] = round(_kabsch_rmsd(esm_xyz, boltz_xyz), 2)
            n_computed += 1
        except Exception:
            n_skipped += 1

    if n_computed or n_skipped:
        rmsds = [d["refolding_rmsd"] for d in designs
                 if "refolding_rmsd" in d and d["refolding_rmsd"] == d["refolding_rmsd"]]
        if rmsds:
            log(f"  Refolding RMSD: computed {n_computed}, skipped {n_skipped}")
            log(f"  RMSD stats: min={min(rmsds):.1f}, median={sorted(rmsds)[len(rmsds)//2]:.1f}, "
                f"max={max(rmsds):.1f} Å (n={len(rmsds)})")
            n_good = sum(1 for r in rmsds if r <= 3.0)
            log(f"  {n_good}/{len(rmsds)} designs with refolding RMSD ≤ 3.0 Å")

    return designs


# ── Solubility prediction (NetSolP) ──────────────────────────────────────────

_NETSOLP_DIR = os.environ.get("NETSOLP_DIR",
    os.path.join(os.environ.get("BINDER_SOFTWARE_DIR", os.path.expanduser("~/data/software")),
                 "NetSolP-1.0", "PredictionServer"))


def compute_solubility(designs, dry_run=False):
    """Predict E.coli solubility using NetSolP (Distilled model).

    Runs NetSolP via subprocess in the esmfold conda env.
    Adds to each design dict:
        netsolp_solubility — predicted solubility probability (0-1, higher = more soluble)
    """
    if dry_run:
        return designs

    predict_script = os.path.join(_NETSOLP_DIR, "predict.py")
    if not os.path.exists(predict_script):
        log("  NetSolP not installed — skipping solubility prediction")
        return designs

    # Check for ONNX model
    model_file = os.path.join(_NETSOLP_DIR, "models", "Solubility_ESM1b_distilled_quantized.onnx")
    if not os.path.exists(model_file):
        log("  NetSolP ONNX models not found — skipping solubility prediction")
        return designs

    # Write sequences to temp FASTA
    import tempfile
    seqs = {}
    for d in designs:
        did = d.get("design_id", "")
        seq = d.get("binder_sequence", "")
        if did and seq and len(seq) > 5:
            seqs[did] = seq

    if not seqs:
        return designs

    fd, fasta_path = tempfile.mkstemp(suffix=".fasta")
    os.close(fd)
    fd2, out_csv = tempfile.mkstemp(suffix=".csv")
    os.close(fd2)

    try:
        with open(fasta_path, "w") as f:
            for did, seq in seqs.items():
                f.write(f">{did}\n{seq}\n")

        # Run NetSolP in esmfold env
        cmd = [
            "conda", "run", "--no-capture-output", "-n", "esmfold",
            "python", predict_script,
            "--FASTA_PATH", fasta_path,
            "--OUTPUT_PATH", out_csv,
            "--MODEL_TYPE", "Distilled",
            "--PREDICTION_TYPE", "S",
            "--MODELS_PATH", os.path.join(_NETSOLP_DIR, "models"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if r.returncode != 0:
            log(f"  NetSolP failed: {r.stderr[-500:]}")
            return designs

        # Parse output CSV
        import csv as _csv
        sol_scores = {}
        with open(out_csv) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                sid = row.get("sid", row.get("ID", ""))
                sol = row.get("predicted_solubility", "")
                if sid and sol:
                    try:
                        sol_scores[sid] = float(sol)
                    except ValueError:
                        pass

        # Apply to designs
        n_scored = 0
        for d in designs:
            did = d.get("design_id", "")
            if did in sol_scores:
                d["netsolp_solubility"] = round(sol_scores[did], 3)
                n_scored += 1

        if n_scored:
            vals = [d["netsolp_solubility"] for d in designs
                    if "netsolp_solubility" in d]
            log(f"  NetSolP: {n_scored} designs scored")
            log(f"  Solubility: min={min(vals):.3f}, median={sorted(vals)[len(vals)//2]:.3f}, "
                f"max={max(vals):.3f}")

    except Exception as e:
        log(f"  NetSolP error: {e}")
    finally:
        for p in (fasta_path, out_csv):
            try:
                os.remove(p)
            except OSError:
                pass

    return designs


# ── Rosetta interface scoring ──────────────────────────────────────────────────

def rosetta_score_interfaces(designs, val_dir, dry_run=False):
    """
    Score Boltz-validated complexes with PyRosetta InterfaceAnalyzerMover.
    Runs in the BindCraft conda env (which has PyRosetta installed).

    Adds to each design dict:
        rosetta_dG, rosetta_sc, rosetta_hbonds, rosetta_bunsats,
        rosetta_dsasa, rosetta_dg_dsasa, rosetta_packstat, rosetta_sap
    """
    # Collect designs that have a complex structure file (CIF or PDB fallback)
    to_score = []
    for d in designs:
        struct = d.get("complex_cif") or d.get("binder_pdb")
        if struct and Path(struct).exists():
            to_score.append(d)

    if not to_score:
        log("Rosetta: no complex structures to score, skipping.")
        return designs

    log(f"Rosetta: scoring {len(to_score)} complex structures...")

    rosetta_dir = Path(val_dir) / "rosetta"
    rosetta_dir.mkdir(parents=True, exist_ok=True)

    # Build list of (design_id, struct_path) pairs — prefer complex_cif, fall back to binder_pdb
    jobs = [(d["design_id"], d.get("complex_cif") or d.get("binder_pdb")) for d in to_score]
    jobs_json = json.dumps(jobs)
    dalpha_path = str(Path(BINDCRAFT_DIR) / "functions" / "DAlphaBall.gcc")
    scores_out = str(rosetta_dir / "scores.json")

    # The Rosetta scoring script runs each design in a subprocess to isolate
    # segfaults (PyRosetta can crash on non-standard CIFs like PXDesign's A0/B0
    # chains).  Results are saved incrementally so a crash doesn't lose progress.
    single_score_script = rosetta_dir / "score_single.py"
    single_score_content = (
        "import json, sys, os, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "import pyrosetta as pr\n"
        "from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover\n"
        "from pyrosetta.rosetta.protocols.relax import FastRelax\n"
        "from pyrosetta.rosetta.core.kinematics import MoveMap\n"
        "from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects\n"
        "\n"
        f"pr.init('-mute all -holes:dalphaball {dalpha_path} "
        "-corrections::beta_nov16 true -relax:default_repeats 1', silent=True)\n"
        "\n"
        "design_id = sys.argv[1]\n"
        "struct_path = sys.argv[2]\n"
        "out_path = sys.argv[3]\n"
        "\n"
        "# Sanitize non-standard chain IDs (e.g., PXDesign's A0/B0 -> A/B)\n"
        "# PyRosetta segfaults on multi-character chain IDs in CIF files.\n"
        "# Use gemmi to convert CIF→PDB with single-char chains.\n"
        "import tempfile\n"
        "clean_path = struct_path\n"
        "if struct_path.endswith('.cif'):\n"
        "    import gemmi\n"
        "    st = gemmi.read_structure(struct_path)\n"
        "    needs_rename = any(len(ch.name) > 1 for mod in st for ch in mod)\n"
        "    if needs_rename:\n"
        "        for model in st:\n"
        "            for chain in model:\n"
        "                if len(chain.name) > 1:\n"
        "                    chain.name = chain.name[0]\n"
        "        fd, clean_path = tempfile.mkstemp(suffix='.pdb')\n"
        "        os.close(fd)\n"
        "        st.write_pdb(clean_path)\n"
        "\n"
        "scorefxn = pr.get_fa_scorefxn()\n"
        "relax = FastRelax()\n"
        "relax.set_scorefxn(scorefxn)\n"
        "relax.constrain_relax_to_start_coords(True)\n"
        "mm = MoveMap()\n"
        "mm.set_bb(True); mm.set_chi(True); mm.set_jump(True)\n"
        "relax.set_movemap(mm)\n"
        "\n"
        "pose = pr.pose_from_file(clean_path)\n"
        "# Clean up temp file if created\n"
        "if clean_path != struct_path:\n"
        "    os.remove(clean_path)\n"
        "chains = set()\n"
        "for i in range(1, pose.total_residue() + 1):\n"
        "    chains.add(pose.pdb_info().chain(i))\n"
        "chains = sorted(chains)\n"
        "if len(chains) >= 2:\n"
        "    interface = f'{chains[0]}_{chains[1]}'\n"
        "else:\n"
        "    interface = 'A_B'\n"
        "\n"
        "relax.apply(pose)\n"
        "iam = InterfaceAnalyzerMover()\n"
        "iam.set_interface(interface)\n"
        "iam.set_scorefunction(scorefxn)\n"
        "iam.set_compute_packstat(True)\n"
        "iam.set_compute_interface_energy(True)\n"
        "iam.set_calc_dSASA(True)\n"
        "iam.set_calc_hbond_sasaE(True)\n"
        "iam.set_compute_interface_sc(True)\n"
        "iam.set_pack_separated(True)\n"
        "iam.apply(pose)\n"
        "data = iam.get_all_data()\n"
        "buns = XmlObjects.static_get_filter(\n"
        "    '<BuriedUnsatHbonds report_all_heavy_atom_unsats=\"true\" '\n"
        "    'scorefxn=\"scorefxn\" ignore_surface_res=\"false\" '\n"
        "    'use_ddG_style=\"true\" dalphaball_sasa=\"1\" probe_radius=\"1.1\" '\n"
        "    'burial_cutoff_apo=\"0.2\" confidence=\"0\" />')\n"
        "# SAP score (Surface Aggregation Propensity)\n"
        "try:\n"
        "    from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import calculate_sap\n"
        "    from pyrosetta.rosetta.core.select.residue_selector import TrueResidueSelector\n"
        "    all_res = TrueResidueSelector()\n"
        "    sap_total = calculate_sap(pose, all_res, all_res, all_res)\n"
        "except Exception:\n"
        "    sap_total = float('nan')\n"
        "\n"
        "result = {\n"
        "    'rosetta_dG': iam.get_interface_dG(),\n"
        "    'rosetta_sc': data.sc_value,\n"
        "    'rosetta_hbonds': data.interface_hbonds,\n"
        "    'rosetta_bunsats': buns.report_sm(pose),\n"
        "    'rosetta_dsasa': iam.get_interface_delta_sasa(),\n"
        "    'rosetta_dg_dsasa': data.dG_dSASA_ratio * 100,\n"
        "    'rosetta_packstat': iam.get_interface_packstat(),\n"
        "    'rosetta_sap': sap_total,\n"
        "}\n"
        "with open(out_path, 'w') as f:\n"
        "    json.dump(result, f)\n"
        "print(f'dG={result[\"rosetta_dG\"]:.1f}  sc={data.sc_value:.3f}')\n"
    )
    single_score_script.write_text(single_score_content)

    # Determine number of parallel workers.
    # Each PyRosetta process uses ~1-2 GB RAM. Use 50% of CPUs, capped by
    # available RAM (2 GB per worker headroom).
    total_cpus = os.cpu_count() or 4
    cpu_cap = max(4, total_cpus // 2)
    try:
        with open("/proc/meminfo") as mf:
            for line in mf:
                if line.startswith("MemAvailable:"):
                    avail_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                avail_gb = 999
        ram_cap = max(4, int(avail_gb / 2))
    except Exception:
        ram_cap = cpu_cap
    n_workers = min(cpu_cap, ram_cap, len(jobs))

    # Orchestrator script: runs designs in parallel using concurrent.futures
    script_lines = [
        "import json, sys, os, subprocess, time",
        "from concurrent.futures import ProcessPoolExecutor, as_completed",
        "",
        f"jobs = {jobs_json}",
        f"scores_out = '{scores_out}'",
        f"single_script = '{single_score_script}'",
        f"n_workers = {n_workers}",
        "results = {}",
        "",
        "# Load any existing partial results",
        "if os.path.exists(scores_out):",
        "    with open(scores_out) as f:",
        "        results = json.load(f)",
        "",
        "# Filter out already-cached jobs",
        "pending = [(did, sp) for did, sp in jobs if did not in results or results[did] is None]",
        "cached = len(jobs) - len(pending)",
        "if cached:",
        "    print(f'Rosetta: {cached} cached, {len(pending)} to score with {n_workers} workers', flush=True)",
        "else:",
        "    print(f'Rosetta: scoring {len(pending)} designs with {n_workers} parallel workers', flush=True)",
        "",
        "def score_one(args):",
        "    design_id, struct_path = args",
        "    result_path = os.path.join(os.path.dirname(scores_out), f'{design_id}.json')",
        "    try:",
        "        proc = subprocess.run(",
        "            ['python', single_script, design_id, struct_path, result_path],",
        "            capture_output=True, text=True, timeout=600)",
        "        if proc.returncode == 0 and os.path.exists(result_path):",
        "            with open(result_path) as f:",
        "                data = json.load(f)",
        "            os.remove(result_path)",
        "            return (design_id, data, proc.stdout.strip())",
        "        else:",
        "            err = (proc.stderr or proc.stdout or '')[-200:]",
        "            return (design_id, None, f'FAILED (exit {proc.returncode}): {err}')",
        "    except subprocess.TimeoutExpired:",
        "        return (design_id, None, 'TIMEOUT')",
        "    except Exception as e:",
        "        return (design_id, None, f'ERROR: {e}')",
        "",
        "n_done = 0",
        "n_total = len(pending)",
        "t0 = time.time()",
        "with ProcessPoolExecutor(max_workers=n_workers) as pool:",
        "    futures = {pool.submit(score_one, job): job[0] for job in pending}",
        "    for future in as_completed(futures):",
        "        design_id, data, msg = future.result()",
        "        results[design_id] = data",
        "        n_done += 1",
        "        elapsed = time.time() - t0",
        "        rate = n_done / elapsed * 60 if elapsed > 0 else 0",
        "        if data:",
        "            print(f'  [{n_done}/{n_total}] {design_id}: {msg}  ({rate:.0f}/min)', flush=True)",
        "        else:",
        "            print(f'  [{n_done}/{n_total}] {design_id}: {msg}', flush=True)",
        "",
        "        # Save incrementally every 10 completions",
        "        if n_done % 10 == 0 or n_done == n_total:",
        "            with open(scores_out, 'w') as f:",
        "                json.dump(results, f)",
        "",
        "# Final save",
        "with open(scores_out, 'w') as f:",
        "    json.dump(results, f)",
        "",
        "scored = sum(1 for v in results.values() if v is not None)",
        "elapsed = time.time() - t0",
        "print(f'Rosetta: {scored}/{len(results)} scored in {elapsed/60:.1f} min '",
        "      f'({scored/elapsed*60:.0f}/min with {n_workers} workers).')",
    ]
    script_content = "\n".join(script_lines) + "\n"

    score_script = rosetta_dir / "batch_score.py"
    score_script.write_text(script_content)

    if dry_run:
        log(f"  [DRY RUN] conda run -n BindCraft python {score_script}")
        return designs

    try:
        run_cmd(["conda", "run", "--no-capture-output", "-n", "BindCraft",
                 "python", str(score_script)],
                timeout=max(7200, len(to_score) * 180))  # ~3 min/design, min 2h
    except RuntimeError as e:
        log(f"Rosetta scoring FAILED: {e}")
        for d in to_score:
            for k in ("rosetta_dG", "rosetta_sc", "rosetta_hbonds", "rosetta_bunsats",
                       "rosetta_dsasa", "rosetta_dg_dsasa", "rosetta_packstat", "rosetta_sap"):
                d[k] = float("nan")
        return designs

    # Parse results
    scores_path = rosetta_dir / "scores.json"
    if scores_path.exists():
        with open(scores_path) as f:
            results = json.load(f)
    else:
        log("Rosetta: scores.json not found after run.")
        results = {}

    scored = 0
    for d in designs:
        did = d["design_id"]
        r = results.get(did)
        if r and isinstance(r, dict):
            for k, v in r.items():
                d[k] = _safe_float(v)
            scored += 1
        else:
            for k in ("rosetta_dG", "rosetta_sc", "rosetta_hbonds", "rosetta_bunsats",
                       "rosetta_dsasa", "rosetta_dg_dsasa", "rosetta_packstat", "rosetta_sap"):
                d[k] = float("nan")

    log(f"Rosetta: scored {scored}/{len(to_score)} structures successfully.")
    return designs

# ── Ranking & output ───────────────────────────────────────────────────────────

def compute_combined_score(d, w_plddt=0.4, w_iptm=0.5, w_dg=0.1, reprediction=True, ss_bias="balanced"):
    """
    Combined quality score:
        w_plddt*(binder_pLDDT/100) + w_iptm*iPTM + w_dg*dG_norm
        - site_pae_penalty (penalizes off-site binding)

    When reprediction=True (default, original behavior):
        Uses boltz_iptm and boltz_binder_plddt directly.

    When reprediction=False:
        Resolves iPTM and pLDDT from native tool scores when Boltz-2 scores
        are unavailable (for tools in IPTM_NATIVE_TOOLS). Site PAE penalty
        is only applied if boltz_site_mean_pae is available (backbone-only
        tools that went through Boltz-2).

    dG_norm: Rosetta dG clamped to [-40, 0] and normalized to [0, 1].
    If Rosetta dG is unavailable, only pLDDT + iPTM components are used
    (re-normalized to sum to w_plddt + w_iptm).

    Site PAE penalty: if boltz_site_mean_pae is available and high (>4.0),
    the score is penalized proportionally (0.02 per unit above 4.0).
    This down-ranks designs that dock away from the specified binding site.
    Penalty is capped at 0.25 to avoid completely zeroing out otherwise
    good designs.

    Returns nan if no iPTM or pLDDT can be resolved.
    Also stores _resolved_iptm and _iptm_source on the design dict.
    """
    if reprediction:
        # Original behavior: use Boltz-2 scores directly
        plddt = d.get("boltz_binder_plddt", float("nan"))
        iptm  = d.get("boltz_iptm",         float("nan"))
        if plddt != plddt or iptm != iptm:  # nan check
            d["_resolved_iptm"] = float("nan")
            d["_iptm_source"] = "N/A"
            return float("nan")
        d["_resolved_iptm"] = iptm
        d["_iptm_source"] = "boltz"
    else:
        # Resolve iPTM: prefer boltz_iptm, fall back to native tool score
        tool_prefix = _get_tool_from_design_id(d.get("design_id", ""))
        iptm = d.get("boltz_iptm", float("nan"))
        iptm_source = "boltz"

        if iptm != iptm:  # boltz_iptm is NaN — try native
            if tool_prefix == "bindcraft":
                iptm = d.get("bc_i_ptm", float("nan"))
                iptm_source = "native_bindcraft"
            elif tool_prefix == "boltzgen":
                iptm = d.get("bg_design_to_target_iptm", float("nan"))
                iptm_source = "native_boltzgen"
            elif tool_prefix == "pxdesign":
                iptm = d.get("px_iptm", float("nan"))
                iptm_source = "native_pxdesign"
            elif tool_prefix == "proteina_complexa":
                iptm = d.get("pc_iptm", float("nan"))
                iptm_source = "native_complexa"

        if iptm != iptm:  # still NaN
            d["_resolved_iptm"] = float("nan")
            d["_iptm_source"] = "N/A"
            return float("nan")

        # Resolve pLDDT: prefer boltz_binder_plddt, fall back to native, then esmfold
        # Native tool pLDDTs may be on 0-1 scale; convert to 0-100 for the formula.
        plddt = d.get("boltz_binder_plddt", float("nan"))
        if plddt != plddt:  # NaN — try native
            if tool_prefix == "bindcraft":
                raw = d.get("bc_binder_plddt", float("nan"))
                # BindCraft stores pLDDT as 0-1; convert to 0-100
                if raw == raw and raw <= 1.0:
                    plddt = raw * 100.0
                else:
                    plddt = raw
            elif tool_prefix == "boltzgen":
                raw = d.get("bg_complex_plddt", float("nan"))
                # BoltzGen stores pLDDT as 0-1; convert to 0-100
                if raw == raw and raw <= 1.0:
                    plddt = raw * 100.0
                else:
                    plddt = raw
            elif tool_prefix == "pxdesign":
                raw = d.get("px_plddt", float("nan"))
                # PXDesign stores pLDDT as 0-1; convert to 0-100
                if raw == raw and raw <= 1.0:
                    plddt = raw * 100.0
                else:
                    plddt = raw
            elif tool_prefix == "proteina_complexa":
                raw = d.get("pc_plddt", float("nan"))
                if raw == raw and raw <= 1.0:
                    plddt = raw * 100.0
                else:
                    plddt = raw
        if plddt != plddt:  # last resort: ESMFold
            plddt = d.get("esmfold_plddt", float("nan"))

        if plddt != plddt:  # still NaN
            d["_resolved_iptm"] = float("nan")
            d["_iptm_source"] = "N/A"
            return float("nan")

        d["_resolved_iptm"] = iptm
        d["_iptm_source"] = iptm_source

    score = w_plddt * (plddt / 100.0) + w_iptm * iptm

    dG = d.get("rosetta_dG", float("nan"))
    if w_dg > 0 and dG == dG:  # dG available and weight > 0
        # Clamp to [-40, 0] and normalize: more negative = better = higher score
        dG_clamped = max(-40.0, min(0.0, dG))
        dG_norm = -dG_clamped / 40.0  # maps -40\u21921.0, 0\u21920.0
        score += w_dg * dG_norm

    # Site-specificity penalty — tighter threshold to penalize off-site binding
    # Penalty starts at sitePAE > 4.0 (was 10.0), scales at 0.02/unit (was 0.005),
    # capped at 0.25 (was 0.15). A design with sitePAE=6 gets -0.04, sitePAE=10 gets -0.12.
    if reprediction:
        # Always apply if available
        site_pae = d.get("boltz_site_mean_pae", float("nan"))
        if site_pae == site_pae:  # not nan
            if site_pae > 4.0:
                penalty = min(0.25, (site_pae - 4.0) * 0.02)
                score -= penalty
    else:
        # Only apply for backbone-only tools that went through Boltz-2
        # Native-iPTM tools don't have Boltz PAE matrices
        tool_prefix = _get_tool_from_design_id(d.get("design_id", ""))
        if tool_prefix not in IPTM_NATIVE_TOOLS:
            site_pae = d.get("boltz_site_mean_pae", float("nan"))
            if site_pae == site_pae:  # not nan
                if site_pae > 4.0:
                    penalty = min(0.25, (site_pae - 4.0) * 0.02)
                    score -= penalty

    # SS bias penalty — scaled to meaningfully affect rankings (0.90-0.95 range)
    if ss_bias != "balanced":
        helix_frac = d.get("binder_helix_frac", float("nan"))
        sheet_frac = d.get("binder_sheet_frac", float("nan"))
        params = SS_BIAS_PARAMS[ss_bias]
        if helix_frac == helix_frac and sheet_frac == sheet_frac:  # not nan
            if ss_bias == "beta":
                if helix_frac > params["penalty_helix_above"]:
                    score -= 0.15 * (helix_frac - params["penalty_helix_above"])
                if sheet_frac < params["penalty_sheet_below"]:
                    score -= 0.10 * (params["penalty_sheet_below"] - sheet_frac)
            elif ss_bias == "helix":
                if sheet_frac > params["penalty_sheet_above"]:
                    score -= 0.15 * (sheet_frac - params["penalty_sheet_above"])
                if helix_frac < params["penalty_helix_below"]:
                    score -= 0.10 * (params["penalty_helix_below"] - helix_frac)

    return score


def rank_designs(all_designs, score_weights=(0.4, 0.5, 0.1),
                 filter_interface_pae=None, filter_site_pae=None,
                 reprediction=True, ss_bias="balanced"):
    """
    Compute combined_score, optionally filter by boltz_min_interface_pae
    and/or boltz_site_mean_pae, rank validated designs, return sorted list.

    Parameters
    ----------
    all_designs : list of design dicts (with boltz_* and rosetta_* fields)
    score_weights : (w_plddt, w_iptm, w_dg) tuple
    filter_interface_pae : float or None -- if set, drop validated designs
        whose boltz_min_interface_pae > threshold (high pAE = poor interface)
    filter_site_pae : float or None -- if set, drop validated designs
        whose boltz_site_mean_pae > threshold (high site pAE = off-site binding)
    reprediction : bool -- if True, original behavior. If False, skip PAE
        filters for native-iPTM tools (they lack Boltz PAE matrices).
    """
    w_plddt, w_iptm, w_dg = score_weights
    for d in all_designs:
        d["combined_score"] = compute_combined_score(
            d, w_plddt=w_plddt, w_iptm=w_iptm, w_dg=w_dg, reprediction=reprediction,
            ss_bias=ss_bias)

    validated   = [d for d in all_designs if d["combined_score"] == d["combined_score"]]
    unvalidated = [d for d in all_designs if d["combined_score"] != d["combined_score"]]

    # Apply interface pAE filter
    if filter_interface_pae is not None:
        before = len(validated)
        passed_pae = []
        for d in validated:
            tool_prefix = _get_tool_from_design_id(d.get("design_id", ""))
            # Skip PAE filter for native-iPTM tools when reprediction is off
            if not reprediction and tool_prefix in IPTM_NATIVE_TOOLS:
                passed_pae.append(d)
                continue
            min_pae = d.get("boltz_min_interface_pae", float("nan"))
            if min_pae != min_pae or min_pae > filter_interface_pae:
                # Move to unvalidated (no combined_score rank)
                d["combined_score"] = float("nan")
                unvalidated.append(d)
            else:
                passed_pae.append(d)
        n_filtered = before - len(passed_pae)
        if n_filtered > 0:
            log(f"  Rank: filtered {n_filtered} designs by interface pAE > {filter_interface_pae}")
        validated = passed_pae

    # Apply site-specific pAE filter (off-site binding detection)
    if filter_site_pae is not None:
        before = len(validated)
        passed_site = []
        for d in validated:
            tool_prefix = _get_tool_from_design_id(d.get("design_id", ""))
            # Skip PAE filter for native-iPTM tools when reprediction is off
            if not reprediction and tool_prefix in IPTM_NATIVE_TOOLS:
                passed_site.append(d)
                continue
            site_pae = d.get("boltz_site_mean_pae", float("nan"))
            if site_pae == site_pae and site_pae > filter_site_pae:
                d["combined_score"] = float("nan")
                unvalidated.append(d)
            else:
                passed_site.append(d)
        n_filtered = before - len(passed_site)
        if n_filtered > 0:
            log(f"  Rank: filtered {n_filtered} designs by site pAE > {filter_site_pae}")
        validated = passed_site

    # SS bias hard-filter: drop designs that strongly violate the bias
    if ss_bias != "balanced":
        before = len(validated)
        passed_ss = []
        params = SS_BIAS_PARAMS[ss_bias]
        for d in validated:
            helix = d.get("binder_helix_frac", float("nan"))
            sheet = d.get("binder_sheet_frac", float("nan"))
            if helix != helix or sheet != sheet:  # no SS data — keep
                passed_ss.append(d)
                continue
            drop = False
            if ss_bias == "beta" and helix > 0.70:
                drop = True  # reject heavily helical designs in beta mode
            elif ss_bias == "helix" and sheet > 0.50:
                drop = True  # reject heavily sheet designs in helix mode
            if drop:
                d["combined_score"] = float("nan")
                unvalidated.append(d)
            else:
                passed_ss.append(d)
        n_filtered = before - len(passed_ss)
        if n_filtered > 0:
            log(f"  Rank: SS hard-filter removed {n_filtered} designs "
                f"(ss_bias={ss_bias})")
        validated = passed_ss

    # Geometric site proximity filter: exclude designs that explicitly failed
    # (site_geometric_pass=False) OR that were never checked (no Boltz CIF)
    # when min_site_fraction > 0.
    before = len(validated)
    passed_geo = []
    for d in validated:
        if d.get("site_geometric_pass") is False:
            d["combined_score"] = float("nan")
            unvalidated.append(d)
        elif d.get("boltz_iptm") != d.get("boltz_iptm"):
            # No Boltz-2 prediction → no complex structure → cannot verify
            # site contact. Exclude from ranking.
            d["combined_score"] = float("nan")
            unvalidated.append(d)
        else:
            passed_geo.append(d)
    n_filtered = before - len(passed_geo)
    if n_filtered > 0:
        log(f"  Rank: geometric filter removed {n_filtered} off-site designs")
    validated = passed_geo

    validated.sort(key=lambda x: x["combined_score"], reverse=True)
    for i, d in enumerate(validated):
        d["rank"] = i + 1
    for d in unvalidated:
        d["rank"] = None

    return validated + unvalidated


def write_rankings_csv(designs, out_path):
    """
    Write rankings.csv with all core and tool-specific columns.

    Columns are present for every row; tool-specific columns are empty/nan
    when not applicable to that design's source tool.
    """
    # Core columns always present
    core_fields = [
        "rank", "design_id", "binder_sequence", "tool", "binder_length",
        "combined_score", "iptm_source",
        "boltz_iptm", "boltz_ptm", "boltz_complex_plddt", "boltz_iplddt",
        "boltz_binder_plddt", "boltz_min_interface_pae", "boltz_mean_interface_pae",
        "boltz_site_min_pae", "boltz_site_mean_pae",
        "esmfold_plddt",
        "native_score", "native_score_name",
        "site_contact_fraction", "site_n_contacted", "site_n_total", "site_geometric_pass",
        "site_interface_fraction", "site_interface_binder_res", "site_interface_total_res",
        "site_centroid_dist_heavy", "site_centroid_dist_CA",
        "site_cos_angle",
        "refolding_rmsd",
    ]

    # RFdiffusion-specific
    rfd_fields = ["ligandmpnn_seq_rec"]

    # BoltzGen-specific (bg_ prefix)
    bg_fields = [
        "bg_design_to_target_iptm", "bg_min_design_to_target_pae", "bg_design_ptm",
        "bg_complex_plddt", "bg_complex_iplddt", "bg_filter_rmsd",
        "bg_delta_sasa_refolded", "bg_plip_hbonds_refolded", "bg_plip_saltbridge_refolded",
        "bg_design_hydrophobicity", "bg_design_largest_hydrophobic_patch_refolded",
        "bg_liability_score", "bg_liability_num_violations",
        "bg_liability_high_severity_violations", "bg_liability_violations_summary",
        "bg_quality_score", "bg_helix", "bg_loop", "bg_sheet",
    ]

    # BindCraft-specific (bc_ prefix) — subset of numeric averages
    bc_fields = [
        "bc_plddt", "bc_ptm", "bc_i_ptm", "bc_pae", "bc_i_pae", "bc_i_plddt",
        "bc_binder_energy_score", "bc_surface_hydrophobicity",
        "bc_shapecomplementarity", "bc_packstat",
        "bc_dg", "bc_dsasa", "bc_interface_hydrophobicity",
        "bc_n_interfaceresidues", "bc_n_interfacehbonds",
        "bc_binder_helix_pct", "bc_binder_betasheet_pct", "bc_binder_loop_pct",
        "bc_hotspot_rmsd", "bc_target_rmsd",
        "bc_binder_plddt", "bc_binder_ptm",
        "bc_mpnn_score", "bc_mpnn_seq_recovery",
    ]

    # PXDesign-specific (px_ prefix)
    px_fields = [
        "px_iptm", "px_ipae", "px_plddt", "px_ptm", "px_rmsd",
        "px_cluster_id", "px_filter_level",
    ]

    # Proteina Complexa-specific (pc_ prefix)
    pc_fields = [
        "pc_iptm", "pc_ipae", "pc_plddt", "pc_ptm", "pc_scrmsd",
        "pc_sc", "pc_hbonds", "pc_bunsats",
    ]

    # Rosetta interface scoring
    rosetta_fields = [
        "rosetta_dG", "rosetta_sc", "rosetta_hbonds", "rosetta_bunsats",
        "rosetta_dsasa", "rosetta_dg_dsasa", "rosetta_packstat", "rosetta_sap",
        "netsolp_solubility",
    ]

    # Universal SS fraction + composition columns (computed for all tools)
    ss_fields = ["binder_helix_frac", "binder_sheet_frac", "binder_loop_frac"]
    composition_fields = [
        "binder_KE_fraction", "binder_K_count", "binder_E_count",
        "interface_KE_fraction", "interface_K_count", "interface_E_count",
        "interface_n_residues", "surface_KE_fraction",
    ]

    all_fields = core_fields + rfd_fields + bg_fields + bc_fields + px_fields + pc_fields + rosetta_fields + ss_fields + composition_fields

    float_fields = set(all_fields) - {"rank", "design_id", "binder_sequence",
                                       "native_score_name", "bg_liability_violations_summary",
                                       "px_cluster_id", "px_filter_level", "iptm_source",
                                       "site_geometric_pass"}

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for d in designs:
            # Populate iptm_source from private field set by compute_combined_score
            if "iptm_source" not in d:
                d["iptm_source"] = d.get("_iptm_source", "N/A")
            row = {}
            for k in all_fields:
                v = d.get(k, "")
                if k in float_fields and isinstance(v, float):
                    row[k] = f"{v:.4f}" if v == v else "N/A"
                elif v == "" or v is None:
                    row[k] = ""
                else:
                    row[k] = v
            writer.writerow(row)

    log(f"Rankings written → {out_path}")


def _merge_binder_with_target(binder_cif, target_pdb, out_pdb):
    """
    Merge a binder from a CIF (with cropped target) onto the full target PDB.

    PXDesign CIFs contain a small target crop (chain A0, ~16 residues) + binder
    (chain B0) in AF2's coordinate frame.  This function:
      1. Extracts CA coords of the crop from the CIF
      2. Extracts matching CA coords from the full target PDB
      3. Computes optimal superposition (Kabsch algorithm)
      4. Transforms binder atoms onto the target frame
      5. Writes a combined PDB: full target (chain A) + transformed binder (chain B)
    """
    import numpy as np

    # ── Parse CIF: get crop CAs and all binder atoms ──────────────────────
    lines = Path(binder_cif).read_text().splitlines()
    col_map = {}; col_count = 0; in_atom = False; data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            in_atom = False; col_map = {}; col_count = 0
        elif s.startswith("_atom_site."):
            in_atom = True; col_map[s] = col_count; col_count += 1
        elif in_atom and s and not s.startswith(("_", "#", "loop_", "data_")):
            data_start = i; break

    if not col_map or data_start is None:
        return False

    def gcol(name):
        return col_map.get(name, col_map.get(name.replace("label_", "auth_"), -1))

    ci_atom  = gcol("_atom_site.label_atom_id")
    ci_chain = gcol("_atom_site.label_asym_id")
    ci_res   = gcol("_atom_site.label_comp_id")
    ci_seq   = gcol("_atom_site.label_seq_id")
    ci_x     = gcol("_atom_site.Cartn_x")
    ci_y     = gcol("_atom_site.Cartn_y")
    ci_z     = gcol("_atom_site.Cartn_z")
    ci_elem  = gcol("_atom_site.type_symbol")

    if any(c < 0 for c in [ci_atom, ci_chain, ci_seq, ci_x, ci_y, ci_z]):
        return False

    crop_cas = {}       # seq_id → (x, y, z)
    binder_atoms = []   # [(atom_name, res_name, seq_id, x, y, z, element)]

    for line in lines[data_start:]:
        s = line.strip()
        if not s or s.startswith(("#", "_", "loop_", "data_")):
            break
        parts = s.split()
        chain = parts[ci_chain]
        try:
            seq_id = int(parts[ci_seq])
        except (ValueError, IndexError):
            continue
        x = float(parts[ci_x])
        y = float(parts[ci_y])
        z = float(parts[ci_z])

        if chain in ("A", "A0"):
            if parts[ci_atom] == "CA":
                crop_cas[seq_id] = np.array([x, y, z])
        elif chain in ("B", "B0"):
            elem = parts[ci_elem] if ci_elem >= 0 and ci_elem < len(parts) else parts[ci_atom][0]
            binder_atoms.append((parts[ci_atom], parts[ci_res], seq_id, x, y, z, elem))

    if len(crop_cas) < 3 or not binder_atoms:
        return False

    # ── Parse target PDB: get matching CAs ────────────────────────────────
    target_lines = Path(target_pdb).read_text().splitlines()

    # Build resnum → 1-based index mapping for target
    target_resnums = []
    for tl in target_lines:
        if tl.startswith("ATOM") and tl[12:16].strip() == "CA":
            rn = int(tl[22:26].strip())
            if rn not in target_resnums:
                target_resnums.append(rn)

    resnum_to_idx = {r: i + 1 for i, r in enumerate(target_resnums)}
    idx_to_resnum = {v: k for k, v in resnum_to_idx.items()}

    # Get target CA coords for the crop residues (matched by 1-based seq_id)
    target_cas = {}   # seq_id → (x, y, z)
    for tl in target_lines:
        if tl.startswith("ATOM") and tl[12:16].strip() == "CA":
            rn = int(tl[22:26].strip())
            idx = resnum_to_idx.get(rn)
            if idx is not None and idx in crop_cas:
                target_cas[idx] = np.array([
                    float(tl[30:38]), float(tl[38:46]), float(tl[46:54])
                ])

    # Match by seq_id
    common = sorted(set(crop_cas) & set(target_cas))
    if len(common) < 3:
        return False

    P = np.array([crop_cas[s] for s in common])      # CIF crop coords
    Q = np.array([target_cas[s] for s in common])     # target PDB coords

    # ── Kabsch superposition: find R, t such that Q ≈ R @ P + t ──────────
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_c = P - P_mean
    Q_c = Q - Q_mean
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.diag([1, 1, d])
    R = Vt.T @ sign_mat @ U.T
    t = Q_mean - R @ P_mean

    # ── Transform binder atoms ────────────────────────────────────────────
    transformed = []
    for atom_name, res_name, seq_id, x, y, z, elem in binder_atoms:
        p = np.array([x, y, z])
        q = R @ p + t
        transformed.append((atom_name, res_name, seq_id, q[0], q[1], q[2], elem))

    # ── Write combined PDB ────────────────────────────────────────────────
    out_lines = []
    # Copy full target as chain A
    for tl in target_lines:
        if tl.startswith(("ATOM", "HETATM", "TER")):
            out_lines.append(tl)

    out_lines.append("TER")

    # Write binder as chain B
    atom_serial = len(out_lines) + 1
    for atom_name, res_name, seq_id, x, y, z, elem in transformed:
        an = f" {atom_name:<3s}" if len(atom_name) < 4 else atom_name
        out_lines.append(
            f"ATOM  {atom_serial:5d} {an} {res_name:>3s} B{seq_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {elem:>2s}"
        )
        atom_serial += 1

    out_lines.append("TER")
    out_lines.append("END")

    Path(out_pdb).write_text("\n".join(out_lines) + "\n")
    return True


def copy_top_designs(designs, top_dir, target_pdb=None, n=50, site_resnums=None,
                     target_residues=None, cos_angle=None):
    """Copy top N complex structures to top_designs/ and generate PyMOL scripts."""
    top_dir = Path(top_dir)
    top_dir.mkdir(exist_ok=True)
    # Clear old rank files to avoid stale entries from previous rankings
    for old in top_dir.glob("rank*.cif"):
        old.unlink()
    for old in top_dir.glob("rank*.pdb"):
        old.unlink()
    copied_files = []   # (filename, design_dict)
    for d in designs:
        if d.get("rank") is None or d["rank"] > n:
            continue
        tool = _get_tool_from_design_id(d.get("design_id", ""))
        # Prefer complex_cif (full complex from Boltz), then binder_pdb (which may
        # be a complex CIF for PXDesign/BindCraft), then backbone_pdb (standalone)
        src = (d.get("complex_cif") or d.get("binder_pdb") or d.get("backbone_pdb"))
        if not src or not Path(src).exists():
            log(f"  Warning: no structure file for rank {d['rank']} {d['design_id']}, skipping")
            continue
        src  = Path(src)
        rank = d["rank"]

        # PXDesign CIFs contain a cropped target (~16 residues) in AF2's frame.
        # Merge binder with full target via Kabsch superposition.
        if tool == "pxdesign" and target_pdb and not d.get("complex_cif"):
            fname = f"rank{rank:03d}_{d['design_id']}.pdb"
            dst = top_dir / fname
            try:
                if _merge_binder_with_target(str(src), str(target_pdb), str(dst)):
                    copied_files.append((fname, d))
                    continue
            except Exception as e:
                log(f"  Warning: merge failed for {d['design_id']}: {e}")
            # Fall through to simple copy if merge fails

        fname = f"rank{rank:03d}_{d['design_id']}{src.suffix}"
        dst  = top_dir / fname
        shutil.copy2(src, dst)
        copied_files.append((fname, d))
    log(f"Top designs: {len(copied_files)} structures copied → {top_dir}")

    if not copied_files:
        return

    # Copy target PDB into top_designs/ for easy loading
    target_name = None
    if target_pdb and Path(target_pdb).exists():
        target_name = f"target_{Path(target_pdb).stem}{Path(target_pdb).suffix}"
        shutil.copy2(target_pdb, top_dir / target_name)

    # ── PML 1: align all on target, color by chain ────────────────────────
    _write_pml_by_chain(top_dir, copied_files, target_name, site_resnums, cos_angle=cos_angle)

    # ── PML 2: align all on target, color binders by iPTM ────────────────
    _write_pml_by_iptm(top_dir, copied_files, target_name, site_resnums)

    # ── Per-tool top-10 folders ───────────────────────────────────────────
    _write_per_tool_folders(designs, top_dir, target_pdb, target_name,
                            site_resnums, n_per_tool=20,
                            target_residues=target_residues)


def _write_pml_by_chain(top_dir, copied_files, target_name, site_resnums=None,
                        cos_angle=None):
    """PyMOL script: load all structures, align to target, color by chain."""
    lines = [
        "# Auto-generated by generate_binders3.py",
        "# Color by chain: target=grey, binder=tool color",
        "bg_color white",
        "set ray_shadow, 0",
        "",
    ]
    tool_colors = {
        "rfdiffusion":       "salmon",
        "boltzgen":          "palegreen",
        "bindcraft":         "lightblue",
        "pxdesign":          "lightorange",
        "proteina":          "lightpink",
        "proteina_complexa": "violet",
    }

    if target_name:
        obj = Path(target_name).stem
        lines.append(f'load {target_name}, {obj}')
        lines.append(f'color grey70, {obj}')
        lines.append(f'show cartoon, {obj}')
        lines.append(f'hide lines, {obj}')
        if site_resnums:
            resi_str = "+".join(str(r) for r in site_resnums)
            lines.append(f'# Highlight binding site residues')
            lines.append(f'color red, {obj} and chain A and resi {resi_str}')
            lines.append(f'show sticks, {obj} and chain A and resi {resi_str}')
        lines.append("")

    for fname, d in copied_files:
        obj = Path(fname).stem
        tool = _get_tool_from_design_id(d.get("design_id", ""))
        color = tool_colors.get(tool, "white")
        lines.append(f'load {fname}, {obj}')
        if target_name:
            lines.append(f'align {obj}, {Path(target_name).stem}')
        # Color target chains grey, binder chains by tool color.
        # Standard chains: A=target, B=binder.
        # PXDesign uses A0=target(crop), B0=binder; handle both.
        lines.append(f'color grey70, {obj} and (chain A or chain A0)')
        lines.append(f'color {color}, {obj} and (chain B or chain B0)')
        # For designs with cropped targets (PXDesign), hide the crop so only
        # the binder + full target are visible
        if tool == "pxdesign":
            lines.append(f'hide everything, {obj} and (chain A or chain A0)')
        lines.append(f'show cartoon, {obj}')
        lines.append(f'hide lines, {obj}')

    lines.extend([
        "",
        "# Show only target + top 5 by default; others hidden",
    ])
    for i, (fname, _) in enumerate(copied_files):
        obj = Path(fname).stem
        if i >= 5:
            lines.append(f'disable {obj}')

    # Add cone visualization if cos_angle filter was used
    if cos_angle is not None and site_resnums and target_name:
        import math
        obj = Path(target_name).stem
        resi_str = "+".join(str(r) for r in site_resnums)
        half_angle = math.acos(cos_angle)
        cone_len = 20.0  # Å, length of cone visualization
        lines.extend([
            "",
            "# ── Surface normal cone (cos filter visualization) ──",
            f"# cos >= {cos_angle:.2f} → half-angle = {math.degrees(half_angle):.0f}°",
            "python",
            "import numpy as np",
            "from pymol import cmd",
            "from pymol.cgo import *",
            "",
            f"# Get site CA centroid",
            f"site_model = cmd.get_model('{obj} and chain A and resi {resi_str} and name CA')",
            "site_coords = np.array([[a.coord[0], a.coord[1], a.coord[2]] for a in site_model.atom])",
            "site_center = site_coords.mean(axis=0)",
            "",
            f"# Get target centroid (all CAs)",
            f"target_model = cmd.get_model('{obj} and chain A and name CA')",
            "target_coords = np.array([[a.coord[0], a.coord[1], a.coord[2]] for a in target_model.atom])",
            "target_center = target_coords.mean(axis=0)",
            "",
            "# Outward normal",
            "normal = site_center - target_center",
            "normal = normal / np.linalg.norm(normal)",
            "",
            "# Draw normal line (yellow cylinder)",
            f"tip = site_center + normal * {cone_len}",
            "cgo_normal = [",
            "    CYLINDER,",
            "    site_center[0], site_center[1], site_center[2],",
            "    tip[0], tip[1], tip[2],",
            "    0.3,  # radius",
            "    1.0, 1.0, 0.0,  # yellow",
            "    1.0, 1.0, 0.0,",
            "]",
            "cmd.load_cgo(cgo_normal, 'site_normal')",
            "",
            "# Draw cone boundary (transparent)",
            f"half_angle = {half_angle}",
            f"cone_len = {cone_len}",
            "cone_radius = cone_len * np.tan(half_angle)",
            "",
            "# Generate cone edge lines",
            "# Find two perpendicular vectors to normal",
            "if abs(normal[0]) < 0.9:",
            "    perp1 = np.cross(normal, [1, 0, 0])",
            "else:",
            "    perp1 = np.cross(normal, [0, 1, 0])",
            "perp1 = perp1 / np.linalg.norm(perp1)",
            "perp2 = np.cross(normal, perp1)",
            "",
            "# Draw 12 lines around the cone edge",
            "cgo_cone = []",
            "for i in range(12):",
            "    angle = 2 * np.pi * i / 12",
            "    edge_dir = normal * cone_len + cone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)",
            "    edge_pt = site_center + edge_dir",
            "    cgo_cone.extend([",
            "        CYLINDER,",
            "        site_center[0], site_center[1], site_center[2],",
            "        edge_pt[0], edge_pt[1], edge_pt[2],",
            "        0.15,  # thin lines",
            "        0.0, 1.0, 1.0,  # cyan",
            "        0.0, 1.0, 1.0,",
            "    ])",
            "",
            "cmd.load_cgo(cgo_cone, 'cos_cone')",
            "",
            "# Sphere at site centroid",
            "cmd.pseudoatom('site_centroid', pos=list(site_center), vdw=1.5)",
            "cmd.color('yellow', 'site_centroid')",
            "cmd.show('spheres', 'site_centroid')",
            "",
            "python end",
        ])

    lines.extend([
        "",
        "zoom",
        "set cartoon_transparency, 0.3",
        f"set cartoon_transparency, 0, {Path(target_name).stem}" if target_name else "",
    ])

    pml_path = top_dir / "view_by_chain.pml"
    pml_path.write_text("\n".join(l for l in lines if l is not None) + "\n")
    log(f"PyMOL script (by chain) → {pml_path}")


def _write_pml_by_iptm(top_dir, copied_files, target_name, site_resnums=None):
    """PyMOL script: load all structures, align to target, color binders by iPTM."""
    lines = [
        "# Auto-generated by generate_binders3.py",
        "# Color binders by Boltz iPTM: red (low) → yellow → green (high)",
        "bg_color white",
        "set ray_shadow, 0",
        "",
        "# iPTM color ramp: 0.3=red, 0.5=orange, 0.65=yellow, 0.8=green, 0.9=blue",
    ]

    if target_name:
        obj = Path(target_name).stem
        lines.append(f'load {target_name}, {obj}')
        lines.append(f'color grey70, {obj}')
        lines.append(f'show cartoon, {obj}')
        lines.append(f'hide lines, {obj}')
        if site_resnums:
            resi_str = "+".join(str(r) for r in site_resnums)
            lines.append(f'# Highlight binding site residues')
            lines.append(f'color red, {obj} and chain A and resi {resi_str}')
            lines.append(f'show sticks, {obj} and chain A and resi {resi_str}')
        lines.append("")

    for fname, d in copied_files:
        obj = Path(fname).stem
        tool = _get_tool_from_design_id(d.get("design_id", ""))
        iptm = d.get("_resolved_iptm")
        if iptm is None or iptm != iptm:
            iptm = d.get("boltz_iptm")
        if iptm is None or iptm != iptm:  # NaN check
            iptm = 0.0

        # Map iPTM 0.3-0.9 to red→yellow→green→blue
        if iptm < 0.4:
            r, g, b = 1.0, 0.2, 0.2        # red
        elif iptm < 0.55:
            frac = (iptm - 0.4) / 0.15
            r, g, b = 1.0, 0.2 + 0.6 * frac, 0.2  # red→orange
        elif iptm < 0.7:
            frac = (iptm - 0.55) / 0.15
            r, g, b = 1.0 - 0.5 * frac, 0.8 + 0.2 * frac, 0.2  # orange→yellow-green
        elif iptm < 0.8:
            frac = (iptm - 0.7) / 0.1
            r, g, b = 0.2 + 0.1 * frac, 0.8 + 0.2 * frac, 0.2 + 0.3 * frac  # green
        else:
            frac = min((iptm - 0.8) / 0.15, 1.0)
            r, g, b = 0.1, 0.7 + 0.3 * frac, 0.3 + 0.7 * frac  # green→cyan/blue

        color_name = f"iptm_{obj}"
        lines.append(f'load {fname}, {obj}')
        if target_name:
            lines.append(f'align {obj}, {Path(target_name).stem}')
        lines.append(f'set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]')
        lines.append(f'color grey70, {obj} and (chain A or chain A0)')
        lines.append(f'color {color_name}, {obj} and (chain B or chain B0)')
        if tool == "pxdesign":
            lines.append(f'hide everything, {obj} and (chain A or chain A0)')
        lines.append(f'show cartoon, {obj}')
        lines.append(f'hide lines, {obj}')

    lines.extend([
        "",
        "# Show only target + top 5 by default",
    ])
    for i, (fname, _) in enumerate(copied_files):
        obj = Path(fname).stem
        if i >= 5:
            lines.append(f'disable {obj}')

    lines.extend([
        "",
        "zoom",
        "set cartoon_transparency, 0.3",
        f"set cartoon_transparency, 0, {Path(target_name).stem}" if target_name else "",
    ])

    pml_path = top_dir / "view_by_iptm.pml"
    pml_path.write_text("\n".join(l for l in lines if l is not None) + "\n")
    log(f"PyMOL script (by iPTM) → {pml_path}")


def _write_per_tool_folders(designs, top_dir, target_pdb, target_name,
                            site_resnums, n_per_tool=20, target_residues=None):
    """
    Create per-tool subdirectories under top_designs/ with top 10 designs
    from each tool ranked by combined_score, plus PML scripts with
    binding site highlighted in red.
    """
    tool_colors = {
        "rfdiffusion":       "salmon",
        "boltzgen":          "palegreen",
        "bindcraft":         "lightblue",
        "pxdesign":          "lightorange",
        "proteina":          "lightpink",
        "proteina_complexa": "plum",
    }

    # Group designs by tool
    by_tool = {}
    for d in designs:
        if d.get("combined_score") != d.get("combined_score"):  # NaN
            continue
        if d.get("combined_score") is None:
            continue
        tool = _get_tool_from_design_id(d.get("design_id", ""))
        by_tool.setdefault(tool, []).append(d)

    # Build PyMOL selection strings for binding site residues
    # site_sel_pdb: for original PDB target (uses PDB residue numbers)
    # site_sel_cif: for Boltz-2 CIF complexes (renumbered 1-based)
    site_sel_pdb = ""
    site_sel_cif = ""
    if site_resnums:
        resi_str = "+".join(str(r) for r in site_resnums)
        site_sel_pdb = f"chain A and resi {resi_str}"
        # For CIF files: convert PDB resnums to 1-based indices
        # (same renumbering as validate_boltz pocket constraint)
        if target_residues:
            resnum_to_idx = {r: i + 1 for i, r in enumerate(target_residues)}
            cif_resi = [resnum_to_idx[r] for r in site_resnums if r in resnum_to_idx]
            if cif_resi:
                cif_resi_str = "+".join(str(r) for r in cif_resi)
                site_sel_cif = f"chain A and resi {cif_resi_str}"
        if not site_sel_cif:
            site_sel_cif = site_sel_pdb  # fallback

    for tool, tool_designs in by_tool.items():
        tool_dir = top_dir / tool
        tool_dir.mkdir(exist_ok=True)

        # Clear old files
        for old in tool_dir.glob("rank*.*"):
            old.unlink()

        # Sort by combined_score descending, take top N
        ranked = sorted(tool_designs,
                        key=lambda x: x.get("combined_score", 0), reverse=True)
        top_n = ranked[:n_per_tool]

        # Copy target
        tool_target = None
        if target_pdb and Path(target_pdb).exists():
            tool_target = f"target_{Path(target_pdb).stem}{Path(target_pdb).suffix}"
            shutil.copy2(target_pdb, tool_dir / tool_target)

        # Copy structures
        copied = []
        for i, d in enumerate(top_n):
            src = (d.get("complex_cif") or d.get("binder_pdb") or d.get("backbone_pdb"))
            if not src or not Path(src).exists():
                continue
            src = Path(src)
            fname = f"rank{i+1:02d}_{d['design_id']}"

            # PXDesign: merge with full target
            if tool == "pxdesign" and target_pdb and not d.get("complex_cif"):
                dst = tool_dir / f"{fname}.pdb"
                try:
                    if _merge_binder_with_target(str(src), str(target_pdb), str(dst)):
                        copied.append((f"{fname}.pdb", d))
                        continue
                except Exception:
                    pass

            dst = tool_dir / f"{fname}{src.suffix}"
            shutil.copy2(src, dst)
            copied.append((f"{fname}{src.suffix}", d))

        if not copied:
            continue

        # Write PML script
        color = tool_colors.get(tool, "white")
        pml_lines = [
            f"# {tool.upper()} — top {len(copied)} designs",
            f"# Binding site residues highlighted in red",
            "bg_color white",
            "set ray_shadow, 0",
            "",
        ]

        if tool_target:
            tobj = Path(tool_target).stem
            pml_lines.append(f"load {tool_target}, {tobj}")
            pml_lines.append(f"color grey70, {tobj}")
            pml_lines.append(f"show cartoon, {tobj}")
            pml_lines.append(f"hide lines, {tobj}")
            # Highlight binding site in red (PDB numbering for target PDB)
            if site_sel_pdb:
                pml_lines.append(f"color red, {tobj} and {site_sel_pdb}")
                pml_lines.append(f"show sticks, {tobj} and {site_sel_pdb}")
            pml_lines.append("")

        for j, (fname, d) in enumerate(copied):
            obj = Path(fname).stem
            iptm = d.get("_resolved_iptm")
            if iptm is None or iptm != iptm:
                iptm = d.get("boltz_iptm", 0)
                if iptm is None or iptm != iptm:
                    iptm = 0
            score = d.get("combined_score", 0)

            pml_lines.append(f"# rank {j+1}  score={score:.3f}  iPTM={iptm:.3f}")
            pml_lines.append(f"load {fname}, {obj}")
            if tool_target:
                pml_lines.append(f"align {obj}, {Path(tool_target).stem}")
            pml_lines.append(f"color grey70, {obj} and (chain A or chain A0)")
            pml_lines.append(f"color {color}, {obj} and (chain B or chain B0)")
            # Highlight binding site in red on complex target chain (CIF renumbered)
            if site_sel_cif:
                pml_lines.append(f"color red, {obj} and {site_sel_cif}")
            if tool == "pxdesign" and not d.get("complex_cif"):
                pml_lines.append(f"hide everything, {obj} and (chain A or chain A0)")
            pml_lines.append(f"show cartoon, {obj}")
            pml_lines.append(f"hide lines, {obj}")
            # Hide all but top 3 by default
            if j >= 3:
                pml_lines.append(f"disable {obj}")
            pml_lines.append("")

        pml_lines.extend([
            "zoom",
            "set cartoon_transparency, 0.3",
        ])
        if tool_target:
            pml_lines.append(f"set cartoon_transparency, 0, {Path(tool_target).stem}")

        pml_path = tool_dir / f"view_{tool}.pml"
        pml_path.write_text("\n".join(pml_lines) + "\n")

        log(f"  {tool}: {len(copied)} designs → {tool_dir}")


# ── PLIP interaction analysis ──────────────────────────────────────────────────

def run_plip_analysis(designs, out_dir, target_path, target_residues=None,
                      plip_top=10, dry_run=False):
    """
    Run PLIP protein-protein interaction analysis on top-ranked designs.

    For each design:
      1. Find the complex structure (CIF from Boltz-2, or native PDB)
      2. Renumber target chain (A) back to original PDB numbering
      3. Convert to PDB via gemmi
      4. Run PLIP with --peptides B (binder as peptide ligand)
      5. Output to plip_analysis/rank{NN}_{design_id}/

    Generates .pse PyMOL sessions, .txt reports, .xml machine-readable reports.

    Parameters
    ----------
    designs : list of ranked design dicts (must have 'rank' set)
    out_dir : Path to output directory (plip_analysis/ created inside)
    target_path : path to original target PDB (for residue numbering)
    target_residues : list of original PDB residue numbers for chain A
    plip_top : number of top designs to analyze (default 10)
    dry_run : if True, skip execution
    """
    if plip_top <= 0 or dry_run:
        return

    plip_dir = Path(out_dir) / "plip_analysis"
    plip_dir.mkdir(parents=True, exist_ok=True)

    # Check PLIP is available
    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        log("  PLIP not installed — skipping interaction analysis. "
            "Install with: conda run -n boltz pip install plip --no-deps")
        return

    # Check gemmi is available (needed for CIF→PDB conversion + renumbering)
    try:
        import gemmi
    except ImportError:
        log("  gemmi not installed — skipping PLIP analysis")
        return

    # Get original target residue numbering
    orig_resnums = []
    if target_residues:
        orig_resnums = list(target_residues)
    else:
        try:
            target_st = gemmi.read_structure(str(target_path))
            orig_resnums = [res.seqid.num for res in target_st[0][0]]  # first chain
        except Exception:
            pass

    # Select top N ranked designs
    ranked = [d for d in designs if d.get("rank") is not None and d["rank"] <= plip_top]
    if not ranked:
        log("  PLIP: no ranked designs to analyze")
        return

    log(f"PLIP: analyzing {len(ranked)} top designs...")

    n_success = 0
    summary_lines = ["PLIP Interaction Summary", "=" * 80, ""]

    for d in ranked:
        rank = d["rank"]
        did = d["design_id"]
        rank_label = f"rank{rank:02d}_{did}"

        # Find complex structure
        struct_path = d.get("complex_cif") or d.get("binder_pdb")
        if not struct_path or not Path(struct_path).exists():
            log(f"  {rank_label}: no structure file, skipping")
            continue

        # Create ranked output folder
        design_plip_dir = plip_dir / rank_label
        design_plip_dir.mkdir(exist_ok=True)

        # Convert CIF→PDB with original target numbering
        pdb_path = plip_dir / f"{rank_label}.pdb"
        try:
            st = gemmi.read_structure(str(struct_path))

            # Renumber chain A to original PDB numbering
            chain_a = None
            for chain in st[0]:
                if chain.name in ("A", "A0"):
                    chain_a = chain
                    break
            if chain_a and orig_resnums:
                for i, res in enumerate(chain_a):
                    if i < len(orig_resnums):
                        res.seqid = gemmi.SeqId(str(orig_resnums[i]))

            # Sanitize chain names (multi-char → single-char)
            for chain in st[0]:
                if len(chain.name) > 1:
                    chain.name = chain.name[0]

            st.write_pdb(str(pdb_path))
        except Exception as e:
            log(f"  {rank_label}: gemmi conversion failed: {e}")
            continue

        # Run PLIP
        try:
            result = subprocess.run(
                ["plip", "-f", str(pdb_path), "-o", str(design_plip_dir),
                 "--peptides", "B", "-txy"],
                capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                log(f"  {rank_label}: PLIP failed: {result.stderr[-200:]}")
                continue
        except FileNotFoundError:
            log("  PLIP command not found — install with: pip install plip --no-deps")
            return
        except subprocess.TimeoutExpired:
            log(f"  {rank_label}: PLIP timeout")
            continue

        # Verify output
        pse_files = list(design_plip_dir.glob("*.pse"))
        report_files = list(design_plip_dir.glob("*_report.txt"))
        if pse_files or report_files:
            n_success += 1

        # Parse interaction counts for summary
        n_hydro = n_hbond = n_salt = 0
        site_res = set()
        for rf in report_files:
            section = ""
            for line in rf.read_text().splitlines():
                if "**Hydrophobic" in line:
                    section = "hydro"
                elif "**Hydrogen" in line:
                    section = "hbond"
                elif "**Salt" in line:
                    section = "salt"
                elif line.startswith("|") and "| A " in line and section:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) > 3 and parts[1].isdigit():
                        rn = int(parts[1])
                        rtype = parts[2][:3]
                        if section == "hydro":
                            n_hydro += 1
                        elif section == "hbond":
                            n_hbond += 1
                        elif section == "salt":
                            n_salt += 1
                        site_res.add(f"{rtype}{rn}")

        pse_name = pse_files[0].name if pse_files else "N/A"
        report_name = report_files[0].name if report_files else "N/A"

        summary_lines.append(
            f"Rank {rank:2d}: {did}  "
            f"score={d.get('combined_score', '?')}  iPTM={d.get('boltz_iptm', d.get('_resolved_iptm', '?'))}  "
            f"frac={d.get('site_contact_fraction', '?')}")
        summary_lines.append(
            f"  Contacts: {n_hydro} hydrophobic, {n_hbond} H-bonds, {n_salt} salt bridges")
        summary_lines.append(
            f"  Site residues: {', '.join(sorted(site_res, key=lambda x: int(''.join(c for c in x if c.isdigit()))))}")
        summary_lines.append(f"  PyMOL: {rank_label}/{pse_name}")
        summary_lines.append(f"  Report: {rank_label}/{report_name}")
        summary_lines.append("")

    # Write summary
    summary_lines.append(f"To view: pymol plip_analysis/rank01_<design>/<NAME>.pse")
    (plip_dir / "PLIP_SUMMARY.txt").write_text("\n".join(summary_lines))

    log(f"PLIP: {n_success}/{len(ranked)} designs analyzed → {plip_dir}")


# ── Dashboard ──────────────────────────────────────────────────────────────────

def plot_dashboard(all_designs, out_path, title="Binder Design Summary"):
    """
    6-panel summary dashboard (3×2 grid):
      Panel 1: Combined score box plot by tool
      Panel 2: ESMFold pLDDT histogram
      Panel 3: ESMFold pLDDT vs Boltz iPTM scatter
      Panel 4: Top 20 horizontal bar chart
      Panel 5: Boltz iPTM vs min interface pAE scatter (validated designs)
      Panel 6: Top 10 key metrics table
    """
    validated = [d for d in all_designs
                 if d.get("combined_score") == d.get("combined_score")
                 and d.get("combined_score") is not None]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    colors = TOOL_COLORS

    # Group by tool
    tool_groups = {}
    for d in validated:
        t = _get_tool_from_design_id(d["design_id"])
        tool_groups.setdefault(t, []).append(d)

    tool_list = [t for t in ALL_TOOLS if t in tool_groups]

    # ── Panel 1: combined score box plot by tool ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if tool_groups:
        data = [[d["combined_score"] for d in tool_groups[t]] for t in tool_list]
        bp   = ax1.boxplot(data, positions=range(len(tool_list)),
                           patch_artist=True, widths=0.5)
        for patch, t in zip(bp["boxes"], tool_list):
            patch.set_facecolor(colors.get(t, "grey")); patch.set_alpha(0.7)
        ax1.set_xticks(range(len(tool_list)))
        ax1.set_xticklabels(tool_list, fontsize=9)
    ax1.set_ylabel("Combined score (pLDDT + iPTM)")
    ax1.set_title("Score distribution by tool")
    ax1.set_ylim(0, 1)

    # ── Panel 2: ESMFold pLDDT histogram ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for t in tool_list:
        plddts = [d.get("esmfold_plddt") for d in tool_groups[t]]
        plddts = [x for x in plddts if x is not None and x == x]  # drop None/NaN
        ax2.hist(plddts, bins=15, alpha=0.6, color=colors.get(t, "grey"),
                 label=t, edgecolor="white")
    ax2.axvline(80, color="red", lw=1, ls="--", label="threshold=80")
    ax2.set_xlabel("ESMFold binder pLDDT"); ax2.set_ylabel("Count")
    ax2.set_title("ESMFold pre-filter distribution")
    ax2.legend(fontsize=8)

    # ── Panel 3: ESMFold pLDDT vs Boltz iPTM scatter ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for t in tool_list:
        pts = [(d.get("esmfold_plddt"), d.get("boltz_iptm")) for d in tool_groups[t]]
        pts = [(x, y) for x, y in pts if x is not None and y is not None and x == x and y == y]
        if pts:
            x, y = zip(*pts)
            ax3.scatter(x, y, c=colors.get(t, "grey"), alpha=0.7, s=30, label=t)
    ax3.set_xlabel("ESMFold binder pLDDT"); ax3.set_ylabel("Boltz-2 iPTM")
    ax3.set_title("ESMFold pLDDT vs Boltz iPTM")
    ax3.legend(fontsize=8)

    # ── Panel 4: top 20 horizontal bar chart ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    top20 = validated[:20]
    ypos  = range(len(top20))
    bar_colors = [colors.get(_get_tool_from_design_id(d["design_id"]), "grey") for d in top20]
    ax4.barh(list(ypos), [d["combined_score"] for d in top20],
             color=bar_colors, alpha=0.8)
    ax4.set_yticks(list(ypos))
    ax4.set_yticklabels([d["design_id"] for d in top20], fontsize=7)
    ax4.set_xlabel("Combined score"); ax4.set_title("Top 20 designs")
    ax4.set_xlim(0, 1); ax4.invert_yaxis()

    # ── Panel 5: Boltz iPTM vs min interface pAE scatter ───────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    for t in tool_list:
        grp = tool_groups[t]
        x = []
        y = []
        for d in grp:
            pae  = d.get("boltz_min_interface_pae", float("nan"))
            iptm = d.get("boltz_iptm", float("nan"))
            if pae == pae and iptm == iptm:  # both non-nan
                x.append(pae)
                y.append(iptm)
        if x:
            ax5.scatter(x, y, c=colors.get(t, "grey"), alpha=0.7,
                        s=40, label=t, edgecolors="none")
    ax5.set_xlabel("Min interface pAE (lower = better contacts)")
    ax5.set_ylabel("Boltz-2 iPTM")
    ax5.set_title("Interface quality: iPTM vs min pAE")
    ax5.legend(fontsize=8)
    # Annotate ideal region
    ax5.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax5.axvline(8.0, color="grey", lw=0.8, ls="--", alpha=0.5)

    # ── Panel 6: top 10 key metrics table ──────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    top10 = validated[:10]
    if top10:
        col_labels = ["Rank", "Design", "Tool", "Score", "iPTM", "pLDDT", "minPAE", "Liability"]
        table_data = []
        for d in top10:
            tool      = _get_tool_from_design_id(d["design_id"])
            score     = d.get("combined_score", float("nan"))
            iptm      = d.get("boltz_iptm", float("nan"))
            plddt     = d.get("boltz_binder_plddt", float("nan"))
            min_pae   = d.get("boltz_min_interface_pae", float("nan"))
            # BoltzGen liability score or N/A
            liab = d.get("bg_liability_score", float("nan"))

            def _fmt(v, dec=3):
                return f"{v:.{dec}f}" if v == v else "N/A"

            table_data.append([
                str(d.get("rank", "?")),
                d["design_id"][-12:],  # truncate long IDs
                tool,
                _fmt(score),
                _fmt(iptm),
                _fmt(plddt, 1),
                _fmt(min_pae, 1),
                _fmt(liab, 0) if liab == liab else "N/A",
            ])

        tbl = ax6.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.3)
        # Color header row
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor("#BBDEFB")
        ax6.set_title("Top 10 designs — key metrics", fontsize=9, pad=10)

    # ── Global legend ───────────────────────────────────────────────────────
    legend_handles = [mpatches.Patch(facecolor=colors[t], label=t, alpha=0.7)
                      for t in tool_list if t in colors]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Dashboard saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-tool binder design: RFdiffusion+LigandMPNN, BoltzGen, BindCraft, PXDesign",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target",    required=True,
                        help="Target protein structure (PDB or CIF)")
    parser.add_argument("--site",      required=True,
                        help='Binding site residues, e.g. "A10 A15 A22"')
    parser.add_argument("--length",    required=True,
                        help="Binder length range, e.g. 80-120")
    parser.add_argument("--tools",     required=True,
                        help=f"Comma-separated tools: {','.join(ALL_TOOLS)}")
    parser.add_argument("--mode",      default=DEFAULT_MODE,
                        choices=list(DESIGN_MODES),
                        help=f"Design scale preset (default: {DEFAULT_MODE}). "
                             f"test=fast, standard=typical, production=thorough")
    parser.add_argument("--n_designs", default=None,
                        help='Override per-tool counts, e.g. "boltzgen=10000" or '
                             '"rfdiffusion=200,boltzgen=5000,bindcraft=100". '
                             'Overrides --mode for specified tools only.')
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory")
    parser.add_argument("--esmfold_plddt_threshold", type=float, default=80.0,
                        help="ESMFold pre-filter threshold (default: 80)")
    parser.add_argument("--max_validate", type=int, default=20,
                        help="Max designs sent to Boltz validation per tool (default: 20)")
    parser.add_argument("--score_weights", default="0.4,0.5,0.1",
                        help='Comma-separated pLDDT, iPTM, and Rosetta dG weights for '
                             'combined_score (default: "0.4,0.5,0.1"). '
                             'Two values also accepted for backward compat (dG weight=0). '
                             'Example: "0.3,0.6,0.1" to weight iPTM more heavily.')
    parser.add_argument("--filter_interface_pae", type=float, default=None,
                        help="Drop Boltz-validated designs whose min interface pAE exceeds "
                             "this threshold (lower pAE = better interface contacts). "
                             "Example: --filter_interface_pae 12.0")
    parser.add_argument("--filter_site_pae", type=float, default=None,
                        help="Drop designs whose site-specific mean pAE exceeds this "
                             "threshold (detects off-site binding). "
                             "Example: --filter_site_pae 15.0")
    parser.add_argument("--max_site_dist", type=float, default=15.0,
                        help="Site contact filter: distance threshold (Angstrom) for "
                             "counting a site residue as 'contacted' by binder heavy atoms. "
                             "0=disable filter. Default: 15.0")
    parser.add_argument("--min_site_fraction", type=float, default=0.0,
                        help="Minimum fraction of site residues that must be contacted "
                             "(within --max_site_dist) for a design to pass. "
                             "0.0=only exclude designs with zero contacts (default). "
                             "0.3=require 30%% of site residues contacted.")
    parser.add_argument("--min_site_interface_fraction", type=float, default=None,
                        help="Minimum fraction of binder interface residues that contact "
                             "SITE residues (vs non-site target residues). Filters designs "
                             "that dock beside the site rather than on top. "
                             "0.5=at least 50%% of binder interface at site. Default: disabled.")
    parser.add_argument("--max_site_centroid_dist", type=float, default=None,
                        help="Maximum distance (Angstrom) between binder centroid and site "
                             "centroid. Selects binders sitting directly on top of the site. "
                             "Typical: 10-15 Å. Default: disabled.")
    parser.add_argument("--centroid_atoms", choices=["CA", "heavy"], default="CA",
                        help="Which atoms to use for site centroid: 'CA' = alpha carbon "
                             "(1 per residue, true geometric center), 'heavy' = all heavy atoms. "
                             "Default: CA.")
    parser.add_argument("--interface_dist", type=float, default=5.0,
                        help="Distance cutoff (Angstrom) defining binder interface residues "
                             "for SIF and centroid calculations. 5.0=direct contacts, "
                             "7.0=broader footprint. Default: 5.0.")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top designs to copy to top_designs/ (default: 50)")
    parser.add_argument("--plip_top", type=int, default=10,
                        help="Number of top designs to run PLIP interaction analysis on "
                             "(default: 10). Set to 0 to disable.")
    parser.add_argument("--boltz_devices", type=int, default=1,
                        help="Number of GPUs for Boltz-2 batch validation (default: 1). "
                             "Only free GPUs (<500 MB used) are used. Safe with other users.")
    parser.add_argument("--bindcraft_filters",
                        default=f"{BINDCRAFT_DIR}/settings_filters/no_filters.json",
                        help="BindCraft filters JSON (default: no_filters.json — "
                             "accept all designs, let Boltz validation rank them. "
                             "Use relaxed_filters.json or default_filters.json for "
                             "stricter quality gates).")
    parser.add_argument("--bindcraft_advanced",
                        default=f"{BINDCRAFT_DIR}/settings_advanced/default_4stage_multimer_flexible.json",
                        help="BindCraft advanced settings JSON (default: "
                             "default_4stage_multimer_flexible.json — removes target "
                             "template constraint for broader sampling). Use "
                             "default_4stage_multimer_flexible_hardtarget.json for "
                             "especially difficult targets like KRAS.")
    parser.add_argument("--device",    default=None,
                        help="CUDA device, e.g. cuda:0 or cuda:1. "
                             "Pins all GPU work to this device via CUDA_VISIBLE_DEVICES. "
                             "Default: use all available GPUs.")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--reprediction", action="store_true",
                        help="Enable Boltz-2 re-prediction for ALL tools including "
                             "those with native iPTM scores (BindCraft, BoltzGen, PXDesign). "
                             "Default behavior: only backbone-only tools (RFdiffusion, Proteina) "
                             "go through Boltz-2 re-prediction; native-iPTM tools keep their "
                             "own scores for ranking.")
    parser.add_argument("--ss_bias", choices=["beta", "helix", "balanced"],
                        default="balanced",
                        help="Secondary structure bias: beta (more sheets), "
                             "helix (more helices), balanced (default, no bias)")
    args = parser.parse_args()

    # ── GPU pinning ──────────────────────────────────────────────────────
    global GPU_ENV
    if args.device:
        # Extract GPU index from "cuda:N" or just "N"
        dev = args.device.replace("cuda:", "")
        GPU_ENV = {"CUDA_VISIBLE_DEVICES": dev}

    # ── Parse & validate args ─────────────────────────────────────────────
    target_path = Path(args.target).resolve()
    if not target_path.exists():
        sys.exit(f"ERROR: target file not found: {target_path}")

    try:
        chain_id, site_resnums = parse_site(args.site)
    except ValueError as e:
        sys.exit(f"ERROR --site: {e}")

    try:
        length_min, length_max = parse_length(args.length)
    except ValueError as e:
        sys.exit(f"ERROR --length: {e}")

    tools = [t.strip() for t in args.tools.split(",")]
    for t in tools:
        if t not in ALL_TOOLS:
            sys.exit(f"ERROR: unknown tool '{t}'. Choose from {ALL_TOOLS}")

    try:
        score_weights = parse_score_weights(args.score_weights)
    except (ValueError, TypeError) as e:
        sys.exit(f"ERROR --score_weights: {e}")

    # RFdiffusion and BindCraft cannot read CIF — require PDB upfront
    pdb_only_tools = [t for t in tools if t in ("rfdiffusion", "bindcraft")]
    if pdb_only_tools and target_path.suffix.lower() == ".cif":
        sys.exit(
            f"ERROR: {pdb_only_tools} require a PDB file, but got: {target_path}\n"
            f"  Use --target with a .pdb file. For KRAS4B:\n"
            f"  --target /data/scripts/protein_folding/outputs/KRASB/8ECR_monomer.pdb"
        )

    # Start from mode presets, then apply any --n_designs overrides
    n_designs = dict(DESIGN_MODES[args.mode])
    if args.n_designs:
        overrides = parse_n_designs(args.n_designs)
        n_designs.update(overrides)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 62)
    log("generate_binders.py — Multi-tool binder design pipeline")
    log("=" * 62)
    log(f"Target:           {target_path}")
    log(f"Site:             chain={chain_id}, residues={site_resnums}")
    log(f"Length:           {length_min}–{length_max} aa")
    log(f"Tools:            {tools}")
    log(f"Device:           {args.device or 'all GPUs'}")
    log(f"Mode:             {args.mode}")
    log(f"N designs:        { {t: n_designs.get(t,'?') for t in tools} }")
    log(f"Score weights:    pLDDT={score_weights[0]}, iPTM={score_weights[1]}, dG={score_weights[2]}")
    log(f"Reprediction:     {args.reprediction}")
    log(f"SS bias:          {args.ss_bias}")
    log(f"Filter PAE:       {args.filter_interface_pae}")
    log(f"Max site dist:    {args.max_site_dist} Å")
    log(f"Min site frac:    {args.min_site_fraction}")
    log(f"Top N copy:       {args.top_n}")
    log(f"Output:           {out_dir}")
    if args.dry_run:
        log("*** DRY RUN — no commands will be executed ***")
    log("=" * 62)

    # ── Extract target sequence ───────────────────────────────────────────
    chain_info = get_chain_info(target_path)
    tchain     = chain_info.get(chain_id, {})
    target_seq = tchain.get("sequence", "")
    target_len = tchain.get("length", 0)

    if not target_seq and not args.dry_run:
        sys.exit(f"ERROR: chain {chain_id} not found in {target_path}. "
                 f"Available chains: {list(chain_info)}")
    log(f"Target chain {chain_id}: {target_len} residues")

    if target_len > 500:
        log(f"NOTE: large target ({target_len} aa) — Boltz validation will use "
            f"reduced MSA depth and recycling steps (auto memory-save)")

    # ── Run design tools sequentially ─────────────────────────────────────
    all_designs  = []
    tool_results = {}

    RUNNERS = {
        "rfdiffusion":       run_rfdiffusion,
        "boltzgen":          run_boltzgen,
        "bindcraft":         run_bindcraft,
        "pxdesign":          run_pxdesign,
        "proteina":          run_proteina,
        "proteina_complexa": run_proteina_complexa,
    }

    for tool in tools:
        n = n_designs.get(tool, 100)
        log(f"\n{'─' * 50}")
        log(f"STEP: {tool.upper()}  (n={n})")
        log(f"{'─' * 50}")
        try:
            extra = {}
            if tool == "bindcraft":
                extra = {"filters_path": args.bindcraft_filters,
                         "advanced_path": args.bindcraft_advanced,
                         "ss_bias": args.ss_bias}
            elif tool in ("boltzgen", "proteina_complexa"):
                extra = {"ss_bias": args.ss_bias}
            designs = RUNNERS[tool](
                target_path, chain_id, site_resnums, length_min, length_max,
                n, out_dir / tool, args.dry_run, **extra)
            all_designs.extend(designs)
            tool_results[tool] = f"OK ({len(designs)} designs)"
        except Exception as e:
            log(f"{tool} FAILED: {e}")
            # Retry once after GPU cleanup (crashed tools may leave zombie GPU processes)
            _cleanup_gpu()
            log(f"  Retrying {tool} after GPU cleanup...")
            try:
                designs = RUNNERS[tool](
                    target_path, chain_id, site_resnums, length_min, length_max,
                    n, out_dir / tool, args.dry_run, **extra)
                all_designs.extend(designs)
                tool_results[tool] = f"OK ({len(designs)} designs, retry)"
                log(f"  {tool} succeeded on retry")
            except Exception as e2:
                log(f"{tool} FAILED on retry: {e2}")
                tool_results[tool] = f"FAILED: {e}"
        # Clean up GPU between tools to prevent memory leaks from crashed subprocesses
        _cleanup_gpu()

    if not all_designs:
        log("No designs collected. Exiting.")
        sys.exit(1)

    log(f"\nTotal designs collected: {len(all_designs)}")

    # ── Validation ────────────────────────────────────────────────────────
    val_dir = out_dir / "validation"

    log(f"\n{'─' * 50}")
    log("STEP: ESMFold pre-filter (Stage 1)")
    log(f"{'─' * 50}")
    if args.reprediction:
        # --reprediction ON: ESMFold all designs (original behavior)
        passing = validate_esmfold(
            all_designs, val_dir, args.esmfold_plddt_threshold, args.dry_run)
    else:
        # --reprediction OFF: only ESMFold backbone-only tools (RFdiffusion, Proteina).
        # iPTM-native tools (BindCraft, BoltzGen, PXDesign, Proteina Complexa)
        # already have internal validation — ESMFold is redundant for them.
        backbone_all = [d for d in all_designs
                        if _get_tool_from_design_id(d["design_id"]) in BACKBONE_ONLY_TOOLS]
        native_all = [d for d in all_designs
                      if _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS]
        log(f"  ESMFold: {len(backbone_all)} backbone-only designs "
            f"(skipping {len(native_all)} native-iPTM designs)")
        passing_backbone = validate_esmfold(
            backbone_all, val_dir, args.esmfold_plddt_threshold, args.dry_run)
        if not passing_backbone and backbone_all and not args.dry_run:
            passing_backbone = backbone_all
        # Native tools pass through directly — no ESMFold gate
        passing = (passing_backbone or []) + native_all

    if not passing and not args.dry_run:
        log("WARNING: no designs passed ESMFold pre-filter. "
            "Consider lowering --esmfold_plddt_threshold. "
            "Proceeding with all designs for ranking.")
        passing = all_designs

    log(f"\n{'─' * 50}")
    log("STEP: Boltz-2 uniform validation (Stage 2)")
    log(f"{'─' * 50}")
    if args.reprediction:
        # --reprediction ON: all designs go through Boltz-2 (original behavior)
        passing = validate_boltz(
            passing, target_seq, val_dir,
            max_per_tool=args.max_validate,
            target_len=target_len,
            dry_run=args.dry_run,
            site_resnums=site_resnums,
            target_residues=tchain.get("residues"),
            boltz_devices=args.boltz_devices)
    else:
        # --reprediction OFF: only backbone-only tools get Boltz-2
        backbone_designs = [d for d in passing
                            if _get_tool_from_design_id(d["design_id"]) in BACKBONE_ONLY_TOOLS]
        native_designs = [d for d in passing
                          if _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS]
        log(f"  Split: {len(backbone_designs)} backbone-only designs for Boltz-2, "
            f"{len(native_designs)} native-iPTM designs skip Boltz-2")

        if backbone_designs:
            backbone_designs = validate_boltz(
                backbone_designs, target_seq, val_dir,
                max_per_tool=args.max_validate,
                target_len=target_len,
                dry_run=args.dry_run,
                site_resnums=site_resnums,
                target_residues=tchain.get("residues"),
                boltz_devices=args.boltz_devices)

        # For native designs: ensure complex_cif or binder_pdb is set
        # (BindCraft stores binder_pdb, BoltzGen stores complex_cif,
        # PXDesign stores binder_pdb)
        for nd in native_designs:
            if not nd.get("complex_cif") and not nd.get("binder_pdb"):
                log(f"  Warning: {nd['design_id']} has no structure file")

        # Recombine for downstream processing
        passing = backbone_designs + native_designs

    need_geometric = (args.max_site_dist > 0 or
                      getattr(args, 'min_site_interface_fraction', None) is not None or
                      getattr(args, 'max_site_centroid_dist', None) is not None or
                      getattr(args, 'min_site_cos', None) is not None)
    if need_geometric:
        max_dist = args.max_site_dist if args.max_site_dist > 0 else 999.0
        log(f"\n{'─' * 50}")
        log("STEP: Geometric site proximity filter")
        log(f"{'─' * 50}")
        geometric_site_filter(all_designs, site_resnums=site_resnums,
                              target_residues=tchain.get("residues"),
                              max_dist=max_dist,
                              min_site_fraction=args.min_site_fraction,
                              interface_dist=args.interface_dist,
                              dry_run=args.dry_run)

    log(f"\n{'─' * 50}")
    log("STEP: Rosetta interface scoring (Stage 3)")
    log(f"{'─' * 50}")
    rosetta_score_interfaces(all_designs, val_dir, dry_run=args.dry_run)

    # ── Compute SS fractions + amino acid composition ────────────────────
    log(f"\n{'─' * 50}")
    log("STEP: Secondary structure + composition analysis")
    log(f"{'─' * 50}")
    populate_binder_composition(all_designs)
    populate_interface_composition(all_designs)
    if not args.dry_run:
        populate_binder_ss(all_designs)
    else:
        log("  (SS skipped in dry-run mode)")

    # ── Rank & output ─────────────────────────────────────────────────────
    log(f"\n{'─' * 50}")
    log("STEP: Ranking and output")
    log(f"{'─' * 50}")

    all_ranked = rank_designs(
        all_designs,
        score_weights=score_weights,
        filter_interface_pae=args.filter_interface_pae,
        filter_site_pae=args.filter_site_pae,
        reprediction=args.reprediction,
        ss_bias=args.ss_bias,
    )
    write_rankings_csv(all_ranked, out_dir / "rankings.csv")

    # Restore nullified structure paths so top designs can be copied
    for d in all_ranked:
        if d.get("_orig_complex_cif"):
            d["complex_cif"] = d.pop("_orig_complex_cif")
        if d.get("_orig_binder_pdb"):
            d["binder_pdb"] = d.pop("_orig_binder_pdb")

    copy_top_designs(all_ranked, out_dir / "top_designs",
                     target_pdb=target_path, n=args.top_n,
                     site_resnums=site_resnums,
                     target_residues=tchain.get("residues"))
    plot_dashboard(
        all_ranked, out_dir / "dashboard.png",
        title=f"Binder Design: {target_path.stem}  site={args.site}")

    # ── PLIP interaction analysis ─────────────────────────────────────────
    if args.plip_top > 0:
        log(f"\n{'─' * 50}")
        log(f"STEP: PLIP interaction analysis (top {args.plip_top})")
        log(f"{'─' * 50}")
        run_plip_analysis(
            all_ranked, out_dir, target_path,
            target_residues=tchain.get("residues"),
            plip_top=args.plip_top,
            dry_run=args.dry_run)

    # ── Summary ───────────────────────────────────────────────────────────
    log(f"\n{'=' * 62}")
    log("DONE")
    log(f"{'=' * 62}")
    for tool, status in tool_results.items():
        log(f"  {tool:12s}: {status}")

    validated = [d for d in all_ranked if d.get("rank") is not None]
    if validated:
        log(f"\nTop 5 designs:")
        for d in validated[:5]:
            min_pae = d.get("boltz_min_interface_pae", float("nan"))
            pae_str = f"{min_pae:.2f}" if min_pae == min_pae else "N/A"
            log(f"  rank {d['rank']:3d}  {d['design_id']:<22s}  "
                f"score={d['combined_score']:.3f}  "
                f"iPTM={d.get('_resolved_iptm', d.get('boltz_iptm', 0)):.3f}  "
                f"ESMFold={d.get('esmfold_plddt', 0):.1f}  "
                f"minPAE={pae_str}")

    log(f"\nOutput directory: {out_dir}")
    log(f"  rankings.csv   — {len(all_ranked)} total designs")
    log(f"  top_designs/   — top {min(args.top_n, len(validated))} structures")
    log(f"  dashboard.png  — summary plot")


if __name__ == "__main__":
    main()
