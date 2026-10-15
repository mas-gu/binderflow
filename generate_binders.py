#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume Mas
# SPDX-License-Identifier: MIT
"""
generate_binders.py — Multi-tool binder design pipeline.

Given a target protein and a binding site, runs up to six complementary design
tools, validates all outputs with Boltz-2 monomer (fast pre-filter) and Boltz-2 complex (uniform
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
    summary.csv are already pre-validated. Our Boltz-2 monomer + Boltz-2 complex stages
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
    production: rfdiffusion=500,  boltzgen=2000,  bindcraft=20,  pxdesign=500,  proteina=500   (~2–4 days)

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
        Selected by highest monomer pLDDT.

    --top_n N
        Number of top designs copied to top_designs/ (default: 50).

═══════════════════════════════════════════════════════════════════════════════
VALIDATION PIPELINE
═══════════════════════════════════════════════════════════════════════════════
Stage 1 — Boltz-2 monomer pre-filter
    Each binder sequence folded independently by Boltz-2. Threshold: pLDDT ≥ 70.
    Filters unstructured/disordered sequences before expensive complex runs.

Stage 2 — Boltz-2 complex scoring (top N per tool by monomer pLDDT)
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
      Monomer: esmfold_plddt (Boltz-2 monomer pLDDT)
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
    ├── validation/    boltz_monomer/ + boltz/ per-design structures & scores
    ├── top_designs/   top --top_n complex structures (CIF/PDB) + PyMOL/ChimeraX scenes
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
import tempfile
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

from config_loader import cfg

_SOFTWARE_DIR = cfg.software_dir
_WEIGHTS_DIR  = cfg.weights_dir

RFDIFFUSION_DIR      = cfg.tool("rfdiffusion_dir",      "RFDIFFUSION_DIR",      f"{_SOFTWARE_DIR}/RFdiffusion")
RFDIFFUSION_CKPT     = cfg.tool("rfdiffusion_ckpt",     "RFDIFFUSION_CKPT",     f"{_WEIGHTS_DIR}/rfdiffusion/Complex_base_ckpt.pt")
LIGANDMPNN_DIR       = cfg.tool("ligandmpnn_dir",       "LIGANDMPNN_DIR",       f"{_SOFTWARE_DIR}/LigandMPNN")
LIGANDMPNN_CKPT      = cfg.tool("ligandmpnn_ckpt",      "LIGANDMPNN_CKPT",      f"{_WEIGHTS_DIR}/ligandmpnn/proteinmpnn_v_48_020.pt")
BINDCRAFT_DIR        = cfg.tool("bindcraft_dir",         "BINDCRAFT_DIR",         f"{_SOFTWARE_DIR}/BindCraft")
BINDCRAFT2_DIR       = cfg.tool("bindcraft2_dir",        "BINDCRAFT2_DIR",        f"{_SOFTWARE_DIR}/BindCraft2")
BOLTZGEN_BIN         = os.environ.get("BOLTZGEN_BIN",    os.path.join(cfg.conda_env_prefix("boltzgen") or cfg.conda_env("boltzgen"), "bin", "boltzgen"))
PXDESIGN_DIR         = cfg.tool("pxdesign_dir",          "PXDESIGN_DIR",          f"{_SOFTWARE_DIR}/PXDesign")
PROTEINA_DIR         = cfg.tool("proteina_dir",          "PROTEINA_DIR",          f"{_SOFTWARE_DIR}/Proteina")
PROTEINA_CKPT        = cfg.tool("proteina_ckpt",         "PROTEINA_CKPT",        f"{_WEIGHTS_DIR}/proteina/proteina_v1.1_DFS_200M_tri.ckpt")
PROTEINA_COMPLEXA_DIR  = cfg.tool("proteina_complexa_dir",  "PROTEINA_COMPLEXA_DIR")
PROTEINA_COMPLEXA_VENV = cfg.tool("proteina_complexa_venv", "PROTEINA_COMPLEXA_VENV")
RFD3_ENV             = cfg.conda_env("rfdiffusion3")
RFD3_WEIGHTS         = cfg.tool("rfd3_weights",          "RFD3_WEIGHTS",         f"{_WEIGHTS_DIR}/foundry")

TOOL_COLORS = {
    # CANONICAL TOOL PALETTE — must stay identical to the copy in
    # binder_browser/config.py. These two were SWAPPED for boltzgen/bindcraft
    # (and differed for both proteina variants) until 2026-09-28, so
    # dashboard.png and the browser's Scatter/Tools/Radar tabs coloured the same
    # tool differently IN THE SAME WINDOW (dashboard_viewer.py renders the PNG
    # next to those tabs). web/config.py re-exports the browser copy, so the web
    # UI inherited the inversion too.
    "rfdiffusion":        "#E53935",
    "rfdiffusion3":       "#00BCD4",
    "boltzgen":           "#43A047",
    "bindcraft":          "#1E88E5",
    "bindcraft2":         "#7CB342",
    "pxdesign":           "#FB8C00",
    "proteina":           "#8E24AA",
    "proteina_complexa":  "#00897B",
}
ALL_TOOLS = list(TOOL_COLORS)

# Tools that produce their own iPTM scores natively (no Boltz-2 re-prediction needed)
IPTM_NATIVE_TOOLS = {"bindcraft", "bindcraft2", "boltzgen", "pxdesign", "proteina_complexa"}
# Backbone-only generators that need Boltz-2 for iPTM scoring
BACKBONE_ONLY_TOOLS = {"rfdiffusion", "proteina", "rfdiffusion3"}

# PyMOL/ChimeraX colour per tool. SINGLE source — the .pml writers and the PNG
# renderer all read this. Deliberately avoids `orange`, which is reserved for the
# epitope in every scene this project produces.
PYMOL_TOOL_COLORS = {
    "rfdiffusion":       "salmon",
    "rfdiffusion3":      "palecyan",
    "boltzgen":          "palegreen",
    "bindcraft":         "lightblue",
    "bindcraft2":        "lightpink",
    "pxdesign":          "wheat",
    "proteina":          "yellow",
    "proteina_complexa": "violet",
}

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
    "test":       {"rfdiffusion":  20, "rfdiffusion3":  20, "boltzgen":    50, "bindcraft":   5, "bindcraft2":   5, "pxdesign":   20, "proteina":   20, "proteina_complexa":   20},
    "standard":   {"rfdiffusion": 200, "rfdiffusion3": 100, "boltzgen":   500, "bindcraft":  10, "bindcraft2":  10, "pxdesign":  200, "proteina":  200, "proteina_complexa":  200},
    "production": {"rfdiffusion": 500, "rfdiffusion3": 250, "boltzgen":  2000, "bindcraft":  20, "bindcraft2":  20, "pxdesign":  500, "proteina":  500, "proteina_complexa":  500},
}
# Rough minutes per design, per tool, on one 20 GB GPU. Used to cost a SUBSET —
# the documented "test ~4h" figure is for all 8 tools, but every real run is a
# subset and those were never costed, so an agent had to extrapolate before
# committing a human's GPU. Order-of-magnitude only; BoltzGen dominates and
# BindCraft/BindCraft2 are slow per accepted design.
# Calibrated so the all-8 `test` case (160 designs) lands near the ~4 h figure
# documented for it, rather than introducing a second, disagreeing number.
TOOL_MINUTES_PER_DESIGN = {
    "rfdiffusion": 0.40, "rfdiffusion3": 0.60, "boltzgen": 1.40,
    "bindcraft": 7.5, "bindcraft2": 7.5, "pxdesign": 0.60,
    "proteina": 0.35, "proteina_complexa": 0.90,
}


def estimate_runtime_minutes(n_designs, tools=None, reprediction=False):
    """Estimate wall-clock minutes for a given per-tool design count."""
    tools = tools or list(n_designs)
    total = sum(TOOL_MINUTES_PER_DESIGN.get(t, 1.0) * n_designs.get(t, 0) for t in tools)
    # Validation (Boltz-2 monomer + complex, Rosetta) scales with the design total
    n_total = sum(n_designs.get(t, 0) for t in tools)
    total += n_total * (0.9 if reprediction else 0.35)
    return total


def format_runtime(minutes):
    if minutes < 90:
        return f"~{minutes:.0f} min"
    if minutes < 60 * 36:
        return f"~{minutes/60:.1f} h"
    return f"~{minutes/60/24:.1f} days"


# Reduced BindCraft counts when filters are active (strict filters = slow per design).
# Shared by v1 and BC2: both spend the same kind of time per *accepted* design, and
# a strict gate multiplies the trajectories needed to reach the requested count.
BINDCRAFT_FILTERED_COUNTS = {
    "test": 2, "standard": 4, "production": 10,
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

def log(msg, err=None):
    """
    Timestamped log line.

    Everything used to go to stdout — including every WARNING and every
    `{tool} FAILED` — so an empty stderr was not evidence of success, and an
    agent had to parse progress chatter to find out whether anything went wrong.
    Problem lines now go to stderr, leaving stdout as a clean progress stream.

    `err=None` auto-detects from the message prefix; pass err=True/False to force.
    """
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if err is None:
        s = str(msg).lstrip()
        # Case-INSENSITIVE: the original matched only the exact spellings
        # "WARNING"/"Warning:"/"ERROR"/"FAILED", so real problem lines written as
        # "failed", "Error:", "could not", "skipping" etc. went to stdout and an
        # empty stderr still did not mean success.
        _s = s.lower()
        err = (_s.startswith(("warning", "error", "failed", "fatal", "traceback"))
               or "failed" in _s[:80] or "error" in _s[:80]
               or "warning" in _s[:80])
    print(line, file=sys.stderr if err else sys.stdout, flush=True)


# GPU pinning env — set by --device flag, merged into every run_cmd call
GPU_ENV = {}

class SkipRequested(Exception):
    """Raised when user requests skipping the current tool via signal file."""
    pass


def _partial_pick_binder_chain(chains, target_seq=None):
    """Pick the binder chain from a gemmi chain list, least-identical-to-target.

    The shortest chain is NOT the binder whenever the binder is longer than the
    target — the bug fixed for the main collectors on 2026-06-17. Without a known
    target sequence we can only fall back to length, so say so at the call site
    rather than pretending.
    """
    def _seq(ch):
        try:
            import gemmi
            return "".join(
                (r.name[-1] if len(r.name) == 1 else
                 (gemmi.find_tabulated_residue(r.name).one_letter_code or "x")).upper()
                for r in ch if r.find_atom("CA", "\0"))
        except Exception:
            return ""
    if target_seq:
        scored = []
        for ch in chains:
            sq = _seq(ch)
            if not sq:
                continue
            # crude identity: fraction of positions equal over the overlap
            n = min(len(sq), len(target_seq))
            ident = (sum(1 for i in range(n) if sq[i] == target_seq[i]) / n) if n else 0.0
            scored.append((ident, ch))
        if scored:
            return min(scored, key=lambda t: t[0])[1]
    return min(chains, key=lambda c: sum(1 for _ in c))


def _collect_partial_output(tool, tool_dir, chain_id="A", target_seq=None):
    """Collect whatever designs exist in a tool's output directory after skip/crash.

    Returns a list of design dicts with at minimum design_id + binder_sequence.
    """
    from pathlib import Path
    tool_dir = Path(tool_dir)
    designs = []

    if tool in ("rfdiffusion", "proteina"):
        # Backbone tools: collect sequences from LigandMPNN output
        seq_dir = tool_dir / "sequences" / "seqs"
        if not seq_dir.exists():
            seq_dir = tool_dir / "sequences"
        for fa in sorted(tool_dir.rglob("*.fa")):
            try:
                # Mirror run_rfdiffusion's own FASTA handling. This used to take the
                # FIRST record and keep it whole, which is wrong twice over:
                #   1. record 1 of a LigandMPNN/ProteinMPNN .fa is the INPUT TEMPLATE,
                #      not a sample — it is the only record with no overall_confidence.
                #   2. the sequence is the full complex `TARGET:BINDER`, so without the
                #      split the "binder_sequence" was a 220-char string containing a
                #      colon (150 target + 69 binder), which is not a binder at all.
                # Both only surfaced on --skip or a crash, so recovered designs were
                # silently garbage.
                best_seq, best_conf = None, -1.0
                header = None
                for line in fa.read_text().splitlines():
                    if line.startswith(">"):
                        header = line
                        continue
                    if not (header and line.strip()):
                        continue
                    m = re.search(r"overall_confidence=([0-9.]+)", header)
                    conf = float(m.group(1)) if m else -1.0   # template -> stays last
                    if conf > best_conf:
                        best_conf, best_seq = conf, line.strip()
                    header = None
                if best_seq and best_conf >= 0:
                    seq = best_seq.split(":")[-1] if ":" in best_seq else best_seq
                    if seq:
                        did = f"{tool}_{len(designs):04d}"
                        designs.append({"design_id": did, "binder_sequence": seq,
                                        "native_score": best_conf})
            except Exception:
                continue

    elif tool == "boltzgen":
        # BoltzGen: collect CIFs from final or intermediate directories
        import gemmi
        for subdir_name in ("final_ranked_designs", "final_50_designs",
                            "intermediate_designs_inverse_folded"):
            cif_dir = tool_dir / "boltzgen_raw" / subdir_name
            if not cif_dir.exists():
                continue
            # Non-recursive: final_ranked_designs/ CONTAINS final_50_designs/, so
            # rglob returned 120 CIFs for a 50-design run (2.4x duplicates, each
            # becoming its own "design" in the panel).
            _cifs = sorted(cif_dir.glob("*.cif")) or sorted(cif_dir.rglob("*.cif"))
            for cif in _cifs:
                try:
                    doc = gemmi.cif.read(str(cif))
                    st = gemmi.make_structure_from_block(doc.sole_block())
                    chains = list(st[0])
                    if len(chains) < 2:
                        continue
                    # Binder = chain LEAST like the target, not the shortest. The
                    # shortest-chain heuristic grabs the TARGET whenever a binder is
                    # longer than it (fixed elsewhere 2026-06-17; these copies were
                    # missed).
                    binder = _partial_pick_binder_chain(chains, target_seq)
                    seq = "".join(r.name[-1] if len(r.name) == 1 else
                                 gemmi.find_tabulated_residue(r.name).one_letter_code
                                 for r in binder if r.find_atom("CA", "\0"))
                    if seq and len(seq) > 5:
                        did = f"boltzgen_{len(designs):04d}"
                        designs.append({"design_id": did, "binder_sequence": seq})
                except Exception:
                    continue
            if designs:
                break

    elif tool == "bindcraft":
        # BindCraft: collect from Accepted/Ranked/ or final_design_stats.csv
        csv_path = tool_dir / "bindcraft_output" / "final_design_stats.csv"
        if csv_path.exists():
            import csv as _csv
            with open(csv_path) as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    seq = row.get("Sequence", "")
                    name = row.get("Design", "")
                    if seq:
                        did = f"bindcraft_{len(designs):04d}"
                        designs.append({"design_id": did, "binder_sequence": seq})

    elif tool == "bindcraft2":
        # BindCraft2 rewrites 3_Ranked/!_Ranked.csv after each acceptance, so a
        # skipped campaign still has every design accepted up to that point —
        # structures included. Reuse the full parser rather than scraping
        # sequences, which is all the v1 branch above can recover.
        # chain_id matters: _bc2_pick_binder_chain's fallback excludes the target
        # BY LETTER, so a default of "A" against a --site on another chain can
        # pick the target as the binder.
        designs.extend(parse_bindcraft2_outputs(tool_dir / "bindcraft2_output",
                                                target_chain=chain_id))

    elif tool == "pxdesign":
        # PXDesign: collect from summary.csv
        import csv as _csv
        for summary in tool_dir.rglob("summary.csv"):
            with open(summary) as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    seq = row.get("sequence", "")
                    if seq:
                        did = f"pxdesign_{len(designs):04d}"
                        designs.append({"design_id": did, "binder_sequence": seq})
            break

    elif tool == "proteina_complexa":
        # Proteina Complexa: collect from inference PDBs
        import gemmi
        inference_base = Path(PROTEINA_COMPLEXA_DIR) / "inference"
        # Use the SAME path-hashed tag run_proteina_complexa writes. This used
        # the bare `{parent}_{name}` form with a trailing glob, so it matched
        # other campaigns' inference dirs and merged every hit — the exact
        # cross-campaign contamination _complexa_run_name() exists to prevent.
        run_name = _complexa_run_name(tool_dir)
        _rds = sorted(inference_base.glob(f"*{run_name}*"))
        if not _rds:
            _rds = sorted(inference_base.glob(f"*{tool_dir.parent.name}_{tool_dir.name}"))
        if len(_rds) > 1:
            _rds = sorted(_rds, key=lambda q: q.stat().st_mtime, reverse=True)[:1]
            log(f"  _collect_partial_output(proteina_complexa): several inference dirs "
                f"matched; using only the newest ({_rds[0].name})", err=True)
        for rd in _rds:
            for pdb in sorted(rd.rglob("*.pdb")):
                if "_binder" in pdb.stem:
                    continue
                try:
                    st = gemmi.read_structure(str(pdb))
                    chains = list(st[0])
                    # Binder = chain LEAST like the target, not the shortest. The
                    # shortest-chain heuristic grabs the TARGET whenever a binder is
                    # longer than it (fixed elsewhere 2026-06-17; these copies were
                    # missed).
                    binder = _partial_pick_binder_chain(chains, target_seq)
                    seq = "".join(
                        gemmi.find_tabulated_residue(r.name).one_letter_code
                        for r in binder if r.find_atom("CA", "\0"))
                    if seq and len(seq) > 5:
                        did = f"proteina_complexa_{len(designs):04d}"
                        designs.append({"design_id": did, "binder_sequence": seq})
                except Exception:
                    continue

    elif tool == "rfdiffusion3":
        import gzip
        import gemmi
        rfd3_out = tool_dir / "rfd3_output"
        if rfd3_out.exists():
            for cif_gz in sorted(rfd3_out.glob("*.cif.gz")):
                try:
                    with gzip.open(str(cif_gz), "rt") as f:
                        doc = gemmi.cif.read_string(f.read())
                    st = gemmi.make_structure_from_block(doc.sole_block())
                    chains = list(st[0])
                    if len(chains) < 2:
                        continue
                    # Binder = chain LEAST like the target, not the shortest. The
                    # shortest-chain heuristic grabs the TARGET whenever a binder is
                    # longer than it (fixed elsewhere 2026-06-17; these copies were
                    # missed).
                    binder = _partial_pick_binder_chain(chains, target_seq)
                    seq = "".join(
                        gemmi.find_tabulated_residue(r.name).one_letter_code
                        for r in binder if r.find_atom("CA", "\0"))
                    if seq and len(seq) > 5:
                        did = f"rfdiffusion3_{len(designs):04d}"
                        designs.append({"design_id": did, "binder_sequence": seq})
                except Exception:
                    continue

    log(f"  _collect_partial_output({tool}): found {len(designs)} designs")
    return designs


class ToolError(RuntimeError):
    """
    A tool subprocess failed, with the failure CLASSIFIED.

    Plain RuntimeError(stderr[-3000:]) makes an OOM, a missing conda env and a
    bad argument indistinguishable — and the 3000-char tail can truncate the
    signature line away entirely. An agent driving this pipeline has to be able
    to tell "retry smaller" from "fix the install" from "fix my arguments", so
    the kind is carried on the exception and recorded in status.json.

    .kind is one of FAILURE_KINDS.
    """
    def __init__(self, message, kind="unknown", cmd=None, returncode=None):
        super().__init__(message)
        self.kind = kind
        self.cmd = cmd
        self.returncode = returncode


FAILURE_KINDS = ("oom", "missing_env", "missing_weights", "cuda_error",
                 "bad_args", "timeout", "killed", "unknown")

# Validation-stage failures (Boltz monomer, Boltz complex, Rosetta, ...).
#
# These stages run AFTER the design-tool loop, and `args._tool_failures` — the
# only thing status.json reads for failure_kinds — is written exclusively inside
# that loop. So a wholesale Boltz monomer crash (OOM, broken env, missing CLI)
# produced `esmfold_plddt = 0.0` for every design, was reported as "0/N passed"
# and left status.json saying ok:true. An infrastructure failure was indistinguishable
# from the scientific result "none of these designs refold".
VALIDATION_FAILURES = {}


def record_validation_failure(stage, exc):
    """Record a validation-stage failure so it reaches status.json, and say so loudly."""
    kind = getattr(exc, "kind", None) or classify_failure(str(exc))
    VALIDATION_FAILURES[stage] = kind
    log(f"  VALIDATION STAGE FAILED: {stage} [{kind}] — {exc}", err=True)
    return kind


# Ordered: first match wins, so the more specific patterns come first.
_FAILURE_SIGNATURES = [
    ("oom", (
        "CUDA out of memory", "torch.cuda.OutOfMemoryError", "OutOfMemoryError",
        "RESOURCE_EXHAUSTED", "Out of memory while trying to allocate",
        "cudaErrorMemoryAllocation", "CUBLAS_STATUS_ALLOC_FAILED",
        "failed to allocate", "XLA_PYTHON_CLIENT_MEM_FRACTION",
    )),
    ("missing_env", (
        "EnvironmentLocationNotFound", "Could not find conda environment",
        "conda: command not found", "No such file or directory: 'conda'",
        "ModuleNotFoundError", "ImportError", "command not found",
    )),
    # cuda_error and bad_args come FIRST: missing_weights used to match the bare
    # substring "checkpoint", so any CUDA traceback or argparse error that merely
    # mentioned a --checkpoint flag was reported as missing weights — sending the
    # caller to "fix the install" instead of "retry smaller" or "fix my args".
    ("cuda_error", (
        "CUDA error", "cuDNN error", "no CUDA-capable device",
        "CUDA driver version is insufficient", "NVIDIA driver",
    )),
    ("missing_weights", (
        "checkpoint not found", "no checkpoint", "missing checkpoint",
        "cannot find checkpoint", "checkpoint file does not exist",
        "No such file or directory: '/", "weights not found",
        "FileNotFoundError",
    )),
    ("bad_args", (
        "unrecognized arguments", "error: argument", "invalid choice",
        "the following arguments are required", "usage:",
    )),
]


def classify_failure(text):
    """Map a subprocess's stderr/stdout tail to one of FAILURE_KINDS."""
    if not text:
        return "unknown"
    for kind, needles in _FAILURE_SIGNATURES:
        for n in needles:
            if n.lower() in text.lower():
                return kind
    return "unknown"


def _tool_error(stdout, stderr, cmd=None, returncode=None, label=None):
    """
    Build a classified ToolError. Keeps BOTH the head and tail of stderr: the
    signature line (e.g. `torch.cuda.OutOfMemoryError`) is often at the top of a
    long traceback and a pure tail slice loses it.
    """
    blob = (stderr or "") + ("\n" + stdout if stdout else "")
    kind = classify_failure(blob)
    # A process killed by a SIGNAL reports a NEGATIVE returncode and usually has
    # empty stderr, so text classification alone always returned "unknown" and
    # the advertised "killed" kind was unreachable. The OOM killer (SIGKILL) is
    # the common case on this box and deserves its own answer: it means the host
    # ran out of RAM, which is a different fix from a CUDA OOM.
    if isinstance(returncode, int) and returncode < 0 and kind == "unknown":
        import signal as _sig
        try:
            _name = _sig.Signals(-returncode).name
        except (ValueError, AttributeError):
            _name = f"signal {-returncode}"
        kind = "killed"
        stderr = (stderr or "") + (
            f"\n[process terminated by {_name}"
            + (" — this is usually the host OOM killer; check RAM, not VRAM]"
               if -returncode == 9 else "]"))
    if stderr and len(stderr) > 4000:
        excerpt = stderr[:1500] + "\n  ...[trimmed]...\n" + stderr[-2500:]
    else:
        excerpt = stderr or (stdout or "")[-3000:]
    prefix = f"[{kind}]" + (f" {label}" if label else "")
    return ToolError(f"{prefix}: {excerpt}", kind=kind, cmd=cmd, returncode=returncode)


def run_cmd(cmd, timeout=86400, extra_env=None, cwd=None, dry_run=False,
            progress_dir=None, progress_pattern=None, progress_total=None,
            progress_label=None, skip_file=None):
    """Run a command with optional file-based progress monitoring and skip support.

    When progress_dir/pattern/total are given, a background thread polls
    the output directory and logs progress every 15 seconds.

    When skip_file is given, the polling thread also checks for a skip signal
    file. If found, the subprocess is terminated and SkipRequested is raised.

    Raises RuntimeError on failure, SkipRequested on user skip.
    """
    if dry_run:
        print(f"  [DRY RUN] {' '.join(str(c) for c in cmd)}")
        return "", ""
    env = {**os.environ, **GPU_ENV, **(extra_env or {})}

    if not (progress_dir and progress_pattern and progress_total):
        # Simple mode: no progress tracking or skip support
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               env=env, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            # subprocess.TimeoutExpired derives from SubprocessError, NOT from
            # RuntimeError — so it slipped past every `except RuntimeError`
            # handler in this module (ToolError IS a RuntimeError, so non-zero
            # exits were handled but timeouts were not). A Rosetta batch that
            # ran long therefore killed the whole pipeline at the very last
            # stage, after all the GPU work was already done. Convert it into
            # the same classified ToolError everything else raises.
            err = _tool_error(
                (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                f"Timed out after {timeout}s",
                cmd=cmd, returncode=None, label=progress_label)
            err.kind = "timeout"
            raise err from e
        if r.returncode != 0:
            raise _tool_error(r.stdout, r.stderr, cmd=cmd,
                              returncode=r.returncode, label=progress_label)
        return r.stdout, r.stderr

    # Progress mode: use Popen + polling thread
    import threading
    import signal as _signal
    from pathlib import Path

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env, cwd=cwd,
                            start_new_session=True)  # new session for killpg
    stop_event = threading.Event()
    skipped = threading.Event()
    label = progress_label or "progress"
    pdir = Path(progress_dir)
    skip_path = Path(skip_file) if skip_file else None

    def _poll():
        last_count = -1
        while not stop_event.is_set():
            # Check skip signal
            if skip_path and skip_path.exists():
                try:
                    skip_path.unlink()  # consume signal
                except OSError:
                    pass
                log(f"  [SKIP] User requested skip of {label}")
                skipped.set()
                # Kill process group
                try:
                    os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
                # Escalate after 5s
                stop_event.wait(5)
                try:
                    os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                return

            # Count progress
            try:
                count = len(list(pdir.glob(progress_pattern)))
            except Exception:
                count = 0
            if count != last_count:
                log(f"  [PROGRESS:{label}] {count}/{progress_total}")
                last_count = count
            stop_event.wait(15)

    poll_thread = threading.Thread(target=_poll, daemon=True)
    poll_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the whole process GROUP. The Popen sets start_new_session=True
        # precisely so killpg works; a bare proc.kill() reaps only the direct
        # child (usually `conda run`) and leaves the grandchild holding VRAM.
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        proc.kill()
        stdout, stderr = proc.communicate()
        stop_event.set()
        raise ToolError(f"[timeout] {progress_label or 'tool'}: timed out after {timeout}s",
                        kind="timeout", cmd=cmd)
    finally:
        stop_event.set()
        poll_thread.join(timeout=8)

    if skipped.is_set():
        raise SkipRequested(f"{label} skipped by user")

    if proc.returncode != 0:
        raise _tool_error(stdout, stderr, cmd=cmd,
                          returncode=proc.returncode, label=progress_label)
    return stdout, stderr


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
            # Is this process a DESCENDANT of ours? The check used to be a single
            # `ppid == my_pid`, but every tool runs as `conda run ... python`, so
            # the process actually holding VRAM is a GRANDchild — its ppid is the
            # `conda run` pid, never ours. The condition was therefore never true
            # and this cleanup killed nothing, so VRAM leaked between tools (which
            # matters on a 20 GB card). Walk the ancestry instead.
            def _ancestors(p, limit=6):
                out, cur = [], p
                for _ in range(limit):
                    try:
                        pr = subprocess.run(["ps", "-o", "ppid=", "-p", str(cur)],
                                            capture_output=True, text=True, timeout=5)
                        nxt = int(pr.stdout.strip())
                    except Exception:
                        break
                    if nxt <= 1:
                        out.append(nxt)
                        break
                    out.append(nxt)
                    cur = nxt
                return out

            chain = _ancestors(pid)
            mem_mb = int(parts[1].strip())
            if my_pid in chain:
                log(f"  GPU cleanup: killing orphan PID {pid} ({mem_mb} MiB on GPU {gpu_id})")
                try:
                    os.kill(pid, 9)
                except (OSError, ProcessLookupError):
                    pass
                import time; time.sleep(2)
            elif chain and chain[-1] <= 1:
                # Re-parented to init: probably ours, but we cannot prove it and
                # this box may be running unrelated GPU work. Report, do not kill.
                log(f"  GPU cleanup: PID {pid} holds {mem_mb} MiB on GPU {gpu_id} and is "
                    f"orphaned (re-parented to init). Not killing it — ownership cannot "
                    f"be proven. Kill it manually if it is a leftover of this pipeline.",
                    err=True)
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
    # "100-60" used to parse to (100, 60) and "0-0" to (0, 0), both accepted in
    # silence — the design tools then received an empty or inverted length range.
    if lo > hi:
        raise ValueError(f"Cannot parse '{length_str}': minimum {lo} is greater "
                         f"than maximum {hi}. Write it as {hi}-{lo}.")
    if lo < 10:
        raise ValueError(f"Binder length {lo} is below 10 residues — too short to "
                         f"fold. Typical miniprotein binders are 55-100 aa.")
    if hi > 500:
        raise ValueError(f"Binder length {hi} exceeds 500 residues. That is a "
                         f"domain, not a binder, and will not fit in VRAM here.")
    return lo, hi


def parse_n_designs(n_str):
    """
    Parse per-tool design counts.
    "rfdiffusion=200,boltzgen=5000,bindcraft=100" → dict
    "100" → {tool: 100 for all tools}
    """
    # Unknown tool keys used to be accepted and silently ignored, so a typo like
    # "bindcraft_2=50" produced a run with default counts and no warning at all.
    # Malformed values raised a bare ValueError traceback from main().
    if "=" not in n_str:
        try:
            n = int(n_str.strip())
        except ValueError:
            raise ValueError(f"Cannot parse '{n_str}'. Expected a number, or "
                             f"'tool=N,tool=N' using names from {ALL_TOOLS}.")
        if n < 1:
            raise ValueError(f"must be >= 1, got {n}.")
        return {t: n for t in ALL_TOOLS}

    result, unknown = {}, []
    for part in n_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Cannot parse '{part}' in --n_designs. Expected "
                             f"'tool=N' (or a bare number for all tools).")
        tool, n = part.split("=", 1)
        tool = tool.strip()
        try:
            n = int(n.strip())
        except ValueError:
            raise ValueError(f"'{n.strip()}' is not a number (in '{part}').")
        if n < 1:
            raise ValueError(f"count for {tool} must be >= 1, got {n}.")
        if tool not in ALL_TOOLS:
            unknown.append(tool)
        result[tool] = n
    if unknown:
        raise ValueError(f"unknown tool name(s) {unknown}. "
                         f"Valid names: {ALL_TOOLS}")
    if not result:
        raise ValueError(f"nothing parsed from '{n_str}'.")
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


# ── PDB renumbering for tools that require contiguous residues ─────────────────

def _renumber_pdb_contiguous(pdb_path, chain_id, out_dir):
    """Renumber a PDB chain to contiguous 1..N residue numbers.

    Returns (renumbered_pdb_path, old_to_new_map) where old_to_new_map
    maps original residue numbers to new ones. Only needed when the PDB
    has gaps in residue numbering (e.g., 761..790, 981..1030).

    If already contiguous, returns (original_path, identity_map).
    """
    out_dir = Path(out_dir)
    # Parse residue numbers in order of appearance
    seen = []
    seen_set = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                ch = line[21].strip()
                if ch != chain_id:
                    continue
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue
                if resnum not in seen_set:
                    seen.append(resnum)
                    seen_set.add(resnum)

    if not seen:
        return pdb_path, {}

    # Check if already contiguous
    is_contiguous = all(seen[i] + 1 == seen[i + 1] for i in range(len(seen) - 1))
    if is_contiguous:
        old_to_new = {r: r for r in seen}
        return pdb_path, old_to_new

    # Build mapping: old resnum → new (1-based contiguous)
    old_to_new = {r: i + 1 for i, r in enumerate(seen)}

    # Rewrite PDB with new residue numbers
    renumbered_path = out_dir / f"{Path(pdb_path).stem}_renum.pdb"
    with open(pdb_path) as fin, open(renumbered_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                ch = line[21].strip()
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    fout.write(line)
                    continue
                if ch == chain_id and resnum in old_to_new:
                    new_num = old_to_new[resnum]
                    line = line[:22] + f"{new_num:4d}" + line[26:]
                fout.write(line)
            else:
                fout.write(line)

    n_gaps = len(seen) - 1 - sum(1 for i in range(len(seen) - 1) if seen[i] + 1 == seen[i + 1])
    log(f"  PDB renumbered: {seen[0]}-{seen[-1]} → 1-{len(seen)} ({n_gaps} gaps filled)")

    return str(renumbered_path), old_to_new


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
    residues = tchain["residues"] if tchain else list(range(1, 201))

    # RFdiffusion requires contiguous residue numbering. Auto-renumber if gaps.
    rf_target = str(target_path)
    rf_site_resnums = list(site_resnums)
    if not dry_run:
        renum_path, old_to_new = _renumber_pdb_contiguous(
            str(target_path), chain_id, out_dir)
        if renum_path != str(target_path):
            rf_target = renum_path
            rf_site_resnums = [old_to_new[r] for r in site_resnums if r in old_to_new]
            residues = [old_to_new[r] for r in residues if r in old_to_new]
    res_start, res_end = residues[0], residues[-1]

    # RFdiffusion works best with 3-6 hotspot residues. If the user specified
    # more, subsample to ~5 evenly spaced residues, preferring those with
    # strong chemical identity (charged, aromatic) over featureless ones (G, A, S).
    MAX_HOTSPOTS = 6
    rf_site = list(rf_site_resnums)
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
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("rfdiffusion"),
        "python", f"{RFDIFFUSION_DIR}/scripts/run_inference.py",
        f"inference.input_pdb={Path(rf_target).resolve()}",
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
    run_cmd(rf_cmd, timeout=None, dry_run=dry_run,
            progress_dir=str(bb_dir), progress_pattern="design_*.pdb",
            progress_total=n_designs, progress_label="rfdiffusion",
            skip_file=str(Path(out_dir).parent / ".skip_rfdiffusion"))

    backbones = sorted(bb_dir.glob("design_*.pdb"))
    if not backbones and not dry_run:
        raise FileNotFoundError(f"No backbone PDBs in {bb_dir}")

    # LigandMPNN: pass all backbones at once via --pdb_path_multi JSON
    multi_json = out_dir / "backbone_list.json"
    multi_json.write_text(json.dumps({str(p): "" for p in backbones}))

    mpnn_cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("mpnn"),
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


def _boltzgen_metrics_index(metrics_csv):
    """Map every usable key -> design_to_target_iptm from BoltzGen's metrics CSV.

    The CSV has NO `name` column — it has `id` ("design_35_1") and `file_name`
    ("design_35_1.cif"). Reading row["name"] therefore keyed every row on "" and
    `iptm_scores.get(cif.stem)` never hit, so EVERY BoltzGen design got
    native_score = 0.0000 (verified: 50/50 zeros in a real run).

    The ranked CIFs are also renamed with a `rankNN_` prefix
    ("rank01_design_35_1.cif"), so the lookup must strip that too.
    """
    import csv as _csv
    from pathlib import Path as _P
    idx = {}
    try:
        with open(metrics_csv) as fh:
            for row in _csv.DictReader(fh):
                raw = (row.get("design_to_target_iptm") or row.get("iptm")
                       or row.get("iPTM") or "")
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                for k in (row.get("id"), row.get("name"),
                          _P(row["file_name"]).stem if row.get("file_name") else None):
                    if k:
                        idx[str(k)] = val
    except (OSError, IOError):
        pass
    return idx


def _boltzgen_lookup(idx, stem, default=0.0):
    """iPTM for a ranked BoltzGen CIF stem, tolerating the `rankNN_` prefix."""
    if not idx or not stem:
        return default
    if stem in idx:
        return idx[stem]
    bare = re.sub(r"^rank\d+_", "", str(stem))
    if bare in idx:
        return idx[bare]
    for k, v in idx.items():
        if k and k in stem:
            return v
    return default


def run_boltzgen(target_path, chain_id, site_resnums, length_min, length_max,
                 n_designs, out_dir, dry_run=False, ss_bias="balanced",
                 filter_bindingsite=True):
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
    # Make BoltzGen enforce the binding site it was given. Its own
    # `filter_bindingsite` defaults to FALSE, so by default it accepts designs
    # that touch nothing near the epitope even though we passed `binding:` in
    # design.yaml. In runs/pdia6_validation_001 that produced 50 designs of
    # which 39 scored SIF < 0.25 and most contacted none of the six site
    # residues — BoltzGen spent its whole budget off-target.
    #
    # The filter is weak by design ("at least one binding-site residue within
    # 4 Å", threshold 1e-4 on bindsite_under_8rmsd), so it removes total misses
    # without gutting yield. Only meaningful when a site was actually given.
    if site_resnums and filter_bindingsite:
        cmd.extend(["--config", "filtering", "filter_bindingsite=true"])
        log("BoltzGen: enforcing its own binding-site filter "
            "(disable with --no_boltzgen_site_filter)")

    # SS bias: filter by helix/sheet fraction
    if ss_bias == "beta":
        cmd.extend(["--additional_filters", "helix<0.4"])
    elif ss_bias == "helix":
        cmd.extend(["--additional_filters", "sheet<0.2"])
    log(f"BoltzGen: running {n_designs} designs (may take hours)...")
    run_cmd(cmd, timeout=None, dry_run=dry_run,
            extra_env={"MKL_THREADING_LAYER": "GNU", "PYTHONNOUSERSITE": "1"},
            progress_dir=str(raw_dir), progress_pattern="intermediate_designs/*.cif",
            progress_total=n_designs, progress_label="boltzgen",
            skip_file=str(Path(out_dir).parent / ".skip_boltzgen"))

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
        metrics_csv = next(raw_dir.glob("final_ranked_designs/all_designs_metrics.csv"), None)
        iptm_scores = _boltzgen_metrics_index(metrics_csv)

        # Load full BoltzGen metrics keyed by binder sequence
        bg_metrics = _load_boltzgen_metrics(raw_dir)
        log(f"  BoltzGen: loaded metrics for {len(bg_metrics)} sequences from CSV")

        # Known target sequence — used to identify the binder chain by DISSIMILARITY rather than
        # by length. The old 'shortest chain = binder' heuristic silently grabs the TARGET when the
        # binder is longer than the target (e.g. 200-250 aa binders vs a 166 aa HRAS target), so the
        # complex YAML ends up with the target sequence duplicated into chain B.
        tgt_seq = pdb_chain_seq(target_path, chain_id)

        for i, cif in enumerate(cifs):
            chain_info = extract_cif_chain_info(cif)
            binder_info = pick_binder_chain_by_identity(chain_info, tgt_seq)
            seq = binder_info.get("sequence", "")
            if not seq:
                continue
            d = {
                "design_id":         f"boltzgen_{i:04d}",
                "binder_sequence":   seq,
                "complex_cif":       str(cif),
                "native_score":      _boltzgen_lookup(iptm_scores, cif.stem),
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


def sanitize_target_pdb(src_path, dst_path):
    """
    Write a cleaned copy of src_path for binder-design tools:
      - HETATM records removed (waters, ligands, ions, cryoprotectants).
        PXDesign's pdb_to_cif splits chain A into A (protein) + B (HETATM),
        then its chain-mapping check raises ValueError. Stripping HETATM
        prevents that and also simplifies Proteina Complexa / Boltz inputs.
      - CONECT records removed (references to stripped HETATMs would dangle).
      - Insertion codes (PDB col 27) cleared — BindCraft/ColabDesign reject them.
    """
    # Convert non-PDB inputs FIRST. This function is pure PDB line filtering, so
    # a .cif target used to be copied verbatim into "<stem>_clean.pdb" — a file
    # named .pdb holding mmCIF text (or, after filtering, nothing). Every
    # downstream fixed-column PDB parser then read garbage, silently. The CLI
    # rejects .cif only for rfdiffusion/bindcraft, so other tools hit this.
    src_path = Path(src_path)
    _suf = src_path.suffix.lower()
    if _suf in (".fasta", ".fa", ".faa", ".seq"):
        raise ValueError(
            f"--target must be a structure, not a sequence file: {src_path}. "
            f"Fold it first (e.g. with Boltz-2 or AlphaFold3) and pass the PDB.")
    if _suf in (".cif", ".mmcif"):
        _converted = Path(dst_path).with_suffix(".from_cif.pdb")
        _n = 0
        try:
            import gemmi
            _st = gemmi.read_structure(str(src_path))
            _st.setup_entities()
            _st.write_pdb(str(_converted))
            _n = sum(1 for l in _converted.read_text().splitlines()
                     if l.startswith(("ATOM", "HETATM")))
        except Exception:
            _n = 0
        if not _n:
            # gemmi 0.6.5 returns 0 models for several producers' mmCIF; fall
            # back to the header-driven reader used everywhere else.
            _n = cif_to_pdb_atoms(src_path, _converted)
        if not _n:
            raise ValueError(f"could not convert mmCIF target to PDB: {src_path}")
        log(f"  Target: converted mmCIF to PDB ({_n} atoms) -> {_converted.name}")
        src_path = _converted

    lines = Path(src_path).read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith(("HETATM", "CONECT", "ANISOU")):
            continue
        if line.startswith("ATOM") and len(line) >= 27 and line[26] != " ":
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

    # Target is pre-sanitized in main() (HETATMs + insertion codes stripped).

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

    effective_filters = filters_path or f"{BINDCRAFT_DIR}/settings_filters/no_filters.json"
    # Validate filter path — fall back to no_filters if invalid
    if not Path(effective_filters).is_file():
        log(f"  WARNING: filter file '{effective_filters}' not found, using no_filters")
        effective_filters = f"{BINDCRAFT_DIR}/settings_filters/no_filters.json"
    filter_name = Path(effective_filters).stem
    log(f"BindCraft: filters = {filter_name} ({effective_filters})")
    cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("bindcraft"),
        "python", f"{BINDCRAFT_DIR}/bindcraft.py",
        "--settings", str(settings_path),
        "--filters",  effective_filters,
        "--advanced", str(effective_advanced),
    ]
    log(f"BindCraft: running until {n_designs} designs accepted...")
    run_cmd(cmd, timeout=None, cwd=BINDCRAFT_DIR, dry_run=dry_run,
            progress_dir=str(design_path / "Accepted"),
            # NOT "**/*.pdb": Accepted/ and Accepted/Ranked/ hold the SAME
            # designs, so a recursive glob counted every design twice and the
            # progress line reported 2x the real yield.
            progress_pattern="*.pdb",
            progress_total=n_designs, progress_label="bindcraft",
            skip_file=str(Path(out_dir).parent / ".skip_bindcraft"))

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


# ── BindCraft2 ────────────────────────────────────────────────────────────────
# BindCraft2 (PacesaLab/BindCraft2) is a rewrite, not an upgrade in place: one
# campaign JSON instead of v1's settings/filters/advanced trio, `bindcraft design`
# instead of `python bindcraft.py`, and mmCIF under 3_Ranked/ instead of PDB under
# Accepted/. It runs in its own Python >=3.12 env — the v1 `bindcraft` env stays
# on 3.10 because it carries the PyRosetta/gemmi/DAlphaBall stack that
# rosetta_score_interfaces() uses for *every* tool.

# BINDCRAFT2_DIR is declared beside the other tool dirs at the top of the file.
#
# BC2 looks for its AlphaFold parameters under $BINDCRAFT_WEIGHTS/alphafold, and
# defaults to ~/.cache/bindcraft when that is unset. This platform keeps weights
# under weights_dir, so the variable has to be exported on every invocation —
# without it BC2 silently re-downloads 5.3 GB into the home partition.
#
# Note alphafold_parameters() returns $BINDCRAFT_AF2_PARAMS verbatim when set,
# without checking that the directory actually holds params_*.npz and without
# downloading. Point BINDCRAFT_WEIGHTS at the cache root instead; setting
# BINDCRAFT_AF2_PARAMS at an empty directory turns `fetch-weights` into a no-op.
BINDCRAFT2_WEIGHTS = cfg.tool("bindcraft2_weights", "BINDCRAFT2_WEIGHTS",
                              f"{_WEIGHTS_DIR}/bindcraft2")

# Modality presets shipped in BindCraft2/settings/modality/. Passed straight
# through to `bindcraft design --modality`; BC2 validates the name itself, this
# list only drives --help text and an early typo check.
BINDCRAFT2_MODALITIES = [
    "binder", "large_binder", "peptide", "cyclic_peptide", "VHH", "scFv",
    "Fab", "ARP", "homo_oligomer", "multidomain", "fold_switch", "induced_fit",
]

# BC2 keeps acceptance filters inline in the campaign JSON rather than in separate
# settings_filters/*.json files, so a preset here is a set of threshold overrides
# merged into the campaign. `None` disables a check — BC2 reads a null threshold
# as "not measured", which is different from a low score. Preset names match the
# v1 --bindcraft_filters vocabulary so the web dropdown carries over unchanged.
#
# BC2 core defaults (settings/core/default.json) for reference:
#   Unbound_Binder_pLDDT >=0.8, pTM >=0.55, i_pTM >=0.7, i_pAE <=0.35,
#   Backbone_Clashes <=0, Interface_Residues >=7  (+ Binder_RMSD <=3.5 for `binder`)
_BC2_CHECKS = ("Unbound_Binder_pLDDT", "pTM", "i_pTM", "i_pAE",
               "Backbone_Clashes", "Interface_Residues", "Binder_RMSD")

# BC2 gates a design TWICE, and the two are independent:
#   1. acceptance `filters`, applied to a finished ProteinMPNN candidate;
#   2. per-stage thresholds min_iptm_<stage> / min_plddt_<stage>, which terminate
#      a gradient trajectory mid-flight (defaults: i_pTM 0.5 at anneal/harden/
#      mutate, 0.7 final; pLDDT 0.6-0.7 throughout).
# Nulling only (1) is not "no filters": a trajectory reading i_pTM 0.4 at anneal
# is killed before acceptance is ever considered, and the campaign ends with
# nothing accepted. Both have to be opened for the v1 meaning of the flag.
# A null threshold disables the check — see BindCraft2 bindcraft/filters.py:825,
# `DesignFilter(..., None if threshold is None else float(threshold), ...)`.
_BC2_STAGES = ("screen", "refine", "anneal", "harden", "mutate", "final")

# BC2's FINAL_CONFIDENCE_FILTERS (bindcraft/settings.py:37) are a third spelling
# of the same idea, set as scalars rather than under `filters`.
_BC2_FINAL_CONFIDENCE = ("min_monomer_plddt_final", "min_ptm_final",
                         "min_iptm_final", "max_ipae_final")

def _bc2_filters(**thresholds):
    """Build a BC2 `filters` block. Unnamed checks are disabled (threshold: null)."""
    return {k: {"threshold": thresholds.get(k)} for k in _BC2_CHECKS}

def _bc2_open_stage_gates():
    """Disable every stage gate, so no trajectory is terminated early."""
    gates = {}
    for stage in _BC2_STAGES:
        gates[f"min_iptm_{stage}"]  = None
        gates[f"min_plddt_{stage}"] = None
    for name in _BC2_FINAL_CONFIDENCE:
        gates[name] = None
    return gates

# A preset is {"filters": <filters block>, "settings": <top-level overrides>}.
BINDCRAFT2_FILTER_PRESETS = {
    # Accept whatever BC2 produces; this pipeline re-scores with Boltz-2 and ranks
    # downstream. Matches v1 no_filters semantics. Costs GPU: with no stage gate,
    # a hopeless trajectory still runs to completion instead of dying at anneal.
    "no_filters": {"filters":  _bc2_filters(),
                   "settings": _bc2_open_stage_gates()},
    "relaxed":    {"filters":  _bc2_filters(i_pTM=0.50, i_pAE=0.45,
                                            Unbound_Binder_pLDDT=0.70,
                                            Backbone_Clashes=0, Interface_Residues=5),
                   "settings": {"min_iptm_anneal": 0.35, "min_iptm_harden": 0.35,
                                "min_iptm_mutate": 0.35, "min_iptm_final": 0.50,
                                "min_monomer_plddt_final": None}},
    # Empty keeps BC2's own core defaults — strict, may stall on difficult targets.
    "default":    {"filters": {}, "settings": {}},
    "kras_relaxed": {"filters":  _bc2_filters(i_pTM=0.45, i_pAE=0.50,
                                              Backbone_Clashes=0, Interface_Residues=4),
                     "settings": {"min_iptm_anneal": 0.30, "min_iptm_harden": 0.30,
                                  "min_iptm_mutate": 0.30, "min_iptm_final": 0.45,
                                  "min_monomer_plddt_final": None,
                                  "min_ptm_final": None}},
    # Peptides need no confident free fold, so the monomer/pTM gates come off —
    # the same reasoning BC2's own `peptide` modality applies.
    "peptide":    {"filters":  _bc2_filters(i_pTM=0.60, i_pAE=0.33,
                                            Backbone_Clashes=0, Interface_Residues=4),
                   "settings": {"min_monomer_plddt_final": None,
                                "min_ptm_final": None}},
    "peptide_relaxed": {"filters":  _bc2_filters(i_pTM=0.45, i_pAE=0.50,
                                                 Backbone_Clashes=0, Interface_Residues=3),
                        "settings": {"min_iptm_anneal": 0.30, "min_iptm_harden": 0.30,
                                     "min_iptm_mutate": 0.30, "min_iptm_final": 0.45,
                                     "min_monomer_plddt_final": None,
                                     "min_ptm_final": None}},
}

# Numeric columns lifted from 3_Ranked/!_Ranked.csv into bc2_* record keys.
# Key transform is a plain lowercase: "i_pTM" -> "bc2_i_ptm".
#
# BC2 computes a measurement only when something asks for it — an active filter,
# the ranking, or an explicit request — so WHICH of these columns exist depends on
# the filter preset. Verified 2026-09-28 on a `no_filters` campaign: the CSV held
# i_pDAE/i_pTM/Target_pLDDT/SS_pLDDT and the geometry columns, but NOT pLDDT, pTM,
# i_pAE, Unbound_Binder_pLDDT, Interface_Residues or Backbone_Clashes. Everything
# here is therefore optional; absent columns simply do not appear on the record.
# Binder pLDDT specifically is backfilled from the CIF B-factor column below,
# because the scoring formula needs it and `no_filters` is the default preset.
BINDCRAFT2_NUMERIC_COLS = [
    # confidence / error
    "i_pDAE", "i_pTM", "i_pAE", "pTM", "pLDDT",
    "Unbound_Binder_pLDDT", "Target_pLDDT", "SS_pLDDT",
    # interface geometry
    "Interface_Residues", "Interface_BuriedArea", "Interface_Hydrophobicity",
    "Backbone_Clashes", "All_Atom_Clashes",
    # binder shape / composition
    "Binder_Helix_Fraction", "Binder_BetaSheet_Fraction", "Binder_Loop_Fraction",
    "Binder_Length", "Binder_Mass_kDa", "Binder_pI", "Binder_Net_Charge",
    "Binder_Extinction", "Binder_Cysteines", "Binder_Disulfides",
    "Binder_Free_Cysteines", "Binder_Chain_Breaks", "Binder_RMSD",
    "Surface_Hydrophobicity",
    # site adherence — BC2's native analogue of this pipeline's SIF
    "Hotspot_Contact_Fraction", "Off_Epitope_Contact_Fraction",
    # emitted under no_filters too — observed on a real run
    "length", "Interface_BuriedArea_Fraction", "Epitope_Residues_Contacted",
    "Target_RMSD", "Interface_Helix_Fraction", "Interface_BetaSheet_Fraction",
    "Interface_Loop_Fraction",
    # format-specific, present only for some modalities
    "Cyclic_Closure_Distance", "Termini_Distance", "Induced_Fit_RMSD",
]


def _bc2_plddt_by_chain(cif_path):
    """
    Mean CA B-factor per chain from a BC2 mmCIF, i.e. mean per-residue pLDDT on
    the 0-100 scale ("The structure's B-factor column stores per-residue pLDDT on
    a 0-100 scale", BC2 outputs.md).

    This is the backfill for `pLDDT` being absent from !_Ranked.csv whenever no
    filter asked for it. Reading it off the coordinates works for every preset.
    Returns {chain_id: mean_plddt_0_100}; {} if the file cannot be parsed.
    """
    try:
        lines = Path(cif_path).read_text().splitlines()
    except Exception:
        return {}

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
    ci_b     = col_map.get("_atom_site.B_iso_or_equiv", -1)
    if min(ci_atom, ci_chain, ci_b) < 0:
        return {}
    max_ci = max(ci_atom, ci_chain, ci_b)

    totals, counts = {}, {}
    for line in lines[data_start:]:
        s = line.strip()
        if not s or s.startswith(("#", "_", "loop_", "data_")):
            break
        parts = s.split()
        if len(parts) <= max_ci or parts[ci_atom] != "CA":
            continue
        try:
            b = float(parts[ci_b])
        except ValueError:
            continue
        c = parts[ci_chain]
        totals[c] = totals.get(c, 0.0) + b
        counts[c] = counts.get(c, 0) + 1

    return {c: totals[c] / counts[c] for c in counts if counts[c]}


def _bc2_cell(raw):
    """
    BC2 writes one column per metric; a multi-target campaign packs one reading
    per target into the cell, semicolon-separated in `targets` order. This
    pipeline drives a single target, so take the first reading and ignore the
    rest rather than letting float() choke on "0.82;0.31".
    """
    if raw is None:
        return float("nan")
    first = str(raw).split(";")[0].strip()
    return _safe_float(first)


def _load_bindcraft2_metrics(design_path):
    """
    Load BindCraft2 3_Ranked/!_Ranked.csv and return dict:
        design_name → {bc2_key: value, ...}

    Returns {} when the campaign accepted nothing (the folder only appears once
    its first result is written, so a missing file is "no accepted designs",
    not an error).
    """
    ranked_csv = Path(design_path) / "3_Ranked" / "!_Ranked.csv"
    if not ranked_csv.exists():
        return {}

    result = {}
    try:
        with open(ranked_csv) as f:
            for row in csv.DictReader(f):
                name = (row.get("design") or "").strip()
                if not name:
                    continue
                entry = {}
                for col in BINDCRAFT2_NUMERIC_COLS:
                    if col in row:
                        entry["bc2_" + col.lower()] = _bc2_cell(row[col])
                seq = (row.get("Binder_Sequence") or "").strip()
                if seq:
                    entry["bc2_binder_sequence"] = seq
                result[name] = entry
    except Exception as e:
        log(f"  Warning: could not load BindCraft2 metrics CSV {ranked_csv}: {e}")

    return result


def _bc2_pick_binder_chain(chain_info, binder_seq, target_chain):
    """
    Resolve which chain of a BC2 complex is the binder.

    v1 could hard-code chain B; BC2 names chains from the prepared assembly and a
    multi-chain target pushes the binder past B, so match on the sequence the CSV
    reported. `/` separates binder chains in Binder_Sequence — the first segment
    is the chain we score.

    Falls back to "the longest chain that is not the target", which is wrong only
    if a binder outgrows a single-chain target (cf. the BoltzGen
    binder-is-shortest-chain bug).
    """
    if not chain_info:
        return None, ""

    wanted = (binder_seq or "").split("/")[0].strip()
    if wanted:
        for cid, info in chain_info.items():
            if info.get("sequence", "") == wanted:
                return cid, wanted
        # Prediction can trim or pad termini; fall back to containment.
        for cid, info in chain_info.items():
            seq = info.get("sequence", "")
            if seq and (seq in wanted or wanted in seq):
                return cid, seq

    candidates = {c: i for c, i in chain_info.items() if c != target_chain}
    if not candidates:
        return None, ""
    cid = max(candidates, key=lambda c: candidates[c].get("length", 0))
    return cid, candidates[cid].get("sequence", "")


def parse_bindcraft2_outputs(design_path, target_chain="A", limit=None):
    """
    Read a finished (or partly finished) BindCraft2 campaign folder into design
    dicts. Shared by run_bindcraft2() and rerank_binders.collect_bindcraft2() so
    the two cannot drift — v1 kept two near-identical copies of this parser.

    Accepted designs live in 3_Ranked/ as <design>_seq<n>.cif, best-first by
    i_pDAE in !_Ranked.csv.
    """
    design_path = Path(design_path)
    ranked_dir  = design_path / "3_Ranked"
    if not ranked_dir.is_dir():
        return []

    metrics = _load_bindcraft2_metrics(design_path)

    # Monomer re-predictions sit beside the complexes; they are the same design,
    # not another one. save_binder_monomers is off in campaigns this pipeline
    # writes, but a resumed/hand-run campaign may have them.
    cifs = sorted(c for c in ranked_dir.glob("*.cif")
                  if not c.stem.endswith("_monomer"))

    # Order by FILENAME, not by rank. BC2 rewrites !_Ranked.csv fully re-sorted by
    # i_pDAE after every acceptance (campaign_output.py: write_ranked_designs), so
    # a design's row index moves as the campaign grows. design_id is positional,
    # and rerank / partial-collection re-parse the same folder later — rank-ordered
    # ids would silently re-point bindcraft2_0003 at a different design and
    # invalidate any cached Boltz-2/Rosetta result keyed to it. Filenames are
    # stable; the ranking is still recoverable from bc2_i_pdae.
    cifs.sort(key=lambda c: c.stem)

    designs = []
    for cif in cifs:
        if limit is not None and len(designs) >= limit:
            break
        matched_key = (cif.stem if cif.stem in metrics
                       else next((k for k in metrics if k in cif.stem), None))
        m = dict(metrics.get(matched_key, {})) if matched_key else {}

        try:
            chain_info = extract_cif_chain_info(cif)
        except Exception as e:
            log(f"  BindCraft2: could not parse {cif.name}: {e}")
            continue

        cid, seq = _bc2_pick_binder_chain(chain_info, m.get("bc2_binder_sequence", ""),
                                          target_chain)
        if not seq:
            log(f"  BindCraft2: no binder chain resolved in {cif.name}, skipping")
            continue
        m.pop("bc2_binder_sequence", None)

        # Backfill binder pLDDT from the coordinates when the CSV has no pLDDT
        # column (BC2 only measures what a filter or the ranking asks for, so
        # `no_filters` campaigns omit it). Stored 0-1 to match BC2's own table
        # convention, which is what the scoring fallback expects.
        if m.get("bc2_plddt", float("nan")) != m.get("bc2_plddt", float("nan")):
            m.pop("bc2_plddt", None)
        if "bc2_plddt" not in m:
            per_chain = _bc2_plddt_by_chain(cif)
            if cid in per_chain:
                m["bc2_plddt"] = per_chain[cid] / 100.0
                m["bc2_plddt_source"] = "bfactor"
            if per_chain:
                for other, val in per_chain.items():
                    if other != cid:
                        m.setdefault("bc2_target_plddt", val / 100.0)
                        break

        d = {
            "design_id":         f"bindcraft2_{len(designs):04d}",
            "binder_sequence":   seq,
            "binder_pdb":        str(cif),   # historical key name; CIF is accepted
            "binder_chain":      cid,
            "native_score":      m.get("bc2_i_ptm", 0.0),
            "native_score_name": "bindcraft2_iptm",
        }
        d.update(m)
        designs.append(d)

    return designs


def _bc2_vram_note(total_n):
    """
    BC2 documents its peak as 2.0 * (3.4 GB + 38 kB * N**2) for an N-residue
    complex. Report rather than silently reshape the run: BC2's own
    subbatch_size=auto is what splits the calculation, and quietly overriding it
    would make a slow run look like a configuration the user chose.
    """
    est_gb = 2.0 * (3.4 + 38e-6 * float(total_n) ** 2)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        totals = [float(x) for x in out.stdout.split()]
        # nvidia-smi enumerates every physical card regardless of
        # CUDA_VISIBLE_DEVICES, so on a pinned multi-GPU host the bare min() can
        # report a card this run will never touch.
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if vis:
            idx = [int(i) for i in vis.split(",") if i.strip().isdigit()]
            totals = [totals[i] for i in idx if i < len(totals)] or totals
        have_gb = min(totals) / 1024.0
    except Exception:
        have_gb = None

    log(f"BindCraft2: total_N={total_n}, estimated peak ~{est_gb:.1f} GB")
    if have_gb and est_gb > have_gb:
        n_fit = int((max(have_gb / 2.0 - 3.4, 0.0) / 38e-6) ** 0.5)
        log(f"  WARNING: ~{have_gb:.1f} GB visible — this complex may not fit. "
            f"This card tops out near N={n_fit}; BC2 will try subbatching, "
            f"expect it to be slow or to OOM.")


def run_bindcraft2(target_path, chain_id, site_resnums, length_min, length_max,
                   n_designs, out_dir, dry_run=False, filters_preset="no_filters",
                   modality="binder", max_trajectories=None, ss_bias="balanced",
                   extra_settings=None, target_len=None):
    """
    BindCraft2 binder design (JAX/AF2, PacesaLab rewrite).

    Writes one campaign JSON, runs `bindcraft design`, then reads
    3_Ranked/!_Ranked.csv + the accepted mmCIFs into bc2_* design records.

    Unlike v1 this accepts PDB, mmCIF or FASTA targets.

    Returns list of design dicts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    design_path   = out_dir / "bindcraft2_output"
    config_path   = out_dir / "campaign.json"

    if modality not in BINDCRAFT2_MODALITIES:
        log(f"BindCraft2: WARNING — unrecognised modality '{modality}'; "
            f"passing through to BC2 anyway (known: {', '.join(BINDCRAFT2_MODALITIES)})")

    # Hotspots are chain-prefixed in BC2: "A54,A56,B12-16". An unprefixed number
    # addresses the first selected chain, which is only the same thing when the
    # target is single-chain — so always prefix.
    hotspots = ",".join(f"{chain_id}{r}" for r in site_resnums) if site_resnums else ""

    campaign = {
        "campaign_name":           "binder",
        "project_folder":          str(design_path),
        "modality":                modality,
        "targets": [{
            "name":        "target",
            "target_path": str(Path(target_path).resolve()),
            "chains":      chain_id,
        }],
        "binder_lengths":          [length_min, length_max],
        "number_of_final_designs": n_designs,
        # 3_Ranked/ is the progress signal; monomer re-predictions would land
        # there too and inflate the count.
        "save_binder_monomers":    False,
    }
    if hotspots:
        campaign["targets"][0]["hotspots"] = hotspots

    # BC2 leaves max_trajectories unset by default, which means it keeps designing
    # until it reaches number_of_final_designs — with no upper bound on attempts.
    # An unattended pipeline run needs a ceiling.
    campaign["max_trajectories"] = int(max_trajectories) if max_trajectories else n_designs * 50

    preset = BINDCRAFT2_FILTER_PRESETS.get(filters_preset)
    if preset is None:
        log(f"BindCraft2: WARNING — unknown filter preset '{filters_preset}', using no_filters")
        preset, filters_preset = BINDCRAFT2_FILTER_PRESETS["no_filters"], "no_filters"
    if preset.get("filters"):
        campaign["filters"] = {k: dict(v) for k, v in preset["filters"].items()}
    # Stage gates are top-level scalars, not part of the `filters` block.
    campaign.update(preset.get("settings", {}))
    n_open = sum(1 for k, v in preset.get("settings", {}).items() if v is None)
    log(f"BindCraft2: filters = {filters_preset}, modality = {modality}"
        + (f", {n_open} stage gate(s) disabled" if n_open else ""))

    # SS bias. BC2 inverts v1's sign convention: weights_binder_helicity is
    # negative to FAVOUR helix (default -0.3), where v1's weights_helicity was
    # positive to favour it. Flipping the sign here is not cosmetic — passing the
    # v1 number through unchanged would bias every run the wrong way.
    if ss_bias != "balanced":
        v1_helicity = SS_BIAS_PARAMS[ss_bias]["bc_helicity"]
        campaign["weights_binder_helicity"] = -v1_helicity
        log(f"BindCraft2: ss_bias={ss_bias} → weights_binder_helicity="
            f"{-v1_helicity} (BC2 sign: negative favours helix)")
        if ss_bias == "beta":
            campaign["max_helix_fraction_final"] = SS_BIAS_PARAMS["beta"]["penalty_helix_above"]

    if extra_settings:
        campaign.update(extra_settings)

    config_path.write_text(json.dumps(campaign, indent=2) + "\n")
    log(f"BindCraft2: campaign.json written → {config_path}")

    if target_len:
        _bc2_vram_note(int(target_len) + int(length_max))

    cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("bindcraft2"),
        "bindcraft", "design", str(config_path),
    ]
    log(f"BindCraft2: running until {n_designs} designs accepted "
        f"(max {campaign['max_trajectories']} trajectories)...")
    run_cmd(cmd, timeout=None, cwd=BINDCRAFT2_DIR, dry_run=dry_run,
            extra_env={"BINDCRAFT_WEIGHTS": str(BINDCRAFT2_WEIGHTS),
                       "PYTHONNOUSERSITE": "1"},
            progress_dir=str(design_path / "3_Ranked"),
            progress_pattern="*.cif",
            progress_total=n_designs, progress_label="bindcraft2",
            skip_file=str(Path(out_dir).parent / ".skip_bindcraft2"))

    if dry_run:
        designs = [{"design_id": "bindcraft2_0000", "binder_sequence": "MOCK" * 25,
                    "binder_pdb": str(out_dir / "mock.cif"),
                    "native_score": 0.60, "native_score_name": "bindcraft2_iptm"}]
    else:
        designs = parse_bindcraft2_outputs(design_path, target_chain=chain_id)
        log(f"  BindCraft2: loaded metrics for {len(designs)} accepted designs")

    log(f"BindCraft2: {len(designs)} designs collected")
    return designs


# ── PXDesign ──────────────────────────────────────────────────────────────────

def _fix_cif_entity_ids(cif_path):
    """Fix gemmi-generated CIF to use numeric entity IDs (Protenix requirement).

    gemmi often writes entity_id as chain letter ('A') and label_asym_id
    as chain+'xp' ('Axp').  Protenix/PXDesign expects numeric entity IDs
    (1, 2, …) and simple label_asym_id matching the chain name.

    UNUSED — nothing calls this. It indexes _atom_site by POSITION
    (`parts[6]`/`parts[7]`), which is safe ONLY because its input is always a
    gemmi-written CIF with a known fixed column order. Do not copy the pattern:
    mmCIF column order is producer-specific, and exactly this assumption made
    four other readers in this module silently return garbage for BoltzGen,
    PXDesign and BindCraft2 output. Use `_iter_struct_atoms()` instead.
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
    # PXDesign's config takes a single int, not a range, so a --length span is
    # collapsed to its midpoint. That is a tool limitation, not a pipeline choice —
    # but it happened silently, so a user asking for 60-80 got only 70-mers from
    # this tool and lost the length diversity they thought they had.
    binder_length = (length_min + length_max) // 2
    if length_min != length_max:
        log(f"  PXDesign: --length {length_min}-{length_max} collapsed to "
            f"binder_length={binder_length} (PXDesign takes a single value, not a "
            f"range). Other tools still sample the full range.")

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
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("pxdesign"),
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
        # PXDesign imports protenix → deepspeed, whose op-builder compat check at
        # import time requires CUDA_HOME with a working nvcc. conda cuda-nvcc is
        # installed into the pxdesign env, so point CUDA_HOME at the env prefix.
        "CUDA_HOME": cfg.conda_env_prefix("pxdesign") or cfg.conda_env("pxdesign"),
    }
    run_cmd(cmd, timeout=86400, extra_env=px_env, dry_run=dry_run,
            progress_dir=str(out_dir / "pxdesign_output"),
            progress_pattern="**/*.cif",
            progress_total=n_designs, progress_label="pxdesign",
            skip_file=str(Path(out_dir).parent / ".skip_pxdesign"))

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
                    # Pick the binder by SEQUENCE, not by chain letter. PXDesign
                    # writes chains "A0" (target) and "B0" (binder) while chain_id
                    # here is "A" — so `cid != chain_id` was True for A0 and this
                    # took the TARGET as the binder and broke immediately. Every
                    # design recovered through this path was the target sequence.
                    _tseq = pdb_chain_seq(target_path, chain_id)
                    _picked = pick_binder_chain_by_identity(ci, _tseq) if _tseq else None
                    if _picked:
                        seq = _picked.get("sequence", "")
                    else:
                        # No target sequence to compare against: fall back to the
                        # chain whose id does not share the target's leading letter.
                        for cid, info in ci.items():
                            if str(cid)[:1].upper() != str(chain_id)[:1].upper():
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
    proteina_python = os.environ.get("PROTEINA_PYTHON", os.path.join(cfg.conda_env_prefix("proteina") or cfg.conda_env("proteina"), "bin", "python"))
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
    run_cmd(proteina_cmd, timeout=None, extra_env=proteina_env, dry_run=dry_run,
            progress_dir=str(bb_dir), progress_pattern="design_*.pdb",
            progress_total=n_designs, progress_label="proteina",
            skip_file=str(Path(out_dir).parent / ".skip_proteina"))

    backbones = sorted(bb_dir.glob("design_*.pdb"))
    if not backbones and not dry_run:
        raise FileNotFoundError(f"No backbone PDBs in {bb_dir}")

    # Proteina outputs CA-only PDBs. Use Proteina's bundled ProteinMPNN CA model
    # (not LigandMPNN which requires full N/CA/C/O backbone).
    proteina_mpnn_script = f"{PROTEINA_DIR}/ProteinMPNN/protein_mpnn_run.py"
    for bb_pdb in backbones:
        mpnn_cmd = [
            "conda", "run", "--no-capture-output", *cfg.conda_run_args("mpnn"),
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
    # Complexa's AtomSelectionStack.from_contig() expands a range into ONE
    # AtomSelection per residue and raises "No atoms found for selection: A/*/N"
    # if any single one is absent. A crystal target with an unresolved loop
    # therefore kills the generate step instantly: 6X18 GLP1R chain A spans
    # 29-423 but is missing 130-135 and 339-343, so "A29-423" died on residue
    # 130. Emit the contiguous runs that are actually PRESENT instead; the
    # contig grammar is comma-separated ("A1-2, B3-10").
    if residues:
        _nums = sorted({int(r) for r in residues})
        _runs, _start, _prev = [], _nums[0], _nums[0]
        for _n in _nums[1:]:
            if _n == _prev + 1:
                _prev = _n
            else:
                _runs.append((_start, _prev)); _start = _prev = _n
        _runs.append((_start, _prev))
        target_input = ", ".join(f"{chain_id}{a}-{b}" for a, b in _runs)
        if len(_runs) > 1:
            log(f"  Proteina Complexa: target has {len(_runs) - 1} unresolved gap(s) — "
                f"selecting {len(_runs)} present ranges ({target_input})")
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


# Keys we accept from Complexa's own environment files. Everything else is either
# irrelevant to this wrapper or deliberately overridden by config below.
_COMPLEXA_ENV_KEYS = (
    "AF2_DIR", "ESM_DIR", "COMMUNITY_MODELS_PATH",
    "LOCAL_CHECKPOINT_PATH", "LOCAL_CACHE_DIR", "LOGURU_LEVEL",
    "FOLDSEEK_EXEC", "SC_EXEC", "DSSP_EXEC", "MMSEQS_EXEC", "TMOL_PATH",
)


def _load_complexa_dotenv(env_path):
    """Parse Proteina Complexa's .env into a resolved {KEY: VALUE} dict.

    Fallback for installs where `complexa init uv` has not been run, so env.sh
    does not exist yet and cannot be sourced. Handles `KEY=VALUE` lines and
    ${VAR} interpolation (the .env defines ESM_DIR=${COMMUNITY_MODELS_PATH}/ckpts/ESM2
    etc.), then applies the same UV_* precedence env.sh would have applied —
    without it FOLDSEEK/SC/MMSEQS/DSSP/TMOL resolve to the docker paths rather
    than the uv venv. Returns {} if the file is missing so callers fall back to
    their built-in defaults.
    """
    env_path = Path(env_path)
    if not env_path.exists():
        return {}
    raw = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        raw[key.strip()] = val.strip().strip('"').strip("'")
    # Iteratively expand ${VAR} references against already-parsed keys.
    resolved = dict(raw)
    for _ in range(10):
        changed = False
        for k, v in list(resolved.items()):
            new_v = re.sub(r"\$\{(\w+)\}",
                           lambda m: resolved.get(m.group(1), m.group(0)), v)
            if new_v != v:
                resolved[k] = new_v
                changed = True
        if not changed:
            break
    # Mirror env.sh: the uv-runtime overrides win over the generic ones.
    for k in ("FOLDSEEK_EXEC", "SC_EXEC", "MMSEQS_EXEC", "DSSP_EXEC", "TMOL_PATH"):
        uv_val = resolved.get(f"UV_{k}")
        if uv_val:
            resolved[k] = uv_val
    return {k: v for k, v in resolved.items() if k in _COMPLEXA_ENV_KEYS and v}


def _complexa_run_name(out_dir):
    """Unique run tag for Proteina Complexa's SHARED inference directory.

    Complexa writes every run into one repo-level `inference/` tree, and the tag
    used to be just `{out_dir.parent.name}_{out_dir.name}`. Two campaigns with
    the same directory layout — e.g. /a/runs/output and /b/runs/output — collide
    on `output_proteina_complexa`, and the collector globs and MERGES every
    matching directory, so one campaign's designs land in another's panel. A
    stale directory from a previous run of the same path does the same thing.
    Appending a hash of the ABSOLUTE path makes the tag unique per output dir.
    """
    import hashlib
    from pathlib import Path as _P
    out_dir = _P(out_dir)
    h = hashlib.sha256(str(out_dir.resolve()).encode()).hexdigest()[:8]
    return f"{out_dir.parent.name}_{out_dir.name}_{h}"


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

    # Inject our target into Complexa's config (Hydra requires this location).
    # NOT during a dry run: this writes into the SHARED software tree, so
    # `--dry_run` — which is supposed to touch nothing — was permanently editing
    # a repo-level config that concurrent real runs read.
    complexa_targets = Path(PROTEINA_COMPLEXA_DIR) / "configs" / "targets" / "targets_dict.yaml"
    if dry_run:
        log(f"  [DRY RUN] would inject target into {complexa_targets} (skipped)")
    else:
        _inject_complexa_target(targets_yaml, complexa_targets)

    # Calculate nsamples: best-of-n with replicas=2 means each "sample"
    # produces ~2 candidates after search; nsamples=n_designs//2 gives ~n_designs outputs
    nsamples = max(2, n_designs // 2)

    # Size-aware batch_size: pair representation memory ~ N². 24 GB GPUs OOM at
    # batch_size=4 once target+binder pushes total N past ~340 (e.g. 382-res target
    # + 90-res binder = 472). Drop to 1 for large inputs; keep 4 for normal range.
    target_len = len(get_chain_info(target_path).get(chain_id, {}).get("residues", []))
    total_n = target_len + length_max
    # RTX 4000 Ada 20 GB (this host, bz-wg3-plm06): Complexa weights alone take ~16 GB,
    # so batch_size=4 OOMs in pair-rep allocation even at N≈296 (see memory
    # feedback_complexa_batch_size.md). Force 1 for any target size on this GPU.
    # On A40 40 GB, restore the `1 if total_n > 340 else 4` rule.
    pc_batch_size = 1
    # VRAM guard: the best-of-n search co-runs the PyTorch flow model AND the JAX/AF2
    # reward model on the same GPU. On <~23 GiB cards (e.g. RTX 4000 Ada 20 GB) the flow
    # model at batch_size>1 leaves no room for JAX to init a GPU stream — AF2 reward fails
    # every batch, then torch OOMs. Force batch_size=1 on small GPUs regardless of total_n.
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _vram_gib = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if _vram_gib < 23.0:
                pc_batch_size = 1
    except Exception:
        pass

    complexa_bin = f"{PROTEINA_COMPLEXA_VENV}/bin/complexa"

    # Run full design pipeline (generate → filter → evaluate → analyze).
    # Requires SC, DSSP, ESM2, foldseek, mmseqs installed (see CLAUDE.md).
    cmd = [
        complexa_bin, "design",
        f"{PROTEINA_COMPLEXA_DIR}/configs/search_binder_local_pipeline.yaml",
        f"++run_name={_complexa_run_name(out_dir)}",
        f"++generation.task_name=binder_target",
        f"++generation.dataloader.dataset.nres.nsamples={nsamples}",
        f"++generation.dataloader.dataset.nres.low={length_min}",
        f"++generation.dataloader.dataset.nres.high={length_max}",
        f"++generation.dataloader.batch_size={pc_batch_size}",
        f"++generation.search.algorithm=best-of-n",
        f"++generation.search.best_of_n.replicas=2",
        f"++gen_njobs=1",
        f"++eval_njobs=1",
        f"++seed=42",
    ]

    log(f"Proteina Complexa: running {n_designs} designs (nsamples={nsamples}, "
        f"target_len={target_len}, total_N={total_n}, batch_size={pc_batch_size})...")

    # Build environment: inherit the machine's resolved paths from env.sh (the single
    # source of truth, produced by `complexa init uv`; it sources .env and resolves all
    # ${...} interpolation), then overlay the config-derived overrides. env.sh carries
    # machine-specific paths the hardcoded defaults below get wrong — notably AF2_DIR,
    # which points at the BindCraft params dir (where the AF2 multimer weights live), not
    # the empty community_models/ckpts/AF2. Without it the AF2 reward model raises
    # "No model parameters found" and every Complexa design dies instantly.
    _envsh = Path(PROTEINA_COMPLEXA_DIR) / "env.sh"
    _envsh_vars = {}
    if _envsh.exists():
        try:
            _out = subprocess.run(
                ["bash", "-c", f"set -a; source '{_envsh}' >/dev/null 2>&1; env -0"],
                capture_output=True, text=True, cwd=PROTEINA_COMPLEXA_DIR, timeout=60,
            ).stdout
            for _kv in _out.split("\0"):
                if "=" in _kv:
                    _k, _v = _kv.split("=", 1)
                    # Skip empty exports: env.sh unconditionally exports TMOL_PATH even
                    # though .env never defines it, and an empty value here would override
                    # the correct venv site-packages default below.
                    if _k in _COMPLEXA_ENV_KEYS and _v:
                        _envsh_vars[_k] = _v
        except Exception as _e:
            log(f"  Proteina Complexa: could not source env.sh ({_e}); falling back to .env")
    # Fallback tier: no env.sh (install predates `complexa init uv`) or sourcing it
    # failed — parse .env directly. Same keys, same UV_* precedence.
    if not _envsh_vars:
        _envsh_vars = _load_complexa_dotenv(Path(PROTEINA_COMPLEXA_DIR) / ".env")
        if _envsh_vars:
            log(f"  Proteina Complexa: resolved {len(_envsh_vars)} paths from .env "
                f"(env.sh absent — run `complexa init uv` for the canonical env)")

    extra_env = {
        # Hardcoded fallbacks (correct on a standard layout; overridden by env.sh below).
        "ESM_DIR": f"{PROTEINA_COMPLEXA_DIR}/community_models/ckpts/ESM2",
        "AF2_DIR": f"{PROTEINA_COMPLEXA_DIR}/community_models/ckpts/AF2",
        "FOLDSEEK_EXEC": f"{PROTEINA_COMPLEXA_VENV}/bin/foldseek",
        "SC_EXEC": f"{PROTEINA_COMPLEXA_DIR}/env/docker/internal/sc",
        "DSSP_EXEC": f"{PROTEINA_COMPLEXA_DIR}/env/docker/internal/dssp",
        "MMSEQS_EXEC": f"{PROTEINA_COMPLEXA_VENV}/bin/mmseqs",
        "TMOL_PATH": f"{PROTEINA_COMPLEXA_VENV}/lib/python3.12/site-packages/tmol",
        # env.sh-resolved machine paths win over the fallbacks above.
        **_envsh_vars,
        # Config-derived overrides (machine-portable; always take precedence).
        "COMPLEXA_INIT": "uv",
        "PYTHONNOUSERSITE": "1",
        "LOCAL_CODE_PATH": PROTEINA_COMPLEXA_DIR,
        "LOCAL_DATA_PATH": _WEIGHTS_DIR,
        "DATA_PATH": _WEIGHTS_DIR,
        "CKPT_PATH": f"{PROTEINA_COMPLEXA_DIR}/ckpts",
        "USE_V2_COMPLEXA_ARCH": "False",
    }

    # Progress: Complexa writes PDBs to inference/ directory
    run_name = _complexa_run_name(out_dir)
    pc_inference_dir = str(Path(PROTEINA_COMPLEXA_DIR) / "inference")
    try:
        run_cmd(cmd, timeout=None, cwd=PROTEINA_COMPLEXA_DIR, extra_env=extra_env, dry_run=dry_run,
                progress_dir=pc_inference_dir, progress_pattern=f"*{run_name}*/job_*/*.pdb",
                progress_total=n_designs, progress_label="proteina_complexa",
                skip_file=str(Path(out_dir).parent / ".skip_proteina_complexa"))
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
        run_name = _complexa_run_name(out_dir)
        run_dirs = sorted(inference_base.glob(f"*{run_name}"))
        if not run_dirs:
            # Fallback: try with binder_target prefix
            run_dirs = sorted(inference_base.glob(f"*binder_target*{run_name}*"))
        if not run_dirs:
            log(f"  WARNING: no inference directory found matching run_name={run_name}",
                err=True)
            run_dirs = []
        elif len(run_dirs) > 1:
            # Never merge several matches — that is how designs from one campaign
            # end up in another's panel. Take the newest and say what was ignored.
            run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
            log(f"  Proteina Complexa: {len(run_dirs)} inference dirs match "
                f"{run_name}; using the newest ({run_dirs[0].name}) and IGNORING "
                f"{', '.join(p.name for p in run_dirs[1:])}", err=True)
            run_dirs = run_dirs[:1]

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

        # Copy the structures INTO out_dir. Complexa writes into the shared
        # software tree (PROTEINA_COMPLEXA_DIR/inference/...), and the design
        # dicts pointed straight at it — so a run's own results lived outside
        # its output directory, survived only until someone cleaned that tree,
        # and could not be archived or moved with the run.
        # The metrics CSVs are keyed on the INFERENCE job-directory name
        # (`job_0_n_226_id_0_bon_orig0_r0`). Capture that key here, before the
        # copy flattens every structure into one `structures/` folder — after
        # the copy `pdb.parent.name` is "structures" and the join silently
        # matched nothing, blanking every pc_* column and dropping all 20
        # designs out of the ranking (runs/pdia6_validation_002).
        _sample_id_for = {
            _p: (_p.parent.name if _p.parent.name.startswith("job_") else _p.stem)
            for _p in sample_pdbs
        }
        _local = Path(out_dir) / "structures"
        try:
            _local.mkdir(parents=True, exist_ok=True)
            _copied = []
            for _p in sample_pdbs:
                _dst = _local / f"{_p.parent.name}__{_p.name}"
                if not _dst.exists():
                    shutil.copy2(_p, _dst)
                _sample_id_for[_dst] = _sample_id_for[_p]
                _copied.append(_dst)
            if _copied:
                sample_pdbs = _copied
                log(f"  Proteina Complexa: copied {len(_copied)} structures into "
                    f"{_local} (they were only in the shared software tree)")
        except Exception as _e:
            log(f"  Proteina Complexa: could not copy structures into out_dir "
                f"({_e}); design paths still point into the shared tree.", err=True)

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
        _n_unjoined = [0]

        for i, pdb in enumerate(sample_pdbs):
            # Extract binder sequence from PDB
            chain_info_pdb = extract_pdb_chain_info(pdb)
            # In Complexa output, binder is typically the shorter chain
            # or the chain that is not the target chain
            # Pick by SEQUENCE IDENTITY to the known target, not by chain letter.
            # `cid != chain_id` returns the TARGET whenever the target's chain in
            # the output is not literally --site's letter — e.g. --site B:... on a
            # 2-chain output picks chain A, the target. Falling back to "shortest"
            # has the twin failure when the binder is longer than the target.
            binder_info = None
            _tseq_pc = pdb_chain_seq(target_path, chain_id) if target_path else ""
            if _tseq_pc:
                binder_info = pick_binder_chain_by_identity(chain_info_pdb, _tseq_pc)
            if not binder_info:
                for cid, info in sorted(chain_info_pdb.items()):
                    if str(cid)[:1].upper() != str(chain_id)[:1].upper():
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
            sample_name = _sample_id_for.get(
                pdb, pdb.parent.name if pdb.parent.name.startswith("job_") else pdb.stem)
            metrics = eval_metrics.get(sample_name, {})
            if eval_metrics and not metrics:
                _n_unjoined[0] += 1

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

    if not dry_run and _n_unjoined[0]:
        # Loud: this is the failure mode that blanked every pc_* column.
        log(f"  Proteina Complexa: WARNING — {_n_unjoined[0]}/{len(sample_pdbs)} "
            f"structures did not match any row in the metrics CSVs "
            f"({len(eval_metrics)} samples loaded). Their pc_* metrics are blank "
            f"and they cannot be ranked on native iPTM.", err=True)
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


def run_rfdiffusion3(target_path, chain_id, site_resnums, length_min, length_max,
                     n_designs, out_dir, dry_run=False, **kwargs):
    """
    RFdiffusion3 (Foundry) all-atom binder design.

    Generates full-atom binder structures with predicted sequences via the
    rfd3 diffusion model.  No separate inverse folding step is needed.

    Each input task produces diffusion_batch_size (4) models with different
    seeds, providing structural diversity.  n_designs=20 → 80 candidates.

    Returns list of design dicts with design_id, binder_sequence, backbone_pdb.
    """
    import gzip

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_subdir = out_dir / "rfd3_output"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # RFD3 uses original PDB residue numbering in contig and hotspots.
    # Contig format for binder design: "{binder_len},/0,{chain}{first}-{last}"
    #   - First segment: new binder chain (diffused)
    #   - /0: chain break
    #   - Second segment: fixed target chain (original PDB numbering)
    chain_info = get_chain_info(target_path)
    tchain = chain_info.get(chain_id)
    if not tchain:
        raise ValueError(f"Chain {chain_id} not found in {target_path}")

    chain_len = tchain["length"]
    residues = tchain.get("residues", [])

    # RFdiffusion3 requires contiguous residue numbering. Auto-renumber if gaps.
    rf3_target = str(target_path)
    rf3_site = list(site_resnums)
    if not dry_run:
        renum_path, old_to_new = _renumber_pdb_contiguous(
            str(target_path), chain_id, out_dir)
        if renum_path != str(target_path):
            rf3_target = renum_path
            rf3_site = [old_to_new[r] for r in site_resnums if r in old_to_new]
            residues = [old_to_new[r] for r in residues if r in old_to_new]

    first_resnum = residues[0] if residues else 1
    last_resnum = residues[-1] if residues else chain_len

    hotspot_str = ",".join(f"{chain_id}{r}" for r in rf3_site)

    # Contig: binder (variable length), chain break, fixed target
    contig = f"{length_min}-{length_max},/0,{chain_id}{first_resnum}-{last_resnum}"
    # Total length = binder + target
    total_len_min = length_min + chain_len
    total_len_max = length_max + chain_len

    # Build input JSON — one task per design for progress tracking
    input_json = {}
    for i in range(n_designs):
        input_json[f"design_{i:04d}"] = {
            "input": str(Path(rf3_target).resolve()),
            "contig": contig,
            "length": f"{total_len_min}-{total_len_max}",
            "select_hotspots": hotspot_str,
            "plddt_enhanced": True,
            "infer_ori_strategy": "hotspots",
            "redesign_motif_sidechains": False,
        }

    json_path = out_dir / "rfd3_input.json"
    with open(json_path, "w") as f:
        json.dump(input_json, f, indent=2)
    log(f"RFdiffusion3: input JSON written → {json_path}")

    # Size-aware batch_size: diffusion pair rep scales ~N². 24 GB GPUs OOM at
    # batch_size=4 once target+binder pushes total N past ~340. Drop to 1 for
    # large inputs; keep 4 for normal range. Goal: avoid OOM, accept slowdown.
    total_n = chain_len + length_max
    rfd3_batch_size = 1 if total_n > 340 else 4

    cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("rfdiffusion3"),
        "rfd3", "design",
        f"inputs={json_path}",
        f"out_dir={output_subdir}",
        f"diffusion_batch_size={rfd3_batch_size}",
        "inference_sampler.num_timesteps=200",
        "inference_sampler.step_scale=1.5",
    ]
    rfd3_env = {
        "PYTHONNOUSERSITE": "1",
        "FOUNDRY_CHECKPOINT_DIRS": RFD3_WEIGHTS,
    }
    # Each input task produces diffusion_batch_size models with different seeds
    n_total_models = n_designs * rfd3_batch_size
    log(f"RFdiffusion3: generating {n_designs} designs × {rfd3_batch_size} models "
        f"= {n_total_models} candidates (target_len={chain_len}, total_N={total_n})...")
    run_cmd(cmd, timeout=None, dry_run=dry_run, extra_env=rfd3_env,
            progress_dir=str(output_subdir), progress_pattern="*.cif.gz",
            progress_total=n_total_models, progress_label="rfdiffusion3",
            skip_file=str(Path(out_dir).parent / ".skip_rfdiffusion3"))

    designs = []
    if not dry_run:
        import gemmi

        for cif_gz in sorted(output_subdir.glob("*.cif.gz")):
            try:
                with gzip.open(str(cif_gz), "rt") as f:
                    doc = gemmi.cif.read_string(f.read())
                st = gemmi.make_structure_from_block(doc.sole_block())
                model = st[0]
                chains = list(model)

                # RFD3 may output binder+target as separate chains or in one chain.
                # Use the metadata JSON to identify the binder region via diffused_index_map.
                meta_path = str(cif_gz).replace(".cif.gz", ".json")
                binder_seq = None

                if len(chains) >= 2:
                    # Multi-chain: identify the binder chain. Prefer the diffused_index_map metadata
                    # (binder = chain with fewest target-mapped residues) — length-agnostic, so it
                    # holds even when the binder is longer than the target. Fall back to shortest chain.
                    binder_ch = None
                    if Path(meta_path).exists():
                        try:
                            with open(meta_path) as mf:
                                # Keep the CHAIN LETTER. The regex captured only the
                                # number, so "A5" became bare 5 and matched residue 5
                                # of ANY chain — on a 2-chain output the binder's own
                                # residue 5 counted as target-mapped and the "least
                                # mapped chain" pick went degenerate.
                                _mapped = set()
                                for _o, _n in json.load(mf).get("diffused_index_map", {}).items():
                                    _m = re.match(r"([A-Za-z])(\d+)", _n)
                                    if _m:
                                        _mapped.add((_m.group(1), int(_m.group(2))))
                            binder_ch = min(chains, key=lambda ch: sum(
                                1 for r in ch if r.find_atom("CA", "\0")
                                and (ch.name, r.seqid.num) in _mapped))
                        except Exception:
                            binder_ch = None
                    if binder_ch is None:
                        chain_sizes = [(ch, sum(1 for r in ch if r.find_atom("CA", "\0"))) for ch in chains]
                        chain_sizes.sort(key=lambda x: x[1])
                        binder_ch = chain_sizes[0][0]
                    binder_seq = "".join(
                        gemmi.find_tabulated_residue(r.name).one_letter_code
                        for r in binder_ch if r.find_atom("CA", "\0")
                        and gemmi.find_tabulated_residue(r.name).one_letter_code != "?")
                elif len(chains) == 1 and Path(meta_path).exists():
                    # Single chain: use diffused_index_map to find which residues
                    # are the fixed target (mapped from original PDB) vs designed binder
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    mapped_resnums = set()
                    for orig, new in meta.get("diffused_index_map", {}).items():
                        # Parse "A5" -> ("A", 5). Keeping the chain letter matters:
                        # a bare 5 matches residue 5 of every chain.
                        m = re.match(r"([A-Za-z])(\d+)", new)
                        if m:
                            mapped_resnums.add((m.group(1), int(m.group(2))))
                    # Binder residues are those NOT in the diffused_index_map
                    ch = chains[0]
                    binder_residues = []
                    for r in ch:
                        if (r.find_atom("CA", "\0")
                                and (ch.name, r.seqid.num) not in mapped_resnums):
                            code = gemmi.find_tabulated_residue(r.name).one_letter_code
                            if code and code != "?":
                                binder_residues.append(code)
                    binder_seq = "".join(binder_residues)

                if not binder_seq or len(binder_seq) < 5:
                    continue

                design_id = f"rfdiffusion3_{len(designs):04d}"
                designs.append({
                    "design_id": design_id,
                    "binder_sequence": binder_seq,
                    "backbone_pdb": str(cif_gz),
                })
            except Exception as e:
                log(f"  RFdiffusion3: failed to parse {cif_gz.name}: {e}")
                continue

        log(f"RFdiffusion3: {len(designs)} designs collected")
    else:
        # Dry-run mock. Every other runner has one; rfdiffusion3 did not, so
        # `--dry_run --tools rfdiffusion3` returned [] and main() hit
        # "No designs collected" and exited 1 — a dry run that cannot succeed.
        designs = [{"design_id": f"rfdiffusion3_{i:04d}",
                    "binder_sequence": "MOCK" * 15,
                    "binder_pdb": str(Path(out_dir) / f"mock_{i:04d}.pdb")}
                   for i in range(min(n_designs, 3))]
        log(f"RFdiffusion3: [DRY RUN] {len(designs)} mock designs")

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
    2. Boltz-2 monomer CIF (or legacy ESMFold PDB)
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


def populate_interface_composition(designs, interface_dist=5.0, target_chain_id="A",
                                   target_residues=None):
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

        # Parse atoms with residue names, via the shared header-driven reader.
        # This block used to index mmCIF _atom_site by POSITION, which is right
        # only for Boltz-2's column order -- so all 21 interface_* columns came
        # out blank for every BoltzGen / PXDesign / BindCraft2 design.
        chain_atoms = {}      # chain_id -> [(resnum, x, y, z)]
        chain_resnames = {}   # chain_id -> {resnum: resname_3letter}
        chain_res_counts = {}
        try:
            for chain, resnum, resname, _name, x, y, z in _iter_struct_atoms(struct_file):
                chain_atoms.setdefault(chain, []).append((resnum, x, y, z))
                chain_resnames.setdefault(chain, {})[resnum] = resname
                chain_res_counts.setdefault(chain, set()).add(resnum)
        except (OSError, IOError):
            n_skipped += 1
            continue

        if len(chain_res_counts) < 2:
            n_skipped += 1
            continue

        # Identify target by the KNOWN site/target chain (length-agnostic — the binder can be longer
        # than the target). Fall back to longest-chain only if the hinted chain is absent.
        sorted_chains = sorted(chain_res_counts.items(), key=lambda x: len(x[1]))
        # Residue COUNT first, chain letter only as a fallback. The letter is wrong
        # for BoltzGen, whose emitted CIF has chain A = BINDER and chain B = target:
        # the hint "A" then selected the binder as the target and inverted all 21
        # interface_* columns. Same rule as _resolve_site_in_structure.
        target_chain = binder_chain = None
        if target_residues:
            n_t = len(target_residues)
            exact = [c for c, r in chain_res_counts.items() if len(r) == n_t]
            if len(exact) == 1:
                target_chain = exact[0]
        if target_chain is None and target_chain_id and target_chain_id in chain_res_counts:
            target_chain = target_chain_id
        if target_chain is not None:
            others = [c for c in chain_res_counts if c != target_chain]
            binder_chain = max(others, key=lambda c: len(chain_res_counts[c])) if others else sorted_chains[0][0]
        else:
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

        # BindCraft2: same idea, but BC2 reports these as fractions already
        # (Binder_Helix_Fraction, 0-1) — dividing by 100 here would silently
        # report every BC2 design as ~1% helix.
        if tool == "bindcraft2" and "bc2_binder_helix_fraction" in d:
            d["binder_helix_frac"] = _safe_float(d["bc2_binder_helix_fraction"])
            d["binder_sheet_frac"] = _safe_float(d["bc2_binder_betasheet_fraction"])
            d["binder_loop_frac"]  = _safe_float(d["bc2_binder_loop_fraction"])
            n_native += 1
            continue

        # All others: compute via DSSP
        if compute_binder_ss(d):
            n_computed += 1
        else:
            n_failed += 1

    log(f"  SS fractions: {n_native} native, {n_computed} DSSP-computed, {n_failed} no structure")


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_boltz_monomer(designs, val_dir, plddt_threshold=70.0, dry_run=False,
                           n_tag="", c_tag=""):
    """Stage 1: Boltz-2 monomer pre-filter (replaces ESMFold).

    Folds each binder sequence independently using Boltz-2 in batch mode
    and filters by mean pLDDT. Uses the same algorithm as the complex
    prediction for apples-to-apples refolding RMSD comparison.

    Adds to each design dict:
        esmfold_plddt — mean binder pLDDT (0-100), key kept for backward compat
        esmfold_pdb   — path to monomer CIF (key kept for refolding RMSD compat)

    Returns list of designs passing the threshold.
    """
    import numpy as np

    val_dir = Path(val_dir) / "boltz_monomer"
    val_dir.mkdir(parents=True, exist_ok=True)
    batch_yaml_dir = val_dir / "batch_inputs"
    batch_yaml_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(designs)
    log(f"Boltz-2 monomer validation: {n_total} designs, pLDDT threshold={plddt_threshold}")
    if n_tag or c_tag:
        log(f"  Terminal tags: N={n_tag or '(none)'}, C={c_tag or '(none)'}")

    if dry_run:
        for d in designs:
            d["esmfold_plddt"] = 80.0
        return designs

    # Check which designs already have results
    needs_prediction = []
    already_done = []
    for d in designs:
        design_id = d["design_id"]
        result_dir = val_dir / "boltz_results_batch_inputs" / "predictions" / design_id
        # Same trap as the complex cache: keyed on design_id alone, so changing
        # --n_tag/--c_tag reused a monomer folded from a DIFFERENT sequence, and
        # the refolding RMSD was then computed between structures of different
        # length (offset by len(n_tag)).
        _mkey = _boltz_input_key("", d.get("binder_sequence", ""), n_tag, c_tag,
                                 None, False, 0.0, None)
        _mcached = _read_boltz_key(val_dir, design_id, "monomer")
        _mhas = result_dir.exists() and bool(list(result_dir.glob("plddt_*.npz")))
        if _mhas and (_mcached == _mkey or not _mcached):
            if _mhas and not _mcached and (n_tag or c_tag):
                log(f"  {design_id}: cached monomer predates tag fingerprinting and "
                    f"cannot be checked against --n_tag/--c_tag.", err=True)
            already_done.append(d)
        else:
            if _mhas and _mcached and _mcached != _mkey:
                log(f"  {design_id}: cached monomer was folded from a DIFFERENT "
                    f"sequence (terminal tags changed) — re-predicting.", err=True)
            d["_boltz_monomer_key"] = _mkey
            needs_prediction.append(d)

    if already_done:
        log(f"  Boltz monomer: {len(already_done)} already predicted, "
            f"{len(needs_prediction)} new")

    # Write YAML files for designs that need prediction (single-chain, no target)
    for d in needs_prediction:
        _write_boltz_key(val_dir, d["design_id"], "monomer",
                         d.get("_boltz_monomer_key", ""))
        seq = (n_tag + d["binder_sequence"] + c_tag) if (n_tag or c_tag) else d["binder_sequence"]
        design_id = d["design_id"]
        yaml_path = batch_yaml_dir / f"{design_id}.yaml"
        yaml_content = (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {seq}\n"
            "      msa: empty\n"
        )
        yaml_path.write_text(yaml_content)

    # Run Boltz-2 batch prediction
    if needs_prediction:
        log(f"  Boltz monomer: predicting {len(needs_prediction)} designs "
            f"(batch mode, single model load)...")
        # boltz skips any design whose predictions/{id} dir exists, so a
        # re-predict must start from a clean slate or it silently keeps the
        # old structure.
        _nrm = _clear_stale_boltz_outputs(val_dir, [d["design_id"] for d in needs_prediction])
        if _nrm:
            log(f"  Boltz monomer: cleared {_nrm} stale prediction dir(s) to force re-predict")
        log(f"  [PROGRESS:esmfold] 0/{n_total} (predicting...)")
        cmd = [
            "conda", "run", "--no-capture-output", *cfg.conda_run_args("boltz"),
            "boltz", "predict", str(batch_yaml_dir),
            "--out_dir",            str(val_dir),
            "--recycling_steps",    "2",
            "--diffusion_samples",  "1",
            "--sampling_steps",     "200",
            "--preprocessing-threads", "48",
            "--no_kernels",
        ]
        try:
            run_cmd(cmd, timeout=None,
                    extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            # boltz can exit 0 having predicted NOTHING (empty manifest after
            # filter_inputs_structure). Without this the run reports
            # "0/N passed" and looks like a scientific result.
            _pred_root = val_dir / "boltz_results_batch_inputs" / "predictions"
            _produced = sum(1 for d in needs_prediction
                            if list((_pred_root / d["design_id"]).glob("plddt_*.npz")))
            if _produced == 0:
                raise RuntimeError(
                    f"boltz exited 0 but produced no monomer output for any of "
                    f"{len(needs_prediction)} designs (no plddt_*.npz under "
                    f"{_pred_root}). This is a tool/environment failure, NOT "
                    f"'every design refolded badly'.")
            if _produced < len(needs_prediction):
                log(f"  Boltz monomer: WARNING — only {_produced}/{len(needs_prediction)} "
                    f"designs produced output", err=True)
            log(f"  Boltz monomer: batch prediction complete")
        except Exception as e:
            # Do not let this look like "every design refolded badly": without
            # this record the run reports 0/N passed and ok:true.
            record_validation_failure("boltz_monomer", e)

    # Parse results — extract pLDDT for each design
    passing = []
    n_parsed = 0
    for i, d in enumerate(designs):
        design_id = d["design_id"]
        pred_dir = val_dir / "boltz_results_batch_inputs" / "predictions" / design_id

        if not pred_dir.exists():
            # NaN, not 0.0 — the prediction never arrived, so this is "not
            # measured", not "refolds terribly". 0.0 is a real value that the
            # pre-filter then reports as a failed refolder, hiding the fact that
            # Boltz produced no output for this design at all.
            d["esmfold_plddt"] = float("nan")
            log(f"  {design_id}: no Boltz monomer output — filtered")
            continue

        # Extract mean pLDDT from per-token npz
        plddt = 0.0
        plddt_files = sorted(pred_dir.glob("plddt_*.npz"))
        if plddt_files:
            try:
                plddt_arr = np.load(plddt_files[0])["plddt"] * 100.0
                # Exclude the terminal tags from the gate. A His-tag is
                # disordered, so its pLDDT is low and averaging it in drags the
                # mean down — good designs then fail the pre-filter for a reason
                # that has nothing to do with the designed binder. The refolding
                # RMSD already excludes tag residues; this is the same contract.
                _a, _b = len(n_tag), len(plddt_arr) - len(c_tag)
                _core = plddt_arr[_a:_b] if (_b > _a and len(plddt_arr) >= _a + len(c_tag) + 5) else plddt_arr
                plddt = float(_core.mean())
            except Exception as e:
                log(f"  Warning: could not parse plddt npz for {design_id}: {e}")

        # Save CIF path for refolding RMSD comparison
        cifs = sorted(pred_dir.glob("*.cif"))
        if cifs:
            d["esmfold_pdb"] = str(cifs[0])

        d["esmfold_plddt"] = plddt
        n_parsed += 1

        if plddt >= plddt_threshold:
            passing.append(d)
        else:
            log(f"  {design_id}: Boltz monomer pLDDT={plddt:.1f} < {plddt_threshold} — filtered")

        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            log(f"  [PROGRESS:esmfold] {i + 1}/{n_total} ({len(passing)} passed)")

    log(f"Boltz monomer: {len(passing)}/{n_total} passed (pLDDT ≥ {plddt_threshold})")
    return passing


def _boltz_input_key(target_seq, binder_seq, n_tag, c_tag, site_resnums,
                     force_site, site_max_distance, holo_ligands):
    """Fingerprint of everything that changes a Boltz complex prediction.

    The batch cache used to hit on `predictions/{design_id}/` plus any
    confidence json — i.e. on the design ID ALONE. Re-running into the same
    out_dir with a different --site (which sets the pocket constraint), with
    --n_tag/--c_tag (which change the sequence actually folded), or with
    different --holo_ligands silently reused the OLD prediction and reported it
    as the new result. Storing this key next to the prediction makes a changed
    input a cache miss.
    """
    import hashlib
    parts = [
        str(target_seq or ""),
        str(n_tag or "") + str(binder_seq or "") + str(c_tag or ""),
        ",".join(str(r) for r in sorted(site_resnums or [])),
        f"force_site={bool(force_site)}",
        f"site_max_distance={site_max_distance}",
        "ligands=" + ",".join(sorted(str(x) for x in (holo_ligands or []))),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ── Boltz input-fingerprint sidecars ─────────────────────────────────────────
# HARD RULE: never create anything under `boltz_results_*/predictions/{id}/`
# before boltz runs.  `boltz/main.py:filter_inputs_structure()` does:
#     existing = {d.name for d in pred_dir.iterdir() if d.is_dir()}
#     manifest = Manifest([r for r in manifest.records if r.id not in existing])
# It skips on the DIRECTORY NAME and never looks inside.  Stamping the key
# there first therefore emptied the manifest: boltz exited 0 in 9 s having
# predicted nothing, and the pipeline reported "0/120 passed" as if every
# design had simply refolded badly (runs/pdia6_validation_002, 2026-09-29).
_BOLTZ_KEY_FILENAME = {"monomer": ".boltz_monomer_key", "complex": ".boltz_input_key"}


def _boltz_key_sidecar(val_dir, design_id, kind):
    return Path(val_dir) / "input_keys" / f"{design_id}.{kind}.key"


def _read_boltz_key(val_dir, design_id, kind):
    """Current sidecar first, then the legacy in-prediction-dir location."""
    legacy = (Path(val_dir) / "boltz_results_batch_inputs" / "predictions"
              / design_id / _BOLTZ_KEY_FILENAME[kind])
    for p in (_boltz_key_sidecar(val_dir, design_id, kind), legacy):
        try:
            v = p.read_text().strip()
        except (OSError, IOError):
            continue
        if v:
            return v
    return ""


def _write_boltz_key(val_dir, design_id, kind, key):
    p = _boltz_key_sidecar(val_dir, design_id, kind)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(key or ""))
    except (OSError, IOError):
        pass


def _clear_stale_boltz_outputs(val_dir, design_ids):
    """Remove prediction dirs for designs we have decided to re-predict.

    Same `filter_inputs_structure` behaviour as above, from the other side: a
    design whose key MISMATCHED (site/tags/ligands changed) is put in
    needs_prediction, but boltz would skip it because its old directory is
    still there — and we would then parse the stale structure as the new
    result.  Returns the number of directories removed.
    """
    pred_root = Path(val_dir) / "boltz_results_batch_inputs" / "predictions"
    n = 0
    for did in design_ids:
        p = pred_root / did
        if p.exists():
            try:
                shutil.rmtree(p)
                n += 1
            except OSError:
                pass
    return n


def validate_boltz(designs, target_chain_seq, val_dir,
                   max_per_tool=20, target_len=0, dry_run=False,
                   site_resnums=None, target_residues=None,
                   boltz_devices=1, n_tag="", c_tag="",
                   force_site=False, site_max_distance=6.0, holo_ligands=None):
    """
    Stage 2: Boltz-2 uniform scoring of binder+target complex.
    BATCH MODE: all monomer-passing designs are predicted in a single Boltz-2
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

    log(f"Boltz validation: {len(to_validate)} designs (all monomer-passing)")

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
        _n_unverified_cache = [0]
        for d in to_validate:
            design_id = d["design_id"]
            result_dir = val_dir / "boltz_results_batch_inputs" / "predictions" / design_id
            _key = _boltz_input_key(target_chain_seq, d.get("binder_sequence", ""),
                                    n_tag, c_tag, site_resnums, force_site,
                                    site_max_distance, holo_ligands)
            _cached_key = _read_boltz_key(val_dir, design_id, "complex")
            _has_result = (result_dir.exists()
                           and bool(list(result_dir.glob("confidence_*.json"))))
            if _has_result and _cached_key == _key:
                pending.append((d, False))  # verified cache hit
            elif _has_result and not _cached_key:
                # Legacy cache written before inputs were fingerprinted. Reuse it
                # rather than forcing an expensive re-prediction of every old run,
                # but say that it could not be verified.
                _n_unverified_cache[0] += 1
                pending.append((d, False))
            else:
                if _has_result and _cached_key != _key:
                    log(f"  {design_id}: cached Boltz prediction was made with DIFFERENT "
                        f"inputs (site / terminal tags / ligands changed) — re-predicting.",
                        err=True)
                d["_boltz_input_key"] = _key
                pending.append((d, True))   # needs prediction

        needs_prediction = [(d, flag) for d, flag in pending if flag]
        already_done = [(d, flag) for d, flag in pending if not flag]

        if _n_unverified_cache[0]:
            log(f"  {_n_unverified_cache[0]} cached Boltz prediction(s) predate input "
                f"fingerprinting and could not be checked against the current --site / "
                f"tags / ligands. If you changed any of those since that run, delete "
                f"{val_dir}/boltz_results_batch_inputs/predictions to force a re-predict.",
                err=True)
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
                    "conda", "run", "--no-capture-output", *cfg.conda_run_args("boltz"),
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
                        log(f"  Boltz batch: WARNING — target MSA not found; the TARGET "
                            f"falls back to an API MSA (the binder stays msa: empty)",
                            err=True)
                except Exception as e:
                    log(f"  Boltz batch: WARNING — target MSA computation failed ({e}); "
                        f"the TARGET falls back to an API MSA (the binder stays msa: empty)",
                        err=True)

        use_precomputed_msa = target_msa_path.exists()
        if use_precomputed_msa:
            log(f"  Boltz batch: using pre-computed target MSA + empty binder MSA (fast mode)")
        else:
            log(f"  Boltz batch: no pre-computed target MSA — target uses the API MSA "
                f"(slow mode). Binder chains remain msa: empty.", err=True)

        # Log terminal tags if provided
        if n_tag or c_tag:
            log(f"  Terminal tags: N={n_tag or '(none)'}, C={c_tag or '(none)'}")

        # Write YAML files for designs that need prediction
        for d, _ in needs_prediction:
            seq = (n_tag + d["binder_sequence"] + c_tag) if (n_tag or c_tag) else d["binder_sequence"]
            design_id = d["design_id"]
            yaml_path = batch_yaml_dir / f"{design_id}.yaml"
            # Stamp the input fingerprint in the sidecar dir, so a later run can
            # tell whether the cached prediction matches its inputs. NOT inside
            # predictions/{id}/ — see _boltz_key_sidecar.
            _write_boltz_key(val_dir, design_id, "complex",
                             d.get("_boltz_input_key", ""))

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
            # ALWAYS empty for the designed binder, not only when the target MSA
            # succeeded. This line used to sit under `if use_precomputed_msa`, so
            # whenever the one-time target MSA failed the pipeline fell back to
            # "API MSA for all designs" — and the binder, an artificial sequence
            # with no natural homologs, got a real ColabFold MSA. A deep MSA
            # overrides the pocket constraint, which is exactly the failure mode
            # the binder-validation mode exists to prevent. The target may fall
            # back to an API MSA; the binder never should.
            yaml_content += "      msa: empty\n"
            # HOLO context: add the nucleotide as CCD ligands (Boltz places GDP/AF3/MG in the HRAS
            # nucleotide pocket). The first ligand (GDP) bears the alpha/beta phosphates that the
            # forced pocket also targets, so the binder is driven to the catalytic phosphate region.
            lig_chain = None
            if holo_ligands:
                for _i, _ccd in enumerate(holo_ligands):
                    cid = "CDEFG"[_i]
                    yaml_content += f"  - ligand:\n      id: {cid}\n      ccd: {_ccd}\n"
                    if lig_chain is None:
                        lig_chain = cid
            if site_indices:
                contacts = "\n".join(f"        - [A, {r}]" for r in site_indices)
                if force_site and lig_chain:                       # also force contact to the nucleotide phosphate
                    contacts += f"\n        - [{lig_chain}, PA]\n        - [{lig_chain}, PB]"
                yaml_content += (
                    "constraints:\n"
                    "  - pocket:\n"
                    "      binder: B\n"
                    "      contacts:\n"
                    f"{contacts}\n"
                )
                if force_site:
                    yaml_content += (
                        f"      max_distance: {site_max_distance}\n"
                        "      force: true\n"
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
            # boltz skips any design whose predictions/{id} dir exists, so a
            # re-predict must start from a clean slate or it silently keeps the
            # old structure.
            _nrm = _clear_stale_boltz_outputs(
                val_dir, [d["design_id"] for d, _ in needs_prediction])
            if _nrm:
                log(f"  Boltz batch: cleared {_nrm} stale prediction dir(s) to force re-predict")
            cmd = [
                "conda", "run", "--no-capture-output", *cfg.conda_run_args("boltz"),
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
                # boltz can exit 0 having predicted NOTHING (empty manifest
                # after filter_inputs_structure). Treat that as a failure so it
                # takes the per-design fallback instead of reporting every
                # design as unscorable.
                _pred_root = val_dir / "boltz_results_batch_inputs" / "predictions"
                _produced = sum(1 for d, _ in needs_prediction
                                if list((_pred_root / d["design_id"]).glob("confidence_*.json")))
                if _produced == 0:
                    raise RuntimeError(
                        f"boltz exited 0 but produced no complex output for any of "
                        f"{len(needs_prediction)} designs (no confidence_*.json under "
                        f"{_pred_root})")
                if _produced < len(needs_prediction):
                    log(f"  Boltz batch: WARNING — only {_produced}/{len(needs_prediction)} "
                        f"designs produced output", err=True)
            except Exception as e:
                record_validation_failure("boltz_complex", e)
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
                        "conda", "run", "--no-capture-output", *cfg.conda_run_args("boltz"),
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
        # NaN, not 0.0: a corrupt/absent confidence JSON must read as "not measured",
        # not as a real design that scored an iPTM of exactly zero. 0.0 survives
        # every filter and is written to rankings.csv as a genuine value.
        boltz_iptm = float("nan")
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
                # NaN, not 0.0 — matches rerank_binders. A confidence JSON that
                # parses but lacks the key must read as "not measured": a 0.0
                # defeats the NaN gate in compute_combined_score and the
                # "no Boltz complex" guard in rank_designs.
                boltz_iptm        = cdata.get("iptm", float("nan"))
                boltz_ptm         = cdata.get("ptm", float("nan"))
                boltz_confidence  = cdata.get("confidence_score", float("nan"))
                boltz_complex_pde = cdata.get("complex_pde", float("nan"))
                boltz_complex_ipde= cdata.get("complex_ipde", float("nan"))
                boltz_protein_iptm= cdata.get("protein_iptm", float("nan"))
                # HOLO mode: the global iptm averages in the binder<->ligand interfaces. Score on the
                # protein-protein iptm (binder<->HRAS) instead, when Boltz reports it.
                if holo_ligands and boltz_protein_iptm == boltz_protein_iptm:
                    boltz_iptm = boltz_protein_iptm
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
                # Cap at the BINDER end, exactly as the interface-PAE block does.
                # An open-ended [tlen:] slice averages in every holo-ligand token
                # (GDP/ALF/MG follow the binder in the YAML), so boltz_binder_plddt
                # — which carries w_plddt in the score — was diluted by ligand
                # confidence in every --holo_ligands run.
                _blen_p = len(n_tag) + len(d.get("binder_sequence", "")) + len(c_tag)
                _bend_p = tlen + _blen_p
                if len(plddt_arr) > tlen:
                    _seg = plddt_arr[tlen:_bend_p] if _bend_p <= len(plddt_arr) else plddt_arr[tlen:]
                    boltz_binder_plddt = float(_seg.mean()) if len(_seg) else float("nan")
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
                # Off-diagonal block: rows=target (0:tlen), cols=binder (tlen:bend).
                # bend caps the binder columns BEFORE any holo ligand tokens (GDP/AF3/MG come after
                # the binder in the YAML), so the interface PAE stays binder<->target, not target<->ligand.
                blen = len(n_tag) + len(d.get("binder_sequence", "")) + len(c_tag)
                bend = tlen + blen
                if pae_mat.shape[0] > tlen and pae_mat.shape[1] > tlen:
                    interface_block = pae_mat[:tlen, tlen:bend]
                    boltz_min_interface_pae  = float(interface_block.min())
                    boltz_mean_interface_pae = float(interface_block.mean())

                    # Site-specific PAE: only rows corresponding to binding site residues
                    if site_row_indices:
                        valid_rows = [r for r in site_row_indices if r < tlen]
                        if valid_rows:
                            site_block = pae_mat[valid_rows][:, tlen:bend]
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
                    _blen = len(n_tag) + len(d.get("binder_sequence", "")) + len(c_tag)
                    pde_block = pde_mat[:tlen, tlen:tlen + _blen]
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

def _cif_atom_site_cols(path):
    """Resolve `_atom_site` column NAME → index for an mmCIF file.

    mmCIF does not fix column order, and every producer in this pipeline writes a
    different one. Boltz-2 puts label_asym_id at index 9 and auth_seq_id at 7;
    BoltzGen, PXDesign and BindCraft2 all emit canonical PDBx order, where index
    9 is pdbx_PDB_ins_code and 7 is label_entity_id. A reader that indexes by
    POSITION is therefore correct for exactly one producer and returns silent
    garbage for the others — which is how 75 of the 130 ranked designs in
    runs/pdia6_validation_001 reached the final panel with no site geometry at
    all (boltzgen parsed to a single pseudo-chain '?', pxdesign and bindcraft2 to
    nothing). Every CIF reader in this module goes through this function.

    Returns (cols, first_data_line_index) or (None, None).
    """
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except (OSError, IOError):
        return None, None
    cols, n, in_loop, start = {}, 0, False, None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            in_loop, cols, n = False, {}, 0
        elif s.startswith("_atom_site."):
            in_loop = True
            cols[s[len("_atom_site."):]] = n
            n += 1
        elif in_loop and s and not s.startswith(("_", "#", "loop_", "data_")):
            start = i
            break
    if not cols or start is None:
        return None, None
    return cols, start


def _cif_col(cols, name, alt=None):
    """Index of an _atom_site column, preferring `name`, else `alt`, else -1."""
    if name in cols:
        return cols[name]
    if alt and alt in cols:
        return cols[alt]
    return -1


def _iter_struct_atoms(filepath):
    """Yield (chain, resnum, resname, atom_name, x, y, z) for heavy atoms.

    One reader for PDB and mmCIF, covering every producer. Hydrogens are skipped;
    for multi-model CIFs only the first model is read.
    """
    path_str = str(filepath)
    if path_str.endswith(".cif"):
        cols, start = _cif_atom_site_cols(filepath)
        if not cols or start is None:
            return
        i_sym = _cif_col(cols, "type_symbol")
        i_atom = _cif_col(cols, "auth_atom_id", "label_atom_id")
        i_comp = _cif_col(cols, "auth_comp_id", "label_comp_id")
        i_ch = _cif_col(cols, "auth_asym_id", "label_asym_id")
        i_seq = _cif_col(cols, "auth_seq_id", "label_seq_id")
        i_x = _cif_col(cols, "Cartn_x")
        i_y = _cif_col(cols, "Cartn_y")
        i_z = _cif_col(cols, "Cartn_z")
        i_model = _cif_col(cols, "pdbx_PDB_model_num")
        if min(i_atom, i_comp, i_ch, i_seq, i_x, i_y, i_z) < 0:
            return
        need = max(i_sym, i_atom, i_comp, i_ch, i_seq, i_x, i_y, i_z, i_model)
        try:
            lines = Path(filepath).read_text(errors="replace").splitlines()
        except (OSError, IOError):
            return
        first_model = None
        for line in lines[start:]:
            s = line.strip()
            if not s or s.startswith(("#", "_", "loop_", "data_")):
                break
            if not s.startswith("ATOM"):
                continue
            p = s.split()
            if len(p) <= need:
                continue
            if i_model >= 0:
                if first_model is None:
                    first_model = p[i_model]
                elif p[i_model] != first_model:
                    break
            if i_sym >= 0 and p[i_sym] == "H":
                continue
            try:
                yield (p[i_ch], int(p[i_seq]), p[i_comp], p[i_atom],
                       float(p[i_x]), float(p[i_y]), float(p[i_z]))
            except (ValueError, IndexError):
                continue
    else:
        try:
            fh = open(filepath)
        except (OSError, IOError):
            return
        with fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                element = line[76:78].strip() if len(line) >= 78 else line[12:16].strip()[:1]
                if element == "H":
                    continue
                try:
                    yield (line[21], int(line[22:26].strip()), line[17:20].strip(),
                           line[12:16].strip(), float(line[30:38]),
                           float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue


def _parse_heavy_atoms(filepath):
    """Parse heavy atoms by chain from a PDB or mmCIF file.

    Returns (chains, chain_res_counts, chain_residues) where:
      chains:           chain_id → [(resnum, x, y, z, atom_name), ...]
      chain_res_counts: chain_id → number of unique residues
      chain_residues:   chain_id → [(resnum, one_letter), ...] in file order
    Returns (None, None, None) on failure.

    `atom_name` and `chain_residues` exist so callers can compute CA geometry and
    identify a chain by sequence WITHOUT re-reading and re-parsing the file with
    a second, independently-written reader. There used to be exactly such a
    duplicate reader inside _compute_site_metrics, and it carried the same
    positional-index bug as this one.
    """
    chains, chain_res, chain_seq = {}, {}, {}
    seen = set()
    any_atom = False
    for chain, resnum, resname, atom_name, x, y, z in _iter_struct_atoms(filepath):
        any_atom = True
        chains.setdefault(chain, []).append((resnum, x, y, z, atom_name))
        chain_res.setdefault(chain, set()).add(resnum)
        key = (chain, resnum)
        if key not in seen:
            seen.add(key)
            chain_seq.setdefault(chain, []).append((resnum, AA3.get(resname.upper(), "X")))
    if not any_atom:
        return None, None, None
    return chains, {c: len(r) for c, r in chain_res.items()}, chain_seq


def _resolve_site_in_structure(chain_res_counts, chain_residues, site_resnums,
                               target_residues, target_chain_hint=None):
    """
    Find the target chain in THIS file and translate the site into ITS numbering.

    Producers disagree on both chain roles and residue numbering. One single run
    (runs/pdia6_validation_001) contains all five of:

        Boltz-2     chain A  = target, renumbered 1..N
        BoltzGen    chain B  = target, numbered 17..N+16   (chain A is the BINDER)
        PXDesign    chain A0 = target, renumbered 1..N
        BindCraft2  chain A  = target, ORIGINAL numbering  (270..419 here)
        BindCraft   chain A  = target, renumbered 1..N     (PDB)

    The previous code used a chain-LETTER hint plus a boolean `use_renumbered`.
    A 2-way flag cannot express a 5-way problem: for BoltzGen the hint "A" names
    the binder, and neither candidate site set matches its +16 frame. So resolve
    per file — identify the target by residue count, then map the site ORDINALLY
    through the reference target's residue list.

    Returns (target_chain_id, binder_chain_id, site_set, how) or None.
    """
    if not chain_res_counts or len(chain_res_counts) < 2:
        return None

    n_target = len(target_residues) if target_residues else None
    target_chain_id = None
    how = ""

    # 1. Residue count is the strongest signal and is numbering-agnostic.
    if n_target:
        exact = [c for c, n in chain_res_counts.items() if n == n_target]
        if len(exact) == 1:
            target_chain_id, how = exact[0], "residue count"
        elif len(exact) > 1:
            # Ambiguous (e.g. binder happens to match the target's length).
            # Prefer the hinted chain, else the one carrying the site numbers.
            if target_chain_hint in exact:
                target_chain_id, how = target_chain_hint, "residue count + hint"
            else:
                site = set(site_resnums)
                best = max(exact, key=lambda c: len({r for r, _ in chain_residues.get(c, [])} & site))
                target_chain_id, how = best, "residue count + site overlap"

    # 2. No count match (a cropped target). Take whichever chain actually
    #    carries the most site residue numbers.
    if target_chain_id is None:
        site = set(site_resnums)
        scored = [(len({r for r, _ in chain_residues.get(c, [])} & site), c)
                  for c in chain_res_counts]
        best_n, best_c = max(scored)
        if best_n:
            target_chain_id, how = best_c, "site-number overlap"

    # 3. Last resort: the hint, else the largest chain.
    if target_chain_id is None:
        if target_chain_hint and target_chain_hint in chain_res_counts:
            target_chain_id, how = target_chain_hint, "chain hint (weak)"
        else:
            target_chain_id = max(chain_res_counts.items(), key=lambda x: x[1])[0]
            how = "largest chain (weak)"

    others = [(c, n) for c, n in chain_res_counts.items() if c != target_chain_id]
    if not others:
        return None
    binder_chain_id = max(others, key=lambda x: x[1])[0]

    # Translate the site into this file's numbering.
    tgt_nums = sorted({r for r, _ in chain_residues.get(target_chain_id, [])})
    site = set(site_resnums)
    direct = site & set(tgt_nums)
    if len(direct) == len(site) and direct:
        return target_chain_id, binder_chain_id, direct, f"{how}; original numbering"

    if target_residues and len(tgt_nums) == len(target_residues):
        pos = {r: i for i, r in enumerate(target_residues)}
        idx = [pos[r] for r in site_resnums if r in pos]
        if idx:
            mapped = {tgt_nums[i] for i in idx if 0 <= i < len(tgt_nums)}
            if mapped:
                return target_chain_id, binder_chain_id, mapped, f"{how}; ordinal mapping"

    if direct:
        return target_chain_id, binder_chain_id, direct, f"{how}; partial original numbering"
    return None


def _compute_site_metrics(args):
    """Worker function for parallel site metric computation.

    Returns dict with site metrics, or None to skip.
    """
    import numpy as np
    (struct_file, site_resnums, target_residues, n_site_residues,
     max_dist, interface_dist, target_chain_hint) = args

    chains, chain_res_counts, chain_residues = _parse_heavy_atoms(struct_file)
    if not chains or not chain_res_counts or len(chain_res_counts) < 2:
        return None

    resolved = _resolve_site_in_structure(chain_res_counts, chain_residues,
                                          site_resnums, target_residues,
                                          target_chain_hint)
    if resolved is None:
        return None
    target_chain_id, binder_chain_id, site_set, _how = resolved

    binder_atoms = chains[binder_chain_id]
    target_atoms = chains[target_chain_id]

    binder_coords = np.array([[a[1], a[2], a[3]] for a in binder_atoms])

    # Site contact fraction
    site_res_atoms = {}
    for resnum, x, y, z, _name in target_atoms:
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
    for resnum, x, y, z, _name in target_atoms:
        target_all_atoms.setdefault(resnum, []).append([x, y, z])

    binder_res_atoms = {}
    for resnum, x, y, z, _name in binder_atoms:
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
    for resnum, x, y, z, _name in target_atoms:
        if resnum in site_set:
            site_heavy_coords.append([x, y, z])

    # 2) CA centroid: one CA per residue, equal weight — true geometric center
    # Taken straight from the already-parsed atoms. This used to be a SECOND,
    # independently-written reader of the same file, carrying its own copy of
    # the positional-index bug — so site_centroid_dist_CA and site_cos_angle
    # were NaN for every producer whose column order was not Boltz-2's, even
    # when the rest of the metrics were fine. One reader cannot drift from
    # itself.
    site_ca_coords = [[x, y, z] for resnum, x, y, z, name in target_atoms
                      if resnum in site_set and name == "CA"]

    binder_iface_center = np.mean(iface_binder_coords, axis=0) if iface_binder_coords else None
    # A design with NO interface at all is not "unmeasured" — it is measured, and
    # the answer is that it touches nothing. Leaving these NaN made
    # --max_site_centroid_dist and --min_site_cos PASS the very designs they
    # exist to reject (NaN passes, by policy). Record the honest extreme instead.
    if binder_iface_center is None:
        centroid_dist_heavy = float("inf")
        centroid_dist_CA = float("inf")
        site_cos_angle = -1.0
    else:
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
            # Outward normal points from the protein body out through the site;
            # binder vector = site_center → binder_interface_center; cos > 0 means
            # the binder approaches from the solvent side.
            #
            # The reference centroid is LOCAL (target atoms within 12 A of the
            # site), not the whole-target centroid. A global centroid makes this a
            # RADIAL vector, which only approximates a surface normal on a compact
            # globular target — for an elongated or multi-domain target, or a
            # concave pocket, it can point along the body rather than out of the
            # surface, and --min_site_cos then filters on a direction that is not
            # the local normal. Falls back to the global centroid when too few
            # local atoms exist to define one.
            _tgt_xyz = np.array([[x, y, z] for _r, x, y, z, _n in target_atoms])
            _d_site = np.linalg.norm(_tgt_xyz - site_center, axis=1)
            _local = _tgt_xyz[_d_site <= 12.0]
            target_center = (_local.mean(axis=0) if len(_local) >= 20
                             else _tgt_xyz.mean(axis=0))
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


def compute_binder_compactness(designs, target_residues=None, dry_run=False):
    """
    Record how globular each binder is: Rg of its CA trace vs the expectation
    for a folded chain of that length.

        binder_rg        — radius of gyration over binder CA atoms (Å)
        binder_rg_ratio  — binder_rg / (2.2 * N**0.38), the empirical globular Rg

    ~1.0 is compact. This exists because refolding RMSD structurally cannot catch
    an extended single helix: a lone helix refolds to a lone helix, scoring a
    near-perfect RMSD, so it can ride iPTM + pLDDT into the top of the panel. In
    runs/pdia6_validation_001 the #2 design by combined_score (rfdiffusion_0014,
    0.833) is a single straight helix at ratio 1.40, and six true single helices
    sit at 2.49-2.81, against a median of ~0.95 for real binders.

    The value is ALWAYS recorded; dropping designs on it is opt-in via
    --max_rg_ratio, because a filter that silently deletes a seventh of the panel
    should be a decision, not a default.
    """
    import numpy as np

    n_done = n_skip = 0
    for d in designs:
        d.setdefault("binder_rg", float("nan"))
        d.setdefault("binder_rg_ratio", float("nan"))
        if dry_run:
            continue
        struct = None
        for key in ("complex_cif", "binder_pdb"):
            v = d.get(key)
            if v and Path(v).exists():
                struct = v
                break
        if not struct:
            n_skip += 1
            continue
        chains, counts, residues = _parse_heavy_atoms(struct)
        if not counts:
            n_skip += 1
            continue

        binder_chain = None
        if len(counts) >= 2 and target_residues:
            # Same rule as the site filter: the target is the chain whose residue
            # count matches the reference; the binder is the largest of the rest.
            n_t = len(target_residues)
            exact = [c for c, n in counts.items() if n == n_t]
            if len(exact) == 1:
                others = [(c, n) for c, n in counts.items() if c != exact[0]]
                if others:
                    binder_chain = max(others, key=lambda x: x[1])[0]
        if binder_chain is None:
            # Monomer file, or an ambiguous complex: fall back to the sequence
            # recorded for this design, else the shortest chain.
            seq = (d.get("binder_sequence") or "").strip()
            if seq:
                for c, rs in residues.items():
                    if len(rs) == len(seq):
                        binder_chain = c
                        break
            if binder_chain is None:
                binder_chain = min(counts, key=lambda c: counts[c])

        ca = np.array([[x, y, z] for rn, x, y, z, name in chains.get(binder_chain, [])
                       if name == "CA"])
        if len(ca) < 10:
            n_skip += 1
            continue
        rg = float(np.sqrt(((ca - ca.mean(axis=0)) ** 2).sum(axis=1).mean()))
        expected = 2.2 * (len(ca) ** 0.38)
        d["binder_rg"] = round(rg, 2)
        d["binder_rg_ratio"] = round(rg / expected, 3) if expected > 0 else float("nan")
        n_done += 1

    if n_done:
        vals = [d["binder_rg_ratio"] for d in designs
                if isinstance(d.get("binder_rg_ratio"), float)
                and d["binder_rg_ratio"] == d["binder_rg_ratio"]]
        if vals:
            vals_sorted = sorted(vals)
            med = vals_sorted[len(vals_sorted) // 2]
            elongated = [v for v in vals if v > 1.25]
            log(f"  Compactness: {n_done} measured, {n_skip} skipped — "
                f"Rg_ratio median={med:.2f}, max={max(vals):.2f}")
            if elongated:
                log(f"  {len(elongated)} design(s) have Rg_ratio > 1.25 (extended / "
                    f"single-helix shape). Refolding RMSD does not catch these; "
                    f"pass --max_rg_ratio 1.25 to drop them.")
    return designs


def geometric_site_filter(designs, site_resnums, target_residues,
                          max_dist=15.0, min_site_fraction=0.0, interface_dist=5.0,
                          dry_run=False, target_chain_id="A"):
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

    # The dual-patch split that used to live here computed patches_orig /
    # patches_renum / all_site_* and NOTHING read them — the per-file site
    # resolver replaced that machinery. It was guarded by
    # `if len(sorted_site) < 2: return designs`, so a SINGLE-RESIDUE --site
    # skipped the entire geometric measurement (silently, with no log line) in
    # order to protect dead code. Removed; a one-residue site now measures like
    # any other.
    pass_counts = {}
    fail_counts = {}
    skip_counts = {}

    # Denominator counts only site residues that actually EXIST in the target.
    # Using len(site_resnums) meant a typo'd or out-of-range residue permanently
    # capped site_contact_fraction below 1.0 for every design, so an SIF/contact
    # threshold could become unreachable through no fault of the designs.
    _present = set(target_residues or [])
    n_site_residues = (len([r for r in site_resnums if r in _present])
                       if _present else len(site_resnums))
    if _present and n_site_residues != len(site_resnums):
        _absent = [r for r in site_resnums if r not in _present]
        log(f"  Geometric filter: {len(_absent)} site residue(s) are not in the "
            f"target chain and are excluded from the contact denominator: "
            f"{', '.join(str(r) for r in _absent[:12])}"
            f"{' ...' if len(_absent) > 12 else ''}", err=True)
    if n_site_residues == 0:
        log("  Geometric filter: no requested site residue exists in the target "
            "chain — cannot measure site geometry.", err=True)
        return designs

    # Prepare worker args for parallelizable designs
    worker_indices = []
    worker_args = []
    for i, d in enumerate(designs):
        tool = d.get("tool") or _get_tool_from_design_id(d.get("design_id", "")) or "unknown"

        # No per-tool special cases, and no `use_renumbered` flag. Both were
        # wrong: the flag is a 2-way switch for a problem with at least five
        # answers (see _resolve_site_in_structure), and the pxdesign skip existed
        # only because the old positional CIF parser could not read PXDesign's
        # output at all. Hand the worker the site in the TARGET's own numbering
        # and let it resolve the frame from the file it is actually holding.
        struct_file = None
        for key in ("complex_cif", "binder_pdb"):
            v = d.get(key)
            if v and Path(v).exists():
                struct_file = v
                break

        if not struct_file or dry_run:
            skip_counts[tool] = skip_counts.get(tool, 0) + 1
            d["site_geometric_pass"] = None if not dry_run else True
            d["site_geometry_status"] = "dry_run" if dry_run else "no_structure"
            continue

        worker_indices.append(i)
        worker_args.append((struct_file, sorted(site_resnums), list(target_residues or []),
                            n_site_residues, max_dist, interface_dist, target_chain_id))

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
            # UNMEASURED, not passed. Writing True here (as this did) is how a
            # design that was never geometrically checked reached the final panel
            # indistinguishable from one that passed the check.
            skip_counts[tool] = skip_counts.get(tool, 0) + 1
            d["site_geometric_pass"] = None
            d["site_geometry_status"] = "parse_failed"
            continue

        d.update(result)
        d["site_geometry_status"] = "measured"
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

    total_skip = sum(skip_counts.values())
    if total_skip and not dry_run:
        by_tool = ", ".join(f"{t}={n}" for t, n in sorted(skip_counts.items()) if n)
        log(f"  {total_skip}/{len(designs)} designs could not be measured ({by_tool}) — "
            f"no usable structure. They carry site_geometric_pass=None and "
            f"site_geometry_status!='measured', so a blank site_* value for them means "
            f"'never checked', NOT 'scored zero'. Most will already have failed an "
            f"earlier stage; how many reach the ranked panel is reported at ranking.",
            err=True)

    fracs = [d.get("site_contact_fraction", 0) for d in designs
             if d.get("site_geometric_pass") is True and "site_contact_fraction" in d]
    if fracs:
        log(f"  Contact fractions: min={min(fracs):.2f}, median={sorted(fracs)[len(fracs)//2]:.2f}, "
            f"max={max(fracs):.2f} (n={len(fracs)})")

    total_f = sum(fail_counts.values())
    if total_f:
        log(f"  Geometric filter: {total_f} off-site designs (0 site contacts) "
            f"will skip Rosetta and will not be ranked")

    return designs


# ── Refolding RMSD ────────────────────────────────────────────────────────────

def compute_refolding_rmsd(designs, dry_run=False, n_tag="", c_tag=""):
    """Compute CA RMSD between Boltz-2 monomer (binder alone) and Boltz-2 complex (binder in complex).

    High RMSD (>3-4 Å) indicates the binder only folds correctly when bound
    to the target — likely disordered on its own.

    When terminal tags are present, the tag residues are excluded from RMSD
    calculation (tags are often disordered and would inflate RMSD).

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
        """Extract CA coords from CIF, return dict chain -> [(resnum, x, y, z)].

        Delegates to the shared header-driven reader. It used to index _atom_site
        by position, which is only correct for Boltz-2's column order — so
        refolding RMSD silently saw an empty structure for every other producer.
        """
        chains = {}
        for ch, rn, _resname, name, x, y, z in _iter_struct_atoms(path):
            if name == "CA":
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
            # Monomer structure: single chain, binder only (PDB or CIF)
            if esm_pdb.endswith(".cif"):
                esm_chains = _parse_ca_coords_cif(esm_pdb)
            else:
                esm_chains = _parse_ca_coords_pdb(esm_pdb)
            esm_cas = list(esm_chains.values())[0] if esm_chains else []

            # Boltz CIF: multi-chain complex. The binder chain is the one whose length matches the
            # monomer (esm_cas) — NOT simply the shortest chain (which mis-selects the target when the
            # binder is longer than the target, e.g. 200-250 aa binders vs a 166 aa target).
            cif_chains = _parse_ca_coords_cif(boltz_cif)
            if len(cif_chains) < 2:
                n_skipped += 1
                continue
            if esm_cas:
                binder_chain = min(cif_chains, key=lambda c: abs(len(cif_chains[c]) - len(esm_cas)))
            else:
                binder_chain = min(cif_chains, key=lambda c: len(cif_chains[c]))
            boltz_cas = cif_chains[binder_chain]

            # Match by residue count (both should be 1-based sequential)
            n = min(len(esm_cas), len(boltz_cas))
            if n < 10:
                n_skipped += 1
                continue

            # Exclude terminal tag residues from RMSD (tags are disordered)
            n_skip_start = len(n_tag)
            n_skip_end = len(c_tag)
            esm_core = esm_cas[n_skip_start:n - n_skip_end if n_skip_end else n]
            boltz_core = boltz_cas[n_skip_start:n - n_skip_end if n_skip_end else n]
            if len(esm_core) < 10:
                n_skipped += 1
                continue

            esm_xyz = [[x, y, z] for _, x, y, z in esm_core]
            boltz_xyz = [[x, y, z] for _, x, y, z in boltz_core]

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

# NetSolP was the one tool path with no config.yaml key — env var or the
# hardcoded ${software_dir} layout only, so a box that keeps it elsewhere had to
# export NETSOLP_DIR for every invocation. Now it follows the same
# env > config > default precedence as every other tool.
_NETSOLP_DIR = cfg.tool("netsolp_dir", "NETSOLP_DIR",
                        os.path.join(_SOFTWARE_DIR, "NetSolP-1.0", "PredictionServer"))


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
            "conda", "run", "--no-capture-output", *cfg.conda_run_args("esmfold"),
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
    n_offsite_skipped = 0
    for d in designs:
        struct = d.get("complex_cif") or d.get("binder_pdb")
        if not (struct and Path(struct).exists()):
            continue
        # Actually skip the off-site designs. The geometric filter has always
        # LOGGED "off-site designs will skip Rosetta", but nothing read
        # site_geometric_pass here, so every rejected design was still scored —
        # pure wasted time on a 12-worker cap. They are excluded from ranking
        # anyway (rank_designs drops site_geometric_pass is False), so this
        # costs nothing. `is False` is deliberate: None means NOT MEASURED and
        # must still be scored.
        if d.get("site_geometric_pass") is False:
            n_offsite_skipped += 1
            continue
        to_score.append(d)

    if n_offsite_skipped:
        log(f"Rosetta: skipping {n_offsite_skipped} design(s) the geometric filter "
            f"rejected as off-site")

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
    cpu_cap = min(12, max(4, total_cpus // 2))
    try:
        with open("/proc/meminfo") as mf:
            for line in mf:
                if line.startswith("MemAvailable:"):
                    avail_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                avail_gb = 999
        # No floor: flooring at 4 defeats the guard on a low-memory box — the
        # whole point is to drop BELOW 4 workers when RAM is short. Rosetta is
        # the stage that OOMs (workers capped at 12 for exactly this reason).
        ram_cap = max(1, int(avail_gb / 2))
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
        "# Re-score when the STRUCTURE is newer than the scores file: scores.json",
        "# is keyed on design_id alone, so a re-predicted complex silently kept",
        "# its old interface dG — and dG is a scoring term.",
        "_sj_mtime = os.path.getmtime(scores_out) if os.path.exists(scores_out) else 0.0",
        "def _stale(sp):",
        "    try:",
        "        return os.path.getmtime(sp) > _sj_mtime + 60",
        "    except OSError:",
        "        return False",
        "pending = [(did, sp) for did, sp in jobs",
        "           if did not in results or results[did] is None or _stale(sp)]",
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
        "error_msgs = {}",
        "t0 = time.time()",
        "with ProcessPoolExecutor(max_workers=n_workers) as pool:",
        "    futures = {pool.submit(score_one, job): job[0] for job in pending}",
        "    for future in as_completed(futures):",
        "        design_id, data, msg = future.result()",
        "        results[design_id] = data",
        "        if not data: error_msgs[design_id] = msg",
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
        "# Save error log for failed designs (helps diagnose env issues)",
        "errors = {k: v for k, v in error_msgs.items() if v}",
        "if errors:",
        "    err_path = os.path.join(os.path.dirname(scores_out), 'errors.log')",
        "    with open(err_path, 'w') as f:",
        "        for did, msg in list(errors.items())[:20]:",
        "            f.write(f'{did}: {msg}\\n')",
        "    print(f'Rosetta: {len(errors)} failures logged to errors.log')",
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
        run_cmd(["conda", "run", "--no-capture-output", *cfg.conda_run_args("bindcraft"),
                 "python", str(score_script)],
                timeout=max(7200, len(to_score) * 180))  # ~3 min/design, min 2h
    except RuntimeError as e:
        record_validation_failure("rosetta", e)
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
            elif tool_prefix == "bindcraft2":
                iptm = d.get("bc2_i_ptm", float("nan"))
                iptm_source = "native_bindcraft2"
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
            elif tool_prefix == "bindcraft2":
                # BC2 reports binder confidence in the bound state as `pLDDT`,
                # 0-1 (its B-factor column is 0-100, the table is not).
                raw = d.get("bc2_plddt", float("nan"))
                plddt = raw * 100.0 if (raw == raw and raw <= 1.0) else raw
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
        d["_score_renormalized"] = False
    elif w_dg > 0:
        # Renormalize, as the docstring has always promised. Without this a design
        # whose Rosetta scoring failed is capped at (w_plddt + w_iptm) while its
        # peers reach 1.0 \u2014 with the recommended 0.3,0.6,0.1 that is a flat 0.1
        # handicap. Rosetta failures are common and only logged (see
        # validation/rosetta/errors.log, and the `import gemmi` failure mode that
        # zeroes ALL Rosetta scores), so this silently down-ranked exactly the
        # subset that failed, invisibly.
        _kept = w_plddt + w_iptm
        if _kept > 0:
            score *= (w_plddt + w_iptm + w_dg) / _kept
        d["_score_renormalized"] = True

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
        n_pae_nan = 0
        for d in validated:
            tool_prefix = _get_tool_from_design_id(d.get("design_id", ""))
            # Skip PAE filter for native-iPTM tools when reprediction is off
            if not reprediction and tool_prefix in IPTM_NATIVE_TOOLS:
                passed_pae.append(d)
                continue
            min_pae = d.get("boltz_min_interface_pae", float("nan"))
            # NaN PASSES, per the project-wide policy ("a design with no
            # measurement is not penalised") and to match filter_site_pae
            # immediately below, which already does this. This branch used
            # `min_pae != min_pae or ...`, i.e. it DROPPED every design with no
            # interface PAE — a different filter from the one documented, and the
            # opposite of its own sibling.
            if min_pae == min_pae and min_pae > filter_interface_pae:
                # Move to unvalidated (no combined_score rank)
                d["combined_score"] = float("nan")
                unvalidated.append(d)
            else:
                if min_pae != min_pae:
                    n_pae_nan += 1
                passed_pae.append(d)
        n_filtered = before - len(passed_pae)
        if n_filtered > 0:
            log(f"  Rank: filtered {n_filtered} designs by interface pAE > {filter_interface_pae}"
                + (f" ({n_pae_nan} had no measurement and passed)" if n_pae_nan else ""))
        elif n_pae_nan:
            log(f"  Rank: interface pAE filter removed 0 designs "
                f"({n_pae_nan}/{before} had no measurement)")
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
    # when they should have one.
    before = len(validated)
    passed_geo = []
    # Two DIFFERENT reasons shared one counter, and the log called both
    # "off-site designs": a design measured as touching nothing, and a design
    # whose Boltz-2 prediction never arrived. The second is an infrastructure
    # casualty, not a scientific verdict, and saying "off-site" about it sends
    # the reader to fix the epitope instead of the crash.
    n_offsite = 0
    n_no_complex = 0
    for d in validated:
        if d.get("site_geometric_pass") is False:
            n_offsite += 1
            d["combined_score"] = float("nan")
            unvalidated.append(d)
        elif d.get("boltz_iptm") != d.get("boltz_iptm"):
            # No Boltz-2 prediction → no complex structure → cannot verify site contact.
            # But native-iPTM tools (BindCraft, BoltzGen, PXDesign, Proteina Complexa)
            # may legitimately skip Boltz-2 when reprediction is off — keep them ranked.
            tool_prefix = _get_tool_from_design_id(d.get("design_id", ""))
            if not reprediction and tool_prefix in IPTM_NATIVE_TOOLS:
                passed_geo.append(d)  # trust native validation
            else:
                n_no_complex += 1
                d["combined_score"] = float("nan")
                unvalidated.append(d)
        else:
            passed_geo.append(d)
    if n_offsite:
        log(f"  Rank: removed {n_offsite} design(s) measured as OFF-SITE "
            f"(0 site contacts)")
    if n_no_complex:
        log(f"  Rank: removed {n_no_complex} design(s) with NO Boltz-2 complex — "
            f"these were never measured against the site, so this is a prediction "
            f"failure, not an off-site verdict. Check the Boltz stage.", err=True)
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
    # The one number that actually matters about unmeasured geometry: how many
    # designs reached the RANKED panel without ever being checked against the
    # site. Reported here, at the single point both scripts funnel through, so
    # it cannot drift out of sync with them. Unmeasured-and-unranked designs are
    # harmless (they failed an earlier stage); unmeasured-and-ranked are the
    # ones that quietly look like site binders.
    try:
        _ranked = [d for d in designs if d.get("rank")]
        _blind = [d for d in _ranked
                  if d.get("site_geometry_status") not in (None, "measured")]
        if _blind:
            _by = {}
            for d in _blind:
                t = d.get("tool") or "?"
                _by[t] = _by.get(t, 0) + 1
            log(f"  NOTE: {len(_blind)}/{len(_ranked)} RANKED designs have no site "
                f"geometry ({', '.join(f'{t}={n}' for t, n in sorted(_by.items()))}). "
                f"Their site_* columns are blank because they were never measured, "
                f"not because they scored zero. Re-rank with --require_site_geometry "
                f"to exclude them.", err=True)
    except Exception:
        pass
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
        "site_geometry_status",
        "binder_rg", "binder_rg_ratio",
        "site_interface_fraction", "site_interface_binder_res", "site_interface_total_res",
        "site_centroid_dist_heavy", "site_centroid_dist_CA",
        "site_cos_angle",
        "refolding_rmsd",
        # Produced by validate_boltz()/run_proteina_complexa() but previously
        # absent from this list, so DictWriter(extrasaction="ignore") dropped them
        # silently. boltz_protein_iptm matters most: in holo mode it OVERRIDES
        # boltz_iptm, so the number that determined the ranking went unrecorded.
        "boltz_confidence", "boltz_protein_iptm",
        "boltz_complex_pde", "boltz_complex_ipde",
        "boltz_min_interface_pde", "boltz_mean_interface_pde",
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

    # BindCraft2-specific (bc2_ prefix). Separate from bc_ on purpose: the two
    # tools report different quantities under similar names (BC2's SS columns are
    # fractions, v1's are percentages; BC2 has no Rosetta dG at all), so merging
    # them into one column set would produce a CSV where a value's meaning depends
    # on which tool wrote the row.
    bc2_fields = [
        "bc2_i_pdae", "bc2_i_ptm", "bc2_i_pae", "bc2_ptm", "bc2_plddt",
        "bc2_unbound_binder_plddt", "bc2_target_plddt", "bc2_ss_plddt",
        "bc2_interface_residues", "bc2_interface_buriedarea",
        "bc2_interface_hydrophobicity", "bc2_backbone_clashes",
        "bc2_all_atom_clashes", "bc2_surface_hydrophobicity",
        "bc2_binder_helix_fraction", "bc2_binder_betasheet_fraction",
        "bc2_binder_loop_fraction", "bc2_binder_length", "bc2_binder_mass_kda",
        "bc2_binder_pi", "bc2_binder_net_charge", "bc2_binder_cysteines",
        "bc2_binder_disulfides", "bc2_binder_free_cysteines",
        "bc2_binder_chain_breaks", "bc2_binder_rmsd",
        "bc2_hotspot_contact_fraction", "bc2_off_epitope_contact_fraction",
        "bc2_epitope_residues_contacted", "bc2_interface_buriedarea_fraction",
        "bc2_target_rmsd", "bc2_interface_helix_fraction",
        "bc2_interface_betasheet_fraction", "bc2_interface_loop_fraction",
        "bc2_length",
        # provenance for bc2_plddt: "bfactor" when backfilled from coordinates
        # because the campaign's filter preset left pLDDT unmeasured. String,
        # so it is excluded from float_fields below.
        "bc2_plddt_source",
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
        "pc_total_reward", "pc_scrmsd_esmfold",
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

    all_fields = core_fields + rfd_fields + bg_fields + bc_fields + bc2_fields + px_fields + pc_fields + rosetta_fields + ss_fields + composition_fields

    # "tool" is in core_fields but was never assigned anywhere in either script,
    # so the column was blank in every rankings.csv ever written (verified 0/196,
    # 0/116, 0/90 non-null on real runs). binder_browser guards on the column
    # being PRESENT rather than populated, so its tool-derivation branch never
    # fired and the run selector always showed "tools: ?".
    for _d in designs:
        if not _d.get("tool"):
            _d["tool"] = _get_tool_from_design_id(_d.get("design_id", ""))
        # Declared in the header since forever, never assigned by either script.
        if not _d.get("binder_length"):
            _seq = _d.get("binder_sequence") or ""
            if _seq:
                _d["binder_length"] = len(_seq)

    float_fields = set(all_fields) - {"rank", "design_id", "binder_sequence",
                                       "native_score_name", "bg_liability_violations_summary",
                                       "px_cluster_id", "px_filter_level", "iptm_source",
                                       "site_geometric_pass", "site_geometry_status", "bc2_plddt_source"}

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


def cif_to_pdb_atoms(cif_path, pdb_path):
    """
    Convert an mmCIF to PDB by reading _atom_site directly. Returns atom count.

    Needed because the host `boltz` env pins **gemmi==0.6.5**, which returns ZERO
    models for BindCraft2's mmCIF (BC2 emits a valid but unusual column order,
    with B_iso_or_equiv and label_entity_id ahead of Cartn_x). gemmi 0.7.5 reads
    them fine, but upgrading is not an option: boltz requires the 0.6.5 pin, and
    boltz-2 is the validation backbone.

    The symptom was silent — `st.write_pdb()` produced a file containing only
    CRYST1 and END, so PLIP ran happily and reported zero interactions for every
    BindCraft2 design.

    Only ATOM/HETATM records are emitted; that is all PLIP and Rosetta need.
    """
    lines = Path(cif_path).read_text().splitlines()
    col, n, in_loop, start = {}, 0, False, None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "loop_":
            in_loop, col, n = False, {}, 0
        elif s.startswith("_atom_site."):
            in_loop = True
            col[s] = n
            n += 1
        elif in_loop and s and not s.startswith(("_", "#", "loop_", "data_")):
            start = i
            break
    if not col or start is None:
        return 0

    def c(name, alt=None):
        return col.get(f"_atom_site.{name}",
                       col.get(f"_atom_site.{alt}", -1) if alt else -1)

    i_group = c("group_PDB")
    i_sym   = c("type_symbol")
    i_atom  = c("auth_atom_id", "label_atom_id")
    i_comp  = c("auth_comp_id", "label_comp_id")
    i_asym  = c("auth_asym_id", "label_asym_id")
    i_seq   = c("auth_seq_id", "label_seq_id")
    i_x, i_y, i_z = c("Cartn_x"), c("Cartn_y"), c("Cartn_z")
    i_b     = c("B_iso_or_equiv")
    i_model = c("pdbx_PDB_model_num")
    if min(i_atom, i_comp, i_asym, i_seq, i_x, i_y, i_z) < 0:
        return 0
    need = max(x for x in (i_group, i_sym, i_atom, i_comp, i_asym, i_seq,
                           i_x, i_y, i_z, i_b, i_model) if x >= 0)

    out, serial, first_model = [], 0, None
    for line in lines[start:]:
        s = line.strip()
        if not s or s.startswith(("#", "_", "loop_", "data_")):
            break
        p = s.split()
        if len(p) <= need:
            continue
        if i_model >= 0:
            if first_model is None:
                first_model = p[i_model]
            elif p[i_model] != first_model:
                break  # only the first model
        serial += 1
        rec = (p[i_group] if i_group >= 0 else "ATOM")
        rec = "HETATM" if rec.upper().startswith("HETATM") else "ATOM"
        name = p[i_atom].strip('"')
        # PDB atom-name column convention: 1-3 char names start at col 14
        name_f = f" {name:<3s}" if len(name) < 4 else name[:4]
        elem = (p[i_sym] if i_sym >= 0 else name[:1]).strip('"')
        try:
            bf = float(p[i_b]) if i_b >= 0 else 0.0
        except ValueError:
            bf = 0.0
        try:
            resnum = int(p[i_seq])
        except ValueError:
            continue
        try:
            x, y, z = float(p[i_x]), float(p[i_y]), float(p[i_z])
        except ValueError:
            continue
        out.append(
            f"{rec:<6s}{serial:>5d} {name_f:<4s} {p[i_comp]:>3s} "
            f"{p[i_asym][:1]:>1s}{resnum:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{bf:>6.2f}"
            f"{'':>10s}{elem:>2s}")
    if not out:
        return 0
    out.append("END")
    Path(pdb_path).write_text("\n".join(out) + "\n")
    return serial


# Maximum solvent-accessible surface area per residue, Tien et al. 2013 (theoretical).
# Used to turn absolute SASA into a relative (0-1) exposure.
_MAX_ASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225,
    "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236,
    "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285,
    "TYR": 263, "VAL": 174,
}


def preview_site(target_path, chain_id, site_resnums, out_dir=None, render=True):
    """
    Check that a --site actually describes a plausible surface epitope, and say so
    in terms an agent can act on.

    The pipeline can already tell you whether site residues EXIST in the target.
    It cannot tell you whether they are the residues you meant — and numbering
    taken from a paper, UniProt, or a different PDB entry is routinely offset.
    A site that exists but is buried, or scattered across opposite faces of the
    protein, will happily consume hours of GPU designing against nothing.

    Three checks, each of which catches a real class of mistake:
      exposure   — buried residues cannot be an epitope (wrong numbering, or an
                   interior site)
      clustering — a real epitope is one contiguous patch, not residues spread
                   over the whole surface (off-by-N numbering)
      size       — 3-25 residues is a sensible epitope

    Writes site_preview.{pml,cxc} for a human, and a rendered PNG that an agent
    with vision can inspect directly. Returns a dict; never raises.
    """
    import math
    # `status` separates a SCIENTIFIC verdict ("this site is buried / not on the
    # surface") from an OPERATIONAL failure ("the check could not run"). Callers
    # mapped `ok` alone onto exit 1, so a missing biopython and a genuinely bad
    # epitope were indistinguishable — the agent is told to fix its --site when
    # the real problem is the install.
    #   ok           — site looks usable
    #   bad_site     — checked, and the site is wrong (bad input)
    #   cannot_check — the check itself could not run (tooling/input unreadable)
    result = {"ok": False, "status": "cannot_check", "warnings": [],
              "residues": [], "verdict": ""}
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
    except ImportError:
        result["verdict"] = "biopython not available — cannot compute exposure"
        result["status"] = "cannot_check"
        return result

    try:
        model = PDBParser(QUIET=True).get_structure("t", str(target_path))[0]
        if chain_id not in model:
            result["verdict"] = f"chain {chain_id} not in {target_path}"
            result["status"] = "bad_site"
            return result
        ShrakeRupley().compute(model, level="R")
        chain = model[chain_id]

        by_num = {r.id[1]: r for r in chain if r.id[0] == " "}
        missing = [r for r in site_resnums if r not in by_num]
        present = [r for r in site_resnums if r in by_num]
        if not present:
            result["verdict"] = "none of the site residues exist in this chain"
            result["status"] = "bad_site"
            return result

        rows, coords = [], []
        for rn in present:
            res = by_num[rn]
            name = res.get_resname()
            rel = res.sasa / _MAX_ASA.get(name, 200.0)
            ca = res["CA"].coord if "CA" in res else None
            if ca is not None:
                coords.append(ca)
            rows.append({"resnum": rn, "resname": name,
                         "sasa": round(res.sasa, 1), "rel_sasa": round(rel, 2),
                         "exposed": rel >= 0.20})
        result["residues"] = rows

        # --- exposure ---
        n_buried = sum(1 for r in rows if not r["exposed"])
        frac_exposed = 1.0 - n_buried / len(rows)

        # --- clustering: pairwise CA spread ---
        max_pair = 0.0
        if len(coords) > 1:
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = float(((coords[i] - coords[j]) ** 2).sum() ** 0.5)
                    max_pair = max(max_pair, d)
        centroid = ([float(sum(c[k] for c in coords) / len(coords)) for k in range(3)]
                    if coords else None)
        rg = (math.sqrt(sum(sum((c[k] - centroid[k]) ** 2 for k in range(3))
                            for c in coords) / len(coords)) if coords else 0.0)

        result.update({
            "chain": chain_id,
            "n_requested": len(site_resnums),
            "n_present": len(present),
            "missing": missing,
            "fraction_exposed": round(frac_exposed, 2),
            "n_buried": n_buried,
            "max_pairwise_ca_dist": round(max_pair, 1),
            "radius_of_gyration": round(rg, 1),
            "centroid": [round(x, 1) for x in centroid] if centroid else None,
            "chain_numbering": [min(by_num), max(by_num)],
        })

        if missing:
            result["warnings"].append(
                f"{len(missing)} residue(s) not in chain {chain_id}: {missing[:10]} "
                f"— they will be silently dropped downstream")
        if frac_exposed < 0.5:
            result["warnings"].append(
                f"{n_buried}/{len(rows)} site residues are BURIED (rel. SASA < 0.20). "
                f"A binder cannot reach them — check the numbering is right for THIS file "
                f"(chain {chain_id} spans {min(by_num)}-{max(by_num)})")
        elif n_buried:
            result["warnings"].append(
                f"{n_buried}/{len(rows)} site residue(s) are buried and contribute nothing")
        # Compactness, scaled to the number of residues. A tight 4-residue hotspot
        # sits at Rg ~4 Å; a 20-residue epitope legitimately reaches ~14. Judging
        # by absolute span alone lets a scattered 4-residue set (Rg 12 Å across
        # 28 Å) pass as "an epitope" when it is really several faces of the
        # protein. A miniprotein interface spans ~25-30 Å, so beyond that a single
        # binder physically cannot touch every residue.
        rg_expected = 4.0 + 0.5 * len(rows)
        result["rg_expected"] = round(rg_expected, 1)
        if rg > 1.8 * rg_expected or max_pair > 30:
            result["warnings"].append(
                f"site residues are SCATTERED: Rg {rg:.0f} Å across {max_pair:.0f} Å "
                f"for {len(rows)} residues (a compact patch this size sits near "
                f"Rg {rg_expected:.0f} Å). These are not one epitope — a binder "
                f"cannot contact them all. Check the numbering.")
        elif rg > 1.3 * rg_expected or max_pair > 22:
            result["warnings"].append(
                f"site is loose: Rg {rg:.0f} Å across {max_pair:.0f} Å "
                f"(compact would be ~{rg_expected:.0f} Å) — usable, but the binder "
                f"will only reach part of it")
        if len(present) > 25:
            result["warnings"].append(
                f"{len(present)} residues is large for an epitope (typical 4-20)")
        if len(present) < 3:
            result["warnings"].append(
                f"only {len(present)} residue(s) — too few to steer docking reliably")

        blocking = [w for w in result["warnings"] if "BURIED" in w or "SCATTERED" in w]
        result["ok"] = not blocking
        # We got far enough to judge the site, so this is a scientific verdict
        # either way — not an operational failure.
        result["status"] = "ok" if result["ok"] else "bad_site"
        result["verdict"] = (
            f"PLAUSIBLE surface epitope: {len(present)} residues, "
            f"{frac_exposed:.0%} exposed, spanning {max_pair:.0f} Å"
            if result["ok"] else
            "IMPLAUSIBLE as a binding site — see warnings; verify the numbering "
            "before spending GPU time")

        if out_dir:
            _write_site_preview_scenes(target_path, chain_id, present,
                                       Path(out_dir), render=render, result=result)
    except Exception as e:
        # Only downgrade if we had not already reached a verdict. Scene writing
        # runs AFTER the judgement, so a PyMOL/ChimeraX failure there must not
        # turn a completed "PLAUSIBLE surface epitope" into cannot_check.
        if result.get("status") in ("ok", "bad_site"):
            result.setdefault("warnings", []).append(
                f"site scenes could not be written: {e}")
        else:
            result["verdict"] = f"site preview failed: {e}"
            result["ok"] = False
            result["status"] = "cannot_check"
    return result


def _write_site_preview_scenes(target_path, chain_id, resnums, out_dir, render=True,
                               result=None):
    """PyMOL + ChimeraX scenes for the site, and a ray-traced PNG an agent can read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = "+".join(str(r) for r in resnums)
    stem = Path(target_path).stem

    pml = out_dir / "site_preview.pml"
    pml.write_text("\n".join([
        f"# Site preview — chain {chain_id}, residues {sel}",
        f"load {Path(target_path).resolve()}, {stem}",
        "hide everything", "bg_color white",
        "set two_sided_lighting, 0", "set specular, 0",
        "set cartoon_fancy_helices, 1",
        f"show surface, {stem}", f"color grey80, {stem}",
        f"color orange, {stem} and chain {chain_id} and resi {sel}",
        f"show sticks, {stem} and chain {chain_id} and resi {sel} and not name N+C+O",
        f"orient {stem} and chain {chain_id} and resi {sel}", "zoom all, 5", "",
    ]) + "\n")

    cxc = out_dir / "site_preview.cxc"
    cxc.write_text("\n".join([
        f"# Site preview — chain {chain_id}, residues {sel}",
        "set bgColor white", f"open {Path(target_path).resolve()}",
        "surface #1", "color #1 #cccccc target s",
        f"color #1/{chain_id}:{sel.replace('+', ',')} #ff8000 target s",
        f"show #1/{chain_id}:{sel.replace('+', ',')} atoms",
        f"style #1/{chain_id}:{sel.replace('+', ',')} stick",
        "hide H atoms", "hide pseudobonds", "lighting soft",
        f"view #1/{chain_id}:{sel.replace('+', ',')}", "",
    ]) + "\n")

    png = out_dir / "site_preview.png"
    if render:
        try:
            import pymol
            pymol.finish_launching(["pymol", "-qc"])
            from pymol import cmd
            cmd.reinitialize()
            cmd.load(str(target_path), stem)
            cmd.hide("everything")
            cmd.bg_color("white")
            cmd.set("two_sided_lighting", 0)
            cmd.set("specular", 0)
            cmd.set("ray_opaque_background", 1)
            cmd.show("surface", stem)
            cmd.color("grey80", stem)
            site = f"{stem} and chain {chain_id} and resi {sel}"
            cmd.color("orange", site)
            cmd.orient(site)
            cmd.zoom(stem, 3)
            cmd.png(str(png), width=900, height=700, ray=1)
            cmd.sync()
            if result is not None and png.exists() and png.stat().st_size:
                result["png"] = str(png)
        except Exception as e:
            if result is not None:
                result.setdefault("warnings", []).append(f"PNG render failed: {e}")
    if result is not None:
        result["pml"] = str(pml)
        result["cxc"] = str(cxc)


def render_site_preview_text(rep, target, site):
    """Terminal rendering of preview_site() — a compact exposure map plus the verdict."""
    L = []
    L.append("=" * 66)
    L.append(f"SITE PREVIEW  {Path(target).name}   site={site}")
    L.append("=" * 66)
    if not rep.get("residues"):
        L.append(f"  {rep.get('verdict', 'no data')}")
        return "\n".join(L)

    lo, hi = rep.get("chain_numbering", (None, None))
    L.append(f"  chain {rep['chain']}  (numbering {lo}-{hi})   "
             f"{rep['n_present']}/{rep['n_requested']} site residues found")
    L.append("")
    L.append("  residue      SASA   rel   exposure")
    L.append("  " + "-" * 52)
    for r in rep["residues"]:
        # 20 cells of relative exposure; a buried residue reads as an empty bar
        filled = max(0, min(20, int(round(r["rel_sasa"] * 20))))
        bar = "#" * filled + "." * (20 - filled)
        flag = "" if r["exposed"] else "   <-- BURIED"
        L.append(f"  {r['resname']} {r['resnum']:<6d} {r['sasa']:>6.1f} "
                 f"{r['rel_sasa']:>5.2f}  {bar}{flag}")
    L.append("")
    L.append(f"  exposed        : {rep['fraction_exposed']:.0%} "
             f"({rep['n_buried']} buried of {rep['n_present']})")
    L.append(f"  patch spread   : {rep['max_pairwise_ca_dist']:.0f} Å max CA-CA, "
             f"Rg {rep['radius_of_gyration']:.0f} Å")
    if rep.get("centroid"):
        L.append(f"  centroid       : {rep['centroid']}")
    for w in rep.get("warnings", []):
        L.append(f"  WARNING: {w}")
    L.append("")
    L.append(f"  VERDICT: {rep['verdict']}")
    for k, label in (("png", "image "), ("pml", "pymol "), ("cxc", "chimerax")):
        if rep.get(k):
            L.append(f"  {label} : {rep[k]}")
    L.append("=" * 66)
    return "\n".join(L)


def _resolve_run_target(results_dir, chain_id="A"):
    """
    Find the target structure of a finished run and return (path, sequence).

    Tried in order: status.json's recorded target, the sanitized copy the run
    writes next to rankings.csv, then the copy placed in top_designs/. Returns
    ("", "") if none is readable — callers must handle that, not assume.
    """
    results_dir = Path(results_dir)
    candidates = []
    st = results_dir / "status.json"
    if st.exists():
        try:
            d = json.loads(st.read_text())
            if d.get("target"):
                candidates.append(Path(d["target"]))
            if d.get("target_chain"):
                chain_id = d["target_chain"]
        except Exception:
            pass
    candidates += sorted(results_dir.glob("*_clean.pdb"))
    candidates += sorted((results_dir / "top_designs").glob("target_*.pdb"))
    for c in candidates:
        try:
            if c.exists():
                seq = pdb_chain_seq(c, chain_id)
                if seq:
                    return str(c), seq
        except Exception:
            continue
    return "", ""


def _pymol_chain_seqs(cmd, obj):
    """One-letter sequence per chain, read from PyMOL so the chain IDs are exactly
    the ones we will colour. Parsing the file separately risks label_asym_id vs
    auth_asym_id disagreeing on multi-character chains."""
    out = {}
    for ch in cmd.get_chains(obj):
        try:
            fa = cmd.get_fastastr(f"{obj} and chain {ch} and polymer.protein")
        except Exception:
            continue
        seq = "".join(l.strip() for l in fa.splitlines() if l and not l.startswith(">"))
        if seq:
            out[ch] = seq
    return out


def render_designs(results_dir, out_dir=None, top=6, site_resnums=None,
                   chain_id="A", width=900, height=700):
    """
    Ray-trace the top designed complexes to PNG so an agent can SEE the result.

    The pipeline already writes .pml and .cxc scenes, but those assume a human at
    a GUI — on a headless box an agent finishes a run, reads numbers, and can
    neither look at what it produced nor show it to anyone. --preview_site renders
    the INPUT (the epitope); this renders the OUTPUT, which is what the user
    actually asked for.

    Convention matches the .pml/.cxc scenes: target grey cartoon, epitope orange
    sticks, binder coloured per tool.

    Returns a list of {design_id, rank, png, score, ...}; never raises.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "renders"
    rankings = results_dir / "rankings.csv"
    if not rankings.exists():
        return {"ok": False, "error": f"no rankings.csv in {results_dir}", "images": []}

    rows = []
    try:
        with open(rankings) as f:
            for r in csv.DictReader(f):
                if r.get("rank") and r["rank"].strip():
                    rows.append(r)
    except Exception as e:
        return {"ok": False, "error": f"could not read rankings.csv: {e}", "images": []}
    if not rows:
        return {"ok": False, "error": "no RANKED designs in rankings.csv "
                                      "(rank column blank for every row)", "images": []}
    rows.sort(key=lambda r: int(r["rank"]))
    rows = rows[:top]

    top_dir = results_dir / "top_designs"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []

    try:
        import pymol
        pymol.finish_launching(["pymol", "-qc"])
        from pymol import cmd
    except Exception as e:
        return {"ok": False, "error": f"pymol unavailable: {e}", "images": []}

    sel_site = "+".join(str(r) for r in (site_resnums or []))
    target_path, target_seq = _resolve_run_target(results_dir, chain_id)
    if not target_seq:
        log("  render: could not read the run's target sequence — binder-chain "
            "identification falls back to sequence match only.", err=True)

    # Where the epitope sits POSITIONALLY in the target chain. Boltz-2 (and Chai,
    # and AF3) hand back every chain renumbered 1..N, so a site given in the
    # target's own numbering — e.g. PDIA6 "A:373-383" on a construct numbered
    # 270-419 — selects nothing at all in the predicted complex. The old code
    # then drew no epitope and said nothing, while the summary line underneath
    # still claimed "epitope orange sticks": an agent reading the PNG would see
    # no orange and conclude the binder misses the site, which is the opposite
    # of what a site_contact_fraction of 1.00 means.
    target_resnums = pdb_chain_resnums(target_path, chain_id) if target_path else []
    site_pos = [target_resnums.index(r) for r in (site_resnums or [])
                if r in target_resnums]
    if site_resnums and not site_pos:
        log(f"  render: none of the site residues {sel_site} are in chain "
            f"{chain_id} of the run's target — the epitope cannot be located "
            f"positionally either.", err=True)

    for row in rows:
        did = row.get("design_id", "")
        rank = row.get("rank", "?")
        # the structure copied into top_designs/ is named rank{NNN}_{design_id}
        hits = sorted(top_dir.glob(f"rank*{did}.*")) if top_dir.is_dir() else []
        struct = next((h for h in hits if h.suffix in (".cif", ".pdb")), None)
        if struct is None:
            # fall back to whatever path the record carries
            for k in ("complex_cif", "binder_pdb"):
                v = row.get(k)
                if v and Path(v).exists():
                    struct = Path(v); break
        if struct is None:
            continue

        png = out_dir / f"rank{int(rank):03d}_{did}.png"
        try:
            cmd.reinitialize()
            cmd.load(str(struct), "cx")
            cmd.hide("everything")
            cmd.bg_color("white")
            cmd.set("two_sided_lighting", 0)
            cmd.set("specular", 0)
            cmd.set("cartoon_fancy_helices", 1)
            cmd.set("ray_opaque_background", 1)
            cmd.show("cartoon", "cx")

            chains = cmd.get_chains("cx")
            chain_seqs = _pymol_chain_seqs(cmd, "cx")
            seq = (row.get("binder_sequence", "") or "").strip()
            binder_ch = None
            how = ""

            # 1. Exact sequence match. The designed sequence is recorded in
            #    rankings.csv, so this is definitive when it works.
            if seq:
                for ch, s in chain_seqs.items():
                    if s == seq:
                        binder_ch, how = ch, "exact sequence"
                        break

            # 2. Least identical to the TARGET. Length-agnostic, which matters:
            #    the old fallback here matched on chain LENGTH and then on
            #    "shortest chain", the same heuristic that made collect_boltzgen
            #    score the target against itself whenever a binder was longer
            #    than its target.
            if binder_ch is None and target_seq and chain_seqs:
                info = {ch: {"sequence": s, "length": len(s)}
                        for ch, s in chain_seqs.items()}
                picked = pick_binder_chain_by_identity(info, target_seq)
                pseq = picked.get("sequence") if picked else None
                if pseq:
                    binder_ch = next((c for c, s in chain_seqs.items() if s == pseq), None)
                    how = "least identical to target"

            # 3. Shortest chain — last resort, and say so, because this is the
            #    one that is wrong for a binder longer than its target.
            if binder_ch is None and chain_seqs:
                binder_ch = min(chain_seqs, key=lambda c: len(chain_seqs[c]))
                how = "shortest chain (UNRELIABLE)"
                log(f"  render: {did}: binder chain guessed by length — no designed "
                    f"sequence in rankings.csv and no target sequence available. "
                    f"Colours may be wrong if the binder is longer than the target.",
                    err=True)

            tool = row.get("tool") or _get_tool_from_design_id(did)
            colour = PYMOL_TOOL_COLORS.get(tool, "marine")
            cmd.color("grey80", "cx")
            if binder_ch:
                cmd.color(colour, f"cx and chain {binder_ch}")
            epitope_shown = False
            if sel_site:
                tgt_chains = [c for c in chains if c != binder_ch] or list(chains)
                tgt = " or ".join(f"chain {c}" for c in tgt_chains)
                site_sel = f"cx and ({tgt}) and resi {sel_site}"

                # Fall back to positional lookup when the tool renumbered the
                # target. Guarded on an exact residue-count match so a wrong
                # chain can never be silently decorated.
                if not cmd.count_atoms(site_sel) and site_pos and target_resnums:
                    for c in tgt_chains:
                        nums = _pymol_ordered_resnums(cmd, "cx", c)
                        if len(nums) == len(target_resnums):
                            site_sel = ("cx and chain %s and resi %s"
                                        % (c, "+".join(str(nums[i]) for i in site_pos)))
                            break

                if cmd.count_atoms(site_sel):
                    cmd.color("orange", site_sel)
                    cmd.show("sticks", f"{site_sel} and not name N+C+O")
                    epitope_shown = True
                else:
                    log(f"  render: {did}: EPITOPE NOT DRAWN — site {sel_site} "
                        f"matched no atoms in {struct.name} and could not be "
                        f"remapped. There is no orange in this image; that is a "
                        f"rendering gap, NOT evidence the binder misses the site.",
                        err=True)
            cmd.orient("cx")
            cmd.zoom("cx", 3)
            cmd.png(str(png), width=width, height=height, ray=1)
            cmd.sync()
            if png.exists() and png.stat().st_size:
                images.append({
                    "rank": int(rank), "design_id": did, "tool": tool,
                    "binder_chain": binder_ch, "binder_chain_by": how,
                    "epitope_shown": epitope_shown,
                    "png": str(png), "structure": str(struct),
                    "combined_score": row.get("combined_score", ""),
                    "iptm": row.get("boltz_iptm") or row.get("_resolved_iptm", ""),
                    "binder_length": row.get("binder_length", ""),
                })
        except Exception as e:
            log(f"  render: {did} failed: {e}", err=True)

    return {"ok": bool(images), "n": len(images), "out_dir": str(out_dir),
            "images": images,
            "error": None if images else "no design could be rendered"}


def render_designs_text(rep):
    L = ["=" * 66, "RENDERED DESIGNS", "=" * 66]
    if not rep.get("ok"):
        L.append(f"  FAILED: {rep.get('error')}")
        L.append("=" * 66)
        return "\n".join(L)
    L.append(f"  {rep['n']} image(s) in {rep['out_dir']}")
    L.append("")
    L.append("  rank  design_id                 tool             score   PNG")
    L.append("  " + "-" * 62)
    for im in rep["images"]:
        try:
            sc = f"{float(im['combined_score']):.3f}"
        except Exception:
            sc = "?"
        L.append(f"  {im['rank']:>4d}  {im['design_id']:<24s}  {im['tool']:<15s} {sc:>6s}  "
                 f"{Path(im['png']).name}")
    L.append("")
    blind = [im for im in rep["images"] if not im.get("epitope_shown", True)]
    if blind:
        L.append(f"  WARNING: {len(blind)}/{rep['n']} image(s) have NO epitope drawn "
                 f"— the site residues could not be located in those structures.")
        L.append("  For those, absence of orange says nothing about where the binder sits;")
        L.append("  judge them by site_contact_fraction in rankings.csv instead.")
        L.append("")
    L.append("  Target grey, epitope orange sticks, binder coloured by tool.")
    L.append("  These are PNGs — read them directly, or show them to the user.")
    L.append("=" * 66)
    return "\n".join(L)


def run_self_test(tools=None, deep=True):
    """
    Actually EXERCISE each tool, instead of checking that its directory exists.

    `--capabilities` reports `available: true` from `Path(env).is_dir()` alone.
    That is necessary but nowhere near sufficient, and every real defect found on
    this machine was invisible to it:
      - `vina` absent from the host env, so molecule docking silently scored 0.0
        for 29,295 molecules across two completed runs
      - `unidock` configured in config.yaml but never installed
      - DATA_ROOT pointing at an unmounted disk

    So each check here imports the module, constructs the object, or invokes the
    binary. Slower (tens of seconds) and worth it: this is what an agent should
    run before trusting a box, and what a human should run after an install.

    Returns a dict; `ok` is true only when nothing FAILED.
    """
    import shutil as _sh
    checks = []

    def _add(name, category, ok, detail, fatal=True):
        checks.append({"name": name, "category": category,
                       "status": "OK" if ok else ("FAIL" if fatal else "WARN"),
                       "detail": detail})

    def _py(env_path):
        p = Path(env_path) / "bin" / "python"
        return str(p) if p.exists() else None

    def _run(cmd, timeout=120):
        try:
            # Probes import jax/torch. Without a preallocation cap jax grabs ~75%
            # of VRAM the moment it is imported, so running --self_test or
            # --capabilities --deep beside a live design run could OOM it.
            _penv = {**os.environ,
                     "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                     "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, env=_penv)
            return r.returncode, (r.stdout or "") + (r.stderr or "")
        except Exception as e:
            return -1, str(e)

    # ---- host env: the one everything else runs inside -------------------
    for mod, why, fatal in (
            ("boltz", "Boltz-2 validation backbone", True),
            ("gemmi", "structure I/O", True),
            ("numpy", "everything", True),
            ("torch", "GPU", True),
            ("plip.structure.preparation", "interaction analysis", False),
            ("vina", "molecule docking (CPU fallback)", False),
            ("meeko", "ligand prep for docking", False),
            ("onnxruntime", "NetSolP solubility", False),
            ("Bio.PDB.SASA", "--preview_site exposure", False)):
        rc, out = _run([sys.executable, "-c", f"import {mod}"], timeout=90)
        _add(f"host:{mod}", "host_env", rc == 0,
             why if rc == 0 else out.strip().splitlines()[-1][:160], fatal=fatal)

    rc, out = _run([sys.executable, "-c",
                    "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"],
                   timeout=180)
    _add("host:cuda", "host_env", rc == 0 and out.strip().startswith("True"),
         out.strip()[:80] if rc == 0 else out.strip()[:160])

    # ---- per-tool envs: import the thing the tool actually needs ---------
    ENV_PROBE = {
        "bindcraft":  ("bindcraft",  "import pyrosetta, gemmi",
                       "PyRosetta+gemmi — used for EVERY tool's Rosetta scoring"),
        "bindcraft2": ("bindcraft2", "import bindcraft, jax", "BindCraft2 + jax"),
        "boltzgen":   ("boltzgen",   "import boltzgen", "BoltzGen"),
        "rfdiffusion":  ("rfdiffusion",  "import torch", "RFdiffusion"),
        "rfdiffusion3": ("rfdiffusion3", "import torch", "RFdiffusion3"),
        "pxdesign":   ("pxdesign",   "import torch", "PXDesign"),
        "proteina":   ("proteina",   "import torch", "Proteina"),
    }
    want = set(tools) if tools else set(ENV_PROBE) | {"proteina_complexa"}
    for tool, (key, probe, why) in ENV_PROBE.items():
        if tool not in want:
            continue
        try:
            # Resolve a bare env NAME to its prefix: _py() builds <env>/bin/python,
            # which is meaningless for a name.
            env = cfg.conda_env_prefix(key) or cfg.conda_env(key)
        except Exception as e:
            _add(f"tool:{tool}", "tools", False, f"config lookup failed: {e}")
            continue
        py = _py(env)
        if not py:
            _add(f"tool:{tool}", "tools", False, f"no python in env: {env}")
            continue
        if not deep:
            _add(f"tool:{tool}", "tools", True, f"env present: {env}")
            continue
        rc, out = _run([py, "-c", probe], timeout=180)
        _add(f"tool:{tool}", "tools", rc == 0,
             why if rc == 0 else out.strip().splitlines()[-1][:160])

    if "proteina_complexa" in want:
        venv = PROTEINA_COMPLEXA_VENV
        py = _py(venv) if venv else None
        _add("tool:proteina_complexa", "tools", bool(py),
             f"UV venv python present" if py else f"UV venv missing: {venv}")

    # ---- conda itself -----------------------------------------------------
    # Every tool runs as `conda run ...`, but self_test only ever invoked
    # <env>/bin/python directly, so a broken or absent conda could not fail it —
    # the check passed and the real run died at the first tool.
    try:
        _cr = subprocess.run(["conda", "run", "--no-capture-output",
                              *cfg.conda_run_args("boltz"),
                              "python", "-c", "import sys; print(sys.prefix)"],
                             capture_output=True, text=True, timeout=120)
        _add("host:conda_run", "host_env", _cr.returncode == 0,
             f"`conda run` works ({(_cr.stdout or '').strip()[:60]})"
             if _cr.returncode == 0 else
             f"`conda run` FAILED rc={_cr.returncode}: "
             f"{((_cr.stderr or '') + (_cr.stdout or '')).strip()[-200:]}")
    except FileNotFoundError:
        _add("host:conda_run", "host_env", False,
             "conda not on PATH — every tool launch would fail")
    except Exception as _e:
        _add("host:conda_run", "host_env", False, f"conda run check failed: {_e}")

    # ---- weights: existence AND non-trivial size ------------------------
    # Covers every tool that has weights. Only 3 of 8 were checked, so a
    # half-installed box passed --self_test and failed hours into a real run.
    for label, path in (("bindcraft2", f"{_WEIGHTS_DIR}/bindcraft2/alphafold"),
                        ("rfdiffusion", str(RFDIFFUSION_CKPT)),
                        ("proteina", str(PROTEINA_CKPT)),
                        ("ligandmpnn", str(LIGANDMPNN_CKPT)),
                        ("rfdiffusion3", str(RFD3_WEIGHTS)),
                        # BindCraft v1 keeps its AF2 params inside its own repo
                        # dir, not under weights_dir.
                        ("bindcraft", f"{BINDCRAFT_DIR}/params")):
        p = Path(path)
        if p.is_dir():
            n = sum(f.stat().st_size for f in p.glob("*") if f.is_file())
            _add(f"weights:{label}", "weights", n > 10_000_000,
                 f"{n/1e9:.1f} GB in {path}" if n else f"EMPTY: {path}")
        else:
            _add(f"weights:{label}", "weights", p.is_file() and p.stat().st_size > 1_000_000,
                 f"{p.stat().st_size/1e6:.0f} MB" if p.is_file() else f"missing: {path}")

    # ---- molecule pipeline: the path that failed silently ---------------
    rc, out = _run([sys.executable, "-c",
                    "from vina import Vina; from meeko import MoleculePreparation; "
                    "Vina(sf_name='vina', verbosity=0)"], timeout=90)
    _add("molecules:vina_cpu", "molecules", rc == 0,
         "CPU Vina docking usable" if rc == 0 else
         "vina/meeko NOT importable in host env — molecule scores would be "
         "renormalised without the 40% docking term", fatal=False)
    try:
        ud = cfg.conda_env_prefix("unidock") or cfg.conda_env("unidock")
        _add("molecules:unidock_gpu", "molecules", bool(ud) and Path(ud).is_dir(),
             "GPU docking available" if ud and Path(ud).is_dir() else
             f"unidock env missing ({ud}) — falls back to CPU Vina", fatal=False)
    except Exception:
        _add("molecules:unidock_gpu", "molecules", False,
             "unidock not configured", fatal=False)

    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    return {"schema": 1, "ok": n_fail == 0,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_checks": len(checks), "n_failed": n_fail, "n_warnings": n_warn,
            "checks": checks}


def render_self_test_text(rep):
    L = ["=" * 70, "SELF TEST — each check EXECUTES the thing, not just stat()s it", "=" * 70]
    cat = None
    for c in rep["checks"]:
        if c["category"] != cat:
            cat = c["category"]
            L.append(f"\n  [{cat}]")
        mark = {"OK": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}[c["status"]]
        L.append(f"  {mark} {c['name']:<32s} {c['detail'][:70]}")
    L.append("")
    L.append(f"  {rep['n_checks']} checks: {rep['n_checks']-rep['n_failed']-rep['n_warnings']} ok, "
             f"{rep['n_warnings']} warn, {rep['n_failed']} FAILED")
    L.append(f"  VERDICT: {'HEALTHY' if rep['ok'] else 'BROKEN — see FAIL lines above'}")
    L.append("=" * 70)
    return "\n".join(L)


def report_capabilities(deep=False):
    """
    Emit a machine-readable description of what this machine can do, as JSON.

    Exists so an agent can DISCOVER the pipeline instead of being told about it.
    Without this, answering "can this box run BindCraft2, and how big a target
    fits?" means grepping Python constants and config.yaml and knowing which
    conda env maps to which tool — every one of which drifts.

    Everything here is read from the live code and the live filesystem, never
    from documentation, so it cannot go stale.
    """
    import shutil as _sh

    def _gpus():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
            out = []
            for line in r.stdout.strip().splitlines():
                i, name, tot, used = [x.strip() for x in line.split(",")]
                out.append({"index": int(i), "name": name,
                            "memory_total_mb": int(tot), "memory_used_mb": int(used)})
            return out
        except Exception:
            return []

    ENV_KEY = {"rfdiffusion": "rfdiffusion", "rfdiffusion3": "rfdiffusion3",
               "boltzgen": "boltzgen", "bindcraft": "bindcraft",
               "bindcraft2": "bindcraft2", "pxdesign": "pxdesign",
               "proteina": "proteina"}
    tools = {}
    for tool in ALL_TOOLS:
        entry = {"available": False, "reason": None, "env": None,
                 "kind": ("iptm_native" if tool in IPTM_NATIVE_TOOLS
                          else "backbone_only" if tool in BACKBONE_ONLY_TOOLS else "other"),
                 "accepts_cif_target": tool not in ("rfdiffusion", "bindcraft")}
        try:
            if tool == "proteina_complexa":
                venv = PROTEINA_COMPLEXA_VENV
                entry["env"] = str(venv) if venv else None
                if venv and Path(venv).is_dir():
                    entry["available"] = True
                else:
                    entry["reason"] = f"UV venv not found: {venv}"
            else:
                env = cfg.conda_env_prefix(ENV_KEY[tool]) or cfg.conda_env(ENV_KEY[tool])
                entry["env"] = env
                if not env:
                    entry["reason"] = f"no conda_envs.{ENV_KEY[tool]} in config.yaml"
                elif not os.path.isabs(env):
                    # A bare NAME that conda_env_prefix() could not resolve does
                    # not exist. The old test was `isabs(env) and not is_dir()`,
                    # so a name fell straight through and the tool was reported
                    # AVAILABLE — and config.example.yaml ships names, so a fresh
                    # clone reported every tool present.
                    entry["reason"] = f"conda env '{env}' not found (no such name, not a path)"
                elif os.path.isabs(env) and not Path(env).is_dir():
                    entry["reason"] = f"conda env missing: {env}"
                elif os.path.isabs(env) and not (Path(env) / "bin" / "python").exists():
                    # A directory can exist and be an empty or half-built env.
                    entry["reason"] = f"env has no python interpreter: {env}"
                else:
                    entry["available"] = True
        except Exception as e:
            entry["reason"] = f"config lookup failed: {e}"
        tools[tool] = entry

    if deep:
        # Ask the self-test, which imports modules and constructs objects, and
        # let its verdict override the directory check.
        try:
            st = run_self_test(tools=[k for k in tools], deep=True)
            for c in st["checks"]:
                if not c["name"].startswith("tool:"):
                    continue
                name = c["name"].split(":", 1)[1]
                if name in tools:
                    tools[name]["available"] = (c["status"] == "OK")
                    tools[name]["reason"] = None if c["status"] == "OK" else c["detail"]
                    tools[name]["verified"] = True
        except Exception as e:
            log(f"  (deep capability check failed: {e})", err=True)

    gpus = _gpus()
    vram_gb = min((g["memory_total_mb"] for g in gpus), default=0) / 1024.0

    caps = {
        "schema": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve()),
        "host_python": sys.executable,
        "tools": tools,
        "tools_available": sorted(k for k, v in tools.items() if v["available"]),
        "tools_unavailable": sorted(k for k, v in tools.items() if not v["available"]),
        "modes": {m: {"per_tool": c, "total_all_tools": sum(c.values())}
                  for m, c in DESIGN_MODES.items()},
        "default_mode": DEFAULT_MODE,
        "ss_bias_choices": ["balanced", "helix", "beta"],
        "bindcraft2": {"modalities": BINDCRAFT2_MODALITIES,
                       "filter_presets": list(BINDCRAFT2_FILTER_PRESETS)},
        "failure_kinds": list(FAILURE_KINDS),
        "exit_codes": {"0": "success", "1": "rejected before compute or unusable",
                       "2": "malformed command line (argparse)",
                       "3": "ran but nothing could be ranked",
                       "4": "partial success: ranked some, >=1 tool failed"},
        "gpus": gpus,
        "vram_gb": round(vram_gb, 1),
        "limits": {
            # BindCraft2 documents peak as 2.0*(3.4GB + 38kB*N^2)
            "bindcraft2_max_total_n": (int(((max(vram_gb / 2.0 - 3.4, 0.0)) / 38e-6) ** 0.5)
                                       if vram_gb else None),
            # rfdiffusion3 only. Proteina Complexa IGNORES this threshold on cards
            # under 23 GiB: run_proteina_complexa() forces batch_size=1 at ANY N,
            # because the ~16 GB of weights plus the JAX/AF2 reward model leave no
            # room for pair-rep at batch_size>1 (OOMs even at N≈296). Reporting 340
            # for Complexa here told an agent asking --capabilities the opposite of
            # what the code does. See memory feedback_complexa_batch_size.md.
            "batch_size_1_above_total_n": 340,
            "complexa_forced_batch_size_1": bool(vram_gb and vram_gb < 23.0),
            "pxdesign_autocrop_above_target_len": 300,
            "boltz_complex_may_oom_above_n": 700,
        },
        "site_format": "A:8-30,120-123 (single-letter chain ids only)",
        "length_format": "60-80 or 70; 10..500 residues",
        "paths": {"software_dir": str(_SOFTWARE_DIR), "weights_dir": str(_WEIGHTS_DIR)},
        # available == the env DIRECTORY exists. It does not import anything, load a
        # checkpoint, or invoke a binary — every real defect found on this box was
        # invisible to that. --self_test is the one that executes things.
        "check_depth": "deep" if deep else "shallow",
        "available_means": (
            "the tool's module actually imports in its env (verified by execution)"
            if deep else
            "conda env exists and has a python interpreter — NOT a functional check. "
            "Re-run with --capabilities --deep, or use --self_test, to verify by execution."),
        "deep_check_command": "generate_binders.py --capabilities --deep",
        "helpers": {
            "plip": bool(_sh.which("plip") or (Path(sys.executable).parent / "plip").exists()),
        },
    }
    return caps


def preflight_tools(tools, dry_run=False):
    """
    Check each requested tool's conda env exists before the run starts.

    Without this, `--tools pxdesign` on a box with no PXDesign is accepted, burns
    the whole tool loop, retries once, and reports a `conda run` error hours in.
    An agent needs "this box cannot do that" up front, not after the GPU time.

    Returns {tool: reason} for tools that cannot run. Env presence is necessary,
    not sufficient — a broken env still fails later, but classified as
    missing_env.
    """
    # tool -> conda_envs key. Complexa uses a UV venv, not a conda env.
    ENV_KEY = {
        "rfdiffusion": "rfdiffusion", "rfdiffusion3": "rfdiffusion3",
        "boltzgen": "boltzgen", "bindcraft": "bindcraft",
        "bindcraft2": "bindcraft2", "pxdesign": "pxdesign",
        "proteina": "proteina",
    }
    problems = {}
    for tool in tools:
        try:
            if tool == "proteina_complexa":
                venv = PROTEINA_COMPLEXA_VENV
                if not venv or not Path(venv).is_dir():
                    problems[tool] = f"UV venv not found: {venv}"
                continue
            key = ENV_KEY.get(tool)
            if not key:
                continue
            env = cfg.conda_env_prefix(key) or cfg.conda_env(key)
            if not env:
                problems[tool] = f"no conda_envs.{key} in config.yaml"
            elif not os.path.isabs(env):
                problems[tool] = f"conda env '{env}' not found (no such name, not a path)"
            elif os.path.isabs(env) and not Path(env).is_dir():
                problems[tool] = f"conda env missing: {env}"
        except Exception as e:  # config lookup problems are themselves a finding
            problems[tool] = f"config lookup failed: {e}"

    if problems:
        for tool, why in sorted(problems.items()):
            log(f"  PREFLIGHT: {tool} cannot run — {why}", err=True)
    else:
        log(f"  Preflight OK: {len(tools)} tool env(s) present")
    return problems


def pdb_chain_seq(path, cid):
    """One-letter sequence of `cid` in a PDB/mmCIF, via gemmi. "" on any failure."""
    try:
        import gemmi
        st = gemmi.read_structure(str(path))
        names = [c.name for c in st[0]]
        ch = st[0][cid] if cid in names else st[0][0]
        out = []
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            if info and info.is_amino_acid():
                out.append(info.one_letter_code.upper())
        return "".join(out)
    except Exception:
        return ""


def pdb_chain_resnums(path, cid):
    """
    Ordered residue NUMBERS of `cid`, index-aligned with `pdb_chain_seq(path, cid)`.

    Must stay a mirror of that function (same gemmi calls, same amino-acid test),
    because callers rely on position i here meaning position i there.
    """
    try:
        import gemmi
        st = gemmi.read_structure(str(path))
        names = [c.name for c in st[0]]
        ch = st[0][cid] if cid in names else st[0][0]
        out = []
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            if info and info.is_amino_acid():
                out.append(int(r.seqid.num))
        return out
    except Exception:
        return []


def _pymol_ordered_resnums(cmd, obj, chain):
    """Residue numbers of one chain in file order, de-duplicated, CA-only."""
    raw = []
    try:
        cmd.iterate(f"{obj} and chain {chain} and polymer.protein and name CA",
                    "raw.append(resi)", space={"raw": raw})
    except Exception:
        return []
    out = []
    for r in raw:
        try:
            v = int(r)
        except (TypeError, ValueError):
            continue
        if not out or out[-1] != v:
            out.append(v)
    return out


def pick_binder_chain_by_identity(chain_info, target_seq, min_len=20):
    """
    Pick the binder chain as the protein chain LEAST identical to the known target.

    Length-agnostic on purpose. The legacy "binder = shortest chain" heuristic
    silently grabs the TARGET whenever the binder is longer than it (e.g. a 200-250
    aa binder against a 166 aa HRAS), so the design gets scored against itself.
    That was fixed in run_boltzgen on 2026-06-17 — but the fix never reached
    rerank's collect_boltzgen, which kept the shortest-chain pick. Module-level so
    there is exactly one implementation for both paths to share.

    Falls back to shortest-chain only when the target sequence is unavailable.
    Returns the chain's info dict, or {} when there is nothing to pick.
    """
    if not chain_info:
        return {}
    # `or chain_info` meant a peptide binder shorter than min_len emptied the
    # pool and put the TARGET back in it — so --length 10-19 returned the target
    # chain as the binder. Only apply the length floor when it still leaves a
    # choice; otherwise keep every chain that has a sequence.
    _long = {k: v for k, v in chain_info.items()
             if v.get("sequence") and v.get("length", 0) >= min_len}
    _any = {k: v for k, v in chain_info.items() if v.get("sequence")}
    pool = _long if len(_long) >= 2 else _any
    if not pool:
        return {}

    def _ident(seq):
        if not seq or not target_seq:
            return 0.0
        n = min(len(seq), len(target_seq))
        return (sum(a == b for a, b in zip(seq[:n], target_seq[:n])) / n) if n else 0.0

    if target_seq:
        return min(pool.values(), key=lambda v: _ident(v.get("sequence", "")))
    return min(pool.values(), key=lambda v: v.get("length", 0))


def apply_post_rank_filter(all_ranked, key, threshold, higher_is_better, label,
                           fmt="{:.2f}"):
    """
    Drop already-ranked designs failing a metric threshold, then renumber.

    Post-rank because these metrics are geometric properties of the final pose:
    a design is scored first, then removed if it docked in the wrong place.
    Removal = combined_score -> NaN and rank -> None, which is how the rest of the
    pipeline marks "present in rankings.csv but not ranked".

    NaN passes, per the project-wide filter policy (a design with no measurement
    is not penalised) — so the drop count is logged to make a filter that silently
    matched nothing distinguishable from one that had nothing to measure.

    Returns (n_removed, n_nan).
    """
    before = sum(1 for d in all_ranked if d.get("rank") is not None)
    n_nan = 0
    for d in all_ranked:
        if d.get("rank") is None:
            continue
        v = d.get(key, float("nan"))
        if v != v:
            n_nan += 1
            continue
        if (v < threshold) if higher_is_better else (v > threshold):
            d["combined_score"] = float("nan")
            d["rank"] = None

    ranked = [d for d in all_ranked if d.get("rank") is not None]
    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    for i, d in enumerate(ranked):
        d["rank"] = i + 1

    n_removed = before - len(ranked)
    op = "<" if higher_is_better else ">"
    if n_removed:
        log(f"  {label}: removed {n_removed}/{before} designs with "
            f"{key} {op} {fmt.format(threshold)}")
    else:
        log(f"  {label}: removed 0/{before} designs"
            + (f" ({n_nan} had no {key} measurement)" if n_nan else ""))
    return n_removed, n_nan


def write_status_json(path, args, tool_results, all_designs, validated,
                      target_path=None, chain_id=None, site_resnums=None,
                      target_residues=None, script="generate_binders.py",
                      extra=None):
    """
    Write a machine-readable run outcome to status.json.

    The pipeline's own summary is print()ed and discarded, and the process exits 0
    on partial failure, so stdout scraping is the only way to learn what happened —
    which is not something an agent driving this can do reliably. This records the
    outcome once, at the end, in a form that can be read back.

    `run_params.txt` is NOT a substitute: it is written at start (intent, not
    outcome), omits ~14 flags, and mislabels one.

    Never raises — a reporting failure must not fail a finished run.
    """
    try:
        path = Path(path)
        by_tool = {}
        for d in all_designs:
            t = _get_tool_from_design_id(d.get("design_id", ""))
            e = by_tool.setdefault(t, {"designs": 0, "ranked": 0})
            e["designs"] += 1
            if d.get("rank") is not None:
                e["ranked"] += 1

        failed = sorted(t for t, s in tool_results.items() if not str(s).startswith("OK"))
        failure_kinds = dict(getattr(args, "_tool_failures", {}) or {})
        # Validation stages are part of the run's success, not a footnote.
        failure_kinds.update({f"stage:{k}": v for k, v in VALIDATION_FAILURES.items()})
        stages_failed = sorted(VALIDATION_FAILURES)

        # Site residues that are not actually in the target chain are silently
        # dropped by every downstream consumer, so surface the mismatch here.
        site_missing = []
        if site_resnums and target_residues:
            present = set(target_residues)
            site_missing = sorted(r for r in site_resnums if r not in present)

        status = {
            "schema": 1,
            "script": script,
            "dry_run": bool(getattr(args, "dry_run", False)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # "ok" means: every requested tool succeeded AND something got ranked.
            "ok": (not failed) and (not stages_failed) and bool(validated),
            "validation_stages_failed": stages_failed,
            "target": str(target_path) if target_path else None,
            "target_chain": chain_id,
            "site": getattr(args, "site", None),
            "site_residues_not_in_target": site_missing,
            # rerank's --tools defaults to None; str(None) would record ["None"].
            "tools_requested": [s.strip() for s in
                                str(getattr(args, "tools", "") or "").split(",")
                                if s.strip()] or sorted(tool_results),
            "tools_failed": failed,
            # A user skip is not a failure; record it separately so an agent can
            # tell "I stopped this" from "this broke".
            "tools_skipped": sorted(getattr(args, "_tools_skipped", []) or []),
            # Classified so a caller can tell "retry smaller" (oom) from "fix the
            # install" (missing_env/missing_weights) from "fix my arguments".
            "failure_kinds": failure_kinds,
            "tool_results": dict(tool_results),
            "per_tool": by_tool,
            "designs_total": len(all_designs),
            "designs_ranked": len(validated),
            "reprediction": bool(getattr(args, "reprediction", False)),
            "mode": getattr(args, "mode", None),
            "top_score": (validated[0].get("combined_score") if validated else None),
            "top_design_id": (validated[0].get("design_id") if validated else None),
            "artifacts": {
                "rankings_csv": "rankings.csv",
                "top_designs": "top_designs",
                "dashboard": "dashboard.png",
            },
        }
        if extra:
            status.update(extra)
        # Non-atomic writes are the norm in this file, but this one is what a
        # reader polls for completion, so a half-file would be read as truth.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2, default=str) + "\n")
        os.replace(tmp, path)
        if site_missing:
            log(f"  WARNING: {len(site_missing)} --site residue(s) absent from chain "
                f"{chain_id}: {site_missing[:10]}{' ...' if len(site_missing) > 10 else ''}")
        return path
    except Exception as e:
        log(f"  (status.json not written: {e})", err=True)
        return None


def _pml_to_cxc(pml_path):
    """Write a ChimeraX .cxc next to a just-generated PyMOL .pml view (same scene).
    Faithful translation of the binder-view command set: load/align/color/show/transparency.
    Best-effort — never breaks the run if a line doesn't parse."""
    import re as _re
    pml_path = Path(pml_path)
    # PyMOL colour name -> explicit hex. ChimeraX does NOT share PyMOL's palette:
    # grey70/grey80/gray70, marine, lightorange and palecyan are all rejected, and
    # an unknown colour is a hard UserError that ABORTS the script, silently losing
    # every command after it. (Measured: 17 of 73 .cxc on disk were dead this way,
    # all via `lightorange`.) So every name the PML writers can emit is pinned to
    # hex here, and anything unrecognised falls back to grey rather than being
    # passed through to abort the file.
    NAMED = {"grey70": "#b3b3b3", "grey80": "#cccccc", "gray70": "#b3b3b3", "red": "#ff0000",
             "palegreen": "#a6e6a6", "plum": "#cc99cc", "violet": "#ff80ff", "wheat": "#fdd1a6",
             "white": "white", "marine": "#0080ff", "orange": "#ff8000", "green": "#00cc00",
             "firebrick": "#b22222", "cyan": "#00ffff", "magenta": "#ff00ff", "yellow": "#ffff00",
             "salmon": "#ff9999", "lightblue": "#bfbfff", "lightpink": "#ffbfde",
             "lightorange": "#ffcc80", "palecyan": "#ccffff", "black": "#000000",
             "tan": "#d2b48c", "purple": "#800080", "teal": "#008080", "skyblue": "#87ceeb"}
    _UNMAPPED = []

    def _hex(rgb):
        r, g, b = (int(max(0, min(1, float(x))) * 255) for x in rgb)
        return f"#{r:02x}{g:02x}{b:02x}"

    objmap, custom, nmodel = {}, {}, 0
    out = [f"# ChimeraX view (auto, equivalent of {pml_path.name})", "set bgColor white", ""]

    def _col(c):
        c = c.strip()
        hit = custom.get(c) or NAMED.get(c)
        if hit:
            return hit
        if c.startswith("#") or c in ("white", "black"):
            return c
        # Never emit an unrecognised name: ChimeraX aborts the whole script on one.
        _UNMAPPED.append(c)
        return "#808080"

    def _sel(s):
        s = s.strip()
        obj = _re.split(r"\s+and\s+", s, 1)[0].strip()
        if obj not in objmap:
            return None
        spec = f"#{objmap[obj]}"
        ch = _re.findall(r"chain\s+(\w+)", s)
        ri = _re.search(r"resi\s+([\d+]+)", s)
        if ch:
            spec += "/" + ",".join(ch)
        if ri:
            spec += ":" + ri.group(1).replace("+", ",")
        return spec

    try:
        for raw in pml_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _re.match(r"set_color\s+(\S+)\s*,\s*\[([^\]]+)\]", line)
            if m:
                custom[m.group(1)] = _hex(m.group(2).split(","))
                continue
            head, _, rest = line.partition(" ")
            args = [a.strip() for a in rest.split(",")]
            if head == "bg_color":
                out.append("set bgColor " + (_col(args[0]) if args else "white"))
            elif head == "load":
                fn = args[0]; name = args[1] if len(args) > 1 else os.path.splitext(args[0])[0]
                nmodel += 1; objmap[name] = nmodel
                out.append(f"open {fn}")
            elif head == "align":
                # Parenthesised: without them the conditional binds to `b` alone,
                # so a 1-arg `align` yields b=(None, None) — truthy — and emits
                # `matchmaker #a to #(None, None)`.
                a, b = ((objmap.get(args[0]), objmap.get(args[1]))
                        if len(args) > 1 else (None, None))
                if a and b:
                    out.append(f"matchmaker #{a} to #{b}")
            elif head == "color":
                sp = _sel(args[1]) if len(args) > 1 else None
                if sp:
                    out.append(f"color {sp} {_col(args[0])}")
            elif head == "show":
                sp = _sel(args[1]) if len(args) > 1 else None
                if sp:
                    if args[0] == "cartoon":
                        # Convention: TARGET (model #1, loaded first) = grey transparent SURFACE
                        # (epitope shows as the orange patch); BINDERS = cartoon, and the redundant
                        # target copy (chain A/A0) in each complex is hidden so only binders show.
                        mid_t = objmap.get(args[1].split(" and")[0].strip())
                        if mid_t == 1:
                            out.append(f"surface {sp}")
                        else:
                            out.append(f"cartoon {sp}")
                            # VIEWER_RULES.md §4.5: `hide atoms` immediately after
                            # `cartoon`, else pocket/site residues stay drawn as
                            # sticks on top of the scene. Trap table §11.
                            out.append(f"hide {sp} atoms")
                            out.append(f"hide #{mid_t}/A,A0 cartoon")
                    elif args[0] == "sticks":
                        out.append(f"show {sp} atoms"); out.append(f"style {sp} stick")
                    elif args[0] == "spheres":
                        out.append(f"style {sp} sphere")
            elif head == "disable":
                mid = objmap.get(args[0])
                if mid:
                    out.append(f"hide #{mid} models")
            elif head == "set" and args and args[0] == "cartoon_transparency":
                val = int(float(args[1]) * 100)
                if len(args) > 2 and objmap.get(args[2]):
                    out.append(f"transparency #{objmap[args[2]]} {val} target c")
                else:
                    out.append(f"transparency {val} target c")
            elif head in ("zoom", "orient"):
                out.append("view")
        # Standard footer used by every other ChimeraX generator in this project.
        out += ["", "hide H atoms", "hide pseudobonds", "lighting soft",
                "clip off", "view"]
        if _UNMAPPED:
            log(f"  ChimeraX: {pml_path.name} had unmapped colour(s) "
                f"{sorted(set(_UNMAPPED))} — rendered grey. Add them to NAMED.")
        cxc = pml_path.with_suffix(".cxc")
        cxc.write_text("\n".join(out) + "\n")
        return cxc
    except Exception as _e:
        log(f"  (ChimeraX .cxc translation skipped for {pml_path.name}: {_e})")
        return None


def _write_pml_by_chain(top_dir, copied_files, target_name, site_resnums=None,
                        cos_angle=None):
    """PyMOL script: load all structures, align to target, color by chain."""
    lines = [
        "# Auto-generated by generate_binders3.py",
        "# Color by chain: target=grey, binder=tool color",
        "bg_color white",
        "set ray_shadow, 0",
        # Mandated session render settings for every .pml/.pse this project builds
        "set two_sided_lighting, 0",
        "set specular, 0",
        "set cartoon_fancy_helices, 1",
        "",
    ]
    tool_colors = PYMOL_TOOL_COLORS

    if target_name:
        obj = Path(target_name).stem
        lines.append(f'load {target_name}, {obj}')
        lines.append(f'color grey70, {obj}')
        lines.append(f'show cartoon, {obj}')
        lines.append(f'hide lines, {obj}')
        if site_resnums:
            resi_str = "+".join(str(r) for r in site_resnums)
            lines.append(f'# Highlight binding-site (epitope) residues — orange')
            lines.append(f'color orange, {obj} and chain A and resi {resi_str}')
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
        # Clamp: --min_site_cos is a user value and math.acos raises outside
        # [-1, 1] — which happened only here, in the PyMOL scene writer, AFTER
        # every design had been scored and ranked. Losing a completed run to a
        # visualisation argument is not acceptable.
        half_angle = math.acos(max(-1.0, min(1.0, float(cos_angle))))
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
    cxc = _pml_to_cxc(pml_path)
    if cxc:
        log(f"ChimeraX script (by chain) → {cxc}")


def _write_pml_by_iptm(top_dir, copied_files, target_name, site_resnums=None):
    """PyMOL script: load all structures, align to target, color binders by iPTM."""
    lines = [
        "# Auto-generated by generate_binders3.py",
        "# Color binders by Boltz iPTM: red (low) → yellow → green (high)",
        "bg_color white",
        "set ray_shadow, 0",
        # Mandated session render settings for every .pml/.pse this project builds
        "set two_sided_lighting, 0",
        "set specular, 0",
        "set cartoon_fancy_helices, 1",
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
            lines.append(f'# Highlight binding-site (epitope) residues — orange')
            lines.append(f'color orange, {obj} and chain A and resi {resi_str}')
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
    cxc = _pml_to_cxc(pml_path)
    if cxc:
        log(f"ChimeraX script (by iPTM) → {cxc}")


def _write_per_tool_folders(designs, top_dir, target_pdb, target_name,
                            site_resnums, n_per_tool=20, target_residues=None):
    """
    Create per-tool subdirectories under top_designs/ with top 10 designs
    from each tool ranked by combined_score, plus PML scripts with
    binding site highlighted in red.
    """
    tool_colors = PYMOL_TOOL_COLORS

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
            # Mandated session render settings for every .pml/.pse this project builds
            "set two_sided_lighting, 0",
            "set specular, 0",
            "set cartoon_fancy_helices, 1",
            "",
        ]

        if tool_target:
            tobj = Path(tool_target).stem
            pml_lines.append(f"load {tool_target}, {tobj}")
            pml_lines.append(f"color grey70, {tobj}")
            pml_lines.append(f"show cartoon, {tobj}")
            pml_lines.append(f"hide lines, {tobj}")
            # Highlight binding site (epitope) in orange (PDB numbering for target PDB)
            if site_sel_pdb:
                pml_lines.append(f"color orange, {tobj} and {site_sel_pdb}")
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
            # Highlight binding site (epitope) in orange on complex target chain (CIF renumbered)
            if site_sel_cif:
                pml_lines.append(f"color orange, {obj} and {site_sel_cif}")
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
        _pml_to_cxc(pml_path)

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
            # Do NOT `continue` here. gemmi 0.6.5 fails on these files in two
            # different ways — it can raise ("no structural models"), or it can
            # return 0 models and write an atom-less PDB — and the recovery path
            # below handles both. Skipping on the raising case is what made
            # PXDesign designs silently drop out of PLIP.
            log(f"  {rank_label}: gemmi conversion failed ({e}); trying direct mmCIF read")

        # gemmi 0.6.5 (pinned by boltz) silently returns 0 models for some mmCIF
        # column orders — notably BindCraft2's — and then write_pdb emits a file
        # with only CRYST1 and END. PLIP accepts that happily and reports zero
        # interactions, so the failure is invisible. Detect the empty result and
        # fall back to reading _atom_site directly.
        try:
            _has_atoms = any(l.startswith(("ATOM", "HETATM"))
                             for l in Path(pdb_path).read_text().splitlines())
        except Exception:
            _has_atoms = False
        if not _has_atoms:
            _src = d.get("complex_cif") or d.get("binder_pdb")
            _n = cif_to_pdb_atoms(_src, pdb_path) if _src else 0
            if _n:
                log(f"  {rank_label}: gemmi produced an empty PDB; "
                    f"recovered {_n} atoms via direct mmCIF read")
            else:
                log(f"  {rank_label}: could not extract atoms from {_src} — "
                    f"skipping PLIP", err=True)
                continue

        # Run PLIP
        try:
            # Find plip binary: PATH first, then next to current Python
            plip_bin = shutil.which("plip")
            if not plip_bin:
                candidate = Path(sys.executable).parent / "plip"
                if candidate.exists():
                    plip_bin = str(candidate)
            if not plip_bin:
                raise FileNotFoundError("plip")
            result = subprocess.run(
                [plip_bin, "-f", str(pdb_path), "-o", str(design_plip_dir),
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
    ax2.set_xlabel("Monomer binder pLDDT"); ax2.set_ylabel("Count")
    ax2.set_title("Monomer pre-filter distribution")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(fontsize=8)

    # ── Panel 3: Monomer pLDDT vs Boltz iPTM scatter ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for t in tool_list:
        pts = [(d.get("esmfold_plddt"), d.get("boltz_iptm")) for d in tool_groups[t]]
        pts = [(x, y) for x, y in pts if x is not None and y is not None and x == x and y == y]
        if pts:
            x, y = zip(*pts)
            ax3.scatter(x, y, c=colors.get(t, "grey"), alpha=0.7, s=30, label=t)
    ax3.set_xlabel("Monomer binder pLDDT"); ax3.set_ylabel("Boltz-2 iPTM")
    ax3.set_title("Monomer pLDDT vs Boltz iPTM")
    if ax3.get_legend_handles_labels()[0]:
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
            iptm = d.get("_resolved_iptm", d.get("boltz_iptm", float("nan")))
            if pae == pae and iptm == iptm:  # both non-nan
                x.append(pae)
                y.append(iptm)
        if x:
            ax5.scatter(x, y, c=colors.get(t, "grey"), alpha=0.7,
                        s=40, label=t, edgecolors="none")
    ax5.set_xlabel("Min interface pAE (lower = better contacts)")
    ax5.set_ylabel("Boltz-2 iPTM")
    ax5.set_title("Interface quality: iPTM vs min pAE")
    if ax5.get_legend_handles_labels()[0]:
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
            iptm      = d.get("_resolved_iptm", d.get("boltz_iptm", float("nan")))
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
                        help='Binding site residues, chain-prefixed and 1-based, e.g. '
                             '"A:54,56,66,115" or with ranges "A:8-30,120-123". '
                             'Single-letter chain ids only. Legacy space-separated '
                             '"A10 A15 A22" is still accepted. Numbers must match THIS '
                             'file\'s numbering — verify with --preview_site first.')
    parser.add_argument("--length",    required=True,
                        help="Binder length range, e.g. 60-80, or a single value like 70. "
                             "Valid 10-500 residues. Ignored by scaffolded BindCraft2 "
                             "modalities (VHH/scFv/Fab/ARP), which carry their own length.")
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
                        help="Boltz-2 monomer pre-filter pLDDT threshold (default: 80)")
    parser.add_argument("--max_validate", type=int, default=20,
                        help="IGNORED — kept only so existing scripts/web forms do not break. "
                             "validate_boltz() batches ALL designs in one invocation "
                             "(its own docstring: 'max_per_tool is kept for API compatibility but ignored'). Was documented as 'max designs sent to Boltz validation per tool', which was never true.")
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
    parser.add_argument("--no_boltzgen_site_filter", action="store_true",
                        help="Do not make BoltzGen enforce the binding site it was given. "
                             "BoltzGen's filter_bindingsite defaults to FALSE, so without this "
                             "enforcement it happily spends its whole budget off-target. "
                             "Only applies when --site is set.")
    parser.add_argument("--min_site_cos", type=float, default=None,
                        help="Minimum cosine between the site's outward surface normal and "
                             "the binder direction. >0 means the binder approaches from the "
                             "solvent side. Typical: 0.3-0.5. Default: disabled. "
                             "(generate_binders computed site_cos_angle but had no flag to "
                             "filter on it, so its internal guard was permanently dead.)")
    parser.add_argument("--max_rg_ratio", type=float, default=None,
                        help="Maximum binder radius-of-gyration ratio Rg/(2.2*N^0.38). "
                             "Rejects extended / single-helix binders that refolding RMSD "
                             "cannot catch (a lone helix refolds to a lone helix). "
                             "~1.0 is globular; 1.25 is a good cut. Default: disabled "
                             "(the metric is still recorded as binder_rg_ratio).")
    parser.add_argument("--require_site_geometry", action="store_true",
                        help="Drop designs whose site geometry could not be measured "
                             "(site_geometry_status != 'measured') instead of ranking them. "
                             "Without this they are ranked with no evidence they engage the site.")
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
                        default="no_filters",
                        help="BindCraft filter preset name or full path to JSON. "
                             "Presets: no_filters (default), relaxed, default, "
                             "kras_relaxed, peptide, peptide_relaxed. "
                             "no_filters = accept all, let pipeline rank. "
                             "relaxed = light quality gate. "
                             "default = strict (may stall on difficult targets).")
    parser.add_argument("--bindcraft_advanced",
                        default=f"{BINDCRAFT_DIR}/settings_advanced/default_4stage_multimer_flexible.json",
                        help="BindCraft advanced settings JSON (default: "
                             "default_4stage_multimer_flexible.json — removes target "
                             "template constraint for broader sampling). Use "
                             "default_4stage_multimer_flexible_hardtarget.json for "
                             "especially difficult targets like KRAS.")
    parser.add_argument("--bindcraft2_filters",
                        default="no_filters",
                        choices=list(BINDCRAFT2_FILTER_PRESETS),
                        help="BindCraft2 acceptance-filter preset. BC2 keeps filters "
                             "inline in its campaign JSON, so these are threshold "
                             "overrides, not file paths. Same vocabulary as "
                             "--bindcraft_filters: no_filters (default) = accept all "
                             "and let the pipeline rank; relaxed = light quality gate; "
                             "default = BC2's own strict defaults (i_pTM>=0.7, "
                             "i_pAE<=0.35) which may stall on difficult targets.")
    parser.add_argument("--bindcraft2_modality",
                        default="binder",
                        help="BindCraft2 modality preset, passed to `bindcraft design "
                             "--modality`. Known: " + ", ".join(BINDCRAFT2_MODALITIES) +
                             ". 'binder' (default) is the de novo miniprotein format "
                             "equivalent to what BindCraft v1 did here. Scaffolded "
                             "formats (VHH, scFv, Fab, ARP) carry their own binder "
                             "length and ignore --length.")
    parser.add_argument("--bindcraft2_max_trajectories",
                        type=int, default=None,
                        help="Cap on BindCraft2 design attempts. BC2 leaves this unset "
                             "by default, which means it keeps designing until it "
                             "reaches the requested accepted count with no upper bound. "
                             "Default here: 50x the requested design count.")
    parser.add_argument("--device",    default=None,
                        help="CUDA device, e.g. cuda:0 or cuda:1. "
                             "Pins all GPU work to this device via CUDA_VISIBLE_DEVICES. "
                             "Default: use all available GPUs.")
    parser.add_argument("--render_designs", action="store_true",
                        help="Ray-trace the top ranked complexes to PNG and exit. "
                             "Needs --results_dir (a finished run) and optionally "
                             "--site (to highlight the epitope) and --top N "
                             "(default 6). No GPU. The .pml/.cxc scenes assume a "
                             "human at a GUI; these are images an agent can read "
                             "and show the user. Exit 0 = rendered, 1 = nothing to "
                             "render.")
    parser.add_argument("--self_test", action="store_true",
                        help="Actually EXERCISE every tool env, weight file and "
                             "dependency — imports the modules, constructs the "
                             "objects — instead of only checking that directories "
                             "exist, which is all --capabilities does. Takes ~1 min, "
                             "no GPU work, needs no other arguments. Exit 0 = healthy, "
                             "1 = something is broken. Add --json for a dict. Run this "
                             "after an install, or before trusting an unfamiliar box.")
    parser.add_argument("--preview_site", action="store_true",
                        help="Check that --site describes a plausible surface epitope and "
                             "exit. Reports per-residue solvent exposure, how far the "
                             "residues are spread, and a verdict; writes "
                             "site_preview.{pml,cxc,png} to --out_dir (or /tmp). Needs "
                             "--target and --site only. No GPU. Exit 0 = plausible, "
                             "1 = implausible (buried or scattered residues). Use this to "
                             "confirm numbering BEFORE committing GPU time. Takes two "
                             "extra flags of its own: --json (machine-readable dict "
                             "instead of the text report) and --no_render (skip the PNG).")
    parser.add_argument("--capabilities", action="store_true",
                        help="Print a machine-readable JSON description of what this "
                             "machine can run (tools, envs, GPUs, VRAM limits, modes, "
                             "exit codes, failure kinds) and exit 0. Intended as the "
                             "FIRST call an agent makes: it needs no other arguments "
                             "and touches no GPU. Add --deep to verify each tool by "
                             "IMPORTING it rather than checking its directory exists "
                             "(~1 min); `check_depth` in the output says which you got.")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Validate everything and print the commands without running "
                             "them or touching the GPU. NOTE: it still CREATES --out_dir "
                             "and writes the sanitized target, per-tool configs and "
                             "placeholder outputs there — so dry-run into a throwaway "
                             "directory, not the one you will use for the real run. It "
                             "deliberately does NOT write status.json.")
    parser.add_argument("--force_site", action="store_true",
                        help="Boltz-2 validation: FORCE the binder pocket onto the site "
                             "(pocket force:true). With --holo_ligands, the forced pocket also "
                             "targets the nucleotide (phosphate), driving the binder to the catalytic region.")
    parser.add_argument("--site_max_distance", type=float, default=6.0,
                        help="max_distance (A) for the forced Boltz-2 pocket constraint (default 6.0).")
    parser.add_argument("--holo_ligands", default=None,
                        help="Comma-separated CCD codes added as ligands to the Boltz-2 complex "
                             "validation, e.g. 'GDP,AF3,MG' (holo/nucleotide context). The first "
                             "ligand bears the phosphate and is included in the forced pocket.")
    parser.add_argument("--reprediction", action="store_true",
                        help="Enable Boltz-2 re-prediction for ALL tools including "
                             "those with native iPTM scores (BindCraft, BoltzGen, PXDesign). "
                             "Default behavior: only backbone-only tools (RFdiffusion, Proteina) "
                             "go through Boltz-2 re-prediction; native-iPTM tools keep their "
                             "own scores for ranking.")
    parser.add_argument("--n_tag", type=str, default="",
                        help="Amino acid sequence to prepend to binder N-terminus for Boltz-2 reprediction "
                             "(e.g. MHHHHHH for His-tag). Only used during reprediction.")
    parser.add_argument("--c_tag", type=str, default="",
                        help="Amino acid sequence to append to binder C-terminus for Boltz-2 reprediction "
                             "(e.g. HHHHHH for C-terminal His-tag). Only used during reprediction.")
    parser.add_argument("--ss_bias", choices=["beta", "helix", "balanced"],
                        default="balanced",
                        help="Secondary structure bias: beta (more sheets), "
                             "helix (more helices), balanced (default, no bias)")
    # --capabilities is a pure query: it must answer without --target/--site/etc,
    # which argparse marks required. Handled before parse_args so an agent can
    # call it as its very first action, knowing nothing about the pipeline.
    if "--capabilities" in sys.argv[1:]:
        _deep = "--deep" in sys.argv[1:]
        print(json.dumps(report_capabilities(deep=_deep), indent=2, default=str))
        return 0

    if "--render_designs" in sys.argv[1:]:
        rv = argparse.ArgumentParser(add_help=False)
        rv.add_argument("--results_dir", required=True)
        rv.add_argument("--site", default=None)
        rv.add_argument("--out_dir", default=None)
        rv.add_argument("--top", type=int, default=6)
        rv.add_argument("--json", action="store_true")
        ra, _ = rv.parse_known_args()
        _chain, _resn = ("A", [])
        if ra.site:
            try:
                _chain, _resn = parse_site(ra.site)
            except ValueError as e:
                sys.exit(f"ERROR --site: {e}")
        rep = render_designs(ra.results_dir, out_dir=ra.out_dir, top=ra.top,
                             site_resnums=_resn, chain_id=_chain)
        print(json.dumps(rep, indent=2, default=str) if ra.json
              else render_designs_text(rep))
        return 0 if rep.get("ok") else 1

    if "--self_test" in sys.argv[1:]:
        rep = run_self_test()
        if "--json" in sys.argv[1:]:
            print(json.dumps(rep, indent=2, default=str))
        else:
            print(render_self_test_text(rep))
        return 0 if rep["ok"] else 1

    # --preview_site is likewise a query: --target and --site only.
    if "--preview_site" in sys.argv[1:]:
        pv = argparse.ArgumentParser(add_help=False)
        pv.add_argument("--target", required=True)
        pv.add_argument("--site", required=True)
        pv.add_argument("--out_dir", default=None)
        pv.add_argument("--json", action="store_true")
        pv.add_argument("--no_render", action="store_true")
        pa, _ = pv.parse_known_args()
        try:
            _chain, _resnums = parse_site(pa.site)
        except ValueError as e:
            sys.exit(f"ERROR --site: {e}")
        if not Path(pa.target).exists():
            sys.exit(f"ERROR: target file not found: {pa.target}")
        _od = Path(pa.out_dir) if pa.out_dir else Path(tempfile.mkdtemp(prefix="site_preview_"))
        rep = preview_site(pa.target, _chain, _resnums, out_dir=_od,
                           render=not pa.no_render)
        if pa.json:
            print(json.dumps(rep, indent=2, default=str))
        else:
            print(render_site_preview_text(rep, pa.target, pa.site))
        # Distinct exit codes: 0 = site usable, 1 = site checked and REJECTED
        # (fix your --site), 2 = the check could not run (fix the install or the
        # target file). These used to collapse into 1, so an agent told to fix
        # its --site could not tell that biopython was simply missing.
        _st = rep.get("status") or ("ok" if rep.get("ok") else "bad_site")
        if _st == "ok":
            return 0
        if _st == "bad_site":
            return 1
        print(f"--preview_site could not run the check: {rep.get('verdict')}",
              file=sys.stderr)
        return 2

    args = parser.parse_args()

    # ── Validate and uppercase terminal tags ─────────────────────────────
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    for tag_name in ("n_tag", "c_tag"):
        tag_val = getattr(args, tag_name).strip().upper()
        if tag_val and not set(tag_val).issubset(VALID_AA):
            bad = set(tag_val) - VALID_AA
            sys.exit(f"ERROR: --{tag_name} contains invalid amino acid characters: {bad}")
        setattr(args, tag_name, tag_val)

    # ── Resolve BindCraft filter preset to full path ─────────────────────
    BINDCRAFT_FILTER_PRESETS = {
        "no_filters": "no_filters.json",
        "relaxed": "relaxed_filters.json",
        "default": "default_filters.json",
        "kras_relaxed": "kras_relaxed_filters.json",
        "peptide": "peptide_filters.json",
        "peptide_relaxed": "peptide_relaxed_filters.json",
    }
    if not args.bindcraft_filters:
        args.bindcraft_filters = "no_filters"
    bindcraft_filters_preset = args.bindcraft_filters  # keep preset name for run_params.txt
    if not os.path.isfile(args.bindcraft_filters):
        preset_file = BINDCRAFT_FILTER_PRESETS.get(args.bindcraft_filters)
        if preset_file:
            args.bindcraft_filters = str(Path(BINDCRAFT_DIR) / "settings_filters" / preset_file)
        else:
            log(f"WARNING: BindCraft filter preset '{args.bindcraft_filters}' not found, using no_filters")
            args.bindcraft_filters = str(Path(BINDCRAFT_DIR) / "settings_filters" / "no_filters.json")
            bindcraft_filters_preset = "no_filters"

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

    # RFdiffusion and BindCraft v1 cannot read CIF — require PDB upfront.
    # BindCraft2 is deliberately absent: it takes PDB, mmCIF or FASTA targets.
    pdb_only_tools = [t for t in tools if t in ("rfdiffusion", "bindcraft")]
    if pdb_only_tools and target_path.suffix.lower() == ".cif":
        sys.exit(
            f"ERROR: {pdb_only_tools} require a PDB file, but got: {target_path}\n"
            f"  Use --target with a .pdb file. For KRAS4B:\n"
            f"  --target /data/scripts/protein_folding/outputs/KRASB/8ECR_monomer.pdb"
        )

    # Start from mode presets, then apply any --n_designs overrides
    n_designs = dict(DESIGN_MODES[args.mode])
    # Reduce BindCraft count when filters are active (strict filters = slow)
    bc_filters_active = (hasattr(args, 'bindcraft_filters') and
                         args.bindcraft_filters and
                         not args.bindcraft_filters.endswith("no_filters.json") and
                         args.bindcraft_filters != "no_filters")
    if bc_filters_active and "bindcraft" in n_designs:
        n_designs["bindcraft"] = BINDCRAFT_FILTERED_COUNTS.get(args.mode, n_designs["bindcraft"])
    # Same reduction for BC2. Its presets are names, not paths — "no_filters"
    # still lets BC2 emit everything, so only a real gate reduces the count.
    bc2_filters_active = (getattr(args, "bindcraft2_filters", "no_filters")
                          not in (None, "", "no_filters"))
    if bc2_filters_active and "bindcraft2" in n_designs:
        n_designs["bindcraft2"] = BINDCRAFT_FILTERED_COUNTS.get(args.mode, n_designs["bindcraft2"])
    if args.n_designs:
        try:
            overrides = parse_n_designs(args.n_designs)
        except ValueError as e:
            sys.exit(f"ERROR --n_designs: {e}")
        n_designs.update(overrides)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_target_path = target_path
    clean_target = out_dir / f"{target_path.stem}_clean.pdb"
    sanitize_target_pdb(target_path, clean_target)
    target_path = clean_target

    log("=" * 62)
    log("generate_binders.py — Multi-tool binder design pipeline")
    log("=" * 62)
    log(f"Target:           {target_path}")
    log(f"Target (raw):     {raw_target_path}")
    log(f"Site:             chain={chain_id}, residues={site_resnums}")
    log(f"Length:           {length_min}–{length_max} aa")
    log(f"Tools:            {tools}")
    log(f"Device:           {args.device or 'all GPUs'}")
    log(f"Mode:             {args.mode}")
    log(f"N designs:        { {t: n_designs.get(t,'?') for t in tools} }")
    _est = estimate_runtime_minutes(n_designs, tools, args.reprediction)
    log(f"Est. runtime:     {format_runtime(_est)} on 1 GPU "
        f"({sum(n_designs.get(t,0) for t in tools)} designs"
        f"{', with reprediction' if args.reprediction else ''}) "
        f"— rough, BoltzGen/BindCraft dominate")
    log(f"Score weights:    pLDDT={score_weights[0]}, iPTM={score_weights[1]}, dG={score_weights[2]}")
    log(f"Reprediction:     {args.reprediction}")
    if args.n_tag or args.c_tag:
        log(f"N-terminal tag:   {args.n_tag or '(none)'}")
        log(f"C-terminal tag:   {args.c_tag or '(none)'}")
    log(f"SS bias:          {args.ss_bias}")
    log(f"Filter PAE:       {args.filter_interface_pae}")
    log(f"Max site dist:    {args.max_site_dist} Å")
    log(f"Min site frac:    {args.min_site_fraction}")
    log(f"Top N copy:       {args.top_n}")
    log(f"Output:           {out_dir}")
    if args.dry_run:
        log("*** DRY RUN — no commands will be executed ***")
    log("=" * 62)

    # ── Save run parameters to file for reproducibility ──────────────────
    params_file = out_dir / "run_params.txt"
    cmd_parts = ["python generate_binders.py"]
    cmd_parts.append(f"--target {target_path}")
    cmd_parts.append(f'--site "{chain_id}:{",".join(str(r) for r in site_resnums)}"')
    cmd_parts.append(f"--length {length_min}-{length_max}")
    cmd_parts.append(f"--tools {','.join(tools)}")
    cmd_parts.append(f"--mode {args.mode}")
    if args.device:
        cmd_parts.append(f"--device {args.device}")
    cmd_parts.append(f"--score_weights {','.join(str(w) for w in score_weights)}")
    cmd_parts.append(f"--ss_bias {args.ss_bias}")
    if args.reprediction:
        cmd_parts.append("--reprediction")
    if args.n_tag:
        cmd_parts.append(f"--n_tag {args.n_tag}")
    if args.c_tag:
        cmd_parts.append(f"--c_tag {args.c_tag}")
    if args.plip_top:
        cmd_parts.append(f"--plip_top {args.plip_top}")
    if args.top_n != 50:
        cmd_parts.append(f"--top_n {args.top_n}")
    if args.max_site_dist != 15.0:
        cmd_parts.append(f"--max_site_dist {args.max_site_dist}")
    if args.min_site_fraction:
        cmd_parts.append(f"--min_site_fraction {args.min_site_fraction}")
    if args.filter_interface_pae:
        cmd_parts.append(f"--filter_interface_pae {args.filter_interface_pae}")
    if args.esmfold_plddt_threshold != 80.0:
        cmd_parts.append(f"--esmfold_plddt_threshold {args.esmfold_plddt_threshold}")
    if bindcraft_filters_preset and bindcraft_filters_preset != "no_filters":
        cmd_parts.append(f"--bindcraft_filters {bindcraft_filters_preset}")
    cmd_parts.append(f"--out_dir {out_dir}")
    if args.dry_run:
        cmd_parts.append("--dry_run")
    with open(params_file, "w") as f:
        f.write("# Run command (reproducible)\n")
        f.write("CUDA_VISIBLE_DEVICES=N \\\n  " + " \\\n  ".join(cmd_parts) + "\n")
        f.write(f"\n# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Machine: {os.uname().nodename}\n")
    log(f"Run parameters saved → {params_file}")

    # ── Extract target sequence ───────────────────────────────────────────
    chain_info = get_chain_info(target_path)
    tchain     = chain_info.get(chain_id, {})
    target_seq = tchain.get("sequence", "")
    target_len = tchain.get("length", 0)

    # Checked even under --dry_run: reading the target costs nothing, and a dry run
    # that returns 0 for a chain that does not exist is worse than useless — it is
    # the check an agent relies on before committing hours of GPU.
    if not target_seq:
        sys.exit(f"ERROR: chain {chain_id} not found in {target_path}. "
                 f"Available chains: {list(chain_info)}")
    log(f"Target chain {chain_id}: {target_len} residues")

    # Site residues absent from the target chain are silently DROPPED by every
    # consumer downstream (BoltzGen YAML, RFdiffusion hotspots, the geometric
    # filter). In the worst case RFdiffusion ends up with hotspots="" and designs
    # against a random surface patch — hours of GPU, exit 0, no error. The only
    # pre-existing check lives inside validate_boltz (Stage 2), which runs long
    # after the design tools have already consumed the bad site. Warn here, before
    # anything is spent. Non-fatal by design: partial site overlap is legitimate
    # when the target was renumbered.
    _site_missing = []
    if site_resnums and tchain.get("residues"):
        _present = set(tchain["residues"])
        _site_missing = sorted(r for r in site_resnums if r not in _present)
        if _site_missing:
            log(f"  WARNING: {len(_site_missing)}/{len(site_resnums)} --site residue(s) "
                f"are NOT in chain {chain_id}: {_site_missing[:15]}"
                f"{' ...' if len(_site_missing) > 15 else ''}")
            log(f"           They will be dropped silently downstream. Check --site "
                f"against the target's actual numbering "
                f"({min(_present)}-{max(_present)}).")
            if len(_site_missing) == len(site_resnums):
                sys.exit(f"ERROR: NONE of the --site residues exist in chain {chain_id}. "
                         f"Refusing to run — every tool would design against an "
                         f"unconstrained surface. Target range: "
                         f"{min(_present)}-{max(_present)}.")

    if target_len > 500:
        log(f"NOTE: large target ({target_len} aa) — Boltz validation will use "
            f"reduced MSA depth and recycling steps (auto memory-save)")

    # These are only threaded into validate_boltz() on the reprediction branch;
    # without --reprediction they were accepted and silently did nothing.
    if not args.reprediction:
        _inert = [f for f, v in (("--force_site", args.force_site),
                                 ("--holo_ligands", args.holo_ligands)) if v]
        if _inert:
            sys.exit(f"ERROR: {', '.join(_inert)} only take effect with "
                     f"--reprediction (they configure the Boltz-2 re-prediction "
                     f"that is skipped otherwise). Add --reprediction, or drop "
                     f"these flags.")

    # ── Preflight: can this box actually run what was asked for? ──────────
    _missing_tools = preflight_tools(tools, dry_run=args.dry_run)
    if _missing_tools and not args.dry_run:
        if set(_missing_tools) >= set(tools):
            sys.exit("ERROR: none of the requested tools can run on this machine:\n  "
                     + "\n  ".join(f"{t}: {w}" for t, w in sorted(_missing_tools.items()))
                     + "\n  Check conda_envs in binders_pipeline_env/config.yaml.")
        log(f"  Continuing with {len(tools) - len(_missing_tools)} of {len(tools)} "
            f"tools; the rest will be recorded as failed.", err=True)

    # ── Run design tools sequentially ─────────────────────────────────────
    all_designs  = []
    tool_results = {}
    tool_failures = {}   # tool -> failure kind, for status.json
    tools_skipped = []   # tools the USER skipped — not failures
    for _t, _why in _missing_tools.items():
        tool_results[_t] = f"FAILED[missing_env]: {_why}"
        tool_failures[_t] = "missing_env"

    RUNNERS = {
        "rfdiffusion":       run_rfdiffusion,
        "boltzgen":          run_boltzgen,
        "bindcraft":         run_bindcraft,
        "bindcraft2":        run_bindcraft2,
        "pxdesign":          run_pxdesign,
        "proteina":          run_proteina,
        "proteina_complexa": run_proteina_complexa,
        "rfdiffusion3":      run_rfdiffusion3,
    }

    for tool in tools:
        if tool in _missing_tools:
            continue
        n = n_designs.get(tool, 100)
        # Clean any stale skip signal from a previous tool
        skip_signal = out_dir / f".skip_{tool}"
        if skip_signal.exists():
            skip_signal.unlink()

        log(f"\n{'─' * 50}")
        log(f"STEP: {tool.upper()}  (n={n})")
        log(f"{'─' * 50}")
        try:
            extra = {}
            if tool == "bindcraft":
                extra = {"filters_path": args.bindcraft_filters,
                         "advanced_path": args.bindcraft_advanced,
                         "ss_bias": args.ss_bias}
            elif tool == "bindcraft2":
                extra = {"filters_preset": args.bindcraft2_filters,
                         "modality": args.bindcraft2_modality,
                         "max_trajectories": args.bindcraft2_max_trajectories,
                         "ss_bias": args.ss_bias,
                         "target_len": target_len}
            elif tool == "boltzgen":
                extra = {"ss_bias": args.ss_bias,
                         "filter_bindingsite": not args.no_boltzgen_site_filter}
            elif tool == "proteina_complexa":
                extra = {"ss_bias": args.ss_bias}
            designs = RUNNERS[tool](
                target_path, chain_id, site_resnums, length_min, length_max,
                n, out_dir / tool, args.dry_run, **extra)
            all_designs.extend(designs)
            # A tool that runs to completion and produces NOTHING is not OK.
            # It used to be recorded as "OK (0 designs)", which starts with "OK",
            # so status.json said ok:true, tools_failed was empty and the run
            # exited 0 — indistinguishable from a successful tool.
            if designs:
                tool_results[tool] = f"OK ({len(designs)} designs)"
            else:
                tool_results[tool] = "FAILED[no_designs]: tool completed but produced 0 designs"
                tool_failures[tool] = "unknown"
                log(f"{tool}: completed but produced 0 designs — recorded as a failure",
                    err=True)
        except SkipRequested:
            # User skipped — collect whatever partial output exists
            log(f"  {tool}: collecting partial output after skip...")
            designs = _collect_partial_output(tool, out_dir / tool, chain_id,
                                              target_seq=(tchain or {}).get("sequence"))
            all_designs.extend(designs)
            n_collected = len(designs)
            log(f"  {tool}: skipped — {n_collected} partial designs collected")
            # "SKIPPED" does not start with "OK", so write_status_json counted a
            # deliberate user skip as a FAILED tool: ok:false and
            # tools_failed:[tool] alongside exit 0. A skip is a user decision, not
            # a failure — mark it OK-with-a-note and record it separately.
            tool_results[tool] = f"OK (skipped by user, {n_collected}/{n} designs collected)"
            tools_skipped.append(tool)
            args._tools_skipped = tools_skipped
        except Exception as e:
            _kind = getattr(e, "kind", None) or classify_failure(str(e))
            log(f"{tool} FAILED [{_kind}]: {e}", err=True)
            tool_failures[tool] = _kind
            # Retrying an OOM unchanged just burns the same memory again.
            if _kind == "oom":
                log(f"  {tool}: out of memory — a retry would fail identically. "
                    f"Reduce --length or the target size, or use a larger GPU.", err=True)
                tool_results[tool] = f"FAILED[oom]: {e}"
                _cleanup_gpu()
                continue
            # Retry once after GPU cleanup (crashed tools may leave zombie GPU processes)
            _cleanup_gpu()
            log(f"  Retrying {tool} after GPU cleanup...")
            try:
                designs = RUNNERS[tool](
                    target_path, chain_id, site_resnums, length_min, length_max,
                    n, out_dir / tool, args.dry_run, **extra)
                all_designs.extend(designs)
                tool_results[tool] = f"OK ({len(designs)} designs, retry)"
                tool_failures.pop(tool, None)
                log(f"  {tool} succeeded on retry")
            except SkipRequested:
                log(f"  {tool}: skipped during retry")
                designs = _collect_partial_output(tool, out_dir / tool, chain_id,
                                              target_seq=(tchain or {}).get("sequence"))
                all_designs.extend(designs)
                tool_results[tool] = f"SKIPPED ({len(designs)} designs)"
            except Exception as e2:
                # Report e2 (the retry's error), not e — the first exception was
                # being recorded here, discarding the more informative one.
                _kind2 = getattr(e2, "kind", None) or classify_failure(str(e2))
                log(f"{tool} FAILED on retry [{_kind2}]: {e2}", err=True)
                tool_failures[tool] = _kind2
                tool_results[tool] = f"FAILED[{_kind2}]: {e2}"
        # Clean up GPU between tools to prevent memory leaks from crashed subprocesses
        _cleanup_gpu()

    if not all_designs:
        # Write status.json BEFORE exiting. This is the one exit code the docs
        # tell an agent to diagnose via status.json, and it was the one path that
        # never wrote it — so "every tool failed" produced an exit 1 with no
        # machine-readable explanation of WHICH tool failed or why.
        log("No designs collected. Exiting.", err=True)
        if not args.dry_run:
            try:
                write_status_json(out_dir / "status.json", args, tool_results, [], [],
                                  target_path=target_path, chain_id=chain_id,
                                  site_resnums=site_resnums,
                                  target_residues=tchain.get("residues"))
                log(f"  status.json    — written with per-tool failure kinds "
                    f"({out_dir / 'status.json'})", err=True)
            except Exception as _e:
                log(f"  could not write status.json: {_e}", err=True)
        sys.exit(1)

    log(f"\nTotal designs collected: {len(all_designs)}")

    # ── Validation ────────────────────────────────────────────────────────
    val_dir = out_dir / "validation"

    log(f"\n{'─' * 50}")
    log("STEP: Boltz-2 monomer pre-filter (Stage 1)")
    log(f"{'─' * 50}")
    if args.reprediction:
        # --reprediction ON: Boltz-2 monomer all designs
        passing = validate_boltz_monomer(
            all_designs, val_dir, args.esmfold_plddt_threshold, args.dry_run,
            n_tag=args.n_tag, c_tag=args.c_tag)
    else:
        # --reprediction OFF: only Boltz-2 monomer for backbone-only tools (RFdiffusion, Proteina).
        # iPTM-native tools (BindCraft, BoltzGen, PXDesign, Proteina Complexa)
        # already have internal validation — monomer filter is redundant for them.
        backbone_all = [d for d in all_designs
                        if _get_tool_from_design_id(d["design_id"]) in BACKBONE_ONLY_TOOLS]
        native_all = [d for d in all_designs
                      if _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS]
        log(f"  Boltz monomer: {len(backbone_all)} backbone-only designs "
            f"(skipping {len(native_all)} native-iPTM designs)")
        passing_backbone = validate_boltz_monomer(
            backbone_all, val_dir, args.esmfold_plddt_threshold, args.dry_run,
            n_tag=args.n_tag, c_tag=args.c_tag)
        if not passing_backbone and backbone_all and not args.dry_run:
            passing_backbone = backbone_all
        # Native tools pass through directly — no monomer gate
        passing = (passing_backbone or []) + native_all

    if not passing and not args.dry_run:
        log("WARNING: no designs passed Boltz-2 monomer pre-filter. "
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
            boltz_devices=args.boltz_devices,
            n_tag=args.n_tag, c_tag=args.c_tag,
            force_site=args.force_site, site_max_distance=args.site_max_distance,
            holo_ligands=([x.strip() for x in args.holo_ligands.split(",")] if args.holo_ligands else None))
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
                boltz_devices=args.boltz_devices,
                n_tag=args.n_tag, c_tag=args.c_tag)

        # For native designs: ensure complex_cif or binder_pdb is set
        # (BindCraft stores binder_pdb, BoltzGen stores complex_cif,
        # PXDesign stores binder_pdb)
        for nd in native_designs:
            if not nd.get("complex_cif") and not nd.get("binder_pdb"):
                log(f"  Warning: {nd['design_id']} has no structure file")

        # Recombine for downstream processing
        passing = backbone_designs + native_designs

    # Measure whenever a site is known. Measuring is cheap (seconds on 12 CPUs)
    # and the columns are how anyone judges whether a design hit the epitope;
    # gating the MEASUREMENT on a filter flag meant site_geometry_status was
    # blank by default, so "never measured" and "never asked for" looked the
    # same in rankings.csv. Only the FILTERING is opt-in, below.
    need_geometric = bool(site_resnums)
    if need_geometric:
        max_dist = args.max_site_dist if args.max_site_dist > 0 else 999.0
        log(f"\n{'─' * 50}")
        log("STEP: Geometric site measurement")
        log(f"{'─' * 50}")
        geometric_site_filter(all_designs, site_resnums=site_resnums,
                              target_residues=tchain.get("residues"),
                              max_dist=max_dist,
                              min_site_fraction=args.min_site_fraction,
                              interface_dist=args.interface_dist,
                              dry_run=args.dry_run,
                              target_chain_id=chain_id)

    # Always measured, never auto-applied — see compute_binder_compactness.
    log(f"\n{'─' * 50}")
    log("STEP: Binder compactness (Rg ratio)")
    log(f"{'─' * 50}")
    compute_binder_compactness(all_designs,
                               target_residues=tchain.get("residues"),
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
    populate_interface_composition(all_designs, target_chain_id=chain_id,
                                   target_residues=tchain.get("residues"),
                                   interface_dist=args.interface_dist)
    if not args.dry_run:
        populate_binder_ss(all_designs)
    else:
        log("  (SS skipped in dry-run mode)")

    # Refolding RMSD + solubility. rankings.csv has always carried
    # refolding_rmsd and netsolp_solubility columns, but only rerank_binders.py
    # ever computed them — so a design run emitted both columns permanently
    # blank, and --max_refolding_rmsd (documented as a core quality filter) had
    # nothing to filter on unless you re-ranked afterwards.
    log(f"\n{'─' * 50}")
    log("STEP: Refolding RMSD")
    log(f"{'─' * 50}")
    compute_refolding_rmsd(all_designs, dry_run=args.dry_run,
                           n_tag=args.n_tag, c_tag=args.c_tag)

    log(f"\n{'─' * 50}")
    log("STEP: Solubility prediction (NetSolP)")
    log(f"{'─' * 50}")
    compute_solubility(all_designs, dry_run=args.dry_run)

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

    # ── Geometric site filters (post-rank) ────────────────────────────────
    # These flags were parsed, documented as filters, and applied NOWHERE in this
    # script: geometric_site_filter() computes site_interface_fraction and the
    # centroid distances and writes them to rankings.csv, but never dropped a
    # design on them. Only rerank_binders.py honoured them, so a run passing
    # --min_site_interface_fraction got an unfiltered panel with a
    # plausible-looking column. Ported here to match rerank's behaviour.
    if args.min_site_interface_fraction is not None:
        apply_post_rank_filter(
            all_ranked, "site_interface_fraction",
            args.min_site_interface_fraction, higher_is_better=True,
            label="Site interface filter (SIF)", fmt="{:.0%}")
    if args.max_site_centroid_dist is not None:
        apply_post_rank_filter(
            all_ranked, f"site_centroid_dist_{args.centroid_atoms}",
            args.max_site_centroid_dist, higher_is_better=False,
            label=f"Centroid distance filter ({args.centroid_atoms})",
            fmt="{:.1f} Å")
    if getattr(args, "min_site_cos", None) is not None:
        apply_post_rank_filter(
            all_ranked, "site_cos_angle",
            args.min_site_cos, higher_is_better=True,
            label="Surface-normal filter (site cos)", fmt="{:.2f}")
    if getattr(args, "max_rg_ratio", None) is not None:
        apply_post_rank_filter(
            all_ranked, "binder_rg_ratio",
            args.max_rg_ratio, higher_is_better=False,
            label="Compactness filter (Rg ratio)", fmt="{:.2f}")
    if getattr(args, "require_site_geometry", False):
        # Must follow the same contract as apply_post_rank_filter: mark rejects
        # with rank=None + NaN score and KEEP them in all_ranked, then re-sort
        # and renumber only the survivors.
        #
        # The first version of this block filtered the list in place and did
        # `enumerate(all_ranked)`, which was wrong three ways: it resurrected
        # designs that SIF/centroid/compactness had already set to rank=None
        # (those filters mark, they do not remove), it assigned ranks in LIST
        # order rather than score order, and it deleted rows that the rest of
        # the pipeline expects to still be present in rankings.csv as unranked.
        before = sum(1 for d in all_ranked if d.get("rank") is not None)
        for d in all_ranked:
            if d.get("rank") is None:
                continue
            if d.get("site_geometry_status") != "measured":
                d["combined_score"] = float("nan")
                d["rank"] = None
        _ranked = [d for d in all_ranked if d.get("rank") is not None]
        _ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, d in enumerate(_ranked):
            d["rank"] = i + 1
        if before - len(_ranked):
            log(f"  Site geometry requirement: removed {before - len(_ranked)}/{before} "
                f"designs whose site geometry was never measured")

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
                     target_residues=tchain.get("residues"),
                     # rerank passes this; generate did not, so the site-cos
                     # acceptance cone was never drawn on a design run even
                     # though --min_site_cos is now a valid generate flag.
                     cos_angle=getattr(args, "min_site_cos", None))
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
                f"monomer_pLDDT={d.get('esmfold_plddt', 0):.1f}  "
                f"minPAE={pae_str}")

    args._tool_failures = tool_failures
    # A dry run must NOT leave a status.json. The documented poll is "status.json
    # exists => the run is over", so a dry run into the directory you are about to
    # use for real plants a file saying ok=false, designs_ranked=0 — and an agent
    # following the procedure literally reports a just-launched 4-hour job as
    # finished and failed. The dry run's own result is its exit code.
    if args.dry_run:
        log("  (dry run: status.json intentionally not written — its presence is "
            "the signal that a REAL run has finished)")
    else:
        write_status_json(out_dir / "status.json", args, tool_results, all_ranked, validated,
                          target_path=target_path, chain_id=chain_id,
                          site_resnums=site_resnums, target_residues=tchain.get("residues"))

    log(f"\nOutput directory: {out_dir}")
    log(f"  rankings.csv   — {len(all_ranked)} total designs")
    log(f"  top_designs/   — top {min(args.top_n, len(validated))} structures")
    log(f"  dashboard.png  — summary plot")
    log(f"  status.json    — machine-readable run outcome")

    # The run used to exit 0 even when every tool crashed and nothing was ranked,
    # so rc carried no information. Now: 0 = everything asked for succeeded and
    # something is ranked; 3 = ran but produced no ranked design; 4 = ranked
    # something, but at least one tool failed (partial success).
    # Argument/validation errors stay 1, argparse stays 2. Details in status.json.
    if args.dry_run:
        # A dry run validates arguments and prints commands; its mock designs are
        # not evidence of anything, so it must not report a run failure.
        return 0
    if not validated:
        log("EXIT 3: no design could be ranked. See status.json for per-tool detail.",
            err=True)
        return 3
    if tool_failures:
        log(f"EXIT 4: partial success — {len(validated)} design(s) ranked, but "
            f"{len(tool_failures)} tool(s) failed: "
            + ", ".join(f"{k}[{v}]" for k, v in sorted(tool_failures.items())), err=True)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
