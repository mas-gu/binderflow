#!/usr/bin/env python3
"""
generate_molecules.py — Structure-based drug design (SBDD) pipeline.

Extends the protein folding platform with small molecule design:

  pocketflow    — Autoregressive flow-based 3D molecule generation (PocketFlow)
  molcraft      — Bayesian flow network SBDD with optimal scheduling (MolCRAFT/MolPilot)
  pocketxmol    — Foundation model for pocket-conditioned generation (PocketXMol)

All tools take a protein pocket as input and generate SDF files with 3D molecules.
Scoring uses QED, SA, and optionally AutoDock Vina re-docking.

===============================================================================
QUICK START
===============================================================================
    conda activate boltz

    # Generate molecules for a binding pocket
    python generate_molecules.py \
        --target target.pdb \
        --site "A:11-17,119-124" \
        --tools pocketflow,molcraft,pocketxmol \
        --mode test \
        --out_dir ./molecules/

    # Single tool, no Vina (faster)
    python generate_molecules.py \
        --target target.pdb \
        --site "A:325-330" \
        --tools pocketflow \
        --mode test \
        --no_vina \
        --out_dir ./molecules/

    # Virtual screening of a compound library
    python generate_molecules.py \
        --target target.pdb \
        --site "A:325-330" \
        --library compounds.sdf \
        --out_dir ./screening/

    # Library screening with SMILES file (3D conformers generated automatically)
    python generate_molecules.py \
        --target target.pdb \
        --site "A:11-17,119-124" \
        --library compounds.smi \
        --no_vina \
        --out_dir ./screening/

    # Combine de novo + library screening
    python generate_molecules.py \
        --target target.pdb \
        --site "A:325-330" \
        --tools pocketflow \
        --library compounds.sdf \
        --mode test \
        --out_dir ./combined/

===============================================================================
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Import shared functions from generate_binders ─────────────────────────────
import generate_binders as _gb
from generate_binders import (
    log, run_cmd, _safe_float,
    parse_site,
)
from pocket_utils import extract_pocket_pdb, compute_pocket_center, compute_bbox_size
from mol_scoring import (
    parse_sdf, score_molecules, rank_molecules, write_rankings_csv,
    compute_diversity,
)

# ── Constants ─────────────────────────────────────────────────────────────────

from config_loader import cfg

POCKETFLOW_DIR  = cfg.tool("pocketflow_dir", default=f"{cfg.software_dir}/PocketFlow")
POCKETFLOW_CKPT = cfg.tool("pocketflow_ckpt",
                            default=f"{POCKETFLOW_DIR}/ckpt/ZINC-pretrained-255000_cpu.pt")

MOLCRAFT_DIR    = cfg.tool("molcraft_dir", default=f"{cfg.software_dir}/MolCRAFT")
MOLCRAFT_CKPT   = cfg.tool("molcraft_ckpt",
                            default=f"{cfg.weights_dir}/molcraft/molpilot_epoch26-val_loss5.42-mol_stable0.48-complete0.83.ckpt")
MOLCRAFT_SCHED  = cfg.tool("molcraft_sched",
                            default=f"{cfg.weights_dir}/molcraft/optimized_path_rect_closest.pt")

POCKETXMOL_DIR  = cfg.tool("pocketxmol_dir", default=f"{cfg.software_dir}/PocketXMol")
POCKETXMOL_CKPT = cfg.tool("pocketxmol_ckpt",
                            default=f"{cfg.weights_dir}/pocketxmol/data/trained_models/pxm/checkpoints/pocketxmol.ckpt")

TOOL_COLORS = {
    "pocketflow":  "#E53935",   # red
    "molcraft":    "#7C3AED",   # purple
    "pocketxmol":  "#0EA5E9",   # sky blue
    "library":     "#FF9800",   # orange (also used for library_* prefixed tools)
}

def _tool_color(name):
    """Get color for a tool name, handling library_* prefix."""
    return TOOL_COLORS.get(name, TOOL_COLORS["library"] if name.startswith("library") else "#888888")

ALL_TOOLS = ["pocketflow", "molcraft", "pocketxmol"]

DESIGN_MODES = {
    "test":       {"pocketflow": 100, "molcraft": 100, "pocketxmol": 100},
    "standard":   {"pocketflow": 1000, "molcraft": 1000, "pocketxmol": 1000},
    "production": {"pocketflow": 5000, "molcraft": 5000, "pocketxmol": 5000},
}
DEFAULT_MODE = "standard"


# ── PocketFlow ────────────────────────────────────────────────────────────────

def run_pocketflow(pocket_pdb, n_mols, out_dir, max_atoms=35, device="cuda:0",
                   dry_run=False):
    """
    Run PocketFlow molecule generation.

    Input: pocket PDB
    Output: SDF files in out_dir/
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(POCKETFLOW_DIR).exists() and not dry_run:
        raise FileNotFoundError(
            f"PocketFlow not found at {POCKETFLOW_DIR}.\n"
            f"Install: see POCKETFLOW/INSTALL_POCKETFLOW.md")

    log(f"PocketFlow: generating {n_mols} molecules (max_atoms={max_atoms})")

    # PocketFlow uses CUDA_VISIBLE_DEVICES + cuda:0 for GPU pinning
    # (its main_generate.py calls torch.cuda.init() early to avoid
    # the PyTorch 2.0.0 deferred capability check bug)
    gpu_id = device.split(":")[1] if ":" in device else "0"

    pf_cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("pocketflow"),
        "python", f"{POCKETFLOW_DIR}/main_generate.py",
        "-pkt", str(Path(pocket_pdb).resolve()),
        "--ckpt", POCKETFLOW_CKPT,
        "-n", str(n_mols),
        "-d", "cuda:0",
        "--max_atom_num", str(max_atoms),
        "--root_path", str(out_dir.resolve()),
        "--with_print", "0",
    ]
    pf_env = {"CUDA_VISIBLE_DEVICES": gpu_id}

    run_cmd(pf_cmd, timeout=None, cwd=POCKETFLOW_DIR, extra_env=pf_env,
            dry_run=dry_run)

    # PocketFlow saves SDFs in root_path/{name}/SDF/
    sdf_files = sorted(out_dir.rglob("*.sdf"))
    if not sdf_files and not dry_run:
        log("  Warning: PocketFlow produced no SDF files")
        return []

    # Collect all SDFs into a single combined file
    all_sdfs = out_dir / "all_molecules.sdf"
    if not dry_run:
        with open(all_sdfs, "w") as f:
            for sdf in sdf_files:
                f.write(sdf.read_text())
        log(f"PocketFlow: collected {len(sdf_files)} SDF files")
    else:
        sdf_files = [out_dir / "dummy.sdf"]

    return sdf_files


# ── MolCRAFT (MolPilot) ──────────────────────────────────────────────────────

def run_molcraft(protein_pdb, pocket_pdb, center, n_mols, out_dir,
                 device="cuda:0", dry_run=False):
    """
    Run MolCRAFT/MolPilot molecule generation.

    MolCRAFT needs a reference ligand to define the pocket. We generate a dummy
    SDF with a single carbon at the pocket center to bootstrap pocket detection.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    molpilot_dir = Path(MOLCRAFT_DIR) / "MolPilot"
    if not molpilot_dir.exists() and not dry_run:
        raise FileNotFoundError(f"MolCRAFT/MolPilot not found at {molpilot_dir}")

    log(f"MolCRAFT: generating {n_mols} molecules (MolPilot)")

    # Create a dummy reference ligand SDF at pocket center
    # MolCRAFT uses this to define the 10Å pocket around the ligand AND as
    # a template for atom count / bond features.  A single-atom molecule
    # causes a degenerate (1-D) bond-index array → IndexError in the model.
    # Use benzene (6 heavy atoms, 6 bonds) to provide a valid template.
    ref_sdf = str(out_dir / "ref_ligand.sdf")
    if not dry_run:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        # Center the molecule at the pocket center
        conf = mol.GetConformer()
        centroid = conf.GetPositions().mean(axis=0)
        shift = [center[0] - centroid[0],
                 center[1] - centroid[1],
                 center[2] - centroid[2]]
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (pos.x + shift[0],
                                     pos.y + shift[1],
                                     pos.z + shift[2]))
        mol = Chem.RemoveHs(mol)
        w = Chem.SDWriter(ref_sdf)
        w.write(mol)
        w.close()

    gpu_id = device.split(":")[1] if ":" in device else "0"

    mc_cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("molcraft"),
        "python", str(molpilot_dir / "sample_for_pocket.py"),
        "--config_file", str(molpilot_dir / "configs" / "crossdock_train_test.yaml"),
        "--protein_path", str(Path(protein_pdb).resolve()),
        "--ligand_path", str(Path(ref_sdf).resolve()),
        "--ckpt_path", MOLCRAFT_CKPT,
        "--time_scheduler_path", MOLCRAFT_SCHED,
        "--num_samples", str(n_mols),
        "--sample_steps", "100",
        "--no_wandb",
        "--skip_chem",
        "--self_condition",
        "--time_emb_dim", "0",
        "--sample_num_atoms", "prior",  # use learned size distribution, not ref ligand count
        "--output_dir", str(out_dir.resolve()),
        "--res_path", str((out_dir / "results.json").resolve()),
    ]
    mc_env = {"CUDA_VISIBLE_DEVICES": gpu_id, "WANDB_MODE": "disabled",
              "PYTHONPATH": str(molpilot_dir)}

    # MolCRAFT + wandb opens many temp files at scale (>100 molecules),
    # hitting "Too many open files". Raise soft limit in this process
    # (inherited by subprocess).
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65536:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    except Exception:
        pass

    # MolCRAFT's sample_for_pocket.py runs internal Vina evaluation after
    # generation. This can crash on molecules with extreme coordinates
    # (PDBQT format overflow). The SDFs are already written before eval,
    # so we catch the error and collect whatever was generated.
    try:
        run_cmd(mc_cmd, timeout=None, cwd=str(molpilot_dir), extra_env=mc_env,
                dry_run=dry_run)
    except RuntimeError as e:
        err = str(e)
        if "PDBQT parsing error" in err or "not valid" in err:
            log("  MolCRAFT: internal Vina eval failed (coordinate overflow) — collecting generated SDFs")
        else:
            raise

    sdf_files = sorted(out_dir.rglob("*.sdf"))
    # Exclude ref_ligand.sdf
    sdf_files = [f for f in sdf_files if f.name != "ref_ligand.sdf"]
    if not sdf_files and not dry_run:
        log("  Warning: MolCRAFT produced no SDF files")
        return []

    log(f"MolCRAFT: collected {len(sdf_files)} SDF files")
    return sdf_files


# ── PocketXMol ───────────────────────────────────────────────────────────────

def run_pocketxmol(protein_pdb, center, bbox_size, n_mols, out_dir,
                   device="cuda:0", dry_run=False):
    """
    Run PocketXMol molecule generation (SBDD mode with AR refinement).

    PocketXMol is a foundation model from the Pocket2Mol authors (Cell 2026).
    Uses YAML config for task specification.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(POCKETXMOL_DIR).exists() and not dry_run:
        raise FileNotFoundError(f"PocketXMol not found at {POCKETXMOL_DIR}")

    log(f"PocketXMol: generating {n_mols} molecules (SBDD + AR refinement)")

    gpu_id = device.split(":")[1] if ":" in device else "0"

    # Generate YAML task config — must include all fields that sample_use.py
    # accesses directly (seed, save_traj_prob, task.transform, noise AR config).
    config_path = out_dir / "task_config.yml"
    if not dry_run:
        import yaml
        cx, cy, cz = round(center[0], 3), round(center[1], 3), round(center[2], 3)
        task_config = {
            "sample": {
                "seed": 2024,
                "num_mols": n_mols,
                "num_steps": 100,
                "batch_size": min(n_mols, 100),
                "save_traj_prob": 0.0,  # don't save trajectory (saves disk)
            },
            "data": {
                "protein_path": str(Path(protein_pdb).resolve()),
                "is_pep": False,
                "pocket_args": {
                    "pocket_coord": [cx, cy, cz],
                    "radius": 15.0,
                },
                "pocmol_args": {
                    "data_id": "sbdd_custom",
                    "pdbid": "custom",
                },
            },
            "transforms": {
                "featurizer_pocket": {
                    "center": [cx, cy, cz],
                },
                "variable_mol_size": {
                    "name": "variable_mol_size",
                    "num_atoms_distri": {
                        "strategy": "mol_atoms_based",
                        "mean": {"coef": 0, "bias": 28},
                        "std": {"coef": 0, "bias": 8},
                        "min": 5,
                    },
                },
            },
            "task": {
                "name": "sbdd",
                "transform": {
                    "name": "ar",
                    "part1_pert": "small",
                },
            },
            "noise": {
                "name": "maskfill",
                "num_steps": 100,
                "ar_config": {
                    "strategy": "refine",
                    "r": 3,
                    "threshold_node": 0.98,
                    "threshold_pos": 0.91,
                    "threshold_bond": 0.98,
                    "max_ar_step": 10,
                    "change_init_step": 1,
                },
                "prior": {
                    "part1": "from_train",
                    "part2": "from_train",
                },
                "level": {
                    "part1": {
                        "name": "uniform",
                        "min": 0.6,
                        "max": 1.0,
                    },
                    "part2": {
                        "name": "advance",
                        "min": 0.0,
                        "max": 1.0,
                        "step2level": {
                            "scale_start": 0.99999,
                            "scale_end": 1.0e-05,
                            "width": 3,
                        },
                    },
                },
            },
        }
        config_path.write_text(yaml.dump(task_config, default_flow_style=False))

    px_cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("pocketxmol"),
        "python", "scripts/sample_use.py",
        "--config_task", str(config_path.resolve()),
        "--config_model", str(Path(POCKETXMOL_DIR) / "configs" / "sample" / "pxm.yml"),
        "--outdir", str(out_dir.resolve()),
        "--device", "cuda:0",
        "--batch_size", str(min(n_mols, 100)),
    ]
    px_env = {"CUDA_VISIBLE_DEVICES": gpu_id}

    run_cmd(px_cmd, timeout=None, cwd=str(POCKETXMOL_DIR), extra_env=px_env,
            dry_run=dry_run)

    # PocketXMol outputs SDFs in {outdir}/*_SDF/ subdirectory
    sdf_files = sorted(out_dir.rglob("*.sdf"))
    sdf_files = [f for f in sdf_files if "config" not in f.name]
    if not sdf_files and not dry_run:
        log("  Warning: PocketXMol produced no SDF files")
        return []

    log(f"PocketXMol: collected {len(sdf_files)} SDF files")
    return sdf_files


# ── Parallel worker functions ─────────────────────────────────────────────────

def _prepare_receptor_pdbqt(protein_pdb):
    """Convert PDB to PDBQT once for parallel Vina docking."""
    if not protein_pdb.endswith(".pdb"):
        return protein_pdb
    receptor_pdbqt = protein_pdb.replace(".pdb", "_receptor.pdbqt")
    if os.path.exists(receptor_pdbqt):
        return receptor_pdbqt
    try:
        subprocess.run(["obabel", protein_pdb, "-O", receptor_pdbqt,
                        "-xr", "-xc", "-xn"],
                       capture_output=True, timeout=60, check=True)
        log(f"  Receptor PDBQT: {receptor_pdbqt}")
        return receptor_pdbqt
    except Exception:
        return protein_pdb


def _vina_dock_worker(args):
    """Worker for parallel Vina docking. Returns (index, vina_score, docked_sdf_path)."""
    idx, sdf_path, receptor_pdbqt, center, bbox_size, docked_sdf = args
    from mol_scoring import compute_vina_score
    score = compute_vina_score(receptor_pdbqt, sdf_path, center, bbox_size,
                               docked_sdf_path=docked_sdf)
    docked_ok = docked_sdf and os.path.exists(docked_sdf) and os.path.getsize(docked_sdf) > 0
    return idx, score, docked_sdf if docked_ok else None


# Metals that cause RDKit BoundsMatrixBuilder invariant violations
_METAL_ATOMS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Ra", "Ac", "Th", "U", "Xe",
}


def _has_metals(mol):
    """Check if molecule contains metal atoms that crash RDKit embedding."""
    return any(a.GetSymbol() in _METAL_ATOMS for a in mol.GetAtoms())


def _embed_mol_worker(args):
    """Worker for parallel 3D embedding. Returns (index, mol_or_None).

    Skips metal-containing molecules (cause RDKit C++ invariant violations)
    and catches all other embedding failures.
    """
    idx, smi, name, seed = args
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return idx, None
        if _has_metals(mol):
            return idx, None
        mol.SetProp("_Name", name)
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return idx, None
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        return idx, Chem.RemoveHs(mol_h)
    except Exception:
        return idx, None


# ── Uni-Dock GPU batch docking ────────────────────────────────────────────────

def _run_unidock_batch(receptor_pdbqt, sdf_paths, center, bbox_size, out_dir):
    """
    Batch-dock molecules using Uni-Dock (GPU-accelerated Vina).
    Returns dict: sdf_path -> (vina_score, docked_sdf_path_or_None).
    Falls back to None if Uni-Dock is not installed.
    """
    unidock_env = cfg.conda_env("unidock")
    unidock_bin = os.path.join(unidock_env, "bin", "unidock") if os.path.isabs(unidock_env) else None
    if not unidock_bin or not os.path.exists(unidock_bin):
        return None  # Uni-Dock not available

    from rdkit import Chem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    # Prepare all ligand PDBQT files
    pdbqt_dir = Path(out_dir) / "_unidock_pdbqt"
    pdbqt_dir.mkdir(parents=True, exist_ok=True)
    docked_dir = Path(out_dir) / "_unidock_docked"
    docked_dir.mkdir(parents=True, exist_ok=True)

    pdbqt_paths = []
    sdf_to_pdbqt = {}

    for sdf_path in sdf_paths:
        try:
            mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
            if mol is None:
                continue
            mol = Chem.AddHs(mol, addCoords=True)
            if mol.GetNumConformers() == 0:
                from rdkit.Chem import AllChem
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            prep = MoleculePreparation()
            mol_setups = prep.prepare(mol)
            pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
            if not is_ok:
                continue

            # Validate PDBQT: must have atoms and parseable coordinates
            # (no box check — Uni-Dock handles placement into search box)
            n_atoms = 0
            valid = True
            for line in pdbqt_str.splitlines():
                if line.startswith(("ATOM", "HETATM")):
                    n_atoms += 1
                    try:
                        float(line[30:38]); float(line[38:46]); float(line[46:54])
                    except (ValueError, IndexError):
                        valid = False
                        break
            if n_atoms == 0 or not valid:
                continue

            stem = Path(sdf_path).stem
            pdbqt_file = str(pdbqt_dir / f"{stem}.pdbqt")
            with open(pdbqt_file, "w") as f:
                f.write(pdbqt_str)
            pdbqt_paths.append(pdbqt_file)
            sdf_to_pdbqt[sdf_path] = pdbqt_file
        except Exception:
            continue

    if not pdbqt_paths:
        return None

    # Run Uni-Dock in chunks via --ligand_index (file-based, no ARG_MAX limit).
    # Per-chunk strategy: try GPU, if chunk fails → dock that chunk on CPU Vina,
    # then continue next chunk on GPU. Isolates segfaults from bad molecules.
    # Uni-Dock batch size — limited by GPU memory. Some DrugBank molecules
    # crash Uni-Dock even in small batches. 25 minimizes CPU fallback scope.
    UNIDOCK_CHUNK = 25
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env = {"CUDA_VISIBLE_DEVICES": gpu_id}
    n_chunks = (len(pdbqt_paths) + UNIDOCK_CHUNK - 1) // UNIDOCK_CHUNK
    n_docked_gpu = 0
    n_docked_cpu = 0

    # Reverse map: pdbqt → sdf for CPU fallback
    pdbqt_to_sdf = {v: k for k, v in sdf_to_pdbqt.items()}

    for chunk_i in range(n_chunks):
        chunk = pdbqt_paths[chunk_i * UNIDOCK_CHUNK:(chunk_i + 1) * UNIDOCK_CHUNK]
        ligand_index = pdbqt_dir / f"ligand_index_{chunk_i}.txt"
        ligand_index.write_text("\n".join(chunk) + "\n")

        cmd = [
            "conda", "run", "--no-capture-output", *cfg.conda_run_args("unidock"),
            "unidock",
            "--receptor", str(receptor_pdbqt),
            "--ligand_index", str(ligand_index),
            "--search_mode", "balance",
            "--scoring", "vina",
            "--center_x", f"{center[0]:.3f}",
            "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{bbox_size:.1f}",
            "--size_y", f"{bbox_size:.1f}",
            "--size_z", f"{bbox_size:.1f}",
            "--num_modes", "1",
            "--dir", str(docked_dir),
        ]

        try:
            run_cmd(cmd, timeout=3600, extra_env=env)
            n_docked_gpu += len(chunk)
            if n_chunks > 1:
                log(f"  GPU chunk {chunk_i + 1}/{n_chunks}: {len(chunk)} OK")
        except Exception:
            # GPU failed for this chunk — dock on CPU Vina per molecule
            chunk_sdfs = [pdbqt_to_sdf[p] for p in chunk if p in pdbqt_to_sdf]
            if chunk_sdfs:
                n_workers = min(max(1, (cpu_count() or 4) - 2), 12, len(chunk_sdfs))
                log(f"  GPU chunk {chunk_i + 1}/{n_chunks} failed → CPU ({len(chunk_sdfs)} mols, {n_workers} workers)")
                cpu_args = []
                for i, sdf_path in enumerate(chunk_sdfs):
                    docked_sdf = str(Path(sdf_path).parent / f"{Path(sdf_path).stem}_docked.sdf")
                    cpu_args.append((i, sdf_path, str(receptor_pdbqt), center, bbox_size, docked_sdf))
                try:
                    with Pool(n_workers) as pool:
                        for idx, score, docked_path in pool.imap_unordered(_vina_dock_worker, cpu_args):
                            # Store result so parse loop picks it up via sdf_to_pdbqt mapping
                            sdf_p = cpu_args[idx][1]
                            if sdf_p in sdf_to_pdbqt:
                                stem = Path(sdf_to_pdbqt[sdf_p]).stem
                                # Write score file for the PDBQT parse loop
                                score_file = docked_dir / f"{stem}_out.pdbqt"
                                score_file.write_text(
                                    f"REMARK VINA RESULT:    {score:.3f}      0.000      0.000\n")
                            n_docked_cpu += 1
                except Exception as e2:
                    log(f"  CPU fallback also failed: {str(e2)[:80]}")

    total = n_docked_gpu + n_docked_cpu
    if total == 0:
        log(f"  Uni-Dock: all chunks failed (GPU + CPU)")
        return None

    if n_docked_gpu > 0 and n_docked_cpu > 0:
        log(f"  Docking: {n_docked_gpu} GPU + {n_docked_cpu} CPU = {total} total")
    elif n_docked_gpu > 0:
        pass  # summary logged by caller

    # Parse results — Uni-Dock writes docked PDBQT files with scores in remarks
    results = {}
    for sdf_path, pdbqt_file in sdf_to_pdbqt.items():
        stem = Path(pdbqt_file).stem
        docked_pdbqt = docked_dir / f"{stem}_out.pdbqt"
        if not docked_pdbqt.exists():
            results[sdf_path] = (float("nan"), None)
            continue

        # Parse score from first MODEL in docked PDBQT
        score = float("nan")
        try:
            with open(docked_pdbqt) as f:
                for line in f:
                    if line.startswith("REMARK VINA RESULT:"):
                        score = float(line.split()[3])
                        break
        except Exception:
            pass

        # Convert docked PDBQT to SDF (or use existing docked SDF from CPU fallback)
        docked_sdf = str(Path(sdf_path).parent / f"{Path(sdf_path).stem}_docked.sdf")
        if os.path.exists(docked_sdf) and os.path.getsize(docked_sdf) > 0:
            # CPU fallback already created the docked SDF
            results[sdf_path] = (score, docked_sdf)
        elif docked_pdbqt.stat().st_size > 200:
            # GPU docked — convert PDBQT to SDF via obabel
            try:
                subprocess.run(["obabel", str(docked_pdbqt), "-O", docked_sdf],
                              capture_output=True, timeout=30)
            except Exception:
                pass
            if os.path.exists(docked_sdf) and os.path.getsize(docked_sdf) > 0:
                results[sdf_path] = (score, docked_sdf)
            else:
                results[sdf_path] = (score, None)
        else:
            results[sdf_path] = (score, None)

    # Cleanup temp PDBQT files
    import shutil
    shutil.rmtree(pdbqt_dir, ignore_errors=True)
    shutil.rmtree(docked_dir, ignore_errors=True)

    return results


# ── Scoring pipeline ──────────────────────────────────────────────────────────

def collect_and_score(tool_name, sdf_files, protein_pdb, center, bbox_size,
                      use_vina=True):
    """
    Parse SDF files from a tool, deduplicate, and score all molecules.
    Returns list of molecule dicts.
    """
    from rdkit import Chem
    all_mols = []
    seen_smiles = set()

    for sdf_path in sdf_files:
        try:
            suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        except Exception:
            continue
        for mol in suppl:
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                continue
            if smi in seen_smiles:
                continue
            seen_smiles.add(smi)
            all_mols.append((mol, smi, str(sdf_path)))

    if not all_mols:
        return []

    from mol_scoring import (
        compute_qed, compute_sa_score, compute_lipinski,
    )

    # Create per-molecule SDF directory for individual files
    mol_sdf_dir = Path(sdf_files[0]).parent / f"{tool_name}_individual"
    mol_sdf_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (mol, smi, sdf_path) in enumerate(all_mols):
        design_id = f"{tool_name}_{i:04d}"

        # Save individual SDF for this molecule
        indiv_sdf = str(mol_sdf_dir / f"{design_id}.sdf")
        w = Chem.SDWriter(indiv_sdf)
        w.write(mol)
        w.close()

        entry = {
            "design_id": design_id,
            "tool": tool_name,
            "smiles": smi,
            "sdf_path": indiv_sdf,
            "qed": compute_qed(mol),
            "sa_score": compute_sa_score(mol),
            "vina_score": float("nan"),
        }
        entry.update(compute_lipinski(mol))
        results.append(entry)

    # ── Docking: try Uni-Dock (GPU) first, fall back to Vina (CPU) ─────────
    if use_vina and center and results:
        receptor_pdbqt = _prepare_receptor_pdbqt(protein_pdb)
        sdf_paths = [entry["sdf_path"] for entry in results]
        n_total = len(sdf_paths)

        # Try Uni-Dock GPU batch docking
        log(f"  Docking {n_total} molecules...")
        unidock_results = _run_unidock_batch(
            receptor_pdbqt, sdf_paths, center, bbox_size,
            str(Path(sdf_paths[0]).parent))

        if unidock_results is not None:
            n_with_score = sum(1 for v, _ in unidock_results.values() if v == v)
            log(f"  Uni-Dock GPU: {n_with_score} molecules docked")
            for i, entry in enumerate(results):
                sdf_path = entry["sdf_path"]
                if sdf_path in unidock_results:
                    score, docked_path = unidock_results[sdf_path]
                    entry["vina_score"] = score
                    if docked_path:
                        entry["sdf_path"] = docked_path
        else:
            # Fallback: parallel Vina on CPU
            vina_args = []
            for i, entry in enumerate(results):
                docked_sdf = str(Path(entry["sdf_path"]).parent /
                               f"{entry['design_id']}_docked.sdf")
                vina_args.append((i, entry["sdf_path"], receptor_pdbqt, center,
                                 bbox_size, docked_sdf))

            n_workers = min(max(1, (cpu_count() or 4) - 2), 20, len(vina_args))
            log(f"  Vina CPU fallback: {n_total} molecules on {n_workers} CPUs")
            with Pool(n_workers) as pool:
                done = 0
                for idx, score, docked_path in pool.imap_unordered(_vina_dock_worker, vina_args):
                    results[idx]["vina_score"] = score
                    if docked_path:
                        results[idx]["sdf_path"] = docked_path
                    done += 1
                    if done % 20 == 0 or done == n_total:
                        log(f"  [PROGRESS:vina] {done}/{n_total} docked")

    return results


# ── Dashboard ─────────────────────────────────────────────────────────────────

def plot_mol_dashboard(all_ranked, out_path, title="SBDD Pipeline"):
    """Generate 6-panel summary dashboard."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    tools_present = sorted(set(d.get("tool", "unknown") for d in all_ranked))

    # Panel 1: Combined score boxplot by tool
    ax1 = fig.add_subplot(gs[0, 0])
    tool_scores = {t: [] for t in tools_present}
    for d in all_ranked:
        t = d.get("tool", "unknown")
        s = d.get("combined_score", 0)
        if s == s:  # not NaN
            tool_scores[t].append(s)
    data_bp = [tool_scores[t] for t in tools_present]
    colors = [_tool_color(t) for t in tools_present]
    bp = ax1.boxplot(data_bp, tick_labels=tools_present, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax1.set_ylabel("Combined Score")
    ax1.set_title("Combined Score by Tool")

    # Panel 2: QED distribution histogram
    ax2 = fig.add_subplot(gs[0, 1])
    for t in tools_present:
        vals = [d["qed"] for d in all_ranked
                if d.get("tool") == t and d.get("qed", 0) == d.get("qed", 0)]
        if vals:
            ax2.hist(vals, bins=20, alpha=0.6, label=t,
                     color=_tool_color(t))
    ax2.set_xlabel("QED")
    ax2.set_ylabel("Count")
    ax2.set_title("QED Distribution")
    ax2.legend(fontsize=8)

    # Panel 3: Vina score vs QED scatter
    ax3 = fig.add_subplot(gs[0, 2])
    for t in tools_present:
        vina = [d.get("vina_score", float("nan")) for d in all_ranked if d.get("tool") == t]
        qed = [d.get("qed", float("nan")) for d in all_ranked if d.get("tool") == t]
        valid = [(v, q) for v, q in zip(vina, qed) if v == v and q == q]
        if valid:
            vs, qs = zip(*valid)
            ax3.scatter(qs, vs, alpha=0.5, s=15, label=t,
                        color=_tool_color(t))
    ax3.set_xlabel("QED")
    ax3.set_ylabel("Vina Score (kcal/mol)")
    ax3.set_title("Vina vs QED")
    ax3.legend(fontsize=8)

    # Panel 4: Top 20 molecules bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    top20 = all_ranked[:20]
    if top20:
        names = [d["design_id"] for d in top20]
        scores = [d.get("combined_score", 0) for d in top20]
        bar_colors = [_tool_color(d.get("tool", "")) for d in top20]
        ax4.barh(range(len(top20)), scores, color=bar_colors)
        ax4.set_yticks(range(len(top20)))
        ax4.set_yticklabels(names, fontsize=6)
        ax4.invert_yaxis()
        ax4.set_xlabel("Combined Score")
        ax4.set_title("Top 20 Molecules")

    # Panel 5: SA score vs Vina scatter
    ax5 = fig.add_subplot(gs[1, 1])
    for t in tools_present:
        sa = [d.get("sa_score", float("nan")) for d in all_ranked if d.get("tool") == t]
        vina = [d.get("vina_score", float("nan")) for d in all_ranked if d.get("tool") == t]
        valid = [(s, v) for s, v in zip(sa, vina) if s == s and v == v]
        if valid:
            ss, vs = zip(*valid)
            ax5.scatter(ss, vs, alpha=0.5, s=15, label=t,
                        color=_tool_color(t))
    ax5.set_xlabel("SA Score (lower=easier)")
    ax5.set_ylabel("Vina Score (kcal/mol)")
    ax5.set_title("SA vs Vina")
    ax5.legend(fontsize=8)

    # Panel 6: Top 10 properties table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    top10 = all_ranked[:10]
    if top10:
        col_labels = ["Rank", "ID", "QED", "SA", "Vina", "MW", "Score"]
        table_data = []
        for d in top10:
            vina_str = f"{d.get('vina_score', float('nan')):.1f}" if d.get("vina_score", float("nan")) == d.get("vina_score", float("nan")) else "N/A"
            table_data.append([
                d.get("rank", ""),
                d.get("design_id", "")[:16],
                f"{d.get('qed', 0):.2f}",
                f"{d.get('sa_score', 0):.1f}",
                vina_str,
                f"{d.get('mw', 0):.0f}",
                f"{d.get('combined_score', 0):.3f}",
            ])
        table = ax6.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)
        ax6.set_title("Top 10 Properties", fontsize=10, pad=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Dashboard saved: {out_path}")


# ── Library loading ───────────────────────────────────────────────────────────

def load_library(library_path, out_dir, dry_run=False):
    """
    Load molecules from a compound library file (.sdf or .smi/.smiles).

    For SDF: reads molecules with existing 3D coordinates. Re-embeds if
    all z-coordinates are zero (2D layout).
    For SMILES: generates 3D conformers with RDKit ETKDGv3 + MMFF.

    Returns list of SDF file paths (single combined SDF).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    library_path = Path(library_path)

    if dry_run:
        return [out_dir / "library_molecules.sdf"]

    combined_sdf = out_dir / "library_molecules.sdf"
    writer = Chem.SDWriter(str(combined_sdf))
    n_loaded = 0
    n_failed = 0

    suffix = library_path.suffix.lower()

    if suffix == ".sdf":
        log(f"Loading SDF library: {library_path}")
        suppl = Chem.SDMolSupplier(str(library_path), removeHs=False)
        for mol in suppl:
            if mol is None:
                n_failed += 1
                continue
            if _has_metals(mol):
                n_failed += 1
                continue
            try:
                # Check for 2D-only coordinates (all z == 0)
                conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
                if conf is not None:
                    positions = conf.GetPositions()
                    if len(positions) > 0 and np.all(np.abs(positions[:, 2]) < 0.01):
                        # 2D layout — re-embed to 3D
                        mol_h = Chem.AddHs(mol)
                        params = AllChem.ETKDGv3()
                        params.randomSeed = 42
                        if AllChem.EmbedMolecule(mol_h, params) == 0:
                            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                            mol = Chem.RemoveHs(mol_h)
                        else:
                            n_failed += 1
                            continue
                elif mol.GetNumConformers() == 0:
                    # No conformer at all — generate one
                    mol_h = Chem.AddHs(mol)
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    if AllChem.EmbedMolecule(mol_h, params) == 0:
                        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                        mol = Chem.RemoveHs(mol_h)
                    else:
                        n_failed += 1
                        continue
                writer.write(mol)
                n_loaded += 1
            except Exception:
                n_failed += 1

    elif suffix in (".smi", ".smiles"):
        log(f"Loading SMILES library: {library_path}")
        # Phase 1: Read all SMILES (fast)
        smiles_data = []
        header_skipped = False
        with open(library_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Skip header lines (e.g., "smiles zinc_id", "SMILES,Name,MW")
                # Check first non-comment line for header-like content
                if not header_skipped:
                    header_skipped = True
                    low = line.lower().replace(",", " ").split()[0]
                    if low in ("smiles", "smile", "smi", "canonical_smiles",
                               "id", "name", "compound"):
                        continue
                # Handle: "SMILES name" or "SMILES,name" or bare SMILES
                parts = line.replace(",", " ").split()
                smi = parts[0]
                name = parts[1] if len(parts) > 1 else f"lib_{line_no}"
                smiles_data.append((len(smiles_data), smi, name, 42 + line_no))

        log(f"  Read {len(smiles_data)} SMILES, embedding in 3D...")

        # Phase 2: Parallel 3D embedding
        n_workers = min(max(1, (cpu_count() or 4) - 2), 22, len(smiles_data))
        with Pool(n_workers) as pool:
            embed_results = pool.map(_embed_mol_worker, smiles_data)

        # Phase 3: Write successful molecules
        for idx, mol in embed_results:
            if mol is not None:
                writer.write(mol)
                n_loaded += 1
                if n_loaded % 500 == 0:
                    log(f"  ... embedded {n_loaded} molecules")
            else:
                n_failed += 1
    else:
        raise ValueError(f"Unsupported library format: {suffix}")

    writer.close()

    if n_failed > 0:
        log(f"  Library: {n_failed} molecules failed parsing/3D embedding")
    log(f"Library: {n_loaded} molecules loaded to {combined_sdf}")

    return [combined_sdf] if n_loaded > 0 else []


# ── Copy top molecules ────────────────────────────────────────────────────────

def copy_top_molecules(ranked, out_dir, n=50, target_pdb=None, pocket_pdb=None,
                        site_resnums=None):
    """Copy top N molecule SDFs to top_molecules/, write SMILES CSV, and generate
    PyMOL visualization scripts showing molecules in the binding pocket."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top = ranked[:n]
    copied_sdfs = []
    smiles_rows = []
    for d in top:
        sdf_src = d.get("sdf_path", "")
        if sdf_src and Path(sdf_src).exists():
            dst = out_dir / f"{d['design_id']}.sdf"
            shutil.copy2(sdf_src, dst)
            copied_sdfs.append((f"{d['design_id']}.sdf", d))
        smiles_rows.append({
            "rank": d.get("rank", ""),
            "design_id": d.get("design_id", ""),
            "smiles": d.get("smiles", ""),
            "combined_score": d.get("combined_score", ""),
            "qed": d.get("qed", ""),
            "sa_score": d.get("sa_score", ""),
            "vina_score": d.get("vina_score", ""),
        })

    csv_path = out_dir / "top_smiles.csv"
    if smiles_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=smiles_rows[0].keys())
            w.writeheader()
            w.writerows(smiles_rows)

    # Copy target and pocket PDBs for visualization
    target_name = None
    if target_pdb and Path(target_pdb).exists():
        target_name = f"target_{Path(target_pdb).stem}.pdb"
        shutil.copy2(target_pdb, out_dir / target_name)
    pocket_name = None
    if pocket_pdb and Path(pocket_pdb).exists():
        pocket_name = "pocket.pdb"
        shutil.copy2(pocket_pdb, out_dir / pocket_name)

    # Generate PyMOL visualization script
    if copied_sdfs:
        _write_molecule_pml(out_dir, copied_sdfs, target_name, pocket_name,
                            site_resnums)

    log(f"Top {len(top)} molecules saved to {out_dir}")


def _write_molecule_pml(out_dir, copied_sdfs, target_name, pocket_name,
                         site_resnums):
    """Write PyMOL script to visualize molecules in the binding pocket."""
    tool_colors = {
        "pocketflow":  "cyan",
        "molcraft":    "purple",
        "pocketxmol":  "lightblue",
        "library":     "orange",
    }

    # Build binding site selection
    site_sel = ""
    if site_resnums:
        resi_str = "+".join(str(r) for r in site_resnums)
        site_sel = f"chain A and resi {resi_str}"

    lines = [
        "# Auto-generated by generate_molecules.py",
        "# Visualize top molecules in the binding pocket",
        "bg_color white",
        "set ray_shadow, 0",
        "",
    ]

    # Load target protein
    if target_name:
        tobj = Path(target_name).stem
        lines.append(f"load {target_name}, {tobj}")
        lines.append(f"color grey70, {tobj}")
        lines.append(f"show cartoon, {tobj}")
        lines.append(f"hide lines, {tobj}")
        lines.append(f"set cartoon_transparency, 0.5, {tobj}")
        # Highlight binding site in red
        if site_sel:
            lines.append(f"color red, {tobj} and {site_sel}")
            lines.append(f"show sticks, {tobj} and {site_sel}")
            lines.append(f"set stick_transparency, 0.3, {tobj} and {site_sel}")
        lines.append("")

    # Load pocket surface (optional, for context)
    if pocket_name:
        lines.append(f"load {pocket_name}, pocket")
        lines.append(f"color grey90, pocket")
        lines.append(f"show surface, pocket")
        lines.append(f"set surface_color, grey90, pocket")
        lines.append(f"set transparency, 0.7, pocket")
        lines.append("")

    # Load molecules
    for i, (sdf_name, d) in enumerate(copied_sdfs):
        obj = Path(sdf_name).stem
        tool = d.get("tool", d.get("design_id", "").split("_")[0])
        color = tool_colors.get(tool, "green")
        score = d.get("combined_score", 0)
        qed = d.get("qed", 0)
        smiles = d.get("smiles", "")

        lines.append(f"# rank {d.get('rank', i+1)}  score={score:.3f}  QED={qed:.2f}  {smiles[:40]}")
        lines.append(f"load {sdf_name}, {obj}")
        lines.append(f"color {color}, {obj}")
        lines.append(f"show sticks, {obj}")
        lines.append(f"set stick_radius, 0.15, {obj}")
        # Show only top 5 by default
        if i >= 5:
            lines.append(f"disable {obj}")
        lines.append("")

    lines.extend([
        "# Zoom to full protein with molecules visible",
        f"zoom {Path(target_name).stem}" if target_name else "zoom",
        "set valence, 1",
        "",
        "# Color legend:",
        "#   red sticks  = binding site residues",
        "#   cyan        = PocketFlow",
        "#   purple      = MolCRAFT",
        "#   lightblue   = PocketXMol",
        "#   orange      = Library hits",
        "#   grey        = target protein",
    ])

    pml_path = out_dir / "view_molecules.pml"
    pml_path.write_text("\n".join(lines) + "\n")
    log(f"PyMOL script → {pml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Structure-based drug design (SBDD) pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tools:
  pocketflow    Autoregressive flow-based 3D molecule generation
  molcraft      Bayesian flow network (MolPilot, ICML 2024)
  pocketxmol    Foundation model (Cell 2026, SOTA)

Examples:
  python generate_molecules.py --target target.pdb --site "A:10-20" \\
      --tools pocketflow,molcraft,pocketxmol --mode test --out_dir ./molecules/

  # Virtual screening of a compound library
  python generate_molecules.py --target target.pdb --site "A:10-20" \\
      --library compounds.sdf --out_dir ./screening/

  # Combine de novo design + library screening
  python generate_molecules.py --target target.pdb --site "A:10-20" \\
      --tools pocketflow --library compounds.smi --out_dir ./combined/
""")
    parser.add_argument("--target", required=True, help="Target PDB file")
    parser.add_argument("--site", required=True,
                        help='Binding site (e.g. "A:325-330")')
    parser.add_argument("--tools", default="pocketflow",
                        help=f'Comma-separated tools ({",".join(ALL_TOOLS)})')
    parser.add_argument("--mode", default=DEFAULT_MODE,
                        choices=DESIGN_MODES.keys(),
                        help=f"Design mode (default: {DEFAULT_MODE})")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--max_atoms", type=int, default=35,
                        help="Max heavy atoms per molecule (default: 35)")
    parser.add_argument("--score_weights", default="0.40,0.35,0.15,0.10",
                        help='Vina,QED,SA,PocketFit weights (default: "0.40,0.35,0.15,0.10")')
    parser.add_argument("--no_vina", action="store_true",
                        help="Skip Vina re-docking (faster, QED+SA only)")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top molecules to copy (default: 50)")
    parser.add_argument("--library", default=None,
                        help="Compound library for virtual screening (.sdf, .smi, or .smiles)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--pocket_dist", type=float, default=10.0,
                        help="Pocket extraction distance in Å (default: 10.0)")
    parser.add_argument("--device", default="cuda:0",
                        help="CUDA device (default: cuda:0)")

    args = parser.parse_args()

    # ── Parse args ─────────────────────────────────────────────────────────
    target_path = Path(args.target).resolve()
    if not target_path.exists():
        sys.exit(f"ERROR: target not found: {target_path}")

    chain_id, site_resnums = parse_site(args.site)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Library screening mode (comma-separated for multiple libraries)
    library_paths = []
    if args.library:
        for lib_str in args.library.split(","):
            lib_str = lib_str.strip()
            if not lib_str:
                continue
            lp = Path(lib_str).resolve()
            if not lp.exists():
                sys.exit(f"ERROR: library not found: {lp}")
            if lp.suffix.lower() not in (".sdf", ".smi", ".smiles"):
                sys.exit(f"ERROR: --library must be .sdf, .smi, or .smiles "
                         f"(got {lp.suffix})")
            library_paths.append(lp)

    # If --library provided without explicit --tools, skip de novo
    tools_explicitly_set = any(a in sys.argv for a in ("--tools",))
    if library_paths and not tools_explicitly_set:
        tools = []
    else:
        tools = [t.strip().lower() for t in args.tools.split(",")]
        for t in tools:
            if t not in ALL_TOOLS:
                sys.exit(f"ERROR: unknown tool '{t}'. Available: {ALL_TOOLS}")

    mode = args.mode
    mode_presets = DESIGN_MODES[mode]

    # Parse score weights
    w_parts = args.score_weights.split(",")
    if len(w_parts) == 3:
        # Legacy 3-weight format (vina, qed, sa) — convert to 4-weight
        wv, wq, ws = (float(w) for w in w_parts)
        weights = (wv, wq, ws, 0.0)
    elif len(w_parts) == 4:
        weights = tuple(float(w) for w in w_parts)
    else:
        sys.exit("--score_weights must be 3 or 4 comma-separated floats")

    use_vina = not args.no_vina

    # ── Header ─────────────────────────────────────────────────────────────
    log("=" * 62)
    log("generate_molecules.py — Structure-Based Drug Design Pipeline")
    log("=" * 62)
    log(f"Target:           {target_path}")
    log(f"Binding site:     {chain_id}:{site_resnums[0]}-{site_resnums[-1]} ({len(site_resnums)} residues)")
    for lp in library_paths:
        log(f"Library:          {lp}")
    if tools:
        log(f"Tools:            {', '.join(tools)}")
    else:
        log(f"Tools:            (none — library screening only)")
    log(f"Mode:             {mode}")
    log(f"Max atoms:        {args.max_atoms}")
    log(f"Score weights:    LE={weights[0]}, QED={weights[1]}, SA={weights[2]}, PocketFit={weights[3]}")
    log(f"Vina re-docking:  {'YES' if use_vina else 'NO'}")
    log(f"Output:           {out_dir}")

    # ── Pocket extraction ──────────────────────────────────────────────────
    log(f"\n{'─' * 50}")
    log("STEP: Pocket extraction")
    log(f"{'─' * 50}")

    pocket_dir = out_dir / "pocket"
    pocket_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        pocket_pdb = extract_pocket_pdb(
            str(target_path), chain_id, site_resnums,
            distance_cutoff=args.pocket_dist,
            out_path=str(pocket_dir / "pocket.pdb"),
        )
        center = compute_pocket_center(str(target_path), chain_id, site_resnums)
        bbox_size = compute_bbox_size(str(target_path), chain_id, site_resnums)
        # Cap bbox at 20 Å — larger boxes spread de novo molecules too thin.
        if bbox_size > 20.0:
            log(f"  Note: bbox {bbox_size:.1f} Å capped to 20.0 Å")
            bbox_size = 20.0

        # Save pocket info for reference
        info = {"center": list(center), "bbox_size": bbox_size,
                "chain": chain_id, "residues": site_resnums}
        (pocket_dir / "pocket_info.json").write_text(json.dumps(info, indent=2))
        log(f"Pocket center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        log(f"Bounding box:  {bbox_size:.1f} Å")
    else:
        pocket_pdb = str(pocket_dir / "pocket.pdb")
        center = (0.0, 0.0, 0.0)
        bbox_size = 25.0

    # ── Run tools ──────────────────────────────────────────────────────────
    tool_sdf_files = {}
    tool_results = {}

    for tool in tools:
        log(f"\n{'─' * 50}")
        log(f"STEP: {tool.upper()}  (n={mode_presets[tool]})")
        log(f"{'─' * 50}")

        tool_dir = out_dir / tool
        n_mols = mode_presets[tool]

        try:
            if tool == "pocketflow":
                sdfs = run_pocketflow(
                    pocket_pdb, n_mols, tool_dir,
                    max_atoms=args.max_atoms, device=args.device,
                    dry_run=args.dry_run)
            elif tool == "molcraft":
                sdfs = run_molcraft(
                    str(target_path), pocket_pdb, center, n_mols, tool_dir,
                    device=args.device, dry_run=args.dry_run)
            elif tool == "pocketxmol":
                sdfs = run_pocketxmol(
                    str(target_path), center, bbox_size, n_mols, tool_dir,
                    device=args.device, dry_run=args.dry_run)
            else:
                raise ValueError(f"Unknown tool: {tool}")

            tool_sdf_files[tool] = sdfs
            tool_results[tool] = f"OK ({len(sdfs)} SDF files)"
        except Exception as e:
            log(f"{tool} FAILED: {e}")
            tool_results[tool] = f"FAILED: {e}"
            tool_sdf_files[tool] = []

    # ── Library screening (supports multiple libraries) ─────────────────
    for lib_path in library_paths:
        # Derive tool name from filename: library_drugbank, library_ppi_500, etc.
        lib_name = f"library_{lib_path.stem}"
        log(f"\n{'─' * 50}")
        log(f"STEP: LIBRARY SCREENING ({lib_path.name})")
        log(f"{'─' * 50}")

        lib_dir = out_dir / lib_name
        try:
            sdfs = load_library(str(lib_path), lib_dir,
                                dry_run=args.dry_run)
            tool_sdf_files[lib_name] = sdfs
            tool_results[lib_name] = f"OK ({len(sdfs)} files)"
        except Exception as e:
            log(f"Library {lib_path.name} FAILED: {e}")
            tool_results[lib_name] = f"FAILED: {e}"
            tool_sdf_files[lib_name] = []

    # ── Scoring & ranking ──────────────────────────────────────────────────
    log(f"\n{'─' * 50}")
    log("STEP: Scoring & Ranking")
    log(f"{'─' * 50}")

    all_molecules = []
    for source in tool_sdf_files:
        sdfs = tool_sdf_files[source]
        if not sdfs or args.dry_run:
            continue

        log(f"Scoring {source} molecules...")
        try:
            mols = collect_and_score(
                source, sdfs, str(target_path), center, bbox_size,
                use_vina=use_vina)
            all_molecules.extend(mols)
            log(f"  {source}: {len(mols)} valid unique molecules scored")
        except Exception as e:
            log(f"  {source} scoring failed: {e}")

    if all_molecules:
        log(f"\nTotal: {len(all_molecules)} molecules to rank")
        ranked = rank_molecules(all_molecules, weights=weights,
                                pocket_pdb=str(pocket_dir / "pocket.pdb"))

        # Compute per-tool diversity
        from rdkit import Chem
        for tool in tool_sdf_files:
            tool_mols_rdkit = []
            for d in ranked:
                if d.get("tool") == tool:
                    m = Chem.MolFromSmiles(d.get("smiles", ""))
                    if m:
                        tool_mols_rdkit.append(m)
            if tool_mols_rdkit:
                div = compute_diversity(tool_mols_rdkit[:200])  # cap for speed
                log(f"  {tool} diversity (Tanimoto): {div:.3f}")

        # Write outputs
        scoring_dir = out_dir / "scoring"
        scoring_dir.mkdir(parents=True, exist_ok=True)

        write_rankings_csv(ranked, out_dir / "rankings.csv")
        copy_top_molecules(ranked, out_dir / "top_molecules", n=args.top_n,
                           target_pdb=str(target_path),
                           pocket_pdb=str(pocket_dir / "pocket.pdb"),
                           site_resnums=site_resnums)

        plot_mol_dashboard(
            ranked, out_dir / "dashboard.png",
            title=f"SBDD: {target_path.stem}  site={args.site}")
    elif not args.dry_run:
        log("No molecules scored. Skipping ranking.")
        ranked = []
    else:
        ranked = []

    # ── Summary ────────────────────────────────────────────────────────────
    log(f"\n{'=' * 62}")
    log("DONE")
    log(f"{'=' * 62}")
    for tool, status in tool_results.items():
        log(f"  {tool:16s}: {status}")

    if ranked:
        log(f"\nTop 5 molecules:")
        for d in ranked[:5]:
            vina = d.get("vina_score", float("nan"))
            vina_str = f"{vina:.1f}" if vina == vina else "N/A"
            log(f"  rank {d['rank']:3d}  {d['design_id']:<22s}  "
                f"score={d['combined_score']:.3f}  "
                f"QED={d.get('qed', 0):.2f}  "
                f"SA={d.get('sa_score', 0):.1f}  "
                f"Vina={vina_str}  "
                f"MW={d.get('mw', 0):.0f}")

    log(f"\nOutput directory: {out_dir}")
    if ranked:
        log(f"  rankings.csv    — {len(ranked)} total molecules")
        log(f"  top_molecules/  — top {min(args.top_n, len(ranked))} SDFs + SMILES")
        log(f"  dashboard.png   — summary plot")
    log(f"  pocket/         — extracted pocket PDB + center info")


if __name__ == "__main__":
    main()
