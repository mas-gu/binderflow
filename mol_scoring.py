#!/usr/bin/env python3
"""
mol_scoring.py — Molecule validation, scoring, and ranking for SBDD pipeline.

Provides QED, synthetic accessibility, Vina re-docking, Lipinski properties,
diversity calculation, and combined ranking. Used by generate_molecules.py.

Dependencies (boltz env): rdkit, numpy, meeko, vina (optional).
"""

import csv
import logging
import os
import tempfile
import warnings
from pathlib import Path

_log = logging.getLogger(__name__)

import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.QED import qed as _rdkit_qed

# Suppress RDKit warnings for invalid molecules
RDLogger.logger().setLevel(RDLogger.ERROR)


# ── SA Score (Ertl & Schuffenhauer, J. Cheminform. 2009) ─────────────────────

_sa_model = None

def _load_sa_model():
    """Load the SA score model (bundled with RDKit contrib)."""
    global _sa_model
    if _sa_model is not None:
        return _sa_model
    try:
        from rdkit.Chem import RDConfig
        import sys
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
        _sa_model = sascorer
        return _sa_model
    except ImportError:
        warnings.warn("SA score module not available — returning default 5.0")
        return None


# ── Core scoring functions ────────────────────────────────────────────────────

def compute_qed(mol):
    """Compute QED (quantitative estimate of drug-likeness), 0–1."""
    if mol is None:
        return float("nan")
    try:
        return _rdkit_qed(mol)
    except Exception:
        return float("nan")


def compute_sa_score(mol):
    """
    Compute synthetic accessibility score, 1 (easy) – 10 (hard).
    Lower is better.
    """
    if mol is None:
        return float("nan")
    sa = _load_sa_model()
    if sa is None:
        return 5.0
    try:
        return sa.calculateScore(mol)
    except Exception:
        return float("nan")


def compute_lipinski(mol):
    """
    Compute physicochemical and drug-likeness properties.

    Returns dict with Lipinski props, extended descriptors (formula, heavy atoms,
    rotatable bonds, molar refractivity, Fsp3), and Lipinski/Veber pass/fail.
    """
    if mol is None:
        return {"mw": float("nan"), "logp": float("nan"),
                "hbd": 0, "hba": 0, "tpsa": float("nan"),
                "lipinski_violations": 4,
                "formula": "", "num_heavy_atoms": 0,
                "num_rotatable_bonds": 0, "molar_refractivity": float("nan"),
                "fsp3": float("nan"), "lipinski_pass": False, "veber_pass": False}
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)

    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    # Extended descriptors
    formula = rdMolDescriptors.CalcMolFormula(mol)
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    molar_refractivity = Descriptors.MolMR(mol)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    # Rule pass/fail
    lipinski_pass = violations == 0
    veber_pass = num_rotatable_bonds <= 10 and tpsa <= 140

    return {
        "mw": mw, "logp": logp,
        "hbd": hbd, "hba": hba,
        "tpsa": tpsa,
        "lipinski_violations": violations,
        "formula": formula,
        "num_heavy_atoms": num_heavy_atoms,
        "num_rotatable_bonds": num_rotatable_bonds,
        "molar_refractivity": molar_refractivity,
        "fsp3": fsp3,
        "lipinski_pass": lipinski_pass,
        "veber_pass": veber_pass,
    }


def compute_vina_score(protein_pdb, sdf_path, center, box_size=25.0,
                       docked_sdf_path=None):
    """
    Re-dock a molecule into the protein pocket using AutoDock Vina.

    Parameters
    ----------
    protein_pdb : str
        Path to protein PDB file.
    sdf_path : str
        Path to ligand SDF file (single molecule).
    center : tuple[float, float, float]
        Pocket center (x, y, z).
    box_size : float
        Docking box side length in Angstroms.

    Returns
    -------
    float
        Best Vina score (kcal/mol), or nan on failure.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from vina import Vina
    except ImportError:
        warnings.warn("meeko/vina not installed — skipping Vina scoring")
        return float("nan")

    try:
        # Prepare receptor: convert PDB to PDBQT using meeko/openbabel
        receptor_pdbqt = protein_pdb
        if protein_pdb.endswith(".pdb"):
            # Use prepare_receptor from meeko or fall back to simple conversion
            receptor_pdbqt = protein_pdb.replace(".pdb", "_receptor.pdbqt")
            if not os.path.exists(receptor_pdbqt):
                try:
                    import subprocess as _sp
                    # Try obabel (Open Babel) for PDB→PDBQT conversion
                    _sp.run(["obabel", protein_pdb, "-O", receptor_pdbqt,
                             "-xr", "-xc", "-xn"],
                            capture_output=True, timeout=60, check=True)
                except (FileNotFoundError, _sp.CalledProcessError):
                    # Fall back to meeko's PDBQTWriterLegacy for receptor
                    from meeko import PDBQTMolecule, RDKitMolCreate
                    try:
                        pdbqt_mol = PDBQTMolecule.from_file(protein_pdb,
                                                             is_receptor=True)
                        with open(receptor_pdbqt, "w") as f:
                            f.write(pdbqt_mol.write_pdbqt_string())
                    except Exception as e:
                        _log.debug("Meeko PDBQT conversion failed: %s", e)
                        # Last resort: Vina can sometimes read PDB directly
                        receptor_pdbqt = protein_pdb

        v = Vina(sf_name="vina")
        v.set_receptor(str(receptor_pdbqt))
        v.compute_vina_maps(center=list(center),
                            box_size=[box_size, box_size, box_size])

        # Prepare ligand: ensure explicit Hs with 3D coordinates
        mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
        if mol is None:
            return float("nan")

        # Always add explicit Hs (meeko requires them)
        mol = Chem.AddHs(mol, addCoords=True)
        # Ensure 3D coordinates exist (generate if missing)
        if mol.GetNumConformers() == 0:
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        # Write PDBQT to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w",
                                          delete=False) as f:
            pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(
                mol_setups[0])
            f.write(pdbqt_str if is_ok else "")
            pdbqt_path = f.name

        if not is_ok:
            try:
                os.unlink(pdbqt_path)
            except OSError:
                pass
            return float("nan")

        v.set_ligand_from_file(pdbqt_path)
        v.dock(exhaustiveness=8, n_poses=1)
        energies = v.energies()

        # Save docked pose as SDF if requested
        if docked_sdf_path and len(energies) > 0:
            try:
                # Write docked PDBQT, convert back to SDF via obabel
                docked_pdbqt = pdbqt_path.replace(".pdbqt", "_docked.pdbqt")
                v.write_poses(docked_pdbqt, n_poses=1, overwrite=True)
                import subprocess as _sp
                # Convert PDBQT→SDF preserving docked coordinates (no --gen3d!)
                _sp.run(["obabel", docked_pdbqt, "-O", docked_sdf_path],
                        capture_output=True, timeout=30)
                try:
                    os.unlink(docked_pdbqt)
                except OSError:
                    pass
            except Exception:
                pass

        try:
            os.unlink(pdbqt_path)
        except OSError:
            pass

        return float(energies[0][0]) if len(energies) > 0 else float("nan")

    except Exception as e:
        _log.debug("Vina docking failed: %s", e)
        try:
            os.unlink(pdbqt_path)  # type: ignore[possibly-undefined]
        except (OSError, NameError):
            pass
        return float("nan")


def compute_diversity(mols):
    """
    Compute average pairwise Tanimoto distance (1 - similarity) using
    Morgan fingerprints (radius=2, 2048 bits).

    Returns float in [0, 1]. Higher = more diverse.
    """
    valid = [m for m in mols if m is not None]
    if len(valid) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
           for m in valid]
    n = len(fps)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_dist += 1.0 - sim
            count += 1

    return total_dist / count if count > 0 else 0.0


# ── Batch scoring ─────────────────────────────────────────────────────────────

def parse_sdf(sdf_path):
    """
    Parse an SDF file and return list of (mol, smiles) tuples.
    Skips invalid molecules. Returns empty list for missing/empty files.
    """
    sdf_path = str(sdf_path)
    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
        return []
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    except OSError:
        return []
    results = []
    for mol in suppl:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            results.append((mol, smi))
        except Exception:
            continue
    return results


def score_molecules(sdf_path, protein_pdb=None, center=None, box_size=25.0,
                    use_vina=True):
    """
    Score all molecules in an SDF file.

    Returns list of dicts with: smiles, qed, sa_score, lipinski props,
    and optionally vina_score.
    """
    mols_smiles = parse_sdf(sdf_path)
    if not mols_smiles:
        return []

    results = []
    seen_smiles = set()

    for mol, smi in mols_smiles:
        # Dedup by canonical SMILES
        if smi in seen_smiles:
            continue
        seen_smiles.add(smi)

        entry = {"smiles": smi}
        entry["qed"] = compute_qed(mol)
        entry["sa_score"] = compute_sa_score(mol)
        entry.update(compute_lipinski(mol))

        # Vina scoring (expensive — write single-mol SDF to temp)
        if use_vina and protein_pdb and center:
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
                w = Chem.SDWriter(f.name)
                w.write(mol)
                w.close()
                entry["vina_score"] = compute_vina_score(
                    protein_pdb, f.name, center, box_size)
                os.unlink(f.name)
        else:
            entry["vina_score"] = float("nan")

        results.append(entry)

    return results


def compute_pocket_fit(mol_sdf_path, pocket_pdb_path, cutoff=4.0):
    """
    Compute fraction of molecule heavy atoms within cutoff of any pocket atom.

    Returns float in [0, 1] — 1.0 means all atoms sit inside the pocket.
    Returns NaN if files can't be read.
    """
    try:
        from rdkit import Chem
        mol = Chem.SDMolSupplier(str(mol_sdf_path), removeHs=True, sanitize=False)[0]
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass  # keep unsanitized mol for coordinate extraction
        if mol is None or mol.GetNumConformers() == 0:
            return float("nan")
        mol_pos = mol.GetConformer().GetPositions()

        # Parse pocket PDB CA/heavy atoms
        pocket_pos = []
        with open(pocket_pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    pocket_pos.append((x, y, z))
        if not pocket_pos:
            return float("nan")

        pocket_pos = np.array(pocket_pos)
        n_in_pocket = 0
        cutoff_sq = cutoff * cutoff
        for pos in mol_pos:
            dists_sq = np.sum((pocket_pos - pos) ** 2, axis=1)
            if np.min(dists_sq) <= cutoff_sq:
                n_in_pocket += 1

        return n_in_pocket / len(mol_pos) if len(mol_pos) > 0 else 0.0
    except Exception:
        return float("nan")


def rank_molecules(all_mols, weights=(0.40, 0.35, 0.15, 0.10),
                   pocket_pdb=None):
    """
    Rank molecules by combined score.

    4-weight form (default):
        combined_score = w_vina * vina_norm + w_qed * qed
                       + w_sa * sa_norm + w_fit * fit_norm

    5-weight form (enables Boltz-2 affinity term):
        combined_score = w_vina * vina_norm + w_qed * qed
                       + w_sa * sa_norm + w_fit * fit_norm
                       + w_aff * prob_binder

    Where:
        vina_norm    = clip(-vina / 12, 0, 1)          — raw Vina, no size bias
        qed          = drug-likeness (0-1)
        sa_norm      = 1 - (sa_score - 1) / 9          — synthetic accessibility
        fit_norm     = clip((pocket_fit - 0.5) / 0.5, 0, 1)
        prob_binder  = boltz_affinity_prob_binder (0-1); NaN -> 0 contribution

    Hard pre-filters:
        Vina outside [-12, -1] → NaN (artifact)

    Parameters
    ----------
    all_mols : list[dict]
        Each dict must have at least 'qed', 'sa_score'. 'vina_score',
        'pocket_fit', and 'boltz_affinity_prob_binder' are optional.
    weights : tuple of 4 or 5 floats
        (vina, qed, sa, fit) or (vina, qed, sa, fit, affinity).
    pocket_pdb : str or None
        Path to pocket PDB for pocket_fit computation.

    Returns
    -------
    list[dict]
        Input dicts with added 'combined_score', 'ligand_efficiency',
        'pocket_fit', and 'rank' fields, sorted by rank.
    """
    if len(weights) == 4:
        w_vina, w_qed, w_sa, w_fit = weights
        w_aff = 0.0
    elif len(weights) == 5:
        w_vina, w_qed, w_sa, w_fit, w_aff = weights
    else:
        raise ValueError(
            f"rank_molecules weights must have 4 or 5 elements, got {len(weights)}")

    for entry in all_mols:
        vina = entry.get("vina_score", float("nan"))
        qed_val = entry.get("qed", 0.0)
        sa = entry.get("sa_score", 5.0)
        n_heavy = entry.get("num_heavy_atoms", 0)

        # Vina artifact detection
        # Realistic range for drug-like molecules: -1 to -12 kcal/mol.
        # Outside this range = Uni-Dock artifact.
        if vina != vina:  # NaN
            vina_norm = 0.0
        elif vina > -1 or vina < -12:
            vina_norm = 0.0  # artifact — treat as undocked
            entry["vina_score"] = float("nan")
        else:
            vina_norm = max(0.0, min(1.0, -vina / 12.0))

        # Ligand efficiency (informational, not used in scoring)
        if vina == vina and vina <= 0 and n_heavy and n_heavy > 0:
            vina_for_le = max(-12.0, vina)
            entry["ligand_efficiency"] = round(-vina_for_le / n_heavy, 3)
        else:
            entry["ligand_efficiency"] = float("nan")

        # SA normalization
        sa_norm = max(0.0, min(1.0, 1.0 - (sa - 1.0) / 9.0))
        qed_val = qed_val if qed_val == qed_val else 0.0

        # Pocket fit — compute if pocket PDB available and docked SDF exists
        fit = entry.get("pocket_fit", float("nan"))
        if (fit != fit) and pocket_pdb:
            sdf_path = entry.get("sdf_path", "")
            if sdf_path and os.path.exists(sdf_path):
                fit = compute_pocket_fit(sdf_path, pocket_pdb)
                entry["pocket_fit"] = round(fit, 3) if fit == fit else float("nan")

        # Rescale pocket_fit: 0.5-1.0 → 0-1 (makes it discriminating)
        if isinstance(fit, (int, float)) and fit == fit:
            fit_norm = max(0.0, min(1.0, (fit - 0.5) / 0.5))
        else:
            fit_norm = 0.0

        # Boltz-2 affinity probability (0-1). NaN -> 0 contribution
        # (same pass-through convention as vina/fit).
        prob = entry.get("boltz_affinity_prob_binder", float("nan"))
        if isinstance(prob, (int, float)) and prob == prob:
            prob_norm = max(0.0, min(1.0, float(prob)))
        else:
            prob_norm = 0.0

        entry["combined_score"] = (
            w_vina * vina_norm +
            w_qed * qed_val +
            w_sa * sa_norm +
            w_fit * fit_norm +
            w_aff * prob_norm
        )

    # Sort by combined score descending
    all_mols.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    for i, entry in enumerate(all_mols):
        entry["rank"] = i + 1

    return all_mols


def write_rankings_csv(molecules, out_path):
    """Write ranked molecules to CSV."""
    if not molecules:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank", "design_id", "tool", "smiles", "sdf_path",
        "qed", "sa_score", "vina_score",
        "boltz_affinity_log_ic50", "boltz_affinity_prob_binder",
        "mw", "logp", "hbd", "hba", "tpsa", "lipinski_violations",
        "combined_score",
    ]
    # Add any extra keys
    all_keys = set()
    for m in molecules:
        all_keys.update(m.keys())
    for k in sorted(all_keys - set(fieldnames)):
        fieldnames.append(k)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in molecules:
            writer.writerow(m)
