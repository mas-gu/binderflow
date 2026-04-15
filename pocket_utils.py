#!/usr/bin/env python3
"""
pocket_utils.py — Shared pocket extraction utilities for SBDD tools.

Used by generate_molecules.py to prepare inputs for PocketFlow, Pocket2Mol,
and DecompDiff. Reuses BioPython patterns from pepflow_wrapper.py.
"""

import numpy as np
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch


def _get_structure(pdb_path):
    """Parse a PDB file and return the first model."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    return structure, structure[0]


def extract_pocket_pdb(protein_pdb, chain_id, site_resnums, distance_cutoff=10.0,
                       out_path=None):
    """
    Extract pocket residues within distance_cutoff of specified binding site.

    Parameters
    ----------
    protein_pdb : str or Path
        Path to the full protein PDB.
    chain_id : str
        Chain containing the binding site.
    site_resnums : list[int]
        Residue numbers defining the binding site.
    distance_cutoff : float
        Angstrom cutoff for pocket residues (default 10.0).
    out_path : str or Path, optional
        Where to save the pocket PDB. If None, saves next to input as
        {stem}_pocket.pdb.

    Returns
    -------
    str
        Path to the saved pocket PDB file.
    """
    protein_pdb = Path(protein_pdb)
    structure, model = _get_structure(protein_pdb)

    if chain_id not in [c.id for c in model]:
        raise ValueError(f"Chain {chain_id} not found in {protein_pdb}")

    chain = model[chain_id]

    # Collect atoms from binding site residues
    site_atoms = []
    for res in chain:
        if res.id[1] in site_resnums:
            site_atoms.extend(res.get_atoms())

    if not site_atoms:
        raise ValueError(
            f"No atoms found for residues {site_resnums} in chain {chain_id}")

    # Find all protein atoms (all chains — pocket may span interfaces)
    all_atoms = list(model.get_atoms())
    search = NeighborSearch(all_atoms)

    pocket_res_ids = set()
    for atom in site_atoms:
        hits = search.search(atom.get_coord(), distance_cutoff, level="R")
        for r in hits:
            # Only include standard residues (skip HETATM water, etc.)
            if r.id[0] == " ":
                pocket_res_ids.add(r.get_full_id())

    class PocketSelect(Select):
        def accept_residue(self, residue):
            return residue.get_full_id() in pocket_res_ids

    if out_path is None:
        out_path = protein_pdb.parent / f"{protein_pdb.stem}_pocket.pdb"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), PocketSelect())

    return str(out_path)


def compute_pocket_center(protein_pdb, chain_id, site_resnums):
    """
    Compute the geometric center of binding site CA atoms.

    Returns
    -------
    tuple[float, float, float]
        (x, y, z) center coordinates.
    """
    _, model = _get_structure(protein_pdb)
    chain = model[chain_id]

    ca_coords = []
    for res in chain:
        if res.id[1] in site_resnums and "CA" in res:
            ca_coords.append(res["CA"].get_vector().get_array())

    if not ca_coords:
        raise ValueError(
            f"No CA atoms found for residues {site_resnums} in chain {chain_id}")

    center = np.mean(ca_coords, axis=0)
    return float(center[0]), float(center[1]), float(center[2])


def compute_bbox_size(protein_pdb, chain_id, site_resnums, padding=5.0):
    """
    Compute bounding box size for the binding pocket.

    Takes the max extent across x/y/z of pocket CA atoms + padding on each side.

    Parameters
    ----------
    padding : float
        Angstroms to add on each side (default 5.0).

    Returns
    -------
    float
        Side length of the cubic bounding box.
    """
    _, model = _get_structure(protein_pdb)
    chain = model[chain_id]

    ca_coords = []
    for res in chain:
        if res.id[1] in site_resnums and "CA" in res:
            ca_coords.append(res["CA"].get_vector().get_array())

    if not ca_coords:
        raise ValueError(
            f"No CA atoms found for residues {site_resnums} in chain {chain_id}")

    coords = np.array(ca_coords)
    extent = coords.max(axis=0) - coords.min(axis=0)
    bbox_size = float(extent.max()) + 2 * padding

    return bbox_size
