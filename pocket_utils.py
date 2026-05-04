#!/usr/bin/env python3
"""
pocket_utils.py — Shared pocket extraction utilities for SBDD tools.

Used by generate_molecules.py and rerank_molecules.py to build pocket inputs
for PocketFlow / MolCRAFT / PocketXMol and to define the Vina docking box.

The molecule pipeline accepts multi-chain binding sites (e.g. an interface
pocket between chains A and B). This module's three geometry helpers all
operate on a flat list of ``(chain_id, resnum)`` pairs returned by
``parse_site_multichain``. Single-chain inputs remain a trivial subset.
"""

import re
import numpy as np
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch


# ── Site parser ───────────────────────────────────────────────────────────────

def parse_site_multichain(site_str):
    """Parse a (possibly multi-chain) binding-site spec.

    Accepted forms (whitespace tolerated around chunks)::

        "A:8-30,120-123"               single chain (current binder syntax)
        "A:8,15,22"                    single chain, individual residues
        "A:8"                          single residue
        "A:11-17,B:119-124"            multi-chain
        "A:11-17,B:119-124,B:140"      multi-chain with mixed ranges
        "A:11-17, B:119-124"           with whitespace around chunks

    Each comma-separated chunk may carry an optional ``<chain>:`` prefix.
    The prefix sets the active chain for that chunk and onward until the
    next prefix. The first chunk **must** include a chain prefix.

    Returns
    -------
    list[tuple[str, int]]
        Sorted, deduplicated list of (chain, resnum) pairs.

    Raises
    ------
    ValueError
        On empty input, missing first-chunk prefix, or unparseable spec.
    """
    site_str = (site_str or "").strip()
    if not site_str:
        raise ValueError("Empty site specification.")

    pairs = []
    active_chain = None

    for chunk in re.split(r",", site_str):
        chunk = chunk.strip()
        if not chunk:
            continue

        m_pref = re.match(r"^([A-Za-z]):(.+)$", chunk)
        if m_pref:
            active_chain = m_pref.group(1).upper()
            spec = m_pref.group(2).strip()
        else:
            if active_chain is None:
                raise ValueError(
                    f"First chunk '{chunk}' must include a chain prefix "
                    f"like 'A:11-17'.")
            spec = chunk

        m_range = re.match(r"^(\d+)-(\d+)$", spec)
        m_single = re.match(r"^(\d+)$", spec)
        if m_range:
            lo, hi = int(m_range.group(1)), int(m_range.group(2))
            if hi < lo:
                raise ValueError(
                    f"Range '{spec}' in '{chunk}' has end < start.")
            for r in range(lo, hi + 1):
                pairs.append((active_chain, r))
        elif m_single:
            pairs.append((active_chain, int(m_single.group(1))))
        else:
            raise ValueError(
                f"Cannot parse residue spec '{spec}' in '{chunk}'. "
                f"Expected N or N-M (e.g. '8-30' or '120').")

    if not pairs:
        raise ValueError("No residues parsed from site specification.")
    return sorted(set(pairs))


def site_pairs_chains(site_pairs):
    """Return the unique chain IDs in a site_pairs list, in stable order."""
    seen = []
    for c, _ in site_pairs:
        if c not in seen:
            seen.append(c)
    return seen


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_structure(pdb_path):
    """Parse a PDB file and return the first model."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    return structure, structure[0]


def _site_atoms(model, site_pairs):
    """Collect all atoms for the residues named in site_pairs.

    Raises ValueError if any chain is missing or if no atoms were collected.
    """
    by_chain = {}
    for c, r in site_pairs:
        by_chain.setdefault(c, set()).add(r)

    available = {c.id for c in model}
    missing = [c for c in by_chain if c not in available]
    if missing:
        raise ValueError(
            f"Chain(s) {missing} not found in target "
            f"(available: {sorted(available)})")

    atoms = []
    for chain_id, resnums in by_chain.items():
        chain = model[chain_id]
        for res in chain:
            if res.id[1] in resnums:
                atoms.extend(res.get_atoms())
    if not atoms:
        raise ValueError(
            f"No atoms found for site residues {site_pairs}")
    return atoms


def _site_ca_coords(model, site_pairs):
    """Collect Cα coordinates for the named residues.

    Falls back to any atom in the residue if no Cα is present (handles
    HETATM-only or partial residues without crashing).
    """
    by_chain = {}
    for c, r in site_pairs:
        by_chain.setdefault(c, set()).add(r)

    coords = []
    for chain_id, resnums in by_chain.items():
        if chain_id not in [c.id for c in model]:
            raise ValueError(f"Chain {chain_id} not found")
        chain = model[chain_id]
        for res in chain:
            if res.id[1] not in resnums:
                continue
            if "CA" in res:
                coords.append(res["CA"].get_vector().get_array())
            else:
                # Fallback: residue centroid from whatever atoms exist
                atoms = list(res.get_atoms())
                if atoms:
                    coords.append(np.mean(
                        [a.get_vector().get_array() for a in atoms], axis=0))
    if not coords:
        raise ValueError(
            f"No Cα atoms found for site residues {site_pairs}")
    return np.array(coords)


# ── Public geometry API ───────────────────────────────────────────────────────

def extract_pocket_pdb(protein_pdb, site_pairs, distance_cutoff=10.0,
                       out_path=None):
    """Extract residues within ``distance_cutoff`` Å of the named site.

    Parameters
    ----------
    protein_pdb : str or Path
        Path to the full protein PDB.
    site_pairs : list[tuple[str, int]]
        Anchor residues as (chain_id, resnum) pairs. May span multiple chains.
    distance_cutoff : float
        Å radius of the pocket sphere around each anchor atom.
    out_path : str or Path, optional
        Destination for the pocket PDB. Defaults to ``{stem}_pocket.pdb``
        next to the input.

    Returns
    -------
    str
        Absolute path to the saved pocket PDB.

    Notes
    -----
    The neighbour search runs across **all** chains of the model, so a
    site anchored on chain A will pull in any residues from chain B (or
    further chains) that fall within range — exactly what's needed for
    interface pockets.
    """
    protein_pdb = Path(protein_pdb)
    structure, model = _get_structure(protein_pdb)

    site_atoms = _site_atoms(model, site_pairs)
    all_atoms = list(model.get_atoms())
    search = NeighborSearch(all_atoms)

    pocket_res_ids = set()
    for atom in site_atoms:
        hits = search.search(atom.get_coord(), distance_cutoff, level="R")
        for r in hits:
            # Standard residues only — skip waters / HETATM
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


def compute_pocket_center(protein_pdb, site_pairs):
    """Geometric centroid of the site Cα atoms across all listed chains.

    Returns
    -------
    tuple[float, float, float]
        (x, y, z) Å coordinates.
    """
    _, model = _get_structure(protein_pdb)
    coords = _site_ca_coords(model, site_pairs)
    center = coords.mean(axis=0)
    return float(center[0]), float(center[1]), float(center[2])


def compute_bbox_size(protein_pdb, site_pairs, padding=5.0):
    """Cubic bounding-box side length covering all site Cα atoms + padding.

    Returns the maximum axis extent of the union of Cα atoms across all
    site chains, plus ``padding`` Å on each side. The molecule pipeline
    applies its own cap on top (20 Å single-chain, 25 Å multi-chain).
    """
    _, model = _get_structure(protein_pdb)
    coords = _site_ca_coords(model, site_pairs)
    extent = coords.max(axis=0) - coords.min(axis=0)
    return float(extent.max()) + 2 * padding
