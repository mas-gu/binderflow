#!/usr/bin/env python3
"""
decompdiff_wrapper.py — Standalone wrapper for DecompDiff molecule generation.

Converts a target PDB + binding site specification into DecompDiff's expected
input format, runs sampling, and collects SDF outputs.

Usage (run inside decompdiff conda env):
    python decompdiff_wrapper.py \
        --target target.pdb \
        --chain A \
        --binding_residues 325,326,327,328,329,330 \
        --num_samples 100 \
        --checkpoint /path/to/uni_o2_bond.pt \
        --output_dir ./decompdiff_out/
"""

import argparse
import os
import sys
import pickle
import warnings

import numpy as np
import torch
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch

# Add DecompDiff to path
DECOMPDIFF_DIR = os.environ.get(
    "DECOMPDIFF_DIR",
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, DECOMPDIFF_DIR)


def extract_pocket_pdb(target_pdb, chain_id, binding_residues, contact_dist=10.0,
                       out_path=None):
    """Extract pocket residues within contact_dist of binding site."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", target_pdb)
    model = structure[0]

    chain = model[chain_id]
    site_atoms = []
    for res in chain:
        if res.id[1] in binding_residues:
            site_atoms.extend(res.get_atoms())

    if not site_atoms:
        raise ValueError(
            f"No atoms for residues {binding_residues} in chain {chain_id}")

    all_atoms = list(chain.get_atoms())
    search = NeighborSearch(all_atoms)
    pocket_res_ids = set()
    for atom in site_atoms:
        hits = search.search(atom.get_coord(), contact_dist, level="R")
        for r in hits:
            if r.id[0] == " ":
                pocket_res_ids.add(r.get_full_id())

    class PocketSelect(Select):
        def accept_chain(self, c):
            return c.id == chain_id
        def accept_residue(self, residue):
            return residue.get_full_id() in pocket_res_ids

    if out_path is None:
        out_path = target_pdb.replace(".pdb", "_pocket.pdb")

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), PocketSelect())

    return str(out_path)


def cluster_pocket_atoms(protein_pos, num_arms=3):
    """Cluster pocket atoms into num_arms groups using k-means for subpocket mode."""
    from scipy.cluster.vq import kmeans2
    coords = protein_pos.cpu().numpy()
    # Use k-means to partition pocket into arm regions
    n_atoms = len(coords)
    if n_atoms < num_arms:
        num_arms = max(1, n_atoms)
    centroids, labels = kmeans2(coords, num_arms, minit='points', iter=20)
    # Build boolean masks: one per arm, shape [num_arms, n_atoms]
    masks = []
    for arm_id in range(num_arms):
        mask = torch.tensor(labels == arm_id, dtype=torch.bool)
        masks.append(mask)
    return masks, num_arms


def run_decompdiff(pocket_pdb, num_samples, checkpoint, output_dir,
                   device="cuda:0", num_steps=1000, batch_size=16,
                   num_arms=3):
    """
    Run DecompDiff sampling on a pocket PDB.

    Uses subpocket mode: clusters pocket atoms into arm regions, then
    samples scaffold+arms conditioned on the pocket structure.
    """
    os.makedirs(output_dir, exist_ok=True)

    from rdkit import Chem, RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    from torch_geometric.transforms import Compose
    from torch_scatter import scatter_sum

    import utils.transforms as trans
    import utils.reconstruct as recon
    from datasets.pl_data import FOLLOW_BATCH, torchify_dict
    from models.decompdiff import DecompScorePosNet3D
    from utils.data import ProteinLigandData, PDBProtein

    # Load checkpoint config
    print(f"Loading model from {checkpoint}...")
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt['config']
    cfg_transform = cfg.data.transform

    # Build transforms — must match training config exactly
    ligand_atom_mode = cfg_transform.ligand_atom_mode
    ligand_bond_mode = cfg_transform.ligand_bond_mode
    max_num_arms = cfg_transform.max_num_arms
    add_ord_feat = getattr(cfg_transform, 'add_ord_feat', False)
    prior_types = cfg.model.get('prior_types', False)

    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(
        mode=ligand_atom_mode, prior_types=prior_types)
    decomp_indicator = trans.AddDecompIndicator(
        max_num_arms=max_num_arms,
        global_prior_index=ligand_featurizer.ligand_feature_dim,
        add_ord_feat=add_ord_feat,
    )

    protein_feat_dim = protein_featurizer.protein_feature_dim + decomp_indicator.protein_feature_dim
    ligand_feat_dim = ligand_featurizer.ligand_feature_dim + decomp_indicator.ligand_feature_dim
    num_classes = ligand_featurizer.ligand_feature_dim

    print(f"  protein_feat_dim={protein_feat_dim}, ligand_feat_dim={ligand_feat_dim}, "
          f"num_classes={num_classes}")

    # Build model
    model = DecompScorePosNet3D(
        cfg.model,
        protein_atom_feature_dim=protein_feat_dim,
        ligand_atom_feature_dim=ligand_feat_dim,
        num_classes=num_classes,
        prior_atom_types=ligand_featurizer.atom_types_prob,
        prior_bond_types=ligand_featurizer.bond_types_prob,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    print("  Model loaded successfully")

    # Load pocket data
    print(f"Loading pocket from {pocket_pdb}...")
    protein = PDBProtein(pocket_pdb)
    protein_dict = torchify_dict(protein.to_dict_atom())
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=protein_dict,
        ligand_dict={
            'element': torch.empty([0], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, num_classes], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0], dtype=torch.long),
        }
    )

    # Apply protein featurizer only (ligand is empty)
    data = protein_featurizer(data)

    # Cluster pocket atoms into arm regions for subpocket mode
    pocket_masks, actual_num_arms = cluster_pocket_atoms(data.protein_pos, num_arms)
    data.pocket_atom_masks = torch.stack(pocket_masks)  # [num_arms, n_protein_atoms]
    data.num_arms = actual_num_arms
    data.num_scaffold = 1
    data.ligand_atom_mask = torch.empty([0], dtype=torch.long)
    print(f"  Pocket clustered into {actual_num_arms} arm regions")

    # NOTE: Do NOT apply decomp_indicator here — init_transform applies it during sampling

    # Build init_transform for sampling (applied inside sample function)
    prior_mode = "subpocket"
    init_transform_list = [
        trans.ComputeLigandAtomNoiseDist(version=prior_mode),
        decomp_indicator,
    ]
    if getattr(cfg.model, 'bond_diffusion', False):
        init_transform_list.append(
            trans.FeaturizeLigandBond(mode=ligand_bond_mode, set_bond_type=False))
    init_transform = Compose(init_transform_list)

    # Load prior num-atoms configs (must be loaded dicts, not paths)
    arms_config_path = os.path.join(DECOMPDIFF_DIR, "utils/evaluation/arm_num_config.pkl")
    scaffold_config_path = os.path.join(DECOMPDIFF_DIR, "utils/evaluation/scaffold_num_config.pkl")
    with open(arms_config_path, 'rb') as f:
        arms_natoms_config = pickle.load(f)
    with open(scaffold_config_path, 'rb') as f:
        scaffold_natoms_config = pickle.load(f)

    # Sample
    print(f"Generating {num_samples} molecules ({num_steps} steps, batch={batch_size})...")

    import scripts.sample_diffusion_decomp as sample_module
    # Set module-level globals used inside sample function
    sample_module.full_protein_pos = data.protein_pos.clone()
    # Fake args object for recon_with_bond flag
    class _FakeArgs:
        recon_with_bond = False  # atom-based recon more robust for novel pockets
    sample_module.args = _FakeArgs()
    from scripts.sample_diffusion_decomp import sample_diffusion_ligand_decomp

    results = sample_diffusion_ligand_decomp(
        model=model,
        data=data,
        init_transform=init_transform,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        prior_mode=prior_mode,
        num_steps=num_steps,
        center_pos_mode='protein',
        num_atoms_mode='prior',
        atom_enc_mode=ligand_atom_mode,
        bond_fc_mode=ligand_bond_mode,
        arms_natoms_config=arms_natoms_config,
        scaffold_natoms_config=scaffold_natoms_config,
    )

    # Collect and save molecules (results is list of dicts from sampling function)
    sdf_dir = os.path.join(output_dir, "SDF")
    os.makedirs(sdf_dir, exist_ok=True)

    n_saved = 0
    smiles_list = []

    for i, res in enumerate(results):
        mol = res.get('mol')
        smi = res.get('smiles', '')
        if mol is None or not smi or '.' in smi:
            # Try fallback reconstruction without bond info
            try:
                pred_pos_i = res['pred_pos']
                pred_v_i = res['pred_v']
                pred_aromatic = trans.is_aromatic_from_index(pred_v_i, mode=ligand_atom_mode)
                pred_atom_type = trans.get_atomic_number_from_index(pred_v_i, mode=ligand_atom_mode)
                mol = recon.reconstruct_from_generated(pred_pos_i, pred_atom_type, pred_aromatic)
                if mol is None:
                    continue
                smi = Chem.MolToSmiles(mol)
                if '.' in smi:
                    continue
            except Exception:
                continue
        sdf_path = os.path.join(sdf_dir, f"{n_saved}.sdf")
        Chem.MolToMolFile(mol, sdf_path)
        smiles_list.append(smi)
        n_saved += 1

    # Save SMILES
    with open(os.path.join(output_dir, "SMILES.txt"), "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

    print(f"Saved {n_saved} molecules to {sdf_dir}")
    return n_saved


def main():
    parser = argparse.ArgumentParser(description="DecompDiff molecule generation wrapper")
    parser.add_argument("--target", required=True, help="Target protein PDB file")
    parser.add_argument("--chain", default="A", help="Target chain ID")
    parser.add_argument("--binding_residues", required=True,
                        help="Comma-separated binding residue numbers")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of molecules to generate")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DecompDiff checkpoint (uni_o2_bond.pt)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--contact_dist", type=float, default=10.0)
    parser.add_argument("--num_arms", type=int, default=3,
                        help="Number of arm regions to cluster pocket into (default: 3)")

    args = parser.parse_args()
    binding_residues = [int(r) for r in args.binding_residues.split(",")]

    # Extract pocket
    work_dir = os.path.join(args.output_dir, "prep")
    os.makedirs(work_dir, exist_ok=True)
    pocket_path = os.path.join(work_dir, "pocket.pdb")

    print(f"Extracting pocket from {args.target} chain {args.chain}...")
    extract_pocket_pdb(
        args.target, args.chain, binding_residues,
        contact_dist=args.contact_dist,
        out_path=pocket_path,
    )

    # Run generation
    run_decompdiff(
        pocket_pdb=pocket_path,
        num_samples=args.num_samples,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        num_arms=args.num_arms,
    )


if __name__ == "__main__":
    main()
