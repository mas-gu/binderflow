#!/usr/bin/env python3
"""
rerank_molecules.py — Re-rank and merge molecule design results.

Reads rankings.csv from one or more previous generate_molecules.py runs,
deduplicates by SMILES, applies hard filters, recomputes pocket_fit,
re-ranks with custom weights, and regenerates all output artifacts.

No GPU needed. No docking. Vina scores are carried from original runs.

Usage:
    conda activate boltz

    # Re-rank single run with filters
    python rerank_molecules.py --target target.pdb --site "A:325-330" \\
        --results_dir ./run1/ --out_dir ./reranked/ \\
        --min_qed 0.5 --max_sa 4.0

    # Merge de novo + library screening runs
    python rerank_molecules.py --target target.pdb --site "A:325-330" \\
        --results_dir ./denovo/,./drugbank/,./ppi/ \\
        --out_dir ./merged/

    # Custom weights (higher Vina emphasis)
    python rerank_molecules.py --target target.pdb --site "A:325-330" \\
        --results_dir ./run1/ --out_dir ./reranked/ \\
        --score_weights 0.50,0.25,0.15,0.10

    # Boltz-2 affinity rescoring on top 100 (adds columns only, no reranking)
    python rerank_molecules.py --target target.pdb --site "A:325-330" \\
        --results_dir ./run1/ --out_dir ./reranked/ \\
        --boltz_affinity --boltz_affinity_top 100

    # Blend Boltz-2 affinity into combined_score (weight 0.25)
    python rerank_molecules.py --target target.pdb --site "A:325-330" \\
        --results_dir ./run1/ --out_dir ./reranked/ \\
        --boltz_affinity --boltz_affinity_top 100 \\
        --boltz_affinity_weight 0.25

    # Final sort by prob_binder (ignore combined_score for rescored mols)
    python rerank_molecules.py --target target.pdb --site "A:325-330" \\
        --results_dir ./run1/ --out_dir ./reranked/ \\
        --boltz_affinity --boltz_affinity_top 100 \\
        --boltz_affinity_sort
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ── Imports from pipeline ────────────────────────────────────────────────────

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_binders as _gb
from generate_binders import log
from pocket_utils import (
    extract_pocket_pdb, compute_pocket_center, compute_bbox_size,
    parse_site_multichain, site_pairs_chains,
)
from mol_scoring import (
    rank_molecules, compute_pocket_fit, write_rankings_csv, compute_diversity,
)
from generate_molecules import (
    plot_mol_dashboard, copy_top_molecules, TOOL_COLORS,
)


# ── Core functions ───────────────────────────────────────────────────────────

def load_rankings_csv(csv_path):
    """Read a rankings.csv and return list of dicts with proper types."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    float_keys = [
        "qed", "sa_score", "vina_score", "mw", "logp", "tpsa",
        "molar_refractivity", "fsp3", "combined_score",
        "ligand_efficiency", "pocket_fit",
        "boltz_affinity_log_ic50", "boltz_affinity_prob_binder",
    ]
    int_keys = [
        "rank", "hbd", "hba", "lipinski_violations",
        "num_heavy_atoms", "num_rotatable_bonds",
    ]

    for r in rows:
        for k in float_keys:
            v = r.get(k, "")
            if v == "" or v == "nan" or v is None:
                r[k] = float("nan")
            else:
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    r[k] = float("nan")
        for k in int_keys:
            v = r.get(k, "")
            if v == "" or v == "nan" or v is None:
                r[k] = 0
            else:
                try:
                    r[k] = int(float(v))
                except (ValueError, TypeError):
                    r[k] = 0
        # Boolean fields
        for k in ["lipinski_pass", "veber_pass"]:
            v = r.get(k, "")
            if v in (True, "True", "true", "1"):
                r[k] = True
            elif v in (False, "False", "false", "0", ""):
                r[k] = False

    return rows


def collect_from_dirs(results_dirs):
    """Collect molecules from one or more results directories.

    Returns (all_molecules, n_dupes) where molecules are deduplicated by
    canonical SMILES. When merging multiple dirs, design_ids are prefixed
    with directory label to avoid collisions.
    """
    all_mols = []
    seen_smiles = set()
    n_dupes = 0
    merge_mode = len(results_dirs) > 1

    for rd in results_dirs:
        rd = Path(rd)
        csv_path = rd / "rankings.csv"
        if not csv_path.exists():
            log(f"  Warning: no rankings.csv in {rd}, skipping")
            continue

        mols = load_rankings_csv(csv_path)
        rd_label = rd.name  # e.g. "2026-04-13_pdia6_denovo"

        for m in mols:
            smi = m.get("smiles", "")
            if not smi:
                continue

            # Canonical SMILES for dedup
            try:
                from rdkit import Chem
                mol_obj = Chem.MolFromSmiles(smi)
                if mol_obj:
                    can_smi = Chem.MolToSmiles(mol_obj)
                else:
                    can_smi = smi
            except Exception:
                can_smi = smi

            if can_smi in seen_smiles:
                n_dupes += 1
                continue
            seen_smiles.add(can_smi)

            # Prefix design_id when merging multiple dirs
            if merge_mode:
                orig_id = m.get("design_id", "")
                m["design_id"] = f"{rd_label}__{orig_id}"
                m["_orig_design_id"] = orig_id

            m["_source_dir"] = str(rd)

            # Resolve sdf_path — may be absolute or relative
            sdf = m.get("sdf_path", "")
            if sdf and not Path(sdf).exists():
                # Try relative to source dir
                alt = rd / sdf
                if alt.exists():
                    m["sdf_path"] = str(alt)

            all_mols.append(m)

        log(f"  {rd_label}: {len(mols)} molecules loaded")

    return all_mols, n_dupes


def _is_nan(v):
    """Check if value is NaN or None."""
    if v is None:
        return True
    try:
        return v != v  # NaN != NaN
    except (TypeError, ValueError):
        return True


def apply_filters(molecules, min_qed=None, max_sa=None, min_pocket_fit=None,
                  mw_range=None, min_vina=None, max_vina=None):
    """Apply hard filters. Molecules with missing values pass through (not penalized)."""
    n_start = len(molecules)
    log_lines = []

    def _filter(key, test, label):
        nonlocal molecules
        before = len(molecules)
        molecules = [m for m in molecules
                     if _is_nan(m.get(key)) or test(m.get(key))]
        removed = before - len(molecules)
        if removed:
            log_lines.append(f"  {label}: removed {removed}")

    if min_qed is not None:
        _filter("qed", lambda v: v >= min_qed, f"min_qed >= {min_qed}")

    if max_sa is not None:
        _filter("sa_score", lambda v: v <= max_sa, f"max_sa <= {max_sa}")

    if min_pocket_fit is not None:
        _filter("pocket_fit", lambda v: v >= min_pocket_fit, f"min_pocket_fit >= {min_pocket_fit}")

    if mw_range is not None:
        mw_min, mw_max = mw_range
        _filter("mw", lambda v: mw_min <= v <= mw_max, f"MW {mw_min}-{mw_max}")

    if min_vina is not None:
        _filter("vina_score", lambda v: v <= min_vina, f"min_vina <= {min_vina}")

    if max_vina is not None:
        _filter("vina_score", lambda v: v >= max_vina, f"max_vina >= {max_vina}")

    n_removed = n_start - len(molecules)
    return molecules, log_lines, n_removed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-rank and merge molecule design results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-rank with filters
  python rerank_molecules.py --target t.pdb --site "A:325-330" \\
      --results_dir ./run1/ --out_dir ./reranked/ --min_qed 0.5

  # Merge multiple runs
  python rerank_molecules.py --target t.pdb --site "A:325-330" \\
      --results_dir ./denovo/,./drugbank/,./ppi/ --out_dir ./merged/
""")

    parser.add_argument("--target", required=True, help="Target protein PDB")
    parser.add_argument("--site", required=True, help='Binding site (e.g. "A:325-330")')
    parser.add_argument("--results_dir", required=True,
                        help="Comma-separated input directories with rankings.csv")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--score_weights", default="0.40,0.35,0.15,0.10",
                        help='Vina,QED,SA,PocketFit[,Affinity] weights — 4 or 5 floats '
                             '(default: "0.40,0.35,0.15,0.10"). '
                             'If 5th weight given, --boltz_affinity_weight is ignored.')
    parser.add_argument("--pocket_dist", type=float, default=10.0,
                        help="Pocket extraction distance (default: 10.0)")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top molecules to copy (default: 50)")

    # Hard filters
    parser.add_argument("--min_qed", type=float, default=None,
                        help="Min QED drug-likeness (0-1)")
    parser.add_argument("--max_sa", type=float, default=None,
                        help="Max SA score (1=easy, 10=hard)")
    parser.add_argument("--min_pocket_fit", type=float, default=None,
                        help="Min pocket fit fraction (0-1)")
    parser.add_argument("--mw_range", default=None,
                        help='MW range, e.g. "150,600"')
    parser.add_argument("--min_vina", type=float, default=None,
                        help="Min (most negative) Vina score, e.g. -10")
    parser.add_argument("--max_vina", type=float, default=None,
                        help="Max (least negative) Vina score, e.g. -3")

    # ── Boltz-2 affinity rescoring ───────────────────────────────────────
    parser.add_argument("--boltz_affinity", action="store_true",
                        help="Rescore top-N molecules with Boltz-2 affinity model")
    parser.add_argument("--boltz_affinity_top", type=int, default=100,
                        help="Rescore top-N molecules by initial combined_score (default: 100)")
    parser.add_argument("--boltz_affinity_weight", type=float, default=0.0,
                        help="Weight of prob_binder in combined_score when rescoring "
                             "(default: 0.0 = adds columns only, no re-ranking). "
                             "Recommended 0.2-0.3. Other weights rescale to sum=1.")
    parser.add_argument("--boltz_affinity_sort", action="store_true",
                        help="Final sort by boltz_affinity_prob_binder (desc); "
                             "rescored molecules float to top, unranked keep "
                             "their combined_score order below them.")
    parser.add_argument("--boltz_affinity_mw_correction", action="store_true",
                        help="Pass --affinity_mw_correction to Boltz-2 (MW-normalized affinity)")
    parser.add_argument("--boltz_affinity_sampling_steps", type=int, default=200,
                        help="Boltz-2 --sampling_steps_affinity (default: 200)")
    parser.add_argument("--boltz_affinity_diffusion_samples", type=int, default=5,
                        help="Boltz-2 --diffusion_samples_affinity ensemble size (default: 5)")
    parser.add_argument("--boltz_devices", type=int, default=1,
                        help="Number of GPUs for Boltz-2 affinity batch (default: 1)")

    args = parser.parse_args()

    # ── Parse inputs ─────────────────────────────────────────────────────
    target_path = Path(args.target).resolve()
    if not target_path.exists():
        sys.exit(f"ERROR: target not found: {target_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dirs = [Path(d.strip()).resolve() for d in args.results_dir.split(",")]
    for rd in results_dirs:
        if not rd.exists():
            sys.exit(f"ERROR: results directory not found: {rd}")

    # Parse weights (3, 4, or 5 floats).
    # 3 → (vina, qed, sa) + fit=0 + aff=args.boltz_affinity_weight
    # 4 → (vina, qed, sa, fit) + aff=args.boltz_affinity_weight
    # 5 → (vina, qed, sa, fit, aff)  — overrides --boltz_affinity_weight
    w_parts = args.score_weights.split(",")
    if len(w_parts) == 3:
        wv, wq, ws = (float(w) for w in w_parts)
        weights = (wv, wq, ws, 0.0, args.boltz_affinity_weight)
    elif len(w_parts) == 4:
        wv, wq, ws, wf = (float(w) for w in w_parts)
        weights = (wv, wq, ws, wf, args.boltz_affinity_weight)
    elif len(w_parts) == 5:
        weights = tuple(float(w) for w in w_parts)
    else:
        sys.exit("--score_weights must be 3, 4, or 5 comma-separated floats")

    # Parse MW range
    mw_range = None
    if args.mw_range:
        parts = args.mw_range.split(",")
        if len(parts) != 2:
            sys.exit('--mw_range must be "min,max" e.g. "150,600"')
        mw_range = (float(parts[0]), float(parts[1]))

    # Parse site (multi-chain supported)
    site_pairs = parse_site_multichain(args.site)
    site_chains = site_pairs_chains(site_pairs)
    is_multichain = len(site_chains) > 1

    # ── Header ───────────────────────────────────────────────────────────
    log("=" * 60)
    log("rerank_molecules.py — Molecule Re-ranking")
    log("=" * 60)
    log(f"Target:           {target_path}")
    if is_multichain:
        per_chain = ", ".join(
            f"{c}:{sum(1 for cc, _ in site_pairs if cc == c)} res"
            for c in site_chains)
        log(f"Binding site:     {args.site} "
            f"({len(site_pairs)} residues across chains {site_chains}; {per_chain})")
    else:
        log(f"Binding site:     {args.site} "
            f"({len(site_pairs)} residues, chain {site_chains[0]})")
    log(f"Results dirs:     {len(results_dirs)}")
    for rd in results_dirs:
        log(f"  {rd}")
    log(f"Score weights:    Vina={weights[0]}, QED={weights[1]}, SA={weights[2]}, "
        f"Fit={weights[3]}, Aff={weights[4]}")
    if args.boltz_affinity:
        log(f"Boltz affinity:   top-{args.boltz_affinity_top}, "
            f"weight={weights[4]}, sort={args.boltz_affinity_sort}, "
            f"mw_corr={args.boltz_affinity_mw_correction}")
    log(f"Output:           {out_dir}")

    # ── Pocket extraction ────────────────────────────────────────────────
    log(f"\n{'─' * 50}")
    log(f"STEP: Pocket extraction")
    log(f"{'─' * 50}")

    pocket_dir = out_dir / "pocket"
    pocket_dir.mkdir(parents=True, exist_ok=True)

    pocket_pdb = extract_pocket_pdb(
        str(target_path), site_pairs,
        distance_cutoff=args.pocket_dist)

    pocket_out = str(pocket_dir / "pocket.pdb")
    import shutil
    shutil.copy2(pocket_pdb, pocket_out)
    pocket_pdb = pocket_out

    center = compute_pocket_center(str(target_path), site_pairs)
    bbox_size = compute_bbox_size(str(target_path), site_pairs)
    bbox_cap = 25.0 if is_multichain else 20.0
    if bbox_size > bbox_cap:
        bbox_size = bbox_cap

    log(f"Pocket center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")

    # ── Collect molecules ────────────────────────────────────────────────
    log(f"\n{'─' * 50}")
    log(f"STEP: Collecting molecules from {len(results_dirs)} dir(s)")
    log(f"{'─' * 50}")

    all_mols, n_dupes = collect_from_dirs(results_dirs)
    log(f"\nCollected: {len(all_mols)} unique molecules ({n_dupes} duplicates removed)")

    if not all_mols:
        log("No molecules found. Exiting.")
        return

    # ── Recompute pocket_fit ─────────────────────────────────────────────
    log(f"\n{'─' * 50}")
    log(f"STEP: Computing pocket_fit")
    log(f"{'─' * 50}")

    n_fit = 0
    for i, m in enumerate(all_mols):
        sdf_path = m.get("sdf_path", "")
        if sdf_path and os.path.exists(sdf_path):
            fit = compute_pocket_fit(sdf_path, pocket_pdb)
            m["pocket_fit"] = round(fit, 3) if fit == fit else float("nan")
            if fit == fit:
                n_fit += 1
        else:
            m["pocket_fit"] = float("nan")

        if (i + 1) % 500 == 0:
            log(f"  ... {i + 1}/{len(all_mols)}")

    log(f"  Pocket fit computed for {n_fit}/{len(all_mols)} molecules")

    # ── Apply filters ────────────────────────────────────────────────────
    filters_active = any(x is not None for x in [
        args.min_qed, args.max_sa, args.min_pocket_fit, mw_range,
        args.min_vina, args.max_vina
    ])

    if filters_active:
        log(f"\n{'─' * 50}")
        log(f"STEP: Applying filters")
        log(f"{'─' * 50}")

        all_mols, filter_log, n_removed = apply_filters(
            all_mols,
            min_qed=args.min_qed,
            max_sa=args.max_sa,
            min_pocket_fit=args.min_pocket_fit,
            mw_range=mw_range,
            min_vina=args.min_vina,
            max_vina=args.max_vina,
        )
        for line in filter_log:
            log(line)
        log(f"  Total removed: {n_removed}, remaining: {len(all_mols)}")

    if not all_mols:
        log("All molecules filtered out. Exiting.")
        return

    # ── Initial rank ─────────────────────────────────────────────────────
    # If Boltz-2 affinity is enabled, we rank once (without affinity) to
    # pick the top-N candidates, then re-rank after affinity is computed.
    log(f"\n{'─' * 50}")
    log(f"STEP: Ranking {len(all_mols)} molecules"
        + (" (initial, for top-N pick)" if args.boltz_affinity else ""))
    log(f"{'─' * 50}")

    ranked = rank_molecules(all_mols, weights=weights, pocket_pdb=pocket_pdb)

    # ── Boltz-2 affinity rescoring ───────────────────────────────────────
    if args.boltz_affinity:
        log(f"\n{'─' * 50}")
        log(f"STEP: Boltz-2 affinity on top-{args.boltz_affinity_top}")
        log(f"{'─' * 50}")
        from mol_affinity import run_boltz_affinity_batch
        ranked = run_boltz_affinity_batch(
            ranked,
            target_pdb=str(target_path),
            site_pairs=site_pairs,
            out_dir=out_dir,
            top_n=args.boltz_affinity_top,
            mw_correction=args.boltz_affinity_mw_correction,
            sampling_steps=args.boltz_affinity_sampling_steps,
            diffusion_samples=args.boltz_affinity_diffusion_samples,
            boltz_devices=args.boltz_devices,
        )
        log(f"\n{'─' * 50}")
        log(f"STEP: Re-ranking with affinity weight={weights[4]}")
        log(f"{'─' * 50}")
        ranked = rank_molecules(ranked, weights=weights, pocket_pdb=pocket_pdb)

        if args.boltz_affinity_sort:
            # Sort by prob_binder desc; NaN (unscored) drops to bottom
            # while preserving combined_score order among themselves.
            def _sort_key(m):
                p = m.get("boltz_affinity_prob_binder", float("nan"))
                has_aff = isinstance(p, (int, float)) and p == p
                return (0 if has_aff else 1,
                        -(p if has_aff else 0.0),
                        -m.get("combined_score", 0.0))
            ranked.sort(key=_sort_key)
            for i, m in enumerate(ranked):
                m["rank"] = i + 1

    # ── Per-tool diversity ───────────────────────────────────────────────
    from rdkit import Chem
    tools_present = sorted(set(m.get("tool", "unknown") for m in ranked))
    for tool in tools_present:
        tool_smiles = [m.get("smiles", "") for m in ranked if m.get("tool") == tool]
        tool_mols = [Chem.MolFromSmiles(s) for s in tool_smiles if s]
        tool_mols = [m for m in tool_mols if m is not None]
        if len(tool_mols) >= 2:
            div = compute_diversity(tool_mols)
            log(f"  {tool} diversity (Tanimoto): {div:.3f}")

    # ── Write outputs ────────────────────────────────────────────────────
    log(f"\n{'─' * 50}")
    log(f"STEP: Writing outputs")
    log(f"{'─' * 50}")

    # Clean internal keys before writing
    for m in ranked:
        for k in ["_source_dir", "_orig_design_id",
                  "_affinity_design_id", "_affinity_skipped_reason"]:
            m.pop(k, None)

    write_rankings_csv(ranked, out_dir / "rankings.csv")

    copy_top_molecules(ranked, out_dir / "top_molecules", n=args.top_n,
                       target_pdb=str(target_path),
                       pocket_pdb=pocket_pdb,
                       site_pairs=site_pairs)

    plot_mol_dashboard(ranked, out_dir / "dashboard.png",
                       title=f"Reranked ({out_dir.name})")

    # ── Summary ──────────────────────────────────────────────────────────
    log(f"\n{'=' * 60}")
    log(f"DONE")
    log(f"{'=' * 60}")

    # Tool breakdown
    tool_counts = Counter(m.get("tool", "unknown") for m in ranked)
    for t, c in tool_counts.most_common():
        log(f"  {t:20s}: {c} molecules")

    # Top 5
    log(f"\nTop 5 molecules:")
    for m in ranked[:5]:
        v = m.get("vina_score", float("nan"))
        vstr = f"{v:.1f}" if v == v else "N/A"
        fit = m.get("pocket_fit", float("nan"))
        fstr = f"{fit:.2f}" if isinstance(fit, float) and fit == fit else "N/A"
        p = m.get("boltz_affinity_prob_binder", float("nan"))
        pstr = f"{p:.2f}" if isinstance(p, (int, float)) and p == p else "N/A"
        log(f"  rank {m['rank']:>4d}  {m.get('tool', ''):15s}  "
            f"Vina={vstr:>7s}  QED={m.get('qed', 0):.2f}  "
            f"Fit={fstr}  Pb={pstr}  MW={m.get('mw', 0):.0f}  "
            f"score={m.get('combined_score', 0):.3f}")

    log(f"\nOutput: {out_dir}")
    log(f"  rankings.csv    — {len(ranked)} molecules")
    log(f"  top_molecules/  — top {min(args.top_n, len(ranked))} SDFs + SMILES")
    log(f"  dashboard.png   — summary plot")


if __name__ == "__main__":
    main()
