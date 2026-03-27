#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume Mas
# Author: Guillaume Mas
# SPDX-License-Identifier: MIT
"""
rerank_binders.py — Re-validate and re-rank existing design outputs.

Scans one or more output directories from previous generate_binders3*.py runs,
collects all designs, and re-runs the validation + ranking pipeline with
(optionally) different parameters. Supports merging results from multiple runs.

═══════════════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════════════
    conda activate boltz

    # Re-rank a single run (fast — reuse existing validation)
    python rerank_binders.py \\
        --target ./outputs/KRASB/8ECR_monomer.pdb \\
        --site "A:11-17,119-124" \\
        --results_dir ./out8/ \\
        --out_dir ./out8_reranked/ \\
        --rank_only \\
        --score_weights 0.3,0.6,0.1

    # Merge multiple runs into one unified ranking
    python rerank_binders.py \\
        --target ./outputs/KRASB/8ECR_monomer.pdb \\
        --site "A:11-17,119-124" \\
        --results_dir ./out_helix,./out_balanced,./out_beta \\
        --out_dir ./merged_ranking/ \\
        --rank_only

    # Re-rank with SS bias + geometric site filter
    python rerank_binders.py \\
        --target ./outputs/KRASB/8ECR_monomer.pdb \\
        --site "A:11-17,119-124" \\
        --results_dir ./out8/ \\
        --out_dir ./out8_helix/ \\
        --rank_only \\
        --ss_bias helix \\
        --max_site_dist 12.0

    # Backward-compatible: --out_dir alone reads+writes same dir (in-place)
    python rerank_binders.py \\
        --target ./outputs/KRASB/8ECR_monomer.pdb \\
        --site "A:11-17,119-124" \\
        --out_dir ./out8/ \\
        --rank_only \\
        --score_weights 0.3,0.6,0.1

    # Re-validate + re-rank with different score weights
    python rerank_binders.py \\
        --target ./outputs/KRASB/8ECR_monomer.pdb \\
        --site "A:11-17,119-124" \\
        --results_dir ./out8/ \\
        --out_dir ./out8_revalidated/ \\
        --score_weights 0.3,0.6,0.1

    # Dry run
    python rerank_binders.py \\
        --target ./outputs/KRASB/8ECR_monomer.pdb \\
        --site "A:11-17,119-124" \\
        --results_dir ./out8/ \\
        --out_dir ./test/ \\
        --dry_run

═══════════════════════════════════════════════════════════════════════════════
WHAT IT DOES
═══════════════════════════════════════════════════════════════════════════════
1. Scans results_dir/{tool}/ for each tool's outputs (backbones, CIFs, CSVs)
   — supports multiple dirs for merging (deduplicates by sequence)
2. Loads existing validation from each results_dir/validation/
3. Runs ESMFold → Boltz-2 → geometric filter → Rosetta → SS analysis → ranking
   (or subset with --rank_only)
4. Writes new rankings.csv, top_designs/, dashboard.png to out_dir

Existing validation results (ESMFold PDBs, Boltz CIFs) are reused if present.
New validation outputs go to out_dir/validation/.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

# Import from generate_binders — has all features including
# SS bias, geometric filter, populate_binder_ss, batch Boltz-2, multi-GPU
from generate_binders import (
    # Constants
    ALL_TOOLS, TOOL_COLORS, PROTEINA_DIR,
    PROTEINA_COMPLEXA_DIR, PROTEINA_COMPLEXA_VENV,
    IPTM_NATIVE_TOOLS, BACKBONE_ONLY_TOOLS,
    SS_BIAS_PARAMS,
    # Helpers
    log, run_cmd, _safe_float, _get_tool_from_design_id,
    extract_pdb_chain_info, extract_cif_chain_info, get_chain_info,
    parse_site, parse_length, parse_score_weights,
    # Metrics loaders
    _load_boltzgen_metrics, _load_bindcraft_metrics,
    # Validation + filtering
    validate_esmfold, validate_boltz, rosetta_score_interfaces,
    geometric_site_filter, populate_binder_composition, populate_interface_composition, populate_binder_ss,
    # Ranking + output
    compute_combined_score, rank_designs,
    write_rankings_csv, copy_top_designs, plot_dashboard,
    run_plip_analysis,
)


# ── Design collectors (scan existing outputs) ────────────────────────────────

def collect_rfdiffusion(tool_dir):
    """Collect designs from an existing rfdiffusion/ output directory."""
    tool_dir = Path(tool_dir)
    bb_dir = tool_dir / "backbones"
    sq_dir = tool_dir / "sequences"

    if not bb_dir.exists():
        return []

    backbones = sorted(bb_dir.glob("design_*.pdb"))
    designs = []

    for i, bb_pdb in enumerate(backbones):
        fasta_path = sq_dir / "seqs" / f"{bb_pdb.stem}.fa"
        if not fasta_path.exists():
            fastas = sorted((sq_dir / "seqs").glob(f"{bb_pdb.stem}*.fa"))
            if not fastas:
                continue
            fasta_path = fastas[0]

        best_seq, best_conf, best_seq_rec = None, -1.0, float("nan")
        current_header = None
        for line in fasta_path.read_text().splitlines():
            if line.startswith(">"):
                current_header = line
            elif current_header and line.strip():
                m_conf = re.search(r"overall_confidence=([0-9.]+)", current_header)
                m_rec = re.search(r"seq_rec=([0-9.]+)", current_header)
                conf = float(m_conf.group(1)) if m_conf else 0.0
                seq_rec = float(m_rec.group(1)) if m_rec else float("nan")
                if conf > best_conf:
                    best_conf = conf
                    best_seq = line.strip()
                    best_seq_rec = seq_rec
                current_header = None

        if best_seq:
            binder_seq = best_seq.split(":")[-1] if ":" in best_seq else best_seq
            designs.append({
                "design_id":          f"rfdiffusion_{i:04d}",
                "binder_sequence":    binder_seq,
                "backbone_pdb":       str(bb_pdb),
                "native_score":       best_conf,
                "native_score_name":  "mpnn_confidence",
                "ligandmpnn_seq_rec": best_seq_rec,
            })

    return designs


def collect_boltzgen(tool_dir):
    """Collect designs from an existing boltzgen/ output directory."""
    tool_dir = Path(tool_dir)
    raw_dir = tool_dir / "boltzgen_raw"
    if not raw_dir.exists():
        return []

    final_dirs = sorted(raw_dir.glob("final_ranked_designs/final_*_designs"))
    if not final_dirs:
        final_dirs = sorted(raw_dir.glob("intermediate_designs_inverse_folded/refold_cif"))

    cifs = []
    for d in final_dirs:
        cifs.extend(sorted(d.glob("*.cif")))

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

    bg_metrics = _load_boltzgen_metrics(raw_dir)

    designs = []
    for i, cif in enumerate(cifs):
        chain_info = extract_cif_chain_info(cif)
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
        if seq in bg_metrics:
            d.update(bg_metrics[seq])
        designs.append(d)

    return designs


def collect_bindcraft(tool_dir):
    """Collect designs from an existing bindcraft/ output directory."""
    tool_dir = Path(tool_dir)
    design_path = tool_dir / "bindcraft_output"
    if not design_path.exists():
        return []

    ranked_dir = design_path / "Accepted" / "Ranked"
    accepted_dir = design_path / "Accepted"
    pdbs = sorted(ranked_dir.glob("*.pdb")) if ranked_dir.exists() else []
    if not pdbs:
        pdbs = sorted(accepted_dir.glob("*.pdb")) if accepted_dir.exists() else []

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

    bc_metrics = _load_bindcraft_metrics(design_path)

    designs = []
    for i, pdb in enumerate(pdbs):
        chain_info = extract_pdb_chain_info(pdb)
        binder_info = chain_info.get("B") or (next(iter(chain_info.values())) if chain_info else {})
        seq = binder_info.get("sequence", "")
        if not seq:
            continue
        native_score = (iptm_scores.get(pdb.stem)
                        or next((v for k, v in iptm_scores.items() if k in pdb.stem), 0.0))
        d = {
            "design_id":         f"bindcraft_{i:04d}",
            "binder_sequence":   seq,
            "binder_pdb":        str(pdb),
            "native_score":      native_score,
            "native_score_name": "bindcraft_iptm",
        }
        matched_key = (pdb.stem if pdb.stem in bc_metrics
                       else next((k for k in bc_metrics if k in pdb.stem), None))
        if matched_key:
            d.update(bc_metrics[matched_key])
        designs.append(d)

    return designs


def collect_pxdesign(tool_dir, chain_id="A"):
    """Collect designs from an existing pxdesign/ output directory."""
    tool_dir = Path(tool_dir)
    px_output = tool_dir / "pxdesign_output"
    if not px_output.exists():
        return []

    summary_csv = None
    for candidate in [
        px_output / "summary.csv",
        *px_output.glob("*/summary.csv"),
        *px_output.glob("design_outputs/*/summary.csv"),
    ]:
        if candidate.exists():
            summary_csv = candidate
            break

    if not summary_csv:
        # Fallback: scan for CIF files
        designs = []
        for cif in sorted(px_output.rglob("*.cif")):
            seq = ""
            try:
                ci = extract_cif_chain_info(cif)
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
        return designs

    designs = []
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            seq = row.get("sequence", "")
            if not seq:
                continue

            rank = row.get("rank", str(i + 1))
            iptm = _safe_float(row.get("ptx_iptm_binder", "") or row.get("af2_iptm", ""))
            ipae = _safe_float(row.get("af2_ipAE", ""))
            plddt = _safe_float(row.get("ptx_plddt", "") or row.get("af2_plddt", ""))
            ptm = _safe_float(row.get("ptx_ptm", "") or row.get("af2_ptm", ""))
            rmsd = _safe_float(row.get("ptx_pred_design_rmsd", "") or
                               row.get("af2_complex_pred_design_rmsd", ""))
            cluster_id = row.get("cluster_id", "")

            filter_level = "none"
            if row.get("Protenix-success", "").strip().lower() == "true":
                filter_level = "Protenix"
            elif row.get("Protenix-basic-success", "").strip().lower() == "true":
                filter_level = "Protenix-basic"
            elif row.get("AF2-IG-success", "").strip().lower() == "true":
                filter_level = "AF2-IG"
            elif row.get("AF2-IG-easy-success", "").strip().lower() == "true":
                filter_level = "AF2-IG-easy"

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

    return designs


def collect_proteina(tool_dir):
    """Collect designs from an existing proteina/ output directory."""
    tool_dir = Path(tool_dir)
    bb_dir = tool_dir / "backbones"
    sq_dir = tool_dir / "sequences"

    if not bb_dir.exists():
        return []

    backbones = sorted(bb_dir.glob("design_*.pdb"))
    designs = []

    for i, bb_pdb in enumerate(backbones):
        fasta_path = sq_dir / "seqs" / f"{bb_pdb.stem}.fa"
        if not fasta_path.exists():
            fastas = sorted((sq_dir / "seqs").glob(f"{bb_pdb.stem}*.fa"))
            if not fastas:
                continue
            fasta_path = fastas[0]

        best_seq, best_conf, best_seq_rec = None, 999.0, float("nan")
        current_header = None
        for line in fasta_path.read_text().splitlines():
            if line.startswith(">"):
                current_header = line
            elif current_header and line.strip():
                seq_candidate = line.strip()
                # Skip backbone sequence (poly-Ala or near-uniform)
                if len(set(seq_candidate)) <= 2:
                    current_header = None
                    continue
                # ProteinMPNN score = negative log-likelihood (lower = better)
                m_conf = re.search(r"(?:overall_confidence|score)=([0-9.]+)", current_header)
                m_rec = re.search(r"(?:seq_rec|seq_recovery)=([0-9.]+)", current_header)
                conf = float(m_conf.group(1)) if m_conf else 999.0
                seq_rec = float(m_rec.group(1)) if m_rec else float("nan")
                if conf < best_conf:  # lower score = better for ProteinMPNN
                    best_conf = conf
                    best_seq = seq_candidate
                    best_seq_rec = seq_rec
                current_header = None

        if best_seq:
            binder_seq = best_seq.split(":")[-1] if ":" in best_seq else best_seq
            designs.append({
                "design_id":          f"proteina_{i:04d}",
                "binder_sequence":    binder_seq,
                "backbone_pdb":       str(bb_pdb),
                "native_score":       best_conf,
                "native_score_name":  "mpnn_score",
                "ligandmpnn_seq_rec": best_seq_rec,
            })

    return designs


def collect_proteina_complexa(tool_dir):
    """Collect designs from an existing proteina_complexa/ output directory."""
    tool_dir = Path(tool_dir)
    if not tool_dir.exists():
        return []

    # Proteina Complexa outputs go to inference/ under PROTEINA_COMPLEXA_DIR,
    # but we also check the tool_dir itself for PDB files that may have been
    # copied or symlinked during the original run.

    # Check for inference outputs in the Complexa repo.
    # Match by parent_dir + tool_dir name to avoid mixing runs.
    inference_base = Path(PROTEINA_COMPLEXA_DIR) / "inference"
    parent_name = tool_dir.parent.name  # e.g., "out345_350"
    run_dirs = sorted(inference_base.glob(f"*binder_target*{parent_name}_{tool_dir.name}*"))
    if not run_dirs:
        run_dirs = sorted(inference_base.glob(f"*binder_target*{tool_dir.name}*"))
    if not run_dirs:
        run_dirs = sorted(inference_base.glob("*binder_target*"))

    sample_pdbs = []
    for rd in run_dirs:
        # Complexa outputs to job_* subdirectories
        for sub in sorted(rd.iterdir()):
            if sub.is_dir() and sub.name.startswith("job_"):
                for p in sorted(sub.glob("*.pdb")):
                    if "_binder" not in p.stem:
                        sample_pdbs.append(p)
        # Fallback: PDBs directly in run dir
        if not sample_pdbs:
            sample_pdbs = sorted(p for p in rd.glob("*.pdb") if "_binder" not in p.stem)

    # Also check tool_dir directly
    if not sample_pdbs:
        sample_pdbs = sorted(p for p in tool_dir.rglob("*.pdb") if "_binder" not in p.stem)

    # Load eval metrics — try rewards CSV first, then binder_results from evaluation_results/
    eval_metrics = {}

    # Rewards CSV (from generate step, always present)
    for rd in run_dirs:
        for rew_csv in sorted(rd.glob("*rewards_*.csv")):
            try:
                with open(rew_csv) as f:
                    for row in csv.DictReader(f):
                        pdb_path = row.get("pdb_path", "")
                        if not pdb_path:
                            continue
                        from pathlib import Path as _P
                        sample_id = _P(pdb_path).parent.name
                        entry = {}
                        for csv_col, pc_key in [
                            ("af2folding_i_ptm_log", "pc_iptm"),
                            ("af2folding_i_pae", "pc_ipae"),
                            ("af2folding_plddt", "pc_plddt"),
                            ("af2folding_ptm_log", "pc_ptm"),
                            ("af2folding_rmsd", "pc_scrmsd"),
                        ]:
                            val = row.get(csv_col, "")
                            if val:
                                entry[pc_key] = _safe_float(val)
                        if entry:
                            eval_metrics[sample_id] = entry
            except Exception:
                pass

    # Binder results CSV (from evaluate step, overrides rewards if present)
    eval_base = Path(PROTEINA_COMPLEXA_DIR) / "evaluation_results"
    for rd in run_dirs:
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
                except Exception:
                    pass

    designs = []
    for i, pdb in enumerate(sample_pdbs):
        chain_info = extract_pdb_chain_info(pdb)
        # Find binder chain (not the target chain)
        binder_info = None
        for cid, info in sorted(chain_info.items()):
            # Assume binder is the shorter chain or any non-target chain
            if binder_info is None or info["length"] < binder_info["length"]:
                binder_info = info
        if not binder_info:
            continue

        seq = binder_info.get("sequence", "")
        if not seq:
            continue

        sample_name = pdb.parent.name if pdb.parent.name.startswith("job_") else pdb.stem
        metrics = eval_metrics.get(sample_name, {})
        native_iptm = metrics.get("pc_iptm", 0.0)
        if native_iptm != native_iptm:
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

    return designs


COLLECTORS = {
    "rfdiffusion":       collect_rfdiffusion,
    "boltzgen":          collect_boltzgen,
    "bindcraft":         collect_bindcraft,
    "pxdesign":          collect_pxdesign,
    "proteina":          collect_proteina,
    "proteina_complexa": collect_proteina_complexa,
}


# ── Reuse existing validation results ────────────────────────────────────────

def load_existing_esmfold(designs, val_dir):
    """Check for existing ESMFold PDBs and populate esmfold_plddt if available.

    Also loads failed ESMFold results from rankings.csv in the parent dir
    (designs that were tested but had pLDDT below threshold — no PDB saved).
    This avoids re-running ESMFold on designs known to fail.
    """
    esm_dir = Path(val_dir) / "esmfold"
    loaded = 0

    # Load failed ESMFold results from rankings.csv (if exists)
    # These designs were tested but pLDDT was below threshold → no PDB saved
    failed_plddt = {}
    rankings_csv = Path(val_dir).parent / "rankings.csv"
    if rankings_csv.exists():
        try:
            import csv
            with open(rankings_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    did = row.get("design_id", "")
                    plddt_str = row.get("esmfold_plddt", "")
                    if did and plddt_str:
                        try:
                            plddt_val = float(plddt_str)
                            if plddt_val == plddt_val:  # not NaN
                                failed_plddt[did] = plddt_val
                        except ValueError:
                            pass
        except Exception:
            pass  # CSV parsing failed, skip

    if not esm_dir.exists() and not failed_plddt:
        return 0

    for d in designs:
        if "esmfold_plddt" in d:
            continue  # already loaded from another results_dir

        # Try PDB file first (designs that passed ESMFold)
        did = d["design_id"]
        pdb_path = esm_dir / f"{did}.pdb" if esm_dir.exists() else None
        if pdb_path and not pdb_path.exists() and "_orig_design_id" in d:
            pdb_path = esm_dir / f"{d['_orig_design_id']}.pdb"

        if pdb_path and pdb_path.exists():
            # Parse pLDDT from B-factors of CA atoms
            bfacs = []
            for line in pdb_path.read_text().splitlines():
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        bfacs.append(float(line[60:66]))
                    except (ValueError, IndexError):
                        pass
            if bfacs:
                import numpy as np
                plddt = float(np.mean(bfacs))
                plddt = plddt * 100.0 if plddt <= 1.0 else plddt
                d["esmfold_plddt"] = plddt
                loaded += 1
                continue

        # Fall back to rankings.csv for failed designs (no PDB but has pLDDT)
        # Use original design_id for lookup (multi-dir merge prefixes IDs)
        orig_did = d.get("_orig_design_id", did)
        if orig_did in failed_plddt:
            d["esmfold_plddt"] = failed_plddt[orig_did]
            loaded += 1
            continue
        if did in failed_plddt:
            d["esmfold_plddt"] = failed_plddt[did]
            loaded += 1
            continue

        # If ESMFold dir exists and has PDBs for the same tool but NOT this
        # design, infer it was tested and failed (no PDB = pLDDT below threshold)
        if esm_dir and esm_dir.exists():
            tool_name = _get_tool_from_design_id(did)
            # Check if any design from same tool has a PDB (tool was processed)
            tool_pdbs = list(esm_dir.glob(f"{tool_name}_*.pdb"))
            if tool_pdbs:
                # This tool was processed but this design has no PDB → failed
                d["esmfold_plddt"] = 0.0  # mark as failed
                loaded += 1

    return loaded


def load_existing_boltz(designs, val_dir, site_row_indices=None):
    """Check for existing Boltz validation results and populate boltz_* fields.

    Parameters
    ----------
    site_row_indices : list of int or None
        0-based row indices in the PAE matrix for site-specific PAE extraction.
    """
    boltz_dir = Path(val_dir) / "boltz"
    if not boltz_dir.exists():
        return 0

    loaded = 0
    # Batch mode output: validation/boltz/boltz_results_batch_inputs/predictions/{design_id}/
    batch_pred_dir = boltz_dir / "boltz_results_batch_inputs" / "predictions"

    for d in designs:
        did = d["design_id"]
        # Try per-design dir first (sequential mode), then batch mode dir
        design_dir = boltz_dir / did
        if not design_dir.exists() and "_orig_design_id" in d:
            design_dir = boltz_dir / d["_orig_design_id"]
        if not design_dir.exists() and batch_pred_dir.exists():
            # Batch mode: predictions stored under boltz_results_batch_inputs/predictions/{design_id}/
            batch_dir = batch_pred_dir / did
            if not batch_dir.exists() and "_orig_design_id" in d:
                batch_dir = batch_pred_dir / d["_orig_design_id"]
            if batch_dir.exists():
                design_dir = batch_dir
        if not design_dir.exists():
            continue

        # Always update complex_cif to Boltz-2 validation CIF (correct chain order:
        # A=target, B=binder). BoltzGen raw CIFs have reversed chains and would
        # break the geometric site filter.
        if True:  # always overwrite with Boltz-2 CIF if available
            cifs = sorted(design_dir.glob("*model*.cif"))
            if not cifs:
                cifs = sorted(design_dir.glob("boltz_results_*/predictions/*model*.cif"))
            if not cifs:
                cifs = sorted(design_dir.glob("boltz_results_*/predictions/*/*model*.cif"))
            if cifs:
                d["complex_cif"] = str(cifs[0])

        if "boltz_iptm" in d:
            continue  # scores already loaded from CSV, only needed CIF path

        # Find confidence JSON — batch mode has it directly, sequential has nested
        conf_jsons = sorted(design_dir.glob("confidence_*.json"))
        if not conf_jsons:
            conf_jsons = sorted(design_dir.glob("boltz_results_*/confidence_*.json"))
        if not conf_jsons:
            conf_jsons = sorted(design_dir.glob("boltz_results_*/predictions/*/confidence_*.json"))
        if not conf_jsons:
            continue

        import json
        try:
            with open(conf_jsons[0]) as f:
                conf = json.load(f)
        except Exception:
            continue

        d["boltz_iptm"] = conf.get("iptm", float("nan"))
        d["boltz_ptm"] = conf.get("ptm", float("nan"))
        d["boltz_confidence"] = conf.get("confidence_score", float("nan"))
        d["boltz_protein_iptm"] = conf.get("protein_iptm", float("nan"))
        d["boltz_complex_plddt"] = conf.get("complex_plddt", float("nan"))
        d["boltz_iplddt"] = conf.get("complex_iplddt", float("nan"))
        d["boltz_complex_pde"] = conf.get("complex_pde", float("nan"))
        d["boltz_complex_ipde"] = conf.get("complex_ipde", float("nan"))

        # Binder pLDDT from per-token array
        plddt_npz = sorted(design_dir.glob("plddt_*.npz"))
        if not plddt_npz:
            plddt_npz = sorted(design_dir.glob("boltz_results_*/plddt_*.npz"))
        if not plddt_npz:
            plddt_npz = sorted(design_dir.glob("boltz_results_*/predictions/*/plddt_*.npz"))
        if plddt_npz:
            import numpy as np
            try:
                arr = np.load(plddt_npz[0])
                plddt_vals = arr[list(arr.keys())[0]]
                # Binder is typically the last chain tokens
                target_len = d.get("_target_len", 0)
                if target_len > 0 and len(plddt_vals) > target_len:
                    d["boltz_binder_plddt"] = float(np.mean(plddt_vals[target_len:])) * 100
                else:
                    d["boltz_binder_plddt"] = float(np.mean(plddt_vals)) * 100
            except Exception:
                pass

        # PAE from NPZ
        pae_npz = sorted(design_dir.glob("pae_*.npz"))
        if not pae_npz:
            pae_npz = sorted(design_dir.glob("boltz_results_*/pae_*.npz"))
        if not pae_npz:
            pae_npz = sorted(design_dir.glob("boltz_results_*/predictions/*/pae_*.npz"))
        if pae_npz:
            import numpy as np
            try:
                arr = np.load(pae_npz[0])
                pae = arr[list(arr.keys())[0]]
                target_len = d.get("_target_len", 0)
                binder_len = len(d.get("binder_sequence", ""))
                if target_len > 0 and pae.shape[0] >= target_len + binder_len:
                    interface_block = pae[:target_len, target_len:target_len + binder_len]
                    d["boltz_min_interface_pae"] = float(np.min(interface_block))
                    d["boltz_mean_interface_pae"] = float(np.mean(interface_block))

                    # Site-specific PAE
                    if site_row_indices:
                        valid_rows = [r for r in site_row_indices if r < target_len]
                        if valid_rows:
                            site_block = pae[valid_rows][:, target_len:target_len + binder_len]
                            d["boltz_site_min_pae"] = float(np.min(site_block))
                            d["boltz_site_mean_pae"] = float(np.mean(site_block))
            except Exception:
                pass

        # PDE from NPZ
        pde_npz = sorted(design_dir.glob("boltz_results_*/pde_*.npz"))
        if not pde_npz:
            pde_npz = sorted(design_dir.glob("boltz_results_*/predictions/*/pde_*.npz"))
        if pde_npz:
            import numpy as np
            try:
                arr = np.load(pde_npz[0])
                pde = arr[list(arr.keys())[0]]
                target_len = d.get("_target_len", 0)
                binder_len = len(d.get("binder_sequence", ""))
                if target_len > 0 and pde.shape[0] >= target_len + binder_len:
                    interface_block = pde[:target_len, target_len:target_len + binder_len]
                    d["boltz_min_interface_pde"] = float(np.min(interface_block))
                    d["boltz_mean_interface_pde"] = float(np.mean(interface_block))
            except Exception:
                pass

        # Complex CIF
        cifs = sorted(design_dir.glob("boltz_results_*/predictions/*model*.cif"))
        if not cifs:
            cifs = sorted(design_dir.glob("boltz_results_*/predictions/*/*model*.cif"))
        if cifs:
            d["complex_cif"] = str(cifs[0])

        loaded += 1

    return loaded


def load_existing_rosetta(designs, val_dir):
    """Check for existing Rosetta scores and populate rosetta_* fields."""
    results_json = Path(val_dir) / "rosetta" / "results.json"
    if not results_json.exists():
        # Also check scores.json (older naming convention)
        results_json = Path(val_dir) / "rosetta" / "scores.json"
    if not results_json.exists():
        return 0

    import json
    try:
        with open(results_json) as f:
            results = json.load(f)
    except Exception:
        return 0

    loaded = 0
    for d in designs:
        if "rosetta_dG" in d:
            continue  # already loaded from another results_dir
        design_id = d["design_id"]
        # Try prefixed ID first, then original
        if design_id not in results and "_orig_design_id" in d:
            design_id = d["_orig_design_id"]
        if design_id in results:
            r = results[design_id]
            if r is None:
                continue
            # Support both old format (dG, sc) and new format (rosetta_dG, rosetta_sc)
            d["rosetta_dG"] = r.get("rosetta_dG", r.get("dG", float("nan")))
            d["rosetta_sc"] = r.get("rosetta_sc", r.get("sc", float("nan")))
            d["rosetta_hbonds"] = r.get("rosetta_hbonds", r.get("hbonds", float("nan")))
            d["rosetta_bunsats"] = r.get("rosetta_bunsats", r.get("bunsats", float("nan")))
            d["rosetta_dsasa"] = r.get("rosetta_dsasa", r.get("dsasa", float("nan")))
            d["rosetta_dg_dsasa"] = r.get("rosetta_dg_dsasa", r.get("dg_dsasa", float("nan")))
            d["rosetta_packstat"] = r.get("rosetta_packstat", r.get("packstat", float("nan")))
            loaded += 1

    return loaded


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-validate and re-rank existing binder designs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target", required=True,
                        help="Target protein structure (PDB or CIF)")
    parser.add_argument("--site", required=True,
                        help='Binding site residues, e.g. "A:11-17,119-124"')
    parser.add_argument("--results_dir", default=None,
                        help="Comma-separated input directories from previous generate_binders3*.py "
                             "runs. Designs and validation results are read from here. "
                             "Multiple dirs are merged (deduplicated by sequence). "
                             "If omitted, --out_dir is used as both input and output.")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for rankings.csv, top_designs/, dashboard.png")
    parser.add_argument("--tools", default=None,
                        help=f"Comma-separated tools to collect (default: auto-detect). "
                             f"Available: {','.join(ALL_TOOLS)}")
    parser.add_argument("--rank_only", action="store_true",
                        help="Skip validation (ESMFold, Boltz, Rosetta) — only re-rank "
                             "using existing validation results")
    parser.add_argument("--skip_esmfold", action="store_true",
                        help="Skip ESMFold pre-filter (reuse existing results if available)")
    parser.add_argument("--skip_boltz", action="store_true",
                        help="Skip Boltz-2 validation (reuse existing results if available)")
    parser.add_argument("--skip_rosetta", action="store_true",
                        help="Skip Rosetta scoring (reuse existing results if available)")
    parser.add_argument("--esmfold_plddt_threshold", type=float, default=70.0,
                        help="ESMFold pre-filter threshold (default: 70)")
    parser.add_argument("--max_validate", type=int, default=20,
                        help="Max designs sent to Boltz validation per tool (default: 20)")
    parser.add_argument("--score_weights", default="0.4,0.5,0.1",
                        help='pLDDT,iPTM,dG weights for combined_score (default: "0.4,0.5,0.1")')
    parser.add_argument("--filter_interface_pae", type=float, default=None,
                        help="Drop designs with min interface pAE above threshold")
    parser.add_argument("--filter_site_pae", type=float, default=None,
                        help="Drop designs with site-specific mean pAE above threshold "
                             "(detects off-site binding)")
    parser.add_argument("--max_site_dist", type=float, default=0.0,
                        help="Site contact filter: distance threshold (Angstrom) for "
                             "counting a site residue as 'contacted' by binder heavy atoms. "
                             "0=disable filter (default). Typical value: 5.0-8.0")
    parser.add_argument("--min_site_fraction", type=float, default=0.0,
                        help="Minimum fraction of site residues that must be contacted "
                             "(within --max_site_dist) for a design to pass. "
                             "0.0=only exclude designs with zero contacts (default). "
                             "0.3=require 30%% of site residues contacted.")
    parser.add_argument("--max_interface_ke", type=float, default=None,
                        help="Maximum K+E fraction at the binder-target interface (0-1). "
                             "Designs with interface_KE_fraction above this are excluded. "
                             "None=disable filter (default). Typical: 0.25")
    parser.add_argument("--reprediction", action="store_true",
                        help="Use Boltz-2 re-prediction scoring for all tools "
                             "(default: use native tool scores for BindCraft/BoltzGen/PXDesign)")
    parser.add_argument("--ss_bias", choices=["beta", "helix", "balanced"],
                        default="balanced",
                        help="SS composition bias for scoring and filtering. "
                             "'helix' boosts helical designs, 'beta' boosts sheet-rich designs, "
                             "'balanced' applies no SS preference (default: balanced)")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top designs to copy (default: 50)")
    parser.add_argument("--plip_top", type=int, default=10,
                        help="Number of top designs to run PLIP interaction analysis on "
                             "(default: 10). Set to 0 to disable.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print actions without executing")
    args = parser.parse_args()

    # ── Parse args ─────────────────────────────────────────────────────────
    target_path = Path(args.target).resolve()
    if not target_path.exists():
        sys.exit(f"ERROR: target file not found: {target_path}")

    try:
        chain_id, site_resnums = parse_site(args.site)
    except ValueError as e:
        sys.exit(f"ERROR --site: {e}")

    try:
        score_weights = parse_score_weights(args.score_weights)
    except (ValueError, TypeError) as e:
        sys.exit(f"ERROR --score_weights: {e}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    val_dir = out_dir / "validation"

    # Parse results_dir(s) — where to read designs + existing validation from
    if args.results_dir:
        results_dirs = [Path(d.strip()).resolve() for d in args.results_dir.split(",")]
        for rd in results_dirs:
            if not rd.exists():
                sys.exit(f"ERROR: results directory not found: {rd}")
    else:
        # Backward compat: read from out_dir itself
        results_dirs = [out_dir]

    # Auto-detect tools from subdirectories across all results_dirs
    if args.tools:
        tools = [t.strip() for t in args.tools.split(",")]
        for t in tools:
            if t not in ALL_TOOLS:
                sys.exit(f"ERROR: unknown tool '{t}'. Choose from {ALL_TOOLS}")
    else:
        tools = list(dict.fromkeys(
            t for rd in results_dirs for t in ALL_TOOLS if (rd / t).exists()
        ))
        if not tools:
            sys.exit(f"ERROR: no tool output directories found in {results_dirs}")

    if args.rank_only:
        args.skip_esmfold = True
        args.skip_boltz = True
        args.skip_rosetta = True

    # ── Header ─────────────────────────────────────────────────────────────
    log("=" * 62)
    log("rerank_binders.py — Re-validate & re-rank pipeline")
    log("=" * 62)
    log(f"Target:           {target_path}")
    log(f"Site:             chain={chain_id}, residues={site_resnums}")
    if len(results_dirs) == 1 and results_dirs[0] == out_dir:
        log(f"Input/Output:     {out_dir}  (in-place)")
    else:
        for i, rd in enumerate(results_dirs):
            log(f"Results dir [{i}]:  {rd}")
        log(f"Output dir:       {out_dir}")
    log(f"Tools:            {tools}")
    log(f"Score weights:    pLDDT={score_weights[0]}, iPTM={score_weights[1]}, dG={score_weights[2]}")
    log(f"Reprediction:     {args.reprediction}")
    log(f"SS bias:          {args.ss_bias}")
    log(f"Filter PAE:       {args.filter_interface_pae}")
    log(f"Max site dist:    {args.max_site_dist} Å")
    mode = "rank-only" if args.rank_only else "re-validate"
    if not args.rank_only:
        skips = []
        if args.skip_esmfold: skips.append("ESMFold")
        if args.skip_boltz:   skips.append("Boltz")
        if args.skip_rosetta: skips.append("Rosetta")
        if skips:
            mode += f" (skip: {', '.join(skips)})"
    log(f"Mode:             {mode}")
    if args.dry_run:
        log("*** DRY RUN — no commands will be executed ***")
    log("=" * 62)

    # ── Extract target sequence ────────────────────────────────────────────
    chain_info = get_chain_info(target_path)
    tchain = chain_info.get(chain_id, {})
    target_seq = tchain.get("sequence", "")
    target_len = tchain.get("length", 0)

    if not target_seq:
        sys.exit(f"ERROR: chain {chain_id} not found in {target_path}. "
                 f"Available chains: {list(chain_info)}")
    log(f"Target chain {chain_id}: {target_len} residues")

    # Convert PDB resnums → 0-based row indices for site-specific PAE
    target_residues = tchain.get("residues", [])
    site_row_indices = None
    if site_resnums and target_residues:
        resnum_to_idx = {r: i for i, r in enumerate(target_residues)}
        site_row_indices = [resnum_to_idx[r] for r in site_resnums if r in resnum_to_idx]
        if site_row_indices:
            log(f"  Site PAE indices: {len(site_row_indices)} residues mapped")

    # ── Collect designs from existing outputs ──────────────────────────────
    all_designs = []
    seen_seqs = {}  # sequence → design_id (for dedup across dirs)
    n_dupes = 0

    for rd in results_dirs:
        rd_label = rd.name if len(results_dirs) > 1 else ""
        for tool in tools:
            tool_dir = rd / tool
            if not tool_dir.exists():
                continue

            log(f"\n{'─' * 50}")
            header = f"COLLECT: {tool.upper()}"
            if rd_label:
                header += f"  ({rd_label})"
            log(header)
            log(f"{'─' * 50}")

            collector = COLLECTORS.get(tool)
            if not collector:
                log(f"  {tool}: no collector available")
                continue

            if tool == "pxdesign":
                designs = collector(tool_dir, chain_id=chain_id)
            else:
                designs = collector(tool_dir)

            # Deduplicate by sequence when merging multiple dirs
            if len(results_dirs) > 1:
                unique_designs = []
                for d in designs:
                    seq = d.get("binder_sequence", "")
                    if seq in seen_seqs:
                        n_dupes += 1
                        continue
                    seen_seqs[seq] = d.get("design_id", "")
                    unique_designs.append(d)
                designs = unique_designs

            # Prefix design_id with dir name when merging to avoid ID collisions
            # (e.g. two runs both have rfdiffusion_0001). Store original for
            # validation file lookups.
            if len(results_dirs) > 1:
                for d in designs:
                    d["_orig_design_id"] = d["design_id"]
                    d["_source_dir"] = str(rd)
                    d["design_id"] = f"{rd_label}__{d['design_id']}"

            # Tag each design with target_len for Boltz PAE parsing
            for d in designs:
                d["_target_len"] = target_len

            log(f"  {tool}: {len(designs)} designs collected")
            all_designs.extend(designs)

    if not all_designs:
        log("No designs found. Exiting.")
        sys.exit(1)

    if n_dupes:
        log(f"\n  Deduplication: {n_dupes} duplicate sequences removed across dirs")
    log(f"\nTotal designs collected: {len(all_designs)}")

    # ── Load existing validation results ───────────────────────────────────
    # Load from all results_dirs (each may have its own validation/)
    n_esm = n_boltz = n_rosetta = 0
    for rd in results_dirs:
        rd_val = rd / "validation"
        n_esm += load_existing_esmfold(all_designs, rd_val)
        n_boltz += load_existing_boltz(all_designs, rd_val, site_row_indices=site_row_indices)
        n_rosetta += load_existing_rosetta(all_designs, rd_val)
    # Also load from out_dir/validation if different from results_dirs
    if out_dir not in results_dirs and val_dir.exists():
        n_esm += load_existing_esmfold(all_designs, val_dir)
        n_boltz += load_existing_boltz(all_designs, val_dir, site_row_indices=site_row_indices)
        n_rosetta += load_existing_rosetta(all_designs, val_dir)

    if n_esm or n_boltz or n_rosetta:
        log(f"\nExisting validation results loaded:")
        if n_esm:     log(f"  ESMFold: {n_esm}/{len(all_designs)} designs")
        if n_boltz:   log(f"  Boltz-2: {n_boltz}/{len(all_designs)} designs")
        if n_rosetta: log(f"  Rosetta: {n_rosetta}/{len(all_designs)} designs")

    # ── Validation ─────────────────────────────────────────────────────────
    # Stage 1: ESMFold
    if not args.skip_esmfold:
        if args.reprediction:
            # --reprediction ON: ESMFold all designs missing plddt
            need_esm = [d for d in all_designs if "esmfold_plddt" not in d]
        else:
            # --reprediction OFF: only ESMFold backbone-only tools
            need_esm = [d for d in all_designs
                        if "esmfold_plddt" not in d
                        and _get_tool_from_design_id(d["design_id"]) in BACKBONE_ONLY_TOOLS]
            n_skip = sum(1 for d in all_designs
                         if "esmfold_plddt" not in d
                         and _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS)
            if n_skip:
                log(f"  ESMFold: skipping {n_skip} native-iPTM designs")
        if need_esm:
            log(f"\n{'─' * 50}")
            log(f"STEP: ESMFold pre-filter ({len(need_esm)} new designs)")
            log(f"{'─' * 50}")
            validate_esmfold(need_esm, val_dir, args.esmfold_plddt_threshold, args.dry_run)
        else:
            log(f"\nESMFold: all applicable designs already validated")

    # Apply ESMFold filter — only gate backbone-only tools; native-iPTM tools pass through
    if args.reprediction:
        passing = [d for d in all_designs
                   if d.get("esmfold_plddt", 0) >= args.esmfold_plddt_threshold]
    else:
        passing = [d for d in all_designs
                   if _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS
                   or d.get("esmfold_plddt", 0) >= args.esmfold_plddt_threshold]
    if not passing:
        log("WARNING: no designs passed ESMFold threshold. Using all designs.")
        passing = all_designs
    else:
        log(f"ESMFold filter: {len(passing)}/{len(all_designs)} passed")

    # Stage 2: Boltz-2
    if not args.skip_boltz:
        if args.reprediction:
            # Reprediction ON: validate all designs missing boltz_iptm
            need_boltz = [d for d in passing if "boltz_iptm" not in d]
        else:
            # Reprediction OFF: only backbone-only tools need Boltz-2
            need_boltz = [d for d in passing
                          if "boltz_iptm" not in d
                          and _get_tool_from_design_id(d["design_id"]) in BACKBONE_ONLY_TOOLS]
            n_skip = sum(1 for d in passing
                         if "boltz_iptm" not in d
                         and _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS)
            if n_skip:
                log(f"  Boltz-2: skipping {n_skip} native-iPTM designs")
        if need_boltz:
            log(f"\n{'─' * 50}")
            log(f"STEP: Boltz-2 validation ({len(need_boltz)} new designs)")
            log(f"{'─' * 50}")
            validate_boltz(
                need_boltz, target_seq, val_dir,
                max_per_tool=args.max_validate,
                target_len=target_len,
                dry_run=args.dry_run,
                site_resnums=site_resnums,
                target_residues=tchain.get("residues"))
        else:
            log(f"\nBoltz-2: all applicable designs already validated")

    # Geometric site proximity filter (between Boltz and Rosetta)
    if args.max_site_dist > 0:
        log(f"\n{'─' * 50}")
        log("STEP: Geometric site proximity filter")
        log(f"{'─' * 50}")
        geometric_site_filter(all_designs, site_resnums=site_resnums,
                              target_residues=tchain.get("residues"),
                              max_dist=args.max_site_dist,
                              min_site_fraction=args.min_site_fraction,
                              dry_run=args.dry_run)

    # Stage 3: Rosetta
    if not args.skip_rosetta:
        need_rosetta = [d for d in all_designs
                        if (d.get("complex_cif") or d.get("binder_pdb")) and "rosetta_dG" not in d]
        if need_rosetta:
            log(f"\n{'─' * 50}")
            log(f"STEP: Rosetta scoring ({len(need_rosetta)} new designs)")
            log(f"{'─' * 50}")
            rosetta_score_interfaces(need_rosetta, val_dir, dry_run=args.dry_run)
        else:
            log(f"\nRosetta: all designs with structures already scored")

    # Restore structure paths for SS computation (not stored in CSV)
    n_restored = 0
    search_dirs = list(results_dirs) + ([out_dir] if out_dir not in results_dirs else [])
    for rd in search_dirs:
        esmfold_dir = rd / "validation" / "esmfold"
        if esmfold_dir.exists():
            for d in all_designs:
                if "esmfold_pdb" not in d:
                    esm_pdb = esmfold_dir / f"{d['design_id']}.pdb"
                    if not esm_pdb.exists() and "_orig_design_id" in d:
                        esm_pdb = esmfold_dir / f"{d['_orig_design_id']}.pdb"
                    if esm_pdb.exists():
                        d["esmfold_pdb"] = str(esm_pdb)
                        n_restored += 1
    if n_restored:
        log(f"  Restored {n_restored} ESMFold PDB paths for SS computation")

    # SS fraction computation (needed for ss_bias scoring/filtering)
    log(f"\n{'─' * 50}")
    log("STEP: Secondary structure + composition analysis")
    log(f"{'─' * 50}")
    populate_binder_composition(all_designs)
    populate_interface_composition(all_designs)
    if not args.dry_run:
        populate_binder_ss(all_designs)
    else:
        log("  (SS skipped in dry-run mode)")

    # ── Rank & output ──────────────────────────────────────────────────────
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

    # Interface K+E filter — exclude designs with too many charged residues at interface
    if args.max_interface_ke is not None:
        before = sum(1 for d in all_ranked if d.get("rank") is not None)
        for d in all_ranked:
            if d.get("rank") is None:
                continue
            ike = d.get("interface_KE_fraction", float("nan"))
            if ike == ike and ike > args.max_interface_ke:
                d["combined_score"] = float("nan")
                d["rank"] = None
        # Re-rank remaining
        ranked = [d for d in all_ranked if d.get("rank") is not None]
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, d in enumerate(ranked):
            d["rank"] = i + 1
        after = len(ranked)
        if before - after > 0:
            log(f"  Interface KE filter: removed {before - after} designs "
                f"with interface K+E > {args.max_interface_ke:.0%}")

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
        title=f"Binder Re-rank: {target_path.stem}  site={args.site}")

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

    # ── Summary ────────────────────────────────────────────────────────────
    log(f"\n{'=' * 62}")
    log("DONE")
    log(f"{'=' * 62}")

    # Per-tool counts
    from collections import Counter
    tool_counts = Counter(_get_tool_from_design_id(d["design_id"]) for d in all_designs)
    for tool, count in sorted(tool_counts.items()):
        log(f"  {tool:12s}: {count} designs")

    validated = [d for d in all_ranked if d.get("rank") is not None]
    if validated:
        log(f"\nTop 5 designs:")
        for d in validated[:5]:
            min_pae = d.get("boltz_min_interface_pae", float("nan"))
            pae_str = f"{min_pae:.2f}" if min_pae == min_pae else "N/A"
            log(f"  rank {d['rank']:3d}  {d['design_id']:<22s}  "
                f"score={d['combined_score']:.3f}  "
                f"iPTM={d.get('boltz_iptm', 0):.3f}  "
                f"ESMFold={d.get('esmfold_plddt', 0):.1f}  "
                f"minPAE={pae_str}")

    # Per-tool K+E composition summary (total + interface)
    ke_by_tool = {}
    ike_by_tool = {}
    for d in all_ranked:
        tool = _get_tool_from_design_id(d.get("design_id", ""))
        ke = d.get("binder_KE_fraction", float("nan"))
        ike = d.get("interface_KE_fraction", float("nan"))
        if ke == ke:
            ke_by_tool.setdefault(tool, []).append(ke)
        if ike == ike:
            ike_by_tool.setdefault(tool, []).append(ike)
    if ke_by_tool:
        log(f"\nK+E composition by tool:")
        log(f"  {'Tool':22s}  {'Total KE':>10s}  {'Interface KE':>14s}  {'Surface KE':>12s}")
        for tool in sorted(ke_by_tool):
            vals = ke_by_tool[tool]
            n_high = sum(1 for v in vals if v > 0.25)
            ke_str = f"mean={sum(vals)/len(vals):.0%}"
            # Interface stats
            ivals = ike_by_tool.get(tool, [])
            if ivals:
                ike_str = f"mean={sum(ivals)/len(ivals):.0%}"
                # Surface = (total_KE*len - iface_KE*n_iface) / n_surface (approx from means)
                svals = []
                for d2 in all_ranked:
                    if _get_tool_from_design_id(d2.get("design_id","")) == tool:
                        s = d2.get("surface_KE_fraction", float("nan"))
                        if s == s:
                            svals.append(s)
                ske_str = f"mean={sum(svals)/len(svals):.0%}" if svals else "N/A"
            else:
                ike_str = "N/A"
                ske_str = "N/A"
            warning = f"  ** {n_high} above 25%" if n_high else ""
            log(f"  {tool:22s}  {ke_str:>10s}  {ike_str:>14s}  {ske_str:>12s}{warning}")

    log(f"\nOutput directory: {out_dir}")
    log(f"  rankings.csv   — {len(all_ranked)} total designs")
    log(f"  top_designs/   — top {min(args.top_n, len(validated))} structures")
    log(f"  dashboard.png  — summary plot")


if __name__ == "__main__":
    main()
