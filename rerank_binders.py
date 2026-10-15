#!/usr/bin/env python3
# Copyright (c) 2026 Guillaume Mas
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
3. Runs Boltz-2 monomer → Boltz-2 complex → geometric filter → Rosetta → SS analysis → ranking
   (or subset with --rank_only)
4. Writes new rankings.csv, top_designs/, dashboard.png to out_dir

Existing validation results (Boltz-2 monomer CIFs, Boltz-2 complex CIFs) are reused if present.
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
    parse_bindcraft2_outputs,
    pdb_chain_seq, pick_binder_chain_by_identity,
    apply_post_rank_filter,
    write_status_json, classify_failure,
    # Validation + filtering
    validate_boltz_monomer, validate_boltz, rosetta_score_interfaces, compute_refolding_rmsd,
    compute_solubility,
    geometric_site_filter, populate_binder_composition, populate_interface_composition, populate_binder_ss,
    compute_binder_compactness,
    sanitize_target_pdb,
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


def collect_boltzgen(tool_dir, chain_id="A", target_path=None):
    """Collect designs from an existing boltzgen/ output directory.

    Binder-chain identification uses pick_binder_chain_by_identity(), the same
    helper run_boltzgen() uses. Until 2026-09-28 this collector still used the
    legacy shortest-chain heuristic, so re-ranking a run whose binder was LONGER
    than the target silently scored the target against itself — the 2026-06-17
    fix had only ever landed in generate_binders.py.
    """
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
    metrics_csv = next(raw_dir.glob("final_ranked_designs/all_designs_metrics.csv"), None)
    iptm_scores = _boltzgen_metrics_index(metrics_csv)

    bg_metrics = _load_boltzgen_metrics(raw_dir)

    tgt_seq = pdb_chain_seq(target_path, chain_id) if target_path else ""
    if target_path and not tgt_seq:
        log("  BoltzGen: could not read target sequence — falling back to "
            "shortest-chain binder pick (wrong if binder > target)")

    designs = []
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
        if seq in bg_metrics:
            d.update(bg_metrics[seq])
        designs.append(d)

    return designs


def collect_bindcraft(tool_dir, chain_id="A", target_path=None):
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
        # "first chain" returns the TARGET when chain B is absent. Prefer the
        # chain least like the known target; fall back to shortest, never to
        # "whatever came first".
        binder_info = chain_info.get("B")
        if not binder_info and chain_info:
            _tseq_bc = pdb_chain_seq(target_path, chain_id) if target_path else ""
            binder_info = (pick_binder_chain_by_identity(chain_info, _tseq_bc)
                           if _tseq_bc else None)
        if not binder_info and chain_info:
            binder_info = min(chain_info.values(), key=lambda v: v.get("length", 0))
        binder_info = binder_info or {}
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


def collect_bindcraft2(tool_dir, chain_id="A"):
    """
    Collect designs from an existing bindcraft2/ output directory.

    Thin wrapper on generate_binders.parse_bindcraft2_outputs — deliberately not
    a second copy of the parser. collect_bindcraft() above is a near-verbatim
    duplicate of run_bindcraft()'s parsing half, and the two have to be kept in
    step by hand; BC2 gets one implementation instead.
    """
    return parse_bindcraft2_outputs(Path(tool_dir) / "bindcraft2_output",
                                    target_chain=chain_id)


def collect_pxdesign(tool_dir, chain_id="A", target_path=None):
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
                # Pick the binder by SEQUENCE, not chain letter. PXDesign writes
                # chains "A0"/"B0" while chain_id is "A", so `cid != chain_id`
                # was True for A0 and this recorded the TARGET as the binder.
                # generate_binders got this fix; its twin here did not.
                _tseq = pdb_chain_seq(target_path, chain_id) if target_path else ""
                _picked = pick_binder_chain_by_identity(ci, _tseq) if _tseq else None
                if _picked:
                    seq = _picked.get("sequence", "")
                else:
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

        # Sentinel and native-entry skip MUST match run_proteina() in
        # generate_binders.py, or the same FASTA yields a different
        # design_id -> sequence mapping depending on which path read it, while the
        # validation cache is keyed on design_id. Until 2026-09-28 this used
        # `len(set(seq)) <= 2` (composition) where the generator uses the header
        # test `"sample=" not in header`, and 999.0 where the generator uses inf.
        best_seq, best_conf, best_seq_rec = None, float("inf"), float("nan")
        current_header = None
        for line in fasta_path.read_text().splitlines():
            if line.startswith(">"):
                current_header = line
            elif current_header and line.strip():
                seq_candidate = line.strip()
                # Skip the native/backbone entry: MPNN designs carry "sample=".
                if "sample=" not in current_header:
                    current_header = None
                    continue
                # ProteinMPNN score = negative log-likelihood (lower = better)
                m_conf = re.search(r"(?:overall_confidence|score)=([0-9.]+)", current_header)
                m_rec = re.search(r"(?:seq_rec|seq_recovery)=([0-9.]+)", current_header)
                conf = float(m_conf.group(1)) if m_conf else float("inf")
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


def _complexa_sample_id(p):
    """Metrics-CSV key for a Complexa structure.

    The CSVs key on the inference job-DIRECTORY name. Structures copied into
    {out_dir}/proteina_complexa/structures/ are flattened and renamed
    "{job_dir}__{file}.pdb", so the directory is gone from the path and
    `pdb.parent.name` is "structures" — recover the key from the filename
    prefix instead. Without this the fallback branch below (which reads those
    flattened copies whenever the shared inference tree has been cleaned)
    joins nothing and every pc_* column comes back blank.
    """
    if p.parent.name.startswith("job_"):
        return p.parent.name
    if "__" in p.name:
        _pref = p.name.split("__", 1)[0]
        if _pref.startswith("job_"):
            return _pref
    return p.stem


def collect_proteina_complexa(tool_dir, chain_id="A", target_path=None):
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
    # 1) the hashed tag generate_binders now writes (unique per absolute out_dir)
    try:
        from generate_binders import _complexa_run_name
        _tag = _complexa_run_name(tool_dir)
        run_dirs = sorted(inference_base.glob(f"*{_tag}*"))
    except Exception:
        run_dirs = []
    # 2) the legacy parent_name-qualified tag, for runs made before that
    if not run_dirs:
        run_dirs = sorted(inference_base.glob(f"*binder_target*{parent_name}_{tool_dir.name}*"))
    # 3) tool_dir.name alone is usually just "proteina_complexa", which matches
    #    EVERY Complexa campaign on this box. Accept it only when unambiguous —
    #    the same rule the bare glob below already follows.
    if not run_dirs:
        _loose = sorted(inference_base.glob(f"*binder_target*{tool_dir.name}*"))
        if len(_loose) == 1:
            run_dirs = _loose
        elif _loose:
            log(f"  Proteina Complexa: {len(_loose)} run dirs match the generic name "
                f"'{tool_dir.name}' — refusing to guess which belongs to this run "
                f"(would mix other campaigns' designs into this panel)", err=True)
    if not run_dirs:
        # Deliberately NOT a bare glob over the shared inference dir: it matches
        # every Complexa run ever done on this box and pulls other campaigns'
        # designs into this rerank. Only accept it when it is unambiguous.
        _candidates = sorted(inference_base.glob("*binder_target*"))
        if len(_candidates) == 1:
            run_dirs = _candidates
        elif _candidates:
            log(f"  Proteina Complexa: {len(_candidates)} run dirs under "
                f"{inference_base} and none matched this run — refusing to guess "
                f"(would mix other campaigns' designs into this panel)")

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
        for rew_csv in sorted(rd.glob("rewards_*.csv")):
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
                            # total_reward lives in rewards_*.csv — generate reads
                            # it here. rerank looked for it in *binder_results_*.csv,
                            # which has no such column, so pc_total_reward came back
                            # permanently empty from every rerank.
                            ("total_reward", "pc_total_reward"),
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

    # Monomer scRMSD — present in run_proteina_complexa(), absent here until
    # 2026-09-28, so pc_scrmsd_esmfold was simply missing from every reranked
    # Complexa panel.
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

    designs = []
    for i, pdb in enumerate(sample_pdbs):
        chain_info = extract_pdb_chain_info(pdb)
        # Binder chain: first chain whose id is NOT the target's, falling back to
        # shortest. Matches run_proteina_complexa(). The previous shortest-only
        # pick silently returned the TARGET whenever the binder was longer than it
        # — and this collector did not even receive chain_id to compare against.
        # Pick by SEQUENCE IDENTITY to the known target, not by chain letter —
        # `cid != chain_id` returns the TARGET for any non-A target chain.
        binder_info = None
        _tseq_pc = pdb_chain_seq(target_path, chain_id) if target_path else ""
        if _tseq_pc:
            binder_info = pick_binder_chain_by_identity(chain_info, _tseq_pc)
        if not binder_info:
            for cid, info in sorted(chain_info.items()):
                if str(cid)[:1].upper() != str(chain_id)[:1].upper():
                    binder_info = info
                    break
        if not binder_info and chain_info:
            binder_info = min(chain_info.values(), key=lambda v: v["length"])
        if not binder_info:
            continue

        seq = binder_info.get("sequence", "")
        if not seq:
            continue

        sample_name = _complexa_sample_id(pdb)
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


def collect_rfdiffusion3(tool_dir):
    """Collect designs from an existing rfdiffusion3/ output directory.

    Mirrors run_rfdiffusion3()'s collection in generate_binders.py EXACTLY so
    design_ids (rfdiffusion3_NNNN, assigned by sorted cif.gz enumeration order)
    match the cached validation files (Boltz/Rosetta keyed by design_id).
    """
    import gzip
    import json as _json
    import gemmi

    tool_dir = Path(tool_dir)
    output_subdir = tool_dir / "rfd3_output"
    designs = []
    if not output_subdir.exists():
        return designs

    for cif_gz in sorted(output_subdir.glob("*.cif.gz")):
        try:
            with gzip.open(str(cif_gz), "rt") as f:
                doc = gemmi.cif.read_string(f.read())
            st = gemmi.make_structure_from_block(doc.sole_block())
            chains = list(st[0])

            meta_path = str(cif_gz).replace(".cif.gz", ".json")
            binder_seq = None

            if len(chains) >= 2:
                # Prefer the diffused_index_map: residues mapped from the original
                # PDB are the fixed TARGET, so the binder is the chain with fewest
                # mapped residues. Length-agnostic, unlike the shortest-chain
                # fallback, which picks the target whenever the binder is longer.
                # This collector's docstring claimed it mirrored the generator
                # "EXACTLY"; until 2026-09-28 it was shortest-chain only, so a
                # divergent pick also mis-keyed the design_id-based validation cache.
                binder_ch = None
                if Path(meta_path).exists():
                    try:
                        with open(meta_path) as mf:
                            # Keep the CHAIN LETTER: "A5" became a bare 5 and
                            # matched residue 5 of ANY chain, so the binder's own
                            # residue 5 counted as target-mapped.
                            _mapped = set()
                            for _o, _n in _json.load(mf).get("diffused_index_map", {}).items():
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
                with open(meta_path) as mf:
                    meta = _json.load(mf)
                mapped_resnums = set()
                for orig, new in meta.get("diffused_index_map", {}).items():
                    # "A5" -> ("A", 5); a bare 5 matches every chain.
                    m = re.match(r"([A-Za-z])(\d+)", new)
                    if m:
                        mapped_resnums.add((m.group(1), int(m.group(2))))
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
                "design_id":       design_id,
                "binder_sequence": binder_seq,
                "backbone_pdb":    str(cif_gz),
            })
        except Exception:
            continue

    return designs


COLLECTORS = {
    "rfdiffusion":       collect_rfdiffusion,
    "rfdiffusion3":      collect_rfdiffusion3,
    "boltzgen":          collect_boltzgen,
    "bindcraft":         collect_bindcraft,
    "bindcraft2":        collect_bindcraft2,
    "pxdesign":          collect_pxdesign,
    "proteina":          collect_proteina,
    "proteina_complexa": collect_proteina_complexa,
}


# ── Reuse existing validation results ────────────────────────────────────────

def _foreign_design(d, val_dir, restrict_to_source):
    """
    True when `d` came from a DIFFERENT results_dir than the one `val_dir` belongs to.

    Multi-dir merge (`--results_dir a,b`) prefixes design_id with the dir label but
    keeps `_orig_design_id` for file lookups — and independent runs reuse the SAME
    bare ids (both dirs hold `rfdiffusion_0000/`). main() then calls each loader over
    the FULL design list once per results_dir, so without this check a design from
    dir B happily resolves dir A's cached prediction. Scores were first-wins and the
    CIF last-wins, so a merged design could carry one run's iPTM and another run's
    structure.

    Only applies to the per-results_dir passes; out_dir's own cache is shared by
    design and must stay unrestricted.
    """
    if not restrict_to_source:
        return False
    src = d.get("_source_dir")
    if not src:
        return False          # single-dir run: ids were never prefixed
    try:
        return Path(src).resolve() != Path(val_dir).parent.resolve()
    except Exception:
        return False


def load_existing_esmfold(designs, val_dir, restrict_to_source=False):
    """Check for existing monomer validation results and populate esmfold_plddt.

    Checks both legacy ESMFold PDBs (validation/esmfold/) and new Boltz-2
    monomer CIFs (validation/boltz_monomer/). Also loads results from
    rankings.csv for designs that were tested but had pLDDT below threshold.
    """
    import numpy as np

    esm_dir = Path(val_dir) / "esmfold"
    monomer_dir = Path(val_dir) / "boltz_monomer"
    loaded = 0

    # Load failed results from rankings.csv (if exists)
    # These designs were tested but pLDDT was below threshold
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

    if not esm_dir.exists() and not monomer_dir.exists() and not failed_plddt:
        return 0

    for d in designs:
        if "esmfold_plddt" in d:
            continue  # already loaded from another results_dir
        if _foreign_design(d, val_dir, restrict_to_source):
            continue

        did = d["design_id"]
        orig_did = d.get("_orig_design_id", did)

        # Try Boltz-2 monomer results first (new format)
        pred_dir = monomer_dir / "boltz_results_batch_inputs" / "predictions" / did
        if not pred_dir.exists() and orig_did != did:
            pred_dir = monomer_dir / "boltz_results_batch_inputs" / "predictions" / orig_did
        if pred_dir.exists():
            plddt_files = sorted(pred_dir.glob("plddt_*.npz"))
            if plddt_files:
                try:
                    plddt_arr = np.load(plddt_files[0])["plddt"] * 100.0
                    d["esmfold_plddt"] = float(plddt_arr.mean())
                    cifs = sorted(pred_dir.glob("*.cif"))
                    if cifs:
                        d["esmfold_pdb"] = str(cifs[0])
                    loaded += 1
                    continue
                except Exception:
                    pass

        # Try legacy ESMFold PDB (old format)
        pdb_path = esm_dir / f"{did}.pdb" if esm_dir.exists() else None
        if pdb_path and not pdb_path.exists() and orig_did != did:
            pdb_path = esm_dir / f"{orig_did}.pdb"

        if pdb_path and pdb_path.exists():
            bfacs = []
            for line in pdb_path.read_text().splitlines():
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        bfacs.append(float(line[60:66]))
                    except (ValueError, IndexError):
                        pass
            if bfacs:
                plddt = float(np.mean(bfacs))
                plddt = plddt * 100.0 if plddt <= 1.0 else plddt
                d["esmfold_plddt"] = plddt
                d["esmfold_pdb"] = str(pdb_path)
                loaded += 1
                continue

        # Fall back to rankings.csv for failed designs
        if orig_did in failed_plddt:
            d["esmfold_plddt"] = failed_plddt[orig_did]
            loaded += 1
            continue
        if did in failed_plddt:
            d["esmfold_plddt"] = failed_plddt[did]
            loaded += 1
            continue

        # Design has no monomer result — leave esmfold_plddt unset so the
        # pipeline knows it needs to be run (don't assume 0.0 = failed)

    return loaded


def load_existing_boltz(designs, val_dir, site_row_indices=None,
                        restrict_to_source=False, n_tag="", c_tag="",
                        target_seq=None, site_resnums=None, holo_ligands=None):
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
    _n_key_mismatch = [0]
    # Batch mode output: validation/boltz/boltz_results_batch_inputs/predictions/{design_id}/
    batch_pred_dir = boltz_dir / "boltz_results_batch_inputs" / "predictions"

    for d in designs:
        if _foreign_design(d, val_dir, restrict_to_source):
            continue
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

        # generate_binders stamps `.boltz_input_key` next to each prediction — a
        # fingerprint of the target seq, the TAGGED binder seq, the site, the
        # ligands and force_site. rerank never read it, so re-ranking with a
        # different --site or different tags silently adopted scores computed
        # under different constraints. Warn rather than drop: --rank_only
        # explicitly means "use the cached scores".
        if target_seq:
            # Sidecar first (current layout), then the legacy in-prediction-dir
            # copy. The sidecar moved out of predictions/{id}/ because creating
            # that dir early makes boltz skip the design entirely.
            _ck = ""
            try:
                from generate_binders import _read_boltz_key as _rbk
                # design_dir is .../predictions/{design_id}
                _ck = _rbk(design_dir.parent.parent.parent, design_dir.name, "complex")
            except Exception:
                pass
            if not _ck:
                try:
                    _ck = (design_dir / ".boltz_input_key").read_text().strip()
                except (OSError, IOError):
                    _ck = ""
            if _ck:
                try:
                    from generate_binders import _boltz_input_key as _bik
                    _want = _bik(target_seq, d.get("binder_sequence", ""),
                                 n_tag, c_tag, site_resnums, False, 6.0, None)
                    if _ck != _want:
                        _n_key_mismatch[0] += 1
                except Exception:
                    pass

        # Always update complex_cif to Boltz-2 validation CIF (correct chain order:
        # A=target, B=binder). BoltzGen raw CIFs have reversed chains and would
        # break the geometric site filter.
        # Take the CIF only from the validation dir that also supplied the scores.
        # This overwrite used to be unconditional and sat ABOVE the `boltz_iptm`
        # guard, making scores first-wins and complex_cif last-wins — so a design
        # could end up scored from one run and rendered/filtered from another.
        if "boltz_iptm" not in d or d.get("_boltz_src") == str(val_dir):
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

        # Remember which validation dir these scores came from, so the CIF
        # overwrite above can keep structure and scores from the same run.
        d["_boltz_src"] = str(val_dir)
        d["boltz_iptm"] = conf.get("iptm", float("nan"))
        # HOLO mode: mirror generate_binders — the global iptm averages in the
        # binder<->ligand interfaces, so score on the protein-protein iptm.
        # Without this a rerank of a holo run re-scored every design on the
        # ligand-diluted value at weight 0.5-0.6.
        _pi = conf.get("protein_iptm", float("nan"))
        if holo_ligands and _pi == _pi:
            d["boltz_iptm"] = _pi
        d["boltz_ptm"] = conf.get("ptm", float("nan"))
        d["boltz_confidence"] = conf.get("confidence_score", float("nan"))
        d["boltz_protein_iptm"] = conf.get("protein_iptm", float("nan"))
        # complex_plddt / complex_iplddt are 0-1 in Boltz's JSON. generate_binders.py
        # stores them ×100 (0-100); this path did not, so the same column carried two
        # different scales depending on which script wrote the CSV — measured on disk:
        # generate-written files peak at 72-97, rerank-written ones at 0.88.
        # boltz_binder_plddt (the one compute_combined_score uses) was already ×100 in
        # both, so rankings were never affected — only cross-run comparison and the
        # browser's pDockQ, which derives from boltz_iplddt.
        _cplddt = conf.get("complex_plddt", float("nan"))
        _iplddt = conf.get("complex_iplddt", float("nan"))
        d["boltz_complex_plddt"] = _cplddt * 100.0 if _cplddt == _cplddt else _cplddt
        d["boltz_iplddt"] = _iplddt * 100.0 if _iplddt == _iplddt else _iplddt
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
                    # Cap at the binder end (see generate_binders): an open-ended
                    # slice averages in holo-ligand tokens and dilutes the term
                    # that carries w_plddt.
                    _blen_p = len(n_tag) + len(d.get("binder_sequence", "")) + len(c_tag)
                    _bend_p = target_len + _blen_p
                    _seg = (plddt_vals[target_len:_bend_p]
                            if _bend_p <= len(plddt_vals) else plddt_vals[target_len:])
                    d["boltz_binder_plddt"] = (float(np.mean(_seg)) * 100
                                               if len(_seg) else float("nan"))
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
                # Terminal tags are PART of the chain Boltz folded, so they occupy
                # PAE/PDE columns. Slicing on the untagged length read the wrong
                # sub-block — shifted by len(n_tag), short by len(c_tag) — so a
                # rerank produced different interface PAE (and a different
                # site-PAE score penalty) from the very prediction it was reading.
                # generate_binders.py has always included the tags here.
                binder_len = len(n_tag) + len(d.get("binder_sequence", "")) + len(c_tag)
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
        # Batch mode writes pde_*.npz DIRECTLY into the design dir (that is what
        # generate_binders.py reads). Only the two nested sequential-mode layouts
        # were globbed here, so every rerank of a batch run silently lost
        # interface PDE.
        pde_npz = sorted(design_dir.glob("pde_*.npz"))
        if not pde_npz:
            pde_npz = sorted(design_dir.glob("boltz_results_*/pde_*.npz"))
        if not pde_npz:
            pde_npz = sorted(design_dir.glob("boltz_results_*/predictions/*/pde_*.npz"))
        if pde_npz:
            import numpy as np
            try:
                arr = np.load(pde_npz[0])
                pde = arr[list(arr.keys())[0]]
                target_len = d.get("_target_len", 0)
                # Terminal tags are PART of the chain Boltz folded, so they occupy
                # PAE/PDE columns. Slicing on the untagged length read the wrong
                # sub-block — shifted by len(n_tag), short by len(c_tag) — so a
                # rerank produced different interface PAE (and a different
                # site-PAE score penalty) from the very prediction it was reading.
                # generate_binders.py has always included the tags here.
                binder_len = len(n_tag) + len(d.get("binder_sequence", "")) + len(c_tag)
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

    if _n_key_mismatch[0]:
        log(f"  Boltz cache: {_n_key_mismatch[0]} cached prediction(s) were made with "
            f"DIFFERENT inputs than this rerank asks for (--site / --n_tag / --c_tag / "
            f"--holo_ligands changed). Their scores describe a different constraint; "
            f"re-run with --reprediction to recompute them.", err=True)
    return loaded


def load_existing_rosetta(designs, val_dir, restrict_to_source=False):
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

    try:
        _results_mtime = results_json.stat().st_mtime
    except OSError:
        _results_mtime = float("inf")
    _n_stale_rosetta = [0]

    loaded = 0
    for d in designs:
        if "rosetta_dG" in d:
            continue  # already loaded from another results_dir
        if _foreign_design(d, val_dir, restrict_to_source):
            continue
        design_id = d["design_id"]
        # Try prefixed ID first, then original
        if design_id not in results and "_orig_design_id" in d:
            design_id = d["_orig_design_id"]
        if design_id in results:
            r = results[design_id]
            if r is None:
                continue
            # scores.json is keyed on design_id ALONE, with nothing recording
            # which structure was scored. If the complex has been re-predicted
            # since (a changed --site, terminal tags, or simply a fresh Boltz
            # run), the cached interface dG belongs to a DIFFERENT structure —
            # and dG is a scoring term. Use mtime as the fingerprint: a complex
            # newer than the scores file cannot have produced those scores.
            _struct = d.get("complex_cif") or d.get("binder_pdb")
            if _struct:
                try:
                    if Path(_struct).stat().st_mtime > _results_mtime + 60:
                        _n_stale_rosetta[0] += 1
                        continue
                except OSError:
                    pass
            # Support both old format (dG, sc) and new format (rosetta_dG, rosetta_sc)
            d["rosetta_dG"] = r.get("rosetta_dG", r.get("dG", float("nan")))
            d["rosetta_sc"] = r.get("rosetta_sc", r.get("sc", float("nan")))
            d["rosetta_hbonds"] = r.get("rosetta_hbonds", r.get("hbonds", float("nan")))
            d["rosetta_bunsats"] = r.get("rosetta_bunsats", r.get("bunsats", float("nan")))
            d["rosetta_dsasa"] = r.get("rosetta_dsasa", r.get("dsasa", float("nan")))
            d["rosetta_dg_dsasa"] = r.get("rosetta_dg_dsasa", r.get("dg_dsasa", float("nan")))
            d["rosetta_packstat"] = r.get("rosetta_packstat", r.get("packstat", float("nan")))
            # SAP (aggregation propensity) was computed and stored by the Rosetta
            # stage but never read back here, so every rerank blanked the column
            # that the browser's developability radar depends on.
            d["rosetta_sap"] = r.get("rosetta_sap", r.get("sap", float("nan")))
            loaded += 1

    if _n_stale_rosetta[0]:
        log(f"  Rosetta cache: ignored {_n_stale_rosetta[0]} score(s) whose complex "
            f"is newer than {results_json.name} — the structure was re-predicted "
            f"after those scores were written, so they describe a different pose.",
            err=True)
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
                        help="Skip validation (Boltz-2 monomer, Boltz-2 complex, Rosetta) — only re-rank "
                             "using existing validation results")
    parser.add_argument("--skip_esmfold", action="store_true",
                        help="Skip Boltz-2 monomer pre-filter (reuse existing results if available)")
    parser.add_argument("--skip_boltz", action="store_true",
                        help="Skip Boltz-2 validation (reuse existing results if available)")
    parser.add_argument("--skip_rosetta", action="store_true",
                        help="Skip Rosetta scoring (reuse existing results if available)")
    parser.add_argument("--rerun_rosetta", action="store_true",
                        help="Force re-run Rosetta scoring on all designs (updates SAP and "
                             "other metrics). Skips Boltz-2 monomer and Boltz-2 complex.")
    parser.add_argument("--esmfold_plddt_threshold", type=float, default=80.0,
                        help="Boltz-2 monomer pre-filter pLDDT threshold (default: 80)")
    parser.add_argument("--max_validate", type=int, default=20,
                        help="IGNORED — kept only so existing scripts/web forms do not break. "
                             "validate_boltz() batches ALL designs in one invocation "
                             "(its own docstring: 'max_per_tool is kept for API compatibility but ignored'). Was documented as 'max designs sent to Boltz validation per tool', which was never true.")
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
    parser.add_argument("--force_site", action="store_true",
                        help="Force the Boltz-2 pocket constraint to also contact the ligand "
                             "phosphates (holo mode). Matches generate_binders; without it a "
                             "--reprediction rerank re-predicts with weaker constraints than "
                             "the design run used.")
    parser.add_argument("--site_max_distance", type=float, default=6.0,
                        help="Max distance (A) for the Boltz-2 pocket constraint. "
                             "Matches generate_binders (default 6.0).")
    parser.add_argument("--holo_ligands", type=str, default="",
                        help="Comma-separated CCD codes present in the complex (e.g. GDP,ALF,MG). "
                             "Only affects SCORING here: with ligands present the global Boltz iPTM "
                             "averages in the binder<->ligand interfaces, so the protein-protein "
                             "iPTM is used instead — matching generate_binders. Without this flag a "
                             "rerank of a holo run silently re-scores on the ligand-diluted iPTM.")
    parser.add_argument("--max_rg_ratio", type=float, default=None,
                        help="Maximum binder Rg/(2.2*N^0.38). Rejects extended / single-helix "
                             "binders that refolding RMSD cannot catch. 1.25 is a good cut. "
                             "Default: disabled (metric still recorded as binder_rg_ratio).")
    parser.add_argument("--require_site_geometry", action="store_true",
                        help="Drop designs whose site geometry could not be measured "
                             "(site_geometry_status != 'measured') instead of ranking them.")
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
    parser.add_argument("--min_site_cos", type=float, default=None,
                        help="Minimum cosine angle between outward surface normal at site "
                             "and binder approach direction. 0.0=any direction from solvent side, "
                             "0.5=within 60 degrees of normal. Default: disabled.")
    parser.add_argument("--max_refolding_rmsd", type=float, default=None,
                        help="Maximum refolding RMSD (Angstrom) between Boltz-2 monomer binder-alone "
                             "and Boltz-2 complex binder-in-complex. Filters target-dependent binders. "
                             "Typical: 3.0-4.0 Å.")
    parser.add_argument("--max_interface_ke", type=float, default=None,
                        help="Maximum K+E fraction at the binder-target interface (0-1). "
                             "Designs with interface_KE_fraction above this are excluded. "
                             "None=disable filter (default). Typical: 0.25")
    parser.add_argument("--no_cys", action="store_true",
                        help="Exclude designs containing cysteine (unpaired Cys cause "
                             "aggregation in E.coli expression).")
    parser.add_argument("--max_aa_fraction", type=float, default=None,
                        help="Maximum fraction of any single amino acid in the binder sequence. "
                             "Catches poly-Ala/poly-Glu hallucinations. Typical: 0.3.")
    parser.add_argument("--min_sc", type=float, default=None,
                        help="Minimum Rosetta shape complementarity (0-1). "
                             "Natural interfaces ~0.65, designed aim >0.60. Default: disabled.")
    parser.add_argument("--reprediction", action="store_true",
                        help="Use Boltz-2 re-prediction scoring for all tools "
                             "(default: use native tool scores for BindCraft/BoltzGen/PXDesign)")
    parser.add_argument("--n_tag", type=str, default="",
                        help="Amino acid sequence to prepend to binder N-terminus for Boltz-2 reprediction "
                             "(e.g. MHHHHHH for His-tag). Only used during reprediction.")
    parser.add_argument("--c_tag", type=str, default="",
                        help="Amino acid sequence to append to binder C-terminus for Boltz-2 reprediction "
                             "(e.g. HHHHHH for C-terminal His-tag). Only used during reprediction.")
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

    raw_target_path = target_path
    clean_target = out_dir / f"{target_path.stem}_clean.pdb"
    sanitize_target_pdb(target_path, clean_target)
    target_path = clean_target

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

    # Validate and uppercase terminal tags
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    for tag_name in ("n_tag", "c_tag"):
        tag_val = getattr(args, tag_name).strip().upper()
        if tag_val and not set(tag_val).issubset(VALID_AA):
            bad = set(tag_val) - VALID_AA
            sys.exit(f"ERROR: --{tag_name} contains invalid amino acid characters: {bad}")
        setattr(args, tag_name, tag_val)

    if args.rank_only and args.reprediction:
        sys.exit("ERROR: --rank_only and --reprediction are contradictory. "
                 "--rank_only re-scores cached results with no GPU; --reprediction "
                 "throws the cache away and re-predicts from scratch. Together they "
                 "load almost nothing and produce an empty ranking. Pick one.")

    if args.rank_only:
        args.skip_esmfold = True
        args.skip_boltz = True
        args.skip_rosetta = True

    if args.rerun_rosetta:
        args.skip_esmfold = True
        args.skip_boltz = True
        args.skip_rosetta = False  # force Rosetta to run
        # Evaluated AFTER rank_only, so it wins — but the mode banner below still
        # printed "rank-only" and suppressed the skip list, claiming nothing would
        # run while Rosetta ran.
        args._mode_note = "rerun-rosetta (overrides --rank_only for the Rosetta stage)"

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
    if args.n_tag or args.c_tag:
        log(f"N-terminal tag:   {args.n_tag or '(none)'}")
        log(f"C-terminal tag:   {args.c_tag or '(none)'}")
    log(f"SS bias:          {args.ss_bias}")
    log(f"Filter PAE:       {args.filter_interface_pae}")
    log(f"Max site dist:    {args.max_site_dist} Å")
    # The skip list was suppressed whenever rank_only was set — so
    # `--rank_only --rerun_rosetta` logged "rank-only" while Rosetta ran, i.e. the
    # banner claimed nothing would execute. Always show what will actually run.
    mode = "rank-only" if args.rank_only else "re-validate"
    skips = []
    if args.skip_esmfold: skips.append("Boltz-2 monomer")
    if args.skip_boltz:   skips.append("Boltz")
    if args.skip_rosetta: skips.append("Rosetta")
    runs = [s for s, on in (("Boltz-2 monomer", not args.skip_esmfold),
                            ("Boltz", not args.skip_boltz),
                            ("Rosetta", not args.skip_rosetta)) if on]
    if skips:
        mode += f" (skip: {', '.join(skips)})"
    if runs:
        mode += f" (RUNS: {', '.join(runs)} — needs GPU)"
    if getattr(args, "_mode_note", None):
        mode += f" [{args._mode_note}]"
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

    # Same site guard as generate_binders.py. Without it, `--site "A:9999"` on a
    # chain numbered 18-132 exited 0 with ok=true and re-ranked every design
    # against a site that does not exist — the mismatch was recorded in
    # status.json but nothing forced anyone to look at it.
    _site_missing = []
    if site_resnums and tchain.get("residues"):
        _present = set(tchain["residues"])
        _site_missing = sorted(r for r in site_resnums if r not in _present)
        if _site_missing:
            log(f"  WARNING: {len(_site_missing)}/{len(site_resnums)} --site residue(s) "
                f"are NOT in chain {chain_id}: {_site_missing[:15]}"
                f"{' ...' if len(_site_missing) > 15 else ''}", err=True)
            if len(_site_missing) == len(site_resnums):
                sys.exit(f"ERROR: NONE of the --site residues exist in chain "
                         f"{chain_id}. Every site-based metric and filter would be "
                         f"meaningless. Target range: {min(_present)}-{max(_present)}.")

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
        # rd.name alone collides when two results_dirs share a basename
        # (e.g. /a/output and /b/output) — the very collision the prefix exists
        # to prevent. Disambiguate with a short hash of the absolute path.
        if len(results_dirs) > 1:
            import hashlib as _hl
            _h = _hl.sha256(str(Path(rd).resolve()).encode()).hexdigest()[:6]
            rd_label = f"{rd.name}_{_h}"
        else:
            rd_label = ""
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

            # Pass target context to every collector that accepts it, rather than
            # a hardcoded allow-list — that list is why collect_boltzgen and
            # collect_proteina_complexa could not do the chain-identity pick at all.
            import inspect as _inspect
            _params = _inspect.signature(collector).parameters
            _kw = {}
            if "chain_id" in _params:
                _kw["chain_id"] = chain_id
            if "target_path" in _params:
                _kw["target_path"] = args.target
            designs = collector(tool_dir, **_kw)

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
        log("No designs found. Exiting.", err=True)
        if not args.dry_run:
            try:
                write_status_json(out_dir / "status.json", args, {}, [], [],
                                  target_path=target_path, chain_id=chain_id,
                                  site_resnums=site_resnums,
                                  target_residues=(tchain or {}).get("residues"),
                                  script="rerank_binders.py")
            except Exception as _e:
                log(f"  could not write status.json: {_e}", err=True)
        sys.exit(1)

    if n_dupes:
        log(f"\n  Deduplication: {n_dupes} duplicate sequences removed across dirs")
    log(f"\nTotal designs collected: {len(all_designs)}")

    # ── Load existing validation results ───────────────────────────────────
    # Load from all results_dirs (each may have its own validation/)
    _holo = [x.strip() for x in str(getattr(args, "holo_ligands", "") or "").split(",") if x.strip()]
    if _holo:
        log(f"  Holo mode: {', '.join(_holo)} — scoring on protein-protein iPTM")
    n_esm = n_boltz = n_rosetta = 0
    for rd in results_dirs:
        rd_val = rd / "validation"
        if not args.reprediction:
            # rank_only: load cached scores from results_dir
            # restrict_to_source: only match designs that actually came from THIS
            # results_dir. Without it the merge silently cross-loads (see
            # _foreign_design).
            _restrict = len(results_dirs) > 1
            n_esm += load_existing_esmfold(all_designs, rd_val,
                                           restrict_to_source=_restrict)
            n_boltz += load_existing_boltz(all_designs, rd_val,
                                           site_row_indices=site_row_indices,
                                           restrict_to_source=_restrict,
                                           n_tag=args.n_tag, c_tag=args.c_tag,
                                           target_seq=(tchain or {}).get("sequence"),
                                           site_resnums=site_resnums,
                                           holo_ligands=_holo)
            n_rosetta += load_existing_rosetta(all_designs, rd_val,
                                               restrict_to_source=_restrict)

    # Load from out_dir/validation (own cache — used for both modes)
    if val_dir.exists():
        n_esm += load_existing_esmfold(all_designs, val_dir)
        n_boltz += load_existing_boltz(all_designs, val_dir,
                                       site_row_indices=site_row_indices,
                                       n_tag=args.n_tag, c_tag=args.c_tag,
                                       target_seq=(tchain or {}).get("sequence"),
                                       site_resnums=site_resnums,
                                       holo_ligands=_holo)
        n_rosetta += load_existing_rosetta(all_designs, val_dir)

    if n_esm or n_boltz or n_rosetta:
        log(f"\nExisting validation results loaded:")
        if n_esm:     log(f"  Boltz-2 monomer: {n_esm}/{len(all_designs)} designs")
        if n_boltz:   log(f"  Boltz-2: {n_boltz}/{len(all_designs)} designs")
        if n_rosetta: log(f"  Rosetta: {n_rosetta}/{len(all_designs)} designs")
    if args.reprediction:
        log(f"  (reprediction mode: only using own out_dir cache, ignoring results_dir scores)")

    # ── Validation ─────────────────────────────────────────────────────────
    # Stage 1: Boltz-2 monomer
    if not args.skip_esmfold:
        if args.reprediction:
            # --reprediction ON: Boltz-2 monomer all designs missing plddt
            need_esm = [d for d in all_designs if "esmfold_plddt" not in d]
        else:
            # --reprediction OFF: only Boltz-2 monomer for backbone-only tools
            need_esm = [d for d in all_designs
                        if "esmfold_plddt" not in d
                        and _get_tool_from_design_id(d["design_id"]) in BACKBONE_ONLY_TOOLS]
            n_skip = sum(1 for d in all_designs
                         if "esmfold_plddt" not in d
                         and _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS)
            if n_skip:
                log(f"  Boltz-2 monomer: skipping {n_skip} native-iPTM designs")
        if need_esm:
            log(f"\n{'─' * 50}")
            log(f"STEP: Boltz-2 monomer pre-filter ({len(need_esm)} new designs)")
            log(f"{'─' * 50}")
            validate_boltz_monomer(need_esm, val_dir, args.esmfold_plddt_threshold, args.dry_run,
                                  n_tag=args.n_tag, c_tag=args.c_tag)
        else:
            log(f"\nBoltz-2 monomer: all applicable designs already validated")

    # Apply Boltz-2 monomer filter — only gate backbone-only tools; native-iPTM tools pass through
    if args.reprediction:
        passing = [d for d in all_designs
                   if d.get("esmfold_plddt", 0) >= args.esmfold_plddt_threshold]
    else:
        passing = [d for d in all_designs
                   if _get_tool_from_design_id(d["design_id"]) in IPTM_NATIVE_TOOLS
                   or d.get("esmfold_plddt", 0) >= args.esmfold_plddt_threshold]
    if not passing:
        log("WARNING: no designs passed Boltz-2 monomer threshold. Using all designs.")
        passing = all_designs
    else:
        log(f"Boltz-2 monomer filter: {len(passing)}/{len(all_designs)} passed")

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
                target_residues=tchain.get("residues"),
                n_tag=args.n_tag, c_tag=args.c_tag,
                # These three shape the Boltz YAML (pocket constraint strength and
                # holo context). They existed only in generate, so a rerank with
                # --reprediction silently re-predicted WITHOUT the constraints the
                # design run used.
                force_site=args.force_site,
                site_max_distance=args.site_max_distance,
                holo_ligands=_holo or None)
        else:
            log(f"\nBoltz-2: all applicable designs already validated")

    # Geometric site measurement (between Boltz and Rosetta).
    # Measure whenever a site is known, not only when a filter flag is passed —
    # otherwise site_geometry_status is blank by default and "never measured" is
    # indistinguishable from "never requested". Filtering stays opt-in.
    need_geometric = bool(site_resnums)
    if need_geometric:
        # --min_site_fraction is meaningless at the 999 A fallback: every site
        # residue "contacts" the binder, so contact_frac is always 1.0 and the
        # filter silently does nothing. If the user asked for a fraction but gave
        # no distance, use the documented typical value and say so.
        if args.max_site_dist <= 0 and args.min_site_fraction > 0:
            args.max_site_dist = 8.0
            log(f"  --min_site_fraction {args.min_site_fraction} needs a contact "
                f"distance; --max_site_dist was unset so defaulting it to 8.0 A "
                f"(without this the fraction filter does nothing).", err=True)
        # Match generate_binders' 15 A default. The old 999 A fallback meant every
        # site residue "contacted" the binder, so site_contact_fraction was 1.0 for
        # everything and a --rank_only rerank RESURRECTED designs the design run had
        # already rejected at 15 A. 999 A is not a measurement.
        max_dist = args.max_site_dist if args.max_site_dist > 0 else 15.0
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

    # Compactness is cheap and always informative — measure it whether or not a
    # geometric site filter was requested.
    log(f"\n{'─' * 50}")
    log("STEP: Binder compactness (Rg ratio)")
    log(f"{'─' * 50}")
    compute_binder_compactness(all_designs,
                               target_residues=tchain.get("residues"),
                               dry_run=args.dry_run)

    # Stage 3: Rosetta
    if not args.skip_rosetta:
        if args.rerun_rosetta:
            # Clear existing Rosetta scores to force re-computation (e.g., to add SAP)
            rosetta_keys = ["rosetta_dG", "rosetta_sc", "rosetta_hbonds", "rosetta_bunsats",
                            "rosetta_dsasa", "rosetta_dg_dsasa", "rosetta_packstat", "rosetta_sap"]
            for d in all_designs:
                for k in rosetta_keys:
                    d.pop(k, None)
            # Also delete cached scores.json so Rosetta re-runs from scratch
            scores_json = val_dir / "rosetta" / "scores.json"
            if scores_json.exists():
                scores_json.unlink()
                log(f"  Cleared cached Rosetta scores for re-computation")

        need_rosetta = [d for d in all_designs
                        if (d.get("complex_cif") or d.get("binder_pdb")) and "rosetta_dG" not in d]
        if need_rosetta:
            log(f"\n{'─' * 50}")
            log(f"STEP: Rosetta scoring ({len(need_rosetta)} {'re-scored' if args.rerun_rosetta else 'new'} designs)")
            log(f"{'─' * 50}")
            rosetta_score_interfaces(need_rosetta, val_dir, dry_run=args.dry_run)
        else:
            log(f"\nRosetta: all designs with structures already scored")

    # Restore structure paths for SS computation (not stored in CSV)
    n_restored = 0
    search_dirs = list(results_dirs) + ([out_dir] if out_dir not in results_dirs else [])
    for rd in search_dirs:
        # Check Boltz-2 monomer CIFs first (new format), then legacy ESMFold PDBs
        monomer_pred_dir = rd / "validation" / "boltz_monomer" / "boltz_results_batch_inputs" / "predictions"
        esmfold_dir = rd / "validation" / "esmfold"
        # Same cross-dir hazard as the cache loaders: `orig_did` is identical
        # across independent runs, so without this a design from dir B picks up
        # dir A's monomer CIF — which then drives SS composition and refolding
        # RMSD. out_dir is shared by design, so it is never restricted.
        _restrict_here = len(results_dirs) > 1 and rd in results_dirs
        for d in all_designs:
            if "esmfold_pdb" in d:
                continue
            if _foreign_design(d, rd / "validation", _restrict_here):
                continue
            did = d["design_id"]
            orig_did = d.get("_orig_design_id", did)
            # Try Boltz monomer CIF
            for try_id in (did, orig_did):
                cifs = sorted((monomer_pred_dir / try_id).glob("*.cif")) if (monomer_pred_dir / try_id).exists() else []
                if cifs:
                    d["esmfold_pdb"] = str(cifs[0])
                    n_restored += 1
                    break
            else:
                # Try legacy ESMFold PDB
                if esmfold_dir.exists():
                    for try_id in (did, orig_did):
                        esm_pdb = esmfold_dir / f"{try_id}.pdb"
                        if esm_pdb.exists():
                            d["esmfold_pdb"] = str(esm_pdb)
                            n_restored += 1
                            break
    if n_restored:
        log(f"  Restored {n_restored} monomer structure paths for SS/RMSD computation")

    # SS fraction computation (needed for ss_bias scoring/filtering)
    log(f"\n{'─' * 50}")
    log("STEP: Secondary structure + composition analysis")
    log(f"{'─' * 50}")
    populate_binder_composition(all_designs)
    # Pass the target identity through: without it the chain pick falls back to
    # "longest chain is the target", which inverts for any tool whose binder is
    # longer than its target.
    populate_interface_composition(all_designs, target_chain_id=chain_id,
                                   target_residues=(tchain or {}).get("residues"),
                                   # generate threads this; rerank did not, so a
                                   # rerank always measured interface_KE at the
                                   # hardcoded 5.0 A whatever --interface_dist said.
                                   interface_dist=args.interface_dist)
    if not args.dry_run:
        populate_binder_ss(all_designs)
    else:
        log("  (SS skipped in dry-run mode)")

    # Refolding RMSD (Boltz-2 monomer vs Boltz-2 complex)
    log(f"\n{'─' * 50}")
    log("STEP: Refolding RMSD")
    log(f"{'─' * 50}")
    compute_refolding_rmsd(all_designs, dry_run=args.dry_run,
                           n_tag=args.n_tag, c_tag=args.c_tag)

    # Solubility prediction (NetSolP)
    log(f"\n{'─' * 50}")
    log("STEP: Solubility prediction (NetSolP)")
    log(f"{'─' * 50}")
    compute_solubility(all_designs, dry_run=args.dry_run)

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

    # ── Quality filters (applied first) ──────────────────────────────────

    # Refolding RMSD filter — exclude target-dependent binders
    if getattr(args, "max_refolding_rmsd", None) is not None:
        # Was a hand-rolled copy that printed nothing when it removed
        # zero designs and never reported how many had no measurement.
        apply_post_rank_filter(
            all_ranked, "refolding_rmsd",
            args.max_refolding_rmsd, higher_is_better=False,
            label="Refolding RMSD filter", fmt="{:.1f} A")

    # Cysteine filter
    if args.no_cys:
        before = sum(1 for d in all_ranked if d.get("rank") is not None)
        for d in all_ranked:
            if d.get("rank") is None:
                continue
            seq = d.get("binder_sequence", "")
            if "C" in seq:
                d["combined_score"] = float("nan")
                d["rank"] = None
        ranked = [d for d in all_ranked if d.get("rank") is not None]
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, d in enumerate(ranked):
            d["rank"] = i + 1
        after = len(ranked)
        if before - after > 0:
            log(f"  Cys filter: removed {before - after} designs containing cysteine")

    # AA composition filter
    if args.max_aa_fraction is not None:
        before = sum(1 for d in all_ranked if d.get("rank") is not None)
        for d in all_ranked:
            if d.get("rank") is None:
                continue
            seq = d.get("binder_sequence", "")
            if seq:
                max_frac = max(seq.count(aa) / len(seq) for aa in set(seq))
                if max_frac > args.max_aa_fraction:
                    d["combined_score"] = float("nan")
                    d["rank"] = None
        ranked = [d for d in all_ranked if d.get("rank") is not None]
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, d in enumerate(ranked):
            d["rank"] = i + 1
        after = len(ranked)
        if before - after > 0:
            log(f"  AA composition filter: removed {before - after} designs "
                f"with single AA > {args.max_aa_fraction:.0%}")

    # Shape complementarity filter
    if getattr(args, "min_sc", None) is not None:
        # Was a hand-rolled copy that printed nothing when it removed
        # zero designs and never reported how many had no measurement.
        apply_post_rank_filter(
            all_ranked, "rosetta_sc",
            args.min_sc, higher_is_better=True,
            label="Shape complementarity filter", fmt="{:.2f}")

    # Interface K+E filter
    if getattr(args, "max_interface_ke", None) is not None:
        # Was a hand-rolled copy that printed nothing when it removed
        # zero designs and never reported how many had no measurement.
        apply_post_rank_filter(
            all_ranked, "interface_KE_fraction",
            args.max_interface_ke, higher_is_better=False,
            label="Interface KE filter", fmt="{:.0%}")

    # ── Geometric site filters (applied last) ─────────────────────────────

    # Site interface fraction filter (SIF)
    if getattr(args, "min_site_interface_fraction", None) is not None:
        # Was a hand-rolled copy that printed nothing when it removed
        # zero designs and never reported how many had no measurement.
        apply_post_rank_filter(
            all_ranked, "site_interface_fraction",
            args.min_site_interface_fraction, higher_is_better=True,
            label="Site interface filter (SIF)", fmt="{:.0%}")

    # Compactness filter (opt-in)
    if getattr(args, "max_rg_ratio", None) is not None:
        # Was a hand-rolled copy that printed nothing when it removed
        # zero designs and never reported how many had no measurement.
        apply_post_rank_filter(
            all_ranked, "binder_rg_ratio",
            args.max_rg_ratio, higher_is_better=False,
            label="Compactness filter (Rg ratio)", fmt="{:.2f}")

    # Unmeasured site geometry (opt-in)
    if getattr(args, "require_site_geometry", False):
        before = sum(1 for d in all_ranked if d.get("rank") is not None)
        for d in all_ranked:
            if d.get("rank") is None:
                continue
            if d.get("site_geometry_status") != "measured":
                d["combined_score"] = float("nan")
                d["rank"] = None
        ranked = [d for d in all_ranked if d.get("rank") is not None]
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, d in enumerate(ranked):
            d["rank"] = i + 1
        if before - len(ranked) > 0:
            log(f"  Site geometry requirement: removed {before - len(ranked)} designs "
                f"whose site geometry was never measured")

    # Site centroid distance filter
    if args.max_site_centroid_dist is not None:
        centroid_key = f"site_centroid_dist_{args.centroid_atoms}"
        before = sum(1 for d in all_ranked if d.get("rank") is not None)
        for d in all_ranked:
            if d.get("rank") is None:
                continue
            cd = d.get(centroid_key, float("nan"))
            if cd == cd and cd > args.max_site_centroid_dist:
                d["combined_score"] = float("nan")
                d["rank"] = None
        ranked = [d for d in all_ranked if d.get("rank") is not None]
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, d in enumerate(ranked):
            d["rank"] = i + 1
        after = len(ranked)
        if before - after > 0:
            log(f"  Centroid distance filter ({args.centroid_atoms}): removed {before - after} designs "
                f"with {centroid_key} > {args.max_site_centroid_dist:.1f} Å")

    # Surface normal filter (cos angle)
    if getattr(args, "min_site_cos", None) is not None:
        # Was a hand-rolled copy that printed nothing when it removed
        # zero designs and never reported how many had no measurement.
        apply_post_rank_filter(
            all_ranked, "site_cos_angle",
            args.min_site_cos, higher_is_better=True,
            label="Surface-normal filter (site cos)", fmt="{:.2f}")

    write_rankings_csv(all_ranked, out_dir / "rankings.csv")

    # (Removed: a "restore nullified structure paths" loop that popped
    # _orig_complex_cif / _orig_binder_pdb. Nothing in the tree has ever
    # ASSIGNED those keys, so the loop was a no-op that implied a nulling step
    # which does not exist — misleading to anyone reading for why paths might
    # be missing.)

    copy_top_designs(all_ranked, out_dir / "top_designs",
                     target_pdb=target_path, n=args.top_n,
                     site_resnums=site_resnums,
                     target_residues=tchain.get("residues"),
                     cos_angle=args.min_site_cos)
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
                f"monomer_pLDDT={d.get('esmfold_plddt', 0):.1f}  "
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

    # Machine-readable outcome, same schema as generate_binders.py so one reader
    # handles both. rerank has no design tools, so `tool_results` records what
    # each COLLECTOR recovered from the existing run.
    _tool_results = {t: f"OK ({c} designs)" for t, c in sorted(tool_counts.items())}
    # Same reason as generate_binders.py: a dry run must not leave a status.json,
    # because the documented poll is "status.json exists => the run finished".
    # A --dry_run here previously wrote one saying ok=true.
    if args.dry_run:
        log("  (dry run: status.json intentionally not written)")
    else:
        write_status_json(
            out_dir / "status.json", args, _tool_results, all_ranked, validated,
            target_path=getattr(args, "target", None),
            chain_id=chain_id, site_resnums=site_resnums,
            target_residues=(tchain.get("residues") if isinstance(tchain, dict) else None),
            script="rerank_binders.py",
            extra={
                "rank_only": bool(getattr(args, "rank_only", False)),
                "results_dirs": [str(x) for x in results_dirs],
                "designs_collected": len(all_designs),
            })

    log(f"\nOutput directory: {out_dir}")
    log(f"  rankings.csv   — {len(all_ranked)} total designs")
    log(f"  top_designs/   — top {min(args.top_n, len(validated))} structures")
    log(f"  dashboard.png  — summary plot")
    log(f"  status.json    — machine-readable run outcome")

    # Same exit taxonomy as generate_binders.py: 0 clean, 3 nothing ranked.
    # (No 4 here — rerank runs no design tools, so there is no partial-tool case.)
    if args.dry_run:
        return 0
    if not validated:
        log("EXIT 3: no design could be ranked. Check status.json — usually every "
            "design was filtered out, or none had an interface score.", err=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
