#!/usr/bin/env python3
"""
mol_affinity.py — Boltz-2 affinity rescoring for small-molecule designs.

Runs Boltz-2 in affinity mode on the top-N molecules of a reranked run.
Pre-computes the target MSA once (re-used across all molecules) and issues
a single batched `boltz predict` call so the model loads once.

Two metrics are populated per molecule:
    boltz_affinity_log_ic50    — affinity_pred_value  (log10 IC50 in uM;
                                  use to rank analogs of a known binder)
    boltz_affinity_prob_binder — affinity_probability_binary  (0-1;
                                  use to triage screening hits)

Boltz-2 affinity constraints:
    - one ligand per YAML (cannot batch multiple ligands in one structure)
    - <=128 atoms per ligand (<=56 heavy atoms = training domain)
    - single-residue CCD or SMILES ligand

This module deliberately mirrors generate_binders.validate_boltz batch
pattern (pre-computed target MSA + single batch dir + cache via existing
JSON output).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_binders as _gb
from generate_binders import log, run_cmd, get_chain_info
from config_loader import cfg


METAL_ATOMS = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
               "Al", "Ga", "In", "Sn", "Pb", "Fe", "Cu", "Zn", "Mn", "Co",
               "Ni", "Cr", "Mo", "W", "Ag", "Au", "Hg", "Cd", "Pt", "Pd"}


def extract_target_sequence(pdb_path, chain_id):
    """Return the amino-acid sequence of chain `chain_id` in `pdb_path`."""
    info = get_chain_info(str(pdb_path))
    if chain_id not in info:
        raise ValueError(f"Chain {chain_id} not found in {pdb_path} "
                         f"(available: {sorted(info)})")
    return info[chain_id]["sequence"]


def extract_target_sequences(pdb_path, chain_ids):
    """Return ``{chain_id: sequence}`` for each chain in ``chain_ids``."""
    info = get_chain_info(str(pdb_path))
    missing = [c for c in chain_ids if c not in info]
    if missing:
        raise ValueError(f"Chain(s) {missing} not found in {pdb_path} "
                         f"(available: {sorted(info)})")
    return {c: info[c]["sequence"] for c in chain_ids}


def _smiles_passes_affinity_domain(smiles):
    """Filter out invalid / out-of-domain ligands.

    Returns (ok, reason). Reason is empty when ok=True.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return True, ""  # no rdkit, skip the filter

    if not smiles:
        return False, "empty SMILES"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "invalid SMILES"

    # Metal filter — Boltz-2 / ligand PDBQT prep both choke on metals
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in METAL_ATOMS:
            return False, f"contains metal atom {atom.GetSymbol()}"

    # Domain filter — Boltz-2 affinity head trained up to ~56 heavy atoms;
    # the model accepts up to 128 total atoms including Hs.
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > 80:  # well beyond training; results unreliable
        return False, f"{n_heavy} heavy atoms (>80, outside training domain)"

    return True, ""


def precompute_target_msa(target_seq, affinity_dir, chain_tag=None):
    """Compute target MSA once via a dummy Boltz-2 prediction.

    Saves to ``{affinity_dir}/target_msa[_{chain_tag}].csv``.
    For multi-chain targets call once per chain, passing the chain id as
    ``chain_tag``. ``chain_tag=None`` keeps the legacy single-chain layout.

    Returns the path, or ``None`` on failure (caller falls back to API MSA).
    """
    affinity_dir = Path(affinity_dir)
    affinity_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{chain_tag}" if chain_tag else ""
    msa_path = affinity_dir / f"target_msa{suffix}.csv"

    if msa_path.exists() and msa_path.stat().st_size > 0:
        log(f"  Affinity: reusing cached target MSA ({msa_path})")
        return msa_path

    # Back-compat: a legacy single-chain run wrote target_msa.csv. If we're
    # being asked for a per-chain file but only the legacy name exists,
    # adopt it for the first chain only — sequences must match for that
    # to be meaningful, so verify length below.
    legacy = affinity_dir / "target_msa.csv"
    if chain_tag and legacy.exists() and legacy.stat().st_size > 0 and not msa_path.exists():
        # Only adopt if the cached MSA matches our target chain length.
        try:
            first_line = legacy.read_text().splitlines()[1]  # header + first row
            cached_seq = first_line.split(",")[-1].strip()
            if len(cached_seq) == len(target_seq):
                import shutil
                shutil.copy2(legacy, msa_path)
                log(f"  Affinity: reusing legacy target MSA for chain {chain_tag}")
                return msa_path
        except Exception:
            pass

    # Check if a previous run nearby already produced an MSA we can reuse
    for parent in [affinity_dir.parent, affinity_dir.parent.parent]:
        existing = list(parent.rglob(f"target_msa{suffix}.csv"))
        existing += list(parent.rglob("boltz_results_batch_inputs/msa/*_0.csv"))
        existing = [p for p in existing if p.stat().st_size > 0]
        if existing:
            import shutil
            shutil.copy2(existing[0], msa_path)
            log(f"  Affinity: reusing target MSA from {existing[0]}")
            return msa_path

    tag_str = f" chain {chain_tag}" if chain_tag else ""
    log(f"  Affinity: computing target MSA{tag_str} (one-time)...")
    msa_yaml_dir = affinity_dir / f"target_msa_input{suffix}"
    msa_yaml_dir.mkdir(parents=True, exist_ok=True)
    msa_yaml = msa_yaml_dir / f"target_msa{suffix}.yaml"
    msa_yaml.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {target_seq}\n"
    )
    cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("boltz"),
        "boltz", "predict", str(msa_yaml),
        "--out_dir", str(affinity_dir / f"target_msa_run{suffix}"),
        "--use_msa_server",
        "--recycling_steps", "1",
        "--diffusion_samples", "1",
        "--sampling_steps", "10",
        "--no_kernels",
    ]
    try:
        run_cmd(cmd, timeout=600,
                extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        csvs = list((affinity_dir / f"target_msa_run{suffix}").rglob("*_0.csv"))
        if csvs:
            import shutil
            shutil.copy2(csvs[0], msa_path)
            log(f"  Affinity: target MSA computed -> {msa_path}")
            return msa_path
        log(f"  Affinity: WARNING — target MSA not found after predict")
    except Exception as e:
        log(f"  Affinity: WARNING — MSA precompute failed: {e}")
    return None


def build_affinity_yaml(design_id, target_chains, smiles, yaml_path):
    """Write one Boltz-2 YAML requesting affinity for ligand L.

    Parameters
    ----------
    target_chains : dict[str, dict]
        Mapping of chain_id -> {"sequence": str, "msa": Path or None}.
        For single-chain targets pass ``{"A": {"sequence": ..., "msa": ...}}``.
        For multi-chain targets pass one entry per chain — each gets its
        own ``protein:`` block with the matching id.
    """
    lines = ["version: 1", "sequences:"]
    for chain_id, info in target_chains.items():
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {info['sequence']}")
        if info.get("msa"):
            lines.append(f"      msa: {info['msa']}")
    lines.append("  - ligand:")
    lines.append("      id: L")
    lines.append(f"      smiles: '{smiles}'")
    lines.append("properties:")
    lines.append("  - affinity:")
    lines.append("      binder: L")
    Path(yaml_path).write_text("\n".join(lines) + "\n")


def _safe_design_id(design_id):
    """Boltz uses the YAML stem as the result dir name; sanitize unsafe chars."""
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in design_id)


def parse_affinity_json(json_path):
    """Parse a Boltz-2 affinity_*.json file.

    Uses the ensemble-averaged fields (`affinity_pred_value`,
    `affinity_probability_binary`) if present; otherwise falls back to the
    per-sample fields (`affinity_pred_value1` ... `affinity_pred_valueN`).
    """
    try:
        data = json.loads(Path(json_path).read_text())
    except Exception as e:
        log(f"  Affinity: parse failed for {json_path}: {e}")
        return float("nan"), float("nan")

    pred = data.get("affinity_pred_value")
    prob = data.get("affinity_probability_binary")

    if pred is None or prob is None:
        preds, probs = [], []
        for k, v in data.items():
            if k.startswith("affinity_pred_value") and k != "affinity_pred_value":
                try:
                    preds.append(float(v))
                except (TypeError, ValueError):
                    pass
            elif k.startswith("affinity_probability_binary") and k != "affinity_probability_binary":
                try:
                    probs.append(float(v))
                except (TypeError, ValueError):
                    pass
        if pred is None and preds:
            pred = sum(preds) / len(preds)
        if prob is None and probs:
            prob = sum(probs) / len(probs)

    try:
        pred = float(pred) if pred is not None else float("nan")
    except (TypeError, ValueError):
        pred = float("nan")
    try:
        prob = float(prob) if prob is not None else float("nan")
    except (TypeError, ValueError):
        prob = float("nan")
    return pred, prob


def run_boltz_affinity_batch(molecules, target_pdb, out_dir,
                             site_pairs=None, chain_id=None,
                             top_n=100, mw_correction=False,
                             sampling_steps=200, diffusion_samples=5,
                             boltz_devices=1):
    """Compute Boltz-2 affinity for the top-N molecules (by current rank).

    Either ``site_pairs`` (list of (chain, resnum) — multi-chain supported)
    **or** the legacy ``chain_id`` (single chain) must be provided.

    For multi-chain sites the Boltz target is built with one ``protein:``
    block per unique chain in the site, each with its own pre-computed
    MSA cached at ``affinity/target_msa_{chain}.csv``. Compute scales
    roughly with the number of chains.

    Mutates each selected molecule dict, adding:
        boltz_affinity_log_ic50    — ensemble-averaged log10(IC50 / uM)
        boltz_affinity_prob_binder — ensemble-averaged binder probability

    Molecules beyond top-N (or skipped/failed) keep NaN — the ranking
    formula treats NaN as a 0 contribution, so they are not silently
    penalised on other axes.
    """
    affinity_dir = Path(out_dir) / "affinity"
    affinity_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = affinity_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    results_root = affinity_dir / "boltz_results_batch" / "predictions"

    # ── Pick top-N molecules (by current combined_score) ──────────────────
    # Caller is expected to have run initial rank_molecules() already.
    ranked = sorted(molecules, key=lambda m: m.get("combined_score", 0.0),
                    reverse=True)
    candidates = ranked[:top_n]

    # ── Resolve target chains ─────────────────────────────────────────────
    if site_pairs:
        chains = []
        for c, _ in site_pairs:
            if c not in chains:
                chains.append(c)
    elif chain_id:
        chains = [chain_id]
    else:
        raise ValueError("run_boltz_affinity_batch requires site_pairs or chain_id")

    seqs = extract_target_sequences(target_pdb, chains)
    if len(chains) == 1:
        log(f"  Affinity: target chain {chains[0]}, "
            f"{len(seqs[chains[0]])} residues")
    else:
        log(f"  Affinity: multi-chain target chains {chains}: "
            + ", ".join(f"{c}={len(seqs[c])} res" for c in chains))

    # ── Pre-compute MSA once per chain ───────────────────────────────────
    target_chains_info = {}
    for c in chains:
        # Single-chain runs keep the legacy filename (no chain_tag) so
        # existing caches are reused untouched. Multi-chain runs get
        # per-chain files.
        tag = c if len(chains) > 1 else None
        msa_path = precompute_target_msa(seqs[c], affinity_dir, chain_tag=tag)
        target_chains_info[c] = {"sequence": seqs[c], "msa": msa_path}

    # ── Build per-molecule YAMLs, skipping already-done + out-of-domain ──
    needs_prediction = []
    id_to_mol = {}
    n_skip_domain = 0
    n_cached = 0

    for m in candidates:
        did = _safe_design_id(m.get("design_id", "")) or f"mol_{id(m)}"
        m["_affinity_design_id"] = did
        id_to_mol[did] = m

        smi = m.get("smiles", "")
        ok, reason = _smiles_passes_affinity_domain(smi)
        if not ok:
            m["boltz_affinity_log_ic50"] = float("nan")
            m["boltz_affinity_prob_binder"] = float("nan")
            m["_affinity_skipped_reason"] = reason
            n_skip_domain += 1
            continue

        # Existing result?
        existing = list((results_root / did).glob("affinity_*.json")) if results_root.exists() else []
        if existing:
            pred, prob = parse_affinity_json(existing[0])
            m["boltz_affinity_log_ic50"] = pred
            m["boltz_affinity_prob_binder"] = prob
            n_cached += 1
            continue

        yaml_path = batch_dir / f"{did}.yaml"
        build_affinity_yaml(did, target_chains_info, smi, yaml_path)
        needs_prediction.append(did)

    log(f"  Affinity: top-{top_n} candidates "
        f"-> {len(needs_prediction)} to predict, "
        f"{n_cached} cached, {n_skip_domain} skipped (domain/invalid)")

    if not needs_prediction:
        _fill_nan_for_unranked(molecules, candidates)
        return molecules

    # ── Detect free GPUs when boltz_devices > 1 ──────────────────────────
    actual_devices = 1
    extra_env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if boltz_devices > 1:
        try:
            gpu_check = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            free = []
            for ln in gpu_check.stdout.strip().split("\n"):
                idx, mem = ln.split(",")
                if int(mem.strip()) < 500:
                    free.append(idx.strip())
            actual_devices = min(boltz_devices, len(free))
            if actual_devices > 1:
                extra_env["CUDA_VISIBLE_DEVICES"] = ",".join(free[:actual_devices])
                log(f"  Affinity: {actual_devices} free GPUs "
                    f"(cuda:{','.join(free[:actual_devices])})")
            else:
                log(f"  Affinity: requested {boltz_devices} GPUs, "
                    f"only {len(free)} free -> using 1")
        except Exception:
            log(f"  Affinity: GPU detection failed, using 1 GPU")

    # ── Run Boltz-2 batch affinity ───────────────────────────────────────
    log(f"  Affinity: running Boltz-2 on {len(needs_prediction)} molecules "
        f"(sampling_steps={sampling_steps}, diffusion_samples={diffusion_samples}, "
        f"mw_correction={mw_correction})")

    cmd = [
        "conda", "run", "--no-capture-output", *cfg.conda_run_args("boltz"),
        "boltz", "predict", str(batch_dir),
        "--out_dir", str(affinity_dir),
        "--use_msa_server",
        "--sampling_steps_affinity", str(sampling_steps),
        "--diffusion_samples_affinity", str(diffusion_samples),
        "--preprocessing-threads", "48",
        "--no_kernels",
    ]
    if mw_correction:
        cmd.append("--affinity_mw_correction")
    if actual_devices > 1:
        cmd += ["--devices", str(actual_devices)]

    try:
        run_cmd(cmd, timeout=None, extra_env=extra_env)
    except Exception as e:
        log(f"  Affinity: Boltz-2 batch failed: {e}")
        # Fall through — whatever got written will be parsed; rest stay NaN.

    # ── Parse results ────────────────────────────────────────────────────
    n_parsed = 0
    for did in needs_prediction:
        m = id_to_mol[did]
        result_dir = results_root / did
        jsons = list(result_dir.glob("affinity_*.json")) if result_dir.exists() else []
        if not jsons:
            m["boltz_affinity_log_ic50"] = float("nan")
            m["boltz_affinity_prob_binder"] = float("nan")
            continue
        pred, prob = parse_affinity_json(jsons[0])
        m["boltz_affinity_log_ic50"] = pred
        m["boltz_affinity_prob_binder"] = prob
        if pred == pred and prob == prob:
            n_parsed += 1

    log(f"  Affinity: parsed {n_parsed}/{len(needs_prediction)} predictions")

    _fill_nan_for_unranked(molecules, candidates)
    _write_affinity_summary(molecules, affinity_dir / "affinity_summary.csv")
    _cleanup_internal_keys(molecules)
    return molecules


def _fill_nan_for_unranked(all_mols, rescored):
    """Molecules outside top-N get NaN (explicit) so columns exist uniformly."""
    rescored_ids = {id(m) for m in rescored}
    for m in all_mols:
        if id(m) in rescored_ids:
            continue
        m.setdefault("boltz_affinity_log_ic50", float("nan"))
        m.setdefault("boltz_affinity_prob_binder", float("nan"))


def _write_affinity_summary(molecules, csv_path):
    import csv
    rows = []
    for m in molecules:
        pred = m.get("boltz_affinity_log_ic50", float("nan"))
        prob = m.get("boltz_affinity_prob_binder", float("nan"))
        if pred != pred and prob != prob:
            continue
        rows.append({
            "design_id": m.get("design_id", ""),
            "smiles": m.get("smiles", ""),
            "boltz_affinity_log_ic50": pred,
            "boltz_affinity_prob_binder": prob,
        })
    if not rows:
        return
    rows.sort(key=lambda r: (
        -(r["boltz_affinity_prob_binder"] if r["boltz_affinity_prob_binder"] == r["boltz_affinity_prob_binder"] else -1)
    ))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _cleanup_internal_keys(molecules):
    for m in molecules:
        m.pop("_affinity_design_id", None)
        m.pop("_affinity_skipped_reason", None)
