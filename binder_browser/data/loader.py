"""Scan results directories and load rankings CSV files."""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from ..config import TOOL_PREFIXES
from .models import RunInfo


def identify_tool(design_id: str) -> str:
    """Identify the tool from a design_id string."""
    design_id_lower = design_id.lower()
    # A multi-dir rerank prefixes ids with the source directory
    # ("out_helix__rfdiffusion_0007"), which defeats every startswith() test
    # below. Strip it before matching.
    if "__" in design_id_lower:
        design_id_lower = design_id_lower.rsplit("__", 1)[-1]
    # Try longer prefixes first
    for prefix in sorted(TOOL_PREFIXES.keys(), key=len, reverse=True):
        if design_id_lower.startswith(prefix):
            return TOOL_PREFIXES[prefix]
    # Fallback: check if tool name is anywhere in design_id. LONGEST FIRST —
    # "rfdiffusion3_0013" contains "rfdiffusion", so with rfdiffusion3 missing
    # (or listed later) every RFdiffusion3 design was mislabelled "rfdiffusion".
    for tool_name in ["rfdiffusion3", "rfdiffusion", "boltzgen",
                      "bindcraft2", "bindcraft", "pxdesign",
                      "proteina_complexa", "proteina",
                      "pocketflow", "molcraft", "pocketxmol",
                      "pocket2mol", "decompdiff", "library"]:
        if tool_name in design_id_lower:
            return tool_name
    return "unknown"


def scan_results_dir(results_dir: Path) -> List[RunInfo]:
    """Scan a results directory for run folders containing rankings.csv.

    Looks for:
    1. Direct rankings.csv in results_dir
    2. Subdirectories with rankings.csv
    """
    runs = []

    # Check if the results_dir itself has a rankings.csv
    direct_csv = results_dir / "rankings.csv"
    if direct_csv.is_file():
        run = _make_run_info(results_dir.name, results_dir, direct_csv)
        if run:
            runs.append(run)

    # Check subdirectories (immediate children)
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "rankings.csv"
        if csv_path.is_file():
            run = _make_run_info(subdir.name, subdir, csv_path)
            if run:
                runs.append(run)
        # Also check reranks/ and revalidated_*/ subfolders
        for nested in sorted(subdir.iterdir()) if subdir.is_dir() else []:
            if not nested.is_dir():
                continue
            if nested.name.startswith("rerank") or nested.name.startswith("revalidat"):
                nested_csv = nested / "rankings.csv"
                if nested_csv.is_file():
                    run = _make_run_info(f"{subdir.name}/{nested.name}", nested, nested_csv)
                    if run:
                        runs.append(run)

    return runs


def _make_run_info(name: str, path: Path, csv_path: Path) -> Optional[RunInfo]:
    """Create a RunInfo from a rankings.csv path with summary stats."""
    try:
        df = pd.read_csv(csv_path, nrows=5)  # Quick peek
        df_full = pd.read_csv(csv_path)
    except Exception:
        return None

    # Add tool column
    if "tool" not in df_full.columns and "design_id" in df_full.columns:
        df_full["tool"] = df_full["design_id"].apply(identify_tool)

    n_designs = len(df_full)
    best_combined = pd.to_numeric(df_full["combined_score"], errors="coerce").max() if "combined_score" in df_full.columns else 0.0
    best_iptm = pd.to_numeric(df_full["boltz_iptm"], errors="coerce").max() if "boltz_iptm" in df_full.columns else 0.0
    if pd.isna(best_combined): best_combined = 0.0
    if pd.isna(best_iptm): best_iptm = 0.0
    tools = sorted([t for t in df_full["tool"].unique().tolist()
                     if isinstance(t, str)]) if "tool" in df_full.columns else []

    dashboard = path / "dashboard.png"
    if not dashboard.is_file():
        dashboard = None

    return RunInfo(
        name=name,
        path=path,
        rankings_csv=csv_path,
        dashboard_png=dashboard,
        n_designs=n_designs,
        best_combined=best_combined,
        best_iptm=best_iptm,
        tools=tools,
    )


def load_rankings(csv_path: Path) -> pd.DataFrame:
    """Load a full rankings CSV and add derived columns."""
    df = pd.read_csv(csv_path)

    # Derive the tool ONLY when the pipeline did not record one. This used to be
    # unconditional, overwriting the authoritative `tool` column from
    # rankings.csv — and identify_tool() maps every library source to the generic
    # "library", so a molecule run's 7 distinct libraries collapsed into one.
    if "design_id" in df.columns:
        if "tool" not in df.columns:
            df["tool"] = df["design_id"].apply(identify_tool)
        else:
            _missing = df["tool"].isna() | (df["tool"].astype(str).str.strip() == "")
            if _missing.any():
                df.loc[_missing, "tool"] = df.loc[_missing, "design_id"].apply(identify_tool)

    # Ensure numeric columns are numeric
    numeric_cols = [
        "rank", "combined_score", "boltz_iptm", "boltz_ptm",
        "boltz_complex_plddt", "boltz_binder_plddt", "boltz_site_mean_pae",
        "boltz_mean_interface_pae", "esmfold_plddt", "rosetta_dG",
        "rosetta_sc", "rosetta_hbonds", "rosetta_dsasa",
        "binder_helix_frac", "binder_sheet_frac", "binder_loop_frac",
        "bg_design_to_target_iptm", "bc_i_ptm", "bc_dg", "bc_binder_plddt",
        "bc2_i_ptm", "bc2_i_pdae", "bc2_i_pae", "bc2_plddt",
        "bc2_interface_buriedarea", "bc2_hotspot_contact_fraction",
        "px_iptm", "px_plddt", "pc_iptm", "pc_plddt", "pc_sc",
        "site_interface_fraction", "site_contact_fraction",
        "site_centroid_dist_CA", "site_centroid_dist_heavy", "site_cos_angle",
        "refolding_rmsd", "interface_KE_fraction", "rosetta_sap",
        "netsolp_solubility",
        # Molecule pipeline descriptors
        "qed", "sa_score", "vina_score", "mw", "logp", "tpsa",
        "num_heavy_atoms", "num_rotatable_bonds", "molar_refractivity", "fsp3",
        "ligand_efficiency", "pocket_fit",
        # Boltz-2 affinity rescoring (rerank_molecules.py --boltz_affinity)
        "boltz_affinity_log_ic50", "boltz_affinity_prob_binder",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute all derived columns at once to avoid fragmentation warning
    derived = {}

    # binder_length from sequence (compute if missing or all-null)
    if "binder_sequence" in df.columns:
        if "binder_length" in df.columns and df["binder_length"].isna().all():
            df = df.drop(columns=["binder_length"])
        if "binder_length" not in df.columns:
            derived["binder_length"] = df["binder_sequence"].apply(
            lambda x: len(str(x)) if pd.notna(x) and x != "" else np.nan)

    # pDockQ2 (PAE-based, Zhu et al. 2023, Bioinformatics 39(7):btad424)
    # x = <1/(1+(PAE/d0)^2)> * <pLDDT>_int,  d0=10 Å
    # pDockQ2 = 1.31 / (1 + exp(-0.075 * (x - 84.733))) + 0.005
    if "boltz_mean_interface_pae" in df.columns and "boltz_iplddt" in df.columns:
        pae = df["boltz_mean_interface_pae"].clip(lower=0.0)
        iplddt = df["boltz_iplddt"].clip(lower=0.0)
        # iplddt on 0-100 scale (matching the original pDockQ2 calibration).
        # Scale PER ROW, not on the column max: generate-written rows are 0-100
        # while rerank-written rows were 0-1 until 2026-09-29, so a merged or
        # historical file contains BOTH. A column-max test sees >1.5 and leaves
        # every 0-1 row unscaled, giving them a pDockQ computed from an iPLDDT of
        # ~0.9 instead of ~90.
        iplddt = iplddt.where(iplddt > 1.5, iplddt * 100.0)
        d0 = 10.0
        pae_factor = 1.0 / (1.0 + (pae / d0) ** 2)
        x = pae_factor * iplddt
        derived["pDockQ"] = 1.31 / (1 + np.exp(-0.075 * (x - 84.733))) + 0.005

    # Tier classification (top 7.5% = Tier 1, next 17.5% = Tier 2, rest = Tier 3)
    if "combined_score" in df.columns:
        valid = df["combined_score"].dropna()
        if len(valid) > 0:
            t1 = valid.quantile(0.925)
            t2 = valid.quantile(0.75)
            # A NaN combined_score means the design was never ranked (filtered
            # out, or missing a scoring term) — it is NOT "Tier 3", which reads
            # as "ranked, and poor". Give it no tier so the browser can show it
            # as unranked instead of burying it among genuinely weak designs.
            derived["tier"] = df["combined_score"].apply(
                lambda x: (1 if x >= t1 else (2 if x >= t2 else 3))
                          if pd.notna(x) else pd.NA)

    # Add all derived columns at once
    if derived:
        df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)

    return df
