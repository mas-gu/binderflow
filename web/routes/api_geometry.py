"""Geometry API — lightweight 3D geometry data for Live Rerank visualization."""

import asyncio
import json
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import gemmi
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from scipy.spatial import ConvexHull

from .. import database
from .api_results import _parse_site_string

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])

_INTERFACE_DIST = 8.0  # Angstrom cutoff for interface residue detection
_MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kabsch_align(mobile: np.ndarray, target: np.ndarray) -> tuple:
    """Compute Kabsch alignment: rotation R and translation t such that
    R @ mobile + t ≈ target.  Uses matched CA positions.

    Both arrays must be (N, 3) with same N.
    Returns (R, t) where R is (3,3), t is (3,).
    """
    assert mobile.shape == target.shape
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    m = mobile - mobile_center
    t = target - target_center
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    translation = target_center - R @ mobile_center
    return R, translation


def _rotation_to_quaternion(R: np.ndarray) -> list[float]:
    """Convert 3x3 rotation matrix to [qw, qx, qy, qz]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def _extract_ca_positions(chain) -> np.ndarray:
    """Extract CA atom positions from a gemmi chain, returns (N, 3) array."""
    cas = []
    for residue in chain:
        ca = residue.find_atom("CA", "\0")
        if ca is not None:
            pos = ca.pos
            cas.append([pos.x, pos.y, pos.z])
    return np.array(cas, dtype=np.float64) if cas else np.empty((0, 3), dtype=np.float64)


def _read_structure(path: str) -> Optional[gemmi.Structure]:
    """Read a CIF or PDB file, return gemmi.Structure or None."""
    try:
        p = str(path)
        if p.endswith(".cif"):
            doc = gemmi.cif.read(p)
            st = gemmi.make_structure_from_block(doc.sole_block())
        else:
            st = gemmi.read_structure(p)
        return st
    except Exception:
        return None


def _identify_chains(structure: gemmi.Structure) -> tuple:
    """Identify binder and target chains. Returns (binder_chain, target_chain).

    Heuristic: the longer chain is the target. For Boltz CIFs, chain A is
    binder and chain B is target (binder is shorter).
    """
    model = structure[0]
    chains = list(model)
    if len(chains) < 2:
        return None, None

    # Sort by number of residues (polymer residues)
    chain_sizes = []
    for ch in chains:
        n_res = sum(1 for r in ch if r.find_atom("CA", "\0") is not None)
        chain_sizes.append((ch, n_res))

    chain_sizes.sort(key=lambda x: x[1])
    binder_chain = chain_sizes[0][0]
    target_chain = chain_sizes[-1][0]
    return binder_chain, target_chain


def _find_structure_file(design_id: str, tool: str, out_dir: Path) -> tuple[Optional[str], str]:
    """Locate complex structure file for a design. Returns (path, source)."""

    # Priority 1: Boltz-2 validated complex
    boltz_dir = out_dir / "validation" / "boltz" / "boltz_results_batch_inputs" / "predictions" / design_id
    if boltz_dir.is_dir():
        cifs = list(boltz_dir.glob("*.cif"))
        if cifs:
            return str(cifs[0]), "boltz"

    # Priority 2: top_designs per-tool directory
    if tool:
        tool_dir = out_dir / "top_designs" / tool
        if tool_dir.is_dir():
            for ext in ("*.cif", "*.pdb"):
                for f in tool_dir.glob(ext):
                    if design_id in f.stem:
                        return str(f), "top_designs"

    # Priority 2b: top_designs root
    top_dir = out_dir / "top_designs"
    if top_dir.is_dir():
        for ext in ("*.cif", "*.pdb"):
            for f in top_dir.glob(ext):
                if design_id in f.stem:
                    return str(f), "top_designs"

    # Priority 3: Tool-native outputs
    if tool == "bindcraft":
        bc_dir = out_dir / "bindcraft" / "bindcraft_output" / "Accepted" / "Ranked"
        if bc_dir.is_dir():
            for f in bc_dir.glob("*.pdb"):
                if design_id in f.stem:
                    return str(f), "native"

    elif tool == "boltzgen":
        # Try both directory names (varies by BoltzGen version/settings)
        for subdir_name in ("final_ranked_designs", "final_50_designs", "final_designs"):
            bg_dir = out_dir / "boltzgen" / "boltzgen_raw" / subdir_name
            if bg_dir.is_dir():
                design_subdir = bg_dir / design_id
                if design_subdir.is_dir():
                    cifs = list(design_subdir.glob("*.cif"))
                    if cifs:
                        return str(cifs[0]), "native"
                # Also try glob for files containing design_id
                for f in bg_dir.rglob(f"*{design_id}*.cif"):
                    return str(f), "native"

    elif tool == "pxdesign":
        px_dir = out_dir / "pxdesign" / "pxdesign_output"
        if px_dir.is_dir():
            for f in px_dir.rglob(f"{design_id}*.cif"):
                return str(f), "native"

    elif tool == "proteina_complexa":
        pc_dir = out_dir / "proteina_complexa" / "inference"
        if pc_dir.is_dir():
            for f in pc_dir.rglob(f"{design_id}*.pdb"):
                return str(f), "native"

    return None, ""


_TOOL_PREFIXES = {
    "rfdiffusion3": "rfdiffusion3",
    "rfdiffusion": "rfdiffusion",
    "boltzgen": "boltzgen",
    "bindcraft": "bindcraft",
    "pxdesign": "pxdesign",
    "proteina_complexa": "proteina_complexa",
    "proteina": "proteina",
}


def _infer_tool(design_id: str) -> str:
    """Infer tool name from design_id prefix (e.g. 'rfdiffusion_0042' -> 'rfdiffusion')."""
    did = design_id.lower()
    # Check longer prefixes first (proteina_complexa before proteina)
    for prefix in sorted(_TOOL_PREFIXES, key=len, reverse=True):
        if did.startswith(prefix):
            return _TOOL_PREFIXES[prefix]
    return "unknown"


def _sanitize_metrics(row: dict) -> dict:
    """Convert NaN/inf floats to None for JSON serialization."""
    clean = {}
    for k, v in row.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean


def _process_design(args: tuple) -> Optional[dict]:
    """Process a single design: find structure, extract raw CA coords.

    Called in a thread pool. Returns raw data for alignment in main function.
    """
    design_id, tool, out_dir, metrics = args

    struct_path, source = _find_structure_file(design_id, tool, out_dir)
    if struct_path is None:
        return None

    st = _read_structure(struct_path)
    if st is None:
        return None

    binder_chain, target_chain = _identify_chains(st)
    if binder_chain is None or target_chain is None:
        return None

    binder_cas = _extract_ca_positions(binder_chain)
    target_cas = _extract_ca_positions(target_chain)
    if len(binder_cas) < 3 or len(target_cas) < 3:
        return None

    return {
        "design_id": design_id,
        "tool": tool,
        "structure_source": source,
        "metrics": _sanitize_metrics(metrics),
        "_binder_cas": binder_cas,
        "_target_cas": target_cas,
    }


def _compute_design_geometry(binder_cas: np.ndarray, target_cas: np.ndarray) -> dict:
    """Compute ellipsoid, interface, etc. from already-aligned CA coordinates."""
    centroid = binder_cas.mean(axis=0)

    # PCA of binder CA positions
    centered = binder_cas - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    semi_axes = 1.0 * np.sqrt(np.maximum(eigenvalues, 0.0))
    R = eigenvectors
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    quat = _rotation_to_quaternion(R)

    # Interface residues: binder CAs within 8A of any target CA
    try:
        diff = binder_cas[:, np.newaxis, :] - target_cas[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        min_dists = dists.min(axis=1)
    except MemoryError:
        min_dists = np.array([
            np.sqrt(((target_cas - bca) ** 2).sum(axis=1)).min()
            for bca in binder_cas
        ])

    interface_mask = min_dists < _INTERFACE_DIST
    interface_centroid = binder_cas[interface_mask].mean(axis=0) if interface_mask.any() else centroid

    return {
        "centroid": centroid.tolist(),
        "ellipsoid_axes": semi_axes.tolist(),
        "ellipsoid_rotation": quat,
        "interface_centroid": interface_centroid.tolist(),
        "approach_vector": None,
    }


def _compute_geometry(out_dir: Path, site_str: str,
                      progress_callback=None,
                      target_pdb: Optional[str] = None) -> dict:
    """Main computation: extract geometry for all designs in a job.

    Args:
        progress_callback: optional callable(done, total, found) invoked
            after each design is processed.  Enables real-time progress.
        target_pdb: path to the original target PDB (for correct Boltz
            renumbering offset when extracting site residues).
    """

    csv_path = out_dir / "rankings.csv"
    if not csv_path.exists():
        raise FileNotFoundError("rankings.csv not found")

    df = pd.read_csv(csv_path)
    if "design_id" not in df.columns:
        raise ValueError("rankings.csv missing design_id column")

    # Determine tool column name
    tool_col = "tool" if "tool" in df.columns else None

    # Build work items
    work_items = []
    for _, row in df.iterrows():
        design_id = str(row["design_id"])
        tool = str(row[tool_col]) if tool_col and pd.notna(row[tool_col]) else ""
        # Infer tool from design_id prefix when tool column is empty/NaN
        if not tool or tool == "nan":
            tool = _infer_tool(design_id)
        metrics = row.to_dict()
        # Ensure tool is in metrics for JS filtering
        metrics["tool"] = tool
        work_items.append((design_id, tool, out_dir, metrics))

    total = len(work_items)

    # --- Phase 1: Extract raw CA coords in parallel (I/O bound) ---
    raw_results = []
    n_no_structure = 0
    done_count = 0

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_process_design, item): item for item in work_items}
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            if result is None:
                n_no_structure += 1
            else:
                raw_results.append(result)

            if progress_callback and done_count % 5 == 0:
                progress_callback(done_count, total, len(raw_results))

    if not raw_results:
        return {
            "target": {"hull_vertices": [], "hull_faces": [], "center": [0, 0, 0]},
            "site": {"residue_positions": [], "centroid": [0, 0, 0], "normal": [0, 0, 1]},
            "designs": [],
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "n_designs": 0,
            "n_no_structure": n_no_structure,
        }

    # --- Phase 2: Align all designs into a common reference frame ---
    # Use the first design's target chain as the reference coordinate system.
    ref_target_cas = raw_results[0]["_target_cas"]
    ref_n = len(ref_target_cas)

    results = []
    for raw in raw_results:
        mobile_target = raw["_target_cas"]
        binder_cas = raw["_binder_cas"]

        # Align this design's target onto the reference target via Kabsch
        # Only align if target chains have the same length (same protein)
        if len(mobile_target) == ref_n:
            try:
                R_align, t_align = _kabsch_align(mobile_target, ref_target_cas)
                aligned_binder = (R_align @ binder_cas.T).T + t_align
                aligned_target = ref_target_cas  # use reference directly
            except Exception:
                aligned_binder = binder_cas
                aligned_target = mobile_target
        else:
            # Different chain length — can't align, use raw coords
            aligned_binder = binder_cas
            aligned_target = mobile_target

        # Compute geometry from aligned coordinates
        geo = _compute_design_geometry(aligned_binder, aligned_target)
        geo["design_id"] = raw["design_id"]
        geo["tool"] = raw["tool"]
        geo["structure_source"] = raw["structure_source"]
        geo["metrics"] = raw["metrics"]
        results.append(geo)

    # --- Phase 3: Target hull from reference target ---
    target_data = {"hull_vertices": [], "hull_faces": [], "center": [0, 0, 0]}
    if len(ref_target_cas) >= 4:
        target_center = ref_target_cas.mean(axis=0)
        target_data["center"] = target_center.tolist()
        try:
            hull = ConvexHull(ref_target_cas)
            target_data["hull_vertices"] = ref_target_cas.tolist()
            target_data["hull_faces"] = hull.simplices.tolist()
        except Exception as exc:
            logger.warning("ConvexHull failed: %s", exc)

    # --- Phase 4: Site residues from the reference target ---
    site_residues = _parse_site_string(site_str)
    site_data = {"residue_positions": [], "centroid": [0, 0, 0], "normal": [0, 0, 1]}

    if site_residues:
        # Extract site residue positions from the reference design's structure.
        # This is in the reference coordinate frame (no alignment needed).
        site_positions = np.empty((0, 3), dtype=np.float64)
        ref_design = raw_results[0]
        ref_path, _ = _find_structure_file(
            ref_design["design_id"], ref_design["tool"], out_dir
        )
        if ref_path:
            site_positions = _extract_site_from_structure(
                ref_path, site_residues, original_target_pdb=target_pdb
            )

        # Fallback: try other structures
        if len(site_positions) == 0:
            site_positions = _extract_site_positions(
                out_dir, site_residues, raw_results
            )

        if len(site_positions) > 0:

            site_centroid = site_positions.mean(axis=0)
            site_data["residue_positions"] = site_positions.tolist()
            site_data["centroid"] = site_centroid.tolist()

            # Site normal: outward from target center
            target_center = np.array(target_data["center"])
            outward_vecs = site_positions - target_center
            norms = np.linalg.norm(outward_vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            unit_vecs = outward_vecs / norms
            normal = unit_vecs.mean(axis=0)
            n_len = np.linalg.norm(normal)
            if n_len > 1e-8:
                normal = normal / n_len
            site_data["normal"] = normal.tolist()

            # Compute approach vectors for each design
            for design in results:
                ifc = np.array(design["interface_centroid"])
                direction = site_centroid - ifc
                d_len = np.linalg.norm(direction)
                if d_len > 1e-8:
                    design["approach_vector"] = (direction / d_len).tolist()
                else:
                    design["approach_vector"] = [0.0, 0.0, 0.0]

    return {
        "target": target_data,
        "site": site_data,
        "designs": results,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_designs": len(results),
        "n_no_structure": n_no_structure,
    }


def _extract_site_positions(
    out_dir: Path,
    site_residues: list[tuple[str, int]],
    results: list[dict],
) -> np.ndarray:
    """Extract CA positions for site residues from a reference structure.

    For Boltz CIFs the target is chain B (renumbered from 1). For native PDBs
    the target keeps original numbering. We try both strategies.
    """
    # Strategy: find the first available structure file and extract
    # site residue CAs from the target chain.

    # Collect candidate structure files (prefer Boltz, then top_designs, then native)
    candidates = []
    boltz_pred_dir = out_dir / "validation" / "boltz" / "boltz_results_batch_inputs" / "predictions"
    if boltz_pred_dir.is_dir():
        for d in boltz_pred_dir.iterdir():
            if d.is_dir():
                cifs = list(d.glob("*.cif"))
                if cifs:
                    candidates.append(("boltz", str(cifs[0])))
                    break

    if not candidates:
        top_dir = out_dir / "top_designs"
        if top_dir.is_dir():
            for f in list(top_dir.glob("*.cif"))[:1] + list(top_dir.glob("*.pdb"))[:1]:
                candidates.append(("top_designs", str(f)))
                break

    if not candidates and results:
        # Use the first design's structure path
        d = results[0]
        path, _ = _find_structure_file(d["design_id"], d["tool"], out_dir)
        if path:
            candidates.append(("fallback", path))

    site_resnums = {r for _, r in site_residues}
    site_chain_id = site_residues[0][0] if site_residues else "A"

    for source, path in candidates:
        st = _read_structure(path)
        if st is None:
            continue

        _, target_chain = _identify_chains(st)
        if target_chain is None:
            continue

        # For Boltz CIFs: target is renumbered from 1. We need to find the
        # offset. Try direct match first, then offset-shifted match.
        positions = _match_site_residues(target_chain, site_resnums)
        if len(positions) > 0:
            return np.array(positions, dtype=np.float64)

        # Boltz renumbering: original numbering shifted so first residue = 1
        # Try to detect offset
        residue_nums = [r.seqid.num for r in target_chain]
        if residue_nums:
            # Find offset that maximizes matches
            min_site = min(site_resnums)
            min_chain = min(residue_nums)
            offset = min_site - min_chain
            shifted = {r - offset for r in site_resnums}
            positions = _match_site_residues(target_chain, shifted)
            if len(positions) > 0:
                return np.array(positions, dtype=np.float64)

    return np.empty((0, 3), dtype=np.float64)


def _get_boltz_offset(struct_path: str, site_chain: str,
                      original_target_pdb: Optional[str] = None) -> int:
    """Compute the Boltz renumbering offset.

    Boltz renumbers all chains from 1. The offset is:
        original_first_resnum - 1

    We determine original_first_resnum from the original target PDB.
    """
    if original_target_pdb and os.path.exists(original_target_pdb):
        try:
            orig_st = _read_structure(original_target_pdb)
            if orig_st:
                for ch in orig_st[0]:
                    if ch.name == site_chain:
                        res_nums = [r.seqid.num for r in ch
                                    if r.find_atom("CA", "\0") is not None]
                        if res_nums:
                            return min(res_nums) - 1
        except Exception:
            pass
    return 0


def _extract_site_from_structure(
    struct_path: str, site_residues: list[tuple[str, int]],
    original_target_pdb: Optional[str] = None,
) -> np.ndarray:
    """Extract site CA positions from a specific structure file.

    Handles Boltz renumbering using the original target PDB to determine
    the correct residue offset.
    """
    st = _read_structure(struct_path)
    if st is None:
        return np.empty((0, 3), dtype=np.float64)

    _, target_chain = _identify_chains(st)
    if target_chain is None:
        return np.empty((0, 3), dtype=np.float64)

    site_resnums = {r for _, r in site_residues}
    site_chain_id = site_residues[0][0] if site_residues else "A"

    # Try direct match first (works for native PDBs with original numbering)
    positions = _match_site_residues(target_chain, site_resnums)
    if positions:
        return np.array(positions, dtype=np.float64)

    # Compute offset from original PDB numbering
    offset = _get_boltz_offset(struct_path, site_chain_id, original_target_pdb)
    if offset > 0:
        shifted = {r - offset for r in site_resnums}
        positions = _match_site_residues(target_chain, shifted)
        if positions:
            return np.array(positions, dtype=np.float64)

    # Fallback: try chain's min resnum as offset base
    residue_nums = [r.seqid.num for r in target_chain
                    if r.find_atom("CA", "\0") is not None]
    if residue_nums:
        min_chain = min(residue_nums)
        # Try every plausible offset: first chain residue maps to each site residue's neighborhood
        min_site = min(site_resnums)
        fallback_offset = min_site - min_chain
        shifted = {r - fallback_offset for r in site_resnums}
        # Only accept if ALL shifted values are within chain range
        max_chain = max(residue_nums)
        if all(min_chain <= s <= max_chain for s in shifted):
            positions = _match_site_residues(target_chain, shifted)
            if positions:
                return np.array(positions, dtype=np.float64)

    return np.empty((0, 3), dtype=np.float64)


def _match_site_residues(chain, resnums: set) -> list[list[float]]:
    """Extract CA positions for residues whose seqid.num is in resnums."""
    positions = []
    for residue in chain:
        if residue.seqid.num in resnums:
            ca = residue.find_atom("CA", "\0")
            if ca is not None:
                positions.append([ca.pos.x, ca.pos.y, ca.pos.z])
    return positions


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/{job_id}/geometry")
async def get_geometry(job_id: str):
    """Compute and return lightweight 3D geometry for all designs in a job."""
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    out_dir = Path(job["out_dir"])
    csv_path = out_dir / "rankings.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Rankings not ready yet")

    cache_path = out_dir / "geometry_cache.json"

    # Return cached result if newer than rankings.csv
    if cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        csv_mtime = csv_path.stat().st_mtime
        if cache_mtime > csv_mtime:
            try:
                return json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass  # Corrupted cache, recompute

    site_str = job.get("site", "") or ""
    target_pdb = job.get("target_file", "") or ""

    # Run computation in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, _compute_geometry, out_dir, site_str, None, target_pdb
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("Geometry computation failed for job %s", job_id)
        raise HTTPException(500, f"Geometry computation failed: {exc}")

    # Cache result
    try:
        cache_path.write_text(json.dumps(result))
    except OSError as exc:
        logger.warning("Failed to write geometry cache: %s", exc)

    return result


@router.get("/{job_id}/geometry/stream")
async def stream_geometry(job_id: str):
    """SSE endpoint: streams progress events then the final geometry JSON.

    Events:
      data: {"type":"progress","done":42,"total":1300,"found":38}
      data: {"type":"done","data":{...full geometry...}}
      data: {"type":"cached"}   (if served from cache — data follows immediately)
      data: {"type":"error","message":"..."}
    """
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    out_dir = Path(job["out_dir"])
    csv_path = out_dir / "rankings.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Rankings not ready yet")

    cache_path = out_dir / "geometry_cache.json"
    site_str = job.get("site", "") or ""
    target_pdb = job.get("target_file", "") or ""

    async def event_generator():
        # Check cache first — if valid, skip computation entirely
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            csv_mtime = csv_path.stat().st_mtime
            if cache_mtime > csv_mtime:
                try:
                    # Validate cache is readable JSON
                    cache_path.read_bytes()[:10]
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
                except OSError:
                    pass

        # Count total designs for initial progress event
        try:
            df = pd.read_csv(csv_path)
            total = len(df)
        except Exception:
            total = 0

        yield f"data: {json.dumps({'type': 'progress', 'done': 0, 'total': total, 'found': 0})}\n\n"

        # Run computation in a thread with progress callback
        progress_queue = asyncio.Queue(maxsize=500)

        def on_progress(done, total_count, found):
            try:
                progress_queue.put_nowait((done, total_count, found))
            except asyncio.QueueFull:
                pass

        # Start computation in background thread
        loop = asyncio.get_event_loop()
        compute_task = loop.run_in_executor(
            None, _compute_geometry, out_dir, site_str, on_progress, target_pdb
        )

        # Stream progress events until computation completes
        while True:
            if compute_task.done():
                break
            # Drain all queued progress, emit the latest
            latest = None
            while not progress_queue.empty():
                try:
                    latest = progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if latest:
                done, total_count, found = latest
                yield f"data: {json.dumps({'type': 'progress', 'done': done, 'total': total_count, 'found': found})}\n\n"
            await asyncio.sleep(0.15)

        # Drain remaining
        latest = None
        while not progress_queue.empty():
            try:
                latest = progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if latest:
            done, total_count, found = latest
            yield f"data: {json.dumps({'type': 'progress', 'done': done, 'total': total_count, 'found': found})}\n\n"

        # Get result and cache it
        try:
            result = compute_task.result()
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        try:
            cache_path.write_text(json.dumps(result))
        except OSError:
            pass

        # Final progress showing completion
        n_total = result["n_designs"] + result["n_no_structure"]
        yield f"data: {json.dumps({'type': 'progress', 'done': n_total, 'total': n_total, 'found': result['n_designs']})}\n\n"

        # Signal done — JS will fetch the full data via GET /geometry (now cached)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
