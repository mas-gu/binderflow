"""Results API — rankings, dashboard, structure files."""

import math
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import database
from binders_pipeline_env.binder_browser.data.loader import load_rankings

router = APIRouter(prefix="/api/results", tags=["results"])


def _parse_rankings(csv_path: Path) -> list[dict]:
    """Load rankings CSV via binder_browser loader (adds pDockQ, tier, tool)."""
    df = load_rankings(csv_path)
    # Convert to list of dicts with JSON-safe types (NaN → None)
    rows = []
    for _, row in df.iterrows():
        clean = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                clean[k] = None
            elif isinstance(v, float) and v == int(v) and k in ("rank", "binder_length", "site_n_contacted", "site_n_total"):
                clean[k] = int(v)
            else:
                clean[k] = v
        rows.append(clean)
    return rows


@router.get("/{job_id}/rankings")
async def get_rankings(job_id: str, tool: str = None, limit: int = None, include_unranked: bool = False):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    csv_path = Path(job["out_dir"]) / "rankings.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Rankings not ready yet")

    rows = _parse_rankings(csv_path)
    # Sort: ranked first (by rank), then unranked at the end
    rows.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 9999))
    # By default, exclude unranked (filtered-out) designs
    if not include_unranked:
        rows = [r for r in rows if r.get("rank") is not None]
    if tool:
        rows = [r for r in rows if r.get("tool") == tool]
    if limit:
        rows = rows[:limit]
    return rows


@router.get("/{job_id}/rankings/csv")
async def download_rankings(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    csv_path = Path(job["out_dir"]) / "rankings.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Rankings not ready")
    return FileResponse(csv_path, filename="rankings.csv", media_type="text/csv")


@router.get("/{job_id}/dashboard")
async def get_dashboard(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    img = Path(job["out_dir"]) / "dashboard.png"
    if not img.exists():
        raise HTTPException(404, "Dashboard not ready")
    return FileResponse(img, media_type="image/png")


@router.get("/{job_id}/structures")
async def list_structures(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    top_dir = Path(job["out_dir"]) / "top_designs"
    if not top_dir.exists():
        return []

    structures = []
    # Only list top-level files (ranked structures), not per-tool subdirectories
    for f in sorted(top_dir.glob("*.cif")) + sorted(top_dir.glob("*.pdb")):
        structures.append({
            "filename": f.name,
            "path": f.name,
            "tool": "global",
            "size_kb": round(f.stat().st_size / 1024, 1),
        })
    # If no top-level files, fall back to subdirectories
    if not structures:
        for f in sorted(top_dir.rglob("*.cif")) + sorted(top_dir.rglob("*.pdb")):
            if f.name.startswith("target_"):
                continue
            rel = f.relative_to(top_dir)
            structures.append({
                "filename": f.name,
                "path": str(rel),
                "tool": rel.parts[0] if len(rel.parts) > 1 else "global",
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return structures


@router.get("/{job_id}/structure/{path:path}")
async def get_structure(job_id: str, path: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    file_path = Path(job["out_dir"]) / "top_designs" / path
    if not file_path.exists() or ".." in path:
        raise HTTPException(404, "Structure file not found")

    media = "chemical/x-cif" if file_path.suffix == ".cif" else "chemical/x-pdb"
    return FileResponse(file_path, media_type=media)


@router.get("/{job_id}/plip")
async def get_plip_summary(job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    plip_dir = Path(job["out_dir"]) / "plip_analysis"
    summary = plip_dir / "PLIP_SUMMARY.txt"
    if summary.exists():
        return {"summary": summary.read_text()}

    if not plip_dir.exists():
        return {"summary": "PLIP analysis not available"}

    designs = []
    for d in sorted(plip_dir.iterdir()):
        if d.is_dir():
            designs.append(d.name)
    return {"designs": designs}


def _parse_site_string(site_str: str) -> list[tuple[str, int]]:
    """Parse site like 'A:373-374,379-384' into [(chain, resnum), ...]."""
    residues = []
    if not site_str:
        return residues
    for part in site_str.strip().split():
        m = __import__('re').match(r'^([A-Za-z]):(.+)$', part)
        if not m:
            continue
        chain, ranges = m.group(1), m.group(2)
        for rng in ranges.split(','):
            rng = rng.strip()
            dash = rng.split('-')
            if len(dash) == 2:
                for r in range(int(dash[0]), int(dash[1]) + 1):
                    residues.append((chain, r))
            elif rng.isdigit():
                residues.append((chain, int(rng)))
    return residues


@router.get("/{job_id}/site_residues")
async def get_site_residues(job_id: str):
    """Return Boltz-renumbered site residues for 3D viewer highlighting.

    Boltz CIFs renumber all chains from 1. The target is chain B (binder=A).
    We compute the offset from the original PDB numbering.
    """
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    site_str = job.get("site", "")
    if not site_str:
        return {"residues": [], "chain": "B"}

    site_residues = _parse_site_string(site_str)
    if not site_residues:
        return {"residues": [], "chain": "B"}

    # Find the target PDB to determine the offset
    out_dir = Path(job["out_dir"])
    target_file = job.get("target_file", "")

    # Try to read target PDB to get first residue number
    first_resnum = None
    try:
        import gemmi
        # Try target in top_designs first, then the original
        candidates = list(out_dir.glob("top_designs/target_*.pdb")) + [Path(target_file)]
        for pdb_path in candidates:
            if pdb_path.exists():
                st = gemmi.read_structure(str(pdb_path))
                target_chain = site_residues[0][0] if site_residues else 'A'
                for chain in st[0]:
                    if chain.name == target_chain:
                        residues_list = list(chain)
                        if residues_list:
                            first_resnum = residues_list[0].seqid.num
                        break
                if first_resnum is not None:
                    break
    except Exception:
        pass

    # Compute Boltz-renumbered residue numbers
    # Boltz renumbers from 1, so: boltz_num = original_num - first_resnum + 1
    offset = (first_resnum - 1) if first_resnum is not None else 0
    boltz_residues = [r - offset for _, r in site_residues]

    return {"residues": boltz_residues, "chain": "B", "offset": offset}
