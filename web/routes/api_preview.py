"""Pocket preview API.

Lets the launch UI stage a target file, validate the site syntax, and
render the extracted pocket before submitting the job. Two endpoints:

    POST /api/preview/stage-target  — multipart upload, returns a token
    POST /api/preview/pocket        — given token + site + pocket_dist,
                                      returns parsed site, pocket geometry,
                                      warnings, and the pocket PDB content.

Staged files live under ``SCRATCH_UPLOADS_DIR``. They are promoted into
the job's permanent uploads dir on submit (handled in api_jobs); orphaned
scratch entries are wiped on app startup.
"""

import secrets
import sys
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

# pocket_utils lives one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pocket_utils import (
    parse_site_multichain,
    site_pairs_chains,
    extract_pocket_pdb,
    compute_pocket_center,
    compute_bbox_size,
)

from ..config import SCRATCH_UPLOADS_DIR


router = APIRouter(prefix="/api/preview", tags=["preview"])


def _scratch_dir_for_token(token: str) -> Path:
    """Resolve scratch dir for a token, refusing path traversal."""
    if not token or "/" in token or ".." in token or "\\" in token:
        raise HTTPException(400, "Invalid token")
    return SCRATCH_UPLOADS_DIR / token


def resolve_staged_target(token: str) -> Path:
    """Locate the single staged target file inside a token's scratch dir.

    Raises HTTPException(404) if missing.
    """
    d = _scratch_dir_for_token(token)
    if not d.is_dir():
        raise HTTPException(404, "Staged target not found")
    files = [p for p in d.iterdir()
             if p.is_file() and p.suffix.lower() in (".pdb", ".cif")]
    if not files:
        raise HTTPException(404, "Staged target not found")
    return files[0]


def _get_chains_in_target(target_path: Path) -> list[str]:
    """Return chain IDs present in a PDB/CIF file."""
    try:
        import gemmi
        st = gemmi.read_structure(str(target_path))
        if len(st) == 0:
            return []
        return [chain.name for chain in st[0]]
    except Exception:
        # Fall back to a textual scan of ATOM records
        chains = []
        try:
            for line in target_path.read_text().splitlines():
                if line.startswith(("ATOM", "HETATM")) and len(line) >= 22:
                    cid = line[21]
                    if cid not in chains and cid != " ":
                        chains.append(cid)
                    if len(chains) > 26:
                        break
        except Exception:
            pass
        return chains


@router.post("/stage-target")
async def stage_target(
    request: Request,
    target: UploadFile = File(None),
    target_path: str = Form(""),
):
    """Stage a target file for preview/launch.

    Accepts either a multipart upload (``target``) or a server-side path
    (``target_path``). Returns a token that subsequent /pocket calls and
    the eventual /api/jobs submission reference.
    """
    SCRATCH_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(12)
    scratch = SCRATCH_UPLOADS_DIR / token
    scratch.mkdir(parents=True, exist_ok=True)

    # Validate inputs first so we don't leak an empty scratch dir on bad
    # requests. Any HTTPException raised below cleans up the scratch dir.
    import shutil
    try:
        if target and target.filename:
            suffix = Path(target.filename).suffix.lower()
            if suffix not in (".pdb", ".cif"):
                raise HTTPException(400, "Target must be .pdb or .cif")
            dest = scratch / Path(target.filename).name
            content = await target.read()
            if not content:
                raise HTTPException(400, "Empty target file")
            dest.write_bytes(content)
        elif target_path and target_path.strip():
            src = Path(target_path.strip())
            if not src.exists():
                raise HTTPException(400, f"Target path not found: {src}")
            if src.suffix.lower() not in (".pdb", ".cif"):
                raise HTTPException(400, "Target must be .pdb or .cif")
            # Symlink to keep the scratch lightweight; if symlink fails (e.g.
            # cross-filesystem permissions), fall back to copy.
            dest = scratch / src.name
            try:
                dest.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dest)
        else:
            raise HTTPException(400, "Provide a file upload or a server path")
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise

    chains = _get_chains_in_target(dest)
    return {
        "token": token,
        "filename": dest.name,
        "chains": chains,
    }


@router.post("/pocket")
async def preview_pocket(
    token: str = Form(...),
    site: str = Form(...),
    pocket_dist: float = Form(8.0),
):
    """Validate the site against a staged target and render the pocket.

    Returns parsed site_pairs, anchor counts per chain, pocket residue
    count, geometric center, bbox size, the bbox cap that will be applied
    by the pipeline, warnings, and the pocket PDB inline (text).
    """
    target_file = resolve_staged_target(token)

    # Parse — surface clear errors for the form
    try:
        site_pairs = parse_site_multichain(site)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    # Validate referenced chains exist
    chains_in_target = _get_chains_in_target(target_file)
    site_chains = site_pairs_chains(site_pairs)
    missing = [c for c in site_chains if c not in chains_in_target]
    if missing:
        return {
            "ok": False,
            "error": (f"Chain(s) {missing} not in target "
                      f"(available: {chains_in_target})"),
        }

    # Run extraction in a scratch subdir so we don't pollute the staging dir
    work = SCRATCH_UPLOADS_DIR / token / "_preview"
    work.mkdir(parents=True, exist_ok=True)
    pocket_pdb_path = work / "pocket.pdb"
    try:
        extract_pocket_pdb(
            str(target_file), site_pairs,
            distance_cutoff=float(pocket_dist),
            out_path=str(pocket_pdb_path),
        )
        center = compute_pocket_center(str(target_file), site_pairs)
        bbox_raw = compute_bbox_size(str(target_file), site_pairs)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    is_multichain = len(site_chains) > 1
    bbox_cap = 25.0 if is_multichain else 20.0
    bbox_capped = bbox_raw > bbox_cap
    bbox_size = min(bbox_raw, bbox_cap)

    # PocketXMol radius the pipeline will use
    pocketxmol_radius = max(15.0, round(bbox_size * 0.75, 1))

    pocket_pdb_text = pocket_pdb_path.read_text() if pocket_pdb_path.exists() else ""
    pocket_residue_count = sum(
        1 for ln in pocket_pdb_text.splitlines()
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA"
    )

    anchors_per_chain = {
        c: sum(1 for cc, _ in site_pairs if cc == c)
        for c in site_chains
    }

    warnings = []
    for c in site_chains:
        if anchors_per_chain[c] < 2:
            warnings.append(
                f"Chain {c} has only {anchors_per_chain[c]} anchor residue — "
                f"centroid may be biased toward the larger anchor set")
    if pocket_residue_count == 0:
        warnings.append("Pocket extraction produced 0 residues — check site")
    if bbox_capped:
        warnings.append(
            f"Bbox extent {bbox_raw:.1f} Å capped to {bbox_cap:.1f} Å "
            f"(pipeline cap for {'multi-chain' if is_multichain else 'single-chain'} site)")

    return {
        "ok": True,
        "token": token,
        "site_pairs": [[c, r] for c, r in site_pairs],
        "site_chains": site_chains,
        "is_multichain": is_multichain,
        "anchors_per_chain": anchors_per_chain,
        "pocket_residue_count": pocket_residue_count,
        "center_xyz": list(center),
        "bbox_size_raw": round(bbox_raw, 2),
        "bbox_size": round(bbox_size, 2),
        "bbox_capped": bbox_capped,
        "bbox_cap": bbox_cap,
        "pocketxmol_radius": pocketxmol_radius,
        "warnings": warnings,
        "pocket_pdb": pocket_pdb_text,
    }
