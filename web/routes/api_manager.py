"""Data Manager API — scan, import, archive, delete result directories."""

import asyncio
import csv
import io
import re
import shutil
import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel

from .. import database
from ..config import OUTPUTS_BASE, HOSTNAME

router = APIRouter(prefix="/api/manager", tags=["manager"])


def _safe_path(path_str: str) -> Path:
    """Resolve a path and ensure it's under OUTPUTS_BASE."""
    p = Path(path_str).resolve()
    base = OUTPUTS_BASE.resolve()
    if not str(p).startswith(str(base)):
        raise HTTPException(403, "Path is outside allowed directory")
    return p


def _extract_date(dirname: str) -> str:
    """Extract YYYY-MM-DD date prefix from a directory name."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", dirname)
    return m.group(1) if m else ""


def _extract_tools_from_csv(csv_path: Path, max_rows: int = 200) -> tuple[list[str], int]:
    """Extract unique tool names and row count from rankings.csv."""
    tools = set()
    count = 0
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                count += 1
                if i < max_rows:
                    tool = row.get("tool", "").strip()
                    if tool:
                        tools.add(tool)
                    else:
                        # Try to infer from design_id
                        did = row.get("design_id", "")
                        if did:
                            prefix = did.split("_")[0].lower()
                            if prefix in ("rf", "rfdiff", "rfdiffusion"):
                                tools.add("rfdiffusion")
                            elif prefix in ("bg", "boltzgen"):
                                tools.add("boltzgen")
                            elif prefix in ("bc", "bindcraft"):
                                tools.add("bindcraft")
                            elif prefix in ("px", "pxdesign"):
                                tools.add("pxdesign")
                            elif prefix in ("pn", "proteina"):
                                tools.add("proteina")
                            elif prefix in ("pc",):
                                tools.add("proteina_complexa")
    except Exception:
        pass
    return sorted(tools), count


def _parse_run_params(params_path: Path) -> dict:
    """Parse run_params.txt for metadata extraction."""
    meta = {}
    try:
        text = params_path.read_text()
        for line in text.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                val = val.strip()
                if key in ("target", "target_file"):
                    meta["target_file"] = val
                elif key == "site":
                    meta["site"] = val
                elif key == "length":
                    meta["length"] = val
                elif key == "mode":
                    meta["mode"] = val
                elif key == "tools":
                    meta["tools"] = val
                elif key in ("score_weights", "weights"):
                    meta["score_weights"] = val
                elif key == "ss_bias":
                    meta["ss_bias"] = val
    except Exception:
        pass
    return meta


# ---------- Scan for orphan directories ----------

@router.get("/scan")
async def scan_orphans():
    """Scan OUTPUTS_BASE for result directories not tracked in any DB."""
    known_dirs = await database.list_all_out_dirs()
    # Normalize known dirs
    known_normalized = set()
    for d in known_dirs:
        try:
            known_normalized.add(str(Path(d).resolve()))
        except Exception:
            known_normalized.add(d)

    orphans = []
    base = OUTPUTS_BASE.resolve()
    if not base.is_dir():
        return {"orphans": [], "scanned": 0}

    scanned = 0
    for project_dir in sorted(base.iterdir()):
        if not project_dir.is_dir():
            continue
        for result_dir in sorted(project_dir.iterdir()):
            if not result_dir.is_dir():
                continue
            scanned += 1
            resolved = str(result_dir.resolve())
            if resolved in known_normalized:
                continue

            rankings = result_dir / "rankings.csv"
            has_rankings = rankings.exists()
            tools, design_count = (
                _extract_tools_from_csv(rankings) if has_rankings else ([], 0)
            )

            orphans.append({
                "path": str(result_dir),
                "project": project_dir.name,
                "name": result_dir.name,
                "has_rankings": has_rankings,
                "date": _extract_date(result_dir.name),
                "tools": tools,
                "design_count": design_count,
            })

    return {"orphans": orphans, "scanned": scanned}


# ---------- Disk size ----------

@router.get("/disksize")
async def get_disk_size(path: str = Query(...)):
    """Get directory size via du -sb."""
    safe = _safe_path(path)
    if not safe.is_dir():
        raise HTTPException(404, "Directory not found")

    proc = await asyncio.create_subprocess_exec(
        "du", "-sb", str(safe),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        size_bytes = int(stdout.decode().split()[0])
    except (ValueError, IndexError):
        size_bytes = 0

    # Human-readable
    if size_bytes >= 1_073_741_824:
        size_human = f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        size_human = f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        size_human = f"{size_bytes / 1024:.1f} KB"
    else:
        size_human = f"{size_bytes} B"

    return {"path": str(safe), "size_bytes": size_bytes, "size_human": size_human}


# ---------- Import ----------

class ImportRequest(BaseModel):
    path: str
    project: str = ""
    name: str = ""
    target_file: str = ""
    site: str = ""


@router.post("/import")
async def import_dir(req: ImportRequest, request: Request):
    """Import a results directory into the local DB.

    If the directory is not already under OUTPUTS_BASE/{project}/, it is
    copied there first (the project subfolder is created if needed).
    """
    source = Path(req.path).resolve()
    if not source.is_dir():
        raise HTTPException(404, "Directory not found")

    rankings = source / "rankings.csv"
    if not rankings.exists():
        raise HTTPException(400, "No rankings.csv found in directory")

    # Extract metadata early (needed for project name)
    tools, design_count = _extract_tools_from_csv(rankings)
    run_params = _parse_run_params(source / "run_params.txt")

    # Determine project and name
    project = req.project or source.parent.name
    name = req.name or source.name

    # If source is not under OUTPUTS_BASE/{project}/, copy it there
    base = OUTPUTS_BASE.resolve()
    dest_project_dir = base / project
    dest_dir = dest_project_dir / source.name

    if str(source).startswith(str(base)):
        # Already under OUTPUTS_BASE — use as-is
        final_dir = source
        copied = False
    else:
        # Copy into OUTPUTS_BASE/{project}/ (run in thread to not block event loop)
        dest_project_dir.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            raise HTTPException(
                409, f"Destination already exists: {dest_dir}"
            )
        await asyncio.to_thread(shutil.copytree, str(source), str(dest_dir))
        final_dir = dest_dir
        copied = True

    # Check for duplicates across all machines (fast pre-check)
    if await database.job_exists_for_path(str(final_dir)):
        raise HTTPException(409, "This directory is already registered in the database")

    # Parse date from dir name
    date_str = _extract_date(final_dir.name)
    if date_str:
        try:
            from datetime import datetime
            created_at = datetime.strptime(date_str, "%Y-%m-%d").timestamp()
        except ValueError:
            created_at = final_dir.stat().st_mtime
    else:
        created_at = final_dir.stat().st_mtime

    # Get username from cookie
    username = request.cookies.get("proteaflow_user", "")

    # Create DB entry — unique index on out_dir prevents duplicates at DB level
    params = {
        "imported": True,
        "design_count": design_count,
        "source_path": str(source) if final_dir != source else "",
        **run_params,
    }

    # User-provided fields override run_params extraction
    final_target = req.target_file or run_params.get("target_file", "")
    final_site = req.site or run_params.get("site", "")

    try:
        job_id = await database.create_job(
            name=name,
            job_type="imported",
            gpu_id=-1,
            target_file=final_target,
            site=final_site,
            length=run_params.get("length", ""),
            tools=run_params.get("tools", ",".join(tools)),
            mode=run_params.get("mode", ""),
            params=params,
            out_dir=str(final_dir),
            log_file=str(final_dir / "run.log") if (final_dir / "run.log").exists() else "",
            username=username,
            project=project,
        )
    except Exception as e:
        # Unique constraint violation (race condition or cross-machine duplicate)
        if "UNIQUE" in str(e).upper() or isinstance(e.__cause__, sqlite3.IntegrityError):
            raise HTTPException(409, "This directory is already registered in the database")
        raise

    # Mark as completed and set timestamps to match directory date
    await database.update_job(
        job_id,
        status="completed",
        created_at=created_at,
        started_at=created_at,
        finished_at=created_at,
        exit_code=0,
    )

    # Add project to shared list
    database.add_shared_project(project)

    return {
        "job_id": job_id,
        "name": name,
        "project": project,
        "tools": tools,
        "design_count": design_count,
        "copied_to": str(final_dir) if copied else None,
    }


# ---------- Unarchive ----------

@router.post("/{job_id}/unarchive")
async def unarchive_job(job_id: str):
    """Unarchive a job (set archived=0)."""
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found in local DB (can only manage local entries)")
    await database.update_job(job_id, archived=0)
    return {"unarchived": True}


# ---------- Delete ----------

@router.delete("/{job_id}")
async def delete_job(job_id: str, delete_files: bool = Query(False)):
    """Delete a job entry from the local DB, optionally remove files."""
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found in local DB (can only manage local entries)")

    out_dir = job.get("out_dir", "")
    files_removed = False

    if delete_files and out_dir:
        try:
            safe = _safe_path(out_dir)
            if safe.is_dir():
                shutil.rmtree(str(safe))
                files_removed = True
        except (HTTPException, OSError):
            pass  # Files not removed, but DB deletion continues

    deleted = await database.delete_job(job_id)
    return {"deleted": deleted, "files_removed": files_removed}


# ---------- Bulk operations ----------

class BulkRequest(BaseModel):
    action: str  # archive, unarchive, delete
    job_ids: list[str]
    delete_files: bool = False


@router.post("/bulk")
async def bulk_action(req: BulkRequest):
    """Bulk archive/unarchive/delete jobs (local DB only)."""
    if req.action not in ("archive", "unarchive", "delete"):
        raise HTTPException(400, f"Unknown action: {req.action}")

    processed = 0
    errors = []

    for job_id in req.job_ids:
        job = await database.get_job(job_id)
        if not job:
            errors.append(f"{job_id}: not in local DB")
            continue

        try:
            if req.action == "archive":
                await database.update_job(job_id, archived=1)
            elif req.action == "unarchive":
                await database.update_job(job_id, archived=0)
            elif req.action == "delete":
                out_dir = job.get("out_dir", "")
                if req.delete_files and out_dir:
                    try:
                        safe = _safe_path(out_dir)
                        if safe.is_dir():
                            shutil.rmtree(str(safe))
                    except HTTPException:
                        errors.append(f"{job_id}: path outside allowed directory")
                        continue
                await database.delete_job(job_id)
            processed += 1
        except Exception as e:
            errors.append(f"{job_id}: {str(e)}")

    return {"processed": processed, "errors": errors}
