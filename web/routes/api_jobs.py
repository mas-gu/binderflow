"""Job management API — create, list, cancel."""

import json
import re
import shutil
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Request

from .. import database
from ..config import (
    UPLOADS_DIR, SCRATCH_UPLOADS_DIR, OUTPUTS_BASE,
    MOLECULE_OUTPUTS_BASE, MOLECULE_TOOLS,
)
from ..gpu import query_gpus, invalidate_cache
from .. import job_runner

# Make pocket_utils importable for site validation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pocket_utils import parse_site_multichain, site_pairs_chains  # noqa: E402

router = APIRouter(prefix="/api", tags=["jobs"])


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:80]


@router.post("/jobs")
async def create_job(
    request: Request,
    job_name: str = Form(...),
    job_type: str = Form("generate"),
    project: str = Form(""),
    target: UploadFile = File(None),
    target_path: str = Form(""),
    target_token: str = Form(""),
    site: str = Form(...),
    length: str = Form("60-80"),
    tools: str = Form("rfdiffusion,boltzgen,bindcraft,pxdesign,proteina,proteina_complexa"),
    mode: str = Form("test"),
    gpu_id: int = Form(...),
    ss_bias: str = Form("balanced"),
    score_weights: str = Form("0.3,0.6,0.1"),
    reprediction: bool = Form(False),
    plip_top: int = Form(10),
    no_cys: bool = Form(False),
    max_aa_fraction: float = Form(None),
    min_sc: float = Form(None),
    max_refolding_rmsd: float = Form(None),
    min_site_interface_fraction: float = Form(None),
    max_site_dist: float = Form(None),
    min_site_fraction: float = Form(None),
    filter_site_pae: float = Form(None),
    esmfold_plddt_threshold: float = Form(None),
    top_n: int = Form(50),
    results_dir: str = Form(""),
    out_dir_override: str = Form(""),
    max_interface_ke: float = Form(None),
    bindcraft_filters: str = Form("no_filters"),
    n_tag: str = Form(""),
    c_tag: str = Form(""),
    # Molecule-specific params (optional, only used when job_type="molecule")
    library: str = Form(""),
    no_vina: str = Form(""),
    pocket_dist: str = Form("8.0"),
    max_atoms: str = Form("35"),
    # Boltz-2 affinity rescoring (mol_rerank only)
    boltz_affinity: bool = Form(False),
    boltz_affinity_top: int = Form(100),
    boltz_affinity_weight: float = Form(0.0),
    boltz_affinity_sort: bool = Form(False),
    boltz_affinity_mw_correction: bool = Form(False),
    boltz_affinity_sampling_steps: int = Form(200),
    boltz_affinity_diffusion_samples: int = Form(5),
):
    username = request.cookies.get("proteaflow_user", "")
    project = project.strip() or "unassigned"
    # Validate GPU — skip for rank_only reranks (no GPU needed)
    # GPU busy check is informational only — don't block job creation
    is_rerank_no_gpu = (
        (job_type == "rerank" and not reprediction)
        or (job_type == "mol_rerank" and not boltz_affinity)
    )
    if not is_rerank_no_gpu:
        gpus = query_gpus()
        gpu = next((g for g in gpus if g.index == gpu_id), None)
        if not gpu:
            raise HTTPException(400, f"GPU {gpu_id} not found")

    safe_name = _sanitize_name(job_name)
    date_str = time.strftime("%Y-%m-%d")

    # Handle target file. Three sources, in priority order:
    #   1. target_token  — file already staged via /api/preview/stage-target
    #                      (after the user previewed the pocket); promote it
    #                      into the permanent uploads dir and wipe scratch.
    #   2. target file   — multipart upload alongside the form submit.
    #   3. target_path   — server-side path supplied by the user.
    upload_dir = UPLOADS_DIR / safe_name
    if target_token:
        # Reject obvious traversal attempts before touching the filesystem.
        if "/" in target_token or "\\" in target_token or ".." in target_token:
            raise HTTPException(400, "Invalid target_token")
        scratch = SCRATCH_UPLOADS_DIR / target_token
        if not scratch.is_dir():
            raise HTTPException(400, "Staged target not found (token expired?)")
        staged = [p for p in scratch.iterdir()
                  if p.is_file() and p.suffix.lower() in (".pdb", ".cif")]
        if not staged:
            raise HTTPException(400, "No staged target file in scratch dir")
        upload_dir.mkdir(parents=True, exist_ok=True)
        src = staged[0]
        dest = upload_dir / src.name
        # Symlinks (when the staging step linked an on-disk path) get
        # resolved to the real file; uploaded blobs get copied/moved.
        if src.is_symlink():
            target_file = str(src.resolve())
        else:
            shutil.move(str(src), str(dest))
            target_file = str(dest)
        # Wipe the scratch dir — token is single-use.
        shutil.rmtree(scratch, ignore_errors=True)
    elif target and target.filename:
        upload_dir.mkdir(parents=True, exist_ok=True)
        target_file = str(upload_dir / target.filename)
        content = await target.read()
        with open(target_file, "wb") as f:
            f.write(content)
    elif target_path:
        target_file = target_path
        if not Path(target_file).exists():
            raise HTTPException(400, f"Target file not found: {target_file}")
    else:
        raise HTTPException(400, "No target file provided")

    # Validate site syntax up front. Molecule jobs use the multi-chain
    # parser; binder jobs go through the existing single-chain parser
    # later in the pipeline (parse_site rejects multi-chain itself).
    if job_type in ("molecule", "mol_rerank"):
        try:
            site_pairs = parse_site_multichain(site)
        except ValueError as e:
            raise HTTPException(400, f"Invalid site spec: {e}")
        # Best-effort chain-existence check — surface a clear error before
        # the GPU is acquired. Fall back gracefully if gemmi isn't installed.
        try:
            import gemmi
            st = gemmi.read_structure(target_file)
            chains_in_target = {ch.name for ch in st[0]} if len(st) else set()
            missing = [c for c in site_pairs_chains(site_pairs)
                       if c not in chains_in_target]
            if missing:
                raise HTTPException(
                    400,
                    f"Chain(s) {missing} not found in target "
                    f"(available: {sorted(chains_in_target)})")
        except HTTPException:
            raise
        except Exception:
            pass  # gemmi unavailable / unreadable: defer to runtime

    # Output directory — use override for reranks into subfolder
    if not project or not project.strip():
        project = "unassigned"
    if out_dir_override:
        out_dir = out_dir_override
    else:
        safe_project = _sanitize_name(project)
        if job_type in ("molecule", "mol_rerank"):
            outputs_base = MOLECULE_OUTPUTS_BASE
        else:
            outputs_base = OUTPUTS_BASE
        out_dir = str(outputs_base / safe_project / f"{date_str}_{safe_name}")
    log_file = str(Path(out_dir) / "run.log")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # For molecule jobs, the 'tools' Form param may contain binder defaults
    # when no molecule tools are checked. Filter to valid molecule tools only.
    if job_type == "molecule":
        valid_mol_tools = set(MOLECULE_TOOLS)
        mol_tools = [t.strip() for t in tools.split(",") if t.strip() in valid_mol_tools]
        tools_str = ",".join(mol_tools)  # empty string if library-only
    else:
        tools_str = None  # not used for binder jobs here

    params = {
        "job_type": job_type,
        "target_file": target_file,
        "site": site,
        "length": length,
        "tools": tools,
        "mode": mode,
        "ss_bias": ss_bias,
        "score_weights": score_weights,
        "reprediction": reprediction,
        "plip_top": plip_top,
        "no_cys": no_cys,
        "max_aa_fraction": max_aa_fraction,
        "min_sc": min_sc,
        "max_refolding_rmsd": max_refolding_rmsd,
        "min_site_interface_fraction": min_site_interface_fraction,
        "max_site_dist": max_site_dist,
        "min_site_fraction": min_site_fraction,
        "filter_site_pae": filter_site_pae,
        "esmfold_plddt_threshold": esmfold_plddt_threshold,
        "max_interface_ke": max_interface_ke,
        "top_n": top_n,
        "bindcraft_filters": bindcraft_filters,
        "out_dir": out_dir,
        "log_file": log_file,
        "results_dir": results_dir,
        "n_tag": n_tag.strip(),
        "c_tag": c_tag.strip(),
    }

    if job_type == "molecule":
        params["library"] = library.strip() if library else ""
        params["no_vina"] = bool(no_vina)
        params["pocket_dist"] = pocket_dist
        params["max_atoms"] = max_atoms
        params["tools"] = tools_str
    elif job_type == "mol_rerank":
        params["pocket_dist"] = pocket_dist
        params["score_weights"] = score_weights
        params["boltz_affinity"] = boltz_affinity
        if boltz_affinity:
            params["boltz_affinity_top"] = boltz_affinity_top
            params["boltz_affinity_weight"] = boltz_affinity_weight
            params["boltz_affinity_sort"] = boltz_affinity_sort
            params["boltz_affinity_mw_correction"] = boltz_affinity_mw_correction
            params["boltz_affinity_sampling_steps"] = boltz_affinity_sampling_steps
            params["boltz_affinity_diffusion_samples"] = boltz_affinity_diffusion_samples

    job_id = await database.create_job(
        name=job_name,
        job_type=job_type,
        gpu_id=gpu_id,
        target_file=target_file,
        site=site,
        length=length,
        tools=tools,
        mode=mode,
        params=params,
        out_dir=out_dir,
        log_file=log_file,
        username=username,
        project=project,
    )

    await job_runner.launch_job(job_id, params, gpu_id)
    invalidate_cache()
    database.add_shared_project(project)

    return {"job_id": job_id, "status": "running", "out_dir": out_dir}


@router.get("/jobs")
async def list_jobs(status: str = None, limit: int = 50):
    jobs = await database.list_jobs(status=status, limit=limit)
    for j in jobs:
        if j.get("params_json"):
            j["params"] = json.loads(j["params_json"])
    return jobs


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("params_json"):
        job["params"] = json.loads(job["params_json"])
    return job


@router.post("/jobs/{job_id}/archive")
async def archive_job(job_id: str):
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    await database.update_job(job_id, archived=1)
    return {"archived": True}


@router.patch("/jobs/{job_id}/rename")
async def rename_job(job_id: str, name: str = Form(...)):
    """Rename a completed/failed job (display name only, does not move files)."""
    job = await database.get_job_any_machine(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "running":
        raise HTTPException(400, "Cannot rename a running job")
    new_name = name.strip()
    if not new_name:
        raise HTTPException(400, "Name cannot be empty")
    await database.update_job(job_id, name=new_name)
    return {"renamed": True, "name": new_name}


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "running":
        raise HTTPException(400, f"Job is {job['status']}, cannot cancel")
    ok = await job_runner.cancel_job(job_id)
    invalidate_cache()
    return {"cancelled": ok}


@router.post("/jobs/{job_id}/skip/{tool}")
async def skip_tool(job_id: str, tool: str):
    """Write a skip signal file for the specified tool.

    The pipeline polls for .skip_{tool} in the output directory and
    terminates the running subprocess when found. Only affects the
    current generation step — validation steps are not skippable.
    """
    VALID_TOOLS = {"rfdiffusion", "boltzgen", "bindcraft", "pxdesign",
                   "proteina", "proteina_complexa",
                   "pocketflow", "molcraft", "pocketxmol", "pocket2mol", "decompdiff"}
    if tool not in VALID_TOOLS:
        raise HTTPException(400, f"Invalid tool: {tool}. Must be one of {VALID_TOOLS}")

    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "running":
        raise HTTPException(400, f"Job is {job['status']}, cannot skip tool")

    out_dir = Path(job["out_dir"])
    skip_file = out_dir / f".skip_{tool}"
    skip_file.write_text(f"skip requested at {time.time()}\n")
    return {"skip_requested": True, "tool": tool, "signal_file": str(skip_file)}
