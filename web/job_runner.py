"""Subprocess management for pipeline jobs."""

import asyncio
import os
import signal
import time
from pathlib import Path

from . import database
from .config import PIPELINE_SCRIPT, RERANK_SCRIPT, MOLECULE_SCRIPT, RERANK_MOLECULE_SCRIPT, PROJECT_ROOT, PYTHON_BIN


_processes: dict[str, "asyncio.subprocess.Process"] = {}


def build_generate_cmd(params: dict) -> list[str]:
    """Build generate_binders.py CLI command from params dict."""
    cmd = [
        PYTHON_BIN, str(PIPELINE_SCRIPT),
        "--target", params["target_file"],
        "--site", params["site"],
        "--length", params["length"],
        "--tools", params["tools"],
        "--mode", params["mode"],
        "--out_dir", params["out_dir"],
        "--score_weights", params.get("score_weights", "0.3,0.6,0.1"),
        "--ss_bias", params.get("ss_bias", "balanced"),
    ]
    if params.get("reprediction"):
        cmd.append("--reprediction")
    if params.get("n_tag"):
        cmd.extend(["--n_tag", params["n_tag"]])
    if params.get("c_tag"):
        cmd.extend(["--c_tag", params["c_tag"]])
    if params.get("plip_top"):
        cmd.extend(["--plip_top", str(params["plip_top"])])
    if params.get("esmfold_plddt_threshold"):
        cmd.extend(["--esmfold_plddt_threshold", str(params["esmfold_plddt_threshold"])])
    if params.get("top_n"):
        cmd.extend(["--top_n", str(params["top_n"])])
    if params.get("max_site_dist"):
        cmd.extend(["--max_site_dist", str(params["max_site_dist"])])
    if params.get("min_site_fraction"):
        cmd.extend(["--min_site_fraction", str(params["min_site_fraction"])])
    if params.get("filter_site_pae"):
        cmd.extend(["--filter_site_pae", str(params["filter_site_pae"])])
    bf = params.get("bindcraft_filters", "no_filters")
    if bf and bf != "no_filters":
        cmd.extend(["--bindcraft_filters", bf])
    return cmd


def build_rerank_cmd(params: dict) -> list[str]:
    """Build rerank_binders.py CLI command from params dict."""
    cmd = [
        PYTHON_BIN, str(RERANK_SCRIPT),
        "--target", params["target_file"],
        "--site", params["site"],
        "--out_dir", params["out_dir"],
        "--score_weights", params.get("score_weights", "0.3,0.6,0.1"),
        "--ss_bias", params.get("ss_bias", "balanced"),
    ]
    if not params.get("reprediction"):
        cmd.append("--rank_only")
    if params.get("results_dir"):
        cmd.extend(["--results_dir", params["results_dir"]])
    if params.get("plip_top"):
        cmd.extend(["--plip_top", str(params["plip_top"])])
    if params.get("no_cys"):
        cmd.append("--no_cys")
    if params.get("max_aa_fraction"):
        cmd.extend(["--max_aa_fraction", str(params["max_aa_fraction"])])
    if params.get("min_sc"):
        cmd.extend(["--min_sc", str(params["min_sc"])])
    if params.get("max_refolding_rmsd"):
        cmd.extend(["--max_refolding_rmsd", str(params["max_refolding_rmsd"])])
    if params.get("min_site_interface_fraction"):
        cmd.extend(["--min_site_interface_fraction", str(params["min_site_interface_fraction"])])
    if params.get("max_site_dist"):
        cmd.extend(["--max_site_dist", str(params["max_site_dist"])])
    if params.get("min_site_fraction"):
        cmd.extend(["--min_site_fraction", str(params["min_site_fraction"])])
    if params.get("max_interface_ke"):
        cmd.extend(["--max_interface_ke", str(params["max_interface_ke"])])
    if params.get("reprediction"):
        cmd.append("--reprediction")
        # Remove --rank_only if it was added (shouldn't be, but safety check)
        if "--rank_only" in cmd:
            cmd.remove("--rank_only")
    if params.get("n_tag"):
        cmd.extend(["--n_tag", params["n_tag"]])
    if params.get("c_tag"):
        cmd.extend(["--c_tag", params["c_tag"]])
    if params.get("esmfold_plddt_threshold"):
        cmd.extend(["--esmfold_plddt_threshold", str(params["esmfold_plddt_threshold"])])
    if params.get("top_n"):
        cmd.extend(["--top_n", str(params["top_n"])])
    return cmd


def build_molecule_cmd(params: dict) -> list[str]:
    """Build generate_molecules.py CLI command."""
    cmd = [
        PYTHON_BIN, str(MOLECULE_SCRIPT),
        "--target", params["target_file"],
        "--site", params["site"],
        "--out_dir", params["out_dir"],
        "--device", "cuda:0",
    ]
    if params.get("tools"):
        cmd += ["--tools", params["tools"]]
    if params.get("mode"):
        cmd += ["--mode", params["mode"]]
    if params.get("library") and params["library"].strip():
        cmd += ["--library", params["library"]]
    if params.get("no_vina"):
        cmd += ["--no_vina"]
    if params.get("score_weights"):
        cmd += ["--score_weights", params["score_weights"]]
    if params.get("pocket_dist"):
        cmd += ["--pocket_dist", str(params["pocket_dist"])]
    if params.get("max_atoms"):
        cmd += ["--max_atoms", str(params["max_atoms"])]
    if params.get("top_n"):
        cmd += ["--top_n", str(params["top_n"])]
    return cmd


def build_mol_rerank_cmd(params: dict) -> list[str]:
    """Build rerank_molecules.py CLI command."""
    cmd = [
        PYTHON_BIN, str(RERANK_MOLECULE_SCRIPT),
        "--target", params["target_file"],
        "--site", params["site"],
        "--out_dir", params["out_dir"],
    ]
    if params.get("results_dir"):
        cmd += ["--results_dir", params["results_dir"]]
    if params.get("score_weights"):
        cmd += ["--score_weights", params["score_weights"]]
    if params.get("pocket_dist"):
        cmd += ["--pocket_dist", str(params["pocket_dist"])]
    if params.get("top_n"):
        cmd += ["--top_n", str(params["top_n"])]
    # Hard filters
    for flag in ["min_qed", "max_sa", "min_pocket_fit", "min_vina", "max_vina"]:
        val = params.get(flag)
        if val is not None and str(val).strip():
            cmd += [f"--{flag}", str(val)]
    if params.get("mw_range") and str(params["mw_range"]).strip():
        cmd += ["--mw_range", str(params["mw_range"])]
    # Boltz-2 affinity rescoring
    if params.get("boltz_affinity"):
        cmd += ["--boltz_affinity"]
        if params.get("boltz_affinity_top") is not None:
            cmd += ["--boltz_affinity_top", str(params["boltz_affinity_top"])]
        if params.get("boltz_affinity_weight") is not None:
            cmd += ["--boltz_affinity_weight", str(params["boltz_affinity_weight"])]
        if params.get("boltz_affinity_sort"):
            cmd += ["--boltz_affinity_sort"]
        if params.get("boltz_affinity_mw_correction"):
            cmd += ["--boltz_affinity_mw_correction"]
        if params.get("boltz_affinity_sampling_steps") is not None:
            cmd += ["--boltz_affinity_sampling_steps", str(params["boltz_affinity_sampling_steps"])]
        if params.get("boltz_affinity_diffusion_samples") is not None:
            cmd += ["--boltz_affinity_diffusion_samples", str(params["boltz_affinity_diffusion_samples"])]
    return cmd


async def launch_job(job_id: str, params: dict, gpu_id: int):
    """Start a pipeline subprocess."""
    job_type = params.get("job_type", "generate")
    if job_type == "rerank":
        cmd = build_rerank_cmd(params)
    elif job_type == "molecule":
        cmd = build_molecule_cmd(params)
    elif job_type == "mol_rerank":
        cmd = build_mol_rerank_cmd(params)
    else:
        cmd = build_generate_cmd(params)

    log_path = params["log_file"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    log_fh = open(log_path, "w")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=log_fh,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
        start_new_session=True,
    )
    _processes[job_id] = proc

    await database.update_job(
        job_id,
        status="running",
        pid=proc.pid,
        started_at=time.time(),
    )

    # Wait for completion in background
    asyncio.create_task(_wait_for_job(job_id, proc, log_fh))


async def _wait_for_job(job_id: str, proc, log_fh):
    """Wait for subprocess to finish and update DB."""
    try:
        returncode = await proc.wait()
    finally:
        log_fh.close()
        _processes.pop(job_id, None)

    status = "completed" if returncode == 0 else "failed"
    error_msg = None
    if returncode != 0:
        try:
            with open(log_fh.name) as f:
                lines = f.readlines()
                error_msg = "".join(lines[-20:])
        except Exception:
            error_msg = f"Exit code {returncode}"

    await database.update_job(
        job_id,
        status=status,
        finished_at=time.time(),
        exit_code=returncode,
        error_msg=error_msg,
    )


def _get_all_descendants(pid: int) -> list[int]:
    """Get all descendant PIDs of a process (children, grandchildren, etc.).

    Walks /proc to find the full process tree, including processes in
    different sessions (e.g. tool subprocesses started with start_new_session).
    """
    descendants = []
    try:
        import subprocess as _sp
        result = _sp.run(
            ["ps", "--ppid", str(pid), "-o", "pid=", "--no-headers"],
            capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().splitlines():
            child_pid = int(line.strip())
            descendants.append(child_pid)
            descendants.extend(_get_all_descendants(child_pid))
    except Exception:
        pass
    return descendants


async def cancel_job(job_id: str) -> bool:
    """Cancel a running job.

    Works both when the process is tracked in-memory (_processes) and when
    the server was restarted and only the PID in the database remains.

    Kills the full process tree including tool subprocesses that run in
    separate sessions (start_new_session=True in run_cmd).
    """
    proc = _processes.get(job_id)
    pid = proc.pid if proc and proc.returncode is None else None

    # If not in memory, try to recover PID from the database
    if pid is None:
        job = await database.get_job(job_id)
        if job and job["status"] == "running" and job.get("pid"):
            pid = job["pid"]

    if pid is None:
        return False

    # Collect ALL descendant PIDs before killing (the tree disappears after kill)
    all_pids = _get_all_descendants(pid)
    all_pids.append(pid)
    # Collect unique process groups from all descendants
    pgids = set()
    for p in all_pids:
        try:
            pgids.add(os.getpgid(p))
        except (OSError, ProcessLookupError):
            pass

    # SIGTERM all process groups (covers both pipeline and tool sessions)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    # Also SIGTERM individual PIDs in case they escaped process groups
    for p in all_pids:
        try:
            os.kill(p, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    # Give SIGTERM 3 seconds, then SIGKILL everything
    await asyncio.sleep(3)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    for p in all_pids:
        try:
            os.kill(p, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    await database.update_job(
        job_id,
        status="cancelled",
        finished_at=time.time(),
        exit_code=-15,
    )
    _processes.pop(job_id, None)
    return True
