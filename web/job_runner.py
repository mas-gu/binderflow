"""Subprocess management for pipeline jobs."""

import asyncio
import os
import signal
import time
from pathlib import Path

from . import database
from .config import PIPELINE_SCRIPT, RERANK_SCRIPT, MOLECULE_SCRIPT, RERANK_MOLECULE_SCRIPT, PROJECT_ROOT, PYTHON_BIN
import logging

log = logging.getLogger(__name__)


_processes: dict[str, "asyncio.subprocess.Process"] = {}

# Valid --ss_bias choices, mirrored from generate_binders.py / rerank_binders.py
# argparse (choices=["beta", "helix", "balanced"]). Anything else (notably the
# empty string produced when a UI filter value like "any"/"sheet" is assigned
# into the rerank <select>) would make argparse exit rc=2.
SS_BIAS_CHOICES = ("beta", "helix", "balanced")
SS_BIAS_DEFAULT = "balanced"
# Browser-side SS *filter* words that are not CLI choices (mirrors
# livererank.js SS_BIAS_TO_CLI) — accepted here so a direct API POST keeps
# the intended meaning instead of silently falling back to the default.
SS_BIAS_ALIASES = {"any": "balanced", "sheet": "beta"}


def _num_arg(params: dict, key: str) -> str | None:
    """Return the stringified value of a numeric param, or None if unset.

    A plain ``if params.get(key):`` test silently drops legitimate zeros, so
    ``--plip_top 0`` ("set to 0 to disable") and ``--max_site_dist 0``
    ("0=disable filter") were unreachable from the web UI. Only None and
    blank strings count as "not supplied".
    """
    val = params.get(key)
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
    return str(val)


def _ss_bias(params: dict) -> str:
    """Return a valid --ss_bias value, falling back to the default."""
    val = params.get("ss_bias")
    val = val.strip().lower() if isinstance(val, str) else val
    val = SS_BIAS_ALIASES.get(val, val)
    return val if val in SS_BIAS_CHOICES else SS_BIAS_DEFAULT


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
        "--ss_bias", _ss_bias(params),
    ]
    if params.get("reprediction"):
        cmd.append("--reprediction")
    if params.get("n_tag"):
        cmd.extend(["--n_tag", params["n_tag"]])
    if params.get("c_tag"):
        cmd.extend(["--c_tag", params["c_tag"]])
    # Numeric params: 0 / 0.0 are meaningful values, not "unset".
    for key in ("plip_top", "esmfold_plddt_threshold", "top_n",
                "max_site_dist", "min_site_fraction", "filter_site_pae"):
        val = _num_arg(params, key)
        if val is not None:
            cmd.extend([f"--{key}", val])
    bf = params.get("bindcraft_filters", "no_filters")
    if bf and bf != "no_filters":
        cmd.extend(["--bindcraft_filters", bf])
    bf2 = params.get("bindcraft2_filters", "no_filters")
    if bf2 and bf2 != "no_filters":
        cmd.extend(["--bindcraft2_filters", bf2])
    mod2 = params.get("bindcraft2_modality", "binder")
    if mod2 and mod2 != "binder":
        cmd.extend(["--bindcraft2_modality", mod2])
    return cmd


def build_rerank_cmd(params: dict) -> list[str]:
    """Build rerank_binders.py CLI command from params dict."""
    cmd = [
        PYTHON_BIN, str(RERANK_SCRIPT),
        "--target", params["target_file"],
        "--site", params["site"],
        "--out_dir", params["out_dir"],
        "--score_weights", params.get("score_weights", "0.3,0.6,0.1"),
        "--ss_bias", _ss_bias(params),
    ]
    if not params.get("reprediction"):
        cmd.append("--rank_only")
    if params.get("results_dir"):
        cmd.extend(["--results_dir", params["results_dir"]])
    if params.get("no_cys"):
        cmd.append("--no_cys")
    # Numeric params: 0 / 0.0 are meaningful values, not "unset".
    for key in ("plip_top", "max_aa_fraction", "min_sc", "max_refolding_rmsd",
                "min_site_interface_fraction", "max_site_dist",
                "min_site_fraction", "max_interface_ke",
                "esmfold_plddt_threshold", "top_n"):
        val = _num_arg(params, key)
        if val is not None:
            cmd.extend([f"--{key}", val])
    if params.get("reprediction"):
        cmd.append("--reprediction")
        # Remove --rank_only if it was added (shouldn't be, but safety check)
        if "--rank_only" in cmd:
            cmd.remove("--rank_only")
    if params.get("n_tag"):
        cmd.extend(["--n_tag", params["n_tag"]])
    if params.get("c_tag"):
        cmd.extend(["--c_tag", params["c_tag"]])
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
    # rerank_molecules.py declares --results_dir required=True (unlike
    # rerank_binders.py, where it is optional and means "rerank in place"), so
    # the conditional below would launch a job that dies instantly with an
    # argparse usage dump and rc=2. The web form marks the field required, but
    # the API accepts an empty string, so a direct API call still reaches here.
    # Fail with something a caller can act on instead.
    if not params.get("results_dir"):
        raise ValueError(
            "mol_rerank requires results_dir — rerank_molecules.py cannot "
            "rerank in place (its binder counterpart can).")
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


def _rotate_log(log_path: str) -> str | None:
    """Move an existing log aside so a re-run cannot destroy it.

    ``log_file`` is the deterministic ``{out_dir}/run.log``, so re-submitting
    the same job name on the same day (or re-running a rerank into an existing
    results dir) used to truncate the very log needed to diagnose the previous
    attempt. The old log is renamed to ``run.log.<YYYYmmdd-HHMMSS>`` and the
    fresh run then starts from an empty file (keeps the SSE tailer simple).

    Returns the archived path, or None if there was nothing to rotate.
    """
    p = Path(log_path)
    try:
        if not p.is_file() or p.stat().st_size == 0:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(p.stat().st_mtime))
        archived = p.with_name(f"{p.name}.{stamp}")
        n = 1
        while archived.exists():
            archived = p.with_name(f"{p.name}.{stamp}.{n}")
            n += 1
        p.rename(archived)
        return str(archived)
    except OSError:
        return None


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
    _rotate_log(log_path)

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

    # Do not clobber a terminal status someone else already set. cancel_job()
    # writes status="cancelled" and then the process dies, so this task ran a
    # moment later and unconditionally overwrote it with "failed" — a cancelled
    # job reported as a failure. Re-read before writing.
    try:
        _cur = await database.get_job(job_id)
        _cur_status = (_cur or {}).get("status")
    except Exception:
        _cur_status = None
    if _cur_status in ("cancelled", "completed"):
        log.info("_wait_for_job(%s): status already %r — not overwriting with %r",
                 job_id, _cur_status, status)
        return

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


def pid_matches_job(pid: int, job) -> bool:
    """Is `pid` really THIS job's pipeline, or has the PID been recycled?

    A PID recovered from the database after a server restart is just an integer.
    Linux recycles PIDs, so on a long-lived box (or after a reboot) that integer
    can belong to something else entirely — and cancel_job() SIGKILLs the whole
    process GROUP, so a wrong match kills an unrelated process tree, potentially
    the user's own shell.

    Every job has a unique out_dir (there is a UNIQUE index on it), and the
    pipeline is always launched with it on the command line, so it is a reliable
    identity token. Fails CLOSED: if we cannot read /proc or cannot see the
    out_dir, we refuse to claim the PID is ours.
    """
    try:
        out_dir = (job or {}).get("out_dir") or ""
    except Exception:
        return False
    if not out_dir:
        return False
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except (OSError, ValueError):
        return False
    return str(out_dir).rstrip("/") in cmdline


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
            # Verify identity before trusting a bare integer from the DB — see
            # pid_matches_job. Without this a recycled PID gets its whole process
            # group SIGKILLed.
            if pid_matches_job(job["pid"], job):
                pid = job["pid"]
            else:
                log.warning(
                    "cancel_job(%s): stored PID %s is not this job (recycled or "
                    "already gone) — refusing to kill it; marking job failed",
                    job_id, job["pid"])
                await database.update_job(job_id, status="failed", exit_code=-1,
                                          error_msg="Process no longer exists (stale PID)")
                return True

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
