"""HTML page routes (Jinja2 templates)."""

import json
import os
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from ..config import (
    TEMPLATES_DIR, TOOL_COLORS, TOOL_DESCRIPTIONS, DEFAULT_TOOLS,
    DEFAULT_MODE, DEFAULT_SS_BIAS, DEFAULT_SCORE_WEIGHTS, DEFAULT_LENGTH,
    MODE_ESTIMATES, SCATTER_PRESETS,
    SCORE_COLUMNS_HIGHER_BETTER, SCORE_COLUMNS_LOWER_BETTER, DEFAULT_COLUMNS,
    OUTPUTS_BASE, HOSTNAME,
    MOLECULE_TOOLS, MOLECULE_TOOL_COLORS, MOLECULE_TOOL_DESCRIPTIONS,
    MOLECULE_MODE_ESTIMATES, MOLECULE_DEFAULT_TOOLS, MOLECULE_SCORE_WEIGHTS,
    COMPOUND_LIBRARIES,
)
from ..gpu import query_gpus
from .. import database


async def _check_stale_jobs():
    """Check running jobs for dead PIDs and mark as completed/failed.

    Lightweight — runs on every dashboard load. Only checks local machine jobs.
    """
    jobs = await database.list_jobs(status="running")
    for job in jobs:
        pid = job.get("pid")
        if not pid:
            continue
        # Check if PID is alive
        try:
            os.kill(pid, 0)
            continue  # alive — skip
        except (OSError, ProcessLookupError):
            pass  # dead — check if completed

        out_dir = job.get("out_dir", "")
        if out_dir and (Path(out_dir) / "rankings.csv").exists():
            status = "completed"
        else:
            log_file = job.get("log_file", "")
            status = "failed"
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file) as f:
                        tail = f.readlines()[-10:]
                    if any(m in line for line in tail
                           for m in ("DONE", "dashboard.png", "rankings.csv",
                                     "summary plot", "total designs")):
                        status = "completed"
                except Exception:
                    pass

        await database.update_job(
            job["id"], status=status,
            exit_code=0 if status == "completed" else -1,
            error_msg="Recovered (stale PID)" if status == "failed" else None,
        )

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _get_username(request: Request) -> str:
    return request.cookies.get("proteaflow_user", "")


def _render(request: Request, name: str, ctx: dict):
    """Compat wrapper for Starlette 1.x TemplateResponse."""
    ctx["request"] = request
    ctx.setdefault("username", _get_username(request))
    return templates.TemplateResponse(request, name, ctx)


@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"request": request})


@router.post("/login")
async def login_submit(request: Request, username: str = Form(...)):
    name = username.strip()
    if not name:
        return templates.TemplateResponse(request, "login.html",
                                          {"request": request, "error": "Please enter your name"})
    database.add_shared_user(name)
    response = RedirectResponse("/", status_code=302)
    response.set_cookie("proteaflow_user", name, max_age=30 * 24 * 3600)  # 30 days
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("proteaflow_user")
    return response


@router.get("/")
async def index(request: Request):
    username = _get_username(request)
    # Check for stale jobs on every dashboard load (lightweight PID check)
    await _check_stale_jobs()
    # Local jobs for GPU status + active monitoring
    local_jobs = await database.list_jobs(limit=50)
    running = await database.list_jobs(status="running")
    gpus = query_gpus()
    for gpu in gpus:
        for j in running:
            if j["gpu_id"] == gpu.index:
                gpu.busy = True
                gpu.job_name = j["name"]
                gpu.job_id = j["id"]

    # All jobs across machines for results + history
    all_jobs = await database.list_all_jobs(limit=100)

    # Group completed jobs by project
    completed = [j for j in all_jobs if j["status"] == "completed"]
    projects_dict = {}
    for j in completed:
        proj = j.get("project") or "unassigned"
        projects_dict.setdefault(proj, []).append(j)

    return _render(request, "index.html", {
        "jobs": all_jobs,
        "gpus": gpus,
        "projects_dict": projects_dict,
    })


@router.get("/launch")
async def launch(request: Request):
    gpus = query_gpus()
    running = await database.list_jobs(status="running")
    projects = await database.list_projects()
    for gpu in gpus:
        for j in running:
            if j["gpu_id"] == gpu.index:
                gpu.busy = True
                gpu.job_name = j["name"]
    return _render(request, "launch.html", {
        "gpus": gpus,
        "tool_colors": TOOL_COLORS,
        "tool_descriptions": TOOL_DESCRIPTIONS,
        "default_tools": DEFAULT_TOOLS,
        "default_mode": DEFAULT_MODE,
        "default_ss_bias": DEFAULT_SS_BIAS,
        "default_score_weights": DEFAULT_SCORE_WEIGHTS,
        "default_length": DEFAULT_LENGTH,
        "mode_estimates": MODE_ESTIMATES,
        "projects": projects,
        "molecule_tools": MOLECULE_TOOLS,
        "molecule_tool_colors": MOLECULE_TOOL_COLORS,
        "molecule_tool_descriptions": MOLECULE_TOOL_DESCRIPTIONS,
        "molecule_mode_estimates": MOLECULE_MODE_ESTIMATES,
        "molecule_default_tools": MOLECULE_DEFAULT_TOOLS,
        "molecule_score_weights": MOLECULE_SCORE_WEIGHTS,
        "compound_libraries": COMPOUND_LIBRARIES,
    })


@router.get("/monitor/{job_id}")
async def monitor(request: Request, job_id: str):
    job = await database.get_job(job_id)
    return _render(request, "monitor.html", {
        "job": job,
        "tool_colors": TOOL_COLORS,
    })


@router.get("/rerank")
async def rerank_page(request: Request, results_dir: str = "", type: str = ""):
    gpus = query_gpus()
    running = await database.list_jobs(status="running")
    all_completed = await database.list_all_jobs(status="completed")
    completed_jobs = [j for j in all_completed
                      if j.get("job_type") in ("generate", "molecule")]
    for gpu in gpus:
        for j in running:
            if j["gpu_id"] == gpu.index:
                gpu.busy = True
                gpu.job_name = j["name"]
    return _render(request, "rerank.html", {
        "gpus": gpus,
        "completed_jobs": completed_jobs,
        "tool_colors": TOOL_COLORS,
        "prefill_results_dir": results_dir,
        "prefill_type": type or "",
        "score_weights": MOLECULE_SCORE_WEIGHTS,
    })



@router.get("/results/{job_id}")
async def results(request: Request, job_id: str):
    job = await database.get_job_any_machine(job_id)
    if not job:
        from fastapi import HTTPException
        raise HTTPException(404, "Job not found")

    # Route molecule jobs to dedicated results page
    if job.get("job_type") in ("molecule", "mol_rerank"):
        return _results_molecule(request, job)

    return _results_binder(request, job)


def _results_molecule(request: Request, job: dict):
    """Render molecule-specific results page."""
    mol_scatter_presets = {
        "QED vs Vina Score": ("qed", "vina_score"),
        "SA vs Vina Score": ("sa_score", "vina_score"),
        "MW vs QED": ("mw", "qed"),
        "LogP vs QED": ("logp", "qed"),
        "Combined Score vs Vina": ("combined_score", "vina_score"),
        "Combined Score vs QED": ("combined_score", "qed"),
        "MW vs Vina Score": ("mw", "vina_score"),
        "TPSA vs LogP": ("tpsa", "logp"),
        "Pocket Fit vs Vina": ("pocket_fit", "vina_score"),
        "Pocket Fit vs QED": ("pocket_fit", "qed"),
        "LE vs QED": ("ligand_efficiency", "qed"),
        "Fsp3 vs QED": ("fsp3", "qed"),
    }
    mol_higher_better = ["combined_score", "qed"]
    mol_lower_better = ["sa_score", "vina_score", "lipinski_violations"]
    mol_default_columns = [
        "rank", "design_id", "tool", "smiles", "combined_score",
        "qed", "sa_score", "vina_score", "mw", "logp",
        "hbd", "hba", "tpsa", "lipinski_violations",
    ]
    mol_tool_metrics = ["combined_score", "qed", "sa_score", "vina_score", "mw", "logp", "tpsa"]

    return _render(request, "results_mol.html", {
        "job": job,
        "tool_colors_json": json.dumps(MOLECULE_TOOL_COLORS),
        "scatter_presets_json": json.dumps(mol_scatter_presets),
        "higher_better_json": json.dumps(mol_higher_better),
        "lower_better_json": json.dumps(mol_lower_better),
        "default_columns_json": json.dumps(mol_default_columns),
        "tool_metrics_json": json.dumps(mol_tool_metrics),
    })


def _results_binder(request: Request, job: dict):
    """Render binder results page (original behavior)."""
    radar_axes = [
        {
            "name": "Binding (iPTM)", "col": "boltz_iptm", "higher": True,
            "strategy": "fixed", "min": 0, "max": 1,
            "good": ">0.8", "tip": "Interface pTM from Boltz-2. 0–1 probability scale. >0.8 = confident binding.",
        },
        {
            "name": "Shape (Sc)", "col": "rosetta_sc", "higher": True,
            "strategy": "threshold", "threshold_default": 0.45, "threshold_param": "min_sc",
            "threshold_pct": 0.70,
            "good": ">0.6", "tip": "Rosetta shape complementarity. 0.45 \u2192 70% on radar, 1.0 \u2192 100%. Beta-sheet designs have intrinsically lower Sc.",
        },
        {
            "name": "Stability (RMSD)", "col": "refolding_rmsd", "higher": False,
            "strategy": "threshold", "threshold_default": 2.5, "threshold_param": "max_refolding_rmsd",
            "good": "<2.5 \u00c5", "tip": "Binder RMSD: ESMFold vs Boltz-2 structure. Lower = more stable fold. Threshold-anchored at 50%.",
        },
        {
            "name": "Interface (PAE)", "col": "boltz_mean_interface_pae", "higher": False,
            "strategy": "threshold", "threshold_default": 10, "threshold_param": "filter_interface_pae",
            "good": "<10 \u00c5", "tip": "Boltz-2 mean interface PAE. Lower = more confident interface prediction. Threshold-anchored at 10\u00c5 \u2192 50%.",
        },
        {
            "name": "Low K+E", "col": "interface_KE_fraction", "higher": False,
            "strategy": "threshold", "threshold_default": 0.25, "threshold_param": "max_interface_ke",
            "good": "<25%", "tip": "Charged residues (Lys+Glu) at interface. Lower = less electrostatic noise. Anchored on user filter threshold.",
        },
        {
            "name": "Low Agg (SAP)", "col": "rosetta_sap", "higher": False,
            "strategy": "data_driven",
            "good": "low", "tip": "Rosetta surface aggregation propensity. Scales with protein size — no universal scale. Normalized to dataset: median=50%, 5th pct=95%.",
        },
        {
            "name": "Solubility", "col": "netsolp_solubility", "higher": True,
            "strategy": "fixed", "min": 0.5, "max": 1,
            "good": ">0.7", "tip": "NetSolP predicted solubility. 0.5–1 range. >0.7 = likely soluble in E. coli.",
        },
        {
            "name": "Site Focus (SIF)", "col": "site_interface_fraction", "higher": True,
            "strategy": "threshold", "threshold_default": 0.30, "threshold_param": "min_site_interface_fraction",
            "good": ">30%", "tip": "Fraction of binder interface contacting the target site. Threshold-anchored. Scale depends on site size.",
        },
    ]

    tool_metrics = [
        "combined_score", "boltz_iptm", "boltz_binder_plddt", "esmfold_plddt",
        "rosetta_dG", "boltz_site_mean_pae", "boltz_mean_interface_pae", "rosetta_sc",
        "site_interface_fraction", "refolding_rmsd",
        "interface_KE_fraction", "binder_helix_frac", "binder_sheet_frac",
    ]

    return _render(request, "results.html", {
        "job": job,
        "tool_colors_json": json.dumps(TOOL_COLORS),
        "scatter_presets_json": json.dumps(SCATTER_PRESETS),
        "higher_better_json": json.dumps(list(SCORE_COLUMNS_HIGHER_BETTER)),
        "lower_better_json": json.dumps(list(SCORE_COLUMNS_LOWER_BETTER)),
        "default_columns_json": json.dumps(DEFAULT_COLUMNS),
        "radar_axes_json": json.dumps(radar_axes),
        "tool_metrics_json": json.dumps(tool_metrics),
    })


@router.get("/manager")
async def manager_page(request: Request):
    all_jobs = await database.list_all_jobs(include_archived=True, limit=500)
    projects = await database.list_projects()
    users = sorted({j.get("username", "") for j in all_jobs if j.get("username")})
    machines = sorted({j.get("machine", "") for j in all_jobs if j.get("machine")})
    return _render(request, "manager.html", {
        "jobs": all_jobs,
        "projects": projects,
        "users": users,
        "machines": machines,
        "hostname": HOSTNAME,
        "outputs_base": str(OUTPUTS_BASE),
    })
