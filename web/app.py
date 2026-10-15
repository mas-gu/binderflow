"""ProteaFlow Web — FastAPI application."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .config import STATIC_DIR, UPLOADS_DIR, SCRATCH_UPLOADS_DIR
from . import database
from .routes import (
    pages, api_gpus, api_jobs, api_results, api_stream,
    api_manager, api_geometry, api_preview,
)


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a proteaflow_user cookie for everything except /login and /static.

    /api used to be exempt, which left every mutating endpoint unauthenticated —
    launching jobs, cancelling them, archiving, and the manager's delete (which
    rmtree's a run directory) were all reachable by anyone who could open the
    port. Pages redirect to /login; API calls get a 401 instead, so fetch()
    callers see a real error rather than an HTML login page with status 200.
    """
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/login") or path.startswith("/static"):
            return await call_next(request)
        username = request.cookies.get("proteaflow_user")
        if not username:
            if path.startswith("/api"):
                return JSONResponse({"error": "authentication required"}, status_code=401)
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)


async def _cleanup_stale_jobs():
    """Mark jobs with dead PIDs as failed/completed on server startup."""
    from pathlib import Path
    jobs = await database.list_jobs(status="running")
    for job in jobs:
        pid = job.get("pid")
        if pid and not _pid_alive(pid):
            status = "failed"
            # Check if the job actually completed:
            # 1. rankings.csv was written *by this run* (mtime after job start).
            #    A bare existence test marks crashed reranks as completed,
            #    because a rerank runs INTO a results dir that already has a
            #    rankings.csv from the original generate run.
            # 2. Log contains completion markers.
            out_dir = job.get("out_dir", "")
            started = job.get("started_at") or job.get("created_at") or 0
            rankings = Path(out_dir) / "rankings.csv" if out_dir else None
            fresh_rankings = False
            if rankings is not None and started:
                try:
                    # 60 s of slack absorbs clock/filesystem timestamp skew.
                    fresh_rankings = (
                        rankings.is_file()
                        and rankings.stat().st_mtime >= (started - 60)
                    )
                except OSError:
                    fresh_rankings = False
            if fresh_rankings:
                status = "completed"
            else:
                log_file = job.get("log_file", "")
                if log_file and os.path.exists(log_file):
                    try:
                        with open(log_file) as f:
                            tail = f.readlines()[-10:]
                        if any(marker in line for line in tail
                               for marker in ("DONE", "dashboard.png", "rankings.csv",
                                              "summary plot", "total designs")):
                            status = "completed"
                    except Exception:
                        pass
            await database.update_job(
                job["id"], status=status, exit_code=-1 if status == "failed" else 0,
                error_msg="Recovered on server restart" if status == "failed" else None,
            )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # EPERM means the process EXISTS, we just may not signal it (different
        # uid). Treating that as "dead" marked live jobs failed on startup.
        return True
    except (OSError, ProcessLookupError):
        return False


def _cleanup_orphaned_scratch():
    """Remove scratch upload dirs older than 24 h.

    Token dirs that didn't end up tied to a submitted job persist as
    leftover bytes; sweep them on startup. Live submissions promote
    their staged file into UPLOADS_DIR and remove the scratch dir
    immediately, so anything still here is orphaned.
    """
    import shutil
    import time
    cutoff = time.time() - 24 * 3600
    if not SCRATCH_UPLOADS_DIR.exists():
        return
    for entry in SCRATCH_UPLOADS_DIR.iterdir():
        if entry.is_dir():
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_orphaned_scratch()
    await database.init_db()
    await _cleanup_stale_jobs()
    yield


app = FastAPI(title="ProteaFlow", lifespan=lifespan)
app.add_middleware(AuthMiddleware)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(pages.router)
app.include_router(api_gpus.router)
app.include_router(api_jobs.router)
app.include_router(api_results.router)
app.include_router(api_stream.router)
app.include_router(api_manager.router)
app.include_router(api_geometry.router)
app.include_router(api_preview.router)
