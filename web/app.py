"""ProteaFlow Web — FastAPI application."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .config import STATIC_DIR, UPLOADS_DIR
from . import database
from .routes import pages, api_gpus, api_jobs, api_results, api_stream, api_manager, api_geometry


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect to /login if no proteaflow_user cookie.

    Skips /login, /static, and /api paths.
    """
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/login") or path.startswith("/static") or path.startswith("/api"):
            return await call_next(request)
        username = request.cookies.get("proteaflow_user")
        if not username:
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
            # 1. rankings.csv exists (pipeline finished ranking)
            # 2. Log contains completion markers
            out_dir = job.get("out_dir", "")
            if out_dir and (Path(out_dir) / "rankings.csv").exists():
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
    except (OSError, ProcessLookupError):
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
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
