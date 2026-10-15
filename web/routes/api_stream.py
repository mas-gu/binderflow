"""Server-Sent Events for live log streaming."""

import asyncio
from pathlib import Path

import aiofiles
from fastapi import APIRouter
from starlette.responses import StreamingResponse

from .. import database

router = APIRouter(prefix="/api", tags=["stream"])


async def _tail_log(job_id: str):
    """Async generator that tails a log file and yields SSE events."""
    job = await database.get_job(job_id)
    if not job or not job["log_file"]:
        yield "data: Log file not found\n\n"
        return

    log_path = Path(job["log_file"])

    # Wait for log file to appear
    for _ in range(30):
        if log_path.exists():
            break
        await asyncio.sleep(1)
    else:
        yield "data: Log file not created\n\n"
        return

    # The 0.5 s poll below exists for LOG responsiveness. Job status changes
    # maybe twice in a run, but the status query ran on every idle pass — and
    # database.get_db() opens a fresh SQLite connection and issues a PRAGMA each
    # time, so a single open browser tab drove ~2 connect/PRAGMA/close cycles a
    # second for the whole run (~36k over 5 hours). It closes properly, so this
    # was churn rather than a leak. Check status every 4th idle pass (~2 s) and
    # leave log latency untouched.
    _idle = 0
    _STATUS_EVERY = 4
    async with aiofiles.open(log_path, "r") as f:
        while True:
            line = await f.readline()
            if line:
                _idle = 0
                # Escape for SSE (multi-line messages need data: prefix per line)
                for sub in line.rstrip("\n").split("\n"):
                    yield f"data: {sub}\n"
                yield "\n"
            else:
                _idle += 1
                if _idle % _STATUS_EVERY:
                    await asyncio.sleep(0.5)
                    continue
                # Check if job is still running
                job = await database.get_job(job_id)
                if job and job["status"] not in ("queued", "running"):
                    # Read any remaining lines
                    remaining = await f.read()
                    if remaining:
                        for rline in remaining.splitlines():
                            yield f"data: {rline}\n\n"
                    yield f"event: status\ndata: {job['status']}\n\n"
                    return
                await asyncio.sleep(0.5)


@router.get("/stream/{job_id}")
async def stream_log(job_id: str):
    return StreamingResponse(
        _tail_log(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
