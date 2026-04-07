"""GPU status API."""

from dataclasses import asdict

from fastapi import APIRouter

from ..gpu import query_gpus
from .. import database

router = APIRouter(prefix="/api", tags=["gpus"])


@router.get("/gpus")
async def get_gpus():
    gpus = query_gpus()
    running_jobs = await database.list_jobs(status="running")

    gpu_dicts = []
    for gpu in gpus:
        d = asdict(gpu)
        # Check if a tracked job is using this GPU
        for job in running_jobs:
            if job["gpu_id"] == gpu.index:
                d["busy"] = True
                d["job_id"] = job["id"]
                d["job_name"] = job["name"]
                break
        d["memory_pct"] = round(gpu.memory_used_mb / gpu.memory_total_mb * 100) if gpu.memory_total_mb else 0
        gpu_dicts.append(d)
    return gpu_dicts
