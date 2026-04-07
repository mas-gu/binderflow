"""GPU status via nvidia-smi."""

import subprocess
import time
from dataclasses import dataclass

_cache: list = []
_cache_time: float = 0.0
_CACHE_TTL = 5.0


@dataclass
class GPUStatus:
    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int
    temperature: int
    busy: bool = False
    job_id: str | None = None
    job_name: str | None = None


def query_gpus() -> list[GPUStatus]:
    """Parse nvidia-smi output. Cached for 5 seconds."""
    global _cache, _cache_time
    now = time.time()
    if _cache and (now - _cache_time) < _CACHE_TTL:
        return _cache

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpu = GPUStatus(
                index=int(parts[0]),
                name=parts[1],
                memory_used_mb=int(parts[2]),
                memory_total_mb=int(parts[3]),
                utilization_pct=int(parts[4]),
                temperature=int(parts[5]),
                busy=int(parts[2]) > 500,
            )
            gpus.append(gpu)
        _cache = gpus
        _cache_time = now
        return gpus
    except Exception:
        return _cache or []


def invalidate_cache():
    global _cache_time
    _cache_time = 0.0
