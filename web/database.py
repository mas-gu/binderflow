"""SQLite job database (async via aiosqlite)."""

import json
import time
import uuid
from pathlib import Path

import aiosqlite

from .config import DB_PATH, SHARED_JSON, HOSTNAME, _SHARED_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    job_type    TEXT DEFAULT 'generate',
    status      TEXT DEFAULT 'queued',
    gpu_id      INTEGER,
    pid         INTEGER,
    target_file TEXT,
    site        TEXT,
    length      TEXT,
    tools       TEXT,
    mode        TEXT,
    params_json TEXT,
    out_dir     TEXT,
    log_file    TEXT,
    created_at  REAL,
    started_at  REAL,
    finished_at REAL,
    exit_code   INTEGER,
    error_msg   TEXT,
    username    TEXT DEFAULT '',
    project     TEXT DEFAULT '',
    machine     TEXT DEFAULT ''
);
"""

MIGRATIONS = [
    "ALTER TABLE jobs ADD COLUMN username TEXT DEFAULT ''",
    "ALTER TABLE jobs ADD COLUMN project TEXT DEFAULT ''",
    "ALTER TABLE jobs ADD COLUMN machine TEXT DEFAULT ''",
    "ALTER TABLE jobs ADD COLUMN archived INTEGER DEFAULT 0",
]

# Separate migration that needs dedup before index creation
_DEDUP_INDEX_SQL = """
DELETE FROM jobs WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY out_dir ORDER BY created_at DESC
        ) AS rn
        FROM jobs
        WHERE out_dir IS NOT NULL AND out_dir != ''
    ) WHERE rn = 1
) AND out_dir IS NOT NULL AND out_dir != '';
"""
_UNIQUE_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_out_dir "
    "ON jobs(out_dir) WHERE out_dir IS NOT NULL AND out_dir != ''"
)


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    return db


async def init_db():
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript(SCHEMA)
        # Run migrations for existing DBs (ignore if columns already exist)
        for migration in MIGRATIONS:
            try:
                await db.execute(migration)
            except Exception:
                pass
        # Deduplicate out_dir values and create unique index
        try:
            # Check if index already exists
            cursor = await db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_jobs_out_dir'"
            )
            if not await cursor.fetchone():
                await db.executescript(_DEDUP_INDEX_SQL)
                await db.execute(_UNIQUE_INDEX_SQL)
        except Exception:
            pass
        await db.commit()


async def create_job(
    name: str,
    job_type: str,
    gpu_id: int,
    target_file: str,
    site: str,
    length: str,
    tools: str,
    mode: str,
    params: dict,
    out_dir: str,
    log_file: str,
    username: str = "",
    project: str = "",
) -> str:
    job_id = uuid.uuid4().hex[:12]
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO jobs
            (id, name, job_type, status, gpu_id, target_file, site, length,
             tools, mode, params_json, out_dir, log_file, created_at, username, project, machine)
            VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, name, job_type, gpu_id, target_file, site, length,
                tools, mode, json.dumps(params), out_dir, log_file, time.time(),
                username, project, HOSTNAME,
            ),
        )
        await db.commit()
    finally:
        await db.close()
    return job_id


async def update_job(job_id: str, **fields):
    db = await get_db()
    try:
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [job_id]
        await db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", vals)
        await db.commit()
    finally:
        await db.close()


async def get_job(job_id: str) -> dict | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def list_jobs(status: str | None = None, limit: int = 50,
                    username: str | None = None, project: str | None = None,
                    include_archived: bool = False) -> list[dict]:
    db = await get_db()
    try:
        conditions = []
        params = []
        if not include_archived:
            conditions.append("(archived IS NULL OR archived = 0)")
        if status:
            conditions.append("status = ?")
            params.append(status)
        if username:
            conditions.append("username = ?")
            params.append(username)
        if project:
            conditions.append("project = ?")
            params.append(project)
        where = " AND ".join(conditions)
        query = "SELECT * FROM jobs"
        if where:
            query += " WHERE " + where
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def list_projects() -> list[str]:
    """List all projects across all machines."""
    projects = set()
    for db_path in _iter_all_dbs():
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT DISTINCT project FROM jobs WHERE project != ''"
                )
                rows = await cursor.fetchall()
                projects.update(r[0] for r in rows)
        except Exception:
            pass
    # Also include projects from shared.json
    projects.update(_read_shared().get("projects", []))
    return sorted(projects)


async def list_all_jobs(status: str | None = None, limit: int = 100,
                        username: str | None = None, project: str | None = None,
                        include_archived: bool = False) -> list[dict]:
    """List jobs from ALL machines (read-only cross-machine query)."""
    all_jobs = []
    for db_path in _iter_all_dbs():
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                conditions = []
                params = []
                if not include_archived:
                    conditions.append("(archived IS NULL OR archived = 0)")
                if status:
                    conditions.append("status = ?")
                    params.append(status)
                if username:
                    conditions.append("username = ?")
                    params.append(username)
                if project:
                    conditions.append("project = ?")
                    params.append(project)
                where = " AND ".join(conditions)
                query = "SELECT * FROM jobs"
                if where:
                    query += " WHERE " + where
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                for r in rows:
                    d = dict(r)
                    # Tag with machine name from filename if not set
                    if not d.get("machine"):
                        d["machine"] = db_path.stem.replace("binderflow_", "")
                    all_jobs.append(d)
        except Exception:
            pass
    # Sort by created_at descending, limit total
    all_jobs.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    return all_jobs[:limit]


async def get_job_any_machine(job_id: str) -> dict | None:
    """Find a job by ID across all machine DBs."""
    # Try local first (fast path)
    job = await get_job(job_id)
    if job:
        return job
    # Try other machines
    for db_path in _iter_all_dbs():
        if db_path == DB_PATH:
            continue
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
                row = await cursor.fetchone()
                if row:
                    d = dict(row)
                    if not d.get("machine"):
                        d["machine"] = db_path.stem.replace("binderflow_", "")
                    return d
        except Exception:
            pass
    return None


async def list_all_out_dirs() -> set[str]:
    """Collect all out_dir values across all machine DBs."""
    dirs = set()
    for db_path in _iter_all_dbs():
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT out_dir FROM jobs WHERE out_dir IS NOT NULL AND out_dir != ''"
                )
                rows = await cursor.fetchall()
                dirs.update(r[0] for r in rows if r[0])
        except Exception:
            pass
    return dirs


async def delete_job(job_id: str) -> bool:
    """Hard-delete a job row from the local DB."""
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def job_exists_for_path(out_dir: str) -> bool:
    """Check if any DB has a job with this out_dir (normalized)."""
    target = str(Path(out_dir).resolve())
    for db_path in _iter_all_dbs():
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute(
                    "SELECT out_dir FROM jobs WHERE out_dir IS NOT NULL AND out_dir != ''"
                )
                rows = await cursor.fetchall()
                for r in rows:
                    if r[0] and str(Path(r[0]).resolve()) == target:
                        return True
        except Exception:
            pass
    return False


def _iter_all_dbs():
    """Yield all binderflow_*.db files in the shared directory."""
    return sorted(_SHARED_DIR.glob("binderflow_*.db"))


def _read_shared() -> dict:
    """Read the shared.json file."""
    if SHARED_JSON.exists():
        try:
            return json.loads(SHARED_JSON.read_text())
        except Exception:
            pass
    return {"projects": [], "users": []}


def _write_shared(data: dict):
    """Write the shared.json file."""
    SHARED_JSON.write_text(json.dumps(data, indent=2))


def add_shared_project(project: str):
    """Add a project to the shared list."""
    data = _read_shared()
    if project and project not in data.get("projects", []):
        data.setdefault("projects", []).append(project)
        data["projects"].sort()
        _write_shared(data)


def add_shared_user(username: str):
    """Add a user to the shared list."""
    data = _read_shared()
    if username and username not in data.get("users", []):
        data.setdefault("users", []).append(username)
        data["users"].sort()
        _write_shared(data)
