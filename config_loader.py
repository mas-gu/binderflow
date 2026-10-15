"""ProteaFlow configuration loader.

Loads paths and settings from config.yaml with environment variable overrides.

Priority chain (highest to lowest):
    1. Environment variables (e.g., BINDER_SOFTWARE_DIR)
    2. config.yaml (in the same directory as this file)
    3. Generic defaults (~/software, ~/weights)

Usage:
    from config_loader import cfg
    print(cfg.software_dir)
    print(cfg.conda_env("esmfold"))
    print(cfg.tool("rfdiffusion_dir"))
"""

import os
import re
from pathlib import Path
from typing import Optional

import yaml


_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

_GENERIC_DEFAULTS = {
    "software_dir": os.path.expanduser("~/software"),
    "weights_dir": os.path.expanduser("~/weights"),
}


def _expand(value: str, context: dict) -> str:
    """Expand ~ and ${var} references in a string value."""
    if not isinstance(value, str):
        return value
    # Expand ${software_dir}, ${weights_dir}, etc.
    def _repl(m):
        key = m.group(1)
        return context.get(key, m.group(0))
    result = re.sub(r'\$\{(\w+)\}', _repl, value)
    # Expand ~
    result = os.path.expanduser(result)
    return result


class ProteaFlowConfig:
    """Central configuration for all ProteaFlow paths and settings."""

    def __init__(self):
        self._raw = {}
        self._load()

    def _load(self):
        """Load config.yaml if it exists."""
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH) as f:
                self._raw = yaml.safe_load(f) or {}

        # Resolve base directories (env var > config > default)
        self.software_dir = os.environ.get(
            "BINDER_SOFTWARE_DIR",
            self._raw.get("software_dir", _GENERIC_DEFAULTS["software_dir"]))
        self.software_dir = os.path.expanduser(self.software_dir)

        self.weights_dir = os.environ.get(
            "BINDER_WEIGHTS_DIR",
            self._raw.get("weights_dir", _GENERIC_DEFAULTS["weights_dir"]))
        self.weights_dir = os.path.expanduser(self.weights_dir)

        # Context for ${var} expansion
        self._context = {
            "software_dir": self.software_dir,
            "weights_dir": self.weights_dir,
        }

        # Resolve tool paths
        self._tools = {}
        for key, val in self._raw.get("tools", {}).items():
            self._tools[key] = _expand(str(val), self._context)

        # Resolve conda env paths
        self._conda_envs = {}
        for key, val in self._raw.get("conda_envs", {}).items():
            # _expand() runs on tools: and web: but was skipped here, so a
            # conda_envs entry written as ${software_dir}/envs/foo stayed literal
            # and conda_run_args() passed the raw "${software_dir}/..." string
            # straight to `conda run`.
            self._conda_envs[key] = os.path.expanduser(
                _expand(str(val), self._context))

        # Web settings
        self._web = {}
        for key, val in self._raw.get("web", {}).items():
            self._web[key] = _expand(str(val), self._context)

    def tool(self, key: str, env_var: Optional[str] = None,
             default: Optional[str] = None) -> str:
        """Get a tool path. Priority: env_var > config > default."""
        if env_var:
            val = os.environ.get(env_var)
            if val:
                return val
        val = self._tools.get(key)
        if val:
            return val
        if default:
            return os.path.expanduser(default)
        import logging
        logging.getLogger(__name__).warning(
            f"Tool '{key}' not configured (no env var, config, or default)")
        return ""

    def conda_env(self, name: str) -> str:
        """Get conda env name or prefix path for a tool.

        Returns the value from config.yaml, or the name itself as fallback.
        If the value is an absolute path, it should be used with `conda run -p`.
        If it's a name, use `conda run -n`.
        """
        return self._conda_envs.get(name, name)

    def conda_env_prefix(self, name: str) -> str:
        """Absolute PREFIX PATH for a tool env, resolving a bare name if needed.

        `conda_env()` may legitimately return either a name or a prefix, and
        `conda run` handles both via `conda_run_args()`. But several callers use
        the value as a PATH — to build `<env>/bin/<binary>`, to set CUDA_HOME, or
        to test `Path(env).is_dir()` in --capabilities. Handed a bare name (which
        is exactly what the shipped config.example.yaml contains, e.g.
        `rfdiffusion3: rfdiffusion3`) those silently produced relative paths like
        `rfdiffusion3/bin/python`, so a fresh clone looked mis-installed.

        Resolves via `conda env list`, cached. Returns "" if it cannot be found,
        which callers already treat as unavailable.
        """
        env = self.conda_env(name)
        if not env:
            return ""
        if os.path.isabs(env):
            return env
        cache = getattr(self, "_env_prefix_cache", None)
        if cache is None:
            cache = {}
            try:
                import subprocess, json as _json
                out = subprocess.run(["conda", "env", "list", "--json"],
                                     capture_output=True, text=True, timeout=30)
                if out.returncode == 0:
                    for p in _json.loads(out.stdout).get("envs", []):
                        cache[os.path.basename(p)] = p
            except Exception:
                pass
            self._env_prefix_cache = cache
        return cache.get(env, "")

    def conda_run_args(self, name: str) -> list:
        """Build conda run prefix args for a tool env.

        Returns ["-p", "/path/to/env"] for prefix envs or ["-n", "envname"] for named envs.
        """
        env = self.conda_env(name)
        if os.path.isabs(env):
            return ["-p", env]
        return ["-n", env]

    def web(self, key: str, default: str = "") -> str:
        """Get a web server config value."""
        return self._web.get(key, default)

    @property
    def outputs_dir(self) -> str:
        return self.web("outputs_dir", "./outputs")

    @property
    def shared_data_dir(self) -> str:
        return self.web("shared_data_dir", "./data")


# Singleton — import and use directly
cfg = ProteaFlowConfig()
