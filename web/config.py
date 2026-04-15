"""Web app configuration — paths, constants, reuse from binder_browser."""

import os
import sys
import types
from pathlib import Path

# Stub PyQt6 so binder_browser.config imports cleanly
_qt_stub = types.ModuleType("PyQt6")
_qt_core = types.ModuleType("PyQt6.QtCore")
_qt_core.QSettings = lambda *a, **k: None
sys.modules.setdefault("PyQt6", _qt_stub)
sys.modules.setdefault("PyQt6.QtCore", _qt_core)

from binders_pipeline_env.binder_browser.config import (
    TOOL_COLORS,
    TOOL_PREFIXES,
    DEFAULT_COLUMNS,
    SCORE_COLUMNS_HIGHER_BETTER,
    SCORE_COLUMNS_LOWER_BETTER,
    SCATTER_PRESETS,
)

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # protein_folding/
WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
import socket as _socket
from binders_pipeline_env.config_loader import cfg as _cfg

_HOST = _socket.gethostname().split(".")[0]
_SHARED_DIR = Path(_cfg.shared_data_dir)
_SHARED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = _SHARED_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = _SHARED_DIR / f"proteaflow_{_HOST}.db"
SHARED_JSON = _SHARED_DIR / "shared.json"
HOSTNAME = _HOST

PIPELINE_SCRIPT = PROJECT_ROOT / "binders_pipeline_env" / "generate_binders.py"
RERANK_SCRIPT = PROJECT_ROOT / "binders_pipeline_env" / "rerank_binders.py"
OUTPUTS_BASE = Path(_cfg.outputs_dir)

CONDA_ENV = os.environ.get("CONDA_PREFIX", _cfg.conda_env("boltz"))
PYTHON_BIN = sys.executable

# ---------- defaults for launch form ----------
DEFAULT_TOOLS = ["rfdiffusion", "rfdiffusion3", "boltzgen", "bindcraft", "pxdesign", "proteina", "proteina_complexa"]
DEFAULT_MODE = "test"
DEFAULT_SS_BIAS = "balanced"
DEFAULT_SCORE_WEIGHTS = "0.3,0.6,0.1"
DEFAULT_LENGTH = "60-80"

MODE_ESTIMATES = {
    "test": "~4 hours (1 GPU)",
    "standard": "~37 hours (1 GPU)",
    "production": "~5 days (1 GPU)",
}

TOOL_DESCRIPTIONS = {
    "rfdiffusion": "Backbone diffusion design",
    "rfdiffusion3": "All-atom diffusion design (Foundry)",
    "boltzgen": "Diffusion-based binder design",
    "bindcraft": "Iterative AF2-based binder design",
    "pxdesign": "DiT diffusion + AF2-IG design",
    "proteina": "Flow-based backbone generation",
    "proteina_complexa": "Target-conditioned full-atom design",
}

# ── Molecule design constants ────────────────────────────────────────────────
MOLECULE_SCRIPT = PROJECT_ROOT / "binders_pipeline_env" / "generate_molecules.py"
MOLECULE_OUTPUTS_BASE = Path("/Colossus/software/webserver/outputs_mol")

MOLECULE_TOOLS = ["pocketflow", "molcraft", "pocketxmol"]
MOLECULE_TOOL_COLORS = {
    "pocketflow": "#E53935",
    "molcraft":   "#7C3AED",
    "pocketxmol": "#0EA5E9",
    "library":    "#FF9800",
}
MOLECULE_TOOL_DESCRIPTIONS = {
    "pocketflow": "Autoregressive flow-based 3D generation",
    "molcraft":   "Bayesian flow network, ICML 2024 (fast, high validity)",
    "pocketxmol": "Foundation model, Cell 2026 (SOTA, from Pocket2Mol authors)",
}
MOLECULE_MODE_ESTIMATES = {
    "test": "~10 min (1 GPU)",
    "standard": "~2-4 hours (1 GPU)",
    "production": "~1-2 days (1 GPU)",
}
MOLECULE_DEFAULT_TOOLS = ["pocketflow", "molcraft", "pocketxmol"]
MOLECULE_SCORE_WEIGHTS = "0.40,0.35,0.15,0.10"  # Vina, QED, SA, PocketFit
RERANK_MOLECULE_SCRIPT = PROJECT_ROOT / "binders_pipeline_env" / "rerank_molecules.py"

# Available compound libraries — paths resolved from config.yaml (libraries_dir)
_lib = _cfg.web("libraries_dir")
COMPOUND_LIBRARIES = {
    "": "(none — de novo only)",
    f"{_lib}/zinc_fda_approved/zinc_fda_approved.smi": "ZINC FDA Approved (103 drugs)",
    f"{_lib}/drugbank_open/drugbank_open_structures.smi": "DrugBank Open (12,309 compounds)",
    f"{_lib}/ppi_compounds/ppi_compounds_combined.smi": "PPI Compounds (5,001)",
    f"{_lib}/ppi_compounds/ppi_500.smi": "PPI Test Set (500)",
    f"{_lib}/zinc_fragments/zinc_fragments_combined.smi": "ZINC Fragments (5,000)",
}
