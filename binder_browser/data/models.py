"""Data models for binder design results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd


@dataclass
class RunInfo:
    """Metadata about a single design run."""
    name: str
    path: Path
    rankings_csv: Path
    dashboard_png: Optional[Path] = None
    n_designs: int = 0
    best_combined: float = 0.0
    best_iptm: float = 0.0
    tools: list = field(default_factory=list)

    def __str__(self):
        return self.name


@dataclass
class DesignResult:
    """A single binder design with all scores."""
    design_id: str
    tool: str
    rank: int
    combined_score: float
    data: dict = field(default_factory=dict)  # all CSV columns
    run_path: Path = None

    @property
    def sequence(self) -> str:
        return self.data.get("binder_sequence", "")

    def find_structure(self) -> Optional[Path]:
        """Find the best available structure file for this design."""
        if self.run_path is None:
            return None

        # Check top_designs/ first (ranked CIF files)
        top_dir = self.run_path / "top_designs"
        if top_dir.is_dir():
            # Per-tool subfolder
            tool_dir = top_dir / self.tool
            if tool_dir.is_dir():
                for f in sorted(tool_dir.iterdir()):
                    if self.design_id in f.name and f.suffix in (".cif", ".pdb"):
                        return f
            # Top-level ranked files
            for f in sorted(top_dir.iterdir()):
                if self.design_id in f.name and f.suffix in (".cif", ".pdb"):
                    return f

        # Check validation/boltz/
        boltz_dir = self.run_path / "validation" / "boltz" / self.design_id
        if boltz_dir.is_dir():
            cifs = list(boltz_dir.rglob("*.cif"))
            if cifs:
                return cifs[0]

        return None
