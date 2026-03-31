"""Detail view for a selected design: scores, sequence, structure launch."""

import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGridLayout, QGroupBox, QTextEdit, QApplication, QProgressBar,
    QScrollArea, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from ..config import TOOL_COLORS
from ..data.models import DesignResult


# Score cards: (label, csv_key, format, higher_better)
SCORE_CARDS = [
    ("Combined Score", "combined_score", ".4f", True),
    ("Boltz iPTM", "boltz_iptm", ".4f", True),
    ("Boltz pLDDT", "boltz_binder_plddt", ".1f", True),
    ("ESMFold pLDDT", "esmfold_plddt", ".1f", True),
    ("pDockQ", "pDockQ", ".3f", True),
    ("Rosetta dG", "rosetta_dG", ".1f", False),
    ("Rosetta SC", "rosetta_sc", ".3f", True),
    ("Rosetta SAP", "rosetta_sap", ".1f", False),
    ("Refolding RMSD", "refolding_rmsd", ".2f", False),
    ("SIF", "site_interface_fraction", ".3f", True),
    ("Centroid Dist", "site_centroid_dist_CA", ".1f", False),
    ("Cos Angle", "site_cos_angle", ".3f", True),
    ("Interface KE", "interface_KE_fraction", ".3f", False),
    ("NetSolP", "netsolp_solubility", ".3f", True),
    ("Site Mean PAE", "boltz_site_mean_pae", ".2f", False),
    ("Rosetta DSASA", "rosetta_dsasa", ".0f", True),
]


class DesignDetail(QWidget):
    """Detailed view of a single selected design."""

    design_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_path = None
        self._current_data = None
        self._ranked_designs = []  # list of dicts sorted by rank
        self._setup_ui()

    def set_data(self, df, run_path=None):
        """Set the full dataframe for rank navigation."""
        import pandas as pd
        self._run_path = run_path
        ranked = df[df["rank"].notna() & (df["rank"] > 0)].sort_values("rank")
        self._ranked_designs = [row.to_dict() for _, row in ranked.iterrows()]

        self.rank_combo.blockSignals(True)
        self.rank_combo.clear()
        self.rank_combo.addItem("Select by rank...")
        for d in self._ranked_designs[:200]:  # top 200
            rank = int(d.get("rank", 0))
            did = d.get("design_id", "?")
            score = d.get("combined_score", 0)
            try:
                score_str = f"{float(score):.3f}"
            except (ValueError, TypeError):
                score_str = "?"
            tool = d.get("tool", "?")
            self.rank_combo.addItem(f"#{rank}  {did}  ({score_str})  [{tool}]")
        self.rank_combo.blockSignals(False)

    def _on_rank_selected(self, index):
        if index <= 0 or index > len(self._ranked_designs):
            return
        data = self._ranked_designs[index - 1]
        self.show_design(data)
        self.design_selected.emit(data)

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Rank navigation dropdown
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Navigate:"))
        self.rank_combo = QComboBox()
        self.rank_combo.setMinimumWidth(400)
        self.rank_combo.addItem("Select by rank...")
        self.rank_combo.currentIndexChanged.connect(self._on_rank_selected)
        nav_layout.addWidget(self.rank_combo)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        # Header
        self.header_label = QLabel("No design selected")
        self.header_label.setFont(QFont("", 12, QFont.Weight.Bold))
        layout.addWidget(self.header_label)

        # Tool badge
        self.tool_label = QLabel("")
        self.tool_label.setFont(QFont("", 10))
        layout.addWidget(self.tool_label)

        # Score card grid
        scores_group = QGroupBox("Scores")
        self.scores_grid = QGridLayout()
        scores_group.setLayout(self.scores_grid)
        self._score_labels = {}
        for i, (label, key, fmt, higher) in enumerate(SCORE_CARDS):
            row, col = divmod(i, 2)
            name_lbl = QLabel(f"{label}:")
            name_lbl.setFont(QFont("", 9, QFont.Weight.Bold))
            val_lbl = QLabel("--")
            val_lbl.setFont(QFont("Monospace", 10))
            self.scores_grid.addWidget(name_lbl, row, col * 2)
            self.scores_grid.addWidget(val_lbl, row, col * 2 + 1)
            self._score_labels[key] = val_lbl
        layout.addWidget(scores_group)

        # SS composition
        ss_group = QGroupBox("Secondary Structure")
        ss_layout = QVBoxLayout()
        self.ss_label = QLabel("Helix: -- | Sheet: -- | Loop: --")
        self.ss_label.setFont(QFont("Monospace", 9))
        ss_layout.addWidget(self.ss_label)
        self.helix_bar = QProgressBar()
        self.helix_bar.setFormat("Helix %p%")
        self.helix_bar.setMaximum(100)
        ss_layout.addWidget(self.helix_bar)
        self.sheet_bar = QProgressBar()
        self.sheet_bar.setFormat("Sheet %p%")
        self.sheet_bar.setMaximum(100)
        ss_layout.addWidget(self.sheet_bar)
        self.loop_bar = QProgressBar()
        self.loop_bar.setFormat("Loop %p%")
        self.loop_bar.setMaximum(100)
        ss_layout.addWidget(self.loop_bar)
        ss_group.setLayout(ss_layout)
        layout.addWidget(ss_group)

        # Sequence
        seq_group = QGroupBox("Binder Sequence")
        seq_layout = QVBoxLayout()
        self.seq_text = QTextEdit()
        self.seq_text.setReadOnly(True)
        self.seq_text.setFont(QFont("Monospace", 10))
        self.seq_text.setMaximumHeight(100)
        seq_layout.addWidget(self.seq_text)
        self.seq_len_label = QLabel("Length: --")
        seq_layout.addWidget(self.seq_len_label)
        seq_group.setLayout(seq_layout)
        layout.addWidget(seq_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.pymol_btn = QPushButton("Open in PyMOL")
        self.pymol_btn.clicked.connect(self._open_pymol)
        self.pymol_btn.setEnabled(False)
        btn_layout.addWidget(self.pymol_btn)

        self.copy_btn = QPushButton("Copy Sequence")
        self.copy_btn.clicked.connect(self._copy_sequence)
        self.copy_btn.setEnabled(False)
        btn_layout.addWidget(self.copy_btn)

        copy_id_btn = QPushButton("Copy Design ID")
        copy_id_btn.clicked.connect(self._copy_design_id)
        btn_layout.addWidget(copy_id_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Structure path
        self.struct_label = QLabel("")
        self.struct_label.setFont(QFont("", 8))
        self.struct_label.setWordWrap(True)
        layout.addWidget(self.struct_label)

        layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_run_path(self, path: Path):
        self._run_path = path

    def show_design(self, row_data: dict):
        self._current_data = row_data
        design_id = str(row_data.get("design_id", "?"))
        tool = str(row_data.get("tool", "unknown"))
        rank = row_data.get("rank", "?")

        self.header_label.setText(f"#{rank}  {design_id}")
        color = TOOL_COLORS.get(tool, "#888888")
        self.tool_label.setText(f"Tool: {tool}")
        self.tool_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        # Scores
        for key, label in self._score_labels.items():
            val = row_data.get(key)
            if val is not None and str(val) != "" and str(val) != "nan":
                try:
                    fmt = next(f for l, k, f, h in SCORE_CARDS if k == key)
                    label.setText(f"{float(val):{fmt}}")
                except (ValueError, StopIteration):
                    label.setText(str(val))
            else:
                label.setText("--")

        # SS composition
        h = row_data.get("binder_helix_frac")
        s = row_data.get("binder_sheet_frac")
        l = row_data.get("binder_loop_frac")
        try:
            h_val = float(h) if h and str(h) != "nan" else 0
            s_val = float(s) if s and str(s) != "nan" else 0
            l_val = float(l) if l and str(l) != "nan" else 0
        except (ValueError, TypeError):
            h_val = s_val = l_val = 0
        self.ss_label.setText(f"Helix: {h_val:.1%} | Sheet: {s_val:.1%} | Loop: {l_val:.1%}")
        self.helix_bar.setValue(int(h_val * 100))
        self.sheet_bar.setValue(int(s_val * 100))
        self.loop_bar.setValue(int(l_val * 100))

        # Sequence
        seq = str(row_data.get("binder_sequence", ""))
        self.seq_text.setPlainText(seq)
        self.seq_len_label.setText(f"Length: {len(seq)} aa")
        self.copy_btn.setEnabled(bool(seq))

        # Structure
        dr = DesignResult(
            design_id=design_id, tool=tool, rank=0,
            combined_score=0, run_path=self._run_path,
        )
        struct = dr.find_structure()
        if struct:
            self.struct_label.setText(f"Structure: {struct}")
            self.pymol_btn.setEnabled(True)
        else:
            self.struct_label.setText("No structure file found")
            self.pymol_btn.setEnabled(False)

    def _open_pymol(self):
        if not self._current_data or not self._run_path:
            return
        design_id = str(self._current_data.get("design_id", ""))
        tool = str(self._current_data.get("tool", "unknown"))
        dr = DesignResult(
            design_id=design_id, tool=tool, rank=0,
            combined_score=0, run_path=self._run_path,
        )
        struct = dr.find_structure()
        if struct and struct.exists():
            subprocess.Popen(["pymol", str(struct)], cwd=str(struct.parent))

    def _copy_sequence(self):
        if self._current_data:
            seq = str(self._current_data.get("binder_sequence", ""))
            QApplication.clipboard().setText(seq)

    def _copy_design_id(self):
        if self._current_data:
            QApplication.clipboard().setText(str(self._current_data.get("design_id", "")))
