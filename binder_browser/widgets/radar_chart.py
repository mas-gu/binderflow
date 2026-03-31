"""Developability radar chart — multi-axis profile comparison."""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                              QLabel, QLineEdit, QCompleter, QPushButton,
                              QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..config import TOOL_COLORS

# Radar axes: (display_name, column, higher_is_better, min_val, max_val, description)
RADAR_AXES = [
    ("Binding (iPTM)\n0→1 | good >0.8", "boltz_iptm", True, 0.0, 1.0,
     "Interface predicted TM-score from Boltz-2.\n"
     "Good: > 0.8 | Excellent: > 0.9 | Poor: < 0.5"),
    ("Structure (Sc)\n0→1 | good >0.55", "rosetta_sc", True, 0.0, 1.0,
     "Rosetta shape complementarity — geometric fit between surfaces.\n"
     "Natural interfaces: 0.64-0.68 | Good designed: > 0.55 | Poor: < 0.45"),
    ("Stability (RMSD)\n0→5 Å | good <2.0", "refolding_rmsd", False, 0.0, 5.0,
     "CA RMSD between binder alone (ESMFold) vs in complex (Boltz-2).\n"
     "Stable fold: < 2.0 Å | Acceptable: < 3.0 Å | Target-dependent: > 4.0 Å"),
    ("Docking (pDockQ)\n0→1 | good >0.8", "pDockQ", True, 0.0, 1.0,
     "Predicted docking quality from iPTM × iPLDDT.\n"
     "High confidence: > 0.8 | Medium: 0.5-0.8 | Low: < 0.3"),
    ("Low KE\n0→100% | good <20%", "interface_KE_fraction", False, 0.0, 1.0,
     "Lysine + Glutamate fraction at the binder-target interface.\n"
     "Good: < 20% | Acceptable: < 25% | Risky: > 30% (non-specific binding)"),
    ("Low Agg (SAP)\n0→300 | good <80", "rosetta_sap", False, 0.0, 300.0,
     "Surface Aggregation Propensity — hydrophobic exposure.\n"
     "Lower is better. Good: < 80 | Moderate: 80-150 | High risk: > 200"),
    ("Solubility (NetSolP)\n0→1 | good >0.7", "netsolp_solubility", True, 0.0, 1.0,
     "Predicted E.coli solubility (NetSolP, ESM-based).\n"
     "Soluble: > 0.7 | Borderline: 0.4-0.7 | Insoluble: < 0.4"),
    ("Site Focus (SIF)\n0→1 | good >0.5", "site_interface_fraction", True, 0.0, 1.0,
     "Fraction of binder interface at the specified binding site.\n"
     "On-site: > 0.5 | Partial: 0.25-0.5 | Off-site: < 0.2"),
]


class RadarChart(QWidget):
    """6-axis developability radar for tier comparison or single design."""

    design_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._active_axes = RADAR_AXES
        self._selected_design = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Background:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["None (design only)", "Tiers (1 vs 2 vs 3)", "Tools (median)", "Top 10 vs All"])
        self.mode_combo.currentTextChanged.connect(self._replot)
        controls.addWidget(self.mode_combo)

        controls.addWidget(QLabel("  Design:"))
        self.design_input = QLineEdit()
        self.design_input.setPlaceholderText("Type design_id or rank...")
        self.design_input.setMinimumWidth(200)
        self.design_input.returnPressed.connect(self._on_design_entered)
        controls.addWidget(self.design_input)

        self.rank_combo = QComboBox()
        self.rank_combo.setMinimumWidth(100)
        self.rank_combo.addItem("None")
        self.rank_combo.currentTextChanged.connect(self._on_rank_selected)
        controls.addWidget(self.rank_combo)

        info_btn = QPushButton("? Scores Guide")
        info_btn.setFixedWidth(120)
        info_btn.clicked.connect(self._show_info)
        controls.addWidget(info_btn)

        controls.addStretch()
        layout.addLayout(controls)

        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def set_data(self, df: pd.DataFrame):
        self._df = df
        self._active_axes = self._get_available_axes(df)
        self._selected_design = None

        # Populate rank selector with top 50
        self.rank_combo.blockSignals(True)
        self.rank_combo.clear()
        self.rank_combo.addItem("None")
        if "rank" in df.columns and "design_id" in df.columns:
            ranked = df[df["rank"].notna() & (df["rank"] > 0)].nsmallest(50, "rank")
            for _, row in ranked.iterrows():
                self.rank_combo.addItem(f"#{int(row['rank'])} {row['design_id']}")
        self.rank_combo.blockSignals(False)

        # Setup autocomplete for design_id
        if "design_id" in df.columns:
            completer = QCompleter(df["design_id"].dropna().tolist())
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self.design_input.setCompleter(completer)

        self._replot()

    def _on_design_entered(self):
        text = self.design_input.text().strip()
        if not text or self._df.empty:
            self._selected_design = None
            self._replot()
            return
        # Try as design_id
        match = self._df[self._df["design_id"] == text]
        if match.empty:
            # Try as rank number
            try:
                rank_num = int(text.replace("#", ""))
                match = self._df[self._df["rank"] == rank_num]
            except ValueError:
                # Try partial match
                match = self._df[self._df["design_id"].str.contains(text, case=False, na=False)]
        if not match.empty:
            self._selected_design = match.iloc[0].to_dict()
            self._replot()

    def _on_rank_selected(self, text):
        if text == "None" or not text:
            self._selected_design = None
            self._replot()
            return
        # Parse "# 1 design_id"
        parts = text.split(" ", 1)
        if len(parts) >= 2:
            design_id = parts[1]
            match = self._df[self._df["design_id"] == design_id]
            if not match.empty:
                self._selected_design = match.iloc[0].to_dict()
                self._replot()

    def _normalize(self, values, col_info):
        _, _, higher_better, vmin, vmax = col_info[:5]
        clipped = np.clip(values, vmin, vmax)
        normed = (clipped - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        if not higher_better:
            normed = 1.0 - normed
        return normed

    def _get_available_axes(self, df):
        available = []
        for axis_info in RADAR_AXES:
            col = axis_info[1]
            if col in df.columns and df[col].notna().sum() > 0:
                available.append(axis_info)
        return available if len(available) >= 3 else RADAR_AXES

    def _get_radar_values(self, df_subset, axes=None):
        if axes is None:
            axes = self._active_axes
        values = []
        for axis_info in axes:
            col = axis_info[1]
            if col in df_subset.columns:
                med = df_subset[col].dropna().median()
                if pd.notna(med):
                    values.append(self._normalize(med, axis_info))
                else:
                    values.append(0.5)
            else:
                values.append(0.5)
        return values

    def _get_single_design_values(self, design_data, axes=None):
        """Get radar values for a single design dict."""
        if axes is None:
            axes = self._active_axes
        values = []
        for axis_info in axes:
            col = axis_info[1]
            val = design_data.get(col)
            if val is not None and pd.notna(val):
                try:
                    values.append(self._normalize(float(val), axis_info))
                except (ValueError, TypeError):
                    values.append(0.5)
            else:
                values.append(0.5)
        return values

    def _replot(self, _=None):
        self.figure.clear()
        df = self._df
        if df.empty:
            self.canvas.draw()
            return

        mode = self.mode_combo.currentText()
        active_axes = self._active_axes
        labels = [a[0] for a in active_axes]
        n_axes = len(labels)
        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]

        self.figure.patch.set_facecolor('#1a1a2e')

        ax = self.figure.add_subplot(111, polar=True)
        ax.set_facecolor('#16213e')
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color='#e0e0e0')
        ax.set_ylim(0, 1.05)
        # Dark theme for grid and spines
        ax.tick_params(colors='#e0e0e0')
        ax.xaxis.label.set_color('#e0e0e0')
        ax.yaxis.label.set_color('#e0e0e0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        ax.yaxis.grid(color='#444444', linestyle='-', linewidth=0.5)
        ax.xaxis.grid(color='#444444', linestyle='-', linewidth=0.5)

        # Background: none / tiers / tools / top10
        if mode == "None (design only)":
            pass  # Empty background — only selected design will show

        elif mode == "Tiers (1 vs 2 vs 3)":
            tier_colors = {1: "#4CAF50", 2: "#FFC107", 3: "#F44336"}
            tier_labels = {1: "Tier 1 (Top 7.5%)", 2: "Tier 2", 3: "Tier 3"}
            if "tier" in df.columns:
                for tier in [1, 2, 3]:
                    subset = df[df["tier"] == tier]
                    if subset.empty:
                        continue
                    vals = self._get_radar_values(subset, active_axes) + \
                           self._get_radar_values(subset, active_axes)[:1]
                    ax.plot(angles, vals, 'o-', linewidth=1.5, label=tier_labels[tier],
                            color=tier_colors[tier], markersize=3, alpha=0.7)
                    ax.fill(angles, vals, alpha=0.08, color=tier_colors[tier])

        elif mode == "Tools (median)":
            if "tool" in df.columns:
                for tool in sorted(df["tool"].unique()):
                    subset = df[df["tool"] == tool]
                    if len(subset) < 3:
                        continue
                    color = TOOL_COLORS.get(tool, "#888888")
                    vals = self._get_radar_values(subset, active_axes) + \
                           self._get_radar_values(subset, active_axes)[:1]
                    ax.plot(angles, vals, 'o-', linewidth=1.2, label=tool,
                            color=color, markersize=2, alpha=0.6)
                    ax.fill(angles, vals, alpha=0.05, color=color)

        elif mode == "Top 10 vs All":
            ranked = df[df["rank"].notna() & (df["rank"] > 0)].nsmallest(10, "rank")
            if not ranked.empty:
                vals_top = self._get_radar_values(ranked, active_axes) + \
                           self._get_radar_values(ranked, active_axes)[:1]
                vals_all = self._get_radar_values(df, active_axes) + \
                           self._get_radar_values(df, active_axes)[:1]
                ax.plot(angles, vals_top, 'o-', linewidth=1.5, label="Top 10",
                        color="#4CAF50", markersize=3, alpha=0.7)
                ax.fill(angles, vals_top, alpha=0.1, color="#4CAF50")
                ax.plot(angles, vals_all, 'o-', linewidth=1, label="All",
                        color="#9E9E9E", markersize=2, alpha=0.5)
                ax.fill(angles, vals_all, alpha=0.05, color="#9E9E9E")

        # Overlay selected design (bold, on top)
        if self._selected_design is not None:
            vals = self._get_single_design_values(self._selected_design, active_axes)
            vals_closed = vals + vals[:1]
            did = self._selected_design.get("design_id", "?")
            tool = self._selected_design.get("tool", "unknown")
            color = TOOL_COLORS.get(tool, "#FFFFFF")
            ax.plot(angles, vals_closed, 'o-', linewidth=3, label=did,
                    color=color, markersize=6, zorder=10)
            ax.fill(angles, vals_closed, alpha=0.25, color=color, zorder=9)

            # Add value annotations
            for i, (angle, val, axis_info) in enumerate(zip(angles[:-1], vals, active_axes)):
                col = axis_info[1]
                raw = self._selected_design.get(col)
                if raw is not None and pd.notna(raw):
                    try:
                        raw_str = f"{float(raw):.2f}"
                    except (ValueError, TypeError):
                        raw_str = str(raw)
                    ax.annotate(raw_str, xy=(angle, val), fontsize=7, fontweight="bold",
                                ha="center", va="bottom", color=color,
                                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.7))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([], fontsize=8)  # clear default labels
        # Add two-line labels: name (bold) + scale (small)
        for angle, label_text in zip(angles[:-1], labels):
            parts = label_text.split("\n", 1)
            name = parts[0]
            scale = parts[1] if len(parts) > 1 else ""
            # Position labels outside the chart with spacing
            r_offset = 1.20
            ax.text(angle, r_offset, name, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#e0e0e0",
                    transform=ax.get_xaxis_transform())
            if scale:
                ax.text(angle, r_offset + 0.12, scale, ha="center", va="center",
                        fontsize=7.5, color="#999999",
                        transform=ax.get_xaxis_transform())
        handles, labels_leg = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7,
                               facecolor='#1a1a2e', edgecolor='#444444')
            for text in legend.get_texts():
                text.set_color('#e0e0e0')

        title = "Developability Profile"
        if self._selected_design:
            title += f" — {self._selected_design.get('design_id', '')}"
        ax.set_title(title, fontsize=11, y=1.08, color='#e0e0e0')

        self.figure.tight_layout(pad=2.5)
        self.canvas.draw()

    def _show_info(self):
        """Show score descriptions dialog."""
        lines = ["<h3>Radar Chart — Score Guide</h3><table>"]
        lines.append("<tr><th style='text-align:left;padding:4px'>Axis</th>"
                     "<th style='text-align:left;padding:4px'>Description</th></tr>")
        for axis_info in RADAR_AXES:
            name = axis_info[0].replace("\n", " ")
            desc = axis_info[5] if len(axis_info) > 5 else ""
            desc_html = desc.replace("\n", "<br>")
            lines.append(f"<tr><td style='padding:4px'><b>{name}</b></td>"
                         f"<td style='padding:4px'>{desc_html}</td></tr>")
        lines.append("</table>")
        QMessageBox.information(self, "Score Guide", "".join(lines))

    def highlight_design(self, design_data: dict):
        """Called externally (e.g., from table selection) to overlay a design."""
        self._selected_design = design_data
        # Update the combo to reflect selection
        did = design_data.get("design_id", "")
        self.design_input.setText(did)
        self._replot()
