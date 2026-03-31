"""Per-tool comparison plots: box plots, histograms, score vs length."""

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt6.QtCore import pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..config import TOOL_COLORS


def _style_ax(fig, ax):
    """Apply dark theme to a matplotlib figure and axes."""
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#e0e0e0')
    ax.xaxis.label.set_color('#e0e0e0')
    ax.yaxis.label.set_color('#e0e0e0')
    ax.title.set_color('#e0e0e0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')


def _style_legend(ax, **kwargs):
    """Create a dark-themed legend."""
    legend = ax.legend(facecolor='#1a1a2e', edgecolor='#444444', **kwargs)
    for text in legend.get_texts():
        text.set_color('#e0e0e0')
    return legend


METRICS = [
    "combined_score", "boltz_iptm", "boltz_binder_plddt", "esmfold_plddt",
    "rosetta_dG", "boltz_site_mean_pae", "rosetta_sc",
    "site_interface_fraction", "refolding_rmsd", "pDockQ",
    "interface_KE_fraction",
    "binder_helix_frac", "binder_sheet_frac",
]

PLOT_MODES = ["Box Plot", "Histogram", "Score vs Length"]


class ToolComparison(QWidget):
    """Tool comparison with box plots, overlaid histograms, and score vs length."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Plot:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(PLOT_MODES)
        self.mode_combo.currentTextChanged.connect(self._replot)
        controls.addWidget(self.mode_combo)

        controls.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(METRICS)
        self.metric_combo.currentTextChanged.connect(self._replot)
        controls.addWidget(self.metric_combo)

        controls.addStretch()
        layout.addLayout(controls)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def set_data(self, df: pd.DataFrame):
        self._df = df
        available = [m for m in METRICS if m in df.columns and df[m].notna().any()]
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        self.metric_combo.addItems(available)
        self.metric_combo.blockSignals(False)
        if available:
            self._replot()

    def _replot(self, _=None):
        mode = self.mode_combo.currentText()
        metric = self.metric_combo.currentText()

        if mode == "Score vs Length":
            self._plot_score_vs_length()
        elif mode == "Histogram":
            self._plot_histogram(metric)
        else:
            self._plot_boxplot(metric)

    def _plot_boxplot(self, metric):
        if not metric or metric not in self._df.columns:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        df = self._df.dropna(subset=[metric])
        if "tool" not in df.columns or df.empty:
            self.canvas.draw()
            return

        tools = sorted(df["tool"].unique())
        data, colors, labels = [], [], []
        for tool in tools:
            vals = df[df["tool"] == tool][metric].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                colors.append(TOOL_COLORS.get(tool, "#888888"))
                labels.append(f"{tool}\n(n={len(vals)})")

        if not data:
            self.canvas.draw()
            return

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by tool", fontsize=11)
        ax.tick_params(axis="x", labelsize=8, colors='#e0e0e0')
        # Style boxplot median/whisker/cap lines for visibility
        for element in ['whiskers', 'caps', 'medians', 'fliers']:
            for item in bp[element]:
                item.set_color('#e0e0e0')
        _style_ax(self.figure, ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_histogram(self, metric):
        """Overlaid per-tool histograms (like generator comparison in dashboard)."""
        if not metric or metric not in self._df.columns:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        df = self._df.dropna(subset=[metric])
        if "tool" not in df.columns or df.empty:
            self.canvas.draw()
            return

        tools = sorted(df["tool"].unique())
        all_vals = df[metric].dropna()
        if all_vals.empty:
            self.canvas.draw()
            return

        bins = np.linspace(all_vals.min(), all_vals.max(), 30)

        for tool in tools:
            vals = df[df["tool"] == tool][metric].dropna()
            if len(vals) > 0:
                color = TOOL_COLORS.get(tool, "#888888")
                ax.hist(vals, bins=bins, alpha=0.5, color=color,
                        label=f"{tool} (n={len(vals)})", edgecolor="white", linewidth=0.3)

        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.set_title(f"{metric} distribution by tool", fontsize=11)
        _style_legend(ax, fontsize=8, loc="upper right")
        _style_ax(self.figure, ax)
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_score_vs_length(self):
        """Score vs binder length colored by tool."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        df = self._df.copy()
        score_col = "combined_score"
        length_col = "binder_length"

        # Compute length if missing
        if length_col not in df.columns and "binder_sequence" in df.columns:
            df[length_col] = df["binder_sequence"].apply(
                lambda x: len(str(x)) if pd.notna(x) and x != "" else np.nan)

        if score_col not in df.columns or length_col not in df.columns:
            self.canvas.draw()
            return

        df = df.dropna(subset=[score_col, length_col])
        if df.empty or "tool" not in df.columns:
            self.canvas.draw()
            return

        tools = sorted(df["tool"].unique())
        for tool in tools:
            subset = df[df["tool"] == tool]
            color = TOOL_COLORS.get(tool, "#888888")
            ax.scatter(subset[length_col], subset[score_col],
                       c=color, label=f"{tool} ({len(subset)})",
                       alpha=0.5, s=15, edgecolors="none")

        ax.set_xlabel("Binder Length (aa)")
        ax.set_ylabel("Combined Score")
        ax.set_title("Score vs Length by Tool", fontsize=11)
        _style_legend(ax, fontsize=7, loc="lower right", markerscale=2)
        _style_ax(self.figure, ax)
        self.figure.tight_layout()
        self.canvas.draw()
