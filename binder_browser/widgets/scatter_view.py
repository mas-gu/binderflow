"""Matplotlib scatter plots with tool coloring and point selection."""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt6.QtCore import pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..config import TOOL_COLORS, SCATTER_PRESETS


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


class ScatterView(QWidget):
    """Interactive scatter plot with preset and custom axis selection."""

    design_selected = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._numeric_cols = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(SCATTER_PRESETS.keys()))
        self.preset_combo.addItem("Custom...")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        controls.addWidget(self.preset_combo)

        controls.addWidget(QLabel("X:"))
        self.x_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(self._on_custom_axis)
        controls.addWidget(self.x_combo)

        controls.addWidget(QLabel("Y:"))
        self.y_combo = QComboBox()
        self.y_combo.currentTextChanged.connect(self._on_custom_axis)
        controls.addWidget(self.y_combo)

        controls.addStretch()
        layout.addLayout(controls)

        # Matplotlib canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        layout.addWidget(self.canvas)

    def set_data(self, df: pd.DataFrame):
        self._df = df
        self._numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != "rank"
        ]
        # Update combo boxes
        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(self._numeric_cols)
        self.y_combo.addItems(self._numeric_cols)
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)

        # Apply current preset
        self._on_preset_changed(self.preset_combo.currentText())

    def _on_preset_changed(self, text):
        if text == "Custom...":
            return
        if text in SCATTER_PRESETS:
            x_col, y_col = SCATTER_PRESETS[text]
            if x_col in self._numeric_cols and y_col in self._numeric_cols:
                self.x_combo.blockSignals(True)
                self.y_combo.blockSignals(True)
                self.x_combo.setCurrentText(x_col)
                self.y_combo.setCurrentText(y_col)
                self.x_combo.blockSignals(False)
                self.y_combo.blockSignals(False)
                self._plot(x_col, y_col)

    def _on_custom_axis(self, _text=None):
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Custom...")
        self.preset_combo.blockSignals(False)
        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        if x_col and y_col and x_col in self._df.columns and y_col in self._df.columns:
            self._plot(x_col, y_col)

    def _plot(self, x_col: str, y_col: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self._df.empty:
            _style_ax(self.figure, ax)
            self.canvas.draw()
            return

        df = self._df.dropna(subset=[x_col, y_col])
        if df.empty:
            ax.set_title("No data for selected axes")
            _style_ax(self.figure, ax)
            self.canvas.draw()
            return

        # Plot per tool
        tools = df["tool"].unique() if "tool" in df.columns else ["all"]
        for tool in tools:
            mask = df["tool"] == tool if "tool" in df.columns else pd.Series(True, index=df.index)
            subset = df[mask]
            color = TOOL_COLORS.get(tool, "#888888")
            ax.scatter(
                subset[x_col], subset[y_col],
                c=color, label=tool, alpha=0.7, s=30,
                edgecolors="white", linewidths=0.3,
                picker=True, pickradius=5,
            )
            # Store index for pick events
            for idx in subset.index:
                pass  # scatter handles pick via indices

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        legend = ax.legend(loc="best", fontsize=8, framealpha=0.8,
                           facecolor='#1a1a2e', edgecolor='#444444')
        for text in legend.get_texts():
            text.set_color('#e0e0e0')
        ax.set_title(f"{y_col} vs {x_col}", fontsize=10)
        _style_ax(self.figure, ax)
        self.figure.tight_layout()
        self.canvas.draw()

        # Store plot data for pick resolution
        self._plot_df = df[[x_col, y_col, "tool"]].copy() if "tool" in df.columns else df[[x_col, y_col]].copy()
        self._plot_df["_orig_idx"] = df.index
        self._plot_x = x_col
        self._plot_y = y_col

    def _on_pick(self, event):
        if not hasattr(self, "_plot_df") or self._plot_df.empty:
            return
        ind = event.ind
        if len(ind) == 0:
            return
        # Find the original DataFrame row
        # The scatter collections are per-tool, so we need to map back
        # Use the first picked point
        artist = event.artist
        offsets = artist.get_offsets()
        if len(ind) > 0:
            picked_x, picked_y = offsets[ind[0]]
            # Find closest in original df
            df = self._df.dropna(subset=[self._plot_x, self._plot_y])
            dists = (df[self._plot_x] - picked_x)**2 + (df[self._plot_y] - picked_y)**2
            if not dists.empty:
                best_idx = dists.idxmin()
                row_data = self._df.loc[best_idx].to_dict()
                self.design_selected.emit(row_data)
