"""Left panel: list of runs with summary stats."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from ..data.models import RunInfo


class RunSelector(QWidget):
    """List of discovered runs. Emits run_selected when clicked."""

    run_selected = pyqtSignal(object)  # RunInfo

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Runs")
        header.setFont(QFont("", 11, QFont.Weight.Bold))
        layout.addWidget(header)

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_selection)
        layout.addWidget(self.list_widget)

        self._runs = []

    def set_runs(self, runs: list):
        """Populate the run list."""
        self._runs = runs
        self.list_widget.clear()
        for run in runs:
            tools_str = ", ".join(run.tools) if run.tools else "?"
            text = (
                f"{run.name}\n"
                f"  {run.n_designs} designs | best score: {run.best_combined:.3f}\n"
                f"  best iPTM: {run.best_iptm:.3f} | tools: {tools_str}"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, run)
            self.list_widget.addItem(item)

        # Auto-select first
        if runs:
            self.list_widget.setCurrentRow(0)

    def _on_selection(self, current, previous):
        if current is None:
            return
        run = current.data(Qt.ItemDataRole.UserRole)
        if run:
            self.run_selected.emit(run)
