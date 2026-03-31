"""Main rankings table with sorting, filtering, color gradients, and context menu."""

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView,
    QMenu, QAbstractItemView, QLabel, QPushButton, QDialog,
    QListWidget, QListWidgetItem, QDialogButtonBox, QApplication,
)
from PyQt6.QtCore import (
    pyqtSignal, Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel,
)
from PyQt6.QtGui import QColor, QAction, QFont

from ..config import (
    DEFAULT_COLUMNS, SCORE_COLUMNS_HIGHER_BETTER,
    SCORE_COLUMNS_LOWER_BETTER, TOOL_COLORS,
)


class RankingsDataModel(QAbstractTableModel):
    """Virtual table model backed by a DataFrame for efficient display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._visible_columns = list(DEFAULT_COLUMNS)
        self._all_columns = []
        self._col_min = {}
        self._col_max = {}

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self._all_columns = list(df.columns)
        # Keep only visible columns that exist
        self._visible_columns = [c for c in self._visible_columns if c in self._all_columns]
        # Precompute min/max for gradient columns
        gradient_cols = SCORE_COLUMNS_HIGHER_BETTER | SCORE_COLUMNS_LOWER_BETTER
        for col in gradient_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                self._col_min[col] = vals.min() if not vals.isna().all() else 0
                self._col_max[col] = vals.max() if not vals.isna().all() else 1
        self.endResetModel()

    def set_visible_columns(self, columns: list):
        self.beginResetModel()
        self._visible_columns = [c for c in columns if c in self._all_columns]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._visible_columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        col_name = self._visible_columns[index.column()]
        value = self._df.iloc[index.row()][col_name]

        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        elif role == Qt.ItemDataRole.BackgroundRole:
            return self._get_gradient_color(col_name, value)

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col_name in ("design_id", "tool", "binder_sequence"):
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        elif role == Qt.ItemDataRole.UserRole:
            # Return the full row as dict
            return self._df.iloc[index.row()].to_dict()

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._visible_columns[section]
            return str(section + 1)
        return None

    def sort(self, column, order):
        if column < 0 or column >= len(self._visible_columns):
            return
        col_name = self._visible_columns[column]
        if col_name not in self._df.columns or self._df.empty:
            return
        ascending = order == Qt.SortOrder.AscendingOrder
        self.beginResetModel()
        self._df = self._df.sort_values(col_name, ascending=ascending, na_position="last").reset_index(drop=True)
        self.endResetModel()

    def _get_gradient_color(self, col_name, value):
        if pd.isna(value):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if col_name in SCORE_COLUMNS_HIGHER_BETTER:
            vmin = self._col_min.get(col_name, 0)
            vmax = self._col_max.get(col_name, 1)
            if vmax == vmin:
                return None
            frac = (value - vmin) / (vmax - vmin)
            frac = max(0.0, min(1.0, frac))
            # Red (bad) -> Yellow -> Green (good)
            r = int(255 * (1 - frac))
            g = int(200 * frac)
            return QColor(r, g, 80, 60)

        elif col_name in SCORE_COLUMNS_LOWER_BETTER:
            vmin = self._col_min.get(col_name, 0)
            vmax = self._col_max.get(col_name, 1)
            if vmax == vmin:
                return None
            frac = (value - vmin) / (vmax - vmin)
            frac = max(0.0, min(1.0, frac))
            # Inverted: low is good (green), high is bad (red)
            r = int(255 * frac)
            g = int(200 * (1 - frac))
            return QColor(r, g, 80, 60)

        return None

    def get_row_data(self, row: int) -> dict:
        if 0 <= row < len(self._df):
            return self._df.iloc[row].to_dict()
        return {}

    def get_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


class RankingsTable(QWidget):
    """Main sortable/filterable rankings table."""

    design_selected = pyqtSignal(dict)  # row data as dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_path = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        self.count_label = QLabel("No data loaded")
        toolbar.addWidget(self.count_label)
        toolbar.addStretch()
        col_btn = QPushButton("Columns...")
        col_btn.clicked.connect(self._show_column_chooser)
        toolbar.addWidget(col_btn)
        layout.addLayout(toolbar)

        # Table
        self.model = RankingsDataModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.selectionModel().currentRowChanged.connect(self._on_row_changed)

        layout.addWidget(self.table)

    def set_data(self, df: pd.DataFrame, run_path: Path = None):
        """Load a DataFrame into the table."""
        self._run_path = run_path
        self.model.set_dataframe(df)
        self.count_label.setText(f"{len(df)} designs")
        # Auto-resize first few columns
        for i in range(min(8, self.model.columnCount())):
            self.table.resizeColumnToContents(i)

    def apply_filter(self, mask: pd.Series):
        """Apply a boolean filter mask to the displayed data."""
        # We re-set the model with filtered data
        # The parent holds the full df and provides the mask
        pass  # Handled by app.py calling set_data with filtered df

    def _on_row_changed(self, current, previous):
        if not current.isValid():
            return
        row_data = self.model.get_row_data(current.row())
        if row_data:
            self.design_selected.emit(row_data)

    def _context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        row_data = self.model.get_row_data(index.row())
        if not row_data:
            return

        menu = QMenu(self)

        copy_seq = QAction("Copy sequence", self)
        copy_seq.triggered.connect(lambda: self._copy_sequence(row_data))
        menu.addAction(copy_seq)

        copy_id = QAction("Copy design ID", self)
        copy_id.triggered.connect(
            lambda: QApplication.clipboard().setText(str(row_data.get("design_id", "")))
        )
        menu.addAction(copy_id)

        open_pymol = QAction("Open in PyMOL", self)
        open_pymol.triggered.connect(lambda: self._open_pymol(row_data))
        menu.addAction(open_pymol)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _copy_sequence(self, row_data):
        seq = row_data.get("binder_sequence", "")
        QApplication.clipboard().setText(str(seq))

    def _open_pymol(self, row_data):
        from ..data.models import DesignResult
        design_id = str(row_data.get("design_id", ""))
        tool = str(row_data.get("tool", "unknown"))
        dr = DesignResult(
            design_id=design_id, tool=tool, rank=0,
            combined_score=0, run_path=self._run_path,
        )
        struct = dr.find_structure()
        if struct and struct.exists():
            subprocess.Popen(["pymol", str(struct)], cwd=str(struct.parent))

    def _show_column_chooser(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select columns")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)

        lw = QListWidget()
        current_visible = set(self.model._visible_columns)
        for col in self.model._all_columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if col in current_visible else Qt.CheckState.Unchecked
            )
            lw.addItem(item)
        layout.addWidget(lw)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            cols = []
            for i in range(lw.count()):
                item = lw.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    cols.append(item.text())
            self.model.set_visible_columns(cols)
