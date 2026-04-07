"""Main application window."""

from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QTabWidget, QStatusBar, QMenuBar, QFileDialog, QMessageBox,
    QDockWidget, QLabel,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction

from .data import scan_results_dir, load_rankings
from .data.models import RunInfo
from .widgets import (
    RunSelector, RankingsTable, ScatterView, DesignDetail,
    ToolComparison, FilterPanel, DashboardViewer, RadarChart,
)


class MainWindow(QMainWindow):

    def __init__(self, results_dir: Path, parent=None):
        super().__init__(parent)
        self.results_dir = results_dir
        self._full_df = pd.DataFrame()  # unfiltered
        self._filtered_df = pd.DataFrame()
        self._current_run = None

        self.setWindowTitle(f"Binder Browser - {results_dir.name}")
        self.setMinimumSize(QSize(1200, 700))
        self.resize(1600, 900)

        self._build_menu()
        self._build_ui()
        self._build_statusbar()

        # Load runs
        runs = scan_results_dir(results_dir)
        if runs:
            self.run_selector.set_runs(runs)
        else:
            QMessageBox.warning(
                self, "No runs found",
                f"No rankings.csv files found in:\n{results_dir}\n\n"
                "Expected subdirectories with rankings.csv inside."
            )

    def _build_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open results dir...", self)
        open_action.triggered.connect(self._open_dir)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        export_action = QAction("Export filtered CSV...", self)
        export_action.triggered.connect(self._export_csv)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("View")
        self._filter_toggle = QAction("Show filters", self)
        self._filter_toggle.setCheckable(True)
        self._filter_toggle.setChecked(True)
        self._filter_toggle.triggered.connect(self._toggle_filters)
        view_menu.addAction(self._filter_toggle)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Main splitter: run selector | tabs + filter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: run selector
        self.run_selector = RunSelector()
        self.run_selector.setMinimumWidth(200)
        self.run_selector.setMaximumWidth(400)
        self.run_selector.run_selected.connect(self._on_run_selected)
        splitter.addWidget(self.run_selector)

        # Right side: tabs + filter in a vertical split
        right_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Center: tabs
        self.tabs = QTabWidget()

        self.rankings_table = RankingsTable()
        self.rankings_table.design_selected.connect(self._on_design_selected)
        self.tabs.addTab(self.rankings_table, "Rankings")

        self.scatter_view = ScatterView()
        self.scatter_view.design_selected.connect(self._on_design_selected)
        self.tabs.addTab(self.scatter_view, "Scatter")

        self.tool_comparison = ToolComparison()
        self.tabs.addTab(self.tool_comparison, "Tools")

        self.radar_chart = RadarChart()
        self.tabs.addTab(self.radar_chart, "Radar")

        self.design_detail = DesignDetail()
        self.design_detail.design_selected.connect(self._on_design_selected)
        self.tabs.addTab(self.design_detail, "Detail")

        self.dashboard_viewer = DashboardViewer()
        self.tabs.addTab(self.dashboard_viewer, "Dashboard")

        right_splitter.addWidget(self.tabs)

        # Right: filter panel
        self.filter_panel = FilterPanel()
        self.filter_panel.setMinimumWidth(200)
        self.filter_panel.setMaximumWidth(320)
        self.filter_panel.filters_changed.connect(self._apply_filters)
        right_splitter.addWidget(self.filter_panel)

        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)

        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)

        main_layout.addWidget(splitter)

    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_total = QLabel("No data")
        self.status_filtered = QLabel("")
        self.status_best = QLabel("")
        self.status.addWidget(self.status_total)
        self.status.addWidget(self.status_filtered)
        self.status.addPermanentWidget(self.status_best)

    # --- Slots ---

    def _on_run_selected(self, run: RunInfo):
        self._current_run = run
        self._full_df = load_rankings(run.rankings_csv)
        self._filtered_df = self._full_df.copy()

        # Apply current filters
        self._apply_filters(self.filter_panel.get_filters())

        # Dashboard
        self.dashboard_viewer.load_image(run.dashboard_png)

        # Design detail run path
        self.design_detail.set_run_path(run.path)

        self._update_status()

    def _on_design_selected(self, row_data: dict):
        self.design_detail.show_design(row_data)
        self.radar_chart.highlight_design(row_data)
        # Switch to detail tab if not already there and user clicked from scatter
        if self.tabs.currentWidget() == self.scatter_view:
            self.tabs.setCurrentWidget(self.design_detail)

    def _apply_filters(self, filters: dict):
        if self._full_df.empty:
            return

        df = self._full_df.copy()
        mask = pd.Series(True, index=df.index)

        # Unranked filter
        if not filters.get("include_unranked", False):
            if "rank" in df.columns:
                mask &= df["rank"].notna()

        # Score thresholds (NaN values pass when filter is at default/floor)
        if "combined_score" in df.columns and filters["min_combined"] > -10:
            mask &= df["combined_score"].fillna(-999) >= filters["min_combined"]
        if "boltz_iptm" in df.columns and filters["min_iptm"] > 0:
            mask &= df["boltz_iptm"].fillna(-999) >= filters["min_iptm"]
        if "esmfold_plddt" in df.columns and filters["min_plddt"] > 0:
            mask &= df["esmfold_plddt"].fillna(-999) >= filters["min_plddt"]
        if "rosetta_dG" in df.columns and filters["max_dg"] < 100:
            mask &= df["rosetta_dG"].fillna(999) <= filters["max_dg"]
        if "boltz_site_mean_pae" in df.columns and filters["max_site_pae"] < 50:
            mask &= df["boltz_site_mean_pae"].fillna(999) <= filters["max_site_pae"]

        # Geometric filters
        if "refolding_rmsd" in df.columns and filters.get("max_rmsd", 20) < 20:
            mask &= df["refolding_rmsd"].fillna(999) <= filters["max_rmsd"]
        if "site_interface_fraction" in df.columns and filters.get("min_sif", 0) > 0:
            mask &= df["site_interface_fraction"].fillna(-1) >= filters["min_sif"]
        if "interface_KE_fraction" in df.columns and filters.get("max_ke", 1) < 1:
            mask &= df["interface_KE_fraction"].fillna(999) <= filters["max_ke"]
        if filters.get("no_cys", False) and "binder_sequence" in df.columns:
            mask &= ~df["binder_sequence"].fillna("").str.contains("C")
        if filters.get("max_aa", 1) < 1 and "binder_sequence" in df.columns:
            def _max_aa_frac(seq):
                if not seq or len(seq) == 0:
                    return 0
                return max(seq.count(aa) / len(seq) for aa in set(seq))
            mask &= df["binder_sequence"].fillna("").apply(_max_aa_frac) <= filters["max_aa"]

        # Tool filter
        if "tool" in df.columns:
            mask &= df["tool"].isin(filters["active_tools"])

        # SS bias
        ss = filters.get("ss_bias", "any")
        if ss == "helix" and "binder_helix_frac" in df.columns:
            mask &= df["binder_helix_frac"].fillna(0) > 0.4
        elif ss == "sheet" and "binder_sheet_frac" in df.columns:
            mask &= df["binder_sheet_frac"].fillna(0) > 0.3
        elif ss == "balanced":
            if "binder_helix_frac" in df.columns:
                mask &= df["binder_helix_frac"].fillna(0) < 0.6
            if "binder_sheet_frac" in df.columns:
                mask &= df["binder_sheet_frac"].fillna(0) < 0.4

        self._filtered_df = df[mask].reset_index(drop=True)

        # Update all widgets
        run_path = self._current_run.path if self._current_run else None
        self.rankings_table.set_data(self._filtered_df, run_path)
        self.scatter_view.set_data(self._filtered_df)
        self.tool_comparison.set_data(self._filtered_df)
        self.radar_chart.set_data(self._filtered_df)
        self.design_detail.set_data(self._filtered_df, run_path)

        self._update_status()

    def _update_status(self):
        total = len(self._full_df)
        shown = len(self._filtered_df)
        self.status_total.setText(f"{shown} / {total} designs")
        if not self._filtered_df.empty:
            best_score = self._filtered_df["combined_score"].max() if "combined_score" in self._filtered_df.columns else 0
            best_iptm = self._filtered_df["boltz_iptm"].max() if "boltz_iptm" in self._filtered_df.columns else 0
            self.status_best.setText(f"Best: score={best_score:.3f}  iPTM={best_iptm:.3f}")

            active_filters = []
            f = self.filter_panel.get_filters()
            if f["min_combined"] > -10:
                active_filters.append(f"score>{f['min_combined']:.2f}")
            if f["min_iptm"] > 0:
                active_filters.append(f"iPTM>{f['min_iptm']:.2f}")
            if f["max_dg"] < 100:
                active_filters.append(f"dG<{f['max_dg']:.0f}")
            if f["min_plddt"] > 0:
                active_filters.append(f"pLDDT>{f['min_plddt']:.0f}")
            if f["max_site_pae"] < 50:
                active_filters.append(f"sitePAE<{f['max_site_pae']:.0f}")
            self.status_filtered.setText(
                f"Filter: {', '.join(active_filters)}" if active_filters else ""
            )
        else:
            self.status_best.setText("")
            self.status_filtered.setText("")

    def _toggle_filters(self, checked):
        self.filter_panel.setVisible(checked)

    def _open_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select results directory", str(self.results_dir)
        )
        if dir_path:
            self.results_dir = Path(dir_path)
            self.setWindowTitle(f"Binder Browser - {self.results_dir.name}")
            runs = scan_results_dir(self.results_dir)
            self.run_selector.set_runs(runs)

    def _export_csv(self):
        if self._filtered_df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export filtered rankings", "filtered_rankings.csv",
            "CSV files (*.csv)"
        )
        if path:
            self._filtered_df.to_csv(path, index=False)
            self.status.showMessage(f"Exported {len(self._filtered_df)} designs to {path}", 5000)
