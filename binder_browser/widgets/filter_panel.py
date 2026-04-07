"""Score filters: min iPTM, max dG, SS composition, tool toggles."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QGroupBox, QPushButton, QRadioButton, QButtonGroup,
    QGridLayout, QScrollArea,
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont

from ..config import TOOL_COLORS


class FilterPanel(QWidget):
    """Filter controls for score thresholds, tool selection, SS bias."""

    filters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tool_checkboxes = {}
        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        header = QLabel("Filters")
        header.setFont(QFont("", 11, QFont.Weight.Bold))
        layout.addWidget(header)

        self.include_unranked = QCheckBox("Include unranked designs")
        self.include_unranked.setChecked(False)
        self.include_unranked.stateChanged.connect(lambda _: self._emit_filters())
        layout.addWidget(self.include_unranked)

        # Score filters
        score_group = QGroupBox("Score Thresholds")
        sg = QGridLayout()

        # Min combined score
        sg.addWidget(QLabel("Min combined score:"), 0, 0)
        self.min_combined = QDoubleSpinBox()
        self.min_combined.setRange(-10, 2)
        self.min_combined.setValue(-10.0)
        self.min_combined.setSingleStep(0.05)
        self.min_combined.setDecimals(3)
        sg.addWidget(self.min_combined, 0, 1)

        # Min iPTM
        sg.addWidget(QLabel("Min iPTM:"), 1, 0)
        self.min_iptm = QDoubleSpinBox()
        self.min_iptm.setRange(0, 1)
        self.min_iptm.setValue(0.0)
        self.min_iptm.setSingleStep(0.05)
        self.min_iptm.setDecimals(3)
        sg.addWidget(self.min_iptm, 1, 1)

        # Min pLDDT
        sg.addWidget(QLabel("Min ESMFold pLDDT:"), 2, 0)
        self.min_plddt = QDoubleSpinBox()
        self.min_plddt.setRange(0, 100)
        self.min_plddt.setValue(0.0)
        self.min_plddt.setSingleStep(5)
        self.min_plddt.setDecimals(1)
        sg.addWidget(self.min_plddt, 2, 1)

        # Max dG
        sg.addWidget(QLabel("Max Rosetta dG:"), 3, 0)
        self.max_dg = QDoubleSpinBox()
        self.max_dg.setRange(-500, 100)
        self.max_dg.setValue(100.0)
        self.max_dg.setSingleStep(5)
        self.max_dg.setDecimals(1)
        sg.addWidget(self.max_dg, 3, 1)

        # Max site PAE
        sg.addWidget(QLabel("Max site PAE:"), 4, 0)
        self.max_site_pae = QDoubleSpinBox()
        self.max_site_pae.setRange(0, 50)
        self.max_site_pae.setValue(50.0)
        self.max_site_pae.setSingleStep(1)
        self.max_site_pae.setDecimals(1)
        sg.addWidget(self.max_site_pae, 4, 1)

        score_group.setLayout(sg)
        layout.addWidget(score_group)

        # Geometric filters
        geo_group = QGroupBox("Geometric Filters")
        gl = QGridLayout()

        gl.addWidget(QLabel("Max refolding RMSD:"), 0, 0)
        self.max_rmsd = QDoubleSpinBox()
        self.max_rmsd.setRange(0, 20)
        self.max_rmsd.setValue(20.0)
        self.max_rmsd.setSingleStep(0.5)
        self.max_rmsd.setDecimals(1)
        gl.addWidget(self.max_rmsd, 0, 1)

        gl.addWidget(QLabel("Min SIF:"), 1, 0)
        self.min_sif = QDoubleSpinBox()
        self.min_sif.setRange(0, 1)
        self.min_sif.setValue(0.0)
        self.min_sif.setSingleStep(0.05)
        self.min_sif.setDecimals(2)
        gl.addWidget(self.min_sif, 1, 1)

        gl.addWidget(QLabel("Max interface KE:"), 2, 0)
        self.max_ke = QDoubleSpinBox()
        self.max_ke.setRange(0, 1)
        self.max_ke.setValue(1.0)
        self.max_ke.setSingleStep(0.05)
        self.max_ke.setDecimals(2)
        gl.addWidget(self.max_ke, 2, 1)

        gl.addWidget(QLabel("Max AA fraction:"), 3, 0)
        self.max_aa = QDoubleSpinBox()
        self.max_aa.setRange(0, 1)
        self.max_aa.setValue(1.0)
        self.max_aa.setSingleStep(0.05)
        self.max_aa.setDecimals(2)
        gl.addWidget(self.max_aa, 3, 1)

        self.no_cys = QCheckBox("No cysteine")
        gl.addWidget(self.no_cys, 4, 0, 1, 2)

        geo_group.setLayout(gl)
        layout.addWidget(geo_group)

        # Tool filter
        tool_group = QGroupBox("Tools")
        tl = QVBoxLayout()
        for tool, color in TOOL_COLORS.items():
            cb = QCheckBox(tool)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {color};")
            self._tool_checkboxes[tool] = cb
            tl.addWidget(cb)
        # Unknown tool
        cb_unk = QCheckBox("unknown")
        cb_unk.setChecked(True)
        self._tool_checkboxes["unknown"] = cb_unk
        tl.addWidget(cb_unk)

        tool_btn_layout = QHBoxLayout()
        sel_all = QPushButton("All")
        sel_all.clicked.connect(lambda: self._set_all_tools(True))
        tool_btn_layout.addWidget(sel_all)
        sel_none = QPushButton("None")
        sel_none.clicked.connect(lambda: self._set_all_tools(False))
        tool_btn_layout.addWidget(sel_none)
        tl.addLayout(tool_btn_layout)

        tool_group.setLayout(tl)
        layout.addWidget(tool_group)

        # SS bias
        ss_group = QGroupBox("SS Bias")
        ssl = QVBoxLayout()
        self.ss_button_group = QButtonGroup()
        self.ss_any = QRadioButton("Any")
        self.ss_any.setChecked(True)
        self.ss_helix = QRadioButton("Helix-rich (>40%)")
        self.ss_sheet = QRadioButton("Sheet-rich (>30%)")
        self.ss_balanced = QRadioButton("Balanced (H<60%, S<40%)")
        self.ss_button_group.addButton(self.ss_any, 0)
        self.ss_button_group.addButton(self.ss_helix, 1)
        self.ss_button_group.addButton(self.ss_sheet, 2)
        self.ss_button_group.addButton(self.ss_balanced, 3)
        for btn in [self.ss_any, self.ss_helix, self.ss_sheet, self.ss_balanced]:
            ssl.addWidget(btn)
        ss_group.setLayout(ssl)
        layout.addWidget(ss_group)

        # Buttons
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._emit_filters)
        btn_layout.addWidget(apply_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _set_all_tools(self, checked):
        for cb in self._tool_checkboxes.values():
            cb.setChecked(checked)

    def _reset(self):
        self.include_unranked.setChecked(False)
        self.min_combined.setValue(-10.0)
        self.min_iptm.setValue(0.0)
        self.min_plddt.setValue(0.0)
        self.max_dg.setValue(100.0)
        self.max_site_pae.setValue(50.0)
        self.max_rmsd.setValue(20.0)
        self.min_sif.setValue(0.0)
        self.max_ke.setValue(1.0)
        self.max_aa.setValue(1.0)
        self.no_cys.setChecked(False)
        self._set_all_tools(True)
        self.ss_any.setChecked(True)
        self._emit_filters()

    def _emit_filters(self):
        active_tools = [
            tool for tool, cb in self._tool_checkboxes.items() if cb.isChecked()
        ]
        ss_id = self.ss_button_group.checkedId()
        ss_bias = {0: "any", 1: "helix", 2: "sheet", 3: "balanced"}.get(ss_id, "any")

        filters = {
            "include_unranked": self.include_unranked.isChecked(),
            "min_combined": self.min_combined.value(),
            "min_iptm": self.min_iptm.value(),
            "min_plddt": self.min_plddt.value(),
            "max_dg": self.max_dg.value(),
            "max_site_pae": self.max_site_pae.value(),
            "max_rmsd": self.max_rmsd.value(),
            "min_sif": self.min_sif.value(),
            "max_ke": self.max_ke.value(),
            "max_aa": self.max_aa.value(),
            "no_cys": self.no_cys.isChecked(),
            "active_tools": active_tools,
            "ss_bias": ss_bias,
        }
        self.filters_changed.emit(filters)

    def get_filters(self) -> dict:
        """Return current filter settings without emitting signal."""
        active_tools = [
            tool for tool, cb in self._tool_checkboxes.items() if cb.isChecked()
        ]
        ss_id = self.ss_button_group.checkedId()
        ss_bias = {0: "any", 1: "helix", 2: "sheet", 3: "balanced"}.get(ss_id, "any")
        return {
            "include_unranked": self.include_unranked.isChecked(),
            "min_combined": self.min_combined.value(),
            "min_iptm": self.min_iptm.value(),
            "min_plddt": self.min_plddt.value(),
            "max_dg": self.max_dg.value(),
            "max_site_pae": self.max_site_pae.value(),
            "max_rmsd": self.max_rmsd.value(),
            "min_sif": self.min_sif.value(),
            "max_ke": self.max_ke.value(),
            "max_aa": self.max_aa.value(),
            "no_cys": self.no_cys.isChecked(),
            "active_tools": active_tools,
            "ss_bias": ss_bias,
        }
