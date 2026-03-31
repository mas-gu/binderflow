"""Dashboard image viewer with zoom."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap


class DashboardViewer(QWidget):
    """Display dashboard.png with zoom controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._scale = 1.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Zoom controls
        controls = QHBoxLayout()
        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(30)
        zoom_in.clicked.connect(lambda: self._zoom(1.25))
        controls.addWidget(zoom_in)
        zoom_out = QPushButton("-")
        zoom_out.setFixedWidth(30)
        zoom_out.clicked.connect(lambda: self._zoom(0.8))
        controls.addWidget(zoom_out)
        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._fit)
        controls.addWidget(fit_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.image_label = QLabel("No dashboard loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll)

    def load_image(self, path: Path):
        if path and path.is_file():
            self._pixmap = QPixmap(str(path))
            self._scale = 1.0
            self._update_display()
        else:
            self._pixmap = None
            self.image_label.setText("No dashboard found")

    def _update_display(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                int(self._pixmap.width() * self._scale),
                int(self._pixmap.height() * self._scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
            self.image_label.resize(scaled.size())

    def _zoom(self, factor):
        self._scale *= factor
        self._scale = max(0.1, min(5.0, self._scale))
        self._update_display()

    def _fit(self):
        if self._pixmap:
            vp = self.scroll.viewport().size()
            w_ratio = vp.width() / self._pixmap.width()
            h_ratio = vp.height() / self._pixmap.height()
            self._scale = min(w_ratio, h_ratio) * 0.95
            self._update_display()
