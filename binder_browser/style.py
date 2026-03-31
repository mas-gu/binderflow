"""Dark theme stylesheet for Binder Browser."""

DARK_THEME = """
/* === Global === */
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Ubuntu", "Noto Sans", sans-serif;
    font-size: 13px;
}

/* === Menu Bar === */
QMenuBar {
    background-color: #0f3460;
    color: #e0e0e0;
    border-bottom: 1px solid #333;
    padding: 2px;
}
QMenuBar::item {
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #e94560;
}
QMenu {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 4px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #0f3460;
}
QMenu::separator {
    height: 1px;
    background: #333;
    margin: 4px 8px;
}

/* === Buttons === */
QPushButton {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #0f3460;
    border-color: #e94560;
}
QPushButton:pressed {
    background-color: #e94560;
}
QPushButton:disabled {
    background-color: #1a1a2e;
    color: #555;
    border-color: #2a2a3e;
}

/* === ComboBox === */
QComboBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 22px;
}
QComboBox:hover {
    border-color: #e94560;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #e0e0e0;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #333;
    selection-background-color: #0f3460;
    selection-color: #e0e0e0;
    border-radius: 4px;
    padding: 2px;
}

/* === LineEdit / SpinBox === */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 22px;
    selection-background-color: #0f3460;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #e94560;
}

/* === Tab Widget === */
QTabWidget::pane {
    background-color: #1a1a2e;
    border: 1px solid #333;
    border-top: none;
    border-radius: 0px 0px 8px 8px;
}
QTabBar::tab {
    background-color: #16213e;
    color: #aaa;
    border: 1px solid #333;
    border-bottom: none;
    border-radius: 8px 8px 0px 0px;
    padding: 8px 18px;
    margin-right: 2px;
    font-size: 13px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background-color: #0f3460;
    color: #e0e0e0;
    border-bottom: 2px solid #e94560;
}
QTabBar::tab:hover:!selected {
    background-color: #1e2d4a;
    color: #e0e0e0;
}

/* === Table View === */
QTableView, QTableWidget {
    background-color: #16213e;
    alternate-background-color: #1a1a2e;
    color: #e0e0e0;
    gridline-color: transparent;
    border: 1px solid #333;
    border-radius: 4px;
    selection-background-color: #0f3460;
    selection-color: #ffffff;
}
QTableView::item:hover {
    background-color: #1e2d4a;
}
QHeaderView::section {
    background-color: #0f3460;
    color: #e0e0e0;
    border: none;
    border-right: 1px solid #333;
    border-bottom: 1px solid #333;
    padding: 6px 8px;
    font-weight: bold;
}
QHeaderView::section:hover {
    background-color: #e94560;
}
QTableCornerButton::section {
    background-color: #0f3460;
    border: none;
}

/* === Splitter === */
QSplitter::handle {
    background-color: #333;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}
QSplitter::handle:hover {
    background-color: #e94560;
}

/* === Scroll Bars === */
QScrollBar:vertical {
    background-color: #1a1a2e;
    width: 10px;
    border: none;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #333;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #e94560;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background-color: #1a1a2e;
    height: 10px;
    border: none;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #333;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #e94560;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* === Status Bar === */
QStatusBar {
    background-color: #0f3460;
    color: #e0e0e0;
    border-top: 1px solid #333;
    padding: 2px;
}
QStatusBar QLabel {
    color: #e0e0e0;
    padding: 0px 8px;
}

/* === Labels === */
QLabel {
    color: #e0e0e0;
    background: transparent;
}

/* === GroupBox === */
QGroupBox {
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 8px;
    margin-top: 8px;
    padding-top: 16px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    color: #e94560;
}

/* === Check Box / Radio Button === */
QCheckBox, QRadioButton {
    color: #e0e0e0;
    spacing: 6px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #555;
    border-radius: 3px;
    background-color: #16213e;
}
QRadioButton::indicator {
    border-radius: 8px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #e94560;
    border-color: #e94560;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #e94560;
}

/* === Slider === */
QSlider::groove:horizontal {
    background: #333;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #e94560;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #ff6b81;
}

/* === ToolTip === */
QToolTip {
    background-color: #16213e;
    color: #e0e0e0;
    border: 1px solid #e94560;
    border-radius: 4px;
    padding: 4px 8px;
}

/* === List / Tree View === */
QListView, QTreeView, QListWidget, QTreeWidget {
    background-color: #16213e;
    alternate-background-color: #1a1a2e;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 4px;
    selection-background-color: #0f3460;
    selection-color: #ffffff;
}
QListView::item:hover, QTreeView::item:hover {
    background-color: #1e2d4a;
}

/* === Dock Widget === */
QDockWidget {
    color: #e0e0e0;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background-color: #0f3460;
    padding: 6px;
    border-radius: 4px 4px 0 0;
}

/* === Message Box / Dialog === */
QMessageBox, QDialog {
    background-color: #1a1a2e;
}
"""
