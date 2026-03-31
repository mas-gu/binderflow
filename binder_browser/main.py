"""Entry point with argument parsing."""

import sys
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


def main():
    # Fix OpenGL rendering on systems with NVIDIA compute + Mesa display
    import os
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    parser = argparse.ArgumentParser(description="Binder Browser — explore protein binder design results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results directory (parent of run folders)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Binder Browser")
    app.setStyle("Fusion")

    from .style import DARK_THEME
    app.setStyleSheet(DARK_THEME)

    from .app import MainWindow
    window = MainWindow(results_dir)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
