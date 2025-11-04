"""Thin runner that keeps ``streamlit run dashboard.py`` working after reorganisation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    # When executed as part of the ``dashboard`` package (e.g. `python -m dashboard.dashboard`).
    from .pages.dashboard import main  # type: ignore[import]
except ImportError:
    # When executed as a standalone script (e.g. `streamlit run dashboard/dashboard.py`).
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from pages.dashboard import main  # type: ignore[import]


if __name__ == "__main__":
    main()
