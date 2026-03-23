"""Shared pytest configuration and fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src layout is on sys.path so tests can import the package
# without a full pip install (works alongside an editable install too).
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
