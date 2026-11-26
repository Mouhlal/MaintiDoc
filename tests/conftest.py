"""
conftest.py — Fixtures partagées pour la suite de tests MaintiDoc
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Assurer que la racine maintidoc/ est dans sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
