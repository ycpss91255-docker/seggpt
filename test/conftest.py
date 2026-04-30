"""pytest bootstrap for the seggpt repo.

Adds ``src/`` to ``sys.path`` so the in-tree ``seggpt`` package is
importable without ``pip install -e .``. Inside the docker image the
package is installed properly; this shim is only for host-side runs
during local TDD.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
