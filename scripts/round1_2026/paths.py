"""Where the Round 1 scripts read and write data.

Raw downloads and the price list stay out of git (third-party data), so they
live in a data directory: ``data/cache/r1`` by default, or ``$R1_DATA``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("R1_DATA", REPO / "data" / "cache" / "r1"))
DATA.mkdir(parents=True, exist_ok=True)

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
