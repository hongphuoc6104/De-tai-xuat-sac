#!/usr/bin/env python3
"""Run original-image inventory, frozen feature extraction, or grouped training."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from histology_data.image_classification_cli import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
