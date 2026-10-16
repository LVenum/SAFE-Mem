#!/usr/bin/env python
"""Worker for the MINJA baseline arms; started by minja_audit.py, not by hand.
Logic in ams.eval.minja_defence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ams.eval.minja_defence import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
