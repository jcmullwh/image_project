from __future__ import annotations

"""Deprecated experiment runner wrapper.

Use:
  - `python -m image_project experiments run profile_v5_3x3 ...`
  - `pdm run experiment-profile-v5-3x3 ...`
"""

import sys
from pathlib import Path


# Ensure project root is importable when running as a loose script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_project.app.experiments_cli import main as experiments_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to the canonical experiment CLI for `profile_v5_3x3`."""

    args = list(argv) if argv is not None else sys.argv[1:]
    return int(experiments_main(["run", "profile_v5_3x3", *args]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

