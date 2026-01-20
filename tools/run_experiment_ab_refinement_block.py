from __future__ import annotations

"""Deprecated experiment runner wrapper.

Use:
  - `python -m image_project experiments run ab_refinement_block ...`
  - `pdm run experiment-ab-refinement-block ...`
"""

import sys
from pathlib import Path


# Ensure project root is importable when running as a loose script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_project.app.experiments_cli import main as experiments_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to the canonical experiment CLI for `ab_refinement_block`."""

    args = list(argv) if argv is not None else sys.argv[1:]
    return int(experiments_main(["run", "ab_refinement_block", *args]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

