from __future__ import annotations

from .cli import main


def _main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    _main()

