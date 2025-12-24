from __future__ import annotations

import sys


try:
    from image_project.app.generate import run_generation as run_generation  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Deprecated entrypoint: import failed. Use `image_project.app.generate.run_generation` "
        "and run via `python -m image_project generate`."
    ) from exc


def main() -> None:
    print(
        "WARNING: Deprecated entrypoint; use `python -m image_project generate` or `pdm run generate`.",
        file=sys.stderr,
    )
    from image_project.app.generate import main as _main

    _main()


if __name__ == "__main__":  # pragma: no cover
    main()

