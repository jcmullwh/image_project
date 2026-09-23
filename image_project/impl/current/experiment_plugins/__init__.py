"""Optional plugin namespace for experiments.

Add new modules under this package that call
`image_project.impl.current.experiments.register_experiment(...)` to extend the
available experiment catalog without editing the canonical runner.
"""

from __future__ import annotations

import importlib
import pkgutil


def discover() -> None:
    """Import all experiment plugin modules under this package."""

    for module in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
        importlib.import_module(module.name)

