"""Built-in context injectors for the current implementation."""

from __future__ import annotations

import importlib
import pkgutil


def discover() -> None:
    for module in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
        importlib.import_module(module.name)

