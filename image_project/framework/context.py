from __future__ import annotations

import logging
import random
import zlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class InjectionResult:
    guidance_lines: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InjectorContext:
    seed: int
    rng: random.Random
    today: date
    config: Mapping[str, Any]
    preferences_guidance: str | None = None
    logger: logging.Logger | None = None

    def get_mapping(self, key: str) -> Mapping[str, Any]:
        value = self.config.get(key)
        return value if isinstance(value, Mapping) else {}


class ContextInjector(Protocol):
    name: str

    def build(self, ctx: InjectorContext) -> InjectionResult | None: ...


_INJECTOR_REGISTRY: dict[str, type[ContextInjector]] = {}


def register_injector(cls: type[ContextInjector]) -> type[ContextInjector]:
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise TypeError("Context injector must define a non-empty 'name' attribute")

    key = name.strip().lower()
    if key in _INJECTOR_REGISTRY:
        raise ValueError(f"Duplicate context injector name: {key}")

    _INJECTOR_REGISTRY[key] = cls
    return cls


def _stable_crc32(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _injector_seed(seed: int, injector_name: str) -> int:
    # Stable mixing so injectors don't share RNG state (adding an injector
    # doesn't shift randomness for others).
    return seed ^ (_stable_crc32(injector_name) << 1) ^ 0xC0FFEE


class ContextManager:
    @classmethod
    def available_injectors(cls) -> tuple[str, ...]:
        return tuple(sorted(_INJECTOR_REGISTRY.keys()))

    @classmethod
    def build(
        cls,
        *,
        enabled: bool,
        injectors: tuple[str, ...],
        context_cfg: Mapping[str, Any],
        seed: int,
        today: date,
        preferences_guidance: str | None = None,
        logger: logging.Logger | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not enabled:
            return "", {}

        requested: list[str] = []
        for raw in injectors:
            if not isinstance(raw, str):
                raise TypeError("context.injectors must contain only strings")
            name = raw.strip().lower()
            if not name:
                raise ValueError("context.injectors must not contain empty strings")
            requested.append(name)

        if logger:
            logger.info("Context injectors enabled: %s", ", ".join(requested) if requested else "<none>")

        guidance_lines: list[str] = []
        metadata: dict[str, Any] = {}

        for injector_name in requested:
            injector_cls = _INJECTOR_REGISTRY.get(injector_name)
            if injector_cls is None:
                raise ValueError(f"Unknown context injector: {injector_name}")

            injector = injector_cls()
            ctx = InjectorContext(
                seed=seed,
                rng=random.Random(_injector_seed(seed, injector_name)),
                today=today,
                config=context_cfg,
                preferences_guidance=preferences_guidance,
                logger=logger,
            )
            result = injector.build(ctx)
            if result is None:
                continue

            metadata[injector_name] = dict(result.metadata)

            for line in result.guidance_lines:
                text = str(line).strip()
                if not text:
                    continue
                if text.startswith("- "):
                    guidance_lines.append(text[2:])
                else:
                    guidance_lines.append(text)

        if not guidance_lines:
            return "", metadata

        guidance_text = "Context guidance (optional):\n" + "\n".join(
            f"- {line}" for line in guidance_lines
        )
        return guidance_text.strip(), metadata

