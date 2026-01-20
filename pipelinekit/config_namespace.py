"""Strict, stage-owned configuration namespace helper for `pipelinekit`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterable, TypeVar

_MISSING = object()

T = TypeVar("T")


def _join_path(parent: str, key: str) -> str:
    if not parent:
        return key
    return f"{parent}.{key}"


@dataclass
class ConfigNamespace:
    """Small helper for stage-owned config parsing with consumed-keys enforcement."""

    data: Mapping[str, Any]
    path: str
    _consumed: set[str] = field(default_factory=set, init=False, repr=False)
    _children: dict[str, "ConfigNamespace"] = field(default_factory=dict, init=False, repr=False)
    _effective: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def empty(cls, *, path: str) -> "ConfigNamespace":
        return cls({}, path=path)

    def _consume(self, key: str) -> None:
        self._consumed.add(key)

    def consumed_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._consumed))

    def unconsumed_keys(self) -> tuple[str, ...]:
        return tuple(sorted(k for k in self.data.keys() if k not in self._consumed))

    def assert_consumed(self) -> None:
        unknown = list(self.unconsumed_keys())
        if unknown:
            path = self.path or "<root>"
            consumed = ", ".join(self.consumed_keys()) or "<none>"
            stage_instance: str | None = None
            resolved_prefix = "prompt.stage_configs.resolved."
            if path.startswith(resolved_prefix):
                stage_instance = path[len(resolved_prefix) :].strip() or None

            if stage_instance:
                raise ValueError(
                    f"Unknown config keys under {path}: {', '.join(unknown)} "
                    f"(stage: {stage_instance}; consumed: {consumed})"
                )
            raise ValueError(
                f"Unknown config keys under {path}: {', '.join(unknown)} (consumed: {consumed})"
            )
        for child in self._children.values():
            child.assert_consumed()

    def effective_values(self) -> dict[str, Any]:
        out = dict(self._effective)
        for key, child in self._children.items():
            child_effective = child.effective_values()
            if child_effective:
                out[key] = child_effective
        return out

    def set_effective(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not key.strip():
            raise TypeError("ConfigNamespace key must be a non-empty string")
        normalized = key.strip()
        if normalized in self._children:
            raise ValueError(
                f"{_join_path(self.path, normalized)} already accessed as a nested namespace"
            )
        self._consume(normalized)
        self._effective[normalized] = value

    def _record_effective(self, key: str, value: Any) -> None:
        normalized = key.strip()
        self._effective[normalized] = value

    def _get_raw(self, key: str, *, default: Any) -> Any:
        if not isinstance(key, str) or not key.strip():
            raise TypeError("ConfigNamespace key must be a non-empty string")
        normalized = key.strip()
        if normalized in self._children:
            raise ValueError(
                f"{_join_path(self.path, normalized)} already accessed as a nested namespace"
            )

        if normalized not in self.data:
            if default is _MISSING:
                raise ValueError(f"Missing required config key: {_join_path(self.path, normalized)}")
            self._consume(normalized)
            return default

        self._consume(normalized)
        return self.data.get(normalized)

    def namespace(
        self,
        key: str,
        *,
        default: Mapping[str, Any] | None | object = _MISSING,
    ) -> "ConfigNamespace":
        normalized = (key or "").strip()
        if not normalized:
            raise TypeError("ConfigNamespace key must be a non-empty string")
        if normalized in self._children:
            return self._children[normalized]

        if normalized not in self.data:
            if default is _MISSING:
                raise ValueError(f"Missing required config namespace: {_join_path(self.path, normalized)}")
            if default is None:
                self._consume(normalized)
                child = ConfigNamespace.empty(path=_join_path(self.path, normalized))
                self._children[normalized] = child
                return child
            if not isinstance(default, Mapping):
                raise TypeError(
                    f"default for {_join_path(self.path, normalized)} must be a mapping or None"
                )
            self._consume(normalized)
            child = ConfigNamespace(dict(default), path=_join_path(self.path, normalized))
            self._children[normalized] = child
            return child

        raw = self.data.get(normalized)
        self._consume(normalized)

        if raw is None:
            if default is _MISSING:
                raise ValueError(f"Missing required config namespace: {_join_path(self.path, normalized)}")
            if default is None:
                child = ConfigNamespace.empty(path=_join_path(self.path, normalized))
                self._children[normalized] = child
                return child
            if not isinstance(default, Mapping):
                raise TypeError(
                    f"default for {_join_path(self.path, normalized)} must be a mapping or None"
                )
            child = ConfigNamespace(dict(default), path=_join_path(self.path, normalized))
            self._children[normalized] = child
            return child

        if not isinstance(raw, Mapping):
            raise TypeError(
                f"{_join_path(self.path, normalized)} must be a mapping (type={type(raw).__name__})"
            )

        child = ConfigNamespace(dict(raw), path=_join_path(self.path, normalized))
        self._children[normalized] = child
        return child

    def get_bool(self, key: str, *, default: bool | object = _MISSING) -> bool:
        if default is not _MISSING and not isinstance(default, bool):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be a boolean")

        raw = self._get_raw(key, default=default)
        if raw is default and default is not _MISSING:
            value = default
        else:
            value = raw

        if not isinstance(value, bool):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be a boolean (type={type(raw).__name__})"
            )
        self._record_effective(key, value)
        return value

    def get_int(
        self,
        key: str,
        *,
        default: int | object = _MISSING,
        min_value: int | None = None,
        max_value: int | None = None,
        choices: Iterable[int] | None = None,
    ) -> int:
        if default is not _MISSING and (isinstance(default, bool) or not isinstance(default, int)):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be an int")

        raw = self._get_raw(key, default=default)
        if raw is default and default is not _MISSING:
            value = int(default)
            self._record_effective(key, value)
            return value
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be an int (type={type(raw).__name__})"
            )
        value = int(raw)
        if min_value is not None and value < int(min_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be >= {int(min_value)} (got {value})"
            )
        if max_value is not None and value > int(max_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be <= {int(max_value)} (got {value})"
            )
        if choices is not None:
            choice_set = {int(item) for item in choices}
            if value not in choice_set:
                allowed = ", ".join(str(item) for item in sorted(choice_set))
                raise ValueError(
                    f"{_join_path(self.path, key.strip())} must be one of: {allowed} (got {value})"
                )
        self._record_effective(key, value)
        return value

    def get_optional_int(
        self,
        key: str,
        *,
        default: int | None | object = _MISSING,
        min_value: int | None = None,
        max_value: int | None = None,
        choices: Iterable[int] | None = None,
    ) -> int | None:
        """Parse an optional integer value.

        Accepts:
          - int
          - None (explicit null)

        Missing keys behave like `get_int`: if `default` is provided, it is used
        and recorded as an effective value; otherwise the key is required.
        """

        if default is not _MISSING and default is not None and (
            isinstance(default, bool) or not isinstance(default, int)
        ):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be an int or None")

        raw = self._get_raw(key, default=default)
        if raw is None:
            self._record_effective(key, None)
            return None

        if raw is default and default is not _MISSING:
            if default is None:
                self._record_effective(key, None)
                return None
            value = int(default)
            self._record_effective(key, value)
            return value

        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be an int or null (type={type(raw).__name__})"
            )
        value = int(raw)
        if min_value is not None and value < int(min_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be >= {int(min_value)} (got {value})"
            )
        if max_value is not None and value > int(max_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be <= {int(max_value)} (got {value})"
            )
        if choices is not None:
            choice_set = {int(item) for item in choices}
            if value not in choice_set:
                allowed = ", ".join(str(item) for item in sorted(choice_set)) or "<none>"
                raise ValueError(
                    f"{_join_path(self.path, key.strip())} must be one of: {allowed} (got {value})"
                )
        self._record_effective(key, value)
        return value

    def get_float(
        self,
        key: str,
        *,
        default: float | object = _MISSING,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        if default is not _MISSING and (
            isinstance(default, bool) or not isinstance(default, (int, float))
        ):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be a float")

        raw = self._get_raw(key, default=default)
        if raw is default and default is not _MISSING:
            value = float(default)
            self._record_effective(key, value)
            return value
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be a float (type={type(raw).__name__})"
            )
        value = float(raw)
        if min_value is not None and value < float(min_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be >= {float(min_value)} (got {value})"
            )
        if max_value is not None and value > float(max_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be <= {float(max_value)} (got {value})"
            )
        self._record_effective(key, value)
        return value

    def get_optional_float(
        self,
        key: str,
        *,
        default: float | None | object = _MISSING,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float | None:
        """Parse an optional float value (float or null)."""

        if default is not _MISSING and default is not None and (
            isinstance(default, bool) or not isinstance(default, (int, float))
        ):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be a float or None")

        raw = self._get_raw(key, default=default)
        if raw is None:
            self._record_effective(key, None)
            return None

        if raw is default and default is not _MISSING:
            if default is None:
                self._record_effective(key, None)
                return None
            value = float(default)
            self._record_effective(key, value)
            return value

        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be a float or null (type={type(raw).__name__})"
            )
        value = float(raw)
        if min_value is not None and value < float(min_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be >= {float(min_value)} (got {value})"
            )
        if max_value is not None and value > float(max_value):
            raise ValueError(
                f"{_join_path(self.path, key.strip())} must be <= {float(max_value)} (got {value})"
            )
        self._record_effective(key, value)
        return value

    def get_str(
        self,
        key: str,
        *,
        default: str | None | object = _MISSING,
        allow_empty: bool = False,
        choices: Iterable[str] | None = None,
    ) -> str | None:
        if default is not _MISSING and default is not None and not isinstance(default, str):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be a string or None")

        raw = self._get_raw(key, default=default)
        if raw is None:
            self._record_effective(key, None)
            return None

        if not isinstance(raw, str):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be a string (type={type(raw).__name__})"
            )
        value = raw.strip()
        if not value and not allow_empty:
            raise ValueError(f"{_join_path(self.path, key.strip())} cannot be empty")
        if choices is not None:
            choice_set = {str(item).strip() for item in choices if str(item).strip()}
            if value not in choice_set:
                allowed = ", ".join(sorted(choice_set)) or "<none>"
                raise ValueError(
                    f"{_join_path(self.path, key.strip())} must be one of: {allowed} (got {value!r})"
                )
        self._record_effective(key, value)
        return value

    def get_list_str(
        self,
        key: str,
        *,
        default: list[str] | tuple[str, ...] | object = _MISSING,
        allow_empty: bool = False,
    ) -> list[str]:
        if default is not _MISSING and not isinstance(default, (list, tuple)):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be a list[str]")

        raw = self._get_raw(key, default=default)
        if raw is default and default is not _MISSING:
            raw = list(default)  # type: ignore[arg-type]

        if not isinstance(raw, (list, tuple)):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be a list[str] (type={type(raw).__name__})"
            )

        items: list[str] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, str):
                raise TypeError(
                    f"{_join_path(self.path, key.strip())}[{idx}] must be a string (type={type(item).__name__})"
                )
            trimmed = item.strip()
            if not trimmed:
                raise ValueError(f"{_join_path(self.path, key.strip())}[{idx}] cannot be empty")
            items.append(trimmed)

        if not items and not allow_empty:
            raise ValueError(f"{_join_path(self.path, key.strip())} cannot be empty")

        self._record_effective(key, list(items))
        return items

    def get_list_mapping(
        self,
        key: str,
        *,
        default: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | object = _MISSING,
        allow_empty: bool = False,
    ) -> list[dict[str, Any]]:
        """Parse a list of mapping objects (converted to dicts)."""

        if default is not _MISSING and not isinstance(default, (list, tuple)):
            raise TypeError(f"{_join_path(self.path, key.strip())} default must be a list[dict]")

        raw = self._get_raw(key, default=default)
        if raw is default and default is not _MISSING:
            raw = list(default)  # type: ignore[arg-type]

        if not isinstance(raw, (list, tuple)):
            raise TypeError(
                f"{_join_path(self.path, key.strip())} must be a list[dict] (type={type(raw).__name__})"
            )

        items: list[dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"{_join_path(self.path, key.strip())}[{idx}] must be a mapping (type={type(item).__name__})"
                )
            items.append(dict(item))

        if not items and not allow_empty:
            raise ValueError(f"{_join_path(self.path, key.strip())} cannot be empty")

        self._record_effective(key, list(items))
        return items
