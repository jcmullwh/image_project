from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Mapping

from context_injectors import InjectionResult, InjectorContext, register_injector


@dataclass(frozen=True)
class _Holiday:
    key: str
    label: str
    date_for_year: Callable[[int], date]


def _fixed(month: int, day: int) -> Callable[[int], date]:
    def _fn(year: int) -> date:
        return date(year, month, day)

    return _fn


def _us_thanksgiving(year: int) -> date:
    # 4th Thursday in November.
    cal = _calendar.Calendar(firstweekday=_calendar.SUNDAY)
    thursdays = [
        d
        for d in cal.itermonthdates(year, 11)
        if d.month == 11 and d.weekday() == _calendar.THURSDAY
    ]
    return thursdays[3]


_HOLIDAYS: tuple[_Holiday, ...] = (
    _Holiday("new_years_day", "New Year’s Day", _fixed(1, 1)),
    _Holiday("valentines_day", "Valentine’s Day", _fixed(2, 14)),
    _Holiday("halloween", "Halloween", _fixed(10, 31)),
    _Holiday("thanksgiving", "Thanksgiving", _us_thanksgiving),
    _Holiday("christmas", "Christmas", _fixed(12, 25)),
)


def _parse_int(value: Any, path: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Invalid config type for {path}: expected int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid config value for {path}: must be an int") from exc
    raise ValueError(f"Invalid config type for {path}: expected int")


def _parse_float(value: Any, path: str, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"Invalid config type for {path}: expected float, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid config value for {path}: must be a float") from exc
    raise ValueError(f"Invalid config type for {path}: expected float")


def _clamp_probability(value: float, path: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Invalid config value for {path}: must be between 0 and 1")
    return value


def _parse_dislikes_contains(preferences_guidance: str | None, needle: str) -> bool:
    if not preferences_guidance:
        return False

    in_dislikes = False
    for raw_line in preferences_guidance.splitlines():
        line = raw_line.strip()
        if not line:
            in_dislikes = False
            continue
        if line.endswith(":"):
            in_dislikes = line[:-1].strip().lower() == "dislikes"
            continue
        if in_dislikes and line.startswith("-"):
            if needle.lower() in line.lower():
                return True

    return False


def _next_occurrence(holiday: _Holiday, today: date) -> date:
    this_year = holiday.date_for_year(today.year)
    if this_year >= today:
        return this_year
    return holiday.date_for_year(today.year + 1)


@register_injector
class HolidayInjector:
    name = "holiday"

    def build(self, ctx: InjectorContext) -> InjectionResult:
        holiday_cfg = ctx.get_mapping("holiday")
        lookahead_days = _parse_int(
            holiday_cfg.get("lookahead_days"), "context.holiday.lookahead_days", default=14
        )
        base_probability = _clamp_probability(
            _parse_float(
                holiday_cfg.get("base_probability"),
                "context.holiday.base_probability",
                default=0.15,
            ),
            "context.holiday.base_probability",
        )
        max_probability = _clamp_probability(
            _parse_float(
                holiday_cfg.get("max_probability"),
                "context.holiday.max_probability",
                default=0.55,
            ),
            "context.holiday.max_probability",
        )

        if lookahead_days <= 0:
            raise ValueError("context.holiday.lookahead_days must be > 0")
        if max_probability < base_probability:
            raise ValueError("context.holiday.max_probability must be >= base_probability")

        today = ctx.today
        window_end = today + timedelta(days=lookahead_days)

        upcoming: list[tuple[_Holiday, date, int]] = []
        for holiday in _HOLIDAYS:
            when = _next_occurrence(holiday, today)
            if when > window_end:
                continue
            days_until = (when - today).days
            if days_until < 0:
                continue
            upcoming.append((holiday, when, days_until))

        if not upcoming:
            if ctx.logger:
                ctx.logger.info("Holiday injector: no holidays within %d days", lookahead_days)
            return InjectionResult(
                guidance_lines=[],
                metadata={
                    "applied": False,
                    "reason": "no_holiday_in_window",
                    "date_used": today.isoformat(),
                    "lookahead_days": lookahead_days,
                },
            )

        holiday, when, days_until = min(upcoming, key=lambda item: item[2])
        closeness = (lookahead_days - days_until) / float(lookahead_days)
        probability = base_probability + (max_probability - base_probability) * closeness
        roll = ctx.rng.random()
        applied = roll < probability

        if ctx.logger:
            ctx.logger.info(
                "Holiday injector: next=%s in %d days (p=%.3f roll=%.3f applied=%s). You MUST adopt it as an additional theme.",
                holiday.label,
                days_until,
                probability,
                roll,
                applied,
            )

        metadata: dict[str, Any] = {
            "applied": applied,
            "date_used": today.isoformat(),
            "holiday": holiday.label,
            "holiday_key": holiday.key,
            "holiday_date": when.isoformat(),
            "days_until": days_until,
            "lookahead_days": lookahead_days,
            "base_probability": base_probability,
            "max_probability": max_probability,
            "computed_probability": probability,
            "roll": roll,
        }

        if not applied:
            return InjectionResult(guidance_lines=[], metadata=metadata)

        dislikes_horror = _parse_dislikes_contains(ctx.preferences_guidance, "horror")

        if holiday.key == "halloween" and dislikes_horror:
            hook = (
                "Halloween proximity - incorporate playful, non-horror motifs (costumes, pumpkins, autumn light)."
            )
        else:
            hook = (
                f"Holiday proximity: {holiday.label} - incorporate it clearly into the scene "
                "(motifs/decor/lighting/colors)."
            )

        guidance = f"{hook} You MUST adopt it as an additional theme."

        return InjectionResult(guidance_lines=[guidance], metadata=metadata)
