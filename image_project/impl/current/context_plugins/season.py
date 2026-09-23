from __future__ import annotations

from datetime import date
from typing import Any, Mapping

from image_project.framework.context import InjectionResult, InjectorContext, register_injector


def _parse_hemisphere(cfg: Mapping[str, Any]) -> str:
    raw = cfg.get("hemisphere")
    if raw is None:
        return "northern"
    if not isinstance(raw, str):
        raise ValueError("Invalid config type for context.season.hemisphere: expected string")
    value = raw.strip().lower()
    if value in {"north", "northern"}:
        return "northern"
    if value in {"south", "southern"}:
        return "southern"
    raise ValueError(
        "Invalid config value for context.season.hemisphere: expected 'northern' or 'southern'"
    )


def _season_for_month(month: int, *, hemisphere: str) -> str:
    # Simple month mapping, with optional hemisphere inversion.
    if hemisphere == "southern":
        # Shift by 6 months.
        month = ((month + 6 - 1) % 12) + 1

    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


@register_injector
class SeasonInjector:
    name = "season"

    def build(self, ctx: InjectorContext) -> InjectionResult:
        season_cfg = ctx.get_mapping("season")
        hemisphere = _parse_hemisphere(season_cfg)

        today: date = ctx.today
        season = _season_for_month(today.month, hemisphere=hemisphere)

        return InjectionResult(
            guidance_lines=[f"Season: {season} — use seasonal light/atmosphere cues if helpful."],
            metadata={
                "season": season,
                "date_used": today.isoformat(),
                "hemisphere": hemisphere,
            },
        )

