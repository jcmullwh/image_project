from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from image_project.framework.config import PromptNoveltyConfig


@dataclass(frozen=True)
class SelectionResult:
    selected_id: str
    selected_score: int
    selected_effective_score: int
    exploration_roll: float
    selection_mode: str
    score_table: list[dict[str, Any]]


def expected_idea_ids(num_ideas: int) -> list[str]:
    if num_ideas <= 0:
        raise ValueError("num_ideas must be > 0")

    def _excel_col(index: int) -> str:
        # 0-based -> A, B, ..., Z, AA, AB, ...
        n = index + 1
        letters: list[str] = []
        while n > 0:
            n, rem = divmod(n - 1, 26)
            letters.append(chr(ord("A") + rem))
        return "".join(reversed(letters))

    return [_excel_col(i) for i in range(num_ideas)]


def _as_nonempty_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid_idea_cards_json: {path} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"invalid_idea_cards_json: {path} must be non-empty")
    return text


def _as_str_list(value: Any, path: str, *, min_len: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"invalid_idea_cards_json: {path} must be a list[str]")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"invalid_idea_cards_json: {path}[{idx}] must be a string")
        text = item.strip()
        if text:
            out.append(text)
    if len(out) < min_len:
        raise ValueError(f"invalid_idea_cards_json: {path} must have >= {min_len} items")
    return out


def _as_optional_str_list(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"invalid_idea_cards_json: {path} must be a list[str] if provided")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"invalid_idea_cards_json: {path}[{idx}] must be a string")
        text = item.strip()
        if text:
            out.append(text)
    return out


def parse_idea_cards_json(text: str, *, expected_num_ideas: int) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid_idea_cards_json: JSON parse error: {exc}") from exc

    ideas_raw: Any
    if isinstance(payload, list):
        ideas_raw = payload
    elif isinstance(payload, Mapping) and "ideas" in payload:
        ideas_raw = payload.get("ideas")
    else:
        raise ValueError("invalid_idea_cards_json: expected a list or an object with key 'ideas'")

    if not isinstance(ideas_raw, list):
        raise ValueError("invalid_idea_cards_json: ideas must be a list")
    if len(ideas_raw) != expected_num_ideas:
        raise ValueError(
            "invalid_idea_cards_json: wrong number of ideas "
            f"(expected={expected_num_ideas} got={len(ideas_raw)})"
        )

    expected_ids = expected_idea_ids(expected_num_ideas)
    seen: set[str] = set()
    ideas: list[dict[str, Any]] = []

    for idx, idea in enumerate(ideas_raw):
        if not isinstance(idea, Mapping):
            raise ValueError(f"invalid_idea_cards_json: ideas[{idx}] must be an object")

        idea_id = _as_nonempty_str(idea.get("id"), f"ideas[{idx}].id")
        if idea_id in seen:
            raise ValueError(f"invalid_idea_cards_json: duplicate idea id: {idea_id}")
        seen.add(idea_id)

        hook = _as_nonempty_str(idea.get("hook"), f"ideas[{idx}].hook")
        narrative = _as_nonempty_str(idea.get("narrative"), f"ideas[{idx}].narrative")

        options = idea.get("options")
        if not isinstance(options, Mapping):
            raise ValueError(f"invalid_idea_cards_json: ideas[{idx}].options must be an object")

        composition = _as_str_list(options.get("composition"), f"ideas[{idx}].options.composition", min_len=2)
        palette = _as_str_list(options.get("palette"), f"ideas[{idx}].options.palette", min_len=2)
        medium = _as_str_list(options.get("medium"), f"ideas[{idx}].options.medium", min_len=1)
        mood = _as_str_list(options.get("mood"), f"ideas[{idx}].options.mood", min_len=1)

        avoid = _as_optional_str_list(idea.get("avoid"), f"ideas[{idx}].avoid")

        ideas.append(
            {
                "id": idea_id,
                "hook": hook,
                "narrative": narrative,
                "options": {
                    "composition": composition,
                    "palette": palette,
                    "medium": medium,
                    "mood": mood,
                },
                "avoid": avoid,
            }
        )

    missing = sorted(set(expected_ids) - seen)
    extra = sorted(seen - set(expected_ids))
    if missing or extra:
        raise ValueError(
            "invalid_idea_cards_json: idea ids must match expected stable ids "
            f"(missing={missing} extra={extra})"
        )

    return ideas


def parse_idea_card_json(text: str, *, expected_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid_idea_card_json: JSON parse error: {exc}") from exc

    idea_raw: Any = payload
    if isinstance(payload, Mapping) and "idea" in payload:
        idea_raw = payload.get("idea")
    elif isinstance(payload, Mapping) and "ideas" in payload:
        ideas = payload.get("ideas")
        if not isinstance(ideas, list) or len(ideas) != 1:
            raise ValueError("invalid_idea_card_json: ideas must be a list with exactly 1 item")
        idea_raw = ideas[0]

    if not isinstance(idea_raw, Mapping):
        raise ValueError("invalid_idea_card_json: expected an object")

    raw_id = idea_raw.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("invalid_idea_card_json: id must be a non-empty string")
    idea_id = raw_id.strip()
    if idea_id != expected_id:
        raise ValueError(
            "invalid_idea_card_json: wrong id "
            f"(expected={expected_id!r} got={idea_id!r})"
        )

    hook = idea_raw.get("hook")
    if not isinstance(hook, str) or not hook.strip():
        raise ValueError("invalid_idea_card_json: hook must be a non-empty string")
    narrative = idea_raw.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("invalid_idea_card_json: narrative must be a non-empty string")

    options = idea_raw.get("options")
    if not isinstance(options, Mapping):
        raise ValueError("invalid_idea_card_json: options must be an object")

    composition = options.get("composition")
    if not isinstance(composition, list):
        raise ValueError("invalid_idea_card_json: options.composition must be a list[str]")
    composition_list = [str(item).strip() for item in composition if isinstance(item, str) and item.strip()]
    if len(composition_list) < 2:
        raise ValueError("invalid_idea_card_json: options.composition must have >= 2 items")

    palette = options.get("palette")
    if not isinstance(palette, list):
        raise ValueError("invalid_idea_card_json: options.palette must be a list[str]")
    palette_list = [str(item).strip() for item in palette if isinstance(item, str) and item.strip()]
    if len(palette_list) < 2:
        raise ValueError("invalid_idea_card_json: options.palette must have >= 2 items")

    medium = options.get("medium")
    if not isinstance(medium, list):
        raise ValueError("invalid_idea_card_json: options.medium must be a list[str]")
    medium_list = [str(item).strip() for item in medium if isinstance(item, str) and item.strip()]
    if len(medium_list) < 1:
        raise ValueError("invalid_idea_card_json: options.medium must have >= 1 items")

    mood = options.get("mood")
    if not isinstance(mood, list):
        raise ValueError("invalid_idea_card_json: options.mood must be a list[str]")
    mood_list = [str(item).strip() for item in mood if isinstance(item, str) and item.strip()]
    if len(mood_list) < 1:
        raise ValueError("invalid_idea_card_json: options.mood must have >= 1 items")

    avoid_raw = idea_raw.get("avoid")
    avoid: list[str] = []
    if avoid_raw is None:
        avoid = []
    elif isinstance(avoid_raw, list):
        for idx, item in enumerate(avoid_raw):
            if not isinstance(item, str):
                raise ValueError(f"invalid_idea_card_json: avoid[{idx}] must be a string")
            text_item = item.strip()
            if text_item:
                avoid.append(text_item)
    else:
        raise ValueError("invalid_idea_card_json: avoid must be a list[str] if provided")

    return {
        "id": idea_id,
        "hook": hook.strip(),
        "narrative": narrative.strip(),
        "options": {
            "composition": composition_list,
            "palette": palette_list,
            "medium": medium_list,
            "mood": mood_list,
        },
        "avoid": avoid,
    }


def parse_judge_scores_json(text: str, *, expected_ids: Iterable[str]) -> dict[str, int]:
    try:
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        preview = (text or "")[:200]
        raise ValueError(f"invalid_judge_output: JSON parse error: {exc}; preview={preview!r}") from exc

    if not isinstance(payload, Mapping) or "scores" not in payload:
        preview = (text or "")[:200]
        raise ValueError(f"invalid_judge_output: expected object with key 'scores'; preview={preview!r}")

    scores_raw = payload.get("scores")
    if not isinstance(scores_raw, list):
        preview = (text or "")[:200]
        raise ValueError(f"invalid_judge_output: scores must be a list; preview={preview!r}")

    expected = set(expected_ids)
    seen: set[str] = set()
    scores: dict[str, int] = {}

    for idx, item in enumerate(scores_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid_judge_output: scores[{idx}] must be an object")

        raw_id = item.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError(f"invalid_judge_output: scores[{idx}].id must be a non-empty string")
        idea_id = raw_id.strip()

        if idea_id in seen:
            raise ValueError(f"invalid_judge_output: duplicate id in scores: {idea_id}")
        seen.add(idea_id)

        raw_score = item.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, int):
            raise ValueError(f"invalid_judge_output: scores[{idx}].score must be an int")
        if not (0 <= raw_score <= 100):
            raise ValueError(f"invalid_judge_output: scores[{idx}].score must be in [0, 100]")

        scores[idea_id] = raw_score

    missing = sorted(expected - set(scores.keys()))
    extra = sorted(set(scores.keys()) - expected)
    if missing or extra:
        raise ValueError(
            "invalid_judge_output: score ids must match idea ids "
            f"(missing={missing} extra={extra})"
        )

    return scores


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "over",
        "the",
        "this",
        "to",
        "with",
        "without",
        "your",
        "you",
        "we",
        "our",
        "they",
        "their",
        "that",
        "these",
        "those",
        "image",
        "prompt",
    }
)


def _load_recent_final_prompts(*, generations_csv_path: str, window: int) -> deque[str]:
    if not generations_csv_path:
        raise ValueError("generations_csv_path is empty")

    prompts: deque[str] = deque(maxlen=window)
    with open(generations_csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("generations csv has no header")
        if "final_image_prompt" not in reader.fieldnames:
            raise ValueError("generations csv missing required column: final_image_prompt")
        for row in reader:
            prompts.append(str(row.get("final_image_prompt", "") or ""))

    return prompts


_TOKEN_ALPHA_RE_V1 = re.compile(r"[a-zA-Z]+")
_TOKEN_ALNUM_RE_V1 = re.compile(r"[a-zA-Z0-9]+")

_PROMPT_SCAFFOLDING_STOPWORDS_V1: frozenset[str] = frozenset(
    {
        "prompt",
        "image",
        "scene",
        "style",
        "composition",
        "lighting",
        "camera",
        "lens",
        "shot",
        "subject",
        "background",
        "foreground",
        "render",
        "rendered",
        "rendering",
        "rules",
        "rule",
        "constraints",
        "constraint",
        "requirements",
        "requirement",
        "instructions",
        "instruction",
        "must",
        "avoid",
        "include",
        "including",
        "exclude",
        "excluding",
        "output",
        "section",
        "format",
        "json",
        "strict",
        "highly",
        "detailed",
        "detail",
        "photorealistic",
        "hyperrealistic",
        "realistic",
    }
)

_STOPWORDS_V1_BASE: frozenset[str] = frozenset(set(_STOPWORDS) | set(_PROMPT_SCAFFOLDING_STOPWORDS_V1))


def tokenize_v1(
    text: str,
    *,
    min_len: int,
    stopwords: frozenset[str],
    alpha_only: bool = True,
) -> list[str]:
    if not text:
        return []
    pattern = _TOKEN_ALPHA_RE_V1 if alpha_only else _TOKEN_ALNUM_RE_V1
    tokens = [t.lower() for t in pattern.findall(text)]
    out: list[str] = []
    for token in tokens:
        if len(token) < int(min_len):
            continue
        if token in stopwords:
            continue
        if not alpha_only and token.isdigit():
            continue
        out.append(token)
    return out


def _stopwords_fingerprint(stopwords: Iterable[str]) -> tuple[int, str]:
    items = sorted({str(s) for s in stopwords if str(s)})
    payload = "\n".join(items).encode("utf-8")
    return len(items), hashlib.sha256(payload).hexdigest()


def _extract_recent_motif_summary_df_overlap_v1(
    *,
    generations_csv_path: str,
    cfg: PromptNoveltyConfig,
) -> dict[str, Any]:
    window = int(cfg.window)
    if window <= 0:
        return {"enabled": False, "window": window, "rows_considered": 0, "top_tokens": []}

    prompts = _load_recent_final_prompts(generations_csv_path=generations_csv_path, window=window)

    extra_stopwords = sorted({s.strip().lower() for s in cfg.stopwords_extra if s.strip()})
    stopwords = frozenset(set(_STOPWORDS_V1_BASE) | set(extra_stopwords))
    stopwords_count, stopwords_hash = _stopwords_fingerprint(stopwords)

    df_counts: Counter[str] = Counter()
    for prompt in prompts:
        df_counts.update(
            set(
                tokenize_v1(
                    prompt,
                    min_len=int(cfg.min_token_len),
                    stopwords=stopwords,
                    alpha_only=bool(cfg.alpha_only),
                )
            )
        )

    df_min = int(cfg.df_min)
    items = [(token, int(df)) for token, df in df_counts.items() if int(df) >= df_min]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    items = items[: int(cfg.max_motifs)]

    df_cap = int(cfg.df_cap)
    motifs = [
        {"token": token, "df": int(df), "w": int(min(int(df), df_cap))} for token, df in items
    ]
    total_weight = sum(int(item["w"]) for item in motifs)

    top_tokens = [{"token": item["token"], "count": int(item["df"])} for item in motifs[:30]]

    motif_watchlist = ("sunset", "sunrise", "tree", "trees", "ocean", "moon")
    motif_counts = {m: int(df_counts.get(m, 0)) for m in motif_watchlist if int(df_counts.get(m, 0)) > 0}

    return {
        "enabled": True,
        "method": "df_overlap_v1",
        "window": window,
        "rows_considered": len(prompts),
        "doc_count": len(prompts),
        "df_min": df_min,
        "df_cap": df_cap,
        "max_motifs": int(cfg.max_motifs),
        "min_token_len": int(cfg.min_token_len),
        "alpha_only": bool(cfg.alpha_only),
        "stopwords_extra": extra_stopwords,
        "stopwords_count": int(stopwords_count),
        "stopwords_hash": str(stopwords_hash),
        "motifs": motifs,
        "motif_counts": motif_counts,
        "total_weight": int(total_weight),
        "top_tokens": top_tokens,
    }


def extract_recent_motif_summary(
    *, generations_csv_path: str, novelty_cfg: PromptNoveltyConfig
) -> dict[str, Any]:
    method = str(getattr(novelty_cfg, "method", "") or "").strip().lower()
    if method == "df_overlap_v1":
        return _extract_recent_motif_summary_df_overlap_v1(
            generations_csv_path=generations_csv_path, cfg=novelty_cfg
        )
    raise ValueError(f"Unknown novelty method: {method!r}")


def _novelty_penalties_df_overlap_v1(
    candidates: list[Mapping[str, Any]],
    novelty_summary: Mapping[str, Any],
    *,
    text_field: str,
    cfg: PromptNoveltyConfig,
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    penalties: dict[str, int] = {str(card.get("id")): 0 for card in candidates}
    breakdown: dict[str, dict[str, Any]] = {
        cid: {"penalty": 0, "reason": "novelty_disabled"} for cid in penalties
    }

    if not novelty_summary or not novelty_summary.get("enabled"):
        return penalties, breakdown

    motifs_raw = novelty_summary.get("motifs")
    if not isinstance(motifs_raw, list):
        breakdown = {cid: {"penalty": 0, "reason": "invalid_novelty_summary"} for cid in penalties}
        return penalties, breakdown

    weight_by_token: dict[str, int] = {}
    df_by_token: dict[str, int] = {}
    for item in motifs_raw:
        if not isinstance(item, Mapping):
            continue
        token = item.get("token")
        df = item.get("df")
        if not isinstance(token, str) or not token:
            continue
        if not isinstance(df, int) or df < 1:
            continue
        w_raw = item.get("w")
        w = int(w_raw) if isinstance(w_raw, int) and w_raw >= 0 else int(min(int(df), int(cfg.df_cap)))
        weight_by_token[token] = w
        df_by_token[token] = int(df)

    motif_tokens = set(weight_by_token.keys())
    total_weight_raw = novelty_summary.get("total_weight")
    total_weight = int(total_weight_raw) if isinstance(total_weight_raw, int) and total_weight_raw >= 0 else sum(weight_by_token.values())

    if total_weight <= 0 or not motif_tokens:
        breakdown = {cid: {"penalty": 0, "reason": "no_motifs"} for cid in penalties}
        return penalties, breakdown

    extra_stopwords = sorted({s.strip().lower() for s in cfg.stopwords_extra if s.strip()})
    stopwords = frozenset(set(_STOPWORDS_V1_BASE) | set(extra_stopwords))

    max_penalty = int(cfg.max_penalty)
    scaling = str(cfg.scaling or "linear").strip().lower()
    for card in candidates:
        cid = str(card.get("id"))
        text = card.get(text_field)
        text_value = text if isinstance(text, str) else ""
        cand_tokens = set(
            tokenize_v1(
                text_value,
                min_len=int(cfg.min_token_len),
                stopwords=stopwords,
                alpha_only=bool(cfg.alpha_only),
            )
        )
        overlap = cand_tokens.intersection(motif_tokens)
        overlap_weight = sum(int(weight_by_token.get(tok, 0)) for tok in overlap)
        frac = float(overlap_weight) / float(total_weight) if total_weight else 0.0
        frac = min(1.0, max(0.0, frac))

        if scaling == "sqrt":
            scaled = math.sqrt(frac)
        elif scaling == "quadratic":
            scaled = frac * frac
        else:
            scaled = frac

        raw_penalty = int(round(float(max_penalty) * float(scaled)))
        penalty = max(0, min(int(max_penalty), int(raw_penalty)))

        overlap_items = sorted(
            overlap, key=lambda tok: (-int(weight_by_token.get(tok, 0)), str(tok))
        )[:10]
        top_motifs = [
            {"token": tok, "w": int(weight_by_token.get(tok, 0)), "df": int(df_by_token.get(tok, 0))}
            for tok in overlap_items
        ]

        penalties[cid] = penalty
        breakdown[cid] = {
            "penalty": int(penalty),
            "max_penalty": int(max_penalty),
            "overlap_weight": int(overlap_weight),
            "total_weight": int(total_weight),
            "overlap_fraction": float(frac),
            "top_motifs": top_motifs,
        }

    return penalties, breakdown


def novelty_penalties(
    candidates: list[Mapping[str, Any]],
    novelty_cfg: PromptNoveltyConfig,
    novelty_summary: Mapping[str, Any] | None,
    *,
    text_field: str,
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    method = str(getattr(novelty_cfg, "method", "") or "").strip().lower()
    if method == "df_overlap_v1":
        return _novelty_penalties_df_overlap_v1(
            candidates,
            novelty_summary or {},
            text_field=text_field,
            cfg=novelty_cfg,
        )
    raise ValueError(f"Unknown novelty method: {method!r}")

def _idea_card_text_for_novelty(card: Mapping[str, Any]) -> str:
    parts: list[str] = []

    hook = card.get("hook")
    if isinstance(hook, str) and hook.strip():
        parts.append(hook.strip())

    narrative = card.get("narrative")
    if isinstance(narrative, str) and narrative.strip():
        parts.append(narrative.strip())

    avoid_raw = card.get("avoid")
    if isinstance(avoid_raw, list):
        avoids = [str(item).strip() for item in avoid_raw if str(item).strip()]
        if avoids:
            parts.append("\n".join(avoids))

    options = card.get("options")
    if isinstance(options, Mapping):
        for key in ("composition", "palette", "medium", "mood"):
            raw = options.get(key)
            if isinstance(raw, list):
                items = [str(item).strip() for item in raw if str(item).strip()]
                if items:
                    parts.append("\n".join(items))

    return "\n".join(parts)


def select_candidate(
    *,
    scores: Mapping[str, int],
    idea_cards: list[Mapping[str, Any]],
    exploration_rate: float,
    rng: random.Random,
    novelty_cfg: PromptNoveltyConfig | None = None,
    novelty_summary: Mapping[str, Any] | None = None,
) -> SelectionResult:
    if not scores:
        raise ValueError("selection inconsistency: empty score table")

    cards_by_id = {str(card.get("id")): card for card in idea_cards}
    missing_cards = sorted(set(scores.keys()) - set(cards_by_id.keys()))
    if missing_cards:
        raise ValueError(f"selection inconsistency: missing idea cards for ids={missing_cards}")

    effective_novelty_cfg = novelty_cfg or PromptNoveltyConfig(enabled=False, window=0)
    novelty_enabled = bool(effective_novelty_cfg.enabled and effective_novelty_cfg.window > 0)
    novelty_method = str(getattr(effective_novelty_cfg, "method", "") or "").strip().lower()

    penalties: dict[str, int] = {str(card.get("id")): 0 for card in idea_cards}
    novelty_breakdown: dict[str, dict[str, Any]] = {
        card_id: {"penalty": 0, "reason": "novelty_disabled"} for card_id in penalties
    }
    if novelty_enabled:
        if novelty_method != "df_overlap_v1":
            raise ValueError(f"Unknown novelty method: {novelty_method!r}")

        novelty_cards = [
            {"id": str(card.get("id")), "text": _idea_card_text_for_novelty(card)}
            for card in idea_cards
        ]
        penalties, novelty_breakdown = novelty_penalties(
            novelty_cards,
            effective_novelty_cfg,
            novelty_summary,
            text_field="text",
        )

    table: list[dict[str, Any]] = []
    for idea_id, raw_score in scores.items():
        novelty_penalty = int(penalties.get(idea_id, 0))
        effective = max(0, int(raw_score) - novelty_penalty)
        table.append(
            {
                "id": idea_id,
                "score": int(raw_score),
                "novelty_penalty": novelty_penalty,
                "novelty_detail": novelty_breakdown.get(idea_id),
                "effective_score": effective,
            }
        )

    table.sort(key=lambda row: (-row["effective_score"], -row["score"], str(row["id"])))

    exploration_roll = float(rng.random())
    if exploration_rate > 0 and exploration_roll < exploration_rate:
        selection_mode = "explore"
        k = max(1, int(math.ceil(len(table) / 4)))
        pool = table[:k]
        weights = [max(1, int(row["effective_score"])) for row in pool]
        total = sum(weights)
        pick = rng.uniform(0, total)
        running = 0.0
        selected = pool[-1]
        for row, weight in zip(pool, weights, strict=False):
            running += weight
            if pick <= running:
                selected = row
                break
    else:
        selection_mode = "exploit"
        selected = table[0]

    selected_id = str(selected["id"])
    return SelectionResult(
        selected_id=selected_id,
        selected_score=int(selected["score"]),
        selected_effective_score=int(selected["effective_score"]),
        exploration_roll=exploration_roll,
        selection_mode=selection_mode,
        score_table=table,
    )
