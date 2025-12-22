from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


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


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")
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


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def extract_recent_motif_summary(*, generations_csv_path: str, window: int) -> dict[str, Any]:
    if window <= 0:
        return {"enabled": False, "window": window, "rows_considered": 0, "top_tokens": []}

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

    counts: Counter[str] = Counter()
    for prompt in prompts:
        counts.update(tokenize(prompt))

    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_tokens = [{"token": token, "count": count} for token, count in top[:30] if count >= 2]

    motif_watchlist = ("sunset", "sunrise", "tree", "trees", "ocean", "moon")
    motif_counts = {m: counts.get(m, 0) for m in motif_watchlist if counts.get(m, 0) > 0}

    return {
        "enabled": True,
        "window": window,
        "rows_considered": len(prompts),
        "top_tokens": top_tokens,
        "motif_counts": motif_counts,
    }


def novelty_penalties(
    idea_cards: list[Mapping[str, Any]], novelty_summary: Mapping[str, Any]
) -> dict[str, int]:
    if not novelty_summary or not novelty_summary.get("enabled"):
        return {str(card.get("id")): 0 for card in idea_cards}

    top_tokens = novelty_summary.get("top_tokens")
    if not isinstance(top_tokens, list):
        return {str(card.get("id")): 0 for card in idea_cards}

    repeated: dict[str, int] = {}
    for item in top_tokens:
        if not isinstance(item, Mapping):
            continue
        token = item.get("token")
        count = item.get("count")
        if isinstance(token, str) and isinstance(count, int) and count >= 2:
            repeated[token] = count

    penalties: dict[str, int] = {}
    for card in idea_cards:
        card_id = str(card.get("id"))
        blob = json.dumps(card, ensure_ascii=False)
        card_tokens = set(tokenize(blob))
        points = 0
        for token, count in repeated.items():
            if token in card_tokens:
                points += min(count, 5)
        penalties[card_id] = min(20, points)

    return penalties


def select_candidate(
    *,
    scores: Mapping[str, int],
    idea_cards: list[Mapping[str, Any]],
    exploration_rate: float,
    rng: random.Random,
    novelty_summary: Mapping[str, Any] | None = None,
) -> SelectionResult:
    if not scores:
        raise ValueError("selection inconsistency: empty score table")

    cards_by_id = {str(card.get("id")): card for card in idea_cards}
    missing_cards = sorted(set(scores.keys()) - set(cards_by_id.keys()))
    if missing_cards:
        raise ValueError(f"selection inconsistency: missing idea cards for ids={missing_cards}")

    penalties = novelty_penalties(idea_cards, novelty_summary or {})

    table: list[dict[str, Any]] = []
    for idea_id, raw_score in scores.items():
        penalty = int(penalties.get(idea_id, 0))
        effective = max(0, int(raw_score) - penalty)
        table.append(
            {
                "id": idea_id,
                "score": int(raw_score),
                "novelty_penalty": penalty,
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

