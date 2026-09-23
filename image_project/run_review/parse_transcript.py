from __future__ import annotations

import json
from typing import Dict, List, Tuple

from .report_model import RunMetadata, TranscriptStep


class TranscriptParseError(Exception):
    pass


def parse_transcript(path: str) -> Tuple[RunMetadata, List[TranscriptStep]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - safety
        raise TranscriptParseError(f"Invalid JSON transcript: {exc}") from exc

    generation_id = payload.get("generation_id") or payload.get("id")

    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        experiment = None

    prompt_pipeline = None
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        candidate = outputs.get("prompt_pipeline")
        if isinstance(candidate, dict):
            prompt_pipeline = candidate

    metadata = RunMetadata(
        generation_id=generation_id or "unknown",
        seed=payload.get("seed"),
        created_at=payload.get("created_at"),
        selected_concepts=payload.get("selected_concepts"),
        image_path=payload.get("image_path"),
        experiment=experiment,
        prompt_pipeline=prompt_pipeline,
        context=payload.get("context"),
        title_generation=payload.get("title_generation"),
        concept_filter_log=payload.get("concept_filter_log"),
        artifact_paths={"transcript": path},
    )

    steps_data = payload.get("steps") or []
    steps: List[TranscriptStep] = []
    for transcript_index, step in enumerate(steps_data):
        steps.append(
            TranscriptStep(
                name=step.get("name") or step.get("step_name") or step.get("path", "unknown"),
                path=step.get("path") or step.get("name") or "unknown",
                transcript_index=transcript_index,
                prompt=step.get("prompt"),
                response=step.get("response"),
                prompt_chars=step.get("prompt_chars"),
                input_chars=step.get("input_chars"),
                context_chars=step.get("context_chars"),
                response_chars=step.get("response_chars"),
                metadata={
                    k: v
                    for k, v in step.items()
                    if k
                    not in {
                        "name",
                        "path",
                        "prompt",
                        "response",
                        "prompt_chars",
                        "input_chars",
                        "context_chars",
                        "response_chars",
                    }
                },
            )
        )

    return metadata, steps
