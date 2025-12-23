import json
import os
import re
from pathlib import Path

import pytest

from run_review.cli import main as cli_main
from run_review.parse_oplog import parse_oplog
from run_review.report_builder import build_report, diff_reports, report_to_dict
from run_review.render_html import render_html
from run_review.report_model import RunInputs, RunMetadata, RunReport, SideEffect, StepReport


def _write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def _sample_oplog(tmpdir: Path) -> Path:
    log = tmpdir / "run123_oplog.log"
    _write_file(
        log,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 | INFO | Run started for generation run123",
                "2025-12-21 12:00:00,100 | INFO | No prompt.random_seed configured; generated seed=42",
                "2025-12-21 12:00:00,200 | INFO | Context injectors enabled: season, holiday",
                "2025-12-21 12:00:00,210 | INFO | Holiday injector: next=Christmas in 3 days (p=1.000 roll=0.123 applied=True). You MUST adopt it as an additional theme.",
                "2025-12-21 12:00:00,300 | INFO | Step: pipeline/image_prompt_creation/tot_enclave/consensus_1 (context_chars=30, prompt_chars=90, input_chars=120)",
                "2025-12-21 12:00:00,900 | INFO | Received response for pipeline/image_prompt_creation/tot_enclave/consensus_1 (input_chars=120, chars=240)",
                "2025-12-21 12:00:01,000 | INFO | Image generation request sent (model=gpt-image-1.5, size=1536x1024, quality=high)",
                "2025-12-21 12:00:01,050 | DEBUG | Received image payload length: 1024",
                "2025-12-21 12:00:01,500 | INFO | Upscaling enabled: engine=realesrgan-ncnn-vulkan model=realesrgan-x4plus target=long_edge=3840 aspect=1.778",
                "2025-12-21 12:00:01,800 | INFO | Appended manifest row to manifest.json (seq=1)",
                "2025-12-21 12:00:02,000 | INFO | Uploading final.png to gphotos:album/Test via rclone",
                "2025-12-21 12:00:02,500 | INFO | Uploaded image via rclone to gphotos:album/Test",
                "2025-12-21 12:00:03,000 | INFO | Run completed successfully for generation run123",
            ]
        )
        + "\n",
    )
    return log


def _sample_transcript(tmpdir: Path) -> Path:
    transcript = tmpdir / "run123_transcript.json"
    payload = {
        "generation_id": "run123",
        "seed": 99,
        "created_at": "2025-12-21T12:00:00Z",
        "selected_concepts": ["snow", "lights"],
        "context": {"holiday": "MUST adopt"},
        "title_generation": {"title": "Festive"},
        "concept_filter_log": {"input": "foo", "output": "foo"},
        "steps": [
            {
                "name": "consensus_1",
                "path": "pipeline/image_prompt_creation/tot_enclave/consensus_1",
                "prompt": "Create a winter scene",
                "response": "A snowy village",
                "prompt_chars": 90,
                "input_chars": 120,
                "context_chars": 30,
                "response_chars": 200,
            }
        ],
    }
    transcript.write_text(json.dumps(payload), encoding="utf-8")
    return transcript


def test_parse_oplog_pipe_format_parses_events(tmp_path: Path):
    oplog = _sample_oplog(tmp_path)
    events, unknown, side_effects = parse_oplog(str(oplog))
    assert any(e.type == "run_start" for e in events)
    assert any(e.type == "run_end" for e in events)
    assert any(e.type == "step_start" for e in events)
    assert any(e.type == "step_end" for e in events)
    assert any(se.type == "image_generation_request" for se in side_effects)
    assert unknown == []


def test_parse_oplog_operational_lines_parsed(tmp_path: Path):
    oplog = tmp_path / "run999_oplog.log"
    _write_file(
        oplog,
        "\n".join(
            [
                "2025-12-21 17:29:50,279 | INFO | Operational logging initialized for generation run999",
                r"2025-12-21 17:29:50,282 | DEBUG | Operational log file: C:\logs\run999_oplog.log",
                r"2025-12-21 17:29:50,284 | INFO | Loading prompt data from C:\repo\prompt_files\category_list_v1.csv",
                "2025-12-21 17:29:50,289 | INFO | Loaded 140 category rows",
                r"2025-12-21 17:29:50,289 | INFO | Loading user profile from C:\repo\prompt_files\user_profile_v1.csv",
                "2025-12-21 17:29:50,292 | INFO | Loaded 26 user profile rows",
                "2025-12-21 17:29:50,353 | INFO | Initialized TextAI with model gpt-5.2",
                "2025-12-21 17:29:50,355 | INFO | Random concepts selected (raw): ['Nanotechnology Wonders: The world transformed by advanced nanotechnology.', 'Dutch Angle: Tilting the camera to create a sense of imbalance or tension.', 'Festive: Bright and celebratory colors, often used for holidays or special occasions.']",
                "2025-12-21 17:29:52,676 | INFO | Concept filter dislike_rewrite: input=['Nanotechnology Wonders: The world transformed by advanced nanotechnology.', 'Dutch Angle: Tilting the camera to create a sense of imbalance or tension.', 'Festive: Bright and celebratory colors, often used for holidays or special occasions.'] output=['Nanotechnology Wonders: A vibrant, optimistic world enhanced by friendly everyday nanotechnology, with clear, lively scenes full of human connection.', 'Dutch Angle: A playful, gently tilted viewpoint that adds dynamic energy and excitement while keeping the scene clear and welcoming.', 'Festive: Bright and celebratory colors, often used for holidays or special occasions.']",
                '2025-12-21 17:29:52,678 | DEBUG | Concept filter dislike_rewrite raw response: ["Nanotechnology Wonders: A vibrant, optimistic world enhanced by friendly everyday nanotechnology, with clear, lively scenes full of human connection.", "Dutch Angle: A playful, gently tilted viewpoint that adds dynamic energy and excitement while keeping the scene clear and welcoming.", "Festive: Bright and celebratory colors, often used for holidays or special occasions."]',
                "2025-12-21 17:29:52,680 | INFO | Concepts adjusted after filtering: ['Nanotechnology Wonders: A vibrant, optimistic world enhanced by friendly everyday nanotechnology, with clear, lively scenes full of human connection.', 'Dutch Angle: A playful, gently tilted viewpoint that adds dynamic energy and excitement while keeping the scene clear and welcoming.', 'Festive: Bright and celebratory colors, often used for holidays or special occasions.']",
                "2025-12-21 17:29:52,683 | INFO | Generated first prompt (selected_concepts=3)",
                "2025-12-21 17:38:38,252 | INFO | Initialized ImageAI",
                "2025-12-21 17:38:39,283 | INFO | Assigned image identifier #026 - Neon Winter Rooftop",
            ]
        )
        + "\n",
    )

    events, unknown, side_effects = parse_oplog(str(oplog))
    assert unknown == []

    assert any(se.type == "oplog_init" for se in side_effects)
    assert any(se.type == "data_load" for se in side_effects)
    assert any(se.type == "ai_init" for se in side_effects)
    assert any(se.type == "concepts_raw" for se in side_effects)
    assert any(se.type == "concept_filter" for se in side_effects)
    assert any(se.type == "concept_filter_raw" for se in side_effects)
    assert any(se.type == "concepts_filtered" for se in side_effects)
    assert any(se.type == "first_prompt_generated" for se in side_effects)
    assert any(se.type == "image_identifier" for se in side_effects)

    image_id = next(se for se in side_effects if se.type == "image_identifier")
    assert image_id.data["seq"] == 26
    assert image_id.data["title"] == "Neon Winter Rooftop"

    concept_filter = next(se for se in side_effects if se.type == "concept_filter")
    assert len(concept_filter.data["input"]) == 3
    assert len(concept_filter.data["output"]) == 3


def test_build_report_joins_oplog_and_transcript(tmp_path: Path):
    oplog = _sample_oplog(tmp_path)
    transcript = _sample_transcript(tmp_path)
    report = build_report(RunInputs("run123", oplog_path=str(oplog), transcript_path=str(transcript)))

    assert report.metadata.seed == 99
    assert report.metadata.context == {"holiday": "MUST adopt"}
    assert report.metadata.oplog_stats is not None
    assert len(report.steps) == 1
    step = report.steps[0]
    assert step.step_index == 0
    assert step.timing.duration_ms == pytest.approx(600)
    assert step.prompt_chars == 90
    assert step.oplog_prompt_chars == 90
    assert step.input_chars == 120
    assert step.oplog_response_chars == 240
    assert any(issue.code == "concept_filter_noop" for issue in report.issues)
    assert any(se.type == "image_generation_request" for se in report.side_effects)

    payload = report_to_dict(report)
    assert payload["metadata"]["generation_id"] == "run123"
    assert payload["steps"][0]["step_index"] == 0
    assert payload["steps"][0]["timing"]["duration_ms"] == pytest.approx(600)


def test_best_effort_allows_missing_artifacts(tmp_path: Path):
    oplog = _sample_oplog(tmp_path)
    report = build_report(RunInputs("run123", oplog_path=str(oplog), transcript_path=None), best_effort=True)
    assert any(issue.code == "missing_transcript" for issue in report.issues)


def test_compare_reports_flags_added_and_removed(tmp_path: Path):
    oplog_a = _sample_oplog(tmp_path)
    transcript_a = _sample_transcript(tmp_path)
    report_a = build_report(RunInputs("run123", oplog_path=str(oplog_a), transcript_path=str(transcript_a)))

    # second run has different step path
    oplog_b = tmp_path / "run124_oplog.log"
    _write_file(
        oplog_b,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 | INFO | Run started for generation run124",
                "2025-12-21 12:00:00,100 | INFO | Step: pipeline/image_prompt_creation/tot_enclave/final_consensus (context_chars=0, prompt_chars=5, input_chars=10)",
                "2025-12-21 12:00:00,400 | INFO | Received response for pipeline/image_prompt_creation/tot_enclave/final_consensus (input_chars=10, chars=20)",
                "2025-12-21 12:00:00,900 | INFO | Run completed successfully for generation run124",
            ]
        )
        + "\n",
    )
    transcript_b = tmp_path / "run124_transcript.json"
    transcript_b.write_text(
        json.dumps(
            {
                "generation_id": "run124",
                "steps": [
                    {
                        "name": "final_consensus",
                        "path": "pipeline/image_prompt_creation/tot_enclave/final_consensus",
                        "prompt": "p",
                        "response": "r",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report_b = build_report(RunInputs("run124", oplog_path=str(oplog_b), transcript_path=str(transcript_b)))

    diff = diff_reports(report_a, report_b)
    assert "pipeline/image_prompt_creation/tot_enclave/final_consensus" in diff.added_steps
    assert "pipeline/image_prompt_creation/tot_enclave/consensus_1" in diff.removed_steps


def test_compare_detects_injector_and_upscale_changes(tmp_path: Path):
    oplog_a = tmp_path / "run125_oplog.log"
    _write_file(
        oplog_a,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 | INFO | Run started for generation run125",
                "2025-12-21 12:00:00,200 | INFO | Holiday injector: next=Christmas in 3 days. You should adopt it as an additional theme.",
                "2025-12-21 12:00:01,000 | INFO | upscaling enabled target_long_edge_px=3840",
                "2025-12-21 12:00:02,000 | INFO | Run completed successfully for generation run125",
            ]
        )
        + "\n",
    )
    transcript_a = tmp_path / "run125_transcript.json"
    transcript_a.write_text(json.dumps({"generation_id": "run125", "steps": []}), encoding="utf-8")
    report_a = build_report(RunInputs("run125", oplog_path=str(oplog_a), transcript_path=str(transcript_a)))

    oplog_b = tmp_path / "run126_oplog.log"
    _write_file(
        oplog_b,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 | INFO | Run started for generation run126",
                "2025-12-21 12:00:00,200 | INFO | Holiday injector: next=Christmas in 3 days. You MUST adopt it as an additional theme.",
                "2025-12-21 12:00:01,000 | INFO | Upscaling enabled: engine=realesrgan target=long_edge=3840 aspect=1.778",
                "2025-12-21 12:00:02,000 | INFO | Run completed successfully for generation run126",
            ]
        )
        + "\n",
    )
    transcript_b = tmp_path / "run126_transcript.json"
    transcript_b.write_text(json.dumps({"generation_id": "run126", "steps": []}), encoding="utf-8")
    report_b = build_report(RunInputs("run126", oplog_path=str(oplog_b), transcript_path=str(transcript_b)))

    diff = diff_reports(report_a, report_b)
    assert any("adoption strength" in msg.lower() for msg in diff.injector_diffs)
    assert any("upscale log format" in msg.lower() for msg in diff.post_processing_diffs)


def test_cli_writes_reports(tmp_path: Path):
    oplog = _sample_oplog(tmp_path)
    transcript = _sample_transcript(tmp_path)
    exit_code = cli_main(
        [
            "--generation-id",
            "run123",
            "--oplog",
            str(oplog),
            "--transcript",
            str(transcript),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "run123_run_report.json").exists()
    assert (tmp_path / "run123_run_report.html").exists()


def test_cli_defaults_to_config_and_most_recent(tmp_path: Path):
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "image:",
                f"  log_path: '{tmp_path}'",
                f"  generation_path: '{tmp_path}'",
                f"  upscale_path: '{tmp_path}'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    old_id = "run_old"
    old_oplog = tmp_path / f"{old_id}_oplog.log"
    _write_file(
        old_oplog,
        "\n".join(
            [
                f"2025-12-21 12:00:00,000 | INFO | Run started for generation {old_id}",
                f"2025-12-21 12:00:01,000 | INFO | Run completed successfully for generation {old_id}",
            ]
        )
        + "\n",
    )
    (tmp_path / f"{old_id}_transcript.json").write_text(
        json.dumps({"generation_id": old_id, "steps": []}),
        encoding="utf-8",
    )

    new_id = "run_new"
    new_oplog = tmp_path / f"{new_id}_oplog.log"
    _write_file(
        new_oplog,
        "\n".join(
            [
                f"2025-12-21 12:00:00,000 | INFO | Run started for generation {new_id}",
                f"2025-12-21 12:00:01,000 | INFO | Run completed successfully for generation {new_id}",
            ]
        )
        + "\n",
    )
    (tmp_path / f"{new_id}_transcript.json").write_text(
        json.dumps({"generation_id": new_id, "steps": []}),
        encoding="utf-8",
    )

    os.utime(old_oplog, (1_700_000_000, 1_700_000_000))
    os.utime(tmp_path / f"{old_id}_transcript.json", (1_700_000_000, 1_700_000_000))
    os.utime(new_oplog, (1_700_000_100, 1_700_000_100))
    os.utime(tmp_path / f"{new_id}_transcript.json", (1_700_000_100, 1_700_000_100))

    exit_code = cli_main(["--config", str(config_path), "--output-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / f"{new_id}_run_report.json").exists()
    assert (tmp_path / f"{new_id}_run_report.html").exists()


def test_most_recent_skips_incomplete_runs_without_best_effort(tmp_path: Path):
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "image:",
                f"  log_path: '{tmp_path}'",
                f"  generation_path: '{tmp_path}'",
                f"  upscale_path: '{tmp_path}'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    complete_id = "run_complete"
    complete_oplog = tmp_path / f"{complete_id}_oplog.log"
    _write_file(
        complete_oplog,
        "\n".join(
            [
                f"2025-12-21 12:00:00,000 | INFO | Run started for generation {complete_id}",
                f"2025-12-21 12:00:01,000 | INFO | Run completed successfully for generation {complete_id}",
            ]
        )
        + "\n",
    )
    (tmp_path / f"{complete_id}_transcript.json").write_text(
        json.dumps({"generation_id": complete_id, "steps": []}),
        encoding="utf-8",
    )

    incomplete_id = "run_incomplete"
    (tmp_path / f"{incomplete_id}_transcript.json").write_text(
        json.dumps({"generation_id": incomplete_id, "steps": []}),
        encoding="utf-8",
    )

    os.utime(complete_oplog, (1_700_000_000, 1_700_000_000))
    os.utime(tmp_path / f"{complete_id}_transcript.json", (1_700_000_000, 1_700_000_000))
    os.utime(tmp_path / f"{incomplete_id}_transcript.json", (1_700_000_100, 1_700_000_100))

    exit_code = cli_main(
        [
            "--config",
            str(config_path),
            "--most-recent",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / f"{complete_id}_run_report.json").exists()


def test_transcript_order_is_preserved_when_oplog_has_no_steps(tmp_path: Path):
    transcript = tmp_path / "run200_transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "generation_id": "run200",
                "steps": [
                    {"name": "b", "path": "pipeline/section/b", "prompt": "p", "response": "r"},
                    {"name": "a", "path": "pipeline/section/a", "prompt": "p", "response": "r"},
                    {"name": "c", "path": "pipeline/section/c", "prompt": "p", "response": "r"},
                ],
            }
        ),
        encoding="utf-8",
    )
    oplog = tmp_path / "run200_oplog.log"
    _write_file(
        oplog,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 | INFO | Run started for generation run200",
                "2025-12-21 12:00:01,000 | INFO | Run completed successfully for generation run200",
            ]
        )
        + "\n",
    )
    report = build_report(RunInputs("run200", oplog_path=str(oplog), transcript_path=str(transcript)))
    assert [s.path for s in report.steps[:3]] == ["pipeline/section/b", "pipeline/section/a", "pipeline/section/c"]


def test_repeated_paths_merge_by_occurrence_and_html_anchors_unique(tmp_path: Path):
    transcript = tmp_path / "run300_transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "generation_id": "run300",
                "steps": [
                    {"name": "x1", "path": "pipeline/repeat/x", "prompt": "p1", "response": "r1"},
                    {"name": "x2", "path": "pipeline/repeat/x", "prompt": "p2", "response": "r2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    oplog = tmp_path / "run300_oplog.log"
    _write_file(
        oplog,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 | INFO | Run started for generation run300",
                "2025-12-21 12:00:00,100 | INFO | Step: pipeline/repeat/x (context_chars=0, prompt_chars=1, input_chars=1)",
                "2025-12-21 12:00:00,200 | INFO | Received response for pipeline/repeat/x (input_chars=1, chars=10)",
                "2025-12-21 12:00:00,300 | INFO | Step: pipeline/repeat/x (context_chars=0, prompt_chars=1, input_chars=1)",
                "2025-12-21 12:00:00,600 | INFO | Received response for pipeline/repeat/x (input_chars=1, chars=20)",
                "2025-12-21 12:00:01,000 | INFO | Run completed successfully for generation run300",
            ]
        )
        + "\n",
    )
    report = build_report(RunInputs("run300", oplog_path=str(oplog), transcript_path=str(transcript)))
    assert [s.timing.duration_ms for s in report.steps[:2]] == [pytest.approx(100), pytest.approx(300)]

    html_doc = render_html(report)
    ids = set(re.findall(r'id=\"(step-[^\"]+)\"', html_doc))
    assert len(ids) == len(report.steps)


def test_render_html_fails_fast_on_duplicate_anchor_ids():
    report = RunReport(
        metadata=RunMetadata(generation_id="dup"),
        steps=[
            StepReport(path="pipeline/x", name="x", step_index=0),
            StepReport(path="pipeline/x", name="x", step_index=0),
        ],
        side_effects=[],
        issues=[],
        parser_version="x",
        tool_version="x",
    )
    with pytest.raises(ValueError, match="Duplicate step anchor IDs"):
        render_html(report)
import json
import os
from pathlib import Path

import pytest

from run_review.cli import main as cli_main
from run_review.report_builder import build_report, diff_reports, report_to_dict
from run_review.report_model import RunInputs


def _write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def _sample_oplog(tmpdir: Path) -> Path:
    log = tmpdir / "run123_oplog.log"
    _write_file(
        log,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 INFO Run started for generation run123",
                "2025-12-21 12:00:00,100 INFO No prompt.random_seed configured; generated seed=42",
                "2025-12-21 12:00:00,200 INFO Context injectors enabled: season, holiday",
                "2025-12-21 12:00:00,210 INFO Holiday injector: next=Christmas in 3 days (p=1.000 roll=0.123 applied=True). You MUST adopt it as an additional theme.",
                "2025-12-21 12:00:00,300 INFO Step: pipeline/image_prompt_creation/tot_enclave/consensus_1 (context_chars=30, prompt_chars=90, input_chars=120)",
                "2025-12-21 12:00:00,900 INFO Received response for pipeline/image_prompt_creation/tot_enclave/consensus_1 (input_chars=120, chars=240)",
                "2025-12-21 12:00:01,000 INFO Image generation request sent (model=gpt-image-1.5, size=1536x1024, quality=high)",
                "2025-12-21 12:00:01,050 DEBUG Received image payload length: 1024",
                "2025-12-21 12:00:01,500 INFO Upscaling enabled: engine=realesrgan-ncnn-vulkan model=realesrgan-x4plus target=long_edge=3840 aspect=1.778",
                "2025-12-21 12:00:01,800 INFO Appended manifest row to manifest.json (seq=1)",
                "2025-12-21 12:00:02,000 INFO Uploading final.png to gphotos:album/Test via rclone",
                "2025-12-21 12:00:02,500 INFO Uploaded image via rclone to gphotos:album/Test",
                "2025-12-21 12:00:03,000 INFO Run completed successfully for generation run123",
            ]
        )
        + "\n",
    )
    return log


def _sample_transcript(tmpdir: Path) -> Path:
    transcript = tmpdir / "run123_transcript.json"
    payload = {
        "generation_id": "run123",
        "seed": 99,
        "created_at": "2025-12-21T12:00:00Z",
        "selected_concepts": ["snow", "lights"],
        "context": {"holiday": "MUST adopt"},
        "title_generation": {"title": "Festive"},
        "concept_filter_log": {"input": "foo", "output": "foo"},
        "steps": [
            {
                "name": "consensus_1",
                "path": "pipeline/image_prompt_creation/tot_enclave/consensus_1",
                "prompt": "Create a winter scene",
                "response": "A snowy village",
                "prompt_chars": 90,
                "input_chars": 120,
                "context_chars": 30,
                "response_chars": 200,
            }
        ],
    }
    transcript.write_text(json.dumps(payload), encoding="utf-8")
    return transcript


def test_build_report_joins_oplog_and_transcript(tmp_path: Path):
    oplog = _sample_oplog(tmp_path)
    transcript = _sample_transcript(tmp_path)
    report = build_report(RunInputs("run123", oplog_path=str(oplog), transcript_path=str(transcript)))

    assert report.metadata.seed == 99
    assert report.metadata.context == {"holiday": "MUST adopt"}
    assert len(report.steps) == 1
    step = report.steps[0]
    assert step.timing.duration_ms == pytest.approx(600)
    assert step.prompt_chars == 90
    assert step.oplog_prompt_chars == 90
    assert step.input_chars == 120
    assert step.oplog_response_chars == 240
    assert any(issue.code == "concept_filter_noop" for issue in report.issues)
    assert any(se.type == "image_generation_request" for se in report.side_effects)

    payload = report_to_dict(report)
    assert payload["metadata"]["generation_id"] == "run123"
    assert payload["steps"][0]["timing"]["duration_ms"] == pytest.approx(600)


def test_best_effort_allows_missing_artifacts(tmp_path: Path):
    oplog = _sample_oplog(tmp_path)
    report = build_report(RunInputs("run123", oplog_path=str(oplog), transcript_path=None), best_effort=True)
    assert any(issue.code == "missing_transcript" for issue in report.issues)


def test_compare_reports_flags_added_and_removed(tmp_path: Path):
    oplog_a = _sample_oplog(tmp_path)
    transcript_a = _sample_transcript(tmp_path)
    report_a = build_report(RunInputs("run123", oplog_path=str(oplog_a), transcript_path=str(transcript_a)))

    # second run has different step path
    oplog_b = tmp_path / "run124_oplog.log"
    _write_file(
        oplog_b,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 INFO Run started for generation run124",
                "2025-12-21 12:00:00,100 INFO Step: pipeline/image_prompt_creation/tot_enclave/final_consensus (context_chars=0, prompt_chars=5, input_chars=10)",
                "2025-12-21 12:00:00,400 INFO Received response for pipeline/image_prompt_creation/tot_enclave/final_consensus (input_chars=10, chars=20)",
                "2025-12-21 12:00:00,900 INFO Run completed successfully for generation run124",
            ]
        )
        + "\n",
    )
    transcript_b = tmp_path / "run124_transcript.json"
    transcript_b.write_text(
        json.dumps(
            {
                "generation_id": "run124",
                "steps": [
                    {
                        "name": "final_consensus",
                        "path": "pipeline/image_prompt_creation/tot_enclave/final_consensus",
                        "prompt": "p",
                        "response": "r",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report_b = build_report(RunInputs("run124", oplog_path=str(oplog_b), transcript_path=str(transcript_b)))

    diff = diff_reports(report_a, report_b)
    assert "pipeline/image_prompt_creation/tot_enclave/final_consensus" in diff.added_steps
    assert "pipeline/image_prompt_creation/tot_enclave/consensus_1" in diff.removed_steps


def test_compare_detects_injector_and_upscale_changes(tmp_path: Path):
    oplog_a = tmp_path / "run125_oplog.log"
    _write_file(
        oplog_a,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 INFO Run started for generation run125",
                "2025-12-21 12:00:00,200 INFO Holiday injector: next=Christmas in 3 days. You should adopt it as an additional theme.",
                "2025-12-21 12:00:01,000 INFO upscaling enabled target_long_edge_px=3840",
                "2025-12-21 12:00:02,000 INFO Run completed successfully for generation run125",
            ]
        )
        + "\n",
    )
    transcript_a = tmp_path / "run125_transcript.json"
    transcript_a.write_text(json.dumps({"generation_id": "run125", "steps": []}), encoding="utf-8")
    report_a = build_report(RunInputs("run125", oplog_path=str(oplog_a), transcript_path=str(transcript_a)))

    oplog_b = tmp_path / "run126_oplog.log"
    _write_file(
        oplog_b,
        "\n".join(
            [
                "2025-12-21 12:00:00,000 INFO Run started for generation run126",
                "2025-12-21 12:00:00,200 INFO Holiday injector: next=Christmas in 3 days. You MUST adopt it as an additional theme.",
                "2025-12-21 12:00:01,000 INFO Upscaling enabled: engine=realesrgan target=long_edge=3840 aspect=1.778",
                "2025-12-21 12:00:02,000 INFO Run completed successfully for generation run126",
            ]
        )
        + "\n",
    )
    transcript_b = tmp_path / "run126_transcript.json"
    transcript_b.write_text(json.dumps({"generation_id": "run126", "steps": []}), encoding="utf-8")
    report_b = build_report(RunInputs("run126", oplog_path=str(oplog_b), transcript_path=str(transcript_b)))

    diff = diff_reports(report_a, report_b)
    assert any("adoption strength" in msg.lower() for msg in diff.injector_diffs)
    assert any("upscale log format" in msg.lower() for msg in diff.post_processing_diffs)


def test_cli_writes_reports(tmp_path: Path):
    oplog = _sample_oplog(tmp_path)
    transcript = _sample_transcript(tmp_path)
    exit_code = cli_main(
        [
            "--generation-id",
            "run123",
            "--oplog",
            str(oplog),
            "--transcript",
            str(transcript),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "run123_run_report.json").exists()
    assert (tmp_path / "run123_run_report.html").exists()
