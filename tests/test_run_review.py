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
