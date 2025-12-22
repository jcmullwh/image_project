import json
from pathlib import Path

import pytest

from run_review.evolution import analyze_evolution, thresholds_from_overrides
from run_review.report_builder import build_report, report_to_dict
from run_review.render_html import render_html
from run_review.report_model import RunInputs, StepReport, StepTiming


def _step(*, idx: int, path: str, response: str) -> StepReport:
    return StepReport(
        path=path,
        name=path.split("/")[-1],
        step_index=idx,
        transcript_index=idx,
        prompt="p",
        response=response,
        timing=StepTiming(),
    )


def test_evolution_analyzes_graph_uptake_provenance_and_terms():
    steps = [
        _step(idx=0, path="pipeline/section/draft", response="Base output."),
        _step(
            idx=1,
            path="pipeline/section/tot_enclave/critic_a",
            response="\n".join(
                [
                    "## Issues",
                    "- too plain",
                    "## Edits",
                    '- Use the phrase "A snowy village with lights."',
                ]
            ),
        ),
        _step(idx=2, path="pipeline/section/consensus_1", response="A snowy village with lights."),
        _step(idx=3, path="pipeline/section/consensus_2", response="A snowy village with lights."),
        _step(
            idx=4,
            path="pipeline/section/final_consensus",
            response="A snowy village with lights.\nChristmas lights sparkle.",
        ),
    ]

    evolution, issues = analyze_evolution(metadata_context={"holiday": "Christmas"}, steps=steps)
    assert issues == []
    assert len(evolution.sections) == 1

    section = evolution.sections[0]
    assert section.section_key == "pipeline/section"
    assert section.base_path == "pipeline/section/draft"
    assert section.merge_path == "pipeline/section/final_consensus"
    assert set(section.candidate_paths) == {"pipeline/section/consensus_1", "pipeline/section/consensus_2"}
    assert section.critique_paths == ["pipeline/section/tot_enclave/critic_a"]

    assert any(d.from_path == section.base_path and d.to_path == section.merge_path for d in section.deltas)
    assert len(section.candidate_similarity) == 1
    assert any(f.type == "candidate_redundancy" for f in section.findings)

    assert len(section.critiques) == 1
    uptake = section.critiques[0]
    assert uptake.parse_status == "ok"
    assert uptake.suggestions_total == 1
    assert uptake.adopted_in_candidates_count == 1
    assert uptake.adopted_in_merge_count == 1

    assert len(section.merge_provenance) == 2
    assert any(a.origin_kind == "candidate" for a in section.merge_provenance)

    assert section.critique_provenance_targets == [
        "pipeline/section/consensus_1",
        "pipeline/section/consensus_2",
        "pipeline/section/final_consensus",
    ]
    assert section.critique_provenance_target_step_indices == [2, 3, 4]
    assert section.critique_provenance_segments_total_by_target["pipeline/section/consensus_1"] == 1
    assert section.critique_provenance_segments_total_by_target["pipeline/section/consensus_2"] == 1
    assert section.critique_provenance_segments_total_by_target["pipeline/section/final_consensus"] == 2
    assert section.critique_provenance_by_critic["pipeline/section/tot_enclave/critic_a"][
        "pipeline/section/consensus_1"
    ] == pytest.approx(1.0)
    assert section.critique_provenance_by_critic["pipeline/section/tot_enclave/critic_a"][
        "pipeline/section/consensus_2"
    ] == pytest.approx(1.0)
    assert section.critique_provenance_by_critic["pipeline/section/tot_enclave/critic_a"][
        "pipeline/section/final_consensus"
    ] == pytest.approx(0.5)
    assert section.critique_provenance_unattributed_by_target["pipeline/section/final_consensus"] == pytest.approx(0.5)

    christmas = next((t for t in section.term_first_seen if t.term == "christmas"), None)
    assert christmas is not None
    assert christmas.injected is True


def test_threshold_overrides_are_strict():
    with pytest.raises(ValueError, match="Unknown evolution threshold key"):
        thresholds_from_overrides({"nope": 1})

    with pytest.raises(ValueError, match="bool is not allowed"):
        thresholds_from_overrides({"max_term_index": True})

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        thresholds_from_overrides({"candidate_redundancy_seq_ratio": 2.0})


def test_critique_edits_parses_replace_with_pairs_and_curly_quotes():
    critic_response = "\n".join(
        [
            "## Issues",
            "- none",
            "## Edits",
            "- Replace: \u201cOld opening line.\u201d",
            "  With: \u201cNew opening line.\u201d",
            "- **Replace:** \u201cOld detail.\u201d",
            "- **With:** \u201cNew detail.\u201d",
            "- Replace the opening with:",
            "  New opening line two.",
            "- Replace the ending with:",
            "  - \u201cNested new ending.\u201d",
        ]
    )

    steps = [
        _step(idx=0, path="pipeline/section/draft", response="Base output."),
        _step(idx=1, path="pipeline/section/tot_enclave/critic_a", response=critic_response),
        _step(
            idx=2,
            path="pipeline/section/consensus_1",
            response="New opening line.\nNew detail.\nNew opening line two.\nNested new ending.",
        ),
        _step(
            idx=3,
            path="pipeline/section/final_consensus",
            response="New opening line.\nNested new ending.",
        ),
    ]

    evolution, issues = analyze_evolution(metadata_context=None, steps=steps)
    assert issues == []
    section = evolution.sections[0]
    assert section.critiques
    uptake = section.critiques[0]
    assert uptake.parse_status == "ok"

    snippets = [s.snippet for s in uptake.suggestions]
    assert "Old opening line." not in snippets
    assert "New opening line." in snippets
    assert "New detail." in snippets
    assert "New opening line two." in snippets
    assert "Nested new ending." in snippets


def _write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def test_evolution_integration_build_report_and_html(tmp_path: Path):
    generation_id = "run_evo_001"

    transcript = tmp_path / f"{generation_id}_transcript.json"
    transcript_payload = {
        "generation_id": generation_id,
        "context": {"holiday": "Christmas"},
        "steps": [
            {"name": "draft", "path": "pipeline/section/draft", "prompt": "p", "response": "Base output."},
            {
                "name": "critic_a",
                "path": "pipeline/section/tot_enclave/critic_a",
                "prompt": "p",
                "response": "\n".join(
                    [
                        "## Issues",
                        "- too plain",
                        "## Edits",
                        '- Use the phrase "A snowy village with lights."',
                    ]
                ),
            },
            {
                "name": "consensus_1",
                "path": "pipeline/section/consensus_1",
                "prompt": "p",
                "response": "A snowy village with lights.",
            },
            {
                "name": "consensus_2",
                "path": "pipeline/section/consensus_2",
                "prompt": "p",
                "response": "A snowy village with lights.",
            },
            {
                "name": "final_consensus",
                "path": "pipeline/section/final_consensus",
                "prompt": "p",
                "response": "A snowy village with lights.\nChristmas lights sparkle.",
            },
        ],
    }
    _write_file(transcript, json.dumps(transcript_payload))

    oplog = tmp_path / f"{generation_id}_oplog.log"
    _write_file(
        oplog,
        "\n".join(
            [
                f"2025-12-21 12:00:00,000 | INFO | Run started for generation {generation_id}",
                "2025-12-21 12:00:00,010 | INFO | Step: pipeline/section/draft (context_chars=0, prompt_chars=1, input_chars=1)",
                "2025-12-21 12:00:00,020 | INFO | Received response for pipeline/section/draft (input_chars=1, chars=10)",
                "2025-12-21 12:00:00,030 | INFO | Step: pipeline/section/tot_enclave/critic_a (context_chars=0, prompt_chars=1, input_chars=1)",
                "2025-12-21 12:00:00,040 | INFO | Received response for pipeline/section/tot_enclave/critic_a (input_chars=1, chars=10)",
                "2025-12-21 12:00:00,050 | INFO | Step: pipeline/section/consensus_1 (context_chars=0, prompt_chars=1, input_chars=1)",
                "2025-12-21 12:00:00,060 | INFO | Received response for pipeline/section/consensus_1 (input_chars=1, chars=10)",
                "2025-12-21 12:00:00,070 | INFO | Step: pipeline/section/consensus_2 (context_chars=0, prompt_chars=1, input_chars=1)",
                "2025-12-21 12:00:00,080 | INFO | Received response for pipeline/section/consensus_2 (input_chars=1, chars=10)",
                "2025-12-21 12:00:00,090 | INFO | Step: pipeline/section/final_consensus (context_chars=0, prompt_chars=1, input_chars=1)",
                "2025-12-21 12:00:00,120 | INFO | Received response for pipeline/section/final_consensus (input_chars=1, chars=10)",
                f"2025-12-21 12:00:01,000 | INFO | Run completed successfully for generation {generation_id}",
                "",
            ]
        ),
    )

    report = build_report(RunInputs(generation_id, oplog_path=str(oplog), transcript_path=str(transcript)))
    assert report.evolution is not None
    assert report.evolution.sections

    payload = report_to_dict(report)
    assert payload["evolution"] is not None
    assert payload["evolution"]["sections"][0]["section_key"] == "pipeline/section"
    evo_section = payload["evolution"]["sections"][0]
    assert evo_section["critique_provenance_by_critic"]["pipeline/section/tot_enclave/critic_a"][
        "pipeline/section/consensus_1"
    ] == pytest.approx(1.0)

    html_doc = render_html(report)
    assert "<h2>Evolution</h2>" in html_doc
    assert "Delta metrics" in html_doc
    assert "Critique provenance" in html_doc
