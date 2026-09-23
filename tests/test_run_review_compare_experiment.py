import json
from pathlib import Path

from image_project.run_review.compare_experiment import PairSelection, compare_experiment_pairs


def _write_oplog(path: Path, generation_id: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"2025-12-21 12:00:00,000 | INFO | Run started for generation {generation_id}",
                f"2025-12-21 12:00:01,000 | INFO | Run completed successfully for generation {generation_id}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_transcript(path: Path, generation_id: str) -> None:
    path.write_text(
        json.dumps({"generation_id": generation_id, "steps": []}),
        encoding="utf-8",
    )


def test_compare_experiment_all_pairs_writes_index(tmp_path: Path):
    experiment_dir = tmp_path / "experiment"
    logs_dir = experiment_dir / "logs"
    logs_dir.mkdir(parents=True)

    a_id = "A1_unit"
    b_id = "B1_unit"

    _write_oplog(logs_dir / f"{a_id}_oplog.log", a_id)
    _write_oplog(logs_dir / f"{b_id}_oplog.log", b_id)
    _write_transcript(logs_dir / f"{a_id}_transcript.json", a_id)
    _write_transcript(logs_dir / f"{b_id}_transcript.json", b_id)

    pairs_payload = {
        "schema_version": 1,
        "experiment_id": "exp_unit",
        "created_at": "2025-01-01T00:00:00Z",
        "pairs": [
            {
                "run_index": 1,
                "a_generation_id": a_id,
                "b_generation_id": b_id,
                "metadata": {"mode": "prompt_only"},
            }
        ],
    }
    (experiment_dir / "pairs.json").write_text(json.dumps(pairs_payload), encoding="utf-8")

    output_dir = tmp_path / "review"
    summary = compare_experiment_pairs(
        experiment_dir=str(experiment_dir),
        output_dir=str(output_dir),
        selection=PairSelection(all_pairs=True),
        print_fn=None,
    )

    assert (output_dir / "index.html").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / f"{a_id}_vs_{b_id}_run_compare.html").exists()
    assert summary["counts"]["failed"] == 0

