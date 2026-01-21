import csv
import json
import runpy
from pathlib import Path


def _tool_main():
    repo_root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(repo_root / "tools" / "migrate_generations_csv_legacy_to_v2.py"))
    return module["main"]


def test_migrate_generations_csv_legacy_to_v2_smoke(tmp_path):
    input_path = tmp_path / "generations.csv"
    with open(input_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "Description Prompt", "Generation Prompt", "Image URL"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ID": "g1",
                "Description Prompt": "cat, quest",
                "Generation Prompt": "a cat on a quest",
                "Image URL": "https://example.com/image.png",
            }
        )

    output_path = tmp_path / "generations_v2.csv"
    rc = _tool_main()(["--input", str(input_path), "--output", str(output_path)])
    assert rc == 0

    with open(output_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "generation_id",
            "selected_concepts",
            "final_image_prompt",
            "image_path",
            "created_at",
            "seed",
        ]
        rows = list(reader)
    assert rows == [
        {
            "generation_id": "g1",
            "selected_concepts": "cat, quest",
            "final_image_prompt": "a cat on a quest",
            "image_path": "https://example.com/image.png",
            "created_at": "",
            "seed": "",
        }
    ]

    report_path = output_path.with_suffix(output_path.suffix + ".migration_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows_total"] == 1
    assert report["rows_written"] == 1
    assert report["unmapped_image_values_count"] == 0
    assert report["unmapped_image_values_path"] is None


def test_migrate_generations_csv_legacy_to_v2_preserves_unmapped_image_values(tmp_path):
    input_path = tmp_path / "generations.csv"
    with open(input_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "Description Prompt", "Generation Prompt", "Image URL"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ID": "g1",
                "Description Prompt": "cat, quest",
                "Generation Prompt": "a cat on a quest",
                "Image URL": "{\"not\":\"a url\"}",
            }
        )

    output_path = tmp_path / "generations_v2.csv"
    rc = _tool_main()(["--input", str(input_path), "--output", str(output_path)])
    assert rc == 0

    with open(output_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows[0]["image_path"] == ""

    sidecar_path = output_path.with_suffix(output_path.suffix + ".unmapped_image_values.jsonl")
    lines = sidecar_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["generation_id"] == "g1"
    assert payload["raw_image_url_field"] == "{\"not\":\"a url\"}"

