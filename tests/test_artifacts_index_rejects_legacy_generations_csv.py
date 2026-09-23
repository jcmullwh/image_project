import csv

from image_project.framework.artifacts.index import _read_generations_csv_v2


def test_artifacts_index_rejects_legacy_generations_csv(tmp_path):
    legacy_path = tmp_path / "generations.csv"
    with open(legacy_path, "w", encoding="utf-8", newline="") as handle:
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

    records, errors = _read_generations_csv_v2(legacy_path)
    assert records == {}
    assert errors
    assert "migrate_generations_csv_legacy_to_v2.py" in errors[0]


def test_artifacts_index_reads_v2_generations_csv(tmp_path):
    v2_path = tmp_path / "generations_v2.csv"
    with open(v2_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "generation_id",
                "selected_concepts",
                "final_image_prompt",
                "image_path",
                "created_at",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "generation_id": "g1",
                "selected_concepts": "[\"cat\"]",
                "final_image_prompt": "a cat on a quest",
                "image_path": "./g1.png",
                "created_at": "2026-01-01T00:00:00Z",
                "seed": "123",
            }
        )

    records, errors = _read_generations_csv_v2(v2_path)
    assert errors == []
    assert set(records.keys()) == {"g1"}
    record = records["g1"]
    assert record["generation_id"] == "g1"
    assert record["final_image_prompt"] == "a cat on a quest"
    assert record["image_path"] == "./g1.png"
    assert record["seed"] == 123
