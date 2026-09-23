from tools.generate_stages_doc import _stage_rows, generate_markdown


def test_stage_docs_generation_is_deterministic():
    rows = _stage_rows()
    md1 = generate_markdown(rows=rows)
    md2 = generate_markdown(rows=rows)

    assert md1 == md2
    assert "## ab" in md1
    assert "`ab.scene_draft`" in md1

