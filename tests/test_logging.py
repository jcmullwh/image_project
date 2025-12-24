from pathlib import Path

from image_project.foundation.logging_utils import write_messages_log


def test_writes_unicode_text_with_utf8_encoding(tmp_path: Path):
    unicode_text = "Log entry with arrow \\u2192 and accents \\u00e9."
    log_path = tmp_path / "log.txt"

    write_messages_log(str(log_path), unicode_text)

    with open(log_path, "r", encoding="utf-8") as file:
        content = file.read()

    assert content == unicode_text
