"""Logging helpers that avoid heavy dependencies."""

from __future__ import annotations


def write_messages_log(log_path: str, messages_text: str) -> None:
    """Write transcript text to a UTF-8 file."""
    with open(log_path, "w", encoding="utf-8") as file:
        file.write(messages_text)
