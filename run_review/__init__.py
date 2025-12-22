"""Offline run review tooling."""

TOOL_VERSION = "0.1.0"

__all__ = [
    "parse_oplog",
    "parse_transcript",
    "build_report",
    "render_html",
    "render_compare_html",
    "diff_reports",
]

from .parse_oplog import parse_oplog
from .parse_transcript import parse_transcript
from .report_builder import build_report
from .render_html import render_html, render_compare_html
from .compare import diff_reports
