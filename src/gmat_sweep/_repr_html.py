"""Shared HTML-table builders for ``_repr_html_`` on Sweep, RunOutcome, ManifestEntry.

Pure-Python rendering — no template engine, no extra dependency. Each
helper returns a plain ``str`` of HTML. All values flow through
:func:`html.escape` so the rendered table cannot break on a ``<`` in a
file path, override value, or stderr line.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "build_kv_table",
    "format_overrides_html",
    "format_paths_html",
    "short_sha",
    "summarise_stderr_html",
]

# stderr is rendered as one ellipsised line so a multi-megabyte traceback
# doesn't bloat the cell — full text is still on disk in the worker log.
_STDERR_SUMMARY_WIDTH = 80


def short_sha(sha: str) -> str:
    """First 12 chars of a hex SHA, or the full string when shorter."""
    return sha[:12]


def summarise_stderr_html(stderr: str | None) -> str:
    """First line of ``stderr`` truncated to fit the cell, HTML-escaped.

    Returns ``"(no stderr)"`` for ``None`` or empty input. Unlike the
    plain-text variant in :mod:`gmat_sweep.cli`, this version escapes the
    output so a ``<`` in a traceback can't break the surrounding table.
    """
    if not stderr:
        return "(no stderr)"
    lines = stderr.splitlines()
    first = lines[0] if lines else ""
    if len(first) <= _STDERR_SUMMARY_WIDTH:
        return html.escape(first)
    return html.escape(first[: _STDERR_SUMMARY_WIDTH - 3]) + "..."


def format_paths_html(paths: Mapping[str, Path]) -> str:
    """Render a ``{name: path}`` mapping as a one-line `key → path` list, HTML-escaped.

    Empty input renders as ``"(none)"``. Multiple entries are joined with
    ``<br>`` so the cell stays one column wide but readable.
    """
    if not paths:
        return "(none)"
    parts = [
        f"{html.escape(k)} &rarr; <code>{html.escape(str(v))}</code>" for k, v in paths.items()
    ]
    return "<br>".join(parts)


def format_overrides_html(overrides: Mapping[str, Any]) -> str:
    """Render an ``overrides`` mapping as a one-key-per-line block, HTML-escaped.

    Empty input renders as ``"(none)"``. Values are passed through
    :func:`repr` so numeric vs string vs ``None`` stay distinguishable
    (matches the plain-text :func:`gmat_sweep.cli._format_run_detail`
    layout).
    """
    if not overrides:
        return "(none)"
    parts = [
        f"<code>{html.escape(k)}</code> = <code>{html.escape(repr(v))}</code>"
        for k, v in sorted(overrides.items())
    ]
    return "<br>".join(parts)


def build_kv_table(
    title: str,
    rows: list[tuple[str, str]],
) -> str:
    """Build a ``<table>`` of two-column key/value rows with ``title`` as the caption.

    ``rows`` is an ordered list of ``(label, html_value)`` pairs. Labels
    are HTML-escaped; values are inserted verbatim, so callers must
    pre-escape any user-supplied content (the helpers above do).
    """
    body = "".join(
        f'<tr><th style="text-align:left;padding-right:1em">{html.escape(label)}</th>'
        f"<td>{value}</td></tr>"
        for label, value in rows
    )
    return (
        f'<table><caption style="text-align:left;font-weight:bold">'
        f"{html.escape(title)}</caption>{body}</table>"
    )
