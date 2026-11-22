"""A small SVG badge showing a dataset's robolint grade, for dataset cards and READMEs."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

_COLORS = {"A": "#2e7d32", "B": "#689f38", "C": "#f9a825", "D": "#ef6c00", "F": "#c62828"}


def _width(text: str) -> int:
    return 10 + 7 * len(text)  # close enough for an 11px sans-serif label


def badge_svg(grade: str, score: float, label: str = "robolint") -> str:
    value = f"{grade} · {score:.0f}/100"
    lw, vw = _width(label), _width(value)
    color = _COLORS.get(grade, "#616161")
    lab, val = escape(label), escape(value)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + vw}" height="20" role="img" '
        f'aria-label="{lab}: {val}"><title>{lab}: {val}</title>'
        f'<rect width="{lw}" height="20" rx="3" fill="#555"/>'
        f'<rect x="{lw}" width="{vw}" height="20" rx="3" fill="{color}"/>'
        f'<rect x="{lw - 3}" width="6" height="20" fill="{color}"/>'
        '<g fill="#fff" text-anchor="middle" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
        f'<text x="{lw / 2}" y="14">{lab}</text><text x="{lw + vw / 2}" y="14">{val}</text></g></svg>'
    )


def write_badge(grade: str, score: float, path: Path) -> None:
    path.write_text(badge_svg(grade, score), encoding="utf-8")
