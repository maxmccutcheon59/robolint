"""Grade badge."""

from __future__ import annotations

from robolint.badge import badge_svg
from robolint.cli import main

from .synth import make_dataset


def test_badge_content():
    svg = badge_svg("A", 96.24)
    assert svg.startswith("<svg") and "A · 96/100" in svg and "#2e7d32" in svg


def test_badge_escapes_label():
    assert "<script>" not in badge_svg("B", 80, label="<script>")


def test_cli_writes_badge(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=5)
    out = tmp_path / "badge.svg"
    assert main(["check", str(root), "--no-color", "--badge", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<svg")
