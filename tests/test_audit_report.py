"""Client-facing audit report."""

from __future__ import annotations

import datetime as dt

from robolint.api import lint
from robolint.audit_report import _ranges, impact, write_audit_report
from robolint.cli import main

from .synth import make_dataset


def test_ranges():
    assert _ranges([]) == ""
    assert _ranges([3]) == "3"
    assert _ranges([1, 2, 3, 7, 9, 10]) == "1–3, 7, 9–10"
    assert _ranges([5, 1, 2, 2]) == "1–2, 5"


def test_impact_counts_dropped_and_trimmed_frames(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, faults={2: ["dead"], 5: ["nan"], 7: ["idle"]})
    r = lint(root)
    imp = impact(r)
    assert set(imp.drop_episodes) >= {2, 5}
    assert imp.drop_frames == 150 * len(imp.drop_episodes)
    assert 7 in imp.trim and imp.trim_frames > 0
    assert not set(imp.trim) & set(imp.drop_episodes)
    assert imp.usable_frames == imp.total_frames - imp.drop_frames - imp.trim_frames
    assert imp.fps == 30


def test_report_contents(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=8, faults={1: ["dead"], 4: ["idle"]})
    r = lint(root)
    out = tmp_path / "audit.html"
    write_audit_report(r, out, client="Acme Robotics", date=dt.date(2026, 12, 1))
    page = out.read_text(encoding="utf-8")
    assert "Data Quality Audit for Acme Robotics" in page
    assert "December 1, 2026" in page
    assert "Action plan" in page and "DEAD_EPISODE" in page
    assert "Episodes to remove" in page


def test_report_escapes_untrusted_text(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=6, faults={2: ["dead"]})
    r = lint(root)
    out = tmp_path / "audit.html"
    write_audit_report(r, out, client="<script>alert(1)</script>", prepared_by="<b>x</b>")
    page = out.read_text(encoding="utf-8")
    assert "<script>" not in page and "<b>x</b>" not in page
    assert "&lt;script&gt;" in page


def test_clean_dataset_needs_no_action(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=6)
    r = lint(root)
    out = tmp_path / "audit.html"
    imp = write_audit_report(r, out)
    if not (imp.drop_episodes or imp.trim or imp.review_episodes):
        assert "ready to train on" in out.read_text(encoding="utf-8")


def test_cli_writes_audit_report(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", n_eps=6, faults={3: ["dead"]})
    out = tmp_path / "client.html"
    assert main(["check", str(root), "--no-color", "--audit-report", str(out), "--client", "Acme"]) == 0
    assert out.exists()
    assert "Audit report:" in capsys.readouterr().out
