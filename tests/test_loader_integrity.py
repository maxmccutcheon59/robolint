"""Regressions from the Nov 2026 code review: data the loader cannot use must never pass silently,
and video checks must not leak file handles."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robolint import Severity, lint

from .synth import make_dataset

FAULTS = {2: ["dead"], 5: ["spike"]}


def _rewrite_action(path: Path, edit) -> None:
    t = pq.read_table(path)
    rows = edit(t.column("action").to_pylist())
    i = t.column_names.index("action")
    pq.write_table(t.set_column(i, "action", pa.array(rows, type=pa.list_(pa.float32()))), path)


def _baseline(root):
    r = lint(root)
    assert {"DEAD_EPISODE", "ACTION_SPIKE"} <= {f.code for f in r.findings}
    return r


def test_null_action_row_is_an_error_not_a_pass(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, faults=FAULTS)
    base = _baseline(root)

    def null_row(rows):
        rows[100] = None
        return rows

    _rewrite_action(root / "data/chunk-000/file-000.parquet", null_row)
    r = lint(root)
    codes = {f.code for f in r.findings}
    # the original problems are still caught, and the null row is reported and scored
    assert {"DEAD_EPISODE", "ACTION_SPIKE", "DATA_COLUMN_INVALID", "NONFINITE_VALUES"} <= codes
    assert r.scores.dataset_score < base.scores.dataset_score
    bad = [f for f in r.findings if f.code == "NONFINITE_VALUES"]
    assert {f.episode for f in bad} == {0}  # row 100 sits in episode 0 (150 frames each)
    assert all(f.severity == Severity.ERROR for f in r.findings if f.code == "DATA_COLUMN_INVALID")


def test_mixed_action_width_across_v2_files_is_an_error(tmp_path):
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=10, faults=FAULTS)
    base = _baseline(root)
    _rewrite_action(root / "data/chunk-000/episode_000007.parquet", lambda rows: [r + [0.0] for r in rows])
    r = lint(root)
    codes = {f.code for f in r.findings}
    assert {"DEAD_EPISODE", "ACTION_SPIKE", "DATA_COLUMN_INVALID"} <= codes
    assert any(f.code == "NONFINITE_VALUES" and f.episode == 7 for f in r.findings)
    assert r.scores.dataset_score < base.scores.dataset_score


def test_mixed_width_rows_inside_one_file(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10)

    def widen(rows):
        rows[200] = rows[200] + [1.0]
        return rows

    _rewrite_action(root / "data/chunk-000/file-000.parquet", widen)
    r = lint(root)
    assert any(f.code == "DATA_COLUMN_INVALID" and "width" in f.message for f in r.findings)
    assert any(f.code == "NONFINITE_VALUES" and f.episode == 1 for f in r.findings)
    assert r.scores.dataset_score < 100


def test_non_numeric_action_column_is_an_error(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=6)
    f = root / "data/chunk-000/file-000.parquet"
    t = pq.read_table(f)
    i = t.column_names.index("action")
    t = t.set_column(i, "action", pa.array([str(x) for x in t.column("action").to_pylist()]))
    pq.write_table(t, f)
    r = lint(root)
    assert any(x.code == "DATA_COLUMN_INVALID" and x.severity == Severity.ERROR for x in r.findings)
    assert r.scores.dataset_score < 100


def test_file_without_episode_index_is_reported(tmp_path):
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=8)
    f = root / "data/chunk-000/episode_000003.parquet"
    t = pq.read_table(f)
    pq.write_table(t.drop(["episode_index"]), f)
    r = lint(root)
    assert any(x.code == "DATA_FILE_SKIPPED" for x in r.findings)
    assert r.scores.dataset_score < 100


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_data_file_symlinked_outside_is_reported(tmp_path):
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=8)
    f = root / "data/chunk-000/episode_000004.parquet"
    outside = tmp_path / "elsewhere.parquet"
    f.rename(outside)
    os.symlink(outside, f)
    r = lint(root)
    assert any(x.code == "DATA_FILE_SKIPPED" and "outside" in x.message for x in r.findings)
    assert r.scores.dataset_score < 100


def test_null_scalar_timestamp_is_reported(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=6)
    f = root / "data/chunk-000/file-000.parquet"
    t = pq.read_table(f)
    ts = t.column("timestamp").to_pylist()
    ts[10] = None
    t = t.set_column(t.column_names.index("timestamp"), "timestamp", pa.array(ts, type=pa.float32()))
    pq.write_table(t, f)
    r = lint(root)
    assert any(x.code == "DATA_COLUMN_INVALID" for x in r.findings)
    assert any(x.code == "TIMESTAMP_NONFINITE" and x.episode == 0 for x in r.findings)


def test_clean_data_is_unchanged_by_the_stricter_loader(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10)
    r = lint(root)
    assert not [f for f in r.findings if f.severity >= Severity.WARNING]
    assert r.dataset.episodes[0].frames["action"].dtype == np.float32


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_NOFILE is POSIX-only")
def test_video_check_survives_a_low_file_descriptor_limit(tmp_path):
    pytest.importorskip("av")
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=120, length=40, videos=True)
    script = textwrap.dedent(
        f"""
        import resource, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
        resource.setrlimit(resource.RLIMIT_NOFILE, (60, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        from robolint import lint
        r = lint({str(root)!r}, videos=True)
        bad = [f for f in r.findings if f.code == "VIDEO_UNREADABLE"]
        print(len(bad), r.scores.dataset_score)
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=True)
    n_bad, score = out.stdout.split()
    assert n_bad == "0", out.stdout + out.stderr
    assert float(score) == 100.0
