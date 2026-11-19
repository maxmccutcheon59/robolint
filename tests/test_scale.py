"""Streaming loader and two-pass statistics: same answers as holding everything in memory."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import robolint.stream as stream
from robolint import lint
from robolint.checks import Config
from robolint.checks.meta import _global_index
from robolint.checks.signals import _dataset_stats
from robolint.dataset import load_dataset
from robolint.stream import Moments, QuantileStream

from .synth import make_dataset


def _run_stream(qs, x, chunks, exact_rows):
    s = QuantileStream(qs, len(x), exact_rows=exact_rows)
    parts = np.array_split(x, chunks)
    for c in parts:
        s.add(c)
    if s.needs_second_pass():
        while True:
            for c in parts:
                s.add_second(c)
            if (r := s.result()) is not None:
                return r
    return s.result()


@pytest.mark.parametrize("kind", ["normal", "ties", "walk", "periodic", "constant"])
def test_two_pass_quantiles_equal_numpy_exactly(kind):
    rng = np.random.default_rng(3)
    for n in (1, 2, 7, 1000, 54_321):
        if kind == "normal":
            x = rng.normal(size=(n, 3))
        elif kind == "ties":
            x = rng.integers(0, 4, size=(n, 3)).astype(float)
        elif kind == "walk":
            x = np.cumsum(rng.normal(size=(n, 3)), axis=0)
        elif kind == "periodic":
            x = np.sin(np.arange(n)[:, None] / 5.0 + np.arange(3))
        else:
            x = np.full((n, 3), 2.5)
        qs = [0.005, 0.5, 0.995]
        ref = np.stack([np.quantile(x, q, axis=0) for q in qs])
        assert np.array_equal(_run_stream(qs, x, 17, exact_rows=0), ref)
        assert np.array_equal(_run_stream(qs, x, 17, exact_rows=n), ref)


def test_two_pass_quantiles_recover_from_a_too_narrow_band():
    x = np.random.default_rng(0).normal(size=(20_000, 2))
    parts = np.array_split(x, 9)
    s = QuantileStream([0.005, 0.995], len(x), exact_rows=0)
    for c in parts:
        s.add(c)
    assert s.needs_second_pass()
    s._bands[:] = np.median(x)  # a band that misses both answers
    for c in parts:
        s.add_second(c)
    assert s.result() is None  # detected: the band is widened for another pass
    for c in parts:
        s.add_second(c)
    ref = np.stack([np.quantile(x, q, axis=0) for q in (0.005, 0.995)])
    assert np.array_equal(s.result(), ref)


def test_streaming_moments_match_corrcoef():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 4)) * 30 + 500
    y = 0.3 * x + rng.normal(size=(5000, 4))
    m = Moments()
    for cx, cy in zip(np.array_split(x, 13), np.array_split(y, 13), strict=True):
        m.add(cx, cy)
    ref = [np.corrcoef(x[:, j], y[:, j])[0, 1] for j in range(4)]
    assert np.allclose(m.corr(), ref, rtol=0, atol=1e-12)
    assert np.allclose(m.std_x(), x.std(axis=0), rtol=1e-12)


def _faulty(tmp_path, name, **kw):
    faults = {1: ["spike"], 3: ["dead"], 5: ["state_lag"], 7: ["idle"], 9: ["dup_of:2"], 11: ["nan"]}
    return make_dataset(tmp_path / name, n_eps=14, faults=faults, **kw)


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_streaming_stats_equal_in_memory_stats(tmp_path, version):
    ds = load_dataset(_faulty(tmp_path, "ds", version=version))
    cfg = Config()
    exact = _dataset_stats(ds, ["action", "observation.state"], cfg)
    streamed = _dataset_stats(ds, ["action", "observation.state"], cfg, exact_rows=0)
    for key, a in exact.items():
        b = streamed[key]
        assert a is not None and b is not None
        for f in ("lo", "hi", "rng", "step_scale", "discrete"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), (key, f)
        assert a.names == b.names


def _result(r):
    d = r.to_dict()
    return {k: d[k] for k in ("score", "grade", "findings", "episode_scores")}


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_no_cache_and_forced_streaming_give_identical_results(tmp_path, monkeypatch, version):
    root = _faulty(tmp_path, "ds", version=version, grasp_object=True)
    base = lint(root)
    monkeypatch.setattr(stream, "EXACT_ROWS", 0)
    r = lint(root, cache_mb=0)
    assert r.dataset.store is not None and r.dataset.store.cache.decodes > 0  # columns were re-read from disk
    assert _result(r) == _result(base)
    assert base.findings  # the comparison covers real findings


def _rewrite(root, parts, shuffle, seed=0):
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f)
    idx = np.arange(t.num_rows)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    f.unlink()
    for i, chunk in enumerate(np.array_split(idx, parts)):
        pq.write_table(t.take(pa.array(chunk)), root / "data" / "chunk-000" / f"file-{i:03d}.parquet")
    return t


def test_shuffled_rows_split_across_files_load_like_sorted_data(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=9, length=60)
    original = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    _rewrite(root, parts=4, shuffle=True)
    ds = load_dataset(root, cache_mb=0)
    ep_col = original.column("episode_index").to_numpy()
    for ep in ds.episodes:
        rows = np.flatnonzero(ep_col == ep.index)
        assert ep.length == len(rows)
        want = np.stack(original.column("action").take(pa.array(rows)).to_pylist()).astype(np.float32)
        assert np.array_equal(ep.frames["action"], want)
        assert np.array_equal(ep.frames["frame_index"], np.arange(len(rows)))
    assert not [f for f in lint(root).findings if f.severity.label != "info"]


def test_frames_are_read_only_and_behave_like_a_dict(tmp_path):
    ds = load_dataset(make_dataset(tmp_path / "ds", n_eps=3))
    fr = ds.episodes[0].frames
    assert "action" in fr and "observation.image" not in fr
    assert set(fr) >= {"action", "observation.state", "timestamp", "frame_index", "episode_index"}
    assert len(fr) == len(list(fr.keys()))
    with pytest.raises(ValueError):
        fr["action"][0, 0] = 1.0
    with pytest.raises(KeyError):
        fr["nope"]


def test_memory_stays_within_the_cache_budget(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=40, length=300)
    ds = load_dataset(root, cache_mb=0.05)
    for ep in ds.episodes:
        ep.frames["action"]
    assert ds.store is not None
    largest = max(
        ds.store.file_column(i, "action").nbytes for i in range(len(ds.store.files))
    )  # one file is always kept
    assert ds.store.cache.bytes <= max(0.05 * 1024 * 1024, largest)


@pytest.mark.parametrize("bitmap", [True, False])
def test_global_index_duplicates_and_gaps_streamed(tmp_path, monkeypatch, bitmap):
    if not bitmap:
        monkeypatch.setattr("robolint.checks.meta.BITMAP_MAX", 1)
    root = make_dataset(tmp_path / "ds", n_eps=5, length=40)
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f)
    idx = t.column("index").to_numpy().copy()
    idx[50:53] = idx[10:13]  # three duplicates
    pq.write_table(t.set_column(t.column_names.index("index"), "index", pa.array(idx)), f)
    ds = load_dataset(root)
    assert _global_index(ds)[0] == 3
    r = lint(root)
    assert [f.message for f in r.findings if f.code == "GLOBAL_INDEX_DUPLICATES"] == [
        "3 duplicate values in the global 'index' column"
    ]
    idx = np.arange(200) * 2  # unique but not contiguous
    pq.write_table(t.set_column(t.column_names.index("index"), "index", pa.array(idx)), f)
    assert any(f.code == "GLOBAL_INDEX_GAPS" for f in lint(root).findings)


def test_cli_accepts_cache_budget(tmp_path):
    from robolint.cli import main

    root = make_dataset(tmp_path / "ds", n_eps=4)
    out = tmp_path / "r.json"
    assert main(["check", str(root), "--cache-mb", "1", "--json", str(out), "--no-color"]) == 0
    assert json.loads(out.read_text())["score"] == 100.0
