"""robolint clean: writes a valid, re-checkable copy with broken episodes removed, idle edges
trimmed (v3 video offsets moved to match) and every index rewritten; the source is untouched."""

import json

import pyarrow.parquet as pq
import pytest

from robolint import lint
from robolint.clean import CleanError, write_cleaned
from robolint.cleaning import plan

from .synth import make_dataset

FAULTS = {1: ["dead"], 3: ["idle"], 5: ["nonmono"], 6: ["length_meta"]}


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_clean_roundtrip(tmp_path, version):
    src = make_dataset(tmp_path / "src", version=version, n_eps=8, faults=FAULTS, videos=True)
    before = sorted(p.read_bytes() for p in src.rglob("*.parquet"))
    r = lint(src)
    p = plan(r)
    removed = {x.episode for x in p if x.decision == "remove"}
    assert {1, 5} <= removed
    rep = write_cleaned(src, p, tmp_path / "out")
    assert sorted(x.read_bytes() for x in src.rglob("*.parquet")) == before  # source untouched
    assert rep.kept == 8 - len(removed)

    r2 = lint(tmp_path / "out", videos=True)
    codes = {f.code for f in r2.findings if f.severity.label == "error"}
    assert not codes, codes
    assert len(r2.dataset.episodes) == rep.kept
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
    assert info["total_episodes"] == rep.kept and info["total_frames"] == rep.frames_after
    m = json.loads((tmp_path / "out" / "meta" / "robolint_cleaning.json").read_text())
    assert [e["original_episode"] for e in m["episodes"]] == [e for e in range(8) if e not in removed]
    if version.startswith("v3"):
        assert rep.trimmed_frames > 0  # idle edges cut, video offsets shifted to match
        t = pq.read_table(tmp_path / "out" / "data" / "chunk-000" / "file-000.parquet")
        assert t.column("index").to_pylist() == list(range(t.num_rows))


def test_refuses_unsafe_output(tmp_path):
    src = make_dataset(tmp_path / "src", n_eps=4)
    p = plan(lint(src))
    with pytest.raises(CleanError):
        write_cleaned(src, p, src / "inside")
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    with pytest.raises(CleanError):
        write_cleaned(src, p, tmp_path / "full")
