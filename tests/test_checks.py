from __future__ import annotations

import json

import pytest

from robolint import Severity, lint
from robolint.cli import main

from .synth import make_dataset


def codes(result, episode=None):
    return {f.code for f in result.findings if f.episode == episode}


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_clean_dataset_has_no_problems(tmp_path, version):
    root = make_dataset(tmp_path / "ds", version=version, n_eps=12)
    r = lint(root)
    bad = [f for f in r.findings if f.severity >= Severity.WARNING]
    assert bad == [], [f.message for f in bad]
    assert r.scores.dataset_score >= 99
    assert r.scores.grade == "A"
    assert len(r.dataset.episodes) == 12


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
@pytest.mark.parametrize(
    "fault, code, severity",
    [
        ("dead", "DEAD_EPISODE", Severity.ERROR),
        ("state_frozen", "STATE_FROZEN", Severity.ERROR),
        ("nan", "NONFINITE_VALUES", Severity.ERROR),
        ("spike", "ACTION_SPIKE", Severity.WARNING),
        ("dropped", "DROPPED_FRAMES", Severity.ERROR),
        ("nonmono", "TIMESTAMP_NONMONOTONIC", Severity.ERROR),
        ("jitter", "TIMESTAMP_JITTER", Severity.WARNING),
        ("fi_gap", "FRAME_INDEX_GAP", Severity.ERROR),
        ("short", "EPISODE_TOO_SHORT", Severity.ERROR),
        ("truncated", "EPISODE_SHORT", Severity.WARNING),
        ("idle", "IDLE_PADDING", Severity.WARNING),
        ("hard_limit", "JOINT_AT_LIMIT", Severity.WARNING),
        ("gripper_still", "GRIPPER_NEVER_ACTUATES", Severity.WARNING),
        ("length_meta", "EPISODE_LENGTH_MISMATCH", Severity.ERROR),
        ("bad_task", "TASK_INDEX_UNKNOWN", Severity.ERROR),
    ],
)
def test_each_fault_is_detected_only_where_injected(tmp_path, version, fault, code, severity):
    root = make_dataset(tmp_path / "ds", version=version, n_eps=10, faults={4: [fault]})
    r = lint(root)
    hits = [f for f in r.findings if f.code == code]
    assert hits, f"{code} not detected; got {sorted({f.code for f in r.findings})}"
    assert {f.episode for f in hits} == {4}
    assert max(f.severity for f in hits) == severity
    # healthy episodes stay clean
    for ep in (0, 1, 2, 7):
        assert not [f for f in r.findings if f.episode == ep and f.severity >= Severity.WARNING]
    assert r.scores.episode_scores[4] < 100
    assert r.scores.episode_scores[0] == 100


def test_duplicate_episode(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=8, faults={6: ["dup_of:2"]})
    r = lint(root)
    dup = [f for f in r.findings if f.code == "DUPLICATE_EPISODE"]
    assert len(dup) == 1 and dup[0].episode == 6 and dup[0].details["duplicate_of"] == 2


def test_dead_episodes_not_flagged_as_duplicates(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=8, faults={1: ["dead"], 5: ["dead"]})
    r = lint(root)
    assert not [f for f in r.findings if f.code == "DUPLICATE_EPISODE"]


def test_dataset_level_findings(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=6, dataset_faults=["count_mismatch", "vague_task"])
    r = lint(root)
    ds_codes = {f.code for f in r.dataset_findings}
    assert {"EPISODE_COUNT_MISMATCH", "TASK_VAGUE"} <= ds_codes


def test_scoring_and_bad_episode_list(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, faults={2: ["dead"], 5: ["spike"], 8: ["nan"]})
    r = lint(root)
    assert r.bad_episodes() == [2, 8]
    assert r.bad_episodes(Severity.WARNING) == [2, 5, 8]
    assert r.scores.episodes_with_errors == 2
    assert r.scores.episodes_with_warnings == 1
    assert r.scores.clean_episodes == 7
    assert r.scores.dataset_score < 100


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_video_clean(tmp_path, version):
    root = make_dataset(tmp_path / "ds", version=version, n_eps=5, videos=True, length=90)
    r = lint(root, videos=True)
    video_bad = [f for f in r.findings if f.code.startswith("VIDEO") and f.severity >= Severity.WARNING]
    assert video_bad == [], [f.message for f in video_bad]


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
@pytest.mark.parametrize(
    "fault, code",
    [("black_video", "VIDEO_BLACK"), ("frozen_video", "VIDEO_FROZEN"), ("short_video", "VIDEO_LENGTH_MISMATCH")],
)
def test_video_faults(tmp_path, version, fault, code):
    root = make_dataset(tmp_path / "ds", version=version, n_eps=5, videos=True, length=90, faults={3: [fault]})
    r = lint(root, videos=True)
    hits = [f for f in r.findings if f.code == code]
    assert hits and {f.episode for f in hits} == {3}, sorted((f.code, f.episode) for f in r.findings)


def test_missing_video_v2(tmp_path):
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=4, videos=True, length=60, faults={1: ["missing_video"]})
    r = lint(root, videos=True)
    assert {f.episode for f in r.findings if f.code == "VIDEO_MISSING"} == {1}


def test_cli_check_outputs(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", n_eps=6, faults={3: ["dead"]})
    j, h, x = tmp_path / "r.json", tmp_path / "r.html", tmp_path / "bad.txt"
    rc = main(["check", str(root), "--json", str(j), "--html", str(h), "--exclude-out", str(x), "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DEAD_EPISODE" in out and "Score" in out
    data = json.loads(j.read_text())
    assert data["episodes"]["with_errors"] == 1
    assert x.read_text() == "3\n"
    assert "<html" in h.read_text()
    assert main(["check", str(root), "--fail-under", "99.9", "--no-color"]) == 1


def test_cli_audit(tmp_path):
    a = make_dataset(tmp_path / "a", n_eps=5)
    b = make_dataset(tmp_path / "b", n_eps=5, faults={0: ["dead"]}, version="v2.1")
    out = tmp_path / "audit"
    rc = main(["audit", str(a), str(b), str(tmp_path / "does-not-exist"), "--out", str(out)])
    assert rc == 0
    md = (out / "summary.md").read_text()
    assert "failed to load" in md and "DEAD_EPISODE" in md
    assert len(json.loads((out / "summary.json").read_text())) == 3


def test_html_escapes_untrusted_text(tmp_path):
    evil = "<script>alert(1)</script>"
    root = make_dataset(tmp_path / "ds", n_eps=3, task=evil)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["robot_type"] = evil
    (root / "meta" / "info.json").write_text(json.dumps(info))
    h = tmp_path / "r.html"
    main(["check", str(root), "--html", str(h), "--no-color"])
    assert "<script>" not in h.read_text()


def test_not_a_dataset(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["check", str(tmp_path / "empty")]) == 2
    assert "info.json" in capsys.readouterr().err
    assert main(["check", "not a repo id!"]) == 2


def test_path_traversal_in_video_template_is_ignored(tmp_path):
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=2, videos=True, length=40)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["video_path"] = "../../../../etc/{video_key}/episode_{episode_index:06d}.mp4"
    (root / "meta" / "info.json").write_text(json.dumps(info))
    r = lint(root, videos=True)
    assert all(not ep.videos for ep in r.dataset.episodes)


def test_embedded_image_columns_are_skipped(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = make_dataset(tmp_path / "ds", n_eps=4)
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f)
    images = pa.array(
        [{"bytes": b"\x89PNG fake", "path": None}] * t.num_rows,
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    pq.write_table(t.append_column("observation.image", images), f)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.image"] = {"dtype": "image", "shape": [48, 64, 3], "names": None}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    r = lint(root)
    assert "observation.image" in r.dataset.data_columns
    assert "observation.image" not in r.dataset.episodes[0].frames
    assert not [f for f in r.findings if f.severity >= Severity.WARNING]


def test_scales_to_large_datasets(tmp_path):
    import time

    root = make_dataset(tmp_path / "ds", n_eps=1500, length=200, faults={700: ["dead"], 1200: ["dup_of:3"]})
    t0 = time.perf_counter()
    r = lint(root)
    elapsed = time.perf_counter() - t0
    assert r.dataset.total_frames == 1500 * 200
    assert 700 in r.bad_episodes()
    assert {f.episode for f in r.findings if f.code == "DUPLICATE_EPISODE"} == {1200}
    assert elapsed < 60, elapsed


def test_single_glitch_counts_as_one_event(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=8, faults={3: ["spike"]})
    (hit,) = [f for f in lint(root).findings if f.code == "ACTION_SPIKE"]
    assert hit.details["count"] == 1


def test_no_false_positives_on_idle_or_short_episodes(tmp_path):
    root = make_dataset(
        tmp_path / "ds", n_eps=10, length=180, videos=True, faults={2: ["idle"], 5: ["short"], 8: ["truncated"]}
    )
    r = lint(root, videos=True)
    assert not [f for f in r.findings if f.code in ("JOINT_AT_LIMIT", "VIDEO_FROZEN")], [
        (f.code, f.episode) for f in r.findings
    ]


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_quick_mode_is_partial_without_false_count_errors(tmp_path, version):
    root = make_dataset(tmp_path / "ds", version=version, n_eps=20, faults={2: ["dead"], 15: ["dead"]})
    r = lint(root, max_episodes=5)
    assert len(r.dataset.episodes) == 5 and r.dataset.partial
    found = {f.code for f in r.findings}
    assert "PARTIAL_CHECK" in found
    assert not found & {"EPISODE_COUNT_MISMATCH", "FRAME_COUNT_MISMATCH"}
    assert r.bad_episodes() == [2]


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_quick_patterns_select_only_needed_files(tmp_path, version):
    from robolint.dataset import quick_patterns

    root = make_dataset(tmp_path / "ds", version=version, n_eps=12, videos=True, length=40)
    pats = quick_patterns(root, 3, include_videos=True)
    assert pats and all((root / p).exists() for p in pats)
    if version.startswith("v2"):
        data = [p for p in pats if p.startswith("data/")]
        assert len(data) == 4  # episodes 0..3
    else:
        assert "data/chunk-000/file-000.parquet" in pats


def test_sample_and_hold_actions_are_not_glitches(tmp_path):
    # Real case: a dataset whose controller updates every 4th frame was flagged with hundreds of "jumps".
    root = make_dataset(tmp_path / "ds", n_eps=10, faults={i: ["hold"] for i in range(10)})
    r = lint(root)
    assert not [f for f in r.findings if f.code == "ACTION_SPIKE"]


def test_spike_detected_inside_sample_and_hold_data(tmp_path):
    faults = {i: ["hold"] for i in range(10)}
    faults[6] = ["hold", "spike"]
    r = lint(make_dataset(tmp_path / "ds", n_eps=10, faults=faults))
    assert {f.episode for f in r.findings if f.code == "ACTION_SPIKE"} == {6}


def test_resting_on_hard_stop_while_idle_is_not_flagged(tmp_path):
    # Real case: SO-100 arms rest on the shoulder stop before each take; that is not a fault.
    r = lint(make_dataset(tmp_path / "ds", n_eps=8, faults={3: ["rest_limit"]}))
    assert not [f for f in r.findings if f.code == "JOINT_AT_LIMIT"]
