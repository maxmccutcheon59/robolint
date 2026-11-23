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
        ("state_lag", "STATE_LAG_OUTLIER", Severity.WARNING),
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


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_quick_mode_takes_every_file_when_the_whole_dataset_is_wanted(tmp_path, version):
    # Real case: a dataset whose metadata lists fewer episodes (or 0) than its data holds. Picking
    # files from that metadata skipped the unlisted data files and graded part of the dataset.
    import json

    from robolint.dataset import quick_patterns

    root = make_dataset(tmp_path / "ds", version=version, n_eps=12, length=40)
    assert quick_patterns(root, 12, include_videos=False) is None
    assert quick_patterns(root, 300, include_videos=False) is None
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for declared in (0, 5, None):
        info["total_episodes"] = declared
        info_path.write_text(json.dumps(info))
        assert quick_patterns(root, 8, include_videos=False) is None, declared


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


def test_fixture_faults_actually_change_the_data(tmp_path):
    """Guard against vacuous tests: every fault name used above must alter the generated actions."""
    import numpy as np

    base = lint(make_dataset(tmp_path / "base", n_eps=2, seed=1)).dataset.episodes[0].frames["action"]
    for fault in ["hold", "rest_limit", "one_arm_rest", "binary_gripper", "hard_limit", "spike", "idle"]:
        ds = lint(make_dataset(tmp_path / fault, n_eps=2, seed=1, faults={0: [fault]})).dataset
        assert not np.allclose(ds.episodes[0].frames["action"], base), fault


def test_binary_gripper_toggles_are_not_glitches(tmp_path):
    # Real case: LIBERO's on/off gripper command was flagged as glitching.
    r = lint(make_dataset(tmp_path / "ds", n_eps=8, faults={i: ["binary_gripper"] for i in range(8)}))
    assert not [f for f in r.findings if f.code == "ACTION_SPIKE"]


def test_joint_resting_on_stop_while_others_work_is_calibration_info(tmp_path):
    # Real case: a bimanual SO-100 dataset; the left shoulder rests on its stop while the other arm works.
    r = lint(make_dataset(tmp_path / "ds", n_eps=8, faults={3: ["one_arm_rest"]}))
    assert not [f for f in r.findings if f.code == "JOINT_AT_LIMIT"]
    hits = [f for f in r.findings if f.code == "CALIBRATION_OFFSET"]
    assert {f.episode for f in hits} == {3} and all(f.severity == Severity.INFO for f in hits)


def test_failed_grasp_is_an_error_and_regrasp_is_info(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=12, grasp_object=True, faults={4: ["missed_grasp"], 8: ["regrasp"]})
    r = lint(root)
    failed = [f for f in r.findings if f.code == "FAILED_GRASP"]
    regrasp = [f for f in r.findings if f.code == "REGRASP"]
    assert {f.episode for f in failed} == {4} and failed[0].severity == Severity.ERROR
    assert {f.episode for f in regrasp} == {8} and regrasp[0].severity == Severity.INFO


@pytest.mark.parametrize("grasp_object", [False, True])
def test_no_failed_grasps_when_every_grasp_is_alike(tmp_path, grasp_object):
    # Either every grasp holds an object, or the gripper always closes freely (e.g. a pushing task):
    # neither should produce grasp findings.
    r = lint(make_dataset(tmp_path / "ds", n_eps=12, grasp_object=grasp_object))
    assert not {"FAILED_GRASP", "REGRASP"} & {f.code for f in r.findings}


def test_smaller_objects_are_not_mistaken_for_missed_grasps(tmp_path):
    small = {i: ["small_object"] for i in (1, 5, 9)}
    r = lint(make_dataset(tmp_path / "ds", n_eps=12, grasp_object=True, faults={**small, 6: ["missed_grasp"]}))
    assert {f.episode for f in r.findings if f.code == "FAILED_GRASP"} == {6}


def test_rest_poses_and_open_width_are_not_grasps(tmp_path):
    # Real teleop takes start and end with the gripper resting fully closed on nothing, and operators
    # open by different amounts. Neither is a missed grasp (both were false positives on SO-100 and ALOHA data).
    faults = {i: ["pick_place"] for i in range(12)}
    r = lint(make_dataset(tmp_path / "ds", n_eps=12, faults=faults))
    assert not {"FAILED_GRASP", "REGRASP"} & {f.code for f in r.findings}


def test_missed_grasp_is_found_between_rest_poses(tmp_path):
    faults = {i: ["pick_place"] for i in range(12)}
    faults[7] = ["pick_place_missed"]
    r = lint(make_dataset(tmp_path / "ds", n_eps=12, faults=faults))
    assert {f.episode for f in r.findings if f.code == "FAILED_GRASP"} == {7}
    assert not [f for f in r.findings if f.code == "REGRASP"]


def test_rest_after_soft_grasp_is_not_a_miss(tmp_path):
    # Operators often hold the object without squeezing the leader fully closed, so the grasp never
    # reaches the "held closed" band; the gripper then goes back to rest closed on nothing at the end.
    # That final rest is not a grasp (it was 21 false FAILED_GRASP on svla_so100_stacking).
    # Gentle grasps barely hold the follower off the command (svla_so100_stacking eps 43, 48).
    faults = {i: ["pick_place"] for i in range(12)}
    for i in (8, 9):
        faults[i].append("soft_grasp")
    for i in (10, 11):
        faults[i].append("gentle_grasp")
    r = lint(make_dataset(tmp_path / "ds", n_eps=12, faults=faults))
    assert not {"FAILED_GRASP", "REGRASP"} & {f.code for f in r.findings}


def test_bimanual_grippers_pair_by_name():
    import numpy as np

    from robolint.checks.signals import DimStats, _gripper_pairs

    def st(names):
        z = np.zeros(len(names))
        return DimStats(z, z, z, z, names, z.astype(bool))

    a = st(["left_waist", "left_gripper", "right_waist", "right_gripper"])
    s = st(["right_gripper", "right_waist", "left_gripper", "left_waist"])
    assert _gripper_pairs(a, s) == [(1, 2), (3, 0)]


# --- action-to-state lag and action/video sync ---------------------------------------------------

LAG_CODES = {"STATE_LAG", "STATE_LAG_OUTLIER", "VIDEO_LAG", "VIDEO_ACTION_OFFSET"}


def lag_findings(r):
    return [f for f in r.findings if f.code in LAG_CODES]


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_clean_data_reports_typical_lags_and_flags_nothing(tmp_path, version):
    root = make_dataset(tmp_path / "ds", version=version, n_eps=10, length=150, videos=True)
    r = lint(root, videos=True)
    found = lag_findings(r)
    assert not [f for f in found if f.severity >= Severity.WARNING], [f.message for f in found]
    state = [f for f in found if f.code == "STATE_LAG"]
    video = [f for f in found if f.code == "VIDEO_LAG"]
    assert len(state) == 1 and state[0].episode is None and state[0].severity == Severity.INFO
    assert state[0].details["typical_lag_frames"] == pytest.approx(2.0, abs=0.5)  # synth state trails by 2
    assert len(video) == 1 and video[0].details["typical_lag_frames"] == pytest.approx(0.0, abs=1.0)
    assert r.scores.dataset_score >= 99


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_shifted_video_is_flagged_only_where_injected(tmp_path, version):
    root = make_dataset(
        tmp_path / "ds", version=version, n_eps=10, length=150, videos=True, faults={4: ["video_shift"]}
    )
    r = lint(root, videos=True)
    hits = [f for f in r.findings if f.code == "VIDEO_ACTION_OFFSET"]
    assert [f.episode for f in hits] == [4]
    assert hits[0].severity == Severity.WARNING
    assert hits[0].details["offset_frames"] == pytest.approx(10, abs=2)
    assert not [f for f in r.findings if f.code == "STATE_LAG_OUTLIER"]  # the state stream is fine


def test_lagged_state_is_not_blamed_on_the_video(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, videos=True, faults={4: ["state_lag"]})
    r = lint(root, videos=True)
    lag = [f for f in r.findings if f.code == "STATE_LAG_OUTLIER"]
    assert [f.episode for f in lag] == [4]
    assert lag[0].details["offset_frames"] == pytest.approx(10, abs=1)
    assert not [f for f in r.findings if f.code == "VIDEO_ACTION_OFFSET"]


def test_sync_check_needs_videos_and_degrades_without_them(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, videos=True, faults={4: ["video_shift"]})
    r = lint(root)  # no --videos: the state lag still runs, the video sync does not
    codes_found = {f.code for f in r.findings}
    assert "STATE_LAG" in codes_found
    assert not codes_found & {"VIDEO_LAG", "VIDEO_ACTION_OFFSET"}
    # no camera declared at all
    plain = make_dataset(tmp_path / "plain", n_eps=10, length=150)
    assert not {f.code for f in lint(plain, videos=True).findings} & {"VIDEO_LAG", "VIDEO_ACTION_OFFSET"}


def test_sync_check_without_pyav(tmp_path, monkeypatch):
    import builtins

    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, videos=True, faults={4: ["video_shift"]})
    real_import = builtins.__import__

    def no_av(name, *args, **kwargs):
        if name == "av" or name.startswith("av."):
            raise ImportError("no PyAV")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_av)
    r = lint(root, videos=True)
    assert "VIDEO_SKIPPED" in {f.code for f in r.findings}
    assert not {f.code for f in r.findings} & {"VIDEO_LAG", "VIDEO_ACTION_OFFSET"}


def test_missing_video_files_do_not_flag_sync(tmp_path):
    root = make_dataset(
        tmp_path / "ds", version="v2.1", n_eps=10, length=150, videos=True, faults={3: ["missing_video"]}
    )
    r = lint(root, videos=True)
    assert {f.episode for f in r.findings if f.code == "VIDEO_MISSING"} == {3}
    assert not [f for f in r.findings if f.code == "VIDEO_ACTION_OFFSET"]


def test_single_joint_dataset(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, dims=1, faults={4: ["state_lag"]})
    r = lint(root)
    assert [f.episode for f in r.findings if f.code == "STATE_LAG_OUTLIER"] == [4]
    clean = make_dataset(tmp_path / "clean", n_eps=10, length=150, dims=1)
    assert not [f for f in lint(clean).findings if f.code == "STATE_LAG_OUTLIER"]


def test_static_episodes_are_not_judged(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, videos=True, faults={i: ["dead"] for i in (1, 3, 5, 7)})
    r = lint(root, videos=True)
    flagged = {f.episode for f in r.findings if f.code in {"STATE_LAG_OUTLIER", "VIDEO_ACTION_OFFSET"}}
    assert not flagged
    state = [f for f in r.findings if f.code == "STATE_LAG"]
    assert state and state[0].details["episodes_estimated"] == 6  # the four dead episodes are skipped
    # nothing moves anywhere: no estimate at all, rather than a guess
    still = make_dataset(tmp_path / "still", n_eps=10, length=150, faults={i: ["dead"] for i in range(10)})
    assert not lag_findings(lint(still))


def test_too_few_episodes_gives_no_lag_verdict(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=4, length=150, faults={2: ["state_lag"]})
    assert not lag_findings(lint(root))


def test_state_that_does_not_follow_the_command_is_not_judged(tmp_path):
    # e.g. delta or velocity actions: they say nothing about where the arm is, so neither the state
    # lag nor the video sync is estimated (the sync would be matching the wrong signal).
    root = make_dataset(
        tmp_path / "ds", n_eps=10, length=150, videos=True, faults={i: ["state_unrelated"] for i in range(10)}
    )
    assert not lag_findings(lint(root, videos=True))


def test_repeating_motion_one_cycle_over_is_not_an_outlier():
    import numpy as np

    from robolint.checks.lag import LagCurve, judge

    lags = np.arange(-30, 31)

    def bump(center, height):
        return height * np.exp(-((lags - center) ** 2) / 20.0)

    # Every episode lines up at 0; a rhythmic one also lines up almost as well 20 frames later.
    curves = {i: LagCurve(bump(0, 0.9) + 0.01 * i, 30) for i in range(8)}
    curves[8] = LagCurve(np.maximum(bump(0, 0.9), bump(20, 0.95)), 30)
    kw = dict(min_corr=0.5, flag_corr=0.7, min_episodes=5, tol_seconds=0.15, tol_frames=3, mad_factor=5, margin=0.1)
    summary = judge(curves, 30.0, **kw)
    assert summary is not None and summary.typical == pytest.approx(0, abs=0.5)
    assert summary.outliers == []
    # ... but an episode that only lines up 20 frames later is.
    curves[8] = LagCurve(bump(20, 0.95), 30)
    assert [o.episode for o in judge(curves, 30.0, **kw).outliers] == [8]


def test_rest_steps_are_not_spikes():
    import numpy as np

    from robolint.checks.signals import drop_rest_steps

    col = np.zeros(60)
    col[20:40] = 0.3  # joystick pushed past its deadzone and held
    col[50] = 5.0  # a one-frame glitch out of rest
    a = col[:, None]
    jumps = np.zeros_like(a, dtype=bool)
    jumps[[19, 20, 21, 39, 40, 50], 0] = True
    kept = drop_rest_steps(a, jumps)
    assert not kept[[19, 20, 21, 39, 40], 0].any()
    assert kept[50, 0]


def test_rest_steps_leave_continuous_channels_alone():
    import numpy as np

    from robolint.checks.signals import drop_rest_steps

    a = np.linspace(0, 10, 100)[:, None]  # no value held for a fifth of the episode
    jumps = np.zeros_like(a, dtype=bool)
    jumps[50, 0] = True
    assert drop_rest_steps(a, jumps)[50, 0]
