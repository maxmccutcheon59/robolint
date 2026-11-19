"""Frame timestamps, frozen stretches and per-file stream properties (robolint check --videos)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("av")

from robolint import lint  # noqa: E402
from robolint.checks.video import NO_TS, _Packets, segment_timing  # noqa: E402

from .synth import CAM, H, W, _frame_for, _write_video, make_dataset  # noqa: E402

TIMESTAMP_AND_FREEZE = {
    "VIDEO_DROPPED_FRAMES",
    "VIDEO_DUPLICATE_FRAMES",
    "VIDEO_TIMESTAMPS_UNORDERED",
    "VIDEO_FREEZE",
    "VIDEO_FROZEN",
}


def _video(r, codes=TIMESTAMP_AND_FREEZE):
    return [f for f in r.findings if f.code in codes]


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_dropped_frames_are_found_from_timestamps(tmp_path, version):
    root = make_dataset(tmp_path / "ds", version=version, n_eps=6, length=150, videos=True, faults={3: ["drop_frames"]})
    r = lint(root, videos=True)
    found = _video(r)
    assert [(f.code, f.episode) for f in found] == [("VIDEO_DROPPED_FRAMES", 3)]
    d = found[0].details
    assert d["missing_frames"] == 5 and d["gaps"] == 1
    assert d["first_gap_at_s"] == pytest.approx(74 / 30, abs=1e-3)
    assert found[0].severity.label == "error"
    # The rate comes from the regular spacing of the timestamps, so the lost frames don't make the
    # camera look slow, and the episode still covers its full length.
    assert not {f.code for f in r.findings} & {"VIDEO_FPS_MISMATCH", "VIDEO_LENGTH_MISMATCH"}


def test_segment_timing_counts_gaps_duplicates_and_backwards():
    period = 512
    pts = np.array([0, 1, 2, 3, 7, 8, 8, 9, 10, 20, 21], np.int64) * period  # gaps of 3 and 9 frames, one repeat
    dts = np.arange(len(pts), dtype=np.int64) * period
    dts[5] = dts[4]  # a decode timestamp that does not increase
    pk = _Packets(pts=pts, dts=dts, key=np.ones(len(pts), bool), tb=1 / 15360, period=period / 15360, no_pts=0)
    t = segment_timing(pk, 0.0, None, 30.0, 1.5)
    assert (t.missing, t.gaps, t.duplicates, t.backwards) == (3 + 9, 2, 1, 1)
    assert t.longest_gap == pytest.approx(10 / 30)
    assert t.first_gap_at == pytest.approx(3 / 30)
    # Only the episode's own segment counts: the gap before frame 20 is outside [0, 10/30 s).
    t = segment_timing(pk, 0.0, 11 / 30, 30.0, 1.5)
    assert (t.missing, t.gaps) == (3, 1)
    # Packets without a decode timestamp are not counted as going backwards.
    pk.dts[:] = NO_TS
    assert segment_timing(pk, 0.0, None, 30.0, 1.5).backwards == 0


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_frozen_stretch_mid_episode_is_found(tmp_path, version):
    root = make_dataset(
        tmp_path / "ds", version=version, n_eps=10, length=150, videos=True, faults={4: ["freeze_stretch"]}
    )
    found = _video(lint(root, videos=True))
    assert [(f.code, f.episode) for f in found] == [("VIDEO_FREEZE", 4)]
    d = found[0].details
    # frames 60 to 104 repeat one image: 2.0 s to 3.5 s, seen at four samples a second
    assert 1.9 <= d["start_s"] <= 2.3 and 3.3 <= d["end_s"] <= 3.6
    assert d["change"] < 0.05 < 1.0 < d["expected_change"]


def test_frozen_stretch_is_found_without_frequent_key_frames(tmp_path):
    # A key frame only every 30 frames: the episode is decoded in full and sampled instead.
    root = make_dataset(
        tmp_path / "ds", n_eps=10, length=150, videos=True, faults={4: ["freeze_stretch"]}, video_gop=30
    )
    assert [(f.code, f.episode) for f in _video(lint(root, videos=True))] == [("VIDEO_FREEZE", 4)]


def test_motion_the_camera_does_not_show_is_not_a_freeze(tmp_path):
    # For 2 s of several episodes only the wrist roll moves, which this camera does not show (as when
    # one arm of a bimanual robot works while the camera watches the other). Nothing is frozen.
    faults = {e: ["unseen_motion"] for e in (1, 4, 6, 8)}
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, videos=True, faults=faults)
    assert _video(lint(root, videos=True)) == []


def test_camera_that_sees_none_of_the_moving_joints_is_not_frozen(tmp_path):
    # A whole episode in which only joints the camera does not show move: the old check flagged it
    # as frozen because some action moved.
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, videos=True, faults={5: ["unseen_all"]})
    assert _video(lint(root, videos=True)) == []


def test_whole_frozen_episode_is_reported_once(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, length=150, videos=True, faults={4: ["frozen_video"]})
    assert [(f.code, f.episode) for f in _video(lint(root, videos=True))] == [("VIDEO_FROZEN", 4)]


def test_file_without_a_video_stream_is_unreadable_not_a_crash(tmp_path):
    import av

    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=4, length=60, videos=True)
    path = root / "videos" / "chunk-000" / CAM / "episode_000002.mp4"
    with av.open(str(path), "w") as out:
        st = out.add_stream("aac", rate=8000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp", layout="mono")
        frame.sample_rate = 8000
        for p in st.encode(frame):
            out.mux(p)
        for p in st.encode():
            out.mux(p)
    r = lint(root, videos=True)
    bad = [f for f in r.findings if f.code == "VIDEO_UNREADABLE"]
    assert [f.episode for f in bad] == [2] and "no video stream" in bad[0].message


def _reencode(path, size):
    import av

    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    small = [np.ascontiguousarray(f[:: H // size[1], :: W // size[0]][: size[1], : size[0]]) for f in frames]
    with av.open(str(path), "w") as out:
        st = out.add_stream("mpeg4", rate=30)
        st.width, st.height, st.pix_fmt = size[0], size[1], "yuv420p"
        for f in small:
            for p in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)


@pytest.mark.parametrize("declared", [True, False])
def test_stream_properties_are_checked_in_every_file(tmp_path, declared):
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=5, length=60, videos=True)
    _reencode(root / "videos" / "chunk-000" / CAM / "episode_000003.mp4", (32, 24))
    if not declared:
        info = json.loads((root / "meta" / "info.json").read_text())
        del info["features"][CAM]["shape"]
        (root / "meta" / "info.json").write_text(json.dumps(info))
    r = lint(root, videos=True)
    shape = [f for f in r.findings if f.code == "VIDEO_SHAPE_MISMATCH"]
    assert len(shape) == 1 and shape[0].details["files"] == 1 and shape[0].details["files_checked"] == 5
    if declared:
        assert shape[0].message.endswith("is 32x24 but declared 64x48 (1 of 5 files)")
    else:
        assert shape[0].message.endswith("is 32x24 in 1 of 5 files but 64x48 in the rest")


def test_frame_rate_mismatch_names_the_files_and_skips_only_their_sync(tmp_path):
    root = make_dataset(tmp_path / "ds", version="v2.1", n_eps=10, length=150, videos=True)
    path = root / "videos" / "chunk-000" / CAM / "episode_000007.mp4"
    frames = [_frame_for(np.zeros(6))] * 75
    _write_video(path, frames, 15)  # half the declared rate, same duration
    r = lint(root, videos=True)
    fps = [f for f in r.findings if f.code == "VIDEO_FPS_MISMATCH"]
    assert len(fps) == 1 and fps[0].details == {"key": CAM, "files": 1, "files_checked": 10}
    assert "runs at 15 fps but the dataset declares 30 (1 of 10 files)" in fps[0].message
    lag = [f for f in r.findings if f.code == "VIDEO_LAG"]
    assert lag and lag[0].details["episodes_estimated"] >= 8  # the other files are still lined up
