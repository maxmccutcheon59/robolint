"""Camera stream checks: missing files, length mismatches, black/blown-out/frozen feeds, and
whether each camera's motion lines up in time with the commanded motion."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import Dataset, Episode, VideoRef
from ..findings import Finding, Severity
from . import Config
from .lag import LagCurve, PositionJoints, fit_curve, judge, position_joints, smooth
from .signals import _as2d

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO


def check(ds: Dataset, cfg: Config) -> Iterator[Finding]:
    try:
        import av  # noqa: F401
    except ImportError:
        yield Finding("VIDEO_SKIPPED", I, "video checks need PyAV: pip install 'robolint[video]'")
        return
    if not ds.video_keys:
        return
    fps = ds.fps
    if fps is None:
        return

    motion: dict[str, list[tuple[int, float]]] = {}  # camera -> [(episode, max frame-to-frame change)]
    # Only position commands describe where the arm is, so only they predict what a camera sees.
    joints = position_joints(ds, cfg)
    sync: dict[str, dict[int, LagCurve]] = {}  # camera -> episode -> how well video lines up per shift
    no_sync: set[str] = set()  # cameras whose frame rate differs from the dataset's
    # Keep only a few containers open at once: v2.x has one file per episode and camera, so a
    # dataset can reference thousands of files (and run out of file descriptors). v3.x shares
    # files between consecutive episodes, which a small LRU cache still reuses.
    cache = _ContainerCache(MAX_OPEN_VIDEOS)
    checked_files: set[Path] = set()
    stream_findings: set[tuple[str, str]] = set()
    try:
        for ep in ds.episodes:
            moving = _actions_move(ep)
            for ref in ep.videos:
                if not ref.path.exists():
                    yield Finding(
                        "VIDEO_MISSING", E, f"camera '{ref.key}' video file is missing", ep.index, {"key": ref.key}
                    )
                    continue
                try:
                    container = cache.get(ref.path)
                except Exception as e:  # noqa: BLE001 - any decode/open failure is a finding
                    yield Finding("VIDEO_UNREADABLE", E, f"camera '{ref.key}' video cannot be opened: {e}", ep.index)
                    continue
                if ref.path not in checked_files:  # stream properties of every file, reported once per issue
                    checked_files.add(ref.path)
                    for f in _stream_props(ds, ref.key, container, fps):
                        if f.code == "VIDEO_FPS_MISMATCH":
                            no_sync.add(ref.key)
                        if (f.code, ref.key) not in stream_findings:
                            stream_findings.add((f.code, ref.key))
                            yield f
                found = list(_episode_video(ep, ref, container, fps, moving, cfg, motion))
                yield from found
                # A broken feed (black, blank, wrong length, unreadable) has no motion to line up.
                if joints is not None and ref.key not in no_sync and not any(f.severity >= W for f in found):
                    try:
                        curve = _sync_curve(ep, ref, container, fps, joints, cfg)
                    except Exception:  # noqa: BLE001 - a decode problem here is not a sync finding
                        curve = None
                    if curve is not None:
                        sync.setdefault(ref.key, {})[ep.index] = curve
    finally:
        cache.close()

    # A frozen feed still flickers a little from compression, so judge each episode
    # against how much this camera normally changes across the dataset.
    for key, rows in motion.items():
        typical = float(np.median([m for _, m in rows]))
        threshold = max(cfg.frozen_diff, 0.2 * typical)
        for ep_idx, m in rows:
            if m < threshold:
                yield Finding(
                    "VIDEO_FROZEN",
                    W,
                    f"camera '{key}' image barely changes while the robot moves (change {m:.1f} vs typical {typical:.1f}): "
                    "the feed may be frozen",
                    ep_idx,
                    {"key": key, "change": round(m, 2), "typical": round(typical, 2)},
                )

    yield from _sync_findings(sync, fps, cfg)


MAX_OPEN_VIDEOS = 8


class _ContainerCache:
    """Small LRU of open PyAV containers; evicted containers are closed immediately."""

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self._open: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path):
        c = self._open.get(path)
        if c is not None:
            self._open.move_to_end(path)
            return c
        c = _open(path)
        self._open[path] = c
        while len(self._open) > self.size:
            _, old = self._open.popitem(last=False)
            old.close()
        return c

    def close(self) -> None:
        while self._open:
            _, c = self._open.popitem(last=False)
            try:
                c.close()
            except Exception:  # noqa: BLE001 - closing must never mask a result
                pass


def _open(path: Path):
    import av

    return av.open(str(path))


def _actions_move(ep: Episode) -> bool:
    a = ep.frames.get("action")
    if a is None or len(a) < 2:
        return True
    a = a.reshape(len(a), -1)
    if not np.all(np.isfinite(a)):
        return True
    spread = a.max(0) - a.min(0)
    return bool(np.any(spread > 1e-6 * (np.abs(a).max(0) + 1.0)))


def _file_duration(container, stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / 1_000_000
    return None


def _stream_props(ds: Dataset, key: str, container, fps: float) -> Iterator[Finding]:
    stream = container.streams.video[0]
    rate = float(stream.average_rate) if stream.average_rate else None
    if rate and abs(rate - fps) > 0.5:
        yield Finding("VIDEO_FPS_MISMATCH", W, f"camera '{key}' runs at {rate:g} fps but the dataset declares {fps:g}")
    shape = ds.features.get(key, {}).get("shape")
    cc = stream.codec_context
    if isinstance(shape, list) and len(shape) >= 2 and all(isinstance(x, int) for x in shape[:2]):
        if (cc.height, cc.width) != (shape[0], shape[1]):
            yield Finding(
                "VIDEO_SHAPE_MISMATCH",
                W,
                f"camera '{key}' is {cc.width}x{cc.height} but declared {shape[1]}x{shape[0]}",
            )


def _sample(container, stream, t: float) -> np.ndarray | None:
    tb = stream.time_base
    container.seek(int(t / tb), stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        if frame.pts * tb + 1e-6 >= t:
            return frame.reformat(width=64, height=48, format="gray").to_ndarray().astype(np.float32)
    return None


def _episode_video(
    ep: Episode,
    ref: VideoRef,
    container,
    fps: float,
    moving: bool,
    cfg: Config,
    motion: dict[str, list[tuple[int, float]]],
) -> Iterator[Finding]:
    stream = container.streams.video[0]
    file_dur = _file_duration(container, stream)
    start = ref.from_ts
    end = ref.to_ts if ref.to_ts is not None else file_dur
    expected = ep.length / fps
    if end is None:
        return
    seg = end - start
    tol = max(2.0 / fps, 0.02 * expected)
    if abs(seg - expected) > tol:
        yield Finding(
            "VIDEO_LENGTH_MISMATCH",
            E,
            f"camera '{ref.key}' covers {seg:.2f}s but the episode is {expected:.2f}s: video and actions are out of sync",
            ep.index,
            {"key": ref.key, "video_s": round(seg, 3), "episode_s": round(expected, 3)},
        )
    if file_dur is not None and end > file_dur + 1.0 / fps:
        yield Finding(
            "VIDEO_SEGMENT_OUT_OF_RANGE",
            E,
            f"camera '{ref.key}' segment ends at {end:.2f}s, past the end of the file ({file_dur:.2f}s)",
            ep.index,
        )
        return

    last = max(start, min(end, start + expected) - 1.0 / fps)
    times = np.linspace(start, last, num=max(2, cfg.video_samples))
    frames: list[np.ndarray] = []
    try:
        for t in times:
            f = _sample(container, stream, float(t))
            if f is not None:
                frames.append(f)
    except Exception as e:  # noqa: BLE001
        yield Finding("VIDEO_UNREADABLE", E, f"camera '{ref.key}' failed to decode: {e}", ep.index)
        return
    if not frames:
        yield Finding("VIDEO_UNREADABLE", E, f"camera '{ref.key}' produced no frames for this episode", ep.index)
        return

    means = np.array([f.mean() for f in frames])
    stds = np.array([f.std() for f in frames])
    k = len(frames)
    black = int(np.sum(means < cfg.black_level))
    white = int(np.sum(means > cfg.white_level))
    blank = int(np.sum((stds < cfg.blank_std) & (means >= cfg.black_level) & (means <= cfg.white_level)))
    if black:
        yield Finding(
            "VIDEO_BLACK",
            E if black == k else W,
            f"camera '{ref.key}' is black in {black}/{k} sampled frames (lens cap, disconnected, or no light)",
            ep.index,
            {"key": ref.key},
        )
    if white:
        yield Finding(
            "VIDEO_OVEREXPOSED",
            E if white == k else W,
            f"camera '{ref.key}' is blown out in {white}/{k} sampled frames",
            ep.index,
            {"key": ref.key},
        )
    if blank:
        yield Finding(
            "VIDEO_BLANK",
            E if blank == k else W,
            f"camera '{ref.key}' shows a uniform image in {blank}/{k} sampled frames",
            ep.index,
            {"key": ref.key},
        )
    # needs a couple of seconds of motion to tell a frozen feed from a slow one
    if k >= 3 and moving and expected >= cfg.frozen_min_seconds and not black and not white and not blank:
        # compare each sample to the first: robust to codec flicker between neighbours
        change = max(float(np.mean(np.abs(f - frames[0]))) for f in frames[1:])
        motion.setdefault(ref.key, []).append((ep.index, change))


def _sync_curve(
    ep: Episode, ref: VideoRef, container, fps: float, joints: PositionJoints, cfg: Config
) -> LagCurve | None:
    """How well this camera's image change is explained by the commanded joint speeds, for each
    shift of the video against the actions. None when there is too little motion to go on."""
    a = ep.frames.get("action")
    fi = ep.frames.get("frame_index")
    if a is None or (fi is not None and len(np.unique(fi)) < len(fi)):
        return None
    a = _as2d(a).astype(np.float64)
    if a.shape[1] <= int(joints.action_dims.max()) or not np.all(np.isfinite(a)):
        return None
    n = min(len(a), int(cfg.sync_max_seconds * fps))
    width = max(1, int(round(cfg.sync_smooth_seconds * fps)))
    max_lag = max(3, int(round(cfg.sync_max_lag_seconds * fps)))
    if n < max(int(cfg.lag_min_seconds * fps), 3 * max_lag, 4 * width):
        return None
    a = a[:n, joints.action_dims] / joints.rng
    travel = a.max(axis=0) - a.min(axis=0)
    if not np.any(travel >= cfg.lag_min_travel):
        return None  # the robot barely moves: nothing to line up
    frames = _decode_range(container, ref.from_ts, n, fps)
    if frames is None:
        return None
    video = smooth(np.array([np.mean(np.abs(frames[i] - frames[i - 1])) for i in range(1, n)]), width)
    speeds = smooth(np.abs(np.diff(a[:, travel >= cfg.lag_min_travel], axis=0)), width)
    if np.std(video) < 1e-3:
        return None  # a still image (or one that only flickers): nothing to line up
    return fit_curve(speeds, video, max_lag)


def _decode_range(container, start: float, n: int, fps: float) -> list[np.ndarray] | None:
    """Every frame of the n-frame span starting at `start` seconds, small and grey, or None if any
    frame of it is missing (a gap would fake a jump in motion)."""
    stream = container.streams.video[0]
    tb = stream.time_base
    if tb is None:
        return None
    container.seek(int(start / tb), stream=stream, backward=True, any_frame=False)
    out: list[np.ndarray | None] = [None] * n
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        i = int(round((float(frame.pts * tb) - start) * fps))
        if i < 0:
            continue
        if i >= n:
            break
        out[i] = frame.reformat(width=32, height=24, format="gray").to_ndarray().astype(np.float32)
    if any(f is None for f in out):
        return None
    return out  # type: ignore[return-value]


def _sync_findings(sync: dict[str, dict[int, LagCurve]], fps: float, cfg: Config) -> Iterator[Finding]:
    """Judge each episode's camera delay against that camera's typical delay across the dataset."""
    for key, curves in sync.items():
        summary = judge(
            curves,
            fps,
            min_corr=cfg.sync_min_corr,
            flag_corr=cfg.sync_flag_corr,
            min_episodes=cfg.lag_min_episodes,
            tol_seconds=cfg.sync_tol_seconds,
            tol_frames=cfg.sync_tol_frames,
            mad_factor=cfg.sync_mad_factor,
            margin=cfg.sync_margin,
        )
        if summary is None:
            continue
        typical = summary.typical
        ms = typical / fps * 1000
        if abs(typical) < 0.5:
            when = "in the same frame as it is commanded"
        else:
            when = (
                f"{abs(typical):.1f} frames ({abs(ms):.0f} ms) {'after' if typical > 0 else 'before'} it is commanded"
            )
        yield Finding(
            "VIDEO_LAG",
            I,
            f"camera '{key}' typically shows motion {when}, estimated over {summary.episodes} episodes",
            None,
            {
                "key": key,
                "typical_lag_frames": round(typical, 2),
                "typical_lag_ms": round(ms, 1),
                "spread_frames": round(summary.spread, 2),
                "episodes_estimated": summary.episodes,
            },
        )
        for o in summary.outliers:
            lag = o.estimate.lag
            off = lag - typical
            yield Finding(
                "VIDEO_ACTION_OFFSET",
                W,
                f"camera '{key}' shows motion {abs(off):.1f} frames ({abs(off) / fps * 1000:.0f} ms) "
                f"{'later' if off > 0 else 'earlier'} than in the rest of the dataset (delay {lag:.1f} vs "
                f"typical {typical:.1f} frames): video and actions may be out of sync",
                o.episode,
                {
                    "key": key,
                    "lag_frames": round(lag, 2),
                    "typical_frames": round(typical, 2),
                    "offset_frames": round(off, 2),
                    "confidence": round(o.estimate.peak, 3),
                    "fit_gain": round(o.margin, 3),
                },
            )
