"""Camera stream checks: missing files, length mismatches, black/blown-out/frozen feeds."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from ..dataset import Dataset, Episode, VideoRef
from ..findings import Finding, Severity
from . import Config

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
    with ExitStack() as stack:
        opened: dict[Path, Any] = {}
        reported_stream: set[str] = set()
        for ep in ds.episodes:
            moving = _actions_move(ep)
            for ref in ep.videos:
                if not ref.path.exists():
                    yield Finding(
                        "VIDEO_MISSING", E, f"camera '{ref.key}' video file is missing", ep.index, {"key": ref.key}
                    )
                    continue
                container = opened.get(ref.path)
                if container is None:
                    try:
                        container = stack.enter_context(_open(ref.path))
                    except Exception as e:  # noqa: BLE001 - any decode/open failure is a finding
                        yield Finding(
                            "VIDEO_UNREADABLE", E, f"camera '{ref.key}' video cannot be opened: {e}", ep.index
                        )
                        continue
                    opened[ref.path] = container
                if ref.key not in reported_stream:
                    reported_stream.add(ref.key)
                    yield from _stream_props(ds, ref.key, container, fps)
                yield from _episode_video(ep, ref, container, fps, moving, cfg, motion)

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
