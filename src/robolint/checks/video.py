"""Camera stream checks: missing or unreadable files, stream properties of every file, frames
missing or out of order in the timestamps, length mismatches, black/blown-out/frozen feeds,
stretches where the image stops changing while the robot keeps moving, and whether each camera's
motion lines up in time with the commanded motion.

Timestamps are read from the container's packets without decoding anything, so every frame of
every file is covered. Images are decoded sparingly: a few samples per episode for the
black/blank/frozen checks, about four frames per second (key frames where the encoding has them
that often) for frozen stretches, and the first seconds of each episode for the sync check."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dataset import Dataset, Episode, VideoRef
from ..findings import Finding, Severity
from . import Config
from .lag import LagCurve, PositionJoints, fit_curve, judge, nonneg_coef, position_joints, smooth
from .signals import _as2d, dataset_stats

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
    travel_dims = _travel_dims(ds, cfg, joints)
    sync: dict[str, dict[int, LagCurve]] = {}  # camera -> episode -> how well video lines up per shift
    stills: dict[str, list[_Series]] = {}  # camera -> sampled image change per episode, for frozen stretches
    files: dict[str, dict[Path, _FileProps]] = {}  # camera -> stream properties of each of its files
    # Keep only a few containers open at once: v2.x has one file per episode and camera, so a
    # dataset can reference thousands of files (and run out of file descriptors). v3.x shares
    # files between consecutive episodes, which a small LRU cache still reuses.
    cache = _ContainerCache(MAX_OPEN_VIDEOS)
    try:
        for ep in ds.episodes:
            moving = _actions_move(ep)
            pos = _positions(ep, travel_dims) if travel_dims is not None else None
            for ref in ep.videos:
                if not ref.path.exists():
                    yield Finding(
                        "VIDEO_MISSING", E, f"camera '{ref.key}' video file is missing", ep.index, {"key": ref.key}
                    )
                    continue
                try:
                    entry = cache.get(ref.path)
                    container = entry.container
                    if not container.streams.video:
                        raise ValueError("the file has no video stream")
                    packets = entry.packets()
                except Exception as e:  # noqa: BLE001 - any decode/open failure is a finding
                    yield Finding("VIDEO_UNREADABLE", E, f"camera '{ref.key}' video cannot be opened: {e}", ep.index)
                    continue
                cam_files = files.setdefault(ref.key, {})
                if ref.path not in cam_files:  # stream properties of every file, summarized per camera below
                    cam_files[ref.path] = _file_props(container, packets)
                props = cam_files[ref.path]
                found = list(_timestamp_findings(ep, ref, packets, fps, cfg))
                found += list(_episode_video(ep, ref, container, fps, moving, cfg, motion))
                yield from found
                # A broken feed (black, blank, wrong length, unreadable) has no stretches to compare.
                if any(f.severity >= W for f in found if f.code not in TIMESTAMP_CODES):
                    continue
                if pos is not None:
                    try:
                        series = _sample_changes(ep, ref, container, packets, fps, pos, cfg)
                    except Exception:  # noqa: BLE001 - a decode problem here is not a frozen stretch
                        series = None
                    if series is not None:
                        stills.setdefault(ref.key, []).append(series)
                # A camera running at another rate than the actions, or missing frames, cannot be lined
                # up frame by frame.
                if joints is not None and not _rate_mismatch(props, fps) and not any(f.severity >= W for f in found):
                    try:
                        curve = _sync_curve(ep, ref, container, fps, joints, cfg)
                    except Exception:  # noqa: BLE001 - a decode problem here is not a sync finding
                        curve = None
                    if curve is not None:
                        sync.setdefault(ref.key, {})[ep.index] = curve
    finally:
        cache.close()

    yield from _props_findings(ds, files, fps)

    models = {key: m for key, series in stills.items() if (m := _camera_model(series, cfg)) is not None}
    sampled = {key: {s.episode: s for s in series} for key, series in stills.items()}

    # A frozen feed still flickers a little from compression, so judge each episode
    # against how much this camera normally changes across the dataset.
    for key, rows in motion.items():
        typical = float(np.median([m for _, m in rows]))
        threshold = max(cfg.frozen_diff, 0.2 * typical)
        model = models.get(key)
        for ep_idx, m in rows:
            s = sampled.get(key, {}).get(ep_idx)
            if model is not None and s is not None and not model.shows_motion(s, fps, cfg):
                continue  # only joints this camera does not see moved (the other arm, say)
            if m < threshold:
                yield Finding(
                    "VIDEO_FROZEN",
                    W,
                    f"camera '{key}' image barely changes while the robot moves (change {m:.1f} vs typical {typical:.1f}): "
                    "the feed may be frozen",
                    ep_idx,
                    {"key": key, "change": round(m, 2), "typical": round(typical, 2)},
                )

    yield from _freeze_findings(stills, models, fps, cfg)
    yield from _sync_findings(sync, fps, cfg)


MAX_OPEN_VIDEOS = 8
TIMESTAMP_CODES = {"VIDEO_DROPPED_FRAMES", "VIDEO_DUPLICATE_FRAMES", "VIDEO_TIMESTAMPS_UNORDERED"}
NO_TS = np.iinfo(np.int64).min


@dataclass
class _Packets:
    """Timestamps of every packet in a file's video stream, read from the container without decoding."""

    pts: np.ndarray  # presentation timestamps (time-base units), in decode order; packets without one left out
    dts: np.ndarray  # decode timestamps, aligned with pts; NO_TS where a packet has none
    key: np.ndarray  # key frame flags, aligned with pts
    tb: float  # seconds per time-base unit
    period: float | None  # seconds between frames, from the regular spacing of the timestamps
    no_pts: int  # packets with no presentation timestamp

    def times(self) -> np.ndarray:
        return self.pts * self.tb


def _demux(container) -> _Packets:
    stream = container.streams.video[0]
    tb = float(stream.time_base) if stream.time_base else 0.0
    pts: list[int] = []
    dts: list[int] = []
    key: list[bool] = []
    no_pts = 0
    for p in container.demux(stream):
        if p.size == 0:  # end-of-stream marker, not a frame
            continue
        if p.pts is None:
            no_pts += 1
            continue
        pts.append(int(p.pts))
        dts.append(int(p.dts) if p.dts is not None else NO_TS)
        key.append(bool(p.is_keyframe))
    pk = _Packets(
        pts=np.asarray(pts, np.int64),
        dts=np.asarray(dts, np.int64),
        key=np.asarray(key, bool),
        tb=tb,
        period=None,
        no_pts=no_pts,
    )
    u = np.unique(pk.pts)
    if len(u) >= 2 and tb > 0:
        d = np.diff(u).astype(np.float64)
        med = float(np.median(d))
        regular = d[(d >= 0.5 * med) & (d <= 1.5 * med)]  # leave out gaps, so dropped frames don't skew the rate
        pk.period = float(regular.mean()) * tb
    return pk


class _Open:
    """An open container and, read once on first use, the timestamps of all its packets."""

    def __init__(self, container) -> None:
        self.container = container
        self._packets: _Packets | None = None

    def packets(self) -> _Packets:
        if self._packets is None:
            self._packets = _demux(self.container)
        return self._packets


class _ContainerCache:
    """Small LRU of open PyAV containers; evicted containers are closed immediately."""

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self._open: OrderedDict[Path, _Open] = OrderedDict()

    def get(self, path: Path) -> _Open:
        c = self._open.get(path)
        if c is not None:
            self._open.move_to_end(path)
            return c
        c = _Open(_open(path))
        self._open[path] = c
        while len(self._open) > self.size:
            _, old = self._open.popitem(last=False)
            old.container.close()
        return c

    def close(self) -> None:
        while self._open:
            _, c = self._open.popitem(last=False)
            try:
                c.container.close()
            except Exception:  # noqa: BLE001 - closing must never mask a result
                pass


def _open(path: Path):
    import av

    return av.open(str(path))


# -- stream properties of every file ----------------------------------------------------------


@dataclass
class _FileProps:
    rate: float | None  # frames per second
    width: int
    height: int


def _file_props(container, packets: _Packets) -> _FileProps:
    stream = container.streams.video[0]
    if packets.period:
        rate: float | None = 1.0 / packets.period
    else:
        rate = float(stream.average_rate) if stream.average_rate else None
    cc = stream.codec_context
    return _FileProps(rate=rate, width=int(cc.width), height=int(cc.height))


def _rate_mismatch(p: _FileProps, fps: float) -> bool:
    return p.rate is not None and abs(p.rate - fps) > 0.5


def _of(n: int, total: int) -> str:
    return "" if n == total else f" ({n} of {total} files)"


def _props_findings(ds: Dataset, files: dict[str, dict[Path, _FileProps]], fps: float) -> Iterator[Finding]:
    """One finding per camera and problem, counting the files it affects."""
    for key, by_file in files.items():
        props = list(by_file.values())
        total = len(props)
        off = [p for p in props if _rate_mismatch(p, fps)]
        if off:
            rates = sorted({round(p.rate, 2) for p in off if p.rate is not None})
            rate = f"{rates[0]:g}" if len(rates) == 1 else f"{rates[0]:g} to {rates[-1]:g}"
            yield Finding(
                "VIDEO_FPS_MISMATCH",
                W,
                f"camera '{key}' runs at {rate} fps but the dataset declares {fps:g}{_of(len(off), total)}",
                None,
                {"key": key, "files": len(off), "files_checked": total},
            )
        shape = ds.features.get(key, {}).get("shape")
        sizes = [(p.width, p.height) for p in props]
        if isinstance(shape, list) and len(shape) >= 2 and all(isinstance(x, int) for x in shape[:2]):
            want = (shape[1], shape[0])
            bad = [s for s in sizes if s != want]
            if bad:
                w, h = bad[0]
                yield Finding(
                    "VIDEO_SHAPE_MISMATCH",
                    W,
                    f"camera '{key}' is {w}x{h} but declared {want[0]}x{want[1]}{_of(len(bad), total)}",
                    None,
                    {"key": key, "files": len(bad), "files_checked": total},
                )
        elif len(set(sizes)) > 1:
            common = max(set(sizes), key=sizes.count)
            bad = [s for s in sizes if s != common]
            w, h = bad[0]
            yield Finding(
                "VIDEO_SHAPE_MISMATCH",
                W,
                f"camera '{key}' is {w}x{h} in {len(bad)} of {total} files but {common[0]}x{common[1]} in the rest",
                None,
                {"key": key, "files": len(bad), "files_checked": total},
            )


# -- frame timestamps ---------------------------------------------------------------------------


@dataclass
class _Timing:
    missing: int  # frames missing from inside the segment
    gaps: int
    longest_gap: float  # seconds
    first_gap_at: float  # seconds into the segment
    duplicates: int  # frames sharing a timestamp with another
    backwards: int  # decode timestamps that do not increase


def _segment_mask(pk: _Packets, start: float, end: float | None, fps: float) -> np.ndarray:
    t = pk.times()
    half = 0.5 / fps
    mask = t >= start - half
    if end is not None:
        mask &= t < end - half
    return mask


def segment_timing(pk: _Packets, start: float, end: float | None, fps: float, gap_factor: float) -> _Timing:
    """Missing, duplicated and out-of-order frames among the packets of one episode's segment."""
    mask = _segment_mask(pk, start, end, fps)
    pts = np.sort(pk.pts[mask])
    period = pk.period or 1.0 / fps
    d = np.diff(pts).astype(np.float64) * pk.tb
    gap = d > gap_factor * period
    missing = int(np.sum(np.maximum(1, np.rint(d[gap] / period) - 1))) if gap.any() else 0
    first_gap = float(pts[np.argmax(gap)] * pk.tb - start) if gap.any() else 0.0
    dts = pk.dts[mask]
    dts = dts[dts != NO_TS]
    return _Timing(
        missing=missing,
        gaps=int(gap.sum()),
        longest_gap=float(d[gap].max()) if gap.any() else 0.0,
        first_gap_at=first_gap,
        duplicates=int(np.sum(np.diff(pts) == 0)),
        backwards=int(np.sum(np.diff(dts) <= 0)),
    )


def _timestamp_findings(ep: Episode, ref: VideoRef, pk: _Packets, fps: float, cfg: Config) -> Iterator[Finding]:
    if pk.tb <= 0 or not len(pk.pts):
        return
    t = segment_timing(pk, ref.from_ts, ref.to_ts, fps, cfg.gap_factor)
    if t.missing:
        yield Finding(
            "VIDEO_DROPPED_FRAMES",
            E,
            f"camera '{ref.key}' is missing {t.missing} frame{'s' if t.missing != 1 else ''} mid-episode "
            f"({t.gaps} gap{'s' if t.gaps != 1 else ''}, the first {t.first_gap_at:.2f}s in): loaders that fetch "
            "frames by timestamp will fail or return the wrong frame there",
            ep.index,
            {
                "key": ref.key,
                "missing_frames": t.missing,
                "gaps": t.gaps,
                "longest_gap_s": round(t.longest_gap, 4),
                "first_gap_at_s": round(t.first_gap_at, 3),
            },
        )
    if t.duplicates:
        yield Finding(
            "VIDEO_DUPLICATE_FRAMES",
            W,
            f"camera '{ref.key}' has {t.duplicates} frame{'s' if t.duplicates != 1 else ''} with the same timestamp "
            "as another: players and loaders keep only one of them",
            ep.index,
            {"key": ref.key, "duplicates": t.duplicates},
        )
    if t.backwards:
        yield Finding(
            "VIDEO_TIMESTAMPS_UNORDERED",
            W,
            f"camera '{ref.key}' decode timestamps go backwards {t.backwards} time{'s' if t.backwards != 1 else ''}: "
            "the file is badly muxed and decoders may drop or reorder frames",
            ep.index,
            {"key": ref.key, "backwards": t.backwards},
        )


# -- frozen stretches ---------------------------------------------------------------------------


@dataclass
class _TravelDims:
    dims: np.ndarray  # state dims whose movement is compared with the image change
    rng: np.ndarray  # their robust ranges, to put joints in different units on one scale
    width: int  # full state width


def _travel_dims(ds: Dataset, cfg: Config, joints: PositionJoints | None) -> _TravelDims | None:
    """The measured positions of the joints that follow their commands. Measured rather than
    commanded, so an arm pushing against an obstacle or a joint limit (commanded to move, not
    moving) is not mistaken for motion the camera should show. None without such joints: then
    frozen stretches are not looked for."""
    if joints is None:
        return None
    st = dataset_stats(ds, "observation.state", cfg)
    if st is None:
        return None
    rng = st.rng[joints.state_dims]
    keep = rng > 1e-9
    if not keep.any():
        return None
    return _TravelDims(joints.state_dims[keep], rng[keep], joints.s_dim)


def _positions(ep: Episode, td: _TravelDims) -> np.ndarray | None:
    """(frames, joints) measured joint positions, in fractions of each joint's range."""
    st = ep.frames.get("observation.state")
    if st is None or len(st) < 2:
        return None
    st = _as2d(st).astype(np.float64)
    if st.shape[1] != td.width or not np.all(np.isfinite(st[:, td.dims])):
        return None
    return st[:, td.dims] / td.rng


@dataclass
class _Series:
    episode: int
    idx: np.ndarray  # frame index of each sampled image
    change: np.ndarray  # mean pixel change between consecutive samples (0-255)
    travel: np.ndarray  # (len(change), joints) how far each joint moved between consecutive samples


def _sample_changes(
    ep: Episode, ref: VideoRef, container, pk: _Packets, fps: float, pos: np.ndarray, cfg: Config
) -> _Series | None:
    """Image change between frames sampled about cfg.freeze_sample_hz times per second across the
    episode, and how far the robot's joints moved in each interval. Key frames are used when
    the encoding has them often enough (LeRobot encodes every second frame as one), and only those
    are decoded; otherwise the segment is decoded in full and sampled."""
    import av

    n = min(ep.length, len(pos))
    step = max(1, int(round(fps / cfg.freeze_sample_hz)))
    if n < 3 * step or pk.tb <= 0:
        return None
    mask = _segment_mask(pk, ref.from_ts, ref.to_ts, fps)
    pts, key = pk.pts[mask], pk.key[mask]
    idx = np.rint((pts * pk.tb - ref.from_ts) * fps).astype(np.int64)
    inside = (idx >= 0) & (idx < n)
    pts, key, idx = pts[inside], key[inside], idx[inside]
    if len(pts) < 3:
        return None
    targets = np.arange(0, n, step)
    k_idx = np.unique(idx[key])
    dense = len(k_idx) >= 2 and float(np.median(np.diff(k_idx))) <= step
    pool_idx, pool_pts = (idx[key], pts[key]) if dense else (idx, pts)
    order = np.argsort(pool_idx, kind="stable")
    pool_idx, pool_pts = pool_idx[order], pool_pts[order]
    at = np.clip(np.searchsorted(pool_idx, targets), 1, len(pool_idx) - 1) if len(pool_idx) > 1 else np.zeros(1, int)
    if len(pool_idx) > 1:
        left = pool_idx[at - 1]
        at = np.where(np.abs(targets - left) <= np.abs(pool_idx[at] - targets), at - 1, at)
    near = np.abs(pool_idx[at] - targets) <= step / 2
    wanted = {int(p): int(i) for p, i in zip(pool_pts[at[near]], pool_idx[at[near]], strict=True)}
    if len(wanted) < 3:
        return None

    stream = container.streams.video[0]
    cc = av.CodecContext.create(stream.codec_context.name, "r")
    if stream.codec_context.extradata:
        cc.extradata = stream.codec_context.extradata
    container.seek(int(ref.from_ts / stream.time_base), stream=stream, backward=True, any_frame=False)
    stop = (ref.to_ts if ref.to_ts is not None else float("inf")) + 1.0
    images: dict[int, np.ndarray] = {}

    def take(frames) -> None:
        for fr in frames:
            if fr.pts is not None and int(fr.pts) in wanted:
                images[wanted[int(fr.pts)]] = (
                    fr.reformat(width=64, height=48, format="gray").to_ndarray().astype(np.float32)
                )

    for p in container.demux(stream):
        if p.size == 0:
            continue
        if p.pts is not None and p.pts * pk.tb > stop:
            break
        if dense and (p.pts is None or int(p.pts) not in wanted):
            continue  # key frames decode on their own: the rest are skipped
        take(cc.decode(p))
    take(cc.decode(None))
    got = np.array(sorted(images))
    if len(got) < 3:
        return None
    change = np.array([np.mean(np.abs(images[b] - images[a])) for a, b in zip(got[:-1], got[1:], strict=True)])
    # Net displacement, not path length: sensor noise does not add up over the interval.
    return _Series(ep.index, got, change, np.abs(pos[got[1:]] - pos[got[:-1]]))


STILL_TRAVEL = 0.002  # joint movement (fraction of range) below which the robot counts as still


@dataclass
class _CameraModel:
    """How much a camera's image changes for a given movement of each joint."""

    weights: np.ndarray  # image change per unit of each joint's movement (0 for joints it does not see)
    frozen_below: float  # change at or below this: the image is repeating, not just a still scene
    visible: float  # movement-driven change at or above this is clearly visible
    flickers: bool  # the live image always changes a little, even when nothing moves (sensor noise)

    def expected(self, s: _Series) -> np.ndarray:
        return s.travel @ self.weights

    def shows_motion(self, s: _Series, fps: float, cfg: Config) -> bool:
        """Whether, in this episode, the joints the camera sees move visibly for at least
        cfg.freeze_min_seconds in total."""
        seconds = np.diff(s.idx) / fps
        return float(seconds[self.expected(s) >= self.visible].sum()) >= cfg.freeze_min_seconds


def _camera_model(series: list[_Series], cfg: Config) -> _CameraModel | None:
    """Per camera, image change between samples predicted from the joints' measured movement by a
    least-squares fit with non-negative weights, so joints the camera does not see (the other arm
    of a bimanual robot, say) get no weight. None when there is too little to fit, or the camera's
    change is not well explained by the robot's movement."""
    if len(series) < cfg.freeze_min_episodes:
        return None
    X = np.vstack([s.travel for s in series])
    y = np.concatenate([s.change for s in series])
    if len(y) < 30:
        return None
    D = float(np.quantile(y, 0.9))
    if D < 0.25:
        return None  # this camera hardly ever changes: nothing to compare against
    still = X.max(axis=1) <= STILL_TRAVEL
    noise = float(np.median(y[still])) if still.sum() >= 10 else 0.0
    # Below the flicker of a live camera looking at a still scene: the image itself is repeating.
    frozen_below = min(0.1 * D, max(0.5 * noise, 0.02 * D))
    visible = max(5.0 * frozen_below, noise)
    # A camera whose still scenes all flicker (none drops to the repeating level) shows a frozen
    # image by the missing flicker alone; one whose still scenes can be perfectly static (rendered
    # simulation, or an encoder that flattens static scenes) needs clear motion to compare against.
    flickers = bool(still.sum() >= 10 and noise > 0 and np.quantile(y[still], 0.02) > frozen_below)
    coef = nonneg_coef(X, y)
    fit = ~((y <= frozen_below) & (X @ coef[:-1] >= visible))  # refit without the intervals under suspicion
    if fit.sum() < 30:
        return None
    coef = nonneg_coef(X[fit], y[fit])
    resid = y[fit] - (X[fit] @ coef[:-1] + coef[-1])
    var = float(np.var(y[fit]))
    if var <= 1e-12 or 1.0 - float(np.mean(resid**2)) / var < cfg.freeze_min_r2:
        return None
    return _CameraModel(weights=coef[:-1], frozen_below=frozen_below, visible=visible, flickers=flickers)


def _freeze_findings(
    stills: dict[str, list[_Series]], models: dict[str, _CameraModel], fps: float, cfg: Config
) -> Iterator[Finding]:
    """Stretches of at least cfg.freeze_min_seconds where a camera's image stops changing (less
    than it changes when nothing moves at all) while the joints it sees keep moving visibly, and
    the same camera is live elsewhere in the episode."""
    for key, series in stills.items():
        model = models.get(key)
        if model is None:
            continue
        for s in series:
            expected = model.expected(s)
            run = _longest_frozen_run(s, expected, model, fps, cfg)
            if run is None:
                continue
            a, b = run
            t0, t1 = s.idx[a] / fps, s.idx[b + 1] / fps
            yield Finding(
                "VIDEO_FREEZE",
                W,
                f"camera '{key}' image stops changing for {t1 - t0:.1f}s ({t0:.1f}s to {t1:.1f}s into the episode) "
                "while the robot keeps moving: the feed may have frozen there",
                s.episode,
                {
                    "key": key,
                    "start_s": round(t0, 2),
                    "end_s": round(t1, 2),
                    "seconds": round(t1 - t0, 2),
                    "change": round(float(s.change[a : b + 1].mean()), 3),
                    "expected_change": round(float(expected[a : b + 1].mean()), 2),
                },
            )


def _longest_frozen_run(
    s: _Series, expected: np.ndarray, model: _CameraModel, fps: float, cfg: Config
) -> tuple[int, int] | None:
    """(first, last) interval of the longest qualifying frozen stretch, or None."""
    visible = model.visible
    frozen = s.change <= model.frozen_below
    best = None
    i = 0
    n = len(frozen)
    while i < n:
        if not frozen[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and frozen[j + 1]:
            j += 1
        seconds = (s.idx[j + 1] - s.idx[i]) / fps
        e = expected[i : j + 1]
        outside = np.concatenate([s.change[:i], s.change[j + 1 :]])
        if model.flickers:
            # No flicker at all, while the joints it sees move enough in total to show.
            moved = float(e.sum()) >= visible
        else:
            # A static image is normal here, so the robot must move visibly through most of the stretch.
            moved = float(e.mean()) >= visible and float(np.mean(e >= visible)) >= 0.5
        if (
            seconds >= cfg.freeze_min_seconds
            and moved
            and bool(np.any(outside >= visible))  # live elsewhere: a whole frozen episode is VIDEO_FROZEN
            and (best is None or seconds > best[0])
        ):
            best = (seconds, i, j)
        i = j + 1
    return None if best is None else (best[1], best[2])


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
