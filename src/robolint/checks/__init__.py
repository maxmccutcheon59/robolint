"""Check registry and shared configuration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..dataset import Dataset
from ..findings import Finding


@dataclass
class Config:
    # Timing
    gap_factor: float = 1.5  # dt above this many frame periods counts as dropped frames
    jitter_ratio: float = 0.10  # std(dt) / period above this is jittery
    drift_frames: float = 2.0  # end-to-end drift tolerance, in frame periods
    # Signals (fractions of each dimension's dataset-wide range)
    move_tol: float = 0.005  # change smaller than this counts as "not moving"
    dead_tol: float = 0.01  # every dimension moves less than this -> dead episode
    gripper_tol: float = 0.05  # gripper travel smaller than this -> never actuated
    spike_quantile: float = 0.995  # robust scale for per-step action changes
    spike_multiplier: float = 6.0
    spike_min_range: float = 0.15  # a spike must also be at least this fraction of the range
    idle_warn: float = 0.30  # idle lead-in + tail above this fraction of the episode
    idle_info: float = 0.15
    limit_seconds: float = 2.0  # pinned at a joint's extreme for this long
    # Episodes
    min_seconds: float = 1.0
    short_ratio: float = 0.25  # shorter than this fraction of the median length
    long_ratio: float = 3.0
    duplicate_tol: float = 0.002  # mean abs difference (fraction of range) for duplicates
    # Video
    video_samples: int = 8
    black_level: float = 8.0
    white_level: float = 247.0
    blank_std: float = 2.0
    frozen_diff: float = 1.0  # minimum mean pixel change (0-255) before a feed counts as live
    frozen_min_seconds: float = 2.0  # shorter episodes are too brief to judge


CheckFn = Callable[[Dataset, Config], Iterable[Finding]]


def all_checks(include_video: bool) -> list[CheckFn]:
    from . import episodes, meta, signals, timing

    checks: list[CheckFn] = [meta.check, timing.check, signals.check, episodes.check]
    if include_video:
        from . import video

        checks.append(video.check)
    return checks
