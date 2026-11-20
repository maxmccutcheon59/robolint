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
    grasp_hold_seconds: float = 0.3  # gripper command held this long counts as a grasp attempt
    grasp_object_min: float = 0.15  # typical grasp stops this far short of fully closed (fraction of range)
    grasp_air_tol: float = 0.05  # within this of fully closed = closed on nothing
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
    # Action-to-state lag
    lag_tracking_corr: float = 0.8  # a joint's state must follow its command this closely to be used
    lag_max_seconds: float = 1.0  # lags searched in each direction
    lag_smooth_seconds: float = 0.1  # velocity smoothing window
    lag_min_seconds: float = 2.0  # shorter episodes are not estimated
    lag_min_travel: float = 0.10  # a joint must travel this fraction of its range to count as moving
    lag_min_corr: float = 0.5  # estimates with a weaker best match are discarded
    lag_flag_corr: float = 0.7  # an outlier is only reported when the match is at least this strong
    lag_min_episodes: int = 5  # estimated episodes needed before anything is reported
    lag_tol_seconds: float = 0.15  # an outlier must be this far from the typical lag...
    lag_tol_frames: float = 3.0  # ...and at least this many frames...
    lag_mad_factor: float = 5.0  # ...and this many robust standard deviations...
    lag_margin: float = 0.1  # ...and its own shift must fit this much better than the typical one
    # Action/video sync (with --videos)
    sync_max_seconds: float = 30.0  # decode at most this much of each episode per camera
    sync_max_lag_seconds: float = 1.0
    sync_smooth_seconds: float = 0.1
    sync_min_corr: float = 0.5  # estimates with a weaker best match are discarded
    sync_flag_corr: float = 0.7  # an offset is only reported when the match is at least this strong
    sync_tol_seconds: float = 0.25  # an offset must be this far from the camera's typical delay...
    sync_tol_frames: float = 3.0
    sync_mad_factor: float = 5.0
    sync_margin: float = 0.1


CheckFn = Callable[[Dataset, Config], Iterable[Finding]]


def all_checks(include_video: bool) -> list[CheckFn]:
    from . import episodes, lag, meta, signals, timing

    checks: list[CheckFn] = [meta.check, timing.check, signals.check, lag.check, episodes.check]
    if include_video:
        from . import video

        checks.append(video.check)
    return checks
