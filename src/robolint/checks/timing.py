"""Per-episode frame index and timestamp integrity."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..dataset import Dataset
from ..findings import Finding, Severity
from . import Config

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO


def check(ds: Dataset, cfg: Config) -> Iterator[Finding]:
    fps = ds.fps
    for ep in ds.episodes:
        n = ep.length
        if n == 0:
            yield Finding("EMPTY_EPISODE", E, "episode has no frames", ep.index)
            continue

        fi = ep.frames.get("frame_index")
        if fi is not None:
            fi = fi.astype(np.int64)
            if fi[0] != 0:
                yield Finding("FRAME_INDEX_OFFSET", W, f"frame_index starts at {fi[0]}, not 0", ep.index)
            d = np.diff(fi)
            dup = int(np.sum(d == 0))
            if dup:
                yield Finding(
                    "FRAME_INDEX_DUPLICATE", E, f"{dup} duplicated frame_index values", ep.index, {"count": dup}
                )
            gaps = d[d > 1]
            if len(gaps):
                missing = int(np.sum(gaps - 1))
                yield Finding(
                    "FRAME_INDEX_GAP",
                    E,
                    f"{missing} frames missing from frame_index ({len(gaps)} gaps)",
                    ep.index,
                    {"missing_frames": missing, "gaps": len(gaps)},
                )

        ts = ep.frames.get("timestamp")
        if ts is None or fps is None or n < 2:
            continue
        ts = ts.astype(np.float64)
        period = 1.0 / fps
        if not np.all(np.isfinite(ts)):
            yield Finding("TIMESTAMP_NONFINITE", E, "timestamps contain NaN or infinite values", ep.index)
            continue
        if abs(ts[0]) > period / 2:
            yield Finding("TIMESTAMP_OFFSET", I, f"first timestamp is {ts[0]:.3f}s, not 0", ep.index)

        dt = np.diff(ts)
        back = int(np.sum(dt <= 0))
        if back:
            yield Finding(
                "TIMESTAMP_NONMONOTONIC",
                E,
                f"timestamps go backwards or repeat {back} times",
                ep.index,
                {"count": back},
            )

        big = dt > cfg.gap_factor * period
        has_gaps = bool(np.any(big))
        if has_gaps:
            dropped = int(np.sum(np.round(dt[big] / period) - 1))
            sev = E if dropped > max(1, 0.01 * n) else W
            yield Finding(
                "DROPPED_FRAMES",
                sev,
                f"~{dropped} frames dropped across {int(big.sum())} timestamp gaps (largest {dt.max() * 1000:.0f} ms at {fps:g} fps)",
                ep.index,
                {"dropped": dropped, "gaps": int(big.sum()), "max_gap_s": float(dt.max())},
            )

        normal = dt[(dt > 0) & ~big]
        if len(normal) >= 10:
            jitter = float(np.std(normal) / period)
            if jitter > cfg.jitter_ratio:
                yield Finding(
                    "TIMESTAMP_JITTER",
                    W,
                    f"frame timing jitter is {jitter * 100:.0f}% of the frame period",
                    ep.index,
                    {"jitter_ratio": jitter},
                )

        if not has_gaps and not back:
            expected = (n - 1) * period
            actual = ts[-1] - ts[0]
            drift = actual - expected
            if abs(drift) > cfg.drift_frames * period:
                yield Finding(
                    "TIMESTAMP_DRIFT",
                    W,
                    f"timestamps drift {drift * 1000:+.0f} ms from the declared {fps:g} fps over the episode",
                    ep.index,
                    {"drift_s": float(drift)},
                )
