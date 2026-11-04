"""Episode-level checks: lengths, tasks, and duplicate demonstrations."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..dataset import Dataset
from ..findings import Finding, Severity
from . import Config
from .signals import dataset_stats

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO
RESAMPLE = 64


def check(ds: Dataset, cfg: Config) -> Iterator[Finding]:
    if not ds.episodes:
        yield Finding("NO_EPISODES", E, "dataset contains no episodes")
        return
    fps = ds.fps or 30.0
    lengths = np.array([ep.length for ep in ds.episodes])
    median = float(np.median(lengths))

    for ep in ds.episodes:
        n = ep.length
        if ep.length_meta is not None and ep.length_meta != n:
            yield Finding(
                "EPISODE_LENGTH_MISMATCH",
                E,
                f"metadata says {ep.length_meta} frames, data has {n}",
                ep.index,
                {"metadata": ep.length_meta, "data": n},
            )
        if 0 < n < cfg.min_seconds * fps:
            yield Finding("EPISODE_TOO_SHORT", E, f"only {n} frames ({n / fps:.2f}s): likely a false start", ep.index)
        elif len(lengths) >= 5 and n < cfg.short_ratio * median:
            yield Finding(
                "EPISODE_SHORT",
                W,
                f"{n} frames, under {cfg.short_ratio:.0%} of the median ({median:.0f}): possibly truncated",
                ep.index,
            )
        if len(lengths) >= 5 and n > cfg.long_ratio * median and n - median > 5 * fps:
            yield Finding(
                "EPISODE_LONG",
                I,
                f"{n} frames, {n / median:.1f}x the median: recording may not have been stopped",
                ep.index,
            )

        ti = ep.frames.get("task_index")
        if ti is not None and ds.tasks:
            unknown = sorted({int(x) for x in np.unique(ti)} - set(ds.tasks))
            if unknown:
                yield Finding(
                    "TASK_INDEX_UNKNOWN", E, f"task_index values {unknown[:5]} have no task description", ep.index
                )
        if ep.tasks and not any(t.strip() for t in ep.tasks):
            yield Finding("EPISODE_TASK_EMPTY", W, "episode's task description is empty", ep.index)

    yield from _duplicates(ds, cfg)


def _signature(a: np.ndarray, lo: np.ndarray, rng: np.ndarray) -> np.ndarray:
    a = a.reshape(-1, 1) if a.ndim == 1 else a
    t_old = np.linspace(0.0, 1.0, len(a))
    t_new = np.linspace(0.0, 1.0, RESAMPLE)
    res = np.stack([np.interp(t_new, t_old, a[:, j]) for j in range(a.shape[1])], axis=1)
    return ((res - lo) / rng).ravel()


def _duplicates(ds: Dataset, cfg: Config) -> Iterator[Finding]:
    st = dataset_stats(ds, "action", cfg)
    if st is None:
        return
    act = st.active
    if not act.any():
        return
    rng = np.where(act, st.rng, 1.0)
    eps, sigs, lens = [], [], []
    for ep in ds.episodes:
        a = ep.frames.get("action")
        if a is None or ep.length < 4:
            continue
        a2 = a.reshape(-1, 1) if a.ndim == 1 else a
        if a2.shape[1] != len(rng) or not np.all(np.isfinite(a2)):
            continue
        if np.all((a2.max(0) - a2.min(0))[act] < cfg.dead_tol * st.rng[act]):
            continue  # dead episodes are reported elsewhere and would all look alike
        eps.append(ep.index)
        sigs.append(_signature(a2[:, act].astype(np.float64), st.lo[act], rng[act]))
        lens.append(ep.length)
    if len(sigs) < 2:
        return
    S = np.stack(sigs)
    L = np.array(lens, dtype=float)
    # mean|Si - Sj| >= |mean(Si) - mean(Sj)|, so only pairs whose signature means are within
    # the tolerance can be duplicates. Sorting by mean makes this exact pruning, which keeps
    # 100k-episode datasets fast.
    M = S.mean(axis=1)
    order = np.argsort(M, kind="stable")
    M_sorted = M[order]
    seen: set[int] = set()
    for i in range(len(S)):
        if eps[i] in seen:
            continue
        lo_pos = np.searchsorted(M_sorted, M[i] - cfg.duplicate_tol, side="left")
        hi_pos = np.searchsorted(M_sorted, M[i] + cfg.duplicate_tol, side="right")
        cand = order[lo_pos:hi_pos]
        cand = np.sort(cand[cand > i])
        cand = cand[np.abs(L[cand] - L[i]) <= 0.05 * L[i] + 1]
        if not len(cand):
            continue
        dist = np.mean(np.abs(S[cand] - S[i]), axis=1)
        for j in cand[dist < cfg.duplicate_tol]:
            seen.add(eps[j])
            yield Finding(
                "DUPLICATE_EPISODE",
                W,
                f"actions are nearly identical to episode {eps[i]}: likely a duplicated or replayed recording",
                eps[j],
                {"duplicate_of": eps[i]},
            )
