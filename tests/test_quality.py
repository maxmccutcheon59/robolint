"""The learned quality model: loads, scores every episode, and ranks obvious faults above clean demos."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sklearn")
from robolint.quality import score_dataset  # noqa: E402


def _demo(rng, n=300):
    t = np.linspace(0, 1, n)[:, None]
    base = np.array([10, -20, 30, 5, 0, 20.0])
    goal = np.array([40, 10, -10, 25, 15, 60.0]) + rng.normal(0, 2, 6)
    a = base + (goal - base) * (3 * t**2 - 2 * t**3) + rng.normal(0, 0.05, (n, 6))
    s = np.vstack([a[:1], a[:-1]])  # follower lags the leader by a frame
    return a.astype(np.float32), s.astype(np.float32)


def _ds(eps):
    return SimpleNamespace(
        episodes=[
            SimpleNamespace(index=i, frames={"action": a, "observation.state": s}) for i, (a, s) in enumerate(eps)
        ]
    )


def test_scores_every_episode_and_flags_dead_and_noisy():
    rng = np.random.default_rng(0)
    eps = [_demo(rng, int(rng.integers(260, 340))) for _ in range(30)]
    a, s = eps[3]
    eps[3] = (np.repeat(a[:1], len(a), 0), np.repeat(s[:1], len(s), 0))  # dead
    a, s = eps[7]
    eps[7] = (a + rng.normal(0, 8, a.shape).astype(np.float32), s)  # noisy teleop
    q = {x.episode: x for x in score_dataset(_ds(eps))}
    assert sorted(q) == list(range(30))
    clean = np.median([q[i].p_bad for i in range(30) if i not in (3, 7)])
    assert q[3].flagged and q[3].p_bad > clean
    assert q[7].p_bad > clean
    assert all(0.0 <= x.p_bad <= 1.0 for x in q.values())


def test_too_few_episodes_returns_nothing():
    rng = np.random.default_rng(1)
    assert score_dataset(_ds([_demo(rng) for _ in range(3)])) == []
