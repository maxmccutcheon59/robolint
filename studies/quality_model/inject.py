"""Realistic faults injected into clean real episodes, to give the model labelled examples.

Each fault mirrors one seen in community LeRobot datasets or reported by teams collecting teleop data.
"""

from __future__ import annotations

import numpy as np

FAULTS = [
    "dead",
    "frozen_state",
    "glitch",
    "idle_pad",
    "false_start",
    "aborted",
    "stacked",
    "stutter",
    "noisy",
    "hesitation",
    "anomalous",
]
CLASSES = ["clean"] + FAULTS


def _copy(e: dict) -> dict:
    return {"action": e["action"].copy(), "state": None if e.get("state") is None else e["state"].copy()}


def _hold(x: np.ndarray, n: int, at_end: bool) -> np.ndarray:
    pad = np.repeat(x[-1:] if at_end else x[:1], n, axis=0)
    return np.concatenate([x, pad]) if at_end else np.concatenate([pad, x])


def inject(e: dict, fault: str, rng: np.random.Generator, scale: np.ndarray, other: dict | None = None) -> dict:
    o = _copy(e)
    a, s = o["action"], o["state"]
    T = len(a)
    if fault == "dead":
        a[:] = a[0] + rng.normal(0, 1e-4, a.shape) * scale
        if s is not None:
            s[:] = s[0]
    elif fault == "frozen_state" and s is not None:
        k = int(rng.uniform(0, 0.5) * T)
        s[k:] = s[k]
    elif fault == "frozen_state":  # no state column: the follower ignoring commands looks like dead actions
        k = int(rng.uniform(0, 0.5) * T)
        a[k:] = a[k]
    elif fault == "glitch":
        for _ in range(rng.integers(1, 4)):
            t = rng.integers(1, T - 1)
            w = rng.integers(1, 4)
            dims = rng.choice(a.shape[1], size=rng.integers(1, max(2, a.shape[1] // 2)), replace=False)
            a[t : t + w, dims] += rng.choice([-1, 1], len(dims)) * rng.uniform(4, 15) * scale[dims]
    elif fault == "idle_pad":
        n = int(rng.uniform(0.4, 1.0) * T)
        end = rng.random() < 0.5
        a = _hold(a, n, end)
        s = None if s is None else _hold(s, n, end)
    elif fault == "false_start":
        k = max(4, int(rng.uniform(0.08, 0.3) * T))
        a, s = a[:k], (None if s is None else s[:k])
    elif fault == "aborted":
        k = max(4, int(rng.uniform(0.3, 0.65) * T))
        n = T - k
        a = _hold(a[:k], n, True)
        s = None if s is None else _hold(s[:k], n, True)
    elif fault == "stacked" and other is not None:
        a = np.concatenate([a, other["action"]])
        if s is not None and other.get("state") is not None and other["state"].shape[1] == s.shape[1]:
            s = np.concatenate([s, other["state"]])
        else:
            s = None
    elif fault == "stutter":
        reps = rng.random(T) < rng.uniform(0.25, 0.5)
        idx = np.repeat(np.arange(T), 1 + reps * rng.integers(2, 5))
        a = a[idx]
        s = None if s is None else s[idx]
    elif fault == "noisy":
        a = a + rng.normal(0, 1, a.shape) * scale * rng.uniform(0.15, 0.5)
    elif fault == "hesitation":
        k = int(rng.uniform(0.3, 0.7) * T)
        n = int(rng.uniform(0.3, 0.6) * T)
        a = np.concatenate([a[:k], np.repeat(a[k : k + 1], n, axis=0), a[k:]])
        s = None if s is None else np.concatenate([s[:k], np.repeat(s[k : k + 1], n, axis=0), s[k:]])
    elif fault == "anomalous":
        # a demo of a different motion: played backwards, or with the main joints swapped
        if rng.random() < 0.5:
            a, s = a[::-1].copy(), (None if s is None else s[::-1].copy())
        else:
            p = rng.permutation(a.shape[1])
            a = a[:, p]
            if s is not None and s.shape[1] == a.shape[1]:
                s = s[:, p]
    o["action"], o["state"] = a, s
    return o
