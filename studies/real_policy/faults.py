"""Recording faults injected into a copy of a clean ALOHA sim dataset.

The same five problems as studies/filtering_bc, made concrete for a 14-joint bimanual arm with a
camera. Every fault keeps the episode self-consistent where the real fault would: a robot that
was never driven shows a still image, idle padding repeats real frames, and a failed grasp is
re-simulated so the camera and joint readings show the gripper closing on air.

An episode is a dict with "state" (T x 14), "action" (T x 14) and "images" (T x H x W x 3 uint8).
"""

from __future__ import annotations

import numpy as np

# Share of the dataset's episodes given each fault. "heavy" matches the heavy mix of the toy study
# (36% of episodes affected); "light" is about 12%. With 50 source episodes: heavy = 3 dead,
# 2 frozen, 4 spikes, 4 idle, 5 failed; light = 1, 1, 2, 1, 1.
MIXES: dict[str, dict[str, float]] = {
    "heavy": {"dead": 0.06, "state_frozen": 0.04, "spikes": 0.08, "idle": 0.08, "failed": 0.10},
    "light": {"dead": 0.02, "state_frozen": 0.02, "spikes": 0.04, "idle": 0.02, "failed": 0.02},
}
FAULTS = ("dead", "state_frozen", "spikes", "idle", "failed")
GRIPPER_DIMS = (6, 13)  # left_gripper, right_gripper in the ALOHA joint order


def assign(n_episodes: int, seed: int) -> dict[str, dict[int, str]]:
    """Which episode gets which fault, per mix. The light mix's faulty episodes are a subset of the
    heavy mix's (same episodes, same faults), so the two mixes differ only in how much is broken."""
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(n_episodes)]
    heavy: dict[str, list[int]] = {}
    k = 0
    for f in FAULTS:
        n = max(1, round(MIXES["heavy"][f] * n_episodes))
        heavy[f] = order[k : k + n]
        k += n
    out: dict[str, dict[int, str]] = {}
    for mix, shares in MIXES.items():
        labels: dict[int, str] = {}
        for f in FAULTS:
            n = max(1, round(shares[f] * n_episodes)) if shares[f] > 0 else 0
            for ep in heavy[f][:n]:
                labels[ep] = f
        out[mix] = labels
    return out


def dead(ep: dict, rng: np.random.Generator) -> dict:
    """The robot was never driven: the recording runs, nothing moves, the camera sees a still scene."""
    n = len(ep["action"])
    return {
        "state": np.repeat(ep["state"][:1], n, axis=0),
        "action": np.repeat(ep["action"][:1], n, axis=0),
        "images": np.repeat(ep["images"][:1], n, axis=0),
    }


def state_frozen(ep: dict, rng: np.random.Generator) -> dict:
    """The follower's joint readings stop updating (stale bus reads) while the arm keeps moving:
    actions and camera are fine, the recorded state is stuck at its first value."""
    return {**ep, "state": np.repeat(ep["state"][:1], len(ep["state"]), axis=0)}


def spikes(ep: dict, rng: np.random.Generator, ranges: np.ndarray) -> dict:
    """Teleop / bus glitches: three single-frame jumps in the recorded command of two arm joints.
    `ranges` is each action dimension's range over the clean dataset; a jump is 30-50% of it."""
    a = ep["action"].copy()
    arm_dims = [d for d in range(a.shape[1]) if d not in GRIPPER_DIMS]
    for t in rng.choice(np.arange(10, len(a) - 10), size=3, replace=False):
        for d in rng.choice(arm_dims, size=2, replace=False):
            a[t, d] += rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.5) * ranges[d]
    return {**ep, "action": a}


def idle(ep: dict, rng: np.random.Generator) -> dict:
    """Recording started long before and stopped long after the operator moved: still frames at
    rest before the take (40-60% of its length) and after it (20-30%)."""
    n = len(ep["action"])
    lead, tail = int(n * rng.uniform(0.4, 0.6)), int(n * rng.uniform(0.2, 0.3))

    def pad(x):
        return np.concatenate([np.repeat(x[:1], lead, axis=0), x, np.repeat(x[-1:], tail, axis=0)])

    return {"state": pad(ep["state"]), "action": pad(ep["action"]), "images": pad(ep["images"])}


def failed(ep: dict, rng: np.random.Generator) -> dict:
    """A failed demonstration: the operator's real motion, replayed in the simulator with the cube
    somewhere the grasp misses. The grippers close on air; images and states come from the physics."""
    import aloha_sim

    pose = aloha_sim.missed_cube_pose(ep["action"], rng)
    if pose is None:
        raise RuntimeError("could not find a cube placement this episode's grasp misses")
    out = aloha_sim.replay(ep["action"], pose, render=True)
    return {"state": out["state"], "action": ep["action"].copy(), "images": out["images"]}


def apply(name: str, ep: dict, rng: np.random.Generator, ranges: np.ndarray) -> dict:
    if name == "spikes":
        return spikes(ep, rng, ranges)
    return {"dead": dead, "state_frozen": state_frozen, "idle": idle, "failed": failed}[name](ep, rng)
