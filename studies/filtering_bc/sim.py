"""A small 2D pick-and-place task, a scripted "teleoperator", and the recording faults seen in
real community robot datasets.

The robot is a point gripper moving in the plane. It must close its gripper on an object, carry
it to a goal, and open the gripper there. Actions are [vel_x, vel_y, gripper]; the state is
[x, y, object_x, object_y, goal_x, goal_y, gripper].
"""

from __future__ import annotations

import numpy as np

FPS = 10
DT = 1.0 / FPS
VMAX = 1.0
GRASP_R = 0.10  # the gripper must close within this distance of the object to pick it up
GOAL_R = 0.15
HORIZON = 150  # evaluation step limit (15 s)

ACTION_NAMES = ["vel_x", "vel_y", "gripper"]
STATE_NAMES = ["x", "y", "object_x", "object_y", "goal_x", "goal_y", "gripper"]
OBJECT_WIDTH = 0.6  # where the follower gripper stops when it closes on the object (1.0 = fully closed)


class VecEnv:
    """n independent copies of the task, stepped together."""

    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self.p = rng.uniform(-1, 1, (n, 2))
        self.o = rng.uniform(-0.9, 0.9, (n, 2))
        self.g = rng.uniform(-0.9, 0.9, (n, 2))
        # keep the three points apart so every episode needs real motion
        for _ in range(100):
            close = (np.linalg.norm(self.o - self.g, axis=1) < 0.6) | (np.linalg.norm(self.p - self.o, axis=1) < 0.5)
            if not close.any():
                break
            k = int(close.sum())
            self.o[close] = rng.uniform(-0.9, 0.9, (k, 2))
            self.g[close] = rng.uniform(-0.9, 0.9, (k, 2))
        self.grip = np.zeros(n)  # follower gripper position: 0 open, 1 fully closed
        self.cmd = np.zeros(n, bool)
        self.held = np.zeros(n, bool)
        self.done = np.zeros(n, bool)

    def obs(self) -> np.ndarray:
        return np.column_stack([self.p, self.o, self.g, self.grip])

    def step(self, a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.float64)
        live = ~self.done
        v = np.clip(a[:, :2], -VMAX, VMAX)
        self.p[live] = np.clip(self.p[live] + v[live] * DT, -1.2, 1.2)
        cmd = a[:, 2] > 0.5
        closing = cmd & ~self.cmd
        near = np.linalg.norm(self.p - self.o, axis=1) < GRASP_R
        self.held |= live & closing & near
        self.held &= cmd  # opening drops the object
        self.cmd = cmd
        # the follower moves toward the command; an object in the gripper stops it partway
        target = np.where(cmd, np.where(self.held, OBJECT_WIDTH, 1.0), 0.0)
        self.grip[live] += np.clip(target - self.grip, -0.5, 0.5)[live]
        self.o[self.held] = self.p[self.held]
        released_at_goal = live & ~self.held & ~cmd & (np.linalg.norm(self.o - self.g, axis=1) < GOAL_R)
        self.done |= released_at_goal

    @property
    def success(self) -> np.ndarray:
        return self.done


def teleop_episode(rng: np.random.Generator, fail_grasp: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:
    """One human-like demonstration: (states, actions, succeeded).

    fail_grasp: the operator closes the gripper too far from the object, does not notice, and
    finishes the motion anyway, which is how failed demos end up in datasets.
    """
    env = VecEnv(1, rng)
    speed = rng.uniform(0.55, 0.95)  # operators differ in pace
    gain = rng.uniform(4.0, 6.0)
    miss = rng.uniform(0.2, 0.3) if fail_grasp else 0.0
    states, actions = [], []
    phase, hold, vel = "reach", 0, np.zeros(2)
    lead = int(rng.integers(2, 6))  # short, normal pause before moving
    for t in range(400):
        s = env.obs()[0].copy()
        p, o, g = env.p[0], env.o[0], env.g[0]
        grip = 0.0
        if t < lead:
            target_v = np.zeros(2)
        elif phase == "reach":
            # a failed attempt aims at a point short of the object
            aim = o + (p - o) / max(np.linalg.norm(p - o), 1e-9) * miss if fail_grasp else o
            d = aim - p
            target_v = _toward(d, speed, gain)
            if np.linalg.norm(d) < 0.05:
                phase, hold = "close", 5
        elif phase == "close":
            target_v, grip = np.zeros(2), 1.0
            hold -= 1
            if hold <= 0:
                phase = "carry"
        elif phase == "carry":
            grip = 1.0
            d = g - p
            target_v = _toward(d, speed, gain)
            if np.linalg.norm(d) < 0.08:
                phase, hold = "release", 5
        else:  # release
            target_v, grip = np.zeros(2), 0.0
            hold -= 1
        vel = 0.6 * vel + 0.4 * target_v  # hand motion is smooth, not instant
        a = np.r_[vel + rng.normal(0, 0.03, 2), grip]
        states.append(s)
        actions.append(a)
        env.step(a[None])
        if phase == "release" and hold <= 0:
            break
    return np.array(states), np.array(actions), bool(env.success[0])


def _toward(d: np.ndarray, speed: float, gain: float) -> np.ndarray:
    """Move straight at the target at the operator's pace, slowing only in the last few cm."""
    dist = float(np.linalg.norm(d))
    return d / max(dist, 1e-9) * min(speed, gain * dist)


# --- recording faults -------------------------------------------------------------------------


def fault_dead(s: np.ndarray, a: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Robot was never driven: the recording runs but nothing moves."""
    return np.tile(s[0], (len(s), 1)), np.tile(np.r_[0.0, 0.0, 0.0], (len(a), 1))


def fault_state_frozen(s: np.ndarray, a: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Follower disconnected: the operator's commands are recorded, the measured state never changes."""
    return np.tile(s[0], (len(s), 1)), a.copy()


def fault_spikes(s: np.ndarray, a: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Teleop / bus glitches: a few single-frame jumps in the recorded command."""
    a = a.copy()
    for k in rng.choice(np.arange(5, len(a) - 5), size=3, replace=False):
        a[k, :2] += rng.choice([-1, 1], 2) * rng.uniform(4, 6, 2) * VMAX
    return s, a


def fault_idle(s: np.ndarray, a: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Recording started long before and stopped long after the operator moved."""
    n = len(s)
    lead, tail = int(n * rng.uniform(0.4, 0.6)), int(n * rng.uniform(0.2, 0.3))
    s2 = np.vstack([np.tile(s[0], (lead, 1)), s, np.tile(s[-1], (tail, 1))])
    still = np.r_[0.0, 0.0, 0.0]
    end = np.r_[0.0, 0.0, a[-1, 2]]
    a2 = np.vstack([np.tile(still, (lead, 1)), a, np.tile(end, (tail, 1))])
    return s2, a2


FAULTS = {
    "dead": fault_dead,
    "state_frozen": fault_state_frozen,
    "spikes": fault_spikes,
    "idle": fault_idle,
}
