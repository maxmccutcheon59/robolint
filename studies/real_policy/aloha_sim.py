"""The ALOHA transfer-cube simulation used by the study, driven directly (no gym wrapper).

Two jobs:
  * replay(): play a recorded action sequence back in the joint-position sim with the cube at a
    chosen place. The study uses it to make failed demonstrations: the operator's real motion,
    replayed with the cube somewhere the grasp misses, closes the grippers on air. The images,
    joint states and gripper readings all come from the physics, so the fault is consistent.
  * scripted_episode(): a scripted demonstration (the ACT paper's pick-and-transfer recipe), used
    only to build a small stand-in dataset for smoke tests on machines without the Hub dataset.

Needs `pip install gym-aloha` (pulled in by lerobot[aloha]) and a MuJoCo render backend
(MUJOCO_GL=egl, or osmesa); see the README.
"""

from __future__ import annotations

import numpy as np

TASK = "Pick the cube with the right arm and transfer it to the left arm."  # informational only
FPS = 50
EPISODE_LEN = 400
BOX_X, BOX_Y, BOX_Z = (0.0, 0.2), (0.4, 0.6), 0.05  # gym_aloha's cube sampling range


def _envs():
    # Imported lazily: dm_control/MuJoCo are only needed by the steps that run the simulator.
    import collections

    from dm_control import mujoco
    from dm_control.rl import control
    from gym_aloha.constants import ASSETS_DIR, DT, normalize_puppet_gripper_position
    from gym_aloha.tasks import sim as sim_tasks
    from gym_aloha.tasks.sim_end_effector import TransferCubeEndEffectorTask

    class JointTask(sim_tasks.TransferCubeTask):
        """gym_aloha's joint-position task, rendering only the 'top' camera (and only on request)."""

        render = True

        def get_observation(self, physics):
            obs = collections.OrderedDict()
            obs["qpos"] = self.get_qpos(physics)
            obs["images"] = {"top": physics.render(height=480, width=640, camera_id="top")} if self.render else {}
            return obs

    class EETask(TransferCubeEndEffectorTask):
        """End-effector (mocap) task for the scripted policy; no rendering, the cube placed by us."""

        box_pose: np.ndarray | None = None

        def initialize_episode(self, physics):
            self.initialize_robots(physics)
            i = physics.model.name2id("red_box_joint", "joint")
            np.copyto(physics.data.qpos[i : i + 7], self.box_pose)
            super(TransferCubeEndEffectorTask, self).initialize_episode(physics)

        def get_observation(self, physics):
            obs = collections.OrderedDict()
            obs["qpos"] = self.get_qpos(physics)
            obs["env_state"] = physics.data.qpos.copy()[16:]
            obs["mocap_pose_left"] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
            obs["mocap_pose_right"] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
            obs["gripper_ctrl"] = physics.data.ctrl.copy()
            return obs

    def joint_env(render: bool):
        physics = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"))
        task = JointTask()
        task.render = render
        return control.Environment(
            physics, task, float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False
        )

    def ee_env(box_pose):
        physics = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"))
        # Recent MuJoCo computes each weld's offset from the model's default pose; the scripted
        # waypoints (from the ACT paper) assume the mocap body sits exactly at the gripper.
        physics.model.eq_data[:, 3:6] = 0.0
        task = EETask()
        task.box_pose = box_pose
        return control.Environment(
            physics, task, float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False
        )

    return sim_tasks, joint_env, ee_env, normalize_puppet_gripper_position


def sample_box_pose(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(*BOX_X), rng.uniform(*BOX_Y), BOX_Z, 1.0, 0.0, 0.0, 0.0])


def replay(actions: np.ndarray, box_pose: np.ndarray, render: bool = True) -> dict:
    """Play `actions` (T x 14 joint targets) in the joint sim with the cube at `box_pose`.

    Returns states (T x 14, the observation *before* each action, as LeRobot records it), images
    (T x 480 x 640 x 3 uint8, or None), the best task reward reached (4 = cube transferred), and
    whether the robot touched the cube at all."""
    sim_tasks, joint_env, _, _ = _envs()
    sim_tasks.BOX_POSE[0] = np.asarray(box_pose, dtype=np.float64)
    env = joint_env(render)
    ts = env.reset()
    ph = env.physics
    box_geom = ph.model.name2id("red_box", "geom")
    robot = {i for i in range(ph.model.ngeom) if (ph.model.id2name(i, "geom") or "").startswith("vx300s")}
    states, images, best, touched = [], [], 0.0, False
    for a in actions:
        states.append(ts.observation["qpos"].copy())
        if render:
            images.append(ts.observation["images"]["top"])
        ts = env.step(np.asarray(a, dtype=np.float64))
        best = max(best, float(ts.reward or 0.0))
        for c in ph.data.contact[: ph.data.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if box_geom in pair and pair & robot:
                touched = True
    return {
        "state": np.asarray(states, dtype=np.float32),
        "images": np.asarray(images, dtype=np.uint8) if render else None,
        "max_reward": best,
        "touched": touched,  # any part of either robot touched the cube
    }


def missed_cube_pose(actions: np.ndarray, rng: np.random.Generator, tries: int = 40) -> np.ndarray | None:
    """A cube placement where replaying `actions` never touches the cube with any part of the robot.

    Tries cube poses from the task's own sampling range, then just outside it; physics only, no
    rendering, so each try takes a second or two."""
    for k in range(tries):
        pose = sample_box_pose(rng)
        if k >= tries // 2:  # widen the search if the operator's motion sweeps the whole range
            pose[0] += rng.choice([-1, 1]) * rng.uniform(0.08, 0.15)
        if not replay(actions, pose, render=False)["touched"]:
            return pose
    return None


# --- scripted demonstrations (stand-in data for smoke tests) ------------------------------------

_START = np.array([0, -0.96, 1.16, 0, -0.3, 0], dtype=np.float64)


def _fingertips(arm_q: np.ndarray) -> np.ndarray:
    """World position between the right gripper's fingers for right-arm joints `arm_q` (6)."""
    from dm_control import mujoco
    from gym_aloha.constants import ASSETS_DIR

    ph = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"))
    ph.data.qpos[8:14] = arm_q
    ph.forward()
    return (
        ph.named.data.xpos["vx300s_right/left_finger_link"] + ph.named.data.xpos["vx300s_right/right_finger_link"]
    ) / 2


def scripted_episode(rng: np.random.Generator) -> dict:
    """A scripted right-arm pick, lift and place in the transfer-cube scene (joint-space waypoints,
    with the cube placed under the gripper's reach point). It is a stand-in for smoke tests, not
    the transfer task itself: it gives real grasps (the cube stops the fingers partway) on real
    renders, in the exact schema of lerobot/aloha_sim_transfer_cube_human."""
    waist = rng.uniform(-0.25, 0.25)
    down = np.array([waist, 0.4 + rng.uniform(-0.05, 0.05), 0.4, 0, 0.8, 0])
    above = down + [0, -0.45, 0, 0, 0, 0]
    place = above + [rng.uniform(-0.4, 0.4), 0, 0, 0, 0, 0]
    tip = _fingertips(down)
    box = np.array([tip[0], tip[1], BOX_Z, 1.0, 0.0, 0.0, 0.0])
    # (frame, right arm joints, right gripper: 1 open, lower is closed); speed varies like an operator's.
    # Operators in the human ALOHA sim data close the leader to about 0.2, not all the way to 0
    # (the follower's fingers meet at about 0.1).
    k = rng.uniform(0.85, 1.15)
    c = rng.uniform(0.15, 0.25)
    keys = [
        (0, _START, 1),
        (30, _START, 1),
        (130, above, 1),
        (180, down, 1),
        (200, down, 1),
        (230, down, c),
        (290, above, c),
        (340, place, c),
        (360, place, 1),
        (EPISODE_LEN - 1, _START, 1),
    ]
    ts = [int(round(t * k)) if 0 < i < len(keys) - 1 else t for i, (t, _, _) in enumerate(keys)]
    ts = list(np.minimum(ts, EPISODE_LEN - 1))
    actions = np.zeros((EPISODE_LEN, 14))
    for t in range(EPISODE_LEN):
        j = max(i for i in range(len(ts)) if ts[i] <= t)
        j2 = min(j + 1, len(ts) - 1)
        f = 0.0 if ts[j2] == ts[j] else (t - ts[j]) / (ts[j2] - ts[j])
        f = 0.5 - 0.5 * np.cos(np.pi * f)  # smooth in, smooth out
        arm = keys[j][1] + (keys[j2][1] - keys[j][1]) * f
        grip = keys[j][2] + (keys[j2][2] - keys[j][2]) * f
        actions[t, :7] = np.r_[_START, 1.0]  # left arm stays at rest, gripper open
        actions[t, 7:] = np.r_[arm, grip]
    actions[:, :6] += rng.normal(0, 0.002, (EPISODE_LEN, 6))  # an operator's hand is never perfectly still
    out = replay(actions, box, render=True)
    return {
        "state": out["state"],
        "action": actions.astype(np.float32),
        "images": out["images"],
        "success": out["max_reward"] >= 2.0,  # cube lifted off the table
    }
