"""Synthetic LeRobot datasets (v2.1 and v3.0 layouts) with injectable faults."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CAM = "observation.images.front"
W, H = 64, 48
SHIFT = 10


def _trajectory(rng: np.random.Generator, n: int, fps: float, gripper: bool = True) -> np.ndarray:
    t = np.arange(n) / fps
    a = np.zeros((n, 6))
    for j in range(5):
        a[:, j] = rng.uniform(-30, 30) + rng.uniform(20, 50) * np.sin(
            2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6)
        )
    if gripper:
        mid = n / fps / 2
        a[:, 5] = 5 + 40 / (1 + np.exp(-(t - mid) * 4))
    else:
        a[:, 5] = 5.0
    a += rng.normal(0, 0.05, a.shape)
    lead = min(8, n // 10)
    a[:lead] = a[lead]  # short, realistic idle lead-in
    return a


_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


def _frame_for(action_row: np.ndarray) -> np.ndarray:
    """A camera view of the arm: two soft blobs whose sub-pixel positions follow the first four joints,
    so the image changes smoothly and in proportion to how fast the arm moves."""
    img = np.full((H, W), 90.0, np.float32)
    for jx, jy, level in ((0, 1, 140.0), (2, 3, -60.0)):
        x = 10 + np.clip((action_row[jx] + 100) / 200, 0, 1) * (W - 20)
        y = 10 + np.clip((action_row[jy] + 100) / 200, 0, 1) * (H - 20)
        img += level * np.exp(-((_XX - x) ** 2 + (_YY - y) ** 2) / (2 * 5.0**2))
    img[:, :4] = 20
    rgb = np.repeat(np.clip(img, 0, 255)[..., None], 3, axis=2).astype(np.uint8)
    rgb[..., 1] = (rgb[..., 1] * 0.7).astype(np.uint8)
    return rgb


def _write_video(path: Path, frames: list[np.ndarray], fps: int, pts: list[int] | None = None, gop: int = 2) -> None:
    """pts: frame numbers to stamp the frames with (default 0, 1, 2, ...); skipping numbers leaves
    gaps in the timeline, as when frames are dropped while recording."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        st = out.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.codec_context.gop_size = gop
        st.bit_rate = 4_000_000  # near-lossless, like a real recording at sane settings
        for i, f in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(f, format="rgb24")
            if pts is not None:
                vf.pts = pts[i]
            for pkt in st.encode(vf):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def make_dataset(
    root: Path,
    *,
    version: str = "v3.0",
    n_eps: int = 10,
    fps: int = 30,
    length: int = 150,
    faults: dict[int, list[str]] | None = None,
    dataset_faults: list[str] | None = None,
    videos: bool = False,
    seed: int = 0,
    task: str = "Pick up the red cube and place it in the bin",
    grasp_object: bool = False,
    dims: int = 6,
    video_gop: int = 2,
) -> Path:
    """grasp_object: the follower gripper stops on an object when it closes (as in real pick-and-place
    data) instead of following the command to fully closed."""
    faults = faults or {}
    dataset_faults = dataset_faults or []
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    v3 = version.startswith("v3")

    eps_data, ep_meta = [], []
    actions_by_ep: dict[int, np.ndarray] = {}
    global_idx = 0
    for ep in range(n_eps):
        f = faults.get(ep, [])
        n = length
        if "short" in f:
            n = 12
        if "truncated" in f:
            n = max(fps + 3, int(length * 0.22))  # over 1s, under 25% of the median
        a = _trajectory(rng, n, fps, gripper="gripper_still" not in f)
        for fault in f:
            if fault.startswith("dup_of:"):
                src = int(fault.split(":")[1])
                a = actions_by_ep[src].copy() + rng.normal(0, 0.01, actions_by_ep[src].shape)
                n = len(a)
        if "dead" in f:
            a = np.tile(a[0], (n, 1))
        if "idle" in f:
            k = int(n * 0.45)
            a[:k] = a[k]
        if "hold" in f:  # controller updates every 4th frame and holds in between (sample-and-hold)
            a = a[(np.arange(n) // 4) * 4]
        if "spike" in f:
            a[n // 2, 1] += 150.0
        if "unseen_motion" in f:  # for a while only the wrist roll moves, which the camera does not show
            k0, k1 = int(0.3 * n), int(0.7 * n)
            a[k1:, :4] -= a[k1, :4] - a[k0, :4]
            a[k0:k1, :4] = a[k0, :4]
        if "unseen_all" in f:  # only joints the camera does not show move, all episode
            a[:, :4] = a[0, :4]
        if "rest_limit" in f:  # arm rests on its hard stop before the take; leader reads slightly past it
            k = int(n * 0.3)
            a[:k] = a[k]
            a[:k, 1] = -95.0
        if "one_arm_rest" in f:  # bimanual-style: this joint rests on its stop all episode, others work
            a[:, 1] = -95.0
        if "binary_gripper" in f:  # on/off gripper command with quick toggles
            a[:, 5] = np.where((np.arange(n) // 7) % 3 == 0, 1.0, -1.0)
            a[n // 2, 5] = -a[n // 2 - 1, 5]  # a one-frame toggle
        if "hard_limit" in f:  # operator keeps pushing past the joint's reach, mid-task
            a[n // 2 :, 1] = -95.0 + 5.0 * np.sin(np.arange(n - n // 2) / 5.0)
        if "regrasp" in f:  # first close misses, the operator reopens and grasps again
            t_ = np.arange(n)
            a[:, 5] = np.where(((t_ >= 0.25 * n) & (t_ < 0.4 * n)) | (t_ >= 0.6 * n), 45.0, 5.0)
            a[:, 5] += rng.normal(0, 0.05, n)
        pick = "pick_place" in f or "pick_place_missed" in f
        if pick:  # real teleop shape: rest closed, open, close on the object, release, back to rest closed
            t_ = np.arange(n) / n
            wide = 4.0 if ep % 3 == 0 else 11.0  # operators open by different amounts from take to take
            a[:, 5] = np.select([t_ < 0.15, t_ < 0.35, t_ < 0.65, t_ < 0.85], [45.0, wide, 45.0, wide], 45.0)
            if "soft_grasp" in f or "gentle_grasp" in f:  # the operator does not squeeze the leader all the way
                a[(t_ >= 0.35) & (t_ < 0.65), 5] = 34.0
            a[:, 5] += rng.normal(0, 0.05, n)
        s = np.vstack([a[:2], a[:-2]]) + rng.normal(0, 0.05, a.shape)
        if "state_lag" in f:  # the state stream arrives 12 frames late (buffered/stalled recording)
            s = np.vstack([np.repeat(a[:1], 12, axis=0), a[:-12]]) + rng.normal(0, 0.05, a.shape)
        if pick and "pick_place_missed" not in f:
            t_ = np.arange(n) / n
            on_object = (t_ >= 0.35) & (t_ < 0.67)
            width = 26.0 if "soft_grasp" in f else 32.0 if "gentle_grasp" in f else 30.0  # gentle: barely any contact
            s[on_object, 5] = np.minimum(s[on_object, 5], width + rng.normal(0, 0.05, int(on_object.sum())))
        if grasp_object and not pick and "missed_grasp" not in f:
            blocked = np.arange(n) >= (0.6 * n if "regrasp" in f else 0)
            width = 37.0 if "small_object" in f else 30.0  # a smaller object lets the gripper close further
            s[blocked, 5] = np.minimum(s[blocked, 5], width + rng.normal(0, 0.05, int(blocked.sum())))
        if "hard_limit" in f or "rest_limit" in f or "one_arm_rest" in f:
            s[:, 1] = np.maximum(s[:, 1], -85.0)  # follower stops at its hard limit
        if "state_unrelated" in f:  # the state measures something that does not follow the command
            s = np.cumsum(rng.normal(0, 1.0, a.shape), axis=0)
        if "state_frozen" in f:
            s = np.tile(s[0], (n, 1))
        if "nan" in f:
            a[5, 2] = np.nan
        ts = np.arange(n) / fps
        fi = np.arange(n)
        if "dropped" in f:
            ts[n // 2 :] += 10 / fps  # 10 frames lost
        if "nonmono" in f:
            ts[20] = ts[18]
        if "jitter" in f:
            ts = ts + rng.normal(0, 0.3 / fps, n)
            ts = np.sort(ts)
            ts[0] = 0.0
        if "fi_gap" in f:
            fi[n // 3 :] += 3
        actions_by_ep[ep] = a
        eps_data.append(
            {
                "action": a[:, :dims].astype(np.float32),
                "observation.state": s[:, :dims].astype(np.float32),
                "timestamp": ts.astype(np.float32),
                "frame_index": fi.astype(np.int64),
                "episode_index": np.full(n, ep, np.int64),
                "index": np.arange(global_idx, global_idx + n, dtype=np.int64),
                "task_index": np.full(n, 1 if "bad_task" in f else 0, np.int64),
            }
        )
        global_idx += n
        meta_len = n + 5 if "length_meta" in f else n
        row = {"episode_index": ep, "tasks": [task], "length": meta_len}
        if version.startswith("v3"):
            row.update(
                {
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_idx - n,
                    "dataset_to_index": global_idx,
                }
            )
        ep_meta.append(row)

    features = {
        "action": {"dtype": "float32", "shape": [dims], "names": NAMES[:dims]},
        "observation.state": {"dtype": "float32", "shape": [dims], "names": NAMES[:dims]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if videos:
        features[CAM] = {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"]}
    total_eps = n_eps + (3 if "count_mismatch" in dataset_faults else 0)
    info = {
        "codebase_version": version,
        "robot_type": "so101_follower",
        "total_episodes": total_eps,
        "total_frames": global_idx,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "features": features,
    }

    def table(d: dict) -> pa.Table:
        cols = {}
        for k, v in d.items():
            if v.ndim == 2:
                cols[k] = pa.array(list(v), type=pa.list_(pa.float32()))
            else:
                cols[k] = pa.array(v)
        return pa.table(cols)

    (root / "meta").mkdir(exist_ok=True)
    task_text = "test" if "vague_task" in dataset_faults else task
    if v3:
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        merged = {k: np.concatenate([e[k] for e in eps_data]) for k in eps_data[0]}
        pq.write_table(table(merged), root / "data" / "chunk-000" / "file-000.parquet")
        pq.write_table(pa.table({"task_index": [0], "task": [task_text]}), root / "meta" / "tasks.parquet")
    else:
        info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for ep, d in enumerate(eps_data):
            pq.write_table(table(d), root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task_text}) + "\n")

    if videos:
        all_frames: list[np.ndarray] = []
        all_pts: list[int] = []
        t0 = 0.0
        for ep in range(len(eps_data)):
            f = faults.get(ep, [])
            a = actions_by_ep[ep]
            if "black_video" in f:
                frames = [np.zeros((H, W, 3), np.uint8)] * len(a)
            elif "frozen_video" in f:
                frames = [_frame_for(a[0])] * len(a)
            elif "video_shift" in f:  # the camera shows each motion 10 frames after it was commanded
                frames = [_frame_for(a[max(0, t - SHIFT)]) for t in range(len(a))]
            else:
                frames = [_frame_for(row) for row in a]
            if "freeze_stretch" in f:  # the camera repeats one image for 1.5 s mid-episode
                k = int(0.4 * len(frames))
                frames[k : k + int(1.5 * fps)] = [frames[k]] * int(1.5 * fps)
            if "short_video" in f:
                frames = frames[: len(frames) // 2]
            n_written = len(frames)
            pts = list(range(n_written))
            if "drop_frames" in f:  # five frames lost mid-episode; the rest keep their timestamps
                k = n_written // 2
                del frames[k : k + 5]
                del pts[k : k + 5]
            if v3:
                all_frames.extend(frames)
                all_pts.extend(int(round(t0 * fps)) + p for p in pts)
                ep_meta[ep][f"videos/{CAM}/chunk_index"] = 0
                ep_meta[ep][f"videos/{CAM}/file_index"] = 0
                ep_meta[ep][f"videos/{CAM}/from_timestamp"] = t0
                ep_meta[ep][f"videos/{CAM}/to_timestamp"] = t0 + n_written / fps
                t0 += n_written / fps
            elif "missing_video" not in f:
                path = root / "videos" / "chunk-000" / CAM / f"episode_{ep:06d}.mp4"
                _write_video(path, frames, fps, pts if "drop_frames" in f else None, video_gop)
        if v3:
            dropped = any("drop_frames" in faults.get(ep, []) for ep in range(len(eps_data)))
            path = root / "videos" / CAM / "chunk-000" / "file-000.mp4"
            _write_video(path, all_frames, fps, all_pts if dropped else None, video_gop)

    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    if v3:
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        cols: dict[str, list] = {}
        for row in ep_meta:
            for k in row:
                cols.setdefault(k, [])
        for row in ep_meta:
            for k in cols:
                cols[k].append(row.get(k))
        pq.write_table(pa.table(cols), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    else:
        (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_meta))
    return root
