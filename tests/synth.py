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


def _frame_for(action_row: np.ndarray) -> np.ndarray:
    img = np.full((H, W, 3), 90, np.uint8)
    x = int(np.clip((action_row[0] + 100) / 200, 0, 1) * (W - 20))
    y = int(np.clip((action_row[1] + 100) / 200, 0, 1) * (H - 20))
    img[y : y + 20, x : x + 20] = (230, 60, 40)
    img[:, :4] = (20, 20, 20)
    return img


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        st = out.add_stream("mpeg4", rate=fps)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.codec_context.gop_size = 2
        st.bit_rate = 4_000_000  # near-lossless, like a real recording at sane settings
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
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
) -> Path:
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
        s = np.vstack([a[:2], a[:-2]]) + rng.normal(0, 0.05, a.shape)
        if "hard_limit" in f or "rest_limit" in f or "one_arm_rest" in f:
            s[:, 1] = np.maximum(s[:, 1], -85.0)  # follower stops at its hard limit
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
                "action": a.astype(np.float32),
                "observation.state": s.astype(np.float32),
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
        "action": {"dtype": "float32", "shape": [6], "names": NAMES},
        "observation.state": {"dtype": "float32", "shape": [6], "names": NAMES},
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
        t0 = 0.0
        for ep in range(len(eps_data)):
            f = faults.get(ep, [])
            a = actions_by_ep[ep]
            if "black_video" in f:
                frames = [np.zeros((H, W, 3), np.uint8)] * len(a)
            elif "frozen_video" in f:
                frames = [_frame_for(a[0])] * len(a)
            else:
                frames = [_frame_for(row) for row in a]
            if "short_video" in f:
                frames = frames[: len(frames) // 2]
            n_written = len(frames)
            if v3:
                all_frames.extend(frames)
                ep_meta[ep][f"videos/{CAM}/chunk_index"] = 0
                ep_meta[ep][f"videos/{CAM}/file_index"] = 0
                ep_meta[ep][f"videos/{CAM}/from_timestamp"] = t0
                ep_meta[ep][f"videos/{CAM}/to_timestamp"] = t0 + n_written / fps
                t0 += n_written / fps
            elif "missing_video" not in f:
                _write_video(root / "videos" / "chunk-000" / CAM / f"episode_{ep:06d}.mp4", frames, fps)
        if v3:
            _write_video(root / "videos" / CAM / "chunk-000" / "file-000.mp4", all_frames, fps)

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
