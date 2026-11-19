"""Run time and findings of `robolint check --videos` on a synthetic recording that looks like a
real one, before and after a change.

    python benchmarks/video.py --work /tmp/robolint-video [--old-src PATH_TO_OLDER_CHECKOUT/src]

Writes a LeRobot v3.0 dataset shaped like a bimanual recording: 14-dimensional action and state at
30 fps, two cameras (a fixed overhead view showing both arms and a wrist camera on the left arm)
encoded the way LeRobot encodes them (AV1, a key frame every second frame, CRF 30) when the AV1
encoder is available. Images have a textured background and sensor noise. The arms pause, and in
some episodes one arm rests while the other works, so a camera often sees no motion while the robot
moves. Three faults are injected and nothing else should be flagged:

- episode 5, overhead camera: the image repeats for 2 s while both arms move (VIDEO_FREEZE)
- episode 9, wrist camera: 3 frames lost mid-episode (VIDEO_DROPPED_FRAMES)
- episode 13: the left arm is commanded into an obstacle and stays put (must not be flagged)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
EP_SECONDS = 15
N_EPS = 24
W, H = 320, 240
CAMS = ("observation.images.cam_high", "observation.images.cam_left_wrist")
NAMES = [f"{side}_{j}" for side in ("left", "right") for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]  # fmt: skip
FREEZE_EP, DROP_EP, BLOCKED_EP = 5, 9, 13


def _arm(rng: np.random.Generator, n: int, active: bool) -> np.ndarray:
    """(n, 7) joint positions: bursts of motion separated by pauses, or a rest pose."""
    q = np.tile(rng.uniform(-0.3, 0.3, 7), (n, 1))
    if not active:
        return q
    t = 0
    while t < n:
        pause = int(rng.uniform(0.5, 2.5) * FPS)
        move = int(rng.uniform(1.5, 4.0) * FPS)
        a, b = min(n, t + pause), min(n, t + pause + move)
        if b > a:
            target = rng.uniform(-0.8, 0.8, 7)
            s = (1 - np.cos(np.linspace(0, np.pi, b - a))) / 2
            q[a:b] = q[a - 1 if a else 0] + s[:, None] * (target - q[a - 1 if a else 0])
            q[b:] = q[b - 1]
        t = b
    return q


def _texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    small = rng.uniform(40, 200, (h // 16 + 2, w // 16 + 2))
    big = np.kron(small, np.ones((16, 16)))[:h, :w]
    k = np.ones(9) / 9
    big = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, big)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, big).astype(np.float32)


_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


def _overhead(bg: np.ndarray, q: np.ndarray, noise: np.ndarray) -> np.ndarray:
    img = bg.copy()
    for side, x0, level in ((0, 0.25, 90.0), (7, 0.75, -80.0)):
        x = W * (x0 + 0.12 * q[side] + 0.05 * q[side + 2])
        y = H * (0.5 + 0.2 * q[side + 1] + 0.05 * q[side + 3])
        img += level * np.exp(-((_XX - x) ** 2 + (_YY - y) ** 2) / (2 * 18.0**2))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _wrist(bg: np.ndarray, q: np.ndarray, noise: np.ndarray) -> np.ndarray:
    dx = int(round(60 * q[0] + 30 * q[4]))
    dy = int(round(40 * q[1] + 20 * q[5]))
    img = np.roll(bg, (dy, dx), axis=(0, 1))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _encoder(av, choice: str):
    if choice == "h264-sparse":  # a key frame every 2 s, B-frames: the decoder has to decode everything
        return "libx264", {"g": "60", "bf": "2", "crf": "23"}
    if choice == "av1-flat":  # coarse AV1 that flattens static scenes and their sensor noise
        return "libsvtav1", {"g": "2", "crf": "55", "preset": "10"}
    for name, opts in (("libsvtav1", {"g": "2", "crf": "30", "preset": "10"}), ("libx264", {"g": "2", "crf": "23"})):
        try:
            av.codec.Codec(name, "w")
            return name, opts
        except Exception:  # noqa: BLE001 - try the next encoder
            continue
    return "mpeg4", {"g": "2"}


def generate(root: Path, seed: int = 0, noise_std: float = 2.0, encoder: str = "auto") -> Path:
    import av

    if (root / "meta" / "info.json").exists():
        return root
    rng = np.random.default_rng(seed)
    n = EP_SECONDS * FPS
    codec, opts = _encoder(av, encoder)
    bgs = {CAMS[0]: _texture(rng, H, W), CAMS[1]: _texture(rng, H, W)}
    outs, streams = {}, {}
    for cam in CAMS:
        path = root / "videos" / cam / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        outs[cam] = av.open(str(path), "w")
        st = outs[cam].add_stream(codec, rate=FPS, options=opts)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        streams[cam] = st
    rows, ep_meta = [], []
    for ep in range(N_EPS):
        left_active = ep % 4 != 1  # every fourth episode the left arm rests while the right one works
        q = np.hstack([_arm(rng, n, left_active), _arm(rng, n, True)])
        action = q + rng.normal(0, 0.002, q.shape)
        state = np.vstack([q[:2], q[:-2]]) + rng.normal(0, 0.001, q.shape)
        if ep == BLOCKED_EP:  # the left arm is pushed into an obstacle: commanded, not moving
            state[:, :7] = state[0, :7] + rng.normal(0, 0.001, (n, 7))
            q[:, :7] = q[0, :7]
        rows.append((action, state))
        meta = {"episode_index": ep, "tasks": ["transfer the cube"], "length": n, "data/chunk_index": 0,
                "data/file_index": 0, "dataset_from_index": ep * n, "dataset_to_index": (ep + 1) * n}  # fmt: skip
        for cam in CAMS:
            meta[f"videos/{cam}/chunk_index"] = 0
            meta[f"videos/{cam}/file_index"] = 0
            meta[f"videos/{cam}/from_timestamp"] = ep * EP_SECONDS
            meta[f"videos/{cam}/to_timestamp"] = (ep + 1) * EP_SECONDS
        ep_meta.append(meta)
        held = None
        for i in range(n):
            for cam in CAMS:
                if cam == CAMS[1] and ep == DROP_EP and n // 2 <= i < n // 2 + 3:
                    continue
                noise = rng.normal(0, noise_std, (H, W)).astype(np.float32) if noise_std else 0.0
                img = (_overhead if cam == CAMS[0] else _wrist)(bgs[cam], q[i], noise)
                if cam == CAMS[0] and ep == FREEZE_EP and 200 <= i < 260:  # the driver repeats one image
                    held = img if held is None else held
                    img = held
                frame = av.VideoFrame.from_ndarray(np.repeat(img[..., None], 3, axis=2), format="rgb24")
                frame.pts = ep * n + i
                for p in streams[cam].encode(frame):
                    outs[cam].mux(p)
    for cam in CAMS:
        for p in streams[cam].encode():
            outs[cam].mux(p)
        outs[cam].close()

    def lst(a: np.ndarray) -> pa.Array:
        return pa.FixedSizeListArray.from_arrays(pa.array(a.astype(np.float32).reshape(-1)), a.shape[1])

    action = np.vstack([a for a, _ in rows])
    state = np.vstack([s for _, s in rows])
    total = N_EPS * n
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": lst(action),
            "observation.state": lst(state),
            "timestamp": pa.array(np.tile(np.arange(n) / FPS, N_EPS).astype(np.float32)),
            "frame_index": pa.array(np.tile(np.arange(n), N_EPS).astype(np.int64)),
            "episode_index": pa.array(np.repeat(np.arange(N_EPS), n).astype(np.int64)),
            "index": pa.array(np.arange(total, dtype=np.int64)),
            "task_index": pa.array(np.zeros(total, np.int64)),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_meta), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["transfer the cube"]}), root / "meta" / "tasks.parquet")
    vec = {"dtype": "float32", "shape": [14], "names": NAMES}
    scalar = {"shape": [1], "names": None}
    features = {"action": vec, "observation.state": vec}
    for cam in CAMS:
        features[cam] = {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channels"]}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):  # fmt: skip
        features[k] = {"dtype": dt, **scalar}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha",
        "total_episodes": N_EPS,
        "total_frames": total,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return root


RUN = """
import json, sys, time
sys.path.insert(0, {src!r})
from robolint import lint
t0 = time.perf_counter()
r = lint({path!r}, videos=True)
dt = time.perf_counter() - t0
video = [{{"code": f.code, "episode": f.episode, "severity": f.severity.label, "message": f.message}}
         for f in r.findings if f.code.startswith("VIDEO")]
print(json.dumps({{"seconds": dt, "score": r.scores.dataset_score, "video_findings": video}}))
"""


def measure(src: str, path: Path) -> dict:
    p = subprocess.run([sys.executable, "-c", RUN.format(src=src, path=str(path))], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": (p.stderr or "").strip().splitlines()[-1:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--old-src", help="another robolint source tree to compare against")
    ap.add_argument("--noise", type=float, default=2.0, help="sensor noise (pixel standard deviation); 0 = rendered")
    ap.add_argument("--encoder", choices=["auto", "h264-sparse", "av1-flat"], default="auto")
    a = ap.parse_args()
    name = "bimanual-video" if (a.noise, a.encoder) == (2.0, "auto") else f"bimanual-video-{a.encoder}-noise{a.noise:g}"
    ds = generate(a.work / name, noise_std=a.noise, encoder=a.encoder)
    new_src = str(Path(__file__).resolve().parents[1] / "src")
    runs = {"after": measure(new_src, ds)}
    if a.old_src:
        runs["before"] = measure(a.old_src, ds)
    for name, r in runs.items():
        print(f"\n{name}: {r.get('seconds', 0):.1f} s, score {r.get('score')}")
        for f in r.get("video_findings", []):
            if f["code"] != "VIDEO_LAG":
                print(f"  {f['code']} ({f['severity']}) episode {f['episode']}: {f['message']}")
        if "error" in r:
            print("  error:", r["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
