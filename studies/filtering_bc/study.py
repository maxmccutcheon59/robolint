"""Does filtering a dataset with robolint produce a better robot policy?

For each fault mix and dataset seed:
  1. Record 200 scripted teleop demonstrations of a pick-and-place task.
  2. Corrupt a share of them with the faults found in community datasets (dead episodes,
     disconnected follower, teleop glitches, idle padding) and include failed demos.
  3. Save it as a real LeRobot v3 dataset and run robolint on it, unchanged.
  4. Train the same behavior-cloning policy on four versions of the data:
       raw            every episode
       robolint-err   episodes robolint flags as errors removed
       robolint-all   episodes with errors or warnings removed
       oracle         only clean, successful demos (the ceiling no tool can beat)
  5. Roll each policy out on the same 300 held-out task setups and count successes.

Run: python study.py [--quick]   (from this folder; needs numpy, pyarrow, robolint)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from robolint import Severity, lint  # noqa: E402

N_EPISODES = 200
MIXES = {
    # share of episodes affected by each problem
    "heavy": {"dead": 0.05, "state_frozen": 0.05, "spikes": 0.08, "idle": 0.08, "failed": 0.10},
    "light": {"dead": 0.02, "state_frozen": 0.02, "spikes": 0.03, "idle": 0.03, "failed": 0.03},
}
CONDITIONS = ["raw", "robolint-err", "robolint-all", "oracle"]
EVAL_EPISODES = 300
EVAL_SEED = 12345


# --- data --------------------------------------------------------------------------------------


def make_episodes(mix: dict[str, float], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    labels = ["clean"] * N_EPISODES
    order = rng.permutation(N_EPISODES)
    k = 0
    for name, share in mix.items():
        for i in order[k : k + round(share * N_EPISODES)]:
            labels[i] = name
        k += round(share * N_EPISODES)
    eps = []
    for label in labels:
        s, a, ok = sim.teleop_episode(rng, fail_grasp=label == "failed")
        if label in sim.FAULTS:
            s, a = sim.FAULTS[label](s, a, rng)
        eps.append({"label": label, "state": s, "action": a, "success": ok and label == "clean"})
    return eps


def write_lerobot(eps: list[dict], root: Path) -> None:
    """Write the episodes as a LeRobot v3.0 dataset."""
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    cols: dict[str, list] = {
        k: []
        for k in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    meta: dict[str, list] = {
        k: []
        for k in (
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        )
    }
    g = 0
    for i, e in enumerate(eps):
        n = len(e["action"])
        cols["action"].append(e["action"].astype(np.float32))
        cols["observation.state"].append(e["state"].astype(np.float32))
        cols["timestamp"].append((np.arange(n) / sim.FPS).astype(np.float32))
        cols["frame_index"].append(np.arange(n))
        cols["episode_index"].append(np.full(n, i))
        cols["index"].append(np.arange(g, g + n))
        cols["task_index"].append(np.zeros(n, np.int64))
        for k, v in (
            ("episode_index", i),
            ("tasks", ["Pick up the block and place it on the target"]),
            ("length", n),
            ("data/chunk_index", 0),
            ("data/file_index", 0),
            ("dataset_from_index", g),
            ("dataset_to_index", g + n),
        ):
            meta[k].append(v)
        g += n

    def vec(chunks):
        a = np.concatenate(chunks)
        return pa.FixedSizeListArray.from_arrays(pa.array(a.ravel()), a.shape[1])

    table = pa.table(
        {
            "action": vec(cols["action"]),
            "observation.state": vec(cols["observation.state"]),
            **{
                k: np.concatenate(cols[k]) for k in ("timestamp", "frame_index", "episode_index", "index", "task_index")
            },
        }
    )
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.table(meta), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(
        pa.table({"task_index": [0], "task": ["Pick up the block and place it on the target"]}),
        root / "meta/tasks.parquet",
    )
    features = {
        "action": {"dtype": "float32", "shape": [3], "names": sim.ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": [7], "names": sim.STATE_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        **{
            k: {"dtype": "int64", "shape": [1], "names": None}
            for k in ("frame_index", "episode_index", "index", "task_index")
        },
    }
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "sim_point_gripper",
                "fps": sim.FPS,
                "total_episodes": len(eps),
                "total_frames": g,
                "total_tasks": 1,
                "chunks_size": 1000,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "features": features,
            }
        )
    )


# --- policy: a plain MLP trained by behavior cloning (numpy, CPU) ------------------------------


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b, strict=True)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            hs.append(x)
        return x, hs

    def fit(self, X, Y, rng, weights=None, steps=3000, batch=256, lr=1e-3):
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        weights = np.ones_like(Y) if weights is None else weights
        for t in range(1, steps + 1):
            idx = rng.integers(0, len(X), batch)
            out, hs = self.forward(X[idx])
            grad = 2 * weights[idx] * (out - Y[idx]) / batch
            gW, gb = [], []
            for i in range(len(self.W) - 1, -1, -1):
                gW.insert(0, hs[i].T @ grad)
                gb.insert(0, grad.sum(0))
                if i:
                    grad = (grad @ self.W[i].T) * (hs[i] > 0)
            for j, (p, gr) in enumerate(zip(params, gW + gb, strict=True)):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr * gr
                p -= lr * (m[j] / (1 - 0.9**t)) / (np.sqrt(v[j] / (1 - 0.999**t)) + 1e-8)


def features(states: np.ndarray) -> np.ndarray:
    """Policy input: object and goal relative to the gripper, plus the gripper state."""
    p, o, g = states[:, 0:2], states[:, 2:4], states[:, 4:6]
    op, gp = o - p, g - o
    return np.column_stack([op, g - p, gp, np.linalg.norm(op, axis=1), np.linalg.norm(gp, axis=1), states[:, 6]])


def train_and_eval(states: np.ndarray, actions: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    # A binary gripper changes state on one frame per grasp or release; up-weight those frames
    # so a single-step policy learns when to act (chunked policies like ACT get this for free).
    weights = np.ones_like(actions)
    weights[:, 2] = np.where((actions[:, 2] > 0.5) != (states[:, 6] > 0.5), 20.0, 1.0)
    states = features(states)
    # mean/std normalization from the training data, as LeRobot policies do
    s_mu, s_sd = states.mean(0), states.std(0) + 1e-6
    a_mu, a_sd = actions.mean(0), actions.std(0) + 1e-6
    net = MLP([states.shape[1], 256, 256, actions.shape[1]], rng)
    net.fit((states - s_mu) / s_sd, (actions - a_mu) / a_sd, rng, weights)
    env = sim.VecEnv(EVAL_EPISODES, np.random.default_rng(EVAL_SEED))
    for _ in range(sim.HORIZON):
        out, _ = net.forward((features(env.obs()) - s_mu) / s_sd)
        env.step(out * a_sd + a_mu)
    return float(env.success.mean())


# --- one dataset -------------------------------------------------------------------------------


def run_dataset(job: tuple[str, int, int]) -> dict:
    mix_name, data_seed, train_seeds = job
    eps = make_episodes(MIXES[mix_name], data_seed)
    tmp = Path(tempfile.mkdtemp())
    try:
        write_lerobot(eps, tmp / "ds")
        result = lint(tmp / "ds")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    err = set(result.bad_episodes(Severity.ERROR))
    warn = set(result.bad_episodes(Severity.WARNING))
    keep = {
        "raw": range(len(eps)),
        "robolint-err": [i for i in range(len(eps)) if i not in err],
        "robolint-all": [i for i in range(len(eps)) if i not in warn],
        "oracle": [i for i, e in enumerate(eps) if e["success"]],
    }
    scores = {}
    for cond in CONDITIONS:
        S = np.concatenate([eps[i]["state"] for i in keep[cond]])
        A = np.concatenate([eps[i]["action"] for i in keep[cond]])
        scores[cond] = [train_and_eval(S, A, 1000 * data_seed + t) for t in range(train_seeds)]
    detection = {}
    for label in ["clean", *MIXES[mix_name]]:
        idx = [i for i, e in enumerate(eps) if e["label"] == label]
        detection[label] = {
            "episodes": len(idx),
            "flagged_error": sum(i in err for i in idx),
            "flagged_any": sum(i in warn for i in idx),
        }
    return {
        "mix": mix_name,
        "data_seed": data_seed,
        "scores": scores,
        "detection": detection,
        "kept": {c: len(list(k)) for c, k in keep.items()},
        "robolint_score": result.scores.dataset_score,
        "robolint_grade": result.scores.grade,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 dataset seed x 1 training seed per mix")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args()
    data_seeds, train_seeds = (1, 1) if args.quick else (4, 3)
    jobs = [(m, s, train_seeds) for m in MIXES for s in range(data_seeds)]
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        runs = list(pool.map(run_dataset, jobs))
    args.out.write_text(json.dumps(runs, indent=2))
    for mix in MIXES:
        rs = [r for r in runs if r["mix"] == mix]
        print(f"\n{mix} faults ({len(rs)} datasets x {len(rs[0]['scores']['raw'])} training seeds)")
        for c in CONDITIONS:
            v = np.array([x for r in rs for x in r["scores"][c]])
            print(
                f"  {c:13s} success {v.mean() * 100:5.1f}%  (sd {v.std() * 100:4.1f}, "
                f"{np.mean([r['kept'][c] for r in rs]):.0f} episodes kept)"
            )
    print(f"\n{time.time() - t0:.0f}s, results in {args.out}")


if __name__ == "__main__":
    main()
