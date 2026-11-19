"""Peak memory and run time of `robolint check` on large synthetic datasets.

    python benchmarks/scale.py --frames 500000 2000000 10000000 --work /tmp/robolint-bench \
        [--old-src PATH_TO_OLDER_CHECKOUT/src] [--cache-mb 128]

Writes LeRobot v3.0 datasets shaped like a bimanual (ALOHA-style) recording: 14-dimensional
action and state, 50 fps, 600-frame episodes, data split into ~100 MB parquet files, with a few
dead and glitchy episodes. Each run is a fresh process; peak RSS is the process's maximum
resident set size. With --old-src, the same datasets are also checked by that source tree, and
the findings of the two are compared.
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

DIMS = 14
FPS = 50
EP_LEN = 600
EPS_PER_FILE = 800  # 480k rows, about 100 MB of parquet, like LeRobot v3's default file size
NAMES = [f"{side}_{j}" for side in ("left", "right") for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]  # fmt: skip


def _list(a: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1)), a.shape[1])


def generate(root: Path, frames: int, seed: int = 0) -> Path:
    if (root / "meta" / "info.json").exists():
        return root
    rng = np.random.default_rng(seed)
    n_eps = frames // EP_LEN
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    t = (np.arange(EP_LEN) / FPS).astype(np.float32)
    ep_rows = []
    for fi, start in enumerate(range(0, n_eps, EPS_PER_FILE)):
        eps = np.arange(start, min(n_eps, start + EPS_PER_FILE))
        k = len(eps)
        c = rng.uniform(-0.5, 0.5, (k, 1, DIMS)).astype(np.float32)
        amp = rng.uniform(0.2, 0.8, (k, 1, DIMS)).astype(np.float32)
        f = rng.uniform(0.1, 0.5, (k, 1, DIMS)).astype(np.float32)
        ph = rng.uniform(0, 6, (k, 1, DIMS)).astype(np.float32)
        a = c + amp * np.sin(2 * np.pi * f * t[None, :, None] + ph)
        a += rng.normal(0, 0.002, a.shape).astype(np.float32)
        for i, e in enumerate(eps):
            if e % 997 == 5:
                a[i] = a[i, :1]  # dead episode
            if e % 499 == 7:
                a[i, EP_LEN // 2, 1] += 3.0  # teleop glitch
        s = np.concatenate([a[:, :2], a[:, :-2]], axis=1) + rng.normal(0, 0.002, a.shape).astype(np.float32)
        n = k * EP_LEN
        g0 = start * EP_LEN
        table = pa.table(
            {
                "action": _list(a.reshape(n, DIMS)),
                "observation.state": _list(s.reshape(n, DIMS)),
                "timestamp": pa.array(np.tile(t, k)),
                "frame_index": pa.array(np.tile(np.arange(EP_LEN, dtype=np.int64), k)),
                "episode_index": pa.array(np.repeat(eps.astype(np.int64), EP_LEN)),
                "index": pa.array(np.arange(g0, g0 + n, dtype=np.int64)),
                "task_index": pa.array(np.zeros(n, np.int64)),
            }
        )
        pq.write_table(table, data_dir / f"file-{fi:03d}.parquet")
        for e in eps:
            ep_rows.append({"episode_index": int(e), "tasks": ["transfer the cube"], "length": EP_LEN,
                            "data/chunk_index": 0, "data/file_index": fi})  # fmt: skip
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": [0], "task": ["transfer the cube"]}), root / "meta" / "tasks.parquet")
    vec = {"dtype": "float32", "shape": [DIMS], "names": NAMES}
    scalar = {"shape": [1], "names": None}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha",
        "total_episodes": n_eps,
        "total_frames": n_eps * EP_LEN,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "action": vec,
            "observation.state": vec,
            "timestamp": {"dtype": "float32", **scalar},
            "frame_index": {"dtype": "int64", **scalar},
            "episode_index": {"dtype": "int64", **scalar},
            "index": {"dtype": "int64", **scalar},
            "task_index": {"dtype": "int64", **scalar},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return root


RUN = """
import json, resource, sys, time
sys.path.insert(0, {src!r})
from robolint import lint
t0 = time.perf_counter()
r = lint({path!r}, **{kwargs!r})
dt = time.perf_counter() - t0
findings = json.dumps(r.to_dict()["findings"], sort_keys=True)
print(json.dumps({{"seconds": dt, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "score": r.scores.dataset_score, "findings": len(r.findings), "digest": {digest}}}))
"""


def measure(src: str, path: Path, **kwargs) -> dict:
    code = RUN.format(
        src=src, path=str(path), kwargs=kwargs, digest="__import__('hashlib').sha256(findings.encode()).hexdigest()"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-1:] or [f"exit code {p.returncode}"]
        return {"error": tail[0] if p.returncode > 0 else f"killed by signal {-p.returncode} (out of memory?)"}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[500_000, 2_000_000, 10_000_000])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--old-src", help="another robolint source tree to compare against")
    ap.add_argument("--cache-mb", type=float, help="cache budget for this tree's run (default: robolint's default)")
    a = ap.parse_args()
    new_src = str(Path(__file__).resolve().parents[1] / "src")
    rows = []
    for n in a.frames:
        ds = generate(a.work / f"aloha-{n}", n)
        size = sum(f.stat().st_size for f in (ds / "data").rglob("*.parquet")) / 1e6
        res = {
            "frames": n,
            "parquet_mb": round(size),
            "new": measure(new_src, ds, **({"cache_mb": a.cache_mb} if a.cache_mb is not None else {})),
        }
        if a.old_src:
            res["old"] = measure(a.old_src, ds)
            if "digest" in res["old"] and "digest" in res["new"]:
                res["identical_findings"] = res["old"]["digest"] == res["new"]["digest"]
        print(json.dumps(res), flush=True)
        rows.append(res)

    def cell(r: dict) -> str:
        return r.get("error") or f"{r['peak_rss_mb']:,.0f} MB, {r['seconds']:.0f} s"

    print("\n| Frames | Parquet | Before | After | Same findings |\n|---|---|---|---|---|")
    for r in rows:
        old = cell(r["old"]) if "old" in r else "-"
        same = {True: "yes", False: "NO"}.get(r.get("identical_findings"), "-")
        print(f"| {r['frames']:,} | {r['parquet_mb']:,} MB | {old} | {cell(r['new'])} | {same} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
