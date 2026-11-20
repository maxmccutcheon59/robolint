"""A corpus of synthetic LeRobot datasets covering every fault the test generator can inject,
both layouts, videos, and the loader edge cases (null rows, odd widths, missing columns,
shuffled rows, data split over several files). Used to prove a change leaves results unchanged.

    python benchmarks/corpus.py OUT_DIR [--videos]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synth import make_dataset  # noqa: E402

EPISODE_FAULTS = [
    "short", "truncated", "gripper_still", "dead", "idle", "hold", "spike", "rest_limit", "one_arm_rest",
    "binary_gripper", "hard_limit", "regrasp", "state_lag", "state_unrelated", "state_frozen", "nan",
    "dropped", "nonmono", "jitter", "fi_gap", "length_meta", "bad_task", "dup_of:1",
]  # fmt: skip
VIDEO_FAULTS = ["black_video", "frozen_video", "video_shift", "short_video", "missing_video"]


def _split_and_shuffle(root: Path, parts: int, seed: int, shuffle: bool) -> None:
    """Rewrite a v3 dataset's single data file as several files, optionally with rows shuffled."""
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f)
    idx = np.arange(t.num_rows)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    f.unlink()
    for i, chunk in enumerate(np.array_split(idx, parts)):
        pq.write_table(t.take(pa.array(chunk)), root / "data" / "chunk-000" / f"file-{i:03d}.parquet")


def _corrupt(root: Path, kind: str) -> None:
    files = sorted((root / "data").rglob("*.parquet"))
    f = files[len(files) // 2]
    t = pq.read_table(f)
    if kind == "null_action":
        col = t.column("action").to_pylist()
        col[3] = None
        t = t.set_column(t.column_names.index("action"), "action", pa.array(col, type=pa.list_(pa.float32())))
    elif kind == "odd_width":
        col = t.column("action").to_pylist()
        col[2] = col[2][:-1]
        t = t.set_column(t.column_names.index("action"), "action", pa.array(col, type=pa.list_(pa.float32())))
    elif kind == "missing_state":
        t = t.drop_columns(["observation.state"])
    elif kind == "null_episode":
        col = t.column("episode_index").to_pylist()
        col[1] = None
        t = t.set_column(t.column_names.index("episode_index"), "episode_index", pa.array(col, type=pa.int64()))
    elif kind == "null_frame_index":
        col = t.column("frame_index").to_pylist()
        col[4] = None
        t = t.set_column(t.column_names.index("frame_index"), "frame_index", pa.array(col, type=pa.int64()))
    pq.write_table(t, f)


def build(out: Path, videos: bool = False) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    made: list[Path] = []

    def mk(name: str, **kw) -> Path:
        p = make_dataset(out / name, **kw)
        made.append(p)
        return p

    for version in ("v2.1", "v3.0"):
        mk(f"clean-{version}", version=version, n_eps=12)
        for i, fault in enumerate(EPISODE_FAULTS):
            mk(
                f"{fault.replace(':', '_')}-{version}",
                version=version,
                n_eps=10,
                seed=i,
                faults={2: [fault], 6: [fault]},
            )
        mixed = {int(e): list(rng.choice(EPISODE_FAULTS, size=2, replace=False)) for e in rng.choice(30, 12, False)}
        mixed = {e: [f for f in fs if not f.startswith("dup_of") or e > 1] for e, fs in mixed.items()}
        mk(f"mixed-{version}", version=version, n_eps=30, seed=3, faults=mixed, dataset_faults=["count_mismatch"])
        mk(f"vague-{version}", version=version, n_eps=6, dataset_faults=["vague_task"])
        mk(
            f"grasp-{version}",
            version=version,
            n_eps=14,
            grasp_object=True,
            faults={3: ["missed_grasp"], 8: ["regrasp"]},
        )
        mk(
            f"pick-{version}",
            version=version,
            n_eps=14,
            seed=5,
            faults={e: ["pick_place"] for e in range(14)} | {4: ["pick_place_missed"], 9: ["pick_place", "soft_grasp"]},
        )
        mk(f"dims4-{version}", version=version, n_eps=8, dims=4)
        mk(f"bigfps-{version}", version=version, n_eps=8, fps=50, length=300, faults={1: ["state_lag"], 5: ["spike"]})
        if videos:
            vf = {1: ["black_video"], 3: ["frozen_video"], 5: ["video_shift"], 7: ["short_video"]}
            if version == "v2.1":
                vf[9] = ["missing_video"]
            mk(f"video-{version}", version=version, n_eps=12, length=120, videos=True, faults=vf)
    # Loader edge cases (v3 layout, several files, shuffled rows, corrupted columns)
    for kind in ("split", "shuffle", "null_action", "odd_width", "missing_state", "null_episode", "null_frame_index"):
        p = mk(f"loader-{kind}", n_eps=12, seed=11, faults={4: ["spike"], 7: ["dead"]})
        _split_and_shuffle(p, 3, seed=1, shuffle=kind != "split")
        if kind not in ("split", "shuffle"):
            _corrupt(p, kind)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--videos", action="store_true")
    a = ap.parse_args()
    for p in build(a.out, a.videos):
        print(p)
