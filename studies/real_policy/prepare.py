"""Step 1: build the study's datasets and run robolint on them.

  1. Get the clean source dataset (lerobot/aloha_sim_transfer_cube_human by default: 50 human
     teleop demos of the ALOHA transfer-cube task, all successful, robolint grade A).
  2. Write three LeRobot v3 datasets through LeRobot's own writer, so every arm of the study
     trains on data encoded the same way:
       data/clean   the 50 demos unchanged (the no-fault reference)
       data/light   ~12% of episodes given a recording fault
       data/heavy   ~36% of episodes given a recording fault
     Faults (faults.py): dead episodes, frozen state readings, action spikes, idle padding and
     failed grasps (re-simulated so the grippers close on air).
  3. Run robolint, unchanged, on each dataset and record which episodes it flags.
  4. Write data/study.json: per dataset, the ground-truth labels and the episodes each arm drops.

Run:  python prepare.py                 (downloads ~70 MB, about 10-20 minutes on a laptop CPU)
      python prepare.py --standin 12    (no download: scripted stand-in episodes, for smoke tests)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))  # robolint from this checkout, unchanged

os.environ.setdefault("MUJOCO_GL", "egl")

import faults  # noqa: E402

SOURCE = "lerobot/aloha_sim_transfer_cube_human"
CAMERA = "observation.images.top"
TASK = "Pick up the cube with the right arm and transfer it to the left arm."


# --- reading the source dataset ----------------------------------------------------------------


def _table(paths):
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa.concat_tables([pq.read_table(p) for p in sorted(paths)]).to_pydict()


def read_source(root: Path):
    """Yield (episode_index, {"state", "action", "images"}) for every episode of a LeRobot v3 dataset."""
    import av

    info = json.loads((root / "meta/info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise SystemExit(f"{root}: expected a LeRobot v3 dataset, got {info.get('codebase_version')}")
    data = _table((root / "data").rglob("*.parquet"))
    meta = _table((root / "meta/episodes").rglob("*.parquet"))
    ep_col = np.asarray(data["episode_index"])
    state = np.asarray(data["observation.state"], dtype=np.float32)
    action = np.asarray(data["action"], dtype=np.float32)
    fps = float(info["fps"])
    for row in range(len(meta["episode_index"])):
        ep = int(meta["episode_index"][row])
        idx = np.flatnonzero(ep_col == ep)
        chunk, file = meta[f"videos/{CAMERA}/chunk_index"][row], meta[f"videos/{CAMERA}/file_index"][row]
        t0 = float(meta[f"videos/{CAMERA}/from_timestamp"][row])
        path = root / info["video_path"].format(video_key=CAMERA, chunk_index=chunk, file_index=file)
        frames = []
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            c.seek(int(t0 / stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in c.decode(stream):
                if frame.time is not None and frame.time < t0 - 0.5 / fps:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) == len(idx):
                    break
        if len(frames) != len(idx):
            raise SystemExit(f"episode {ep}: decoded {len(frames)} frames, expected {len(idx)}")
        yield ep, {"state": state[idx], "action": action[idx], "images": np.stack(frames)}


def get_source(args) -> Path:
    if args.standin:
        root = args.out / "standin"
        if not (root / "meta/info.json").exists():
            write_standin(root, args.standin, args.seed)
        return root
    if Path(args.source).exists():
        return Path(args.source)
    from huggingface_hub import snapshot_download

    root = args.out / "source"
    print(f"Downloading {args.source} ...")
    snapshot_download(args.source, repo_type="dataset", local_dir=root)
    return root


# --- writing LeRobot datasets ------------------------------------------------------------------


def features(names: list[str]) -> dict:
    vec = {"dtype": "float32", "shape": [len(names)], "names": {"motors": names}}
    return {
        CAMERA: {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"]},
        "observation.state": dict(vec),
        "action": dict(vec),
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }


class Writer:
    """A LeRobot v3 dataset written with LeRobot's own writer."""

    def __init__(self, root: Path, names: list[str], fps: int):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if root.exists():
            shutil.rmtree(root)
        self.ds = LeRobotDataset.create(
            repo_id=f"robolint-study/{root.name}",
            fps=fps,
            features=features(names),
            root=root,
            robot_type="aloha",
            use_videos=True,
            image_writer_threads=4,
        )

    def add(self, ep: dict) -> None:
        n = len(ep["action"])
        for t in range(n):
            self.ds.add_frame(
                {
                    CAMERA: ep["images"][t],
                    "observation.state": ep["state"][t].astype(np.float32),
                    "action": ep["action"][t].astype(np.float32),
                    "next.done": np.array([t == n - 1]),
                    "task": TASK,
                }
            )
        self.ds.save_episode()

    def close(self) -> None:
        self.ds.finalize()


MOTORS = [
    f"{side}_{j}"
    for side in ("left", "right")
    for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
]


def write_standin(root: Path, n: int, seed: int) -> None:
    """Scripted pick-and-lift episodes in the transfer-cube scene, in the source dataset's schema."""
    import aloha_sim

    print(f"Writing {n} scripted stand-in episodes to {root} (smoke tests only) ...")
    rng = np.random.default_rng(seed + 7)
    w = Writer(root, MOTORS, aloha_sim.FPS)
    for i in range(n):
        ep = aloha_sim.scripted_episode(rng)
        print(f"  stand-in episode {i}: {'lifted the cube' if ep['success'] else 'missed'}")
        w.add(ep)
    w.close()


# --- robolint ----------------------------------------------------------------------------------


def run_robolint(root: Path) -> dict:
    from robolint import Severity, lint

    result = lint(root)
    (root / "robolint.json").write_text(json.dumps(result.to_dict(), indent=2))
    return {
        "errors": result.bad_episodes(Severity.ERROR),
        "errors_or_warnings": result.bad_episodes(Severity.WARNING),
        "score": result.scores.dataset_score,
        "grade": result.scores.grade,
        "codes": {
            str(ep): sorted({f.code for f in result.episode_findings(ep) if f.severity >= Severity.WARNING})
            for ep in result.bad_episodes(Severity.WARNING)
        },
    }


def detection_table(labels: dict[int, str], lint_out: dict, n: int) -> dict:
    err, warn = set(lint_out["errors"]), set(lint_out["errors_or_warnings"])
    table = {}
    for name in ("clean", *faults.FAULTS):
        eps = [i for i in range(n) if labels.get(i, "clean") == name]
        if eps:
            table[name] = {
                "episodes": len(eps),
                "flagged_error": sum(i in err for i in eps),
                "flagged_error_or_warning": sum(i in warn for i in eps),
            }
    return table


# --- main --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE, help="Hub dataset id or local path of a clean LeRobot v3 dataset")
    ap.add_argument("--standin", type=int, default=0, metavar="N", help="use N scripted stand-in episodes instead")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--seed", type=int, default=0, help="which episodes get which fault")
    ap.add_argument("--mixes", default="light,heavy", help="comma-separated fault mixes to build")
    args = ap.parse_args()
    t_start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    src = get_source(args)
    info = json.loads((src / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    names = names["motors"] if isinstance(names, dict) else names
    fps = int(Fraction(info["fps"]).limit_denominator())
    n = int(info["total_episodes"])
    mixes = [m for m in args.mixes.split(",") if m]
    labels = faults.assign(n, args.seed)

    # Action ranges of the clean data set the size of injected spikes.
    import pyarrow.parquet as pq

    acts = np.concatenate(
        [
            np.asarray(pq.read_table(p, columns=["action"])["action"].to_pylist())
            for p in (src / "data").rglob("*.parquet")
        ]
    )
    ranges = acts.max(0) - acts.min(0)

    writers = {"clean": Writer(args.out / "clean", names, fps)}
    for m in mixes:
        writers[m] = Writer(args.out / m, names, fps)
    print(f"Writing clean + {', '.join(mixes)} from {n} source episodes ...")
    for ep_idx, ep in read_source(src):
        writers["clean"].add(ep)
        cache: dict[str, dict] = {}  # the light mix reuses the heavy mix's faulty episode as is
        for m in mixes:
            fault = labels[m].get(ep_idx)
            if fault is None:
                writers[m].add(ep)
                continue
            if fault not in cache:
                rng = np.random.default_rng([args.seed, ep_idx])
                cache[fault] = faults.apply(fault, ep, rng, ranges)
            writers[m].add(cache[fault])
        tags = ", ".join(f"{m}: {labels[m][ep_idx]}" for m in mixes if ep_idx in labels[m])
        print(f"  episode {ep_idx:3d}  {tags or 'clean'}")
    for w in writers.values():
        w.close()

    study = {"source": str(args.source if not args.standin else "standin"), "episodes": n, "fps": fps, "datasets": {}}
    for name in ["clean", *mixes]:
        root = args.out / name
        lab = {} if name == "clean" else labels[name]
        lint_out = run_robolint(root)
        study["datasets"][name] = {
            "root": str(root.resolve()),
            "repo_id": f"robolint-study/{name}",
            "labels": {str(k): v for k, v in sorted(lab.items())},
            "exclude": {
                "raw": [],
                "robolint_errors": lint_out["errors"],
                "robolint_all": lint_out["errors_or_warnings"],
                "clean_only": sorted(lab),
            },
            "robolint": lint_out,
            "detection": detection_table(lab, lint_out, n),
        }
        det = study["datasets"][name]["detection"]
        print(f"\n{name}: robolint {lint_out['score']} ({lint_out['grade']})")
        for label, d in det.items():
            print(
                f"  {label:13s} {d['episodes']:3d} episodes  flagged as error {d['flagged_error']:3d}  "
                f"as error or warning {d['flagged_error_or_warning']:3d}"
            )
    (args.out / "study.json").write_text(json.dumps(study, indent=2))
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Plan inputs in {args.out / 'study.json'}")


if __name__ == "__main__":
    main()
