"""Step 2: train and evaluate an ACT policy for every arm of the study (needs an NVIDIA GPU).

Each run is LeRobot's own `lerobot-train` followed by `lerobot-eval`, with the same settings for
every arm; the arms differ only in which episodes are left out (`--dataset.exclude_episodes`).
All arms are evaluated on the same simulated cube placements (same eval seed).

Arms (episodes dropped before training):
  raw               none: train on everything, faults included
  robolint_errors   the episodes robolint flags as errors
  robolint_all      the episodes robolint flags as errors or warnings
  clean_only        every faulty episode (ground truth; the best any filter could do)
  no_faults         trained on the original, unbroken dataset (the reference)
Arms that drop exactly the same episodes share one run.

Plans:
  quick     heavy faults, raw vs robolint_errors, 1 training seed          (2 runs)
  standard  heavy faults, all arms + no_faults, 3 training seeds           (up to 15 runs)
  full      standard + light faults, 3 training seeds                      (up to 27 runs)

Runs that already finished are skipped, so the study can be stopped and restarted at any time.

Run:  python run_study.py --plan quick
      python run_study.py --plan standard
      python run_study.py --smoke          (a few training steps on CPU or GPU: checks the pipeline)
      python run_study.py --plan standard --dry-run   (print the commands without running them)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ARMS = ["raw", "robolint_errors", "robolint_all", "clean_only"]
PLANS = {
    "quick": {"cells": [("heavy", "raw"), ("heavy", "robolint_errors")], "seeds": 1},
    "standard": {"cells": [("heavy", a) for a in ARMS] + [("clean", "no_faults")], "seeds": 3},
    "full": {
        "cells": [("heavy", a) for a in ARMS] + [("light", a) for a in ARMS] + [("clean", "no_faults")],
        "seeds": 3,
    },
}
EVAL_SEED = 100_000  # first cube placement of the evaluation; the same for every arm
ENV_ARGS = ["--env.type=aloha", "--env.task=AlohaTransferCube-v0"]


def exclusions(study: dict, dataset: str, arm: str) -> list[int]:
    ds = study["datasets"][dataset]
    return [] if arm == "no_faults" else list(ds["exclude"][arm])


def plan_runs(study: dict, plan: str, seeds: int | None) -> tuple[list[dict], dict]:
    """Unique runs, and which run each (dataset, arm, seed) cell uses."""
    spec = PLANS[plan]
    n_seeds = seeds or spec["seeds"]
    runs: dict[tuple, dict] = {}
    cells: dict[str, str] = {}
    for dataset, arm in spec["cells"]:
        if dataset not in study["datasets"]:
            raise SystemExit(f"prepare.py did not build the '{dataset}' dataset; rerun it with that mix")
        ex = exclusions(study, dataset, arm)
        for seed in range(n_seeds):
            key = (dataset, tuple(ex), seed)
            if key not in runs:
                runs[key] = {"name": f"{dataset}-{arm}-seed{seed}", "dataset": dataset, "exclude": ex, "seed": seed}
            cells[f"{dataset}/{arm}/{seed}"] = runs[key]["name"]
    return list(runs.values()), cells


def train_cmd(study: dict, run: dict, out: Path, args) -> list[str]:
    ds = study["datasets"][run["dataset"]]
    cmd = [
        "lerobot-train",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.amp).lower()}",
        f"--dataset.repo_id={ds['repo_id']}",
        f"--dataset.root={ds['root']}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--seed={1000 + run['seed']}",
        "--env_eval_freq=0",
        f"--save_freq={args.steps}",
        f"--log_freq={min(200, args.steps)}",
        f"--output_dir={out / 'train'}",
        f"--job_name={run['name']}",
    ]
    if run["exclude"]:
        cmd.append(f"--dataset.exclude_episodes=[{','.join(map(str, run['exclude']))}]")
    return cmd + list(args.train_arg)


def eval_cmd(run_dir: Path, args) -> list[str]:
    cmd = [
        "lerobot-eval",
        f"--policy.path={run_dir / 'train/checkpoints/last/pretrained_model'}",
        *ENV_ARGS,
        f"--eval.n_episodes={args.eval_episodes}",
        f"--eval.batch_size={min(args.eval_batch, args.eval_episodes)}",
        f"--seed={EVAL_SEED}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={str(args.amp).lower()}",
        f"--output_dir={run_dir / 'eval'}",
    ]
    if args.smoke:
        cmd += ["--eval.max_episode_steps=20", "--eval.use_async_envs=false"]
    return cmd


def run_logged(cmd: list[str], log: Path) -> None:
    print("  $ " + " ".join(cmd))
    with log.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
    if proc.returncode:
        tail = log.read_text(errors="replace").splitlines()[-25:]
        print("\n".join("    " + line for line in tail))
        raise SystemExit(f"command failed (exit {proc.returncode}); full log: {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=sorted(PLANS), default="standard")
    ap.add_argument("--data", type=Path, default=HERE / "data", help="folder prepare.py wrote")
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--seeds", type=int, default=None, help="training seeds per arm (default: the plan's)")
    ap.add_argument("--steps", type=int, default=50_000, help="training steps per run")
    ap.add_argument("--batch-size", type=int, default=8, help="8 fits an 8 GB GPU; raise it on a bigger one")
    ap.add_argument("--num-workers", type=int, default=4, help="data loader processes")
    ap.add_argument("--eval-episodes", type=int, default=50, help="simulated test episodes per run")
    ap.add_argument("--eval-batch", type=int, default=10, help="simulator copies run in parallel during eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true", help="mixed precision: faster, less memory")
    ap.add_argument("--train-arg", action="append", default=[], help="extra lerobot-train flag (repeatable)")
    ap.add_argument("--smoke", action="store_true", help="tiny runs (20 steps, 2 short eval episodes) to test setup")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()
    if args.smoke:
        args.plan, args.seeds, args.steps, args.eval_episodes, args.eval_batch = "quick", 1, 20, 2, 1
        args.num_workers = min(args.num_workers, 2)
        args.runs = args.runs.parent / (args.runs.name + "-smoke")
    os.environ.setdefault("MUJOCO_GL", "egl")

    study_path = args.data / "study.json"
    if not study_path.exists():
        raise SystemExit(f"{study_path} not found: run prepare.py first")
    study = json.loads(study_path.read_text())
    runs, cells = plan_runs(study, args.plan, args.seeds)
    args.runs.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ("data", "runs", "dry_run")}
    (args.runs / "plan.json").write_text(json.dumps({"settings": settings, "cells": cells, "runs": runs}, indent=2))

    todo = [r for r in runs if not (args.runs / r["name"] / "eval" / "eval_info.json").exists()]
    print(f"Plan '{args.plan}': {len(runs)} runs ({len(runs) - len(todo)} already done), {args.steps} steps each.")
    if args.dry_run:
        for r in todo:
            d = args.runs / r["name"]
            print(f"\n# {r['name']}  (drops {len(r['exclude'])} episodes)")
            print(" ".join(train_cmd(study, r, d, args)))
            print(" ".join(eval_cmd(d, args)))
        return

    if not shutil.which("lerobot-train"):
        raise SystemExit("lerobot-train not found: activate the study's environment (see README)")
    durations: list[float] = []
    for i, r in enumerate(todo):
        d = args.runs / r["name"]
        print(f"\n[{i + 1}/{len(todo)}] {r['name']}: training on {study['episodes'] - len(r['exclude'])} episodes")
        t0 = time.time()
        if not (d / "train/checkpoints/last/pretrained_model").exists():
            if (d / "train").exists():
                shutil.rmtree(d / "train")  # an interrupted run: start it over
            d.mkdir(parents=True, exist_ok=True)
            run_logged(train_cmd(study, r, d, args), d / "train.log")
        if (d / "eval").exists():
            shutil.rmtree(d / "eval")
        run_logged(eval_cmd(d, args), d / "eval.log")
        durations.append(time.time() - t0)
        info = json.loads((d / "eval/eval_info.json").read_text())
        pc = info.get("overall", info.get("aggregated", {})).get("pc_success")
        left = len(todo) - i - 1
        eta = sum(durations) / len(durations) * left / 3600
        print(f"  success {pc}%  ({durations[-1] / 60:.0f} min; about {eta:.1f} h left for {left} runs)")
    print(f"\nAll runs done. Next: python report.py --runs {args.runs}")


if __name__ == "__main__":
    main()
