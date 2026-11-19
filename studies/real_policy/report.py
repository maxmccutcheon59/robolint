"""Step 3: collect the evaluation results into a table, results.json and a chart.

Run:  python report.py            (reads runs/ and data/study.json; needs matplotlib for the chart)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {
    "raw": "Raw (faults included)",
    "robolint_errors": "robolint: drop errors",
    "robolint_all": "robolint: drop errors + warnings",
    "clean_only": "Only clean episodes",
    "no_faults": "No faults (reference)",
}


def success(run_dir: Path) -> tuple[float, int] | None:
    p = run_dir / "eval" / "eval_info.json"
    if not p.exists():
        return None
    info = json.loads(p.read_text())
    overall = info.get("overall") or info.get("aggregated") or {}
    return float(overall["pc_success"]), int(overall.get("n_episodes", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "runs")
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args()
    plan = json.loads((args.runs / "plan.json").read_text())
    study = json.loads((args.data / "study.json").read_text())

    # cell "dataset/arm/seed" -> run name; gather success per (dataset, arm) and seed
    table: dict[str, dict[str, dict[int, float]]] = {}
    n_eval = 0
    for cell, run in plan["cells"].items():
        dataset, arm, seed = cell.split("/")
        r = success(args.runs / run)
        if r is not None:
            table.setdefault(dataset, {}).setdefault(arm, {})[int(seed)] = r[0]
            n_eval = r[1]
    if not table:
        raise SystemExit("no finished runs yet")

    results = {"settings": plan["settings"], "eval_episodes_per_run": n_eval, "datasets": {}}
    lines = []
    for dataset in ("heavy", "light", "clean"):
        if dataset not in table:
            continue
        ds = study["datasets"][dataset]
        n_faulty = len(ds["labels"])
        title = (
            "No faults"
            if dataset == "clean"
            else f"{dataset.capitalize()} faults ({n_faulty} of {study['episodes']} episodes)"
        )
        lines.append(f"\n{title}")
        raw = table[dataset].get("raw", {})
        out = {}
        for arm, by_seed in table[dataset].items():
            v = np.array([by_seed[s] for s in sorted(by_seed)])
            kept = study["episodes"] - (0 if arm == "no_faults" else len(ds["exclude"][arm]))
            row = {"success_pct": float(v.mean()), "per_seed": by_seed, "episodes_trained": kept}
            paired = [by_seed[s] - raw[s] for s in by_seed if s in raw]
            if arm != "raw" and paired:
                row["vs_raw_pct_points"] = float(np.mean(paired))
                row["better_than_raw_in"] = f"{sum(p > 0 for p in paired)}/{len(paired)} seeds"
            out[arm] = row
            diff = (
                f"  {row['vs_raw_pct_points']:+.1f} vs raw ({row['better_than_raw_in']})"
                if "vs_raw_pct_points" in row
                else ""
            )
            sd = f", sd {v.std():.1f}" if len(v) > 1 else ""
            lines.append(f"  {LABELS[arm]:34s} {v.mean():5.1f}% success ({len(v)} seeds{sd}, {kept} episodes){diff}")
        results["datasets"][dataset] = {"arms": out, "detection": ds.get("detection")}
    args.out.write_text(json.dumps(results, indent=2))
    print("\n".join(lines))
    print(f"\n{n_eval} simulated test episodes per run. Results in {args.out}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(pip install matplotlib for the chart)")
        return
    groups = [d for d in ("light", "heavy") if d in results["datasets"]]
    arms = [
        a
        for a in ("raw", "robolint_errors", "robolint_all", "clean_only")
        if any(a in results["datasets"][g]["arms"] for g in groups)
    ]
    colors = ["#9aa0a6", "#1a73e8", "#8ab4f8", "#34a853"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        ys = [results["datasets"][g]["arms"].get(arm, {}).get("success_pct", np.nan) for g in groups]
        ax.bar(np.arange(len(groups)) + (i - (len(arms) - 1) / 2) * w, ys, w, label=LABELS[arm], color=colors[i])
    ref = results["datasets"].get("clean", {}).get("arms", {}).get("no_faults")
    if ref:
        ax.axhline(ref["success_pct"], color="#202124", ls="--", lw=1, label=LABELS["no_faults"])
    ax.set_xticks(np.arange(len(groups)), [f"{g.capitalize()} faults" for g in groups])
    ax.set_ylabel("Task success (%)")
    ax.set_title("ACT on ALOHA sim transfer-cube, by training data")
    ax.legend(fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=150)
    print(f"Chart in {png}")


if __name__ == "__main__":
    main()
