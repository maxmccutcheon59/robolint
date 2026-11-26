"""Eidran-01, experiment 3: real robot datasets with naturally occurring faults.

For public LeRobot datasets where robolint (rules + quality model) flags real problem episodes,
train a behaviour-cloning policy three ways and compare how well it imitates held-out CLEAN demos:
  all       - every training episode, flagged ones included
  filtered  - flagged episodes removed
  random    - the same number of episodes removed at random (control)
Metric: action-chunk prediction error (MSE, normalised) on held-out clean episodes. This is an
offline proxy: real-robot task success needs the physical robot. Lower is better.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/claude/qmodel")
from robolint.quality import score_dataset  # noqa: E402

torch.set_num_threads(2)
OUT = Path(__file__).parent
CACHE = Path("/home/claude/qmodel/cache")
SETS = ["Embodied-AI-6/dataset", "adityx23/icl-dataset", "jchandru08/so100_press_red_button",
        "kmg0620/so101_bomb_pick_place_v1", "mliu48/telearms-yam-cube-v4", "BitRobot/HIW-500-LeRobot"]
H, CH = 2, 8


def xy(eps):
    X, Y = [], []
    for e in eps:
        s, a = e["state"], e["action"]
        T = len(a)
        for t in range(T):
            X.append(np.concatenate([s[max(0, t - k)] for k in reversed(range(H))]))
            Y.append(np.stack([a[min(T - 1, t + k)] for k in range(CH)]).ravel())
    return np.array(X, np.float32), np.array(Y, np.float32)


def fit_eval(train_eps, test_eps, seed, steps=2500):
    torch.manual_seed(seed)
    X, Y = xy(train_eps)
    Xt_, Yt_ = xy(test_eps)
    xm, xs, ym, ys = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    # normalise test with the clean-test statistics' scale so every condition is scored on the same footing
    tm, ts = Yt_.mean(0), Yt_.std(0) + 1e-6
    Xn, Yn = torch.tensor((X - xm) / xs), torch.tensor((Y - ym) / ys)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(Xn), (256,), generator=g)
        loss = nn.functional.mse_loss(net(Xn[b]), Yn[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(torch.tensor((Xt_ - xm) / xs)).numpy() * ys + ym
    return float((((pred - Yt_) / ts) ** 2).mean())


def flags(d):
    eps = d["episodes"]
    ds = SimpleNamespace(episodes=[SimpleNamespace(index=i, frames={"action": e["action"], "observation.state": e["state"]})
                                   for i, e in enumerate(eps)])
    q = {x.episode: x.flagged for x in score_dataset(ds)}
    return {i for i, e in enumerate(eps) if any(s >= 2 for _, s in e["findings"]) or q.get(i, False)}


def main(splits=3):
    t0 = time.time()
    res = {}
    for src in SETS:
        d = pickle.load(open(CACHE / (src.replace("/", "__") + ".pkl"), "rb"))
        eps = [e for e in d["episodes"] if e.get("state") is not None and e["state"].shape == e["action"].shape and len(e["action"]) >= 10]
        d["episodes"] = eps
        fl = flags(d)
        clean = [i for i in range(len(eps)) if i not in fl]
        if len(fl) < 3 or len(clean) < 12:
            print(src, "skipped: flagged", len(fl), "clean", len(clean)); continue
        rows = []
        for sp in range(splits):
            rng = np.random.default_rng(sp)
            test = set(rng.choice(clean, max(4, len(clean) // 4), replace=False).tolist())
            pool = [i for i in range(len(eps)) if i not in test]
            fpool = [i for i in pool if i not in fl]
            n_drop = len(pool) - len(fpool)
            rpool = [i for i in pool if i not in set(rng.choice(pool, n_drop, replace=False).tolist())]
            te = [eps[i] for i in test]
            r = {name: fit_eval([eps[i] for i in idx], te, sp) for name, idx in
                 (("all", pool), ("filtered", fpool), ("random", rpool))}
            rows.append(r)
            print(src, sp, {k: round(v, 4) for k, v in r.items()}, f"{time.time()-t0:.0f}s", flush=True)
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res[src] = {"episodes": len(eps), "flagged": len(fl), "test_clean_eps": len(test), "mean_error": m,
                    "filtered_vs_all_pct": 100 * (m["filtered"] - m["all"]) / m["all"],
                    "random_vs_all_pct": 100 * (m["random"] - m["all"]) / m["all"], "splits": rows}
        json.dump(res, open(OUT / "results_real.json", "w"), indent=1)


if __name__ == "__main__":
    main()
