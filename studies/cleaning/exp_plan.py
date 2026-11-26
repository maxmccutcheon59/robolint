"""Experiment 4: fault-specific cleaning plan vs dropping everything flagged, on real datasets.

Conditions (same held-out clean test episodes and splits as experiment 3):
  all          - every training episode
  drop_flagged - every flagged episode removed (the naive approach)
  plan         - robolint's cleaning plan: remove broken recordings, trim idle edges, keep the rest
  random       - the same number of episodes the plan removes, removed at random
"""

import json
import sys
from types import SimpleNamespace

import numpy as np

import exp_real as R
from robolint import lint
from robolint.cleaning import plan as make_plan
from robolint.quality import score_dataset

res = {}
for src in ["kmg0620/so101_bomb_pick_place_v1", "jchandru08/so100_press_red_button",
            "mliu48/telearms-yam-cube-v4", "Embodied-AI-6/dataset", "adityx23/icl-dataset"]:
    r = lint(src, max_episodes=120)
    eps, idx = [], []
    for ep in r.dataset.episodes:
        f = ep.frames
        if "observation.state" not in f:
            continue
        a, s = np.asarray(f["action"], np.float32), np.asarray(f["observation.state"], np.float32)
        if a.shape != s.shape or len(a) < 10:
            continue
        eps.append({"action": a, "state": s}); idx.append(ep.index)
    pos = {e: i for i, e in enumerate(idx)}
    q = score_dataset(SimpleNamespace(episodes=[SimpleNamespace(index=idx[i], frames={"action": e["action"], "observation.state": e["state"]})
                                                for i, e in enumerate(eps)]))
    plans = {p.episode: p for p in make_plan(r, quality=q)}
    flagged = {pos[e] for e, p in plans.items() if e in pos and p.actions}
    removed = {pos[e] for e, p in plans.items() if e in pos and p.decision == "remove"}
    trims = {pos[e]: p.trim for e, p in plans.items() if e in pos and p.trim and p.decision == "keep"}
    clean = [i for i in range(len(eps)) if i not in flagged]
    if len(clean) < 12:
        print(src, "skip", len(clean)); continue

    def trimmed(i):
        e = eps[i]
        if i not in trims:
            return e
        a, b = trims[i]
        sl = slice(a, len(e["action"]) - b if b else None)
        return {"action": e["action"][sl], "state": e["state"][sl]} if len(e["action"][sl]) >= 10 else e

    rows = []
    for sp in range(3):
        rng = np.random.default_rng(sp)
        test = set(rng.choice(clean, max(4, len(clean) // 4), replace=False).tolist())
        pool = [i for i in range(len(eps)) if i not in test]
        te = [eps[i] for i in test]
        n_rm = sum(i in removed for i in pool)
        rnd = set(rng.choice(pool, n_rm, replace=False).tolist()) if n_rm else set()
        conds = {"all": [eps[i] for i in pool],
                 "drop_flagged": [eps[i] for i in pool if i not in flagged],
                 "plan": [trimmed(i) for i in pool if i not in removed],
                 "random": [eps[i] for i in pool if i not in rnd]}
        rows.append({k: R.fit_eval(v, te, sp) for k, v in conds.items()})
    m = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
    res[src] = {"episodes": len(eps), "flagged": len(flagged), "plan_removes": len(removed), "plan_trims": len(trims),
                "mean_error": m, **{f"{k}_vs_all_pct": 100 * (m[k] - m["all"]) / m["all"] for k in ("drop_flagged", "plan", "random")}}
    print(src, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in res[src].items() if k != "mean_error"}, flush=True)
    json.dump(res, open("results_plan.json", "w"), indent=1)
