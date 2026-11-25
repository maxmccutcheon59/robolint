"""Run the trained model on the REAL episodes (as published) and compare with robolint's rules."""

import json
import pickle

import numpy as np
from train import WARN, load

from robolint.quality.features import DatasetContext, episode_features

M = pickle.load(open("quality_model.pkl", "rb"))
m, thr, classes = M["model"], M["threshold"], M["classes"]
rows, summ = [], []
for d in load():
    eps = d["episodes"]
    ctx = DatasetContext(eps)
    X = np.stack([episode_features(e, ctx) for e in eps])
    pr = m.predict_proba(X)
    cols = list(m.classes_)
    pb = 1 - pr[:, cols.index(0)]
    worst = [max([s for _, s in e["findings"]] or [0]) for e in eps]
    rule_bad = np.array([w >= WARN for w in worst])
    model_bad = pb > thr
    summ.append(
        {
            "dataset": d["source"],
            "episodes": len(eps),
            "rules": int(rule_bad.sum()),
            "model": int(model_bad.sum()),
            "both": int((rule_bad & model_bad).sum()),
            "model_only": int((model_bad & ~rule_bad).sum()),
            "rules_only": int((rule_bad & ~model_bad).sum()),
        }
    )
    for i in np.flatnonzero(model_bad & ~rule_bad):
        p = pr[i].copy()
        p[cols.index(0)] = 0
        rows.append(
            {
                "dataset": d["source"],
                "episode": eps[i]["index"],
                "p_bad": round(float(pb[i]), 3),
                "likely": classes[cols[int(p.argmax())]],
                "frames": len(eps[i]["action"]),
            }
        )
json.dump(
    {"summary": summ, "model_only": sorted(rows, key=lambda r: -r["p_bad"])}, open("scan_real.json", "w"), indent=1
)
t = {k: sum(s[k] for s in summ) for k in ("episodes", "rules", "model", "both", "model_only", "rules_only")}
print(t)
for s in summ:
    print(s)
