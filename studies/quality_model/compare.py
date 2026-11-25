"""Head-to-head on held-out datasets: robolint rules vs the quality model vs both together.

For each dataset: train the model on every OTHER dataset, then build a test mix of this dataset's
clean episodes plus one faulty copy of each, and ask both detectors which episodes are bad.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from rules import rule_flags
from sklearn.model_selection import GroupKFold
from train import WARN, build, load, make_mix, model

from robolint.quality.features import DatasetContext, episode_features

sets = load()
names = [d["source"] for d in sets]
X, y, g = build(sets)
rng = np.random.default_rng(1)
hits = defaultdict(lambda: {"n": 0, "rules": 0, "model": 0, "either": 0})
fp = {"n": 0, "rules": 0, "model": 0, "either": 0}
folds = list(GroupKFold(n_splits=6).split(X, y, g))
for k, (tr_idx, _) in enumerate(folds):
    tr = np.zeros(len(y), bool)
    tr[tr_idx] = True
    held = sorted(set(g[~tr]))
    m = model().fit(X[tr], y[tr])
    # threshold: 2% false alarms on clean episodes of datasets this model never trained on
    oof = []
    for itr, ite in GroupKFold(n_splits=3).split(X[tr], y[tr], g[tr]):
        mi = model().fit(X[tr][itr], y[tr][itr])
        keep = y[tr][ite] == 0
        oof.append(1 - mi.predict_proba(X[tr][ite][keep])[:, list(mi.classes_).index(0)])
    thr = float(np.quantile(np.concatenate(oof), 0.98))
    for gi in held:
        d = sets[gi]
        mix, labels = make_mix(d, 0.25, rng)
        if not mix:
            continue
        ctx = DatasetContext(mix)
        Xm = np.stack([episode_features(e, ctx) for e in mix])
        pb = 1 - m.predict_proba(Xm)[:, list(m.classes_).index(0)]
        rf = rule_flags(mix, d["fps"] or 30)
        for i, lab in enumerate(labels):
            r = rf.get(i, 0) >= WARN
            md = pb[i] > thr
            tgt = fp if lab == "clean" else hits[lab]
            tgt["n"] += 1
            tgt["rules"] += r
            tgt["model"] += md
            tgt["either"] += r or md
    print("fold", k, "done", [names[i] for i in held], flush=True)

tot = {k: sum(h[k] for h in hits.values()) for k in ("n", "rules", "model", "either")}
out = {
    "faults": {f: {k: (v if k == "n" else round(v / h["n"], 3)) for k, v in h.items()} for f, h in hits.items()},
    "all_faults": {k: (v if k == "n" else round(v / tot["n"], 3)) for k, v in tot.items()},
    "false_alarms_on_clean": {k: (v if k == "n" else round(v / fp["n"], 3)) for k, v in fp.items()},
}
print(json.dumps(out, indent=1))
json.dump(out, open("compare.json", "w"), indent=1)
