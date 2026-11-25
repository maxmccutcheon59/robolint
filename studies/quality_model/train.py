"""Train and evaluate the episode quality model.

Training data: clean real episodes (no robolint warning or error) from public LeRobot datasets,
plus copies of them with realistic faults injected. Evaluation holds out whole datasets, so every
score is on robots and tasks the model never saw.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from inject import CLASSES, FAULTS, inject
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from robolint.quality.features import FEATURES, DatasetContext, episode_features

HERE = Path(__file__).parent
WARN = 2


def load():
    sets = []
    for p in sorted((HERE / "cache").glob("*.pkl")):
        d = pickle.load(open(p, "rb"))
        eps = [e for e in d["episodes"] if len(e["action"]) >= 8 and e["action"].shape[1] >= 1]
        if len(eps) >= 8:
            d["episodes"] = eps
            sets.append(d)
    return sets


def make_mix(d, rate, rng):
    """A dataset like a real one: every clean episode once, plus faulty copies making up `rate`
    of the result. Features are measured against this mix, as they would be on a real upload."""
    clean = [e for e in d["episodes"] if not any(sev >= WARN for _, sev in e["findings"])]
    if len(clean) < 6:
        return [], []
    scale = DatasetContext(d["episodes"]).scale
    n_bad = max(1, int(round(len(clean) * rate / (1 - rate))))
    eps, labels = list(clean), ["clean"] * len(clean)
    for _ in range(n_bad):
        f = FAULTS[rng.integers(len(FAULTS))]
        e, other = clean[rng.integers(len(clean))], clean[rng.integers(len(clean))]
        eps.append(inject(e, f, rng, scale, other))
        labels.append(f)
    return eps, labels


def build(sets, rates=(0.1, 0.25, 0.4), seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for gi, d in enumerate(sets):
        for rate in rates:
            eps, labels = make_mix(d, rate, rng)
            if not eps:
                continue
            ctx = DatasetContext(eps)
            for e, lab in zip(eps, labels, strict=True):
                X.append(episode_features(e, ctx))
                y.append(CLASSES.index(lab))
                g.append(gi)
    return np.array(X), np.array(y), np.array(g)


def model():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0
    )


def evaluate(X, y, g, names):
    p_bad = np.zeros(len(y))
    pred = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=min(6, len(set(g)))).split(X, y, g):
        m = model().fit(X[tr], y[tr])
        pr = m.predict_proba(X[te])
        cols = list(m.classes_)
        p_bad[te] = 1 - pr[:, cols.index(0)]
        pred[te] = np.array(cols)[pr.argmax(1)]
    bad = y > 0
    auc = roc_auc_score(bad, p_bad)
    thr = float(np.quantile(p_bad[~bad], 0.98))  # 2% of clean episodes flagged
    rec = {f: float((p_bad[y == i] > thr).mean()) for i, f in enumerate(CLASSES) if i and (y == i).any()}
    per_ds = {}
    for gi in sorted(set(g)):
        mk = g == gi
        if bad[mk].any() and (~bad[mk]).any():
            per_ds[names[gi]] = round(roc_auc_score(bad[mk], p_bad[mk]), 3)
    typed = float((pred[bad] == y[bad]).mean())
    return {
        "auc": round(auc, 4),
        "threshold_at_2pct_fpr": thr,
        "recall_at_2pct_fpr": rec,
        "overall_recall": float((p_bad[bad] > thr).mean()),
        "fault_type_accuracy": typed,
        "per_dataset_auc": per_ds,
        "n": dict(Counter(CLASSES[i] for i in y)),
    }


if __name__ == "__main__":
    sets = load()
    names = [d["source"] for d in sets]
    print(len(sets), "datasets:", ", ".join(names))
    X, y, g = build(sets)
    print("samples", X.shape)
    res = evaluate(X, y, g, names)
    print(json.dumps(res, indent=1))
    m = model().fit(X, y)
    thr = res["threshold_at_2pct_fpr"]
    pickle.dump(
        {"model": m, "features": FEATURES, "classes": CLASSES, "threshold": thr}, open(HERE / "model.pkl", "wb")
    )
    json.dump({**res, "datasets": names}, open(HERE / "results.json", "w"), indent=1)
