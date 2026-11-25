"""Learned episode quality model (optional; needs scikit-learn: pip install 'robolint[model]').

A gradient-boosted classifier over robot-agnostic per-episode features, trained on clean episodes
from 26 public LeRobot datasets plus copies with realistic faults injected (dead, frozen follower,
glitches, idle padding, false starts, aborted demos, stacked recordings, stutter, noise,
hesitation, anomalous motion). It complements the rule checks: on held-out datasets, rules plus
model caught 82% of injected faults versus 55% for rules alone, at 1.6% false alarms on clean
episodes. See studies/quality_model/README.md.
"""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import DatasetContext, episode_features

MODEL_PATH = Path(__file__).with_name("model.pkl.gz")


@dataclass
class EpisodeQuality:
    episode: int
    p_bad: float  # model's probability the episode has a fault
    flagged: bool  # p_bad above the threshold set for 2% false alarms on held-out clean data
    likely_fault: str | None  # the most probable fault type when flagged


def _load():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("the quality model needs scikit-learn: pip install 'robolint[model]'") from e
    # The model file ships inside this package (it is not user input), so unpickling it is safe.
    with gzip.open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_dataset(dataset) -> list[EpisodeQuality]:
    """Score every episode of a loaded robolint Dataset that has an action column."""
    eps = []
    for ep in dataset.episodes:
        fr = ep.frames
        if "action" not in fr:
            continue
        a = np.asarray(fr["action"], dtype=np.float32)
        if a.ndim != 2 or len(a) < 8:
            continue
        s = np.asarray(fr["observation.state"], dtype=np.float32) if "observation.state" in fr else None
        eps.append({"index": ep.index, "action": a, "state": s})
    if len(eps) < 4:
        return []
    m = _load()
    model, thr, classes = m["model"], m["threshold"], m["classes"]
    ctx = DatasetContext(eps)
    X = np.stack([episode_features(e, ctx) for e in eps])
    pr = model.predict_proba(X)
    cols = list(model.classes_)
    clean = cols.index(0)
    out = []
    for e, p in zip(eps, pr, strict=True):
        pb = float(1 - p[clean])
        q = p.copy()
        q[clean] = 0
        flagged = pb > thr
        out.append(
            EpisodeQuality(e["index"], round(pb, 4), flagged, classes[cols[int(q.argmax())]] if flagged else None)
        )
    return out
