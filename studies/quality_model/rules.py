"""Run robolint's rule checks on in-memory episodes (for a fair rules-vs-model comparison)."""

import numpy as np

from robolint.checks import Config, all_checks
from robolint.dataset import Dataset, Episode


def rule_flags(eps, fps):
    D = eps[0]["action"].shape[1]
    feats = {"action": {"dtype": "float32", "shape": [D]}}
    has_state = all(e.get("state") is not None and e["state"].shape[1] == D for e in eps)
    if has_state:
        feats["observation.state"] = {"dtype": "float32", "shape": [D]}
    episodes = []
    for i, e in enumerate(eps):
        T = len(e["action"])
        fr = {
            "action": e["action"],
            "timestamp": (np.arange(T) / fps).astype(np.float32),
            "frame_index": np.arange(T),
            "episode_index": np.full(T, i),
            "index": np.arange(T),
        }
        if has_state:
            fr["observation.state"] = e["state"]
        episodes.append(Episode(index=i, frames=fr, length_meta=T, n_rows=T))
    info = {"codebase_version": "v3.0", "fps": fps, "features": feats, "total_episodes": len(eps)}
    ds = Dataset(
        root=__import__("pathlib").Path("."),
        info=info,
        episodes=episodes,
        tasks={0: "t"},
        data_columns=set(episodes[0].frames),
    )
    cfg = Config()
    out = {}
    for chk in all_checks(include_video=False):
        try:
            for f in chk(ds, cfg):
                if f.episode is not None:
                    out[f.episode] = max(out.get(f.episode, 0), int(f.severity))
        except Exception as ex:
            out.setdefault("_errors", []).append(f"{chk.__name__}: {ex!r}"[:120])
    return out
