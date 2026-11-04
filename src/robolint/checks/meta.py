"""Dataset-level structure and metadata consistency."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..dataset import Dataset
from ..findings import Finding, Severity
from . import Config

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO


def check(ds: Dataset, cfg: Config) -> Iterator[Finding]:
    info = ds.info
    for msg in ds.load_warnings:
        yield Finding("METADATA_INCOMPLETE", W, msg)

    if ds.fps is None:
        yield Finding("FPS_MISSING", E, "meta/info.json has no valid fps; timing cannot be verified")

    major = ds.major_version
    if major and major < 2:
        yield Finding("OLD_FORMAT", W, f"codebase_version {ds.version} predates LeRobot v2; results may be incomplete")

    if ds.partial:
        yield Finding(
            "PARTIAL_CHECK",
            I,
            f"quick mode: only the first {len(ds.episodes)} of {info.get('total_episodes', '?')} episodes were checked",
        )
    # Declared vs found counts (meaningless when only part of the dataset was loaded)
    found_eps = len(ds.episodes)
    declared_eps = info.get("total_episodes")
    if ds.partial:
        declared_eps = None
    if isinstance(declared_eps, int) and declared_eps != found_eps:
        yield Finding(
            "EPISODE_COUNT_MISMATCH",
            E,
            f"info.json declares {declared_eps} episodes but the data contains {found_eps}",
            details={"declared": declared_eps, "found": found_eps},
        )
    declared_frames = None if ds.partial else info.get("total_frames")
    if isinstance(declared_frames, int) and declared_frames != ds.total_frames:
        yield Finding(
            "FRAME_COUNT_MISMATCH",
            E,
            f"info.json declares {declared_frames} frames but the data contains {ds.total_frames}",
            details={"declared": declared_frames, "found": ds.total_frames},
        )

    # Episode indices should be 0..N-1
    idx = sorted(ep.index for ep in ds.episodes)
    if idx and idx != list(range(idx[0], idx[0] + len(idx))):
        missing = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx))
        yield Finding(
            "EPISODE_INDEX_GAPS",
            W,
            f"{len(missing)} episode indices missing between {idx[0]} and {idx[-1]}",
            details={"missing": missing[:50]},
        )

    # Global "index" column should be a contiguous range across the dataset
    globals_ = [ep.frames["index"] for ep in ds.episodes if "index" in ep.frames]
    if globals_:
        g = np.sort(np.concatenate(globals_).astype(np.int64))
        dupes = int(np.sum(np.diff(g) == 0))
        if dupes:
            yield Finding("GLOBAL_INDEX_DUPLICATES", E, f"{dupes} duplicate values in the global 'index' column")
        elif len(g) and (g[-1] - g[0] + 1) != len(g):
            yield Finding("GLOBAL_INDEX_GAPS", W, "global 'index' column is not contiguous")

    # Declared features must exist in the data
    for key, spec in ds.features.items():
        if not isinstance(spec, dict) or spec.get("dtype") in ("video", "image"):
            continue
        if key not in ds.data_columns:
            yield Finding("FEATURE_MISSING", E, f"feature '{key}' is declared in info.json but absent from the data")

    if "action" not in ds.data_columns:
        yield Finding("NO_ACTIONS", E, "dataset has no 'action' column; it cannot be used for policy training")
    if "timestamp" not in ds.data_columns:
        yield Finding("NO_TIMESTAMPS", E, "dataset has no 'timestamp' column")

    # Declared shapes should match
    for key in ("action", "observation.state"):
        spec = ds.features.get(key)
        if not isinstance(spec, dict) or not ds.episodes or key not in ds.episodes[0].frames:
            continue
        shape = spec.get("shape")
        arr = ds.episodes[0].frames[key]
        d = arr.shape[1] if arr.ndim == 2 else 1
        if isinstance(shape, list) and shape and isinstance(shape[0], int) and shape[0] != d:
            yield Finding(
                "SHAPE_MISMATCH",
                E,
                f"'{key}' declared with {shape[0]} dimensions but data has {d}",
            )

    # Tasks
    if not ds.tasks:
        yield Finding("TASKS_MISSING", W, "no task descriptions found (meta/tasks.parquet or tasks.jsonl)")
    else:
        empty = [i for i, t in ds.tasks.items() if not t.strip()]
        if empty:
            yield Finding(
                "TASK_EMPTY", W, f"{len(empty)} task descriptions are empty", details={"task_indices": empty[:20]}
            )
        vague = [t for t in ds.tasks.values() if t.strip() and (len(t.strip()) < 8 or t.strip().lower() in VAGUE_TASKS)]
        if vague:
            yield Finding(
                "TASK_VAGUE",
                I,
                f"{len(vague)} task descriptions are too vague to condition a language policy",
                details={"examples": vague[:5]},
            )


VAGUE_TASKS = {"task", "test", "demo", "default", "debug", "none", "null", "untitled", "do the task", "pick", "place"}
