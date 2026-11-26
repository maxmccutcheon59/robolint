"""Write a cleaned copy of a LeRobot dataset by applying a cleaning plan.

The copy keeps the source's layout (v2.x one file per episode, or v3.x chunked files) and never
touches the source. Removed episodes are dropped and the rest renumbered 0..N-1; idle edges are
trimmed (with v3 video offsets moved to match); lengths, frame indices, global indices and
timestamps are rewritten from the data itself, which repairs bookkeeping errors; labels and the
mapping back to the original episodes go to meta/robolint_cleaning.json.

Limits: per-episode stats in the metadata are copied as they were (slightly stale for trimmed
episodes) and meta/stats.json is copied unchanged; v2.x episodes with videos keep their leading
idle frames (cutting them would mean re-encoding the video), only the tail is trimmed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .cleaning import EpisodePlan


class CleanError(Exception):
    pass


@dataclass
class CleanReport:
    out: Path
    kept: int
    removed: list[int]
    trimmed_frames: int
    frames_before: int
    frames_after: int
    notes: list[str] = field(default_factory=list)


def _inside(root: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_all_data(root: Path) -> pa.Table:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise CleanError("no data files under data/")
    tables = [pq.read_table(f) for f in files]
    t = pa.concat_tables(tables, promote_options="default")
    if "episode_index" not in t.column_names:
        raise CleanError("data has no episode_index column")
    sort = [("episode_index", "ascending")] + [(c, "ascending") for c in ("frame_index", "index") if c in t.column_names][:1]
    return t.sort_by(sort)


def _episode_meta_v3(root: Path) -> pa.Table:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise CleanError("missing meta/episodes for a v3 dataset")
    return pa.concat_tables([pq.read_table(f) for f in files], promote_options="default").sort_by("episode_index")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_cleaned(src: Path, plans: list[EpisodePlan], out: Path) -> CleanReport:
    src, out = Path(src), Path(out)
    if out.exists() and any(out.iterdir()):
        raise CleanError(f"{out} already exists and is not empty")
    if _inside(src, out) or _inside(out, src):
        raise CleanError("the output folder must be outside the source dataset (and not contain it)")
    info = json.loads((src / "meta" / "info.json").read_text())
    major = int(str(info.get("codebase_version", "v0")).lstrip("v").split(".")[0] or 0)
    fps = float(info.get("fps") or 0) or None
    video_keys = [k for k, v in (info.get("features") or {}).items() if isinstance(v, dict) and v.get("dtype") == "video"]
    by_ep = {p.episode: p for p in plans}
    notes: list[str] = []

    data = _read_all_data(src)
    ep_col = data.column("episode_index").to_numpy()
    starts = np.flatnonzero(np.r_[True, ep_col[1:] != ep_col[:-1]])
    ends = np.r_[starts[1:], len(ep_col)]
    runs = {int(ep_col[s]): (int(s), int(e)) for s, e in zip(starts, ends, strict=True)}

    kept_eps = [e for e in sorted(runs) if not (e in by_ep and by_ep[e].decision == "remove")]
    removed = [e for e in sorted(runs) if e not in kept_eps]
    if not kept_eps:
        raise CleanError("the plan removes every episode")

    pieces, new_meta, mapping = [], [], []
    shifts: dict[int, tuple[float, float]] = {}  # old episode -> (seconds cut at start, seconds cut at end)
    g = 0
    trimmed = 0
    for k, old in enumerate(kept_eps):
        s, e = runs[old]
        n = e - s
        lead, tail = (by_ep[old].trim if old in by_ep and by_ep[old].trim else (0, 0))
        if major < 3 and video_keys and lead:
            lead = 0
            if "v2_video_lead" not in notes:
                notes.append("v2_video_lead")
        if n - lead - tail < max(10, n // 4):  # never trim an episode down to almost nothing
            lead, tail = 0, 0
        piece = data.slice(s + lead, n - lead - tail)
        m = piece.num_rows
        trimmed += n - m
        ts = piece.column("timestamp").to_numpy().astype(np.float64) if "timestamp" in piece.column_names else None
        t0 = float(ts[0]) if ts is not None else (lead / fps if fps else 0.0)
        cols = {c: piece.column(c) for c in piece.column_names}
        typ = lambda c, default: cols[c].type if c in cols else default  # noqa: E731
        cols["episode_index"] = pa.array(np.full(m, k), typ("episode_index", pa.int64()))
        cols["frame_index"] = pa.array(np.arange(m), typ("frame_index", pa.int64()))
        cols["index"] = pa.array(np.arange(g, g + m), typ("index", pa.int64()))
        if ts is not None:
            orig_first = float(data.column("timestamp")[s].as_py())
            cols["timestamp"] = pa.array((ts - t0).astype(np.float32), typ("timestamp", pa.float32()))
            cut_start = t0 - orig_first
        else:
            cut_start = lead / fps if fps else 0.0
        cut_end = tail / fps if fps else 0.0
        shifts[old] = (cut_start, cut_end)
        pieces.append(pa.table({c: cols[c] for c in piece.column_names}))
        mapping.append({"episode": k, "original_episode": old, "frames": m, "frames_trimmed": n - m,
                        "labels": by_ep[old].labels if old in by_ep else [],
                        "actions": by_ep[old].actions if old in by_ep else []})
        new_meta.append((k, old, g, m))
        g += m

    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir()
    if major >= 3:
        (out / "data" / "chunk-000").mkdir(parents=True)
        pq.write_table(pa.concat_tables(pieces), out / "data" / "chunk-000" / "file-000.parquet")
        meta = _episode_meta_v3(src)
        rows = {int(r["episode_index"]): r for r in meta.to_pylist()}
        out_rows = []
        for k, old, gfrom, m in new_meta:
            r = dict(rows.get(old, {"episode_index": old}))
            r.update({"episode_index": k, "length": m, "dataset_from_index": gfrom, "dataset_to_index": gfrom + m,
                      "data/chunk_index": 0, "data/file_index": 0})
            if "meta/episodes/chunk_index" in r:
                r["meta/episodes/chunk_index"], r["meta/episodes/file_index"] = 0, 0
            a, b = shifts[old]
            for key in video_keys:
                if r.get(f"videos/{key}/from_timestamp") is not None:
                    r[f"videos/{key}/from_timestamp"] = float(r[f"videos/{key}/from_timestamp"]) + a
                if r.get(f"videos/{key}/to_timestamp") is not None:
                    r[f"videos/{key}/to_timestamp"] = float(r[f"videos/{key}/to_timestamp"]) - b
            out_rows.append(r)
        (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(out_rows, schema=meta.schema), out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        for name in ("tasks.parquet", "stats.json"):
            if (src / "meta" / name).exists():
                shutil.copyfile(src / "meta" / name, out / "meta" / name)
        if (src / "videos").exists():
            for f in (src / "videos").rglob("*.mp4"):
                dst = out / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    else:
        chunks = int(info.get("chunks_size") or 1000)
        data_t = info["data_path"]
        video_t = info.get("video_path")
        for (k, old, _, _), t in zip(new_meta, pieces, strict=True):
            p = out / data_t.format(episode_chunk=k // chunks, episode_index=k)
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, p)
            for key in video_keys if isinstance(video_t, str) else []:
                vs = src / video_t.format(episode_chunk=old // chunks, video_key=key, episode_index=old)
                if vs.exists() and _inside(src, vs.parent):
                    vd = out / video_t.format(episode_chunk=k // chunks, video_key=key, episode_index=k)
                    vd.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(vs, vd)
        old_rows = {int(r["episode_index"]): r for r in _jsonl(src / "meta" / "episodes.jsonl")}
        (out / "meta" / "episodes.jsonl").write_text("".join(
            json.dumps({**old_rows.get(old, {}), "episode_index": k, "length": m}) + "\n" for k, old, _, m in new_meta))
        old_stats = {int(r["episode_index"]): r for r in _jsonl(src / "meta" / "episodes_stats.jsonl")}
        if old_stats:
            (out / "meta" / "episodes_stats.jsonl").write_text("".join(
                json.dumps({**old_stats[old], "episode_index": k}) + "\n" for k, old, _, _ in new_meta if old in old_stats))
        for name in ("tasks.jsonl", "stats.json"):
            if (src / "meta" / name).exists():
                shutil.copyfile(src / "meta" / name, out / "meta" / name)
        info["total_chunks"] = (len(kept_eps) - 1) // chunks + 1
        if video_keys:
            info["total_videos"] = len(kept_eps) * len(video_keys)

    info["total_episodes"] = len(kept_eps)
    info["total_frames"] = g
    info["splits"] = {"train": f"0:{len(kept_eps)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    if (src / "README.md").exists():
        shutil.copyfile(src / "README.md", out / "README.md")
    report = CleanReport(out, len(kept_eps), removed, trimmed, len(ep_col), g, notes)
    (out / "meta" / "robolint_cleaning.json").write_text(json.dumps({
        "source_episodes": len(runs), "kept": len(kept_eps), "removed_original_episodes": removed,
        "frames_before": len(ep_col), "frames_after": g, "frames_trimmed": trimmed,
        "notes": {"v2_video_lead": "v2 episodes with video keep their leading idle frames"} if notes else {},
        "episodes": mapping}, indent=1))
    return report
