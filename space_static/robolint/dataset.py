"""Loading LeRobot datasets (v2.x and v3.x layouts) from disk or the Hugging Face Hub."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class DatasetError(Exception):
    """Raised when a dataset cannot be loaded at all."""


@dataclass
class VideoRef:
    key: str
    path: Path
    from_ts: float = 0.0
    to_ts: float | None = None  # None = until end of file


@dataclass
class Episode:
    index: int
    frames: dict[str, np.ndarray]
    length_meta: int | None = None
    tasks: list[str] = field(default_factory=list)
    videos: list[VideoRef] = field(default_factory=list)

    @property
    def length(self) -> int:
        for arr in self.frames.values():
            return len(arr)
        return 0


@dataclass
class Dataset:
    root: Path
    info: dict[str, Any]
    episodes: list[Episode]
    tasks: dict[int, str]
    data_columns: set[str]
    source: str = ""
    load_warnings: list[str] = field(default_factory=list)
    partial: bool = False  # only the first N episodes were loaded (quick mode)

    @property
    def version(self) -> str:
        return str(self.info.get("codebase_version", "unknown"))

    @property
    def major_version(self) -> int:
        m = re.match(r"v?(\d+)", self.version)
        return int(m.group(1)) if m else 0

    @property
    def fps(self) -> float | None:
        fps = self.info.get("fps")
        return float(fps) if isinstance(fps, (int, float)) and fps > 0 else None

    @property
    def features(self) -> dict[str, Any]:
        f = self.info.get("features")
        return f if isinstance(f, dict) else {}

    @property
    def video_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if isinstance(v, dict) and v.get("dtype") == "video"]

    def feature_names(self, key: str) -> list[str]:
        """Per-dimension names for a vector feature, if the dataset declares them."""
        names = self.features.get(key, {}).get("names")
        if isinstance(names, dict):  # e.g. {"motors": [...]}
            for v in names.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
            return []
        if isinstance(names, list):
            return [str(x) for x in names]
        return []

    @property
    def total_frames(self) -> int:
        return sum(ep.length for ep in self.episodes)


# ---------------------------------------------------------------------------
# Hub download


def resolve_source(
    source: str,
    *,
    include_videos: bool = False,
    revision: str | None = None,
    cache_dir: str | None = None,
    max_episodes: int | None = None,
) -> Path:
    """Return a local directory for `source` (a path, or a Hub repo id like 'lerobot/pusht').

    With max_episodes, only the files holding the first N episodes are downloaded, so huge
    datasets (tens of GB) can be spot-checked quickly.
    """
    p = Path(source).expanduser()
    if p.exists():
        if not p.is_dir():
            raise DatasetError(f"{source} is not a directory")
        return p
    if not REPO_ID_RE.match(source):
        raise DatasetError(f"{source!r} is neither an existing directory nor a Hub repo id (owner/name)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - depends on extras
        raise DatasetError(
            "Downloading from the Hugging Face Hub needs huggingface_hub: pip install 'robolint[hub]'"
        ) from e

    def fetch(patterns: list[str]) -> Path:
        try:
            return Path(
                snapshot_download(
                    repo_id=source,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_patterns=patterns,
                )
            )
        except Exception as e:
            raise DatasetError(f"could not download {source}: {e}") from e

    if max_episodes is None:
        return fetch(["meta/**", "data/**", "README.md"] + (["videos/**"] if include_videos else []))
    local = fetch(["meta/**", "README.md"])
    patterns = quick_patterns(local, max_episodes, include_videos)
    if patterns is None:  # layout not understood: fall back to everything
        patterns = ["data/**"] + (["videos/**"] if include_videos else [])
    return fetch(["meta/**", "README.md", *patterns])


def quick_patterns(root: Path, n: int, include_videos: bool) -> list[str] | None:
    """File paths that hold the first n episodes, worked out from the metadata alone."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = _read_json(info_path)
    data_t, video_t = info.get("data_path"), info.get("video_path")
    if not isinstance(data_t, str):
        return None
    keys = [k for k, v in (info.get("features") or {}).items() if isinstance(v, dict) and v.get("dtype") == "video"]
    chunks = int(info.get("chunks_size") or 1000)
    out: set[str] = set()
    try:
        if "{episode_index" in data_t:  # v2.x: one file per episode
            for ep in range(n + 1):
                out.add(data_t.format(episode_chunk=ep // chunks, episode_index=ep))
                if include_videos and isinstance(video_t, str):
                    for k in keys:
                        out.add(video_t.format(episode_chunk=ep // chunks, episode_index=ep, video_key=k))
        else:  # v3.x: chunked files, mapping lives in meta/episodes
            meta = _load_episode_meta(root)
            if not meta:
                return None
            for ep in sorted(meta)[: n + 1]:
                row = meta[ep]
                if row.get("data/chunk_index") is None:
                    return None
                out.add(data_t.format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])))
                if include_videos and isinstance(video_t, str):
                    for k in keys:
                        ci, fi = row.get(f"videos/{k}/chunk_index"), row.get(f"videos/{k}/file_index")
                        if ci is not None and fi is not None:
                            out.add(video_t.format(video_key=k, chunk_index=int(ci), file_index=int(fi)))
    except (KeyError, IndexError, ValueError):
        return None
    return sorted(out)


# ---------------------------------------------------------------------------
# Loading


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _inside(root: Path, p: Path) -> bool:
    try:
        return p.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _column_to_numpy(col: pa.ChunkedArray) -> np.ndarray | None:
    """Numeric scalar -> (n,), numeric list -> (n, d). Anything else -> None."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    t = arr.type
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t):
        out = arr.to_numpy(zero_copy_only=False)
        return out.astype(np.float64) if out.dtype == bool else out
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        vt = t.value_type
        if not (pa.types.is_integer(vt) or pa.types.is_floating(vt)):
            return None
        n = len(arr)
        if n == 0:
            return np.zeros((0, 0))
        if arr.null_count:
            return None
        lengths = pc.list_value_length(arr).to_numpy(zero_copy_only=False)
        if len(np.unique(lengths)) != 1:
            return None
        d = int(lengths[0])
        flat = pc.list_flatten(arr).to_numpy(zero_copy_only=False).astype(np.float64)
        return flat.reshape(n, d)
    return None


def _load_frames(root: Path, max_episodes: int | None = None) -> tuple[dict[str, np.ndarray], set[str]]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise DatasetError(f"no parquet files under {root / 'data'}")
    tables = []
    all_names: set[str] = set()
    seen_eps: set[int] = set()
    for f in files:
        if not _inside(root, f):
            continue
        try:
            schema = pq.read_schema(f)
            all_names.update(schema.names)
            # Only numeric columns: skips embedded images, which can be gigabytes.
            wanted = [fld.name for fld in schema if _is_numeric(fld.type)]
            t = pq.read_table(f, columns=wanted)
        except Exception as e:
            raise DatasetError(f"unreadable parquet file {f.relative_to(root)}: {e}") from e
        tables.append(t)
        if max_episodes is not None and "episode_index" in t.column_names:
            seen_eps.update(pc.unique(t.column("episode_index")).to_pylist())
            if len(seen_eps) > max_episodes:
                break  # quick mode: enough files read
    if not tables:
        raise DatasetError("no readable parquet files")
    table = pa.concat_tables(tables, promote_options="default")
    columns: dict[str, np.ndarray] = {}
    for name in table.column_names:
        arr = _column_to_numpy(table.column(name))
        if arr is not None:
            columns[name] = arr
    return columns, all_names


def _is_numeric(t: pa.DataType) -> bool:
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t):
        return True
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        vt = t.value_type
        return pa.types.is_integer(vt) or pa.types.is_floating(vt)
    return False


def _load_tasks(root: Path) -> dict[int, str]:
    pq_path = root / "meta" / "tasks.parquet"
    jl_path = root / "meta" / "tasks.jsonl"
    tasks: dict[int, str] = {}
    if pq_path.exists():
        t = pq.read_table(pq_path)
        cols = t.column_names
        idx_col = "task_index" if "task_index" in cols else None
        str_cols = [
            c
            for c in cols
            if pa.types.is_string(t.schema.field(c).type) or pa.types.is_large_string(t.schema.field(c).type)
        ]
        text_col = "task" if "task" in cols else (str_cols[0] if str_cols else None)
        if idx_col and text_col:
            for i, s in zip(t.column(idx_col).to_pylist(), t.column(text_col).to_pylist(), strict=True):
                if i is not None:
                    tasks[int(i)] = "" if s is None else str(s)
    elif jl_path.exists():
        for row in _read_jsonl(jl_path):
            if "task_index" in row:
                tasks[int(row["task_index"])] = str(row.get("task", ""))
    return tasks


def _load_episode_meta(root: Path) -> dict[int, dict[str, Any]]:
    meta: dict[int, dict[str, Any]] = {}
    ep_dir = root / "meta" / "episodes"
    jl = root / "meta" / "episodes.jsonl"
    if ep_dir.is_dir():
        for f in sorted(ep_dir.rglob("*.parquet")):
            t = pq.read_table(f)
            keep = [c for c in t.column_names if not c.startswith("stats/")]
            for row in t.select(keep).to_pylist():
                if row.get("episode_index") is not None:
                    meta[int(row["episode_index"])] = row
    elif jl.exists():
        for row in _read_jsonl(jl):
            if "episode_index" in row:
                meta[int(row["episode_index"])] = row
    return meta


def _video_refs(
    root: Path, info: dict[str, Any], ep: int, row: dict[str, Any], video_keys: list[str], major: int
) -> list[VideoRef]:
    template = info.get("video_path")
    if not isinstance(template, str):
        return []
    chunks_size = int(info.get("chunks_size") or 1000)
    refs = []
    for key in video_keys:
        try:
            if major >= 3:
                ci = row.get(f"videos/{key}/chunk_index")
                fi = row.get(f"videos/{key}/file_index")
                if ci is None or fi is None:
                    continue
                rel = template.format(video_key=key, chunk_index=int(ci), file_index=int(fi))
                ref = VideoRef(
                    key=key,
                    path=root / rel,
                    from_ts=float(row.get(f"videos/{key}/from_timestamp") or 0.0),
                    to_ts=(
                        float(row[f"videos/{key}/to_timestamp"])
                        if row.get(f"videos/{key}/to_timestamp") is not None
                        else None
                    ),
                )
            else:
                rel = template.format(video_key=key, episode_chunk=ep // chunks_size, episode_index=ep)
                ref = VideoRef(key=key, path=root / rel)
        except (KeyError, IndexError, ValueError):
            continue
        if _inside(root, ref.path):
            refs.append(ref)
    return refs


def load_dataset(root: Path, *, source: str = "", max_episodes: int | None = None) -> Dataset:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise DatasetError(f"missing meta/info.json in {root} (is this a LeRobot dataset?)")
    info = _read_json(info_path)
    if not isinstance(info, dict):
        raise DatasetError("meta/info.json is not a JSON object")

    columns, all_columns = _load_frames(root, max_episodes)
    if "episode_index" not in columns:
        raise DatasetError("data has no episode_index column")

    warnings: list[str] = []
    ep_col = columns["episode_index"].astype(np.int64)
    order_keys = [ep_col]
    if "frame_index" in columns:
        order_keys.insert(0, columns["frame_index"])
    order = np.lexsort(order_keys)  # sort by episode, then frame
    sorted_cols = {k: v[order] for k, v in columns.items()}
    ep_sorted = sorted_cols["episode_index"].astype(np.int64)
    uniq, starts = np.unique(ep_sorted, return_index=True)
    ends = list(starts[1:]) + [len(ep_sorted)]

    tasks = _load_tasks(root)
    meta = _load_episode_meta(root)
    major = int(m.group(1)) if (m := re.match(r"v?(\d+)", str(info.get("codebase_version", "")))) else 0
    video_keys = [
        k for k, v in (info.get("features") or {}).items() if isinstance(v, dict) and v.get("dtype") == "video"
    ]

    episodes: list[Episode] = []
    for n, (ep, s, e) in enumerate(zip(uniq.tolist(), starts.tolist(), ends, strict=True)):
        if max_episodes is not None and n >= max_episodes:
            break
        row = meta.get(int(ep), {})
        ep_tasks = row.get("tasks")
        if isinstance(ep_tasks, str):
            ep_tasks = [ep_tasks]
        if not isinstance(ep_tasks, list):
            ep_tasks = []
        length_meta = row.get("length")
        episodes.append(
            Episode(
                index=int(ep),
                frames={k: v[s:e] for k, v in sorted_cols.items()},
                length_meta=int(length_meta) if isinstance(length_meta, (int, float)) else None,
                tasks=[str(t) for t in ep_tasks],
                videos=_video_refs(root, info, int(ep), row, video_keys, major),
            )
        )
    if not meta:
        warnings.append("no episode metadata found (meta/episodes or meta/episodes.jsonl)")
    declared = info.get("total_episodes")
    partial = max_episodes is not None and (
        len(uniq) > len(episodes) or (isinstance(declared, int) and declared > len(episodes))
    )

    return Dataset(
        root=root,
        info=info,
        episodes=episodes,
        tasks=tasks,
        data_columns=all_columns,
        source=source or str(root),
        load_warnings=warnings,
        partial=partial,
    )
