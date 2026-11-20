"""Loading LeRobot datasets (v2.x and v3.x layouts) from disk or the Hugging Face Hub."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .frames import DEFAULT_CACHE_MB, ColumnCache, ColumnSpec, DataFile, FrameStore, LazyFrames, Rows, Segment, rows_len

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
    frames: Mapping[str, np.ndarray]  # column -> per-frame values (decoded on first use when loaded from disk)
    length_meta: int | None = None
    tasks: list[str] = field(default_factory=list)
    videos: list[VideoRef] = field(default_factory=list)
    n_rows: int | None = None  # frame count, known without decoding any column

    @property
    def length(self) -> int:
        if self.n_rows is not None:
            return self.n_rows
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
    load_errors: list[tuple[str, str, dict]] = field(default_factory=list)  # (code, message, details)
    partial: bool = False  # only the first N episodes were loaded (quick mode)
    format: str = "lerobot"  # "lerobot", "hdf5-robomimic", or "hdf5-aloha"
    store: FrameStore | None = field(default=None, repr=False)  # where lazily loaded frames come from
    # Columns read as "action" / "observation.state" when they are not single columns of those names
    key_sources: dict[str, list[str]] = field(default_factory=dict)
    virtual_features: dict[str, Any] = field(default_factory=dict)

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
        f = f if isinstance(f, dict) else {}
        return {**f, **self.virtual_features} if self.virtual_features else f

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


def _within(parent: str, child: str) -> bool:
    """True if absolute path `child` equals or sits below absolute path `parent` (no I/O)."""
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:  # different drives on Windows, or mixed abs/rel
        return False


_MAX_LINK_HOPS = 8


def hub_blob_dirs(snapshot: Path) -> tuple[Path, ...] | None:
    """Blob folders a Hugging Face cache snapshot may legitimately link into.

    huggingface_hub downloads to <cache>/datasets--<org>--<name>/snapshots/<rev>/... and
    stores each file as a relative symlink into that repo's blobs/ folder. Newer releases
    (2.x) add a second hop from the repo's blobs/ into a cache-wide <cache>/blobs/ folder.
    Returns None when `snapshot` does not have that layout.
    """
    real = Path(os.path.realpath(snapshot))
    repo_dir = real.parent.parent
    if real.parent.name != "snapshots" or not repo_dir.name.startswith("datasets--"):
        return None
    dirs = tuple(Path(os.path.realpath(d)) for d in (repo_dir / "blobs", repo_dir.parent / "blobs") if d.is_dir())
    return dirs or None


def _inside(root: Path, p: Path, hub_blobs: tuple[Path, ...] | None = None) -> bool:
    """Path-safety check for files the dataset metadata points us at.

    Always: the path, taken literally, must stay under root (no "..", no absolute paths
    smuggled in through metadata templates).

    Local datasets: the fully resolved file must be under the resolved root, so symlinks
    that leave the dataset folder are refused.

    Hub downloads (hub_blobs given): additionally accept the Hub cache's own symlinks, but
    only if every hop of the link chain lands inside one of the cache's blob folders, the
    chain is short, the folders between root and the file are real (not links out), and
    the final target is a regular file. A link to /etc/passwd, out of the cache, or a chain
    that detours outside the blob folders is refused.
    """
    try:
        root_abs = os.path.abspath(root)
        p_abs = os.path.abspath(p)
        if not _within(root_abs, p_abs):
            return False
        root_real = os.path.realpath(root)
        real = os.path.realpath(p)
        if _within(root_real, real):
            return True
        if not hub_blobs:
            return False
        allowed = [os.path.realpath(b) for b in hub_blobs]
        # Directories inside the snapshot must not themselves be links out of it.
        if not _within(root_real, os.path.realpath(os.path.dirname(p_abs))):
            return False
        cur = p_abs
        for hop in range(_MAX_LINK_HOPS + 1):
            if not os.path.islink(cur):
                break
            if hop == _MAX_LINK_HOPS:
                return False  # chain too long
            target = os.readlink(cur)
            if os.path.isabs(target):
                return False  # Hub cache links are always relative
            cur = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(cur)), target))
            if not any(_within(b, cur) for b in allowed):
                return False  # this hop leaves the cache's blob folders
        return cur == real and any(_within(b, real) for b in allowed) and os.path.isfile(real)
    except (OSError, RuntimeError, ValueError):
        return False


def _column_to_numpy(col: pa.ChunkedArray) -> tuple[np.ndarray | None, str | None]:
    """Numeric scalar -> (n,), numeric list -> (n, d).

    Returns (array, problem). Null entries and list rows whose width differs from the most common
    width become NaN rows (so per-episode checks flag them) and are described in `problem`.
    Non-numeric columns return (None, None); they are not data robolint checks.
    """
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    t = arr.type
    n = len(arr)
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t):
        nulls = arr.null_count
        if nulls:
            out = pc.cast(arr, pa.float64()).to_numpy(zero_copy_only=False)  # nulls -> NaN
            return out, f"{nulls} null values"
        out = arr.to_numpy(zero_copy_only=False)
        return (out.astype(np.float32) if out.dtype == bool else out), None
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        vt = t.value_type
        if not (pa.types.is_integer(vt) or pa.types.is_floating(vt)):
            return None, None
        if n == 0:
            return np.zeros((0, 0), np.float32), None
        lengths = pc.list_value_length(arr).to_numpy(zero_copy_only=False)
        valid = ~np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)
        if not valid.any():
            return np.full((n, 0), np.nan, np.float32), f"all {n} rows are null"
        lens_valid = lengths[valid].astype(np.int64)
        widths, counts = np.unique(lens_valid, return_counts=True)
        d = int(widths[np.argmax(counts)])
        good = valid & (np.nan_to_num(lengths, nan=-1).astype(np.int64) == d)
        problems = []
        if (~valid).any():
            problems.append(f"{int((~valid).sum())} null rows")
        odd = valid & ~good
        if odd.any():
            others = sorted({int(x) for x in lens_valid if x != d})
            problems.append(f"{int(odd.sum())} rows with width {others} instead of {d}")
        if not problems:
            flat = pc.list_flatten(arr).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            return flat.reshape(n, d), None
        out = np.full((n, d), np.nan, np.float32)
        if good.any():
            sub = arr.filter(pa.array(good))
            out[good] = pc.list_flatten(sub).to_numpy(zero_copy_only=False).astype(np.float32).reshape(-1, d)
        # Inner nulls inside otherwise valid rows already arrive as NaN through list_flatten -> float.
        return out, "; ".join(problems)
    return None, None


BOOKKEEPING = ("episode_index", "frame_index", "index", "timestamp", "task_index")
CORE = ("action", "observation.state", "episode_index", "frame_index", "timestamp")
# Columns the checks read; the scan keeps these decoded (cache budget permitting) instead of reading them twice.
HOT = (*BOOKKEEPING, "action", "observation.state")


@dataclass
class _Piece:
    fid: int
    rows: int  # rows that belong to an episode
    widths: dict[str, int]
    dtypes: dict[str, np.dtype]


def _plan(
    pieces: list[_Piece], files: list[DataFile], errors: list[tuple[str, str, dict]]
) -> tuple[list[str], dict[str, ColumnSpec]]:
    """Dataset-wide width and dtype of every column. A column that is missing, or has another width,
    in a file is read as NaN rows for that file's episodes, and reported."""
    names: list[str] = []
    for p in pieces:
        for k in p.widths:
            if k not in names:
                names.append(k)
    spec: dict[str, ColumnSpec] = {}
    for name in names:
        widths: dict[int, int] = {}
        for p in pieces:
            if name in p.widths:
                w = p.widths[name]
                widths[w] = widths.get(w, 0) + p.rows
        w = max(widths, key=lambda k: widths[k])
        dtypes = []
        for p in pieces:
            aw = p.widths.get(name)
            if aw == w:
                dtypes.append(p.dtypes[name])
                continue
            why = "is missing" if aw is None else f"has width {aw} instead of {w}"
            rel = files[p.fid].rel
            errors.append(
                (
                    "DATA_COLUMN_INVALID",
                    f"column '{name}' {why} in {rel}; its {p.rows} rows were treated as missing",
                    {"column": name, "file": rel, "rows": p.rows},
                )
            )
            dtypes.append(np.dtype(np.float32))
        if any(d.kind == "f" for d in dtypes) and any(d.kind != "f" for d in dtypes):
            dt = np.dtype(np.float64)
        else:
            dt = np.result_type(*dtypes) if dtypes else np.dtype(np.float64)
        spec[name] = ColumnSpec(width=w, dtype=np.dtype(dt))
    if "episode_index" in spec:
        spec["episode_index"].dtype = np.dtype(np.int64)
    return names, spec


def _episode_runs(ep: np.ndarray, pos: np.ndarray | None) -> list[tuple[int, Rows]]:
    """(episode, rows) for each run of one episode's rows in a file, in file order. `pos` maps the
    kept rows back to file rows (None when every row was kept)."""
    n = len(ep)
    if n == 0:
        return []
    cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    if len(cuts) > 2 * len(np.unique(ep)) + 16:
        # Rows of different episodes are interleaved: group each episode's rows (kept in file order).
        order = np.argsort(ep, kind="stable")
        e_sorted = ep[order]
        bounds = np.flatnonzero(e_sorted[1:] != e_sorted[:-1]) + 1
        out = []
        for s, e in zip(np.r_[0, bounds].tolist(), np.r_[bounds, n].tolist(), strict=True):
            idx = order[s:e]
            out.append((int(e_sorted[s]), idx if pos is None else pos[idx]))
        return out
    starts = np.r_[0, cuts].tolist()
    ends = np.r_[cuts, n].tolist()
    if pos is None:
        return [(int(ep[s]), slice(s, e)) for s, e in zip(starts, ends, strict=True)]
    return [(int(ep[s]), pos[s:e]) for s, e in zip(starts, ends, strict=True)]


def _scan(
    root: Path,
    max_episodes: int | None,
    hub_blobs: tuple[Path, ...] | None,
    errors: list[tuple[str, str, dict]],
    cache: ColumnCache,
    hot: tuple[str, ...] = HOT,
) -> tuple[FrameStore, set[str], int]:
    """Read every data file once: validate it, and record which of its rows belong to which episode.

    Problems that would otherwise drop data silently (files outside the dataset, unreadable
    columns, null rows, mixed list widths, rows without an episode) are appended to `errors` as
    (code, message, details); the meta check reports them as errors that cost score.

    Returns the frame store, every column name seen, and how many distinct episodes the scanned
    files hold.
    """
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise DatasetError(f"no parquet files under {root / 'data'}")
    files: list[DataFile] = []
    pieces: list[_Piece] = []
    all_names: set[str] = set()
    seen_eps: set[int] = set()
    segments: dict[int, list[Segment]] = {}
    for f in paths:
        rel = str(f.relative_to(root)) if _within(str(root), str(f)) else f.name
        if not _inside(root, f, hub_blobs):
            errors.append(
                (
                    "DATA_FILE_SKIPPED",
                    f"{rel} resolves outside the dataset folder (symlink) and was not read",
                    {"file": rel},
                )
            )
            continue
        try:
            pf = pq.ParquetFile(f)
            schema = pf.schema_arrow
            all_names.update(schema.names)
            # Only numeric columns: skips embedded images, which can be gigabytes.
            wanted = [fld.name for fld in schema if _is_numeric(fld.type)]
            for fld in schema:
                if fld.name in CORE and fld.name not in wanted:
                    errors.append(
                        (
                            "DATA_COLUMN_INVALID",
                            f"column '{fld.name}' in {rel} has non-numeric type {fld.type}",
                            {"column": fld.name, "file": rel, "type": str(fld.type)},
                        )
                    )
            n = pf.metadata.num_rows
        except Exception as e:
            raise DatasetError(f"unreadable parquet file {rel}: {e}") from e
        fid = len(files)
        widths: dict[str, int] = {}
        dtypes: dict[str, np.dtype] = {}
        ep: np.ndarray | None = None
        for name in wanted:
            # One column at a time: decoding a whole file at once can take several times its size.
            try:
                col = pf.read(columns=[name]).column(0)
            except Exception as e:
                raise DatasetError(f"unreadable parquet file {rel}: {e}") from e
            arr, problem = _column_to_numpy(col)
            del col
            if arr is None:
                continue
            if problem:
                errors.append(
                    ("DATA_COLUMN_INVALID", f"column '{name}' in {rel}: {problem}", {"column": name, "file": rel})
                )
            widths[name] = 0 if arr.ndim == 1 else arr.shape[1]
            dtypes[name] = arr.dtype
            if name == "episode_index":
                ep = arr
            if name in hot:
                cache.put((fid, name), arr)
            del arr
        files.append(DataFile(path=f, rel=rel, widths=widths))
        if ep is None:
            errors.append(
                (
                    "DATA_FILE_SKIPPED",
                    f"{rel} has no episode_index column; its {n} rows were not checked",
                    {"file": rel, "rows": n},
                )
            )
            continue
        pos = None
        if ep.dtype.kind == "f" and not np.all(np.isfinite(ep)):
            keep = np.isfinite(ep)
            errors.append(
                (
                    "DATA_FILE_SKIPPED",
                    f"{rel}: {int((~keep).sum())} rows have no episode_index and were not checked",
                    {"file": rel, "rows": int((~keep).sum())},
                )
            )
            pos = np.flatnonzero(keep)
            ep = ep[keep]
            n = int(keep.sum())
        ep = ep.astype(np.int64)
        dtypes["episode_index"] = np.dtype(np.int64)
        pieces.append(_Piece(fid=fid, rows=n, widths=widths, dtypes=dtypes))
        for e, rows in _episode_runs(ep, pos):
            segments.setdefault(e, []).append((fid, rows))
        if max_episodes is not None:
            seen_eps.update(np.unique(ep).tolist())
            if len(seen_eps) > max_episodes:
                break  # quick mode: enough files read
    if not pieces:
        raise DatasetError("no readable parquet files")
    names, spec = _plan(pieces, files, errors)
    return FrameStore(files, spec, names, segments, cache), all_names, len(segments)


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
    root: Path,
    info: dict[str, Any],
    ep: int,
    row: dict[str, Any],
    video_keys: list[str],
    major: int,
    hub_blobs: tuple[Path, ...] | None = None,
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
        if _inside(root, ref.path, hub_blobs):
            refs.append(ref)
    return refs


ACTION, STATE = "action", "observation.state"


def _feature_names(features: dict[str, Any], key: str) -> list[str]:
    names = features.get(key, {}).get("names") if isinstance(features.get(key), dict) else None
    if isinstance(names, dict):
        names = next((v for v in names.values() if isinstance(v, list)), None)
    return [str(x) for x in names] if isinstance(names, list) else []


def _part_order(key: str) -> tuple[int, str]:
    """Left before right, then by name, so action and state parts line up the same way."""
    k = key.lower()
    side = 0 if re.search(r"(^|[._])(left|l)([._]|$)", k) else 1 if re.search(r"(^|[._])(right|r)([._]|$)", k) else 2
    return side, k


def resolve_keys(
    features: dict[str, Any], columns: dict[str, int], action_keys=(), state_keys=()
) -> tuple[dict[str, list[str]], list[str]]:
    """Which data columns make up the action and the state.

    Explicit keys are used as given, in that order. Otherwise a column named exactly 'action'
    ('actions' also works) or 'observation.state' is used; failing that, all numeric columns
    under 'action.' or 'observation.state.' (for example 'action.left_arm' and 'action.right_arm'
    in a bimanual recording) are placed side by side, left before right. `columns` maps each
    numeric data column to its width. Returns {canonical key: source columns} for the keys that are
    not a single column of the canonical name, and problems with explicit keys.
    """
    out: dict[str, list[str]] = {}
    problems: list[str] = []
    for canon, given, fallbacks in ((ACTION, action_keys, ("actions",)), (STATE, state_keys, ())):
        if given:
            missing = [k for k in given if k not in columns]
            if missing:
                problems.append(
                    f"{canon} key{'s' if len(missing) > 1 else ''} {', '.join(repr(k) for k in missing)} "
                    "not found among the numeric data columns"
                )
            parts = [k for k in given if k in columns]
        elif canon in columns:
            continue
        elif any(f in columns for f in fallbacks):
            parts = [next(f for f in fallbacks if f in columns)]
        else:
            parts = sorted((k for k in columns if k.startswith(canon + ".")), key=_part_order)
            if canon == ACTION:  # 'action.*' never includes the state, but guard against odd names
                parts = [k for k in parts if not k.startswith(STATE)]
        if parts and parts != [canon]:
            out[canon] = parts
    return out, problems


def _virtual_feature(features: dict[str, Any], canon: str, parts: list[str], widths: dict[str, int]) -> dict:
    names: list[str] = []
    for p in parts:
        w = max(1, widths[p])
        label = p[len(canon) + 1 :] if p.startswith(canon + ".") else p
        own = _feature_names(features, p)
        if widths[p] == 0:
            names.append(label)
        elif len(own) == w:
            names.extend(f"{label}.{n}" if len(parts) > 1 else n for n in own)
        else:
            names.extend(f"{label}[{i}]" for i in range(w))
    return {"dtype": "float32", "shape": [len(names)], "names": names, "assembled_from": parts}


def load_dataset(
    root: Path,
    *,
    source: str = "",
    max_episodes: int | None = None,
    hub_blobs: tuple[Path, ...] | None = None,
    cache_mb: float | None = None,
    action_keys: tuple[str, ...] | list[str] = (),
    state_keys: tuple[str, ...] | list[str] = (),
) -> Dataset:
    """Load a dataset from a local folder.

    `action_keys` / `state_keys` name the data columns that make up the action and the state when
    they are not single columns called 'action' and 'observation.state'; by default split columns
    such as 'action.left_arm' and 'action.right_arm' are found and placed side by side (see
    resolve_keys).

    Frames are not held in memory: each episode's columns are decoded from the parquet files when a
    check first reads them, with at most `cache_mb` megabytes (default 512) of decoded columns kept.

    Pass hub_blobs (from hub_blob_dirs) only for folders that resolve_source downloaded from
    the Hub; it lets the Hub cache's symlinks into its blob store through the path check.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise DatasetError(f"missing meta/info.json in {root} (is this a LeRobot dataset?)")
    info = _read_json(info_path)
    if not isinstance(info, dict):
        raise DatasetError("meta/info.json is not a JSON object")

    load_errors: list[tuple[str, str, dict]] = []
    cache = ColumnCache(int((DEFAULT_CACHE_MB if cache_mb is None else cache_mb) * 1024 * 1024))
    store, all_columns, n_found = _scan(root, max_episodes, hub_blobs, load_errors, cache)
    if "episode_index" not in store.spec:
        raise DatasetError("data has no episode_index column")

    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    widths = {k: v.width for k, v in store.spec.items()}
    key_sources, problems = resolve_keys(features, widths, tuple(action_keys), tuple(state_keys))
    if problems:
        raise DatasetError("; ".join(problems) + f" (numeric columns: {', '.join(sorted(widths))})")
    virtual: dict[str, Any] = {}
    for canon, parts in key_sources.items():
        store.add_alias(canon, parts)
        all_columns = all_columns | {canon}
        virtual[canon] = _virtual_feature(features, canon, parts, widths)

    warnings: list[str] = []
    tasks = _load_tasks(root)
    meta = _load_episode_meta(root)
    major = int(m.group(1)) if (m := re.match(r"v?(\d+)", str(info.get("codebase_version", "")))) else 0
    video_keys = [
        k for k, v in (info.get("features") or {}).items() if isinstance(v, dict) and v.get("dtype") == "video"
    ]

    episodes: list[Episode] = []
    for n, ep in enumerate(sorted(store.segments)):
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
                frames=LazyFrames(store, ep),
                n_rows=sum(rows_len(r) for _, r in store.segments[ep]),
                length_meta=int(length_meta) if isinstance(length_meta, (int, float)) else None,
                tasks=[str(t) for t in ep_tasks],
                videos=_video_refs(root, info, int(ep), row, video_keys, major, hub_blobs),
            )
        )
    if not meta:
        warnings.append("no episode metadata found (meta/episodes or meta/episodes.jsonl)")
    declared = info.get("total_episodes")
    partial = max_episodes is not None and (
        n_found > len(episodes) or (isinstance(declared, int) and declared > len(episodes))
    )

    return Dataset(
        root=root,
        info=info,
        episodes=episodes,
        tasks=tasks,
        data_columns=all_columns,
        source=source or str(root),
        load_warnings=warnings,
        load_errors=load_errors,
        partial=partial,
        store=store,
        key_sources=key_sources,
        virtual_features=virtual,
    )
