"""Out-of-core access to a dataset's per-frame columns.

The loader scans every data file once to validate it and to record which rows of which file
belong to which episode. Columns are then decoded on demand, one file and one column at a
time, and kept in a small least-recently-used cache with a byte budget. Checks walk the
episodes in order, so a dataset of any size is checked with memory bounded by the cache
budget plus one episode, instead of holding every frame of every column at once.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_CACHE_MB = 512
EPISODE_CACHE = 8  # assembled episodes kept around (checks revisit the episode they are on)


@dataclass
class DataFile:
    path: Path
    rel: str
    widths: dict[str, int]  # numeric column -> width (0 for scalar columns)


@dataclass
class ColumnSpec:
    width: int  # dataset-wide width; files with another width read as missing (NaN rows)
    dtype: np.dtype  # dtype of the column once merged across files


Rows = slice | np.ndarray
Segment = tuple[int, Rows]  # (file id, rows of that file)


def rows_len(rows: Rows) -> int:
    if isinstance(rows, slice):
        return rows.stop - rows.start
    return len(rows)


class ColumnCache:
    """LRU of decoded (file, column) arrays with a byte budget. The most recent entry is always
    kept, even if it alone is over budget."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget = max(0, int(budget_bytes))
        self._items: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
        self.bytes = 0
        self.decodes = 0  # columns decoded from disk after the initial scan (for tests and benchmarks)

    def get(self, key: tuple[int, str]) -> np.ndarray | None:
        a = self._items.get(key)
        if a is not None:
            self._items.move_to_end(key)
        return a

    def put(self, key: tuple[int, str], a: np.ndarray) -> None:
        old = self._items.pop(key, None)
        if old is not None:
            self.bytes -= old.nbytes
        self._items[key] = a
        self.bytes += a.nbytes
        while self.bytes > self.budget and len(self._items) > 1:
            _, ev = self._items.popitem(last=False)
            self.bytes -= ev.nbytes

    def clear(self) -> None:
        self._items.clear()
        self.bytes = 0


class FrameStore:
    """Decodes episode columns on demand from the dataset's parquet files."""

    def __init__(
        self,
        files: list[DataFile],
        spec: dict[str, ColumnSpec],
        names: list[str],
        segments: dict[int, list[Segment]],
        cache: ColumnCache,
    ) -> None:
        self.files = files
        self.spec = spec
        self.names = names
        self.segments = segments
        self.cache = cache
        self.aliases: dict[str, list[str]] = {}  # column assembled from other columns, side by side
        self._episodes: OrderedDict[int, tuple[np.ndarray | None, dict[str, np.ndarray]]] = OrderedDict()

    # -- file columns -------------------------------------------------------------------------

    def file_column(self, fid: int, name: str) -> np.ndarray | None:
        """The whole column of one file as the loader converted it, or None if that file lacks the
        column or has it at another width than the rest of the dataset."""
        f = self.files[fid]
        spec = self.spec[name]
        if f.widths.get(name) != spec.width:
            return None
        a = self.cache.get((fid, name))
        if a is None:
            from .dataset import _column_to_numpy

            a, _ = _column_to_numpy(pq.ParquetFile(f.path).read(columns=[name]).column(0))
            if a is None:  # pragma: no cover - the scan already checked the type
                return None
            self.cache.decodes += 1
            self.cache.put((fid, name), a)
        return a

    # -- episodes -----------------------------------------------------------------------------

    def add_alias(self, name: str, parts: list[str]) -> None:
        """Make `name` read as the columns `parts` placed side by side (e.g. one action vector from
        separate left- and right-arm columns)."""
        width = sum(max(1, self.spec[p].width) for p in parts)
        self.spec[name] = ColumnSpec(width=width, dtype=np.dtype(np.float32))  # as vector columns load
        self.aliases[name] = list(parts)
        if name not in self.names:
            self.names.append(name)
        self._episodes.clear()

    def _raw(self, ep: int, name: str) -> np.ndarray:
        spec = self.spec[name]
        if name in self.aliases:
            cols = [self._raw(ep, p) for p in self.aliases[name]]
            cols = [c.reshape(len(c), -1) for c in cols]
            return np.concatenate(cols, axis=1).astype(spec.dtype)
        parts = []
        for fid, rows in self.segments[ep]:
            a = self.file_column(fid, name)
            if a is None:
                k = rows_len(rows)
                part = np.full((k,) if spec.width == 0 else (k, spec.width), np.nan, np.float32)
            else:
                part = a[rows]
            # Always a copy: a view would keep the whole file's column alive after the cache drops it.
            parts.append(part.astype(spec.dtype))
        if not parts:
            return np.zeros(0, spec.dtype)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def _entry(self, ep: int) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
        e = self._episodes.get(ep)
        if e is not None:
            self._episodes.move_to_end(ep)
            return e
        order = None
        if "frame_index" in self.spec:
            fi = self._raw(ep, "frame_index")
            # Frames are ordered by frame_index; ties keep file order (a stable sort).
            if len(fi) > 1 and not np.all(fi[1:] >= fi[:-1]):
                order = np.argsort(fi, kind="stable")
        e = (order, {})
        self._episodes[ep] = e
        while len(self._episodes) > EPISODE_CACHE:
            self._episodes.popitem(last=False)
        return e

    def column(self, ep: int, name: str) -> np.ndarray:
        order, cols = self._entry(ep)
        a = cols.get(name)
        if a is None:
            a = self._raw(ep, name)
            if order is not None:
                a = a[order]
            a.flags.writeable = False  # shared between checks: nobody may modify it in place
            cols[name] = a
        return a

    def drop_episode_cache(self) -> None:
        self._episodes.clear()


class LazyFrames(Mapping):
    """An episode's columns, decoded when first used. Behaves like a read-only dict."""

    __slots__ = ("_store", "_ep")

    def __init__(self, store: FrameStore, ep: int) -> None:
        self._store = store
        self._ep = ep

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._store.spec:
            raise KeyError(key)
        return self._store.column(self._ep, key)

    def __contains__(self, key: object) -> bool:
        return key in self._store.spec

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.names)

    def __len__(self) -> int:
        return len(self._store.names)
