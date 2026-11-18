"""Path safety with the Hugging Face Hub cache, which stores downloaded files as symlinks.

Layout built here mirrors huggingface_hub:
  <cache>/datasets--org--name/snapshots/<rev>/<file> -> ../../blobs/<sha>          (1.x and earlier)
  <cache>/datasets--org--name/blobs/<sha> -> ../../blobs/<xx>/<hash>               (extra hop in 2.x)
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from robolint import lint
from robolint.dataset import DatasetError, _inside, hub_blob_dirs, load_dataset

from .synth import make_dataset

pytestmark = pytest.mark.skipif(os.name == "nt", reason="needs POSIX symlinks")

REV = "0123456789abcdef0123456789abcdef01234567"


def _make_hub_cache(tmp_path: Path, version: str = "v3.0", shared_blobs: bool = False) -> Path:
    """Build a local dataset, then move it into a fake Hub cache the way huggingface_hub lays it out."""
    src = make_dataset(tmp_path / "src", version=version, n_eps=6, length=60)
    cache = tmp_path / "cache"
    repo = cache / "datasets--org--name"
    snap = repo / "snapshots" / REV
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(REV)
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        repo_blob = repo / "blobs" / sha
        if shared_blobs:
            shared = cache / "blobs" / sha[:2] / sha
            shared.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, shared)
            repo_blob.symlink_to(os.path.relpath(shared, repo_blob.parent))
        else:
            shutil.copyfile(f, repo_blob)
        link = snap / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(repo_blob, link.parent))
    return snap


def _first_parquet(snap: Path) -> Path:
    return sorted((snap / "data").rglob("*.parquet"))[0]


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
@pytest.mark.parametrize("shared_blobs", [False, True])
def test_hub_cache_symlinks_are_accepted(tmp_path, version, shared_blobs):
    snap = _make_hub_cache(tmp_path, version, shared_blobs)
    assert _first_parquet(snap).is_symlink()
    blobs = hub_blob_dirs(snap)
    assert blobs
    ds = load_dataset(snap, hub_blobs=blobs)
    assert len(ds.episodes) == 6
    # Same answer as the plain local copy.
    assert ds.total_frames == load_dataset(tmp_path / "src").total_frames


def test_lint_hub_id_through_cache_symlinks(tmp_path, monkeypatch):
    """End to end through lint() with snapshot_download stubbed to return the fake cache."""
    hf = pytest.importorskip("huggingface_hub")
    snap = _make_hub_cache(tmp_path, shared_blobs=True)
    monkeypatch.setattr(hf, "snapshot_download", lambda **kw: str(snap))
    monkeypatch.chdir(tmp_path)  # "org/name" must not exist as a local folder
    r = lint("org/name")
    assert len(r.dataset.episodes) == 6


def test_local_dataset_keeps_strict_rule(tmp_path):
    """Without hub_blobs, a symlink out of the dataset folder is refused, even in a cache-like layout."""
    snap = _make_hub_cache(tmp_path)
    with pytest.raises(DatasetError, match="no readable parquet"):
        load_dataset(snap)


def test_local_path_to_a_cache_snapshot_is_strict_in_lint(tmp_path):
    snap = _make_hub_cache(tmp_path)
    with pytest.raises(DatasetError, match="no readable parquet"):
        lint(snap)


def _repoint(link: Path, target: Path | str) -> None:
    link.unlink()
    link.symlink_to(target)


def test_symlink_escaping_cache_is_rejected(tmp_path):
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    victim = _first_parquet(snap)
    outside = tmp_path / "outside.parquet"
    shutil.copyfile(victim, outside)  # a perfectly valid parquet file, just outside the cache
    _repoint(victim, os.path.relpath(outside, victim.parent))
    assert not _inside(snap, victim, blobs)
    _repoint(victim, outside)  # absolute form too
    assert not _inside(snap, victim, blobs)


def test_symlink_to_etc_passwd_is_rejected(tmp_path):
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    victim = _first_parquet(snap)
    _repoint(victim, "/etc/passwd")
    assert not _inside(snap, victim, blobs)
    _repoint(victim, os.path.relpath("/etc/passwd", victim.parent))
    assert not _inside(snap, victim, blobs)


def test_symlink_to_other_cache_areas_is_rejected(tmp_path):
    """Inside the cache but not a blob (e.g. refs/main or another repo's snapshot): refused."""
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    victim = _first_parquet(snap)
    refs_main = snap.parent.parent / "refs" / "main"
    _repoint(victim, os.path.relpath(refs_main, victim.parent))
    assert not _inside(snap, victim, blobs)


def test_chained_symlink_escaping_is_rejected(tmp_path):
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    victim = _first_parquet(snap)
    blob = Path(os.path.realpath(victim))
    # snapshot -> repo blobs/<sha> -> /etc/passwd
    blob.unlink()
    blob.symlink_to("/etc/passwd")
    assert not _inside(snap, victim, blobs)


def test_chain_detouring_outside_cache_is_rejected(tmp_path):
    """snapshot -> outside link -> back into blobs: final target is a blob, but the chain left the cache."""
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    victim = _first_parquet(snap)
    blob = Path(os.path.realpath(victim))
    hop = tmp_path / "detour"
    hop.symlink_to(blob)
    _repoint(victim, os.path.relpath(hop, victim.parent))
    assert not _inside(snap, victim, blobs)


def test_symlink_loop_is_rejected(tmp_path):
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    victim = _first_parquet(snap)
    a = snap.parent.parent / "blobs" / "loop_a"
    b = snap.parent.parent / "blobs" / "loop_b"
    a.symlink_to("loop_b")
    b.symlink_to("loop_a")
    _repoint(victim, os.path.relpath(a, victim.parent))
    assert not _inside(snap, victim, blobs)


def test_directory_symlink_out_of_snapshot_is_rejected(tmp_path):
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(snap / "data", elsewhere, symlinks=False)
    shutil.rmtree(snap / "data")
    (snap / "data").symlink_to(elsewhere)
    assert not _inside(snap, _first_parquet(snap), blobs)


def test_dotdot_and_absolute_paths_are_rejected_even_for_hub(tmp_path):
    snap = _make_hub_cache(tmp_path)
    blobs = hub_blob_dirs(snap)
    blob = Path(os.path.realpath(_first_parquet(snap)))
    rel_to_blob = os.path.relpath(blob, snap)  # "../../blobs/<sha>"
    assert ".." in rel_to_blob
    assert not _inside(snap, snap / rel_to_blob, blobs)  # metadata path walking up into blobs
    assert not _inside(snap, snap / str(blob), blobs)  # absolute path in metadata
    assert not _inside(snap, snap / "../../../../etc/passwd", blobs)


def test_hub_blob_dirs_requires_cache_layout(tmp_path):
    plain = make_dataset(tmp_path / "plain", n_eps=2, length=40)
    assert hub_blob_dirs(plain) is None
    fake = tmp_path / "x" / "snapshots" / "rev"  # right leaf names, but not a datasets--* repo
    fake.mkdir(parents=True)
    (tmp_path / "x" / "blobs").mkdir()
    assert hub_blob_dirs(fake) is None
