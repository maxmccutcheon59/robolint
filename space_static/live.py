"""Runs inside Pyodide: download one public LeRobot dataset from the Hub into the
in-browser filesystem, run robolint on it, and return the report as JSON.

The id is a Hub dataset id, optionally followed by a subfolder for repos that keep one
LeRobot dataset per folder: "owner/name" or "owner/name/sub/folder"."""

import json
import re
import shutil
from pathlib import Path

from pyodide.http import pyfetch

from robolint import lint
from robolint.audit_report import write_audit_report
from robolint.dataset import REPO_ID_RE, quick_patterns
from robolint.report import HELP

HUB = "https://huggingface.co"
MAX_BYTES = 400 * 1024 * 1024  # refuse to pull more than this into a browser tab
_SUB_RE = re.compile(r"^[\w.-]+(/[\w.-]+)*$")
_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


async def _get(url):
    resp = await pyfetch(url)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    return resp


def split_id(text):
    """'owner/name/sub/dir' -> ('owner/name', 'sub/dir'); 'owner/name' -> ('owner/name', '')."""
    parts = text.strip().strip("/").split("/")
    repo_id, sub = "/".join(parts[:2]), "/".join(parts[2:])
    if not REPO_ID_RE.match(repo_id) or (sub and (not _SUB_RE.match(sub) or ".." in sub.split("/"))):
        raise ValueError("Enter a Hugging Face dataset id like lerobot/pusht (optionally /subfolder)")
    return repo_id, sub


async def _list_files(repo_id, sub):
    """All files under the subfolder, following the Hub's paginated tree listing."""
    url = f"{HUB}/api/datasets/{repo_id}/tree/main{('/' + sub) if sub else ''}?recursive=true"
    files = {}
    while url:
        resp = await _get(url)
        for t in await resp.json():
            if t.get("type") == "file":
                files[t["path"]] = t.get("size", 0)
        headers = getattr(resp, "headers", None) or {}
        m = _NEXT_RE.search(headers.get("link", "") or headers.get("Link", "") or "")
        url = m.group(1) if m else None
    return files


async def run(text, quick, max_episodes, progress):
    repo_id, sub = split_id(text)
    prefix = f"{sub}/" if sub else ""
    progress("Listing dataset files…")
    listed = await _list_files(repo_id, sub)
    sizes = {p[len(prefix):]: s for p, s in listed.items() if p.startswith(prefix)}
    if "meta/info.json" not in sizes:
        folders = sorted({p.split("/")[0] for p in sizes if p.endswith("/meta/info.json")})
        hint = f" It holds LeRobot datasets in subfolders; try {repo_id}/{folders[0]}" if folders else ""
        raise ValueError(f"{text.strip()} has no meta/info.json, so it is not a LeRobot dataset.{hint}")

    root = Path("/tmp/ds") / text.strip().strip("/").replace("/", "__")
    shutil.rmtree(root, ignore_errors=True)

    async def fetch(paths):
        for i, p in enumerate(paths, 1):
            if ".." in Path(p).parts:
                continue
            progress(f"Downloading {i}/{len(paths)}: {p}")
            data = await (await _get(f"{HUB}/datasets/{repo_id}/resolve/main/{prefix}{p}")).bytes()
            dest = root / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    await fetch([p for p in sizes if p.startswith("meta/") and p.endswith((".json", ".jsonl", ".parquet"))])
    data_files = [p for p in sizes if p.startswith("data/") and p.endswith(".parquet")]
    if quick:
        pats = quick_patterns(root, max_episodes, include_videos=False)
        if pats:
            data_files = [p for p in pats if p in sizes]
    total = sum(sizes[p] for p in data_files)
    if total > MAX_BYTES:
        raise ValueError(
            f"{text.strip()} needs {total / 1e6:.0f} MB of data files, too much for a browser tab. "
            "Use quick mode or the command-line tool."
        )
    await fetch(sorted(data_files))
    progress("Checking…")
    source = text.strip().strip("/")
    r = lint(root, max_episodes=max_episodes if quick else None)
    r.source = source
    report = Path("/tmp/audit.html")
    write_audit_report(r, report)
    d = r.to_dict()
    d["source"] = source
    d.pop("episode_scores", None)
    return json.dumps({"help": HELP, "result": d, "audit_html": report.read_text(encoding="utf-8")})
