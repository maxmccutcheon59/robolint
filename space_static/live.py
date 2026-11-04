"""Runs inside Pyodide: download one public LeRobot dataset from the Hub into the
in-browser filesystem, run robolint on it, and return the report as JSON."""

import json
import shutil
from pathlib import Path

from pyodide.http import pyfetch

from robolint import lint
from robolint.dataset import REPO_ID_RE, quick_patterns
from robolint.report import HELP

HUB = "https://huggingface.co"
MAX_BYTES = 400 * 1024 * 1024  # refuse to pull more than this into a browser tab


async def _get(url):
    resp = await pyfetch(url)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    return resp


async def run(repo_id, quick, max_episodes, progress):
    repo_id = repo_id.strip()
    if not REPO_ID_RE.match(repo_id):
        raise ValueError("Enter a Hugging Face dataset id like lerobot/pusht")
    progress("Listing dataset files…")
    tree = await (await _get(f"{HUB}/api/datasets/{repo_id}/tree/main?recursive=true")).json()
    sizes = {t["path"]: t.get("size", 0) for t in tree if t.get("type") == "file"}
    if "meta/info.json" not in sizes:
        raise ValueError(f"{repo_id} has no meta/info.json, so it is not a LeRobot dataset")

    root = Path("/tmp/ds") / repo_id.replace("/", "__")
    shutil.rmtree(root, ignore_errors=True)

    async def fetch(paths):
        for i, p in enumerate(paths, 1):
            if ".." in Path(p).parts:
                continue
            progress(f"Downloading {i}/{len(paths)}: {p}")
            data = await (await _get(f"{HUB}/datasets/{repo_id}/resolve/main/{p}")).bytes()
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
            f"{repo_id} needs {total / 1e6:.0f} MB of data files, too much for a browser tab. "
            "Use quick mode or the command-line tool."
        )
    await fetch(sorted(data_files))
    progress("Checking…")
    r = lint(root, max_episodes=max_episodes if quick else None)
    d = r.to_dict()
    d["source"] = repo_id
    d.pop("episode_scores", None)
    return json.dumps({"help": HELP, "result": d})
