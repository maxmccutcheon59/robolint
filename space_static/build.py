"""Build the static Hugging Face Space.

Usage (from the repo root):
    python space_static/build.py path/to/audit
where path/to/audit is the folder written by `robolint audit` (it holds reports/*.json).

Writes space_static/results.json (precomputed audit results) and copies the pure-Python
robolint package into space_static/robolint/ so the in-browser live check can import it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" / "robolint"
sys.path.insert(0, str(HERE.parent / "src"))
from robolint.report import HELP  # noqa: E402

KEEP = ("source", "dataset", "score", "grade", "episodes", "issues", "findings", "videos_checked")


def main(audit_dir: str) -> None:
    reports = sorted(Path(audit_dir, "reports").glob("*.json"))
    if not reports:
        raise SystemExit(f"no reports in {audit_dir}/reports")
    results = []
    for p in reports:
        d = json.loads(p.read_text(encoding="utf-8"))
        results.append({k: d[k] for k in KEEP if k in d})
    results.sort(key=lambda r: (r["score"], r["source"]))
    out = {"generated_by": "robolint audit", "audited": "November 2026", "help": HELP, "results": results}
    (HERE / "results.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    dest = HERE / "robolint"
    if dest.exists():
        shutil.rmtree(dest)
    files = []
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / rel)
        files.append(str(Path("robolint") / rel).replace("\\", "/"))
    (HERE / "robolint_files.json").write_text(json.dumps(files), encoding="utf-8")
    print(f"{len(results)} datasets -> results.json; {len(files)} package files copied")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "audit")
