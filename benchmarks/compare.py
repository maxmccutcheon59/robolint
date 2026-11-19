"""Run two robolint source trees on the same datasets and diff their findings and scores.

    python benchmarks/compare.py OLD_SRC NEW_SRC DATASET [DATASET ...] [--videos] [-- extra lint kwargs as JSON]

OLD_SRC and NEW_SRC are folders that contain the `robolint` package (e.g. a `git worktree` of an
older commit's `src`). Exits 1 if any dataset's findings, episode scores or dataset score differ.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

RUN = """
import json, sys
sys.path.insert(0, {src!r})
from robolint import lint
{pre}
kw = json.loads({kw!r})
r = lint({path!r}, videos={videos}, **kw)
d = r.to_dict()
print(json.dumps({{k: d[k] for k in ("score", "grade", "episodes", "findings", "episode_scores", "issues")}}))
"""


def run(src: str, path: str, videos: bool, kw: str, pre: str = "") -> dict:
    out = subprocess.run(
        [sys.executable, "-c", RUN.format(src=src, path=path, videos=videos, kw=kw, pre=pre)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(out.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--new-kwargs", default="{}", help="JSON kwargs for the new tree's lint()")
    ap.add_argument(
        "--force-streaming",
        action="store_true",
        help="make the new tree use its two-pass large-dataset statistics even on small datasets",
    )
    a = ap.parse_args()
    bad = 0
    n_findings = 0
    for p in a.datasets:
        old = run(a.old, p, a.videos, "{}")
        pre = "import robolint.stream as _s; _s.EXACT_ROWS = 0" if a.force_streaming else ""
        new = run(a.new, p, a.videos, a.new_kwargs, pre)
        n_findings += len(old["findings"])
        if old != new:
            bad += 1
            print(f"DIFF {p}")
            for k in old:
                if old[k] != new[k]:
                    print(f"  {k}: {json.dumps(old[k])[:400]}\n  -> {json.dumps(new[k])[:400]}")
        else:
            print(f"same {p}  score {old['score']} ({old['grade']}), {len(old['findings'])} findings")
    print(f"\n{len(a.datasets) - bad}/{len(a.datasets)} datasets identical ({n_findings} findings compared)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
