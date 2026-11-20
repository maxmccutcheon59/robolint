"""Re-score the published 18-dataset audit from its stored findings with this tree's scorer.

    python benchmarks/audit_scores.py [space_static/results.json]

The audit's findings are kept in space_static/results.json (the robolint Space shows them). This
recomputes each dataset's score from those findings, without downloading anything, and prints it
next to the published one, so a change to the scoring can be judged on real datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from robolint.findings import Finding, Severity, score


def audit_results(path: Path) -> list[tuple[dict, list[Finding], list[int]]]:
    """(published result, its findings, its episode indices) per dataset. Clean episodes are not
    stored, so they get stand-in indices: only their number matters to the score."""
    out = []
    for r in json.loads(path.read_text(encoding="utf-8"))["results"]:
        findings = [
            Finding(f["code"], Severity[f["severity"].upper()], f.get("message", ""), f.get("episode"))
            for f in r["findings"]
        ]
        flagged = sorted({f.episode for f in findings if f.episode is not None})
        n = sum(r["episodes"].values())
        taken = set(flagged)
        spare = (i for i in range(n + len(flagged) + 1) if i not in taken)
        out.append((r, findings, flagged + [next(spare) for _ in range(max(0, n - len(flagged)))]))
    return out


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "space_static/results.json")
    changed = 0
    for r, findings, episodes in audit_results(path):
        s = score(findings, episodes)
        note = ""
        if s.grade != r["grade"]:
            changed += 1
            note = f"   grade changed from {r['grade']}"
        elif s.dataset_score != r["score"]:
            note = f"   was {r['score']:.1f}"
        print(f"{r['source']:<58} {s.dataset_score:5.1f} ({s.grade}){note}")
    print(f"\n{changed} grade{'s' if changed != 1 else ''} changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
