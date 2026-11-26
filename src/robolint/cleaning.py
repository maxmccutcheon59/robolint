"""Fault-specific cleaning plans.

Dropping every flagged episode throws away good data: on real datasets it made policies worse
(see studies/cleaning). Each finding code maps to the fix that suits it:

- remove: the recording itself is broken (dead, frozen follower, stacked recordings, non-finite
  values, unreadable camera)
- repair: the motion is fine, the bookkeeping is wrong (length or index mismatches, task labels)
- trim: cut idle frames at the start and end, keep the rest
- label: keep, but mark it (failed grasps and re-grasps teach recovery)
- review: needs a person to look (spikes, joint limits, calibration, lag)

An episode is removed if any of its findings says remove; otherwise it is kept with every repair,
trim and label its findings call for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .findings import Severity

REMOVE = "remove"
REPAIR = "repair"
TRIM = "trim"
LABEL = "label"
REVIEW = "review"

ACTION_FOR_CODE: dict[str, str] = {
    # broken recordings
    "DEAD_EPISODE": REMOVE,
    "EMPTY_EPISODE": REMOVE,
    "EPISODE_TOO_SHORT": REMOVE,
    "STATE_FROZEN": REMOVE,
    "DUPLICATE_EPISODE": REMOVE,
    "FRAME_INDEX_DUPLICATE": REMOVE,
    "TIMESTAMP_NONMONOTONIC": REMOVE,
    "NONFINITE_VALUES": REMOVE,
    "TIMESTAMP_NONFINITE": REMOVE,
    "SHAPE_MISMATCH": REMOVE,
    "VIDEO_BLACK": REMOVE,
    "VIDEO_BLANK": REMOVE,
    "VIDEO_FROZEN": REMOVE,
    "VIDEO_UNREADABLE": REMOVE,
    "VIDEO_MISSING": REMOVE,
    # bookkeeping errors: the motion data is fine
    "EPISODE_LENGTH_MISMATCH": REPAIR,
    "FRAME_COUNT_MISMATCH": REPAIR,
    "FRAME_INDEX_OFFSET": REPAIR,
    "FRAME_INDEX_GAP": REPAIR,
    "GLOBAL_INDEX_GAPS": REPAIR,
    "GLOBAL_INDEX_DUPLICATES": REPAIR,
    "EPISODE_INDEX_GAPS": REPAIR,
    "EPISODE_COUNT_MISMATCH": REPAIR,
    "TIMESTAMP_OFFSET": REPAIR,
    "TIMESTAMP_DRIFT": REPAIR,
    "TASK_EMPTY": REPAIR,
    "EPISODE_TASK_EMPTY": REPAIR,
    "TASK_INDEX_UNKNOWN": REPAIR,
    "VIDEO_ACTION_OFFSET": REPAIR,
    "VIDEO_LAG": REPAIR,
    "VIDEO_LENGTH_MISMATCH": REPAIR,
    # idle time at the edges
    "IDLE_PADDING": TRIM,
    # real attempts worth keeping
    "FAILED_GRASP": LABEL,
    "REGRASP": LABEL,
}


@dataclass
class EpisodePlan:
    episode: int
    decision: str  # "remove" or "keep"
    actions: list[dict] = field(default_factory=list)  # {"action", "code", "detail"}
    trim: tuple[int, int] | None = None  # (frames to cut at start, frames to cut at end)
    labels: list[str] = field(default_factory=list)


def plan(result, min_severity: Severity = Severity.WARNING, quality: list | None = None) -> list[EpisodePlan]:
    """Build a cleaning plan from a robolint Result (and optional quality-model scores)."""
    by_ep: dict[int, EpisodePlan] = {}
    for f in result.findings:
        if f.episode is None or (f.severity < min_severity and f.code != "IDLE_PADDING"):
            continue
        p = by_ep.setdefault(f.episode, EpisodePlan(f.episode, "keep"))
        act = ACTION_FOR_CODE.get(f.code, REVIEW)
        p.actions.append({"action": act, "code": f.code, "detail": f.message})
        if act == REMOVE:
            p.decision = "remove"
        elif act == TRIM:
            lead = int(f.details.get("lead_frames", 0))
            tail = int(f.details.get("tail_frames", 0))
            keep_margin = 3  # leave a few still frames so the policy sees the start and end state
            p.trim = (max(0, lead - keep_margin), max(0, tail - keep_margin))
        elif act == LABEL:
            p.labels.append(f.code.lower())
    for q in quality or []:
        if q.flagged and q.episode not in by_ep:
            by_ep[q.episode] = EpisodePlan(
                q.episode,
                "keep",
                [
                    {
                        "action": REVIEW,
                        "code": "QUALITY_MODEL",
                        "detail": f"quality model: likely {q.likely_fault} (p={q.p_bad:.2f})",
                    }
                ],
            )
    return [by_ep[k] for k in sorted(by_ep)]


def summary(plans: list[EpisodePlan]) -> dict[str, int]:
    out = {"remove": 0, "repair": 0, "trim": 0, "label": 0, "review": 0}
    for p in plans:
        if p.decision == "remove":
            out["remove"] += 1
            continue
        kinds = {a["action"] for a in p.actions}
        for k in ("repair", "trim", "label", "review"):
            out[k] += k in kinds
    return out


def write_plan(plans: list[EpisodePlan], path: Path) -> dict[str, int]:
    s = summary(plans)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps({"summary": s, "episodes": [asdict(p) for p in plans]}, indent=1))
    tmp.replace(path)
    return s
