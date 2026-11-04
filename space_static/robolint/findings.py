"""Findings, severities, and scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    INFO = 1
    WARNING = 2
    ERROR = 3

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    episode: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.label
        return d


# Points removed from an episode's score, counted once per distinct code.
# An error makes an episode unusable for training, so it zeroes the episode.
EPISODE_PENALTY = {Severity.ERROR: 100, Severity.WARNING: 15, Severity.INFO: 2}
# Points removed from the dataset score for dataset-level findings.
DATASET_PENALTY = {Severity.ERROR: 10, Severity.WARNING: 3, Severity.INFO: 0}


def grade(score: float) -> str:
    for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cutoff:
            return letter
    return "F"


@dataclass
class Scores:
    dataset_score: float
    grade: str
    episode_scores: dict[int, float]
    clean_episodes: int
    episodes_with_errors: int
    episodes_with_warnings: int


def score(findings: list[Finding], episode_indices: list[int]) -> Scores:
    per_ep: dict[int, dict[str, Severity]] = {i: {} for i in episode_indices}
    dataset_level: dict[str, Severity] = {}
    for f in findings:
        bucket = dataset_level if f.episode is None else per_ep.setdefault(f.episode, {})
        prev = bucket.get(f.code)
        if prev is None or f.severity > prev:
            bucket[f.code] = f.severity

    ep_scores: dict[int, float] = {}
    clean = with_err = with_warn = 0
    for ep, codes in per_ep.items():
        penalty = sum(EPISODE_PENALTY[s] for s in codes.values())
        ep_scores[ep] = float(max(0, 100 - penalty))
        sevs = set(codes.values())
        if Severity.ERROR in sevs:
            with_err += 1
        elif Severity.WARNING in sevs:
            with_warn += 1
        else:
            clean += 1

    base = sum(ep_scores.values()) / len(ep_scores) if ep_scores else 0.0
    ds_penalty = sum(DATASET_PENALTY[s] for s in dataset_level.values())
    final = round(max(0.0, min(100.0, base - ds_penalty)), 1)
    return Scores(
        dataset_score=final,
        grade=grade(final),
        episode_scores=ep_scores,
        clean_episodes=clean,
        episodes_with_errors=with_err,
        episodes_with_warnings=with_warn,
    )
