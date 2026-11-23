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


GRADES = "ABCDF"
DEFAULT_GRADE_CAPS: tuple[tuple[float, str], ...] = ((0.0, "B"), (0.05, "C"), (0.15, "D"), (0.30, "F"))


@dataclass(frozen=True)
class ScoreWeights:
    """How much of a score each finding takes away: a weight w keeps (1 - w) of it.

    `episode` applies to findings about one episode, `dataset` to findings about the whole
    dataset, and `codes` overrides either for particular finding codes. Each code counts once per
    episode (or once for the dataset), at its worst severity.
    """

    # An error makes an episode unusable for training, so it zeroes the episode.
    episode: dict[Severity, float] = field(
        default_factory=lambda: {Severity.ERROR: 1.0, Severity.WARNING: 0.15, Severity.INFO: 0.02}
    )
    # A dataset-level error (mostly metadata that no longer matches the data) costs as much as a
    # warning on every episode; a dataset-level warning a third of that.
    dataset: dict[Severity, float] = field(
        default_factory=lambda: {Severity.ERROR: 0.15, Severity.WARNING: 0.05, Severity.INFO: 0.0}
    )
    # Problems that leave nothing to train on zero the whole dataset.
    codes: dict[str, float] = field(default_factory=lambda: {"NO_ACTIONS": 1.0, "NO_EPISODES": 1.0})
    # The share of episodes with errors caps the grade, whatever the average score: (share, best
    # grade allowed when the share is above it). An A means there is nothing to drop before
    # training, and a dataset that has to lose many episodes can't grade well because the rest
    # are clean.
    grade_caps: tuple[tuple[float, str], ...] = DEFAULT_GRADE_CAPS

    def __post_init__(self) -> None:
        for table in (self.episode, self.dataset, self.codes):
            for k, w in table.items():
                if not (isinstance(w, (int, float)) and 0.0 <= w <= 1.0):
                    raise ValueError(f"score weight for {getattr(k, 'label', k)} must be between 0 and 1, got {w!r}")
        for share, letter in self.grade_caps:
            if not (isinstance(share, (int, float)) and 0.0 <= share < 1.0) or letter not in GRADES:
                raise ValueError(
                    f"grade cap must be (share in [0, 1), one of {', '.join(GRADES)}), got {(share, letter)!r}"
                )
        for table in (self.episode, self.dataset):
            missing = [s.label for s in Severity if s not in table]
            if missing:
                raise ValueError(f"score weights are missing for: {', '.join(missing)}")

    def weight(self, code: str, severity: Severity, dataset_level: bool) -> float:
        if code in self.codes:
            return float(self.codes[code])
        return float((self.dataset if dataset_level else self.episode)[severity])

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": {s.label: self.episode[s] for s in Severity},
            "dataset": {s.label: self.dataset[s] for s in Severity},
            "codes": dict(sorted(self.codes.items())),
            "grade_caps": [{"error_episodes_over": share, "max_grade": g} for share, g in sorted(self.grade_caps)],
        }

    def with_overrides(self, overrides: dict[str, float]) -> ScoreWeights:
        """A copy with weights changed by key: 'error', 'warning' or 'info' (episode findings),
        'dataset-error', 'dataset-warning' or 'dataset-info', or a finding code."""
        episode, dataset, codes = dict(self.episode), dict(self.dataset), dict(self.codes)
        for key, w in overrides.items():
            k = key.strip().lower()
            if k.startswith("dataset-") and k[8:] in SEVERITY_BY_LABEL:
                dataset[SEVERITY_BY_LABEL[k[8:]]] = w
            elif k in SEVERITY_BY_LABEL:
                episode[SEVERITY_BY_LABEL[k]] = w
            elif key.strip().isupper() or "_" in key:
                codes[key.strip().upper()] = w
            else:
                raise ValueError(
                    f"unknown score weight {key!r}: use error, warning, info, dataset-error, dataset-warning, "
                    "dataset-info or a finding code such as IDLE_PADDING"
                )
        return ScoreWeights(episode=episode, dataset=dataset, codes=codes, grade_caps=self.grade_caps)


SEVERITY_BY_LABEL = {s.label: s for s in Severity}


def grade(score: float) -> str:
    for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
        if score >= cutoff:
            return letter
    return "F"


def capped_grade(letter: str, error_share: float, caps: tuple[tuple[float, str], ...]) -> tuple[str, str | None]:
    """The grade after the error-episode caps, and the cap that lowered it (None if none did)."""
    worst = None
    for share, cap in caps:
        if error_share > share and (worst is None or GRADES.index(cap) > GRADES.index(worst[1])):
            worst = (share, cap)
    if worst is None or GRADES.index(worst[1]) <= GRADES.index(letter):
        return letter, None
    share, cap = worst
    limit = "any" if share == 0 else f"over {share:.0%} of"
    return cap, f"{error_share:.1%} of episodes have errors; {limit} episodes with errors caps the grade at {cap}"


@dataclass
class Scores:
    dataset_score: float
    grade: str
    episode_scores: dict[int, float]
    clean_episodes: int
    episodes_with_errors: int
    episodes_with_warnings: int
    grade_cap: str | None = None  # why the grade is lower than the score alone would give


def score(findings: list[Finding], episode_indices: list[int], weights: ScoreWeights | None = None) -> Scores:
    """Score each episode and the dataset (see ScoreWeights and the README's "Scoring" section).

    An episode starts at 100 and each distinct finding code keeps (1 - weight) of it, so warnings
    stack toward zero but only an error reaches it. The dataset score is the average episode
    score, scaled the same way by each dataset-level finding. The score summarizes how many and
    how serious the issues are; it is not a measured prediction of training results.
    """
    weights = weights or ScoreWeights()
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
        kept = 1.0
        for code, sev in codes.items():
            kept *= 1.0 - weights.weight(code, sev, dataset_level=False)
        ep_scores[ep] = round(100.0 * kept, 2)
        sevs = set(codes.values())
        if Severity.ERROR in sevs:
            with_err += 1
        elif Severity.WARNING in sevs:
            with_warn += 1
        else:
            clean += 1

    base = sum(ep_scores.values()) / len(ep_scores) if ep_scores else 0.0
    kept = 1.0
    for code, sev in dataset_level.items():
        kept *= 1.0 - weights.weight(code, sev, dataset_level=True)
    final = round(max(0.0, min(100.0, base * kept)), 1)
    error_share = with_err / len(per_ep) if per_ep else 0.0
    letter, cap = capped_grade(grade(final), error_share, weights.grade_caps)
    return Scores(
        dataset_score=final,
        grade=letter,
        grade_cap=cap,
        episode_scores=ep_scores,
        clean_episodes=clean,
        episodes_with_errors=with_err,
        episodes_with_warnings=with_warn,
    )
