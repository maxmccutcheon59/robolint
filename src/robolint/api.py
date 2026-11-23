"""Programmatic entry point: lint a dataset and get a structured result."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .checks import Config, all_checks
from .dataset import Dataset, hub_blob_dirs, load_dataset, resolve_source
from .findings import Finding, Scores, ScoreWeights, Severity, score


@dataclass
class Result:
    source: str
    dataset: Dataset
    findings: list[Finding]
    scores: Scores
    videos_checked: bool
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    @property
    def dataset_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.episode is None]

    def episode_findings(self, ep: int) -> list[Finding]:
        return [f for f in self.findings if f.episode == ep]

    def code_counts(self) -> list[tuple[str, Severity, int]]:
        """(code, worst severity, number of episodes or 1 for dataset-level) sorted by impact."""
        eps: dict[str, set] = {}
        sev: dict[str, Severity] = {}
        for f in self.findings:
            eps.setdefault(f.code, set()).add(f.episode)
            sev[f.code] = max(sev.get(f.code, f.severity), f.severity)
        rows = [(c, sev[c], len(e)) for c, e in eps.items()]
        return sorted(rows, key=lambda r: (-int(r[1]), -r[2], r[0]))

    def bad_episodes(self, min_severity: Severity = Severity.ERROR) -> list[int]:
        return sorted({f.episode for f in self.findings if f.episode is not None and f.severity >= min_severity})

    def to_dict(self) -> dict[str, Any]:
        ds = self.dataset
        sev = Counter(f.severity.label for f in self.findings)
        return {
            "tool": {"name": "robolint", "version": __version__},
            "source": self.source,
            "dataset": {
                "codebase_version": ds.version,
                "robot_type": ds.info.get("robot_type"),
                "fps": ds.fps,
                "episodes": len(ds.episodes),
                "frames": ds.total_frames,
                "cameras": ds.video_keys,
                "tasks": len(ds.tasks),
                "columns": columns_used(ds),
            },
            "score": self.scores.dataset_score,
            "grade": self.scores.grade,
            "grade_cap": self.scores.grade_cap,
            "score_weights": self.weights.to_dict(),
            "episodes": {
                "clean": self.scores.clean_episodes,
                "with_warnings": self.scores.episodes_with_warnings,
                "with_errors": self.scores.episodes_with_errors,
            },
            "videos_checked": self.videos_checked,
            "finding_counts": dict(sev),
            "issues": [{"code": c, "severity": s.label, "affected": n} for c, s, n in self.code_counts()],
            "episode_scores": {str(k): v for k, v in sorted(self.scores.episode_scores.items())},
            "findings": [f.to_dict() for f in self.findings],
            "seconds": round(self.seconds, 2),
        }


def columns_used(ds: Dataset) -> dict[str, list[str]]:
    """The data columns read as the action and the state, when they are not single columns of
    those names."""
    return {k: list(v) for k, v in ds.key_sources.items()}


def lint(
    source: str | Path,
    *,
    videos: bool = False,
    revision: str | None = None,
    cache_dir: str | None = None,
    max_episodes: int | None = None,
    config: Config | None = None,
    cache_mb: float | None = None,
) -> Result:
    """Check a dataset. `cache_mb` caps the decoded frame data kept in memory (default 512 MB);
    datasets larger than that are re-read from disk as checks need them."""
    t0 = time.perf_counter()
    cfg = config or Config()
    # Only a Hub download gets the relaxed symlink rule; a local folder keeps the strict one.
    from_hub = not Path(str(source)).expanduser().exists()
    root = resolve_source(
        str(source), include_videos=videos, revision=revision, cache_dir=cache_dir, max_episodes=max_episodes
    )
    hub_blobs = hub_blob_dirs(root) if from_hub else None
    ds = load_dataset(
        root,
        source=str(source),
        max_episodes=max_episodes,
        hub_blobs=hub_blobs,
        cache_mb=cache_mb,
        action_keys=cfg.action_keys,
        state_keys=cfg.state_keys,
    )
    from .checks.signals import configured_grippers

    configured_grippers(ds, cfg)  # an unknown gripper name or index stops here, before any check
    findings: list[Finding] = []
    for check in all_checks(include_video=videos):
        findings.extend(check(ds, cfg))
    findings.sort(key=lambda f: (f.episode if f.episode is not None else -1, -int(f.severity), f.code))
    scores = score(findings, [ep.index for ep in ds.episodes], cfg.score_weights)
    return Result(
        source=str(source),
        dataset=ds,
        findings=findings,
        scores=scores,
        videos_checked=videos,
        seconds=time.perf_counter() - t0,
        weights=cfg.score_weights,
    )
