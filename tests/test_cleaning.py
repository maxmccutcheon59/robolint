"""Cleaning plans: each finding code gets its own fix, and one remove wins over keep."""

from types import SimpleNamespace

from robolint.cleaning import plan, summary
from robolint.findings import Finding, Severity


def _result(findings):
    return SimpleNamespace(findings=findings)


def test_codes_map_to_their_fixes():
    r = _result(
        [
            Finding("FRAME_INDEX_DUPLICATE", Severity.ERROR, "stacked", episode=1),
            Finding("EPISODE_LENGTH_MISMATCH", Severity.ERROR, "length", episode=1),
            Finding("EPISODE_LENGTH_MISMATCH", Severity.ERROR, "length", episode=2),
            Finding("IDLE_PADDING", Severity.WARNING, "idle", episode=3, details={"lead_frames": 40, "tail_frames": 2}),
            Finding("FAILED_GRASP", Severity.ERROR, "missed", episode=4),
            Finding("ACTION_SPIKE", Severity.WARNING, "spike", episode=5),
            Finding("TIMESTAMP_JITTER", Severity.INFO, "minor", episode=6),
            Finding("TASKS_MISSING", Severity.ERROR, "dataset level"),
        ]
    )
    p = {x.episode: x for x in plan(r)}
    assert p[1].decision == "remove"
    assert p[2].decision == "keep" and p[2].actions[0]["action"] == "repair"
    assert p[3].trim == (37, 0)
    assert p[4].decision == "keep" and p[4].labels == ["failed_grasp"]
    assert p[5].actions[0]["action"] == "review"
    assert 6 not in p  # below the severity threshold
    assert summary(list(p.values())) == {"remove": 1, "repair": 1, "trim": 1, "label": 1, "review": 1}


def test_quality_model_flags_become_review():
    q = [
        SimpleNamespace(episode=9, flagged=True, likely_fault="stutter", p_bad=0.99),
        SimpleNamespace(episode=8, flagged=False, likely_fault=None, p_bad=0.1),
    ]
    p = plan(_result([]), quality=q)
    assert [x.episode for x in p] == [9] and p[0].actions[0]["code"] == "QUALITY_MODEL"
