"""Scoring: what each finding costs, configurable weights, and stability on the published audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from robolint import lint
from robolint.checks import Config
from robolint.cli import main
from robolint.findings import Finding, ScoreWeights, Severity, score

from .synth import make_dataset

E, W, I = Severity.ERROR, Severity.WARNING, Severity.INFO
ROOT = Path(__file__).resolve().parents[1]


def f(code, sev, ep=None):
    return Finding(code, sev, "", ep)


def test_episode_scores():
    s = score(
        [
            f("A", W, 0),
            f("A", W, 0),  # the same code twice counts once
            f("A", W, 1),
            f("B", W, 1),
            f("C", I, 2),
            f("D", E, 3),
            f("A", W, 3),
        ],
        [0, 1, 2, 3, 4],
    )
    assert s.episode_scores == {0: 85.0, 1: 72.25, 2: 98.0, 3: 0.0, 4: 100.0}
    assert s.dataset_score == round((85 + 72.25 + 98 + 0 + 100) / 5, 1)


def test_warnings_stack_but_only_an_error_zeroes_an_episode():
    many = [f(f"W{i}", W, 0) for i in range(10)]
    s = score(many, [0])
    assert 0 < s.episode_scores[0] < 25  # before, 7 warnings already scored 0, the same as an error


def test_a_dataset_level_error_costs_more_than_a_warning_on_one_episode():
    eps = list(range(20))
    one_warning = score([f("A", W, 3)], eps).dataset_score
    dataset_error = score([f("FRAME_COUNT_MISMATCH", E)], eps).dataset_score
    assert dataset_error < one_warning  # before, 90.0 against 99.2: the dataset error cost less
    # It costs as much as a warning on every episode.
    assert dataset_error == score([f("A", W, i) for i in eps], eps).dataset_score == 85.0
    assert score([f("TASK_EMPTY", W)], eps).dataset_score == 95.0
    assert score([f("NOTE", I)], eps).dataset_score == 100.0


def test_nothing_to_train_on_scores_zero():
    assert score([f("NO_ACTIONS", E)], [0, 1, 2]).dataset_score == 0.0
    assert score([f("NO_EPISODES", E)], []).dataset_score == 0.0


def test_weights_can_be_changed():
    w = ScoreWeights().with_overrides({"warning": 0.5, "IDLE_PADDING": 0.0, "dataset-error": 0.0})
    s = score([f("IDLE_PADDING", W, 0), f("ACTION_SPIKE", W, 1), f("FRAME_COUNT_MISMATCH", E)], [0, 1], w)
    assert s.episode_scores == {0: 100.0, 1: 50.0} and s.dataset_score == 75.0
    for bad in ({"warning": 1.5}, {"dataset-error": -0.1}):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ScoreWeights().with_overrides(bad)
    with pytest.raises(ValueError, match="unknown score weight"):
        ScoreWeights().with_overrides({"severe": 0.3})


def test_weights_reach_the_result_and_the_cli(tmp_path):
    root = make_dataset(tmp_path / "ds", n_eps=10, faults={2: ["spike"]})
    base = lint(root)
    assert base.scores.episode_scores[2] == 85.0
    r = lint(root, config=Config(score_weights=ScoreWeights().with_overrides({"ACTION_SPIKE": 0.4})))
    assert r.scores.episode_scores[2] == 60.0
    assert r.to_dict()["score_weights"]["codes"]["ACTION_SPIKE"] == 0.4
    out = tmp_path / "r.json"
    assert main(["check", str(root), "--weight", "warning=0.4", "--json", str(out), "--no-color"]) == 0
    d = json.loads(out.read_text())
    assert d["episode_scores"]["2"] == 60.0 and d["score_weights"]["episode"]["warning"] == 0.4
    assert main(["check", str(root), "--weight", "warning", "--no-color"]) == 2
    assert main(["check", str(root), "--weight", "warning=2", "--no-color"]) == 2


def test_grades_on_the_published_audit_do_not_change():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        from audit_scores import audit_results
    finally:
        sys.path.pop(0)
    results = audit_results(ROOT / "space_static" / "results.json")
    assert len(results) == 18
    moved = {}
    for r, findings, episodes in results:
        s = score(findings, episodes)
        assert s.grade == r["grade"], r["source"]
        assert s.clean_episodes == r["episodes"]["clean"]
        if s.dataset_score != r["score"]:
            moved[r["source"]] = (r["score"], s.dataset_score)
    # Only datasets with dataset-level findings move; see the CHANGELOG.
    assert moved == {
        "EPITECH-LILLE/eval_training": (36.6, 40.9),
        "kmg0620/so101_bomb_pick_place_v1": (55.3, 52.4),
        "jchandru08/so100_press_red_button": (55.5, 54.9),
        "SO-101-robot-learning/so101_pick_place": (78.5, 71.1),
        "imstevenpmwork/omx_sentry_1": (82.0, 80.8),
    }
