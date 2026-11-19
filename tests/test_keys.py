"""Choosing the action and state columns, and naming grippers the names don't reveal."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robolint import lint
from robolint.checks import Config
from robolint.cli import main
from robolint.dataset import DatasetError, load_dataset, resolve_keys

from .synth import NAMES, make_dataset

KEYS = ("action", "observation.state")


def _data(root):
    return root / "data" / "chunk-000" / "file-000.parquet"


def _info(root):
    return json.loads((root / "meta" / "info.json").read_text())


def _write_info(root, info):
    (root / "meta" / "info.json").write_text(json.dumps(info))


def _bimanual(tmp_path, left_faults, right_faults, **kw):
    """A v3 dataset whose two arms are stored as separate columns: action.left_arm /
    action.right_arm and observation.state.left_arm / observation.state.right_arm. Each arm is a
    single-arm synthetic recording with its own faults; the camera (with videos=True) shows the
    left arm only."""
    left = make_dataset(tmp_path / "left", n_eps=12, grasp_object=True, faults=left_faults, seed=0, **kw)
    kw.pop("videos", None)
    right = make_dataset(tmp_path / "right", n_eps=12, grasp_object=True, faults=right_faults, seed=1, **kw)
    root = tmp_path / "bimanual"
    shutil.copytree(left, root)
    lt, rt = pq.read_table(_data(left)), pq.read_table(_data(right))
    cols = {n: lt.column(n) for n in lt.column_names if n not in KEYS}
    for key in KEYS:
        cols[f"{key}.left_arm"] = lt.column(key)
        cols[f"{key}.right_arm"] = rt.column(key)
    pq.write_table(pa.table(cols), _data(root))
    info = _info(root)
    for key in KEYS:
        spec = info["features"].pop(key)
        info["features"][f"{key}.left_arm"] = spec
        info["features"][f"{key}.right_arm"] = spec
    _write_info(root, info)
    return root


def _codes(r, *codes):
    return sorted((f.code, f.episode) for f in r.findings if f.code in codes)


def test_split_bimanual_columns_are_found_and_joined_left_then_right(tmp_path):
    root = _bimanual(tmp_path, {3: ["missed_grasp"]}, {7: ["missed_grasp"], 5: ["spike"]})
    r = lint(root)
    assert r.dataset.key_sources == {
        "action": ["action.left_arm", "action.right_arm"],
        "observation.state": ["observation.state.left_arm", "observation.state.right_arm"],
    }
    names = r.dataset.feature_names("action")
    assert names == [f"left_arm.{n}" for n in NAMES] + [f"right_arm.{n}" for n in NAMES]
    assert r.to_dict()["dataset"]["columns"]["action"] == ["action.left_arm", "action.right_arm"]
    # Before, this dataset got NO_ACTIONS and nothing else was checked.
    assert not _codes(r, "NO_ACTIONS", "FEATURE_MISSING", "SHAPE_MISMATCH")
    # Each arm's gripper is paired with its own follower by name, and each arm's faults are found.
    failed = [f for f in r.findings if f.code == "FAILED_GRASP"]
    assert sorted((f.episode, f.details["gripper"]) for f in failed) == [
        (3, "left_arm.gripper.pos"),
        (7, "right_arm.gripper.pos"),
    ]
    spikes = [f for f in r.findings if f.code == "ACTION_SPIKE"]
    assert [f.episode for f in spikes] == [5] and all(d.startswith("right_arm.") for d in spikes[0].details["dims"])


def test_video_checks_use_the_joined_columns(tmp_path):
    pytest.importorskip("av")
    # The camera sees only the left arm. Its frozen stretch is found through the joined columns,
    # and the right arm's motion, which the camera does not show, flags nothing.
    root = _bimanual(tmp_path, {4: ["freeze_stretch"]}, {}, videos=True, length=150)
    r = lint(root, videos=True)
    video = [(f.code, f.episode) for f in r.findings if f.code.startswith("VIDEO_") and f.severity.label != "info"]
    assert video == [("VIDEO_FREEZE", 4)]


def test_split_columns_match_the_same_data_as_single_columns(tmp_path):
    # One arm stored as joints + a scalar gripper column: same findings as the single-column original.
    faults = {2: ["missed_grasp"], 4: ["dead"], 6: ["spike"], 9: ["state_lag"]}
    root = make_dataset(tmp_path / "ds", n_eps=12, grasp_object=True, faults=faults)
    base = lint(root)
    t = pq.read_table(_data(root))
    cols = {n: t.column(n) for n in t.column_names if n not in KEYS}
    info = _info(root)
    for key in KEYS:
        a = np.stack(t.column(key).to_numpy(zero_copy_only=False))
        cols[f"{key}.joints"] = pa.array(list(a[:, :5]), pa.list_(pa.float32()))
        cols[f"{key}.gripper"] = pa.array(a[:, 5], pa.float32())
        info["features"].pop(key)
        info["features"][f"{key}.joints"] = {"dtype": "float32", "shape": [5], "names": NAMES[:5]}
        info["features"][f"{key}.gripper"] = {"dtype": "float32", "shape": [1], "names": None}
    pq.write_table(pa.table(cols), _data(root))
    _write_info(root, info)
    r = lint(root)
    # Order is by name ('gripper' sorts before 'joints'), the same for action and state.
    assert r.dataset.key_sources["action"] == ["action.gripper", "action.joints"]
    assert r.dataset.feature_names("action")[0] == "gripper"
    assert [(f.code, f.episode, f.severity) for f in r.findings] == [
        (f.code, f.episode, f.severity) for f in base.findings
    ]
    assert r.scores.episode_scores == base.scores.episode_scores


def test_explicit_keys_set_the_order_and_missing_ones_are_reported(tmp_path):
    root = _bimanual(tmp_path, {}, {})
    ds = load_dataset(
        root,
        action_keys=("action.right_arm", "action.left_arm"),
        state_keys=("observation.state.right_arm", "observation.state.left_arm"),
    )
    assert ds.feature_names("action")[0] == "right_arm.shoulder_pan.pos"
    assert ds.episodes[0].frames["action"].shape[1] == 12
    with pytest.raises(DatasetError, match="action key 'action.middle' not found"):
        load_dataset(root, action_keys=("action.middle",))


def test_an_existing_action_column_is_used_as_is():
    cols = {"action": 6, "action.extra": 2, "observation.state": 6, "observation.state.extra": 1}
    assert resolve_keys({}, cols) == ({}, [])
    assert resolve_keys({}, {"actions": 7, "observation.state": 7}) == ({"action": ["actions"]}, [])


def _unnamed_gripper(tmp_path, rename=None):
    root = make_dataset(tmp_path / "ds", n_eps=12, grasp_object=True, faults={4: ["missed_grasp"]})
    info = _info(root)
    for key in KEYS:
        info["features"][key]["names"] = rename
    _write_info(root, info)
    return root


@pytest.mark.parametrize("gripper", [5, "5"])
def test_unnamed_gripper_given_by_index(tmp_path, gripper):
    # LIBERO-style data: no dimension names, so the grasp checks could not find the gripper.
    root = _unnamed_gripper(tmp_path)
    assert not _codes(lint(root), "FAILED_GRASP")
    r = lint(root, config=Config(gripper_dims=(gripper,)))
    assert _codes(r, "FAILED_GRASP") == [("FAILED_GRASP", 4)]


def test_gripper_with_another_name_given_by_name(tmp_path):
    root = _unnamed_gripper(tmp_path, rename=NAMES[:5] + ["claw"])
    assert not _codes(lint(root), "FAILED_GRASP")
    # The state gripper defaults to the same name.
    r = lint(root, config=Config(gripper_dims=("claw",)))
    assert _codes(r, "FAILED_GRASP") == [("FAILED_GRASP", 4)]
    assert [f.details["gripper"] for f in r.findings if f.code == "FAILED_GRASP"] == ["claw"]


def test_unnamed_gripper_is_left_out_of_the_lag_check(tmp_path):
    root = _unnamed_gripper(tmp_path)
    from robolint.checks.lag import position_joints

    ds = load_dataset(root)
    # Without names, dims are paired by position and the gripper is treated as a joint...
    pj = position_joints(ds, Config())
    assert pj is not None and 5 in pj.action_dims.tolist()
    # ...unless it is named as a gripper: an object stops it short of the command, which is not lag.
    pj = position_joints(ds, Config(gripper_dims=(5,)))
    assert pj is not None and pj.action_dims.tolist() == [0, 1, 2, 3, 4]


def test_unknown_gripper_is_a_clear_error(tmp_path):
    root = _unnamed_gripper(tmp_path, rename=NAMES[:5] + ["claw"])
    with pytest.raises(DatasetError, match="gripper 'pinch' is not a dimension of 'action'"):
        lint(root, config=Config(gripper_dims=("pinch",)))
    with pytest.raises(DatasetError, match="outside 'action', which has 6"):
        lint(root, config=Config(gripper_dims=(6,)))
    assert main(["check", str(root), "--gripper", "pinch", "--no-color"]) == 2


def test_cli_column_and_gripper_options(tmp_path, capsys):
    root = _bimanual(tmp_path, {}, {})
    out = tmp_path / "r.json"
    args = ["check", str(root), "--json", str(out), "--no-color"]
    args += ["--action-key", "action.left_arm", "--action-key", "action.right_arm"]
    args += ["--state-key", "observation.state.left_arm", "--state-key", "observation.state.right_arm"]
    args += ["--gripper", "left_arm.gripper.pos", "--gripper", "11"]
    assert main(args) == 0
    assert "action read from action.left_arm + action.right_arm" in capsys.readouterr().out
    d = json.loads(out.read_text())
    assert d["dataset"]["columns"]["observation.state"] == ["observation.state.left_arm", "observation.state.right_arm"]
    assert not [f for f in d["findings"] if f["severity"] != "info"]
