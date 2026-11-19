# robolint

[![CI](https://github.com/maxmccutcheon59/robolint/actions/workflows/ci.yml/badge.svg)](https://github.com/maxmccutcheon59/robolint/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/maxmccutcheon59/robolint/blob/main/LICENSE)

**Quality checks for robot learning datasets.** Point it at any [LeRobot](https://github.com/huggingface/lerobot) dataset, local or on the Hugging Face Hub, and find out which episodes will quietly hurt your policy before you spend GPU hours training on them.

```bash
pip install "robolint[all]"
robolint check lerobot/svla_so101_pickplace --videos --html report.html
```

```text
robolint report: demo_ds
  format v3.0 · robot so101_follower · 40 episodes · 7,032 frames · 30.0 fps · cameras: observation.images.front

  Score 85.2/100  (grade B)
  clean episodes 30 (75%) · with warnings 5 (12%) · with errors 5 (12%)

  Issues
  error    DEAD_EPISODE               1 episode
           Commanded actions are constant. Drop the episode; it teaches the policy to do nothing.
  error    STATE_FROZEN               1 episode
           State never updates while actions change. Check the follower arm connection; drop the episode.
  error    VIDEO_BLACK                1 episode
           Camera shows black frames: lens cap, disconnect, or no light.
  warning  DUPLICATE_EPISODE          1 episode
           Nearly identical to another episode: duplicated upload or replay. ...
```

*(Output from a test dataset with deliberately injected faults.)*

**New: [What's broken in public robot learning data](https://github.com/maxmccutcheon59/robolint/blob/main/AUDIT.md)**, an audit of 18 public LeRobot datasets.

[![Data quality scores for 18 public LeRobot datasets](https://raw.githubusercontent.com/maxmccutcheon59/robolint/main/docs/audit-scores.png)](https://github.com/maxmccutcheon59/robolint/blob/main/AUDIT.md)

**Try it in your browser:** the [robolint Space](https://huggingface.co/spaces/maxmccutcheon59/robolint) checks any public LeRobot dataset without installing anything (experimental; no camera checks).

## Why

Robot data is expensive to collect and easy to get subtly wrong. A follower arm unplugs mid-session and you record minutes of actions with no motion. USB bandwidth drops camera frames and video drifts out of sync with actions. A teleop glitch makes a joint jump 150° in one step. An episode is uploaded twice. None of these crash training. They just make the policy worse, and nobody notices.

robolint finds them in seconds and tells you exactly which episodes to drop or fix.

## What it checks

| Area | Checks |
|---|---|
| **Timing** | dropped frames (timestamp gaps), timestamps going backwards, jitter, drift from declared fps, frame_index gaps and duplicates |
| **Actions & state** | NaN/inf values, dead episodes (robot never moves), frozen state (follower disconnected), teleop glitches (action spikes), idle lead-in/tail padding, gripper never actuating, **failed grasps** (the gripper closes on nothing: a failed demo) and recovered re-grasps, joints pinned at limits, **action-to-state lag** (the dataset's typical delay between command and measured state, and episodes whose delay is far off it) |
| **Episodes** | false starts, truncated and runaway episodes, metadata/data length mismatches, near-duplicate episodes, missing or unknown tasks |
| **Video** (`--videos`) | missing or unreadable files, **frames missing mid-episode** (gaps in the frame timestamps of every file, read without decoding), duplicate or out-of-order frame timestamps, video/action length mismatches, black, blown-out, blank, and frozen camera feeds, **frozen stretches mid-episode** (the image stops changing for a second or more while the joints that camera sees keep moving), fps and resolution mismatches in every file, **action/video sync** (each camera's typical delay behind the commanded motion, and episodes whose video is shifted in time against the actions) |
| **Metadata** | declared vs actual episode/frame counts, missing features, shape mismatches, vague task descriptions that can't condition a VLA policy |



Every episode gets a 0–100 score and the dataset gets a score and letter grade (A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60). An **error** makes an episode unusable for training, so it scores 0; each distinct **warning** costs 15 points; **info** costs 2. The dataset score is the average episode score, minus small penalties for dataset-level problems. A heuristic summary: 100 minus penalties per issue, not a measure of usable data.

Supports LeRobot **v2.x** (per-episode files) and **v3.x** (chunked files) layouts.

## Usage

```bash
# Local folder or Hub repo id
robolint check ./my_dataset
robolint check your-org/your-dataset --videos

# Shareable reports
robolint check your-org/your-dataset --json report.json --html report.html

# List bad episodes so you can drop them before training
robolint check your-org/your-dataset --exclude-out bad_episodes.txt
robolint check your-org/your-dataset --exclude-out flagged.txt --exclude-level warning

# Gate a data pipeline in CI
robolint check ./my_dataset --fail-under 85

# Spot-check a huge dataset: downloads only the files for the first N episodes
robolint check cadene/droid_1.0.1 --max-episodes 200

# Cap the decoded data kept in memory (default 512 MB); bigger datasets are streamed from disk
robolint check ./my_big_dataset --cache-mb 256

# Audit many datasets at once
robolint audit lerobot/pusht lerobot/svla_so101_pickplace --out audit/
robolint audit --from-file datasets.txt --out audit/
```

From Python:

```python
from robolint import lint

result = lint("lerobot/pusht")
print(result.scores.dataset_score, result.scores.grade)
for f in result.findings:
    print(f.episode, f.severity.label, f.code, f.message)
```

Install options: `robolint` (local datasets), `robolint[hub]` (Hub downloads), `robolint[video]` (camera checks), `robolint[all]`.

## How the video checks work

Frame timestamps come from the container's packets, which are read without decoding, so every
frame of every file is checked. A gap of more than 1.5 frame periods inside an episode is
reported as missing frames. That is an error, because loaders that fetch frames by timestamp
(LeRobot does) fail or return the wrong frame there. Each file's frame rate is measured from the
regular spacing of its timestamps, so lost frames don't make a camera look slow.

Images are decoded sparingly. To find frozen stretches, about four frames per second are decoded:
only key frames when the encoding has them that often (LeRobot encodes every second frame as one),
otherwise the episode is decoded in full and sampled. For each camera, robolint fits how much the
image changes for a given movement of each joint, with non-negative weights, so joints that
camera does not see (the other arm of a bimanual robot, say) get no weight. It also measures how
much the image flickers when nothing moves. A stretch is flagged when, for at least a second, the
image changes less than that flicker while the joints the camera sees keep moving, and the same
camera is live elsewhere in the episode. The same fit stops the whole-episode frozen check from
flagging a camera when only joints it cannot see moved. When a camera's image change isn't
explained well by joint movement, or the dataset has no joints whose state follows the command,
frozen stretches are not looked for.

On a synthetic 24-episode bimanual recording with two cameras (AV1, sensor noise, pauses, one arm
often resting; `benchmarks/video.py`), `--videos` takes 11.8 s, against 10.9 s before this change. It finds
exactly the two injected faults and nothing else. Before, it missed both faults and wrongly
flagged 6 episodes as frozen. The same holds with a rendered (noise-free) image, sparse key frames
(H.264, every 2 s) and coarse encoding, except that the frozen stretch, which falls at the end of a
motion, is only caught when the camera has sensor noise.

## Large datasets

robolint does not load a dataset into memory. It reads each data file once to validate it and map
its rows to episodes, then decodes columns on demand while the checks walk the episodes, keeping
at most `--cache-mb` (default 512 MB) of decoded columns. Dataset-wide statistics (per-joint
ranges and glitch thresholds) are exact: quantiles are found in two passes from a sample and the
values near each answer, giving the same numbers as `numpy.quantile` without holding every value.
Memory is bounded by the cache, one data file and one episode, not by the size of the dataset.

Measured on synthetic 14-dimensional, 50 fps datasets (`benchmarks/scale.py`, one process per run):

| Frames | Parquet on disk | Peak memory before | Peak memory after | Time before | Time after |
|---|---|---|---|---|---|
| 500,000 | 61 MB | 777 MB | 435 MB | 13 s | 13 s |
| 2,000,000 | 240 MB | 2,160 MB | 567 MB | 58 s | 52 s |
| 10,000,000 | 1,197 MB | 7,281 MB | 1,115 MB | 320 s | 328 s |
| 10,000,000, `--cache-mb 128` | 1,197 MB | | 721 MB | | 416 s |

Before, memory grew by about 0.7 KB per frame, so 20 million frames needed about 14 GB. After, it
stops growing once the cache is full: the default 512 MB cache plus about 0.5 GB of working memory.
A smaller cache trades memory for time spent re-reading columns.

The findings and scores are identical before and after at every size, and on 69 synthetic
datasets covering every fault robolint detects (`benchmarks/corpus.py`, `benchmarks/compare.py`).

## Exit codes

`0` success · `1` score below `--fail-under` · `2` dataset could not be loaded.

## Security

Datasets are treated as untrusted input: no pickle, no code execution, file paths from metadata are confined to the dataset folder, and every string in HTML reports is escaped. Report vulnerabilities to MaxMcCutcheon1@outlook.com.

## License

MIT © Max McCutcheon
