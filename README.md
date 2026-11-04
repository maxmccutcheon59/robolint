# robolint

**Quality checks for robot learning datasets.** Point it at any [LeRobot](https://github.com/huggingface/lerobot) dataset, local or on the Hugging Face Hub, and find out which episodes will quietly hurt your policy before you spend GPU hours training on them.

```bash
pip install "robolint[all] @ git+https://github.com/maxmccutcheon59/robolint.git"
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

## Why

Robot data is expensive to collect and easy to get subtly wrong. A follower arm unplugs mid-session and you record minutes of actions with no motion. USB bandwidth drops camera frames and video drifts out of sync with actions. A teleop glitch makes a joint jump 150° in one step. An episode is uploaded twice. None of these crash training. They just make the policy worse, and nobody notices.

robolint finds them in seconds and tells you exactly which episodes to drop or fix.

## What it checks

| Area | Checks |
|---|---|
| **Timing** | dropped frames (timestamp gaps), timestamps going backwards, jitter, drift from declared fps, frame_index gaps and duplicates |
| **Actions & state** | NaN/inf values, dead episodes (robot never moves), frozen state (follower disconnected), teleop glitches (action spikes), idle lead-in/tail padding, gripper never actuating, joints pinned at limits |
| **Episodes** | false starts, truncated and runaway episodes, metadata/data length mismatches, near-duplicate episodes, missing or unknown tasks |
| **Video** (`--videos`) | missing or unreadable files, video/action length mismatches, black, blown-out, blank, and frozen camera feeds, fps and resolution mismatches |
| **Metadata** | declared vs actual episode/frame counts, missing features, shape mismatches, vague task descriptions that can't condition a VLA policy |

Every episode gets a 0–100 score and the dataset gets a score and letter grade (A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60). An **error** makes an episode unusable for training, so it scores 0; each distinct **warning** costs 15 points; **info** costs 2. The dataset score is the average episode score, minus small penalties for dataset-level problems. Roughly: the share of your data you can actually train on.

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

## Exit codes

`0` success · `1` score below `--fail-under` · `2` dataset could not be loaded.

## Security

Datasets are treated as untrusted input: no pickle, no code execution, file paths from metadata are confined to the dataset folder, and every string in HTML reports is escaped. Report vulnerabilities to MaxMcCutcheon1@outlook.com.

## License

MIT © Max McCutcheon
