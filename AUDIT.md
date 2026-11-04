# What's broken in public robot learning data

*An audit of 18 public LeRobot datasets (4,218 episodes, about 2.7 million frames) with [robolint](README.md). November 2026.*

Robot data is expensive to collect and easy to get subtly wrong. None of the problems below crash training. They quietly make a policy worse: episodes where the robot never moves, recordings stacked on top of each other, metadata that no longer matches the data. We ran robolint over 18 of the most-downloaded LeRobot datasets on the Hugging Face Hub to see how common these problems are.

**Short version:** the simulation benchmarks are clean. Real-robot teleoperation data is where things break, including in official reference datasets, and 3 of the 8 community datasets have structural problems serious enough that we would not train on them as published.

Every number here is reproducible:

```bash
pip install "robolint[all] @ git+https://github.com/maxmccutcheon59/robolint.git"
robolint audit --from-file datasets.txt --out audit
```

## Results

| Dataset | Episodes | Score | Episodes with errors | Main issues |
|---|---:|---:|---:|---|
| EPITECH-LILLE/eval_training | 60 | 36.6 (F) | 20 (33%) | merged recordings, metadata says 0 episodes |
| kmg0620/so101_bomb_pick_place_v1 | 150 | 55.3 (F) | 22 (15%) | merged recordings, 47 episodes missing from metadata |
| jchandru08/so100_press_red_button | 29 | 55.5 (F) | 7 (24%) | metadata/data mismatch, missing episodes |
| SO-101-robot-learning/so101_pick_place | 50 | 78.5 (C) | 0 | metadata says 0 episodes |
| imstevenpmwork/omx_sentry_1 | 7 | 82.0 (B) | 0 | no task description |
| lerobot/svla_so100_stacking | 56 | 90.8 (A) | 5 (9%) | dead and false-start episodes |
| Embodied-AI-6/dataset | 160 | 91.0 (A) | 0 | 33% of all frames are idle |
| lerobot/svla_so100_sorting | 52 | 96.2 (A) | 2 (4%) | dead episodes |
| lerobot/aloha_sim_transfer_cube_scripted | 50 | 98.5 (A) | 0 | near-identical scripted demos |
| lerobot/svla_so101_pickplace | 50 | 98.7 (A) | 0 | idle padding |
| lerobot/svla_so100_pickplace | 50 | 98.9 (A) | 0 | idle padding |
| brandonyang/artem-pour-water | 1,445 | 99.8 (A) | 0 | 22 single-frame tracking glitches |
| lerobot/pusht | 206 | 100 (A) | 0 | none |
| lerobot/aloha_sim_transfer_cube_human | 50 | 100 (A) | 0 | none |
| lerobot/aloha_sim_insertion_human | 50 | 100 (A) | 0 | none |
| lerobot/aloha_sim_insertion_scripted | 50 | 100 (A) | 0 | none |
| lerobot/libero | 1,693 | 100 (A) | 0 | 2 minor glitches |
| recast-robotics/rollout_dagger_r1_s2_b0_20260915_121327 | 10 | 100 (A) | 0 | none |

An **error** means the episode is unusable for training as published. The score is roughly the share of the data you can train on (see [scoring](README.md#what-it-checks)).

## Findings

### 1. Recordings stacked on top of each other

The most serious problem we found. In two datasets, the data files contain two separate recordings under the same episode number: frame numbers restart halfway through the episode and timestamps jump backwards.

- **kmg0620/so101_bomb_pick_place_v1**: the metadata describes 103 episodes and 57,211 frames. The data files actually hold 150 episodes and 88,782 frames, 55% more than declared. Episodes 81–102 each contain two recordings (for example, episode 90: metadata says 512 frames, the data holds 1,225). 17,581 global frame indices are duplicated. The repository also carries a 40 MB `record.log`.
- **EPITECH-LILLE/eval_training**: `info.json` declares 0 episodes and 0 frames. The data holds 60 episodes and 41,686 frames, and episodes 0–19 each contain two recordings (about 596 frames per the metadata, about 1,042 in the data). 11,911 global indices are duplicated. Six leftover `tmp*` upload folders hold about 69 MB of partial video files.

This pattern usually comes from resuming or re-running a recording session into an existing dataset. A loader that indexes by episode will hand the policy two different demonstrations spliced together.

### 2. Metadata that no longer describes the data

- **SO-101-robot-learning/so101_pick_place**: `info.json` declares 0 episodes and 0 frames. The data holds 50 episodes and 17,870 frames. The episodes themselves look healthy, but tools that trust `info.json` will see an empty dataset.
- **jchandru08/so100_press_red_button**: `info.json` declares 19,028 frames, but the data holds 17,156. Three episode numbers between 0 and 31 are missing, and 7 episodes have a frame count that differs from their metadata by 1–3 frames, so any index built from the metadata points at slightly wrong frames.

### 3. Dead and false-start episodes, including in official datasets

- **lerobot/svla_so100_stacking**: episode 20 is a single frame. Episodes 37, 38, 39 and 55 never move: the commanded actions are constant from start to finish. Episodes 37 and 38 are also under a second long.
- **lerobot/svla_so100_sorting**: episodes 8 and 23 never move. Episode 23 is 51 frames, against a median of 694.

These are almost always recordings that were started by accident. Each one teaches the policy that doing nothing is a valid demonstration.

### 4. Idle time

Operators often start recording and then wait, or forget to stop.

- **Embodied-AI-6/dataset**: all 160 episodes have long idle stretches. 27,000 of 81,786 frames (33%) are the robot standing still before the task starts or after it ends; one episode is 57% idle.
- **lerobot/svla_so101_pickplace**: 26 of 50 episodes are at least 15% idle (21% on average).

Idle frames are not wrong, but in large amounts they bias a policy toward hesitating. Trimming them is cheap.

### 5. Smaller issues

- **imstevenpmwork/omx_sentry_1**: the task description is empty, so the data cannot condition a language-driven (VLA) policy.
- **jchandru08/so100_press_red_button**: in 5 episodes the right wrist is pushed into its hard stop mid-task for over two seconds, and in 14 the left shoulder's command sits past the joint's reach at rest (a leader/follower calibration offset; usually harmless).
- **brandonyang/artem-pour-water**: 22 of 1,445 episodes contain a single-frame jump in the tracked pose, typical of tracking dropouts. That's a very clean dataset for its size.
- **lerobot/aloha_sim_transfer_cube_scripted**: 5 scripted demos are near-identical to earlier ones, which is expected when a scripted policy repeats a setup, but worth knowing when counting effective dataset size.

### What was clean

The simulation datasets (PushT, ALOHA sim, LIBERO with 1,693 episodes) and the DAgger rollouts from recast-robotics came back essentially clean. The problems concentrate in real-robot teleoperation data and in how datasets are assembled and uploaded.

## How the audit was run

robolint checks timing (dropped frames, timestamps), actions and state (dead episodes, disconnected followers, glitches, idle padding, joint limits), episode structure (false starts, duplicates, metadata consistency) and, optionally, camera streams. Video was not checked in this audit. Every check is tested against synthetic datasets with injected faults, and this audit itself surfaced three false positives (sample-and-hold controllers, binary gripper commands, and an arm resting on its stop) that were fixed before these numbers were produced.

Thresholds are judgment calls, so findings are a starting point for review, not a verdict. If you maintain one of these datasets and think a finding is wrong, please [open an issue](https://github.com/maxmccutcheon59/robolint/issues).

*Audit by Max McCutcheon. Contact: MaxMcCutcheon1@outlook.com.*
