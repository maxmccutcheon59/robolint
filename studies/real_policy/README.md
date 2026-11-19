# Does robolint filtering train a better real policy?

The toy study in `../filtering_bc` showed that dropping the episodes robolint flags raised a small behavior-cloning policy's success from 63.3% to 73.9% on heavily faulty data. This study asks the same question with a real policy on a standard benchmark: **ACT on the ALOHA sim transfer-cube task, trained with LeRobot.**

## Design

**Data.** `lerobot/aloha_sim_transfer_cube_human`: 50 human teleop demos, all successful, robolint grade A with no flagged episodes. `prepare.py` writes three copies through LeRobot's own dataset writer, so every arm trains on data encoded the same way:

| Dataset | What is broken |
|---|---|
| `clean` | nothing (the reference) |
| `light` | 6 of 50 episodes (12%): 1 dead, 1 frozen state, 2 action spikes, 1 idle padding, 1 failed grasp |
| `heavy` | 18 of 50 episodes (36%): 3 dead, 2 frozen state, 4 action spikes, 4 idle padding, 5 failed grasps |

The faults are the toy study's, made concrete for a 14-joint arm with a camera (`faults.py`). Each stays consistent where the real fault would:

- **Dead episode:** the robot was never driven. State, action and camera are all still.
- **Frozen state:** the joint readings stop updating (stale bus reads) while the arm and camera keep moving.
- **Action spikes:** three single-frame jumps in two arm joints' commands, each 30 to 50% of that joint's range.
- **Idle padding:** 40 to 60% of the episode's length in still frames before the take, 20 to 30% after.
- **Failed grasp:** the operator's real motion, replayed in the simulator with the cube moved to where the grasp misses (checked: no part of either arm touches it). The camera and joint readings come from the physics, so the grippers visibly close on air.

robolint runs on each dataset unchanged. Which episodes it flags is saved in `data/study.json`, next to the true labels.

**Arms.** The same ACT training run, differing only in which episodes are left out (`--dataset.exclude_episodes`):

| Arm | Episodes dropped |
|---|---|
| `raw` | none |
| `robolint_errors` | those robolint flags as errors |
| `robolint_all` | those flagged as errors or warnings |
| `clean_only` | every faulty episode (the true labels; the best any filter could do) |
| `no_faults` | trained on the `clean` dataset (the reference) |

Arms that drop exactly the same episodes share one run. Each run trains with LeRobot's ACT defaults (batch 8, ImageNet ResNet-18 backbone) for 50,000 steps, then `lerobot-eval` runs it on 50 simulated episodes. Every run uses the same eval seed, so every arm faces the same 50 cube placements. Training seeds are 1000, 1001 and 1002.

**Plans.**

| Plan | What it runs | Runs | Estimated GPU time |
|---|---|---|---|
| `quick` | heavy: raw vs robolint_errors, 1 seed | 2 | 3 to 5 h |
| `standard` | heavy: all four arms, plus no_faults, 3 seeds | 15 | 22 to 38 h |
| `full` | standard plus the light mix, 3 seeds | 27 | 40 to 70 h |

These estimates assume an 8 GB laptop GPU at 6 to 12 training steps a second and 5 to 15 minutes of evaluation per run. After the first run, `run_study.py` prints the measured time and how much is left. Finished runs are skipped, so you can stop with Ctrl+C and pick up again the next night. An interrupted run starts over.

## Setup (Windows with WSL2 and an NVIDIA GPU)

1. In Windows, update the NVIDIA driver (Game Ready or Studio, any 2026 release). Don't install a Linux NVIDIA driver inside WSL; WSL uses the Windows one.
2. In an Ubuntu 24.04 WSL terminal:

```bash
sudo apt update && sudo apt install -y python3.12-venv git libegl1 libegl-mesa0 libgl1 libosmesa6
git clone https://github.com/maxmccutcheon59/robolint.git
cd robolint/studies/real_policy
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True` and your GPU's name. LeRobot 0.7.0 needs Python 3.12 or newer. `requirements.txt` pins it, because the commands below were checked against that version's source.

The simulator renders with `MUJOCO_GL=egl` (the scripts set it). If rendering fails, run `export MUJOCO_GL=osmesa` and try again. It's slower, but it works anywhere.

## Run

```bash
source .venv/bin/activate
python prepare.py                    # ~20 min on CPU: download, build clean/light/heavy, run robolint
python run_study.py --smoke          # ~5 min: 20 training steps + 2 short eval episodes, to check the setup
python run_study.py --plan quick     # the first real numbers
python run_study.py --plan standard  # the full heavy-fault comparison (resumes where quick left off)
python report.py                     # table, results.json and results.png
```

Before spending GPU time, check what `prepare.py` prints at the end. It shows how many episodes of each fault robolint caught and how many clean episodes it wrongly flagged. That table is half the result.

Useful options:

- `--batch-size 16` on a GPU with 12 GB or more.
- `--amp` for mixed precision, which is faster and uses less memory.
- `--steps 100000` to match LeRobot's published ACT recipe.
- `--eval-episodes 100` to make each run's success rate tighter.
- `--dry-run` prints every command without running anything.

Each run's logs are in `runs/<name>/train.log` and `eval.log`.

## What was checked without a GPU

On a CPU machine with LeRobot 0.7.0 and gym-aloha 0.1.5 installed, using 10 scripted stand-in episodes in the transfer-cube scene (`python prepare.py --standin 10`, the source dataset's exact schema):

- `prepare.py` built all three datasets with LeRobot's writer and injected all five faults. The failed grasp was re-simulated: the follower gripper settled at 0.10 (closed on air) instead of about 0.63 (closed on the cube).
- robolint flagged every injected fault: DEAD_EPISODE, STATE_FROZEN and FAILED_GRASP as errors, ACTION_SPIKE and IDLE_PADDING as warnings. It flagged none of the clean episodes.
- `run_study.py --smoke` ran `lerobot-train` (20 steps on CPU) and `lerobot-eval` in the ALOHA simulator for both quick-plan arms. The robolint arm trained on 7 of 10 episodes, confirming that `--dataset.exclude_episodes` works.
- `report.py` produced the table and chart from those runs.

Not checked: the real Hub dataset (this machine couldn't reach the Hub), GPU training, and the success rates themselves. Those are what the study is for.
