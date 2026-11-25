# A learned episode quality model

robolint's checks are hand-written rules. This study trains a model to catch faults the rules
were never written for, and measures honestly how much it adds.

**Model.** A gradient-boosted classifier (scikit-learn `HistGradientBoostingClassifier`) over 19
robot-agnostic features per episode: length, idle share and where it sits, action jumps and jerk,
frame-to-frame reversals, repeated frames, frozen state, action-to-state tracking, distance from the
dataset's typical trajectory, start and end poses, path length, range of motion, and returns to the
start pose. Every feature is measured relative to the episode's own dataset, so one model works
across arms, humanoids and bimanual rigs at any control rate. It predicts clean or one of 11 fault
types.

**Data.** 26 public LeRobot datasets (SO-100/SO-101 arms, Agilex, Galaxea, Airbot, AI2 Alphabot,
YAM, DROID, humanoid and bimanual rigs; up to 120 episodes each, 2,406 episodes). Training examples
are real episodes with no robolint warning or error, plus copies with realistic faults injected:
dead, frozen follower, glitches, idle padding, false starts, aborted demos, stacked recordings,
stutter (repeated frames), noisy teleop, mid-episode hesitation, and anomalous motion. Datasets are
built with 10%, 25% and 40% of episodes faulty, and features are measured against that mix, as
they would be on a real upload.

**Evaluation.** Every number is on datasets the model never trained on (grouped cross-validation by
dataset). The alarm threshold is set for 2% false alarms on clean episodes of held-out datasets.

| | Held-out result |
|---|---|
| Separates faulty from clean episodes (AUC) | 0.948 |
| Names the fault type correctly, when faulty | 84% |

**Rules vs model on the same faults** (`compare.py`: 24 held-out datasets, 25% of episodes
faulty, 621 faulty and 1,868 clean episodes):

| Fault | Rules | Model | Rules + model |
|---|---|---|---|
| Dead episode | 100% | 89% | 100% |
| Frozen follower | 100% | 59% | 100% |
| Teleop glitch | 97% | 37% | 97% |
| Idle padding | 100% | 81% | 100% |
| False start | 76% | 55% | 81% |
| Aborted demo | 100% | 90% | 100% |
| Stacked recordings | 0% | 33% | 33% |
| Stutter (repeated frames) | 4% | 85% | 86% |
| Noisy teleop | 15% | 82% | 83% |
| Hesitation | 1% | 77% | 79% |
| Anomalous motion | 20% | 24% | 41% |
| **All faults** | **55%** | **66%** | **82%** |
| False alarms on clean episodes | 1.0% | 0.6% | 1.6% |

**What it says**
1. Rules and model are complementary. Rules are exact on the faults they were written for; the
   model catches stutter, noisy teleop and hesitation, which the rules almost never see. Together
   they catch 82% of faults versus 55% for rules alone, at 1.6% false alarms.
2. On the real published episodes (`scan_real.py`) the model, at this strict threshold, flags 74
   of 2,406 episodes, 73 of which the rules also flag. It confirms the rules' worst findings but
   found only one new problem episode in these datasets. Its value is on fault types that are rare
   in these particular datasets, and on teams' private data.
3. Weak spots: stacked recordings (33%) and anomalous motion (41%) need sequence models or video;
   that is the next version.

**Limits.** Faults are injected, not collected; real faults may look different. The threshold is
conservative. Labels for "clean" come from robolint itself, so episodes the rules miss can be
mislabelled clean in training.

**Use.** `pip install 'robolint[model]'`, then `robolint check DATASET --model`. The JSON report
gains a `quality_model` list (per episode: `p_bad`, `flagged`, `likely_fault`).

**Reproduce.** `python fetch.py` (downloads the datasets; signals only), `python train.py` (writes
`model.pkl` and `results.json`), `python compare.py`, `python scan_real.py`. Runs on a CPU; with
2 cores, about 30 minutes in total.
