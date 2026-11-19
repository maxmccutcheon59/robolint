# Does filtering with robolint train a better policy?

A controlled study on a simulated pick-and-place task. It is a first test of the idea, not proof on real robots.

**Setup.** A scripted teleoperator records 200 demonstrations per dataset. A share of them is corrupted with the faults found in community LeRobot datasets: dead episodes, a disconnected follower (state frozen), teleop glitches (action spikes), long idle padding, and failed demos (the gripper closes short of the object). Each dataset is saved in LeRobot v3 format and checked by robolint unchanged. The same behavior-cloning policy (an MLP) is trained on four versions of the data and evaluated on the same 300 held-out task setups. Each condition uses 4 datasets × 3 training seeds per fault mix.

![Policy success by training data](filtering-study.png)

**Detection.** Across 1,208 clean episodes robolint flagged none. It caught 100% of dead and frozen-state episodes (as errors), 100% of glitch and idle-padding episodes (as warnings), and 0% of failed demos.

**Results** (task success; paired differences across the 12 runs per mix):

| | Light faults (13%) | Heavy faults (36%) |
|---|---|---|
| Raw | 79.4% | 62.8% |
| robolint: drop errors | 80.9% (+1.5, better in 8/12) | 66.5% (+3.7, better in 9/12) |
| robolint: drop errors + warnings | 80.8% (+1.3) | 61.4% (−1.4) |
| Clean, successful demos only | 81.1% | 69.1% |

**What it says**
1. Dropping episodes robolint marks as errors helps, more as contamination grows.
2. Dropping warning-level episodes does not help on heavy data: those episodes still hold good demonstrations. Idle padding should be trimmed and glitches repaired, not whole episodes thrown away.
3. Failed demos are the biggest remaining loss: removing them adds 7.7 points on heavy data (better in 11/12 runs), and robolint cannot detect them yet.

**Limits.** A toy 2D task, a simple policy, fault rates chosen to resemble community data, and effects within a few points with real run-to-run variance. The next step is the same comparison with a real policy (ACT or diffusion) on LeRobot sim tasks.

Reproduce: `python study.py` (about 10 minutes on 2 CPU cores), then `python chart.py`.
