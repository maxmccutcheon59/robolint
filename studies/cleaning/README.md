# Fault-specific cleaning vs dropping every flagged episode

Real public LeRobot datasets, naturally occurring problems. A behaviour-cloning policy (state → 8-step action chunk) is trained four ways and scored by action-prediction error on held-out clean episodes (an offline proxy; real task success needs the robot). Mean of 3 random splits; lower is better.

| Dataset | Drop everything flagged | robolint cleaning plan | Random removal (same count as plan) |
|---|---|---|---|
| kmg0620/so101_bomb_pick_place_v1 (22 stacked recordings) | −37% | **−39%** (removes 22, trims 2) | +2% |
| jchandru08/so100_press_red_button (metadata mismatches) | +84% | **0%** (repairs, removes none) | 0% |
| mliu48/telearms-yam-cube-v4 (failed grasps, idle edges) | +145% | **+2%** (labels, trims 37) | 0% |

Dropping every flagged episode can badly hurt a policy; the plan (`robolint check DATASET --clean-plan plan.json`) removes only broken recordings, repairs bookkeeping, trims idle edges and keeps failed attempts as labelled recovery data. It matched or beat dropping everything on all three. Two other datasets had no unflagged episodes to test on and were skipped. Earlier per-condition numbers: `results_real.json`.
