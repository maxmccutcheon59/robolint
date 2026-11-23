# Changelog

## Unreleased

### Fixed
- **Quick mode (`--max-episodes`) no longer skips data when the metadata is wrong.** It chose
  which files to download from `meta/episodes`, so a dataset whose metadata lists fewer episodes
  than its data holds (the kind of problem robolint exists to catch) was graded on only the
  listed files. EPITECH-LILLE/eval_training scored 60.3 (D) with 32,757 of its 41,686 frames
  instead of 39.9 (F). When the dataset is no bigger than `--max-episodes`, or its metadata
  doesn't say how big it is, every data file is now downloaded.

### Added
- **Grade badge.** `robolint check DATASET --badge robolint.svg` writes an SVG badge with the
  grade and score to show on a dataset card or README.
- **Client-facing audit report.** `robolint check DATASET --audit-report audit.html --client NAME`
  writes a one-page summary for a team lead: grade and verdict, share of frames usable after
  cleanup, an action plan (episodes to remove grouped by problem, idle time to trim, episodes to
  review, dataset-level fixes), recording fixes, the worst episodes, and a copyable drop list. It is
  a single self-contained HTML file that prints cleanly to PDF. Also available as
  `robolint.audit_report.write_audit_report`.

### Changed
- **Episodes with errors cap the grade.** The score is unchanged, but the letter grade can be no
  better than B when any episode has errors, C over 5%, D over 15%, F over 30%, so an A always
  means nothing has to be dropped before training. Reports say when a cap lowered the grade
  (`grade_cap` in the JSON, a line in the text and HTML reports), and the HTML and audit reports
  color the grade by letter instead of by score. Caps are set with `ScoreWeights.grade_caps`
  (`()` turns them off). On the published audit: lerobot/svla_so100_stacking A to C (8.9% of
  episodes with errors), lerobot/svla_so100_sorting A to B; the other 16 grades are unchanged.
- **Scoring is consistent and its weights can be changed.** Each finding now keeps (1 − weight) of
  the score instead of subtracting fixed points. Episode findings: error 1.0, warning 0.15, info
  0.02, so one warning still gives 85, but warnings stack toward 0 without reaching it (before, 7
  warnings scored 0, the same as an error). Dataset-level findings now scale the dataset score:
  error 0.15, warning 0.05. Before, a dataset-level error cost a flat 10 points, less than a
  single warning costs its episode. `NO_ACTIONS` and `NO_EPISODES` score the dataset 0. Weights
  are set with `--weight KEY=W` (CLI) or `Config.score_weights` (`ScoreWeights`), and the JSON
  report lists them under `score_weights`. On the 18-dataset audit no grade changes; recomputed
  from the published findings (`benchmarks/audit_scores.py`): EPITECH-LILLE/eval_training 36.6 to
  40.9 (F), kmg0620/so101_bomb_pick_place_v1 55.3 to 52.4 (F), jchandru08/so100_press_red_button
  55.5 to 54.9 (F), SO-101-robot-learning/so101_pick_place 78.5 to 71.1 (C),
  imstevenpmwork/omx_sentry_1 82.0 to 80.8 (B). The other 13 are unchanged. Episode scores can now
  have decimals (72.25).
- **Large datasets no longer need to fit in memory.** The loader reads each data file once, one
  column at a time, to validate it and record which rows belong to which episode; episode columns
  are then decoded when a check first reads them, with at most `--cache-mb` (default 512 MB) of
  decoded data kept. Dataset-wide statistics, the grasp and lag checks, and the global index check
  were rewritten to stream over episodes. Quantiles stay exact (two passes: a sample locates a
  narrow band around each answer, then only the values in the band are kept), and correlations
  are merged chunk by chunk. On a 10M-frame, 14-dimensional dataset peak memory drops from
  7.3 GB to 1.1 GB (0.72 GB with `--cache-mb 128`) with identical findings; see the README's
  "Large datasets" section.
- `Episode.frames` is now a read-only mapping whose arrays are decoded on first use. Code that
  modified those arrays in place must copy them first.

- **Video: frozen-feed check no longer flags cameras that cannot see the motion.** Before,
  `VIDEO_FROZEN` fired whenever any action moved and the image did not, so a wrist camera on a
  resting arm was flagged while the other arm worked. It now uses the per-camera fit described
  under `VIDEO_FREEZE` and stays quiet when only joints that camera does not see moved. It falls
  back to the old behaviour when no fit is possible.
- **Video: stream properties are summarized over every file.** `VIDEO_FPS_MISMATCH` and
  `VIDEO_SHAPE_MISMATCH` now say how many of the camera's files are affected ("(3 of 50 files)"),
  and are reported at the end of the video checks. When no shape is declared, files whose
  resolution differs from the rest are reported. A file's frame rate is measured from the spacing
  of its frame timestamps, so dropped frames no longer make a camera look slow. Only episodes in
  files at the wrong rate are left out of the sync check, not the whole camera.
- **Video: a file with no video stream** is reported as `VIDEO_UNREADABLE` instead of crashing
  the run.

### Added
- **Split action and state columns.** When a dataset has no `action` or `observation.state` column,
  every numeric column under `action.` or `observation.state.` (for example `action.left_arm` and
  `action.right_arm`) is joined side by side, left before right, then by name; joined dimensions
  are named after their part (`left_arm.gripper.pos`). Before, such datasets got `NO_ACTIONS` and
  no other action or state check ran. A column called `actions` is read as the action. The report
  and the JSON (`dataset.columns`) say which columns were joined. Datasets with an `action` column
  are read exactly as before: 69/69 synthetic datasets give identical findings.
- **`--action-key`, `--state-key` (CLI) and `Config.action_keys`, `Config.state_keys`:** choose the
  columns, in the order to join them. An unknown column stops the run with an error.
- **`--gripper`, `--state-gripper` (CLI) and `Config.gripper_dims`, `Config.state_gripper_dims`:**
  name grippers whose dimension names don't contain "gripper", by name or index (for example
  LIBERO-style data with no names). The grasp, gripper and joint-limit checks then run on them, and
  the lag and video checks leave them out of the joints they use. The state gripper defaults to the
  same name, or the same index when action and state have the same width. An unknown name or
  index stops the run with an error.
- **`VIDEO_DROPPED_FRAMES` (error), `VIDEO_DUPLICATE_FRAMES` and `VIDEO_TIMESTAMPS_UNORDERED`
  (warnings), with `--videos`.** Every packet timestamp of every video file is read without
  decoding. A gap of more than 1.5 frame periods inside an episode's segment counts as missing
  frames: loaders that fetch frames by timestamp fail or return the wrong frame there. Gaps at
  episode boundaries are left to `VIDEO_LENGTH_MISMATCH`.
- **`VIDEO_FREEZE` (warning), with `--videos`.** About four frames per second are decoded (only
  key frames when there is one at least every quarter second, as in LeRobot's encoding). For each
  camera, image change between samples is fitted to the measured movement of each joint with
  non-negative weights, and the flicker of a still scene is measured. A stretch of at least 1 s is
  flagged when the image changes less than that flicker (or, for a camera whose still scenes can
  be perfectly static, when the joints it sees move clearly through most of the stretch), and the
  camera is live elsewhere in the episode. Cameras whose change is poorly explained by the motion
  (R² < 0.3), and datasets with no joints whose state follows the command, are skipped. New config
  fields: `freeze_sample_hz`, `freeze_min_seconds`, `freeze_min_r2`, `freeze_min_episodes`.
- `benchmarks/video.py`: a synthetic bimanual recording with two cameras, sensor noise, pauses and
  three injected cases, timed before and after a change.
- `--cache-mb` (CLI) and `cache_mb=` (`lint()`, `load_dataset()`): memory budget for decoded frames.
- `benchmarks/`: `scale.py` (peak memory and time at any size, optionally against an older
  checkout and with a given `--cache-mb`), `corpus.py` (synthetic datasets covering every fault) and `compare.py` (diff two
  source trees' findings and scores on those datasets).
- **`STATE_LAG_OUTLIER` (warning) and `STATE_LAG` (info).** For joints whose measured position
  follows the command, robolint cross-correlates smoothed action and state velocities to estimate
  how many frames the state trails the action in each episode. `STATE_LAG` reports the dataset's
  typical lag. An episode is flagged only when its lag is far from the typical one (at least
  0.15 s, 3 frames and 5 robust standard deviations), the match at its own lag is strong, and that
  lag fits clearly better than the typical one. Grippers, on/off channels, static and short
  episodes, and datasets with fewer than 5 estimable episodes are skipped.
- **`VIDEO_ACTION_OFFSET` (warning) and `VIDEO_LAG` (info), with `--videos`.** robolint decodes up
  to 30 s of each camera per episode and finds the shift at which frame-to-frame image change is
  best explained by the commanded joint speeds (a non-negative fit, so more motion can only mean
  more image change). `VIDEO_LAG` reports each camera's typical delay; episodes whose shift is far
  from it, by the same rules as above (0.25 s tolerance), are flagged. Feeds that are already black,
  blank, frozen, the wrong length or at the wrong frame rate are skipped, and the check is skipped
  without PyAV.

## 0.1.2 (2026-11-19)

### Added
- **`FAILED_GRASP` (error) and `REGRASP` (info).** For each gripper, robolint compares where the
  follower settles when the gripper is commanded closed. If the dataset's grasps usually stop on an
  object, a grasp that closes all the way on nothing is a missed grasp. If an episode's last grasp
  missed, it gets `FAILED_GRASP` (likely a failed demonstration). If it missed and then grasped,
  it gets `REGRASP` (a recovery).
- `studies/filtering_bc/`: a simulated study of whether dropping robolint errors trains a better
  behavior-cloning policy.

### Fixed (before first release of the grasp check)
- **Rest poses read as grasps.** Holds from the first frame, and a final close on nothing after
  the episode already held an object or closed and reopened, are the gripper at rest, not grasps.
  Without this, ALOHA sim datasets got `REGRASP` on every episode.
- **Open side read as the grasp side.** Operators open the gripper by different amounts. A side
  now counts as grasping only if the follower is actually held short of the command. Without
  this, SO-100/SO-101 episodes that opened fully were flagged `FAILED_GRASP`.
- **Bimanual pairing.** Action and state grippers are paired by name when both list the same
  gripper names, instead of by position.

### Limits
- Recall on real failed demonstrations is not yet measured. No public dataset we audited labels
  its failed demos. On the 18-dataset audit, 5 episodes are flagged: 4 are confirmed recoveries
  (`REGRASP`) and 1 is ambiguous.
- The check only runs on dimensions whose names contain "gripper", present in both `action` and
  `observation.state`. Unnamed grippers (for example LIBERO) are skipped without a message. It
  also assumes the commanded and measured gripper positions use the same units.

## 0.1.1 (2026-11-19)

Correctness release. Upgrade if you use robolint as a quality gate: 0.1.0 could report a
broken dataset as clean.

### Fixed
- **Silent pass on data the loader could not use.** In 0.1.0, one null row in a vector column
  (for example `action`), or rows or files where that column had a different width, made the
  whole column disappear without a message. Every check on it was skipped, so a dataset with real
  faults could score 100 (A). These rows now load as NaN, so the affected episodes are flagged
  (`NONFINITE_VALUES`, error), and the loader also reports the problem itself.
- **Crash on a data file without `episode_index`.** A LeRobot v2 file missing that column could
  exhaust memory. Its rows are now skipped and reported.
- **File-handle leak in video checks (`--videos`).** Every video file stayed open for the whole
  run, so datasets with many files hit the operating system's open-file limit and produced false
  `VIDEO_UNREADABLE` errors. At most 8 files are now open at a time. Stream properties (fps,
  resolution) are now checked on every file, not just the first per camera.

### Added
- New error codes, which count toward the score:
  - `DATA_COLUMN_INVALID`: null values, rows of a different vector width, a column missing from
    some files, or a non-numeric `action` / `observation.state` / index / timestamp column.
  - `DATA_FILE_SKIPPED`: a data file, or some of its rows, could not be checked (no
    `episode_index`, or the file resolves outside the dataset folder via a symlink).

### Changed
- Vector columns load as float32 instead of float64, which uses less memory. Scores on the
  published audit datasets are unchanged.
- README, AUDIT.md and docs/hf-article.md now describe the score as a heuristic (100 minus penalties per issue)
  rather than "the share of your data you can train on". The audit write-ups note that video streams were
  not checked in the audit.
