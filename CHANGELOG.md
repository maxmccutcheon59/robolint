# Changelog

## Unreleased

### Changed
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

### Added
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
