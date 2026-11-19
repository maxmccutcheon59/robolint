# Changelog

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
- README and AUDIT.md now describe the score as a heuristic (100 minus penalties per issue)
  rather than "the share of your data you can train on". AUDIT.md notes that video streams were
  not checked in the audit.
