"""Terminal, JSON, HTML, and Markdown reports."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import TextIO

from .api import Result
from .findings import Severity

HELP: dict[str, str] = {
    "DEAD_EPISODE": "Commanded actions are constant. Drop the episode; it teaches the policy to do nothing.",
    "STATE_FROZEN": "State never updates while actions change. Check the follower arm connection; drop the episode.",
    "NONFINITE_VALUES": "NaN/inf values break training. Drop or repair the affected frames.",
    "DROPPED_FRAMES": "Timestamp gaps mean frames were lost (USB bandwidth, CPU load). Actions and video desync around gaps.",
    "TIMESTAMP_NONMONOTONIC": "Timestamps go backwards or repeat. Recording clock or merge bug.",
    "TIMESTAMP_JITTER": "Irregular frame timing. Check camera/USB load during recording.",
    "TIMESTAMP_DRIFT": "Recording ran faster/slower than the declared fps.",
    "FRAME_INDEX_GAP": "Frames missing from frame_index.",
    "FRAME_INDEX_DUPLICATE": "Repeated frame_index values; usually a bad merge.",
    "ACTION_SPIKE": "Single-step jumps far larger than normal motion: teleop glitches or bus errors. Smooth or trim.",
    "IDLE_PADDING": "Long idle lead-in/tail. Trim it; idle frames bias policies toward hesitating.",
    "GRIPPER_NEVER_ACTUATES": "Gripper never moves. Fine for pushing tasks; a red flag for pick-and-place.",
    "JOINT_AT_LIMIT": "The operator pushes a joint into its hard stop mid-task; the arm cannot follow the command.",
    "CALIBRATION_OFFSET": "Leader and follower calibrations disagree near a joint limit. Usually harmless; recalibrate to be sure.",
    "EPISODE_TOO_SHORT": "Under a second long: almost always a false start. Drop it.",
    "EPISODE_SHORT": "Much shorter than typical: possibly truncated.",
    "EPISODE_LONG": "Much longer than typical: recording may not have been stopped.",
    "EPISODE_LENGTH_MISMATCH": "Episode metadata disagrees with the data. Indexes into this episode will be wrong.",
    "DUPLICATE_EPISODE": "Nearly identical to another episode: duplicated upload or replay. Inflates the dataset without adding information.",
    "VIDEO_MISSING": "Camera file missing; the episode cannot be loaded with images.",
    "VIDEO_LENGTH_MISMATCH": "Video and action streams have different lengths: they are out of sync.",
    "VIDEO_SEGMENT_OUT_OF_RANGE": "Episode points past the end of its video file.",
    "VIDEO_BLACK": "Camera shows black frames: lens cap, disconnect, or no light.",
    "VIDEO_OVEREXPOSED": "Camera is blown out: fix exposure or lighting.",
    "VIDEO_BLANK": "Camera shows a uniform image.",
    "VIDEO_FROZEN": "Camera image never changes while the robot moves: stale or frozen feed.",
    "VIDEO_UNREADABLE": "Video cannot be decoded.",
    "VIDEO_FPS_MISMATCH": "Camera frame rate differs from the dataset fps.",
    "VIDEO_SHAPE_MISMATCH": "Camera resolution differs from what info.json declares.",
    "EPISODE_COUNT_MISMATCH": "info.json disagrees with the data about how many episodes exist.",
    "FRAME_COUNT_MISMATCH": "info.json disagrees with the data about how many frames exist.",
    "FEATURE_MISSING": "A declared feature is absent from the data.",
    "SHAPE_MISMATCH": "Declared and actual dimensions differ.",
    "TASK_VAGUE": "Task descriptions too vague to condition a language-conditioned (VLA) policy.",
    "TASK_EMPTY": "Empty task descriptions.",
    "TASKS_MISSING": "No task descriptions at all.",
    "TASK_INDEX_UNKNOWN": "Frames reference tasks that do not exist.",
    "EPISODE_TASK_EMPTY": "Episode has no task description, so it cannot train a language-conditioned policy.",
    "GLOBAL_INDEX_DUPLICATES": "Frames share global indices: data from different recordings was merged under the same numbers.",
    "GLOBAL_INDEX_GAPS": "Global frame indices skip values: frames or episodes are missing.",
    "EPISODE_INDEX_GAPS": "Episode numbers skip values: episodes were deleted or never uploaded.",
    "FRAME_INDEX_OFFSET": "frame_index does not start at 0.",
    "TIMESTAMP_OFFSET": "Episode timestamps do not start at 0.",
    "TIMESTAMP_NONFINITE": "Timestamps contain NaN or infinite values.",
    "EMPTY_EPISODE": "Episode contains no frames.",
    "NO_EPISODES": "Dataset contains no episodes.",
    "NO_ACTIONS": "No action column: the dataset cannot train a policy.",
    "NO_TIMESTAMPS": "No timestamp column: timing cannot be verified.",
    "FPS_MISSING": "No valid fps in info.json.",
    "METADATA_INCOMPLETE": "Episode metadata files are missing.",
    "OLD_FORMAT": "Dataset predates LeRobot v2.",
    "PARTIAL_CHECK": "Quick mode: only part of the dataset was checked.",
    "VIDEO_SKIPPED": "Video checks need PyAV installed.",
}

_ANSI = {Severity.ERROR: "\033[31m", Severity.WARNING: "\033[33m", Severity.INFO: "\033[36m"}
_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"


def _eps(n: int) -> str:
    return f"{n} episode" if n == 1 else f"{n} episodes"


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "0%"


def write_text(
    r: Result, out: TextIO | None = None, *, color: bool | None = None, max_episodes_listed: int = 15
) -> None:
    out = out or sys.stdout
    color = out.isatty() if color is None else color

    def c(s: str, code: str) -> str:
        return f"{code}{s}{_RESET}" if color else s

    ds, sc = r.dataset, r.scores
    n_eps = len(ds.episodes)
    w = out.write
    w(c(f"robolint report: {r.source}\n", _BOLD))
    w(
        f"  format {ds.version} · robot {ds.info.get('robot_type') or 'unknown'} · "
        f"{n_eps} episodes · {ds.total_frames:,} frames · {ds.fps or '?'} fps · "
        f"cameras: {', '.join(ds.video_keys) or 'none'}\n"
    )
    w(c(f"\n  Score {sc.dataset_score:.1f}/100  (grade {sc.grade})\n", _BOLD))
    w(
        f"  clean episodes {sc.clean_episodes} ({_pct(sc.clean_episodes, n_eps)}) · "
        f"with warnings {sc.episodes_with_warnings} ({_pct(sc.episodes_with_warnings, n_eps)}) · "
        f"with errors {sc.episodes_with_errors} ({_pct(sc.episodes_with_errors, n_eps)})\n"
    )
    if not r.videos_checked and ds.video_keys:
        w(c("  video not checked (add --videos to check camera streams)\n", _DIM))

    rows = r.code_counts()
    if rows:
        w(c("\n  Issues\n", _BOLD))
        for code, sev, n in rows:
            scope = "dataset" if all(f.episode is None for f in r.findings if f.code == code) else _eps(n)
            w(f"  {c(f'{sev.label:<8}', _ANSI[sev])} {code:<26} {scope}\n")
            if code in HELP:
                w(c(f"           {HELP[code]}\n", _DIM))
    else:
        w(c("\n  No issues found.\n", _BOLD))

    ds_f = r.dataset_findings
    if ds_f:
        w(c("\n  Dataset findings\n", _BOLD))
        for f in ds_f:
            w(f"  {c(f.severity.label, _ANSI[f.severity])}: {f.message}\n")

    worst = sorted(((s, ep) for ep, s in sc.episode_scores.items() if s < 100), key=lambda x: (x[0], x[1]))[
        :max_episodes_listed
    ]
    if worst:
        w(c("\n  Lowest-scoring episodes\n", _BOLD))
        for s, ep in worst:
            fs = sorted(r.episode_findings(ep), key=lambda f: -int(f.severity))
            w(f"  episode {ep:<6} score {s:>5.0f}  {fs[0].message if fs else ''}")
            w(f"  (+{len(fs) - 1} more)\n" if len(fs) > 1 else "\n")
    bad = r.bad_episodes()
    if bad:
        w(f"\n  {len(bad)} episodes have errors. Write them out with --exclude-out to drop them before training.\n")
    w(c(f"\n  checked in {r.seconds:.1f}s\n", _DIM))


def write_json(r: Result, path: Path) -> None:
    path.write_text(json.dumps(r.to_dict(), indent=2, default=str), encoding="utf-8")


def write_exclude(r: Result, path: Path, min_severity: Severity) -> int:
    eps = r.bad_episodes(min_severity)
    path.write_text("".join(f"{e}\n" for e in eps), encoding="utf-8")
    return len(eps)


_HTML_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--err:#b91c1c;--warn:#b45309;--info:#0e7490;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--card:#1c1917;--line:#292524;--err:#f87171;--warn:#fbbf24;--info:#67e8f9;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 8px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.big{font-size:28px;font-weight:650}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.error{color:var(--err)}.warning{color:var(--warn)}.info{color:var(--info)}.ok{color:var(--ok)}code{font-size:13px}
.wrap{overflow-x:auto}
@media (max-width:640px){table,tbody,tr,td{display:block;width:100%}tr:first-child{display:none}
tr{border-bottom:1px solid var(--line);padding:8px 0}td{border:none;padding:2px 12px}}
"""


def write_html(r: Result, path: Path) -> None:
    e = html.escape
    ds, sc = r.dataset, r.scores
    n_eps = len(ds.episodes)
    grade_cls = "ok" if sc.dataset_score >= 80 else ("warning" if sc.dataset_score >= 60 else "error")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>robolint · {e(r.source)}</title><style>{_HTML_CSS}</style></head><body><main>",
        f"<h1>{e(r.source)}</h1>",
        f"<div class='muted'>LeRobot {e(ds.version)} · robot {e(str(ds.info.get('robot_type') or 'unknown'))} · "
        f"{ds.fps or '?'} fps · cameras: {e(', '.join(ds.video_keys) or 'none')}"
        f"{'' if r.videos_checked or not ds.video_keys else ' · video not checked'}</div>",
        "<div class='cards'>",
        f"<div class='card'><div class='muted'>Score</div><div class='big {grade_cls}'>{sc.dataset_score:.1f}</div><div>grade {e(sc.grade)}</div></div>",
        f"<div class='card'><div class='muted'>Episodes</div><div class='big'>{n_eps:,}</div><div>{ds.total_frames:,} frames</div></div>",
        f"<div class='card'><div class='muted'>Clean</div><div class='big ok'>{_pct(sc.clean_episodes, n_eps)}</div><div>{_eps(sc.clean_episodes)}</div></div>",
        f"<div class='card'><div class='muted'>With errors</div><div class='big error'>{_pct(sc.episodes_with_errors, n_eps)}</div><div>{_eps(sc.episodes_with_errors)}</div></div>",
        "</div>",
    ]
    rows = r.code_counts()
    parts.append("<h2>Issues</h2>")
    if rows:
        parts.append(
            "<div class='wrap'><table><tr><th>Severity</th><th>Issue</th><th>Affected</th><th>What it means</th></tr>"
        )
        for code, sev, n in rows:
            dataset_level = all(f.episode is None for f in r.findings if f.code == code)
            parts.append(
                f"<tr><td class='{sev.label}'>{sev.label}</td><td><code>{e(code)}</code></td>"
                f"<td>{'dataset' if dataset_level else _eps(n)}</td><td>{e(HELP.get(code, ''))}</td></tr>"
            )
        parts.append("</table></div>")
    else:
        parts.append("<p class='ok'>No issues found.</p>")

    ds_f = r.dataset_findings
    if ds_f:
        parts.append("<h2>Dataset findings</h2><div class='wrap'><table><tr><th>Severity</th><th>Finding</th></tr>")
        for f in ds_f:
            parts.append(f"<tr><td class='{f.severity.label}'>{f.severity.label}</td><td>{e(f.message)}</td></tr>")
        parts.append("</table></div>")

    flagged = [(ep, s) for ep, s in sorted(sc.episode_scores.items(), key=lambda x: (x[1], x[0])) if s < 100]
    if flagged:
        parts.append(
            "<h2>Flagged episodes</h2><div class='wrap'><table><tr><th>Episode</th><th>Score</th><th>Findings</th></tr>"
        )
        for ep, s in flagged[:500]:
            items = "<br>".join(
                f"<span class='{f.severity.label}'>{f.severity.label}</span> {e(f.message)}"
                for f in sorted(r.episode_findings(ep), key=lambda f: -int(f.severity))
            )
            parts.append(f"<tr><td>{ep}</td><td>{s:.0f}</td><td>{items}</td></tr>")
        parts.append("</table></div>")
        if len(flagged) > 500:
            parts.append(f"<p class='muted'>{len(flagged) - 500} more flagged episodes in the JSON report.</p>")
    parts.append(f"<p class='muted'>Generated by robolint in {r.seconds:.1f}s.</p></main></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def audit_markdown(results: list[dict]) -> str:
    """Summary table across many datasets (dicts from Result.to_dict(), or error entries)."""
    lines = [
        "| Dataset | Format | Episodes | Score | Grade | Clean | With errors | Top issues |",
        "|---|---|---:|---:|:---:|---:|---:|---|",
    ]
    ok = [r for r in results if "error" not in r]
    for r in sorted(ok, key=lambda r: r["score"]):
        n = r["dataset"]["episodes"]
        top = ", ".join(f"`{i['code']}`" for i in r["issues"][:3]) or "none"
        lines.append(
            f"| {r['source']} | {r['dataset']['codebase_version']} | {n} | {r['score']:.1f} | {r['grade']} | "
            f"{_pct(r['episodes']['clean'], n)} | {_pct(r['episodes']['with_errors'], n)} | {top} |"
        )
    for r in results:
        if "error" in r:
            lines.append(f"| {r['source']} | — | — | — | — | — | — | failed to load: {r['error']} |")
    if ok:
        tot_eps = sum(r["dataset"]["episodes"] for r in ok)
        tot_err = sum(r["episodes"]["with_errors"] for r in ok)
        tot_warn = sum(r["episodes"]["with_warnings"] for r in ok)
        lines += [
            "",
            f"**{len(ok)} datasets, {tot_eps:,} episodes.** "
            f"{_pct(tot_err, tot_eps)} of episodes have errors and {_pct(tot_warn, tot_eps)} more have warnings.",
        ]
        counts: dict[str, int] = {}
        for r in ok:
            for i in r["issues"]:
                counts[i["code"]] = counts.get(i["code"], 0) + 1
        if counts:
            lines += ["", "Most common issues (datasets affected):", ""]
            for code, n in sorted(counts.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"- `{code}`: {n}/{len(ok)} datasets. {HELP.get(code, '')}")
    return "\n".join(lines) + "\n"
