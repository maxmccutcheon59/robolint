"""Client-facing audit report: the deliverable of a robolint Data Audit.

Where `report.write_html` lists every finding for an engineer, this report answers what a team
lead asks: is the dataset safe to train on, how much of it is bad, and what to do about it.
It is a single self-contained HTML file that prints cleanly to PDF.
"""

from __future__ import annotations

import datetime as _dt
import html
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .api import Result
from .findings import Severity
from .report import HELP

_VERDICT = {
    "A": "Train-ready",
    "B": "Good, needs minor cleanup",
    "C": "Needs cleanup before training",
    "D": "Significant quality problems",
    "F": "Not train-ready",
}


@dataclass
class Impact:
    """How much of the dataset is affected, in episodes, frames and recording time."""

    episodes: int
    total_frames: int
    drop_episodes: list[int]
    drop_frames: int
    trim: dict[int, int]  # episode -> idle frames to trim, for episodes that are kept
    trim_frames: int
    review_episodes: list[int]
    usable_frames: int
    fps: float | None

    def seconds(self, frames: int) -> float | None:
        return frames / self.fps if self.fps else None


def impact(r: Result) -> Impact:
    lengths = {ep.index: ep.length for ep in r.dataset.episodes}
    total = sum(lengths.values()) or int(r.dataset.total_frames or 0)
    drop = r.bad_episodes(Severity.ERROR)
    dropped = set(drop)
    drop_frames = sum(lengths.get(e, 0) for e in drop)
    trim: dict[int, int] = {}
    for f in r.findings:
        if f.code != "IDLE_PADDING" or f.severity < Severity.WARNING or f.episode is None or f.episode in dropped:
            continue
        n = int(f.details.get("lead_frames", 0) or 0) + int(f.details.get("tail_frames", 0) or 0)
        if n > 0:
            trim[f.episode] = max(trim.get(f.episode, 0), n)
    review = sorted(
        {
            f.episode
            for f in r.findings
            if f.episode is not None
            and f.episode not in dropped
            and f.severity == Severity.WARNING
            and f.code != "IDLE_PADDING"
        }
    )
    trim_frames = sum(trim.values())
    fps = float(r.dataset.fps) if r.dataset.fps else None
    return Impact(
        episodes=len(lengths),
        total_frames=total,
        drop_episodes=drop,
        drop_frames=drop_frames,
        trim=trim,
        trim_frames=trim_frames,
        review_episodes=review,
        usable_frames=max(0, total - drop_frames - trim_frames),
        fps=fps,
    )


def _ranges(eps: list[int]) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1–3, 7, 9–10'."""
    out: list[str] = []
    xs = sorted(set(eps))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}–{xs[j]}")
        i = j + 1
    return ", ".join(out)


def _dur(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "0%"


def _eps(n: int) -> str:
    return f"{n} episode" if n == 1 else f"{n} episodes"


def verdict(r: Result, imp: Impact) -> str:
    if not imp.drop_episodes and not imp.trim and not imp.review_episodes:
        return "No problems found in the episodes. This dataset is ready to train on."
    parts = []
    if imp.drop_episodes:
        parts.append(
            f"{_eps(len(imp.drop_episodes))} of {imp.episodes} ({_pct(len(imp.drop_episodes), imp.episodes)}) "
            "have errors and should be removed before training"
        )
    if imp.trim:
        parts.append(f"{_eps(len(imp.trim))} carry idle time that should be trimmed")
    if imp.review_episodes:
        parts.append(f"{_eps(len(imp.review_episodes))} need a human look")
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


_CSS = """
:root{--fg:#1c1917;--muted:#6b6560;--line:#e7e5e4;--card:#fafaf9;--err:#b91c1c;--warn:#b45309;--ok:#15803d;--accent:#1e3a8a}
*{box-sizing:border-box}body{margin:0;color:var(--fg);background:#fff;font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 24px}
header{border-bottom:2px solid var(--accent);padding-bottom:16px;margin-bottom:24px}
.kicker{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{font-size:26px;margin:6px 0 4px}h2{font-size:18px;margin:36px 0 10px;color:var(--accent)}h3{font-size:15px;margin:18px 0 4px}
.muted{color:var(--muted)}.small{font-size:13px}
.verdict{display:flex;gap:20px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.grade{font-size:56px;font-weight:700;line-height:1;min-width:72px;text-align:center}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.big{font-size:24px;font-weight:650}
.err{color:var(--err)}.warn{color:var(--warn)}.ok{color:var(--ok)}
table{width:100%;border-collapse:collapse;margin-top:8px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
ol.plan>li{margin-bottom:14px}code,pre{font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;white-space:pre-wrap;word-break:break-word}
footer{margin-top:40px;padding-top:12px;border-top:1px solid var(--line)}
@media (max-width:640px){.cards{grid-template-columns:repeat(2,1fr)}.verdict{flex-direction:column;align-items:flex-start}}
@media print{main{padding:0}h2{break-after:avoid}table,.verdict,.cards,li{break-inside:avoid}@page{margin:18mm}}
"""


def write_audit_report(
    r: Result,
    path: Path,
    *,
    client: str | None = None,
    prepared_by: str = "Eidran",
    date: _dt.date | None = None,
    max_episodes_listed: int = 40,
) -> Impact:
    """Write the client-facing audit report and return the impact numbers it is built on."""
    e = html.escape
    ds, sc = r.dataset, r.scores
    imp = impact(r)
    d = date or _dt.date.today()
    day = f"{d:%B} {d.day}, {d.year}"  # %-d is not portable to Windows
    grade_cls = "ok" if sc.dataset_score >= 80 else ("warn" if sc.dataset_score >= 60 else "err")
    hours = _dur(imp.seconds(imp.total_frames))

    def frames_card(label: str, frames: int, cls: str, sub: str) -> str:
        t = _dur(imp.seconds(frames))
        return (
            f"<div class='card'><div class='muted small'>{e(label)}</div>"
            f"<div class='big {cls}'>{_pct(frames, imp.total_frames)}</div>"
            f"<div class='small'>{frames:,} frames{(' · ' + t) if t else ''}</div>"
            f"<div class='small muted'>{e(sub)}</div></div>"
        )

    p: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Data Quality Audit · {e(r.source)}</title><style>{_CSS}</style></head><body><main>",
        "<header><div class='kicker'>robolint Data Audit</div>",
        f"<h1>Data Quality Audit{(' for ' + e(client)) if client else ''}</h1>",
        f"<div class='muted'>Dataset <code>{e(r.source)}</code> · {e(day)} · prepared by {e(prepared_by)}</div>",
        "</header>",
        # 1. Verdict
        "<h2>Summary</h2>",
        f"<div class='verdict'><div class='grade {grade_cls}'>{e(sc.grade)}</div><div>",
        f"<div style='font-size:18px;font-weight:600'>{e(_VERDICT.get(sc.grade, ''))}</div>",
        f"<div>{e(verdict(r, imp))}</div>",
        f"<div class='muted small'>Quality score {sc.dataset_score:.1f}/100 across {imp.episodes:,} episodes, "
        f"{imp.total_frames:,} frames{(' (' + hours + ' of recording)') if hours else ''}.</div>",
        "</div></div>",
        "<div class='cards'>",
        frames_card("Usable after cleanup", imp.usable_frames, "ok", "kept for training"),
        frames_card("To remove", imp.drop_frames, "err", _eps(len(imp.drop_episodes))),
        frames_card("Idle to trim", imp.trim_frames, "warn", _eps(len(imp.trim))),
        f"<div class='card'><div class='muted small'>Needs review</div>"
        f"<div class='big warn'>{len(imp.review_episodes)}</div><div class='small'>episodes</div>"
        f"<div class='small muted'>warnings only</div></div>",
        "</div>",
    ]

    # 2. Action plan
    p.append("<h2>Action plan</h2><ol class='plan'>")
    by_code: dict[str, set[int]] = {}
    for f in r.findings:
        if f.episode is not None and f.severity == Severity.ERROR:
            by_code.setdefault(f.code, set()).add(f.episode)
    if imp.drop_episodes:
        p.append(
            f"<li><b>Remove {_eps(len(imp.drop_episodes))}</b> before training. Training on them teaches the "
            "policy the wrong behavior or breaks data loading.<table><tr><th>Problem</th><th>Episodes</th>"
            "<th>Why it matters</th></tr>"
        )
        for code, eps in sorted(by_code.items(), key=lambda x: -len(x[1])):
            p.append(
                f"<tr><td><code>{e(code)}</code></td><td>{e(_ranges(sorted(eps)))}</td>"
                f"<td>{e(HELP.get(code, ''))}</td></tr>"
            )
        p.append("</table></li>")
    if imp.trim:
        p.append(
            f"<li><b>Trim idle time from {_eps(len(imp.trim))}</b> ({imp.trim_frames:,} frames). Idle lead-ins and "
            "tails bias the policy toward hesitating. Episodes: "
            f"{e(_ranges(sorted(imp.trim)))}.</li>"
        )
    if imp.review_episodes:
        codes = sorted(
            {
                f.code
                for f in r.findings
                if f.episode in set(imp.review_episodes) and f.severity == Severity.WARNING and f.code != "IDLE_PADDING"
            }
        )
        p.append(
            f"<li><b>Review {_eps(len(imp.review_episodes))}</b> with warnings ({e(', '.join(codes))}). They may be "
            f"fine for your task; check a few by eye. Episodes: {e(_ranges(imp.review_episodes))}.</li>"
        )
    meta = [f for f in r.dataset_findings if f.severity >= Severity.WARNING]
    if meta:
        p.append("<li><b>Fix dataset-level problems.</b><ul>")
        for f in meta:
            p.append(f"<li>{e(f.message)}</li>")
        p.append("</ul></li>")
    if not (imp.drop_episodes or imp.trim or imp.review_episodes or meta):
        p.append("<li>No action needed.</li>")
    p.append("</ol>")

    # 3. Recording fixes: stop the problems at the source
    fix_codes = [c for c, s, _ in r.code_counts() if s >= Severity.WARNING and c in HELP]
    if fix_codes:
        p.append(
            "<h2>Fix it at the source</h2><p class='muted'>What each problem means for your next recording "
            "session.</p><table><tr><th>Problem</th><th>Affected</th><th>Cause and fix</th></tr>"
        )
        counts = {c: n for c, _, n in r.code_counts()}
        for c in fix_codes:
            dataset_level = all(f.episode is None for f in r.findings if f.code == c)
            p.append(
                f"<tr><td><code>{e(c)}</code></td><td>{'dataset' if dataset_level else _eps(counts[c])}</td>"
                f"<td>{e(HELP[c])}</td></tr>"
            )
        p.append("</table>")

    # 4. Worst episodes
    flagged = sorted(((s, ep) for ep, s in sc.episode_scores.items() if s < 100), key=lambda x: (x[0], x[1]))
    if flagged:
        p.append("<h2>Worst episodes</h2><table><tr><th>Episode</th><th>Score</th><th>Main finding</th></tr>")
        for s, ep in flagged[:max_episodes_listed]:
            fs = sorted(r.episode_findings(ep), key=lambda f: -int(f.severity))
            more = f" <span class='muted'>(+{len(fs) - 1} more)</span>" if len(fs) > 1 else ""
            p.append(f"<tr><td>{ep}</td><td>{s:.0f}</td><td>{e(fs[0].message) if fs else ''}{more}</td></tr>")
        p.append("</table>")
        if len(flagged) > max_episodes_listed:
            p.append(
                f"<p class='muted small'>{len(flagged) - max_episodes_listed} more in the full technical report.</p>"
            )

    # 5. Drop list
    if imp.drop_episodes:
        p.append(
            "<h2>Episodes to remove</h2><p class='muted small'>Copy this list into your training config's "
            "episode filter.</p>"
        )
        p.append(f"<pre>{e(', '.join(str(x) for x in imp.drop_episodes))}</pre>")

    # 6. Method and limits
    checked = "action, state, timing, episode and metadata streams"
    checked += ", plus camera streams" if r.videos_checked else " (camera streams were not checked)"
    p += [
        "<h2>Method and limits</h2>",
        f"<p class='small'>Every episode was checked by robolint {e(__version__)} for {e(checked)}. "
        "Errors make an episode unusable; warnings lower its score. The score summarizes how many and how "
        "serious the problems are; it is not a measured prediction of training results.</p>",
    ]
    if ds.partial:
        p.append("<p class='small warn'>Only part of the dataset was checked (quick mode).</p>")
    p.append(
        f"<footer class='muted small'>Prepared by {e(prepared_by)} with robolint, the open-source quality "
        "checker for robot learning data.</footer></main></body></html>"
    )
    path.write_text("".join(p), encoding="utf-8")
    return imp
