"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .api import lint
from .audit_report import write_audit_report
from .badge import write_badge
from .checks import Config
from .dataset import DatasetError
from .findings import ScoreWeights, Severity
from .report import audit_markdown, write_exclude, write_html, write_json, write_text


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--videos", action="store_true", help="also check camera streams (downloads videos; needs PyAV)")
    p.add_argument("--revision", help="Hub revision (branch, tag, or commit)")
    p.add_argument("--cache-dir", help="where to cache Hub downloads")
    p.add_argument("--max-episodes", type=int, help="only check the first N episodes (quick look)")
    p.add_argument(
        "--cache-mb",
        type=float,
        help="decoded frame data kept in memory, in MB (default 512); larger datasets are streamed from disk",
    )
    p.add_argument(
        "--weight",
        action="append",
        metavar="KEY=W",
        help="change how much of the score a finding takes (0 to 1): KEY is error, warning or info "
        "(episode findings), dataset-error, dataset-warning or dataset-info, or a finding code "
        "such as IDLE_PADDING; repeatable",
    )
    k = p.add_argument_group("columns (found automatically unless given)")
    k.add_argument(
        "--action-key",
        action="append",
        metavar="COLUMN",
        help="data column that makes up the action; repeat to join several side by side, in order "
        "(e.g. --action-key action.left --action-key action.right)",
    )
    k.add_argument("--state-key", action="append", metavar="COLUMN", help="the same for observation.state")
    k.add_argument(
        "--gripper",
        action="append",
        metavar="DIM",
        help="action dimension that is a gripper, by name or 0-based index; repeat for each gripper",
    )
    k.add_argument(
        "--state-gripper",
        action="append",
        metavar="DIM",
        help="state dimension that is a gripper (default: the same name, or the same index if action "
        "and state have the same width)",
    )


def _dim(v: str) -> str | int:
    return int(v) if v.lstrip("-").isdigit() else v


def _weights(items: list[str] | None) -> ScoreWeights:
    overrides: dict[str, float] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        try:
            overrides[key] = float(value)
        except ValueError:
            raise ValueError(f"--weight expects KEY=W with a number W, got {item!r}") from None
        if not sep:
            raise ValueError(f"--weight expects KEY=W, got {item!r}")
    return ScoreWeights().with_overrides(overrides)


def _config(args: argparse.Namespace) -> Config:
    return Config(
        score_weights=_weights(args.weight),
        action_keys=tuple(args.action_key or ()),
        state_keys=tuple(args.state_key or ()),
        gripper_dims=tuple(_dim(v) for v in args.gripper or ()),
        state_gripper_dims=tuple(_dim(v) for v in args.state_gripper or ()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robolint",
        description="Quality checks for robot learning datasets in LeRobot format.",
    )
    parser.add_argument("--version", action="version", version=f"robolint {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="check one dataset (local folder or Hub repo id like lerobot/pusht)")
    c.add_argument("source")
    _common(c)
    c.add_argument("--json", type=Path, help="write the full report as JSON")
    c.add_argument("--html", type=Path, help="write a shareable HTML report")
    c.add_argument(
        "--audit-report",
        type=Path,
        help="write a client-facing audit report (HTML; prints cleanly to PDF)",
    )
    c.add_argument("--client", help="client name shown on the audit report")
    c.add_argument("--badge", type=Path, help="write an SVG badge with the grade, for a dataset card or README")
    c.add_argument("--exclude-out", type=Path, help="write flagged episode indices, one per line")
    c.add_argument(
        "--exclude-level",
        choices=["error", "warning"],
        default="error",
        help="minimum severity for --exclude-out (default: error)",
    )
    c.add_argument("--fail-under", type=float, help="exit 1 if the score is below this (for CI)")
    c.add_argument("--no-color", action="store_true")
    c.add_argument(
        "--clean-plan",
        type=Path,
        help="write a fault-specific cleaning plan (remove / repair / trim / label / review) as JSON",
    )
    c.add_argument(
        "--model",
        action="store_true",
        help="also score episodes with the learned quality model (needs scikit-learn)",
    )

    a = sub.add_parser("audit", help="check many datasets and write a summary table")
    a.add_argument("sources", nargs="*", help="local folders or Hub repo ids")
    a.add_argument("--from-file", type=Path, help="file with one dataset per line")
    _common(a)
    a.add_argument("--out", type=Path, default=Path("robolint-audit"), help="output folder (default: robolint-audit)")
    return parser


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        cfg = _config(args)
    except ValueError as e:
        print(f"robolint: {e}", file=sys.stderr)
        return 2
    try:
        result = lint(
            args.source,
            videos=args.videos,
            revision=args.revision,
            cache_dir=args.cache_dir,
            max_episodes=args.max_episodes,
            cache_mb=args.cache_mb,
            config=cfg,
        )
    except DatasetError as e:
        print(f"robolint: {e}", file=sys.stderr)
        return 2
    if args.model:
        from .quality import score_dataset

        try:
            q = score_dataset(result.dataset)
        except RuntimeError as e:
            print(f"robolint: {e}", file=sys.stderr)
            return 2
        result.extra["quality_model"] = [vars(x) for x in q]
    write_text(result, color=False if args.no_color else None)
    if args.model:
        flagged = [x for x in result.extra["quality_model"] if x["flagged"]]
        rule_bad = set(result.bad_episodes(Severity.WARNING))
        new = [x for x in flagged if x["episode"] not in rule_bad]
        print(
            f"  Quality model: {len(flagged)} of {len(result.extra['quality_model'])} episodes flagged"
            f" ({len(new)} not already flagged by the checks)"
        )
        for x in sorted(new, key=lambda x: -x["p_bad"])[:20]:
            print(f"    episode {x['episode']}: likely {x['likely_fault']} (p={x['p_bad']:.2f})")
    if args.clean_plan:
        from .cleaning import plan, write_plan

        qm = None
        if args.model:
            from .quality import EpisodeQuality

            qm = [EpisodeQuality(**x) for x in result.extra["quality_model"]]
        s = write_plan(plan(result, quality=qm), args.clean_plan)
        print(
            f"  Cleaning plan: {args.clean_plan} (remove {s['remove']}, repair {s['repair']}, "
            f"trim {s['trim']}, label {s['label']}, review {s['review']})"
        )
    if args.json:
        write_json(result, args.json)
        print(f"  JSON report: {args.json}")
    if args.html:
        write_html(result, args.html)
        print(f"  HTML report: {args.html}")
    if args.audit_report:
        imp = write_audit_report(result, args.audit_report, client=args.client)
        print(
            f"  Audit report: {args.audit_report} "
            f"({len(imp.drop_episodes)} episodes to remove, {imp.trim_frames:,} idle frames to trim)"
        )
    if args.badge:
        write_badge(result.scores.grade, result.scores.dataset_score, args.badge)
        print(f"  Badge: {args.badge}")
    if args.exclude_out:
        level = Severity.ERROR if args.exclude_level == "error" else Severity.WARNING
        n = write_exclude(result, args.exclude_out, level)
        print(f"  {n} episode indices written to {args.exclude_out}")
    if args.fail_under is not None and result.scores.dataset_score < args.fail_under:
        return 1
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    sources = list(args.sources)
    if args.from_file:
        for line in args.from_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                sources.append(line)
    if not sources:
        print("robolint audit: no datasets given", file=sys.stderr)
        return 2
    try:
        cfg = _config(args)
    except ValueError as e:
        print(f"robolint audit: {e}", file=sys.stderr)
        return 2
    out: Path = args.out
    (out / "reports").mkdir(parents=True, exist_ok=True)
    summary = []
    for i, src in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] {src} ...", flush=True)
        safe = src.replace("/", "__").replace("\\", "__")
        try:
            r = lint(
                src,
                videos=args.videos,
                revision=args.revision,
                cache_dir=args.cache_dir,
                max_episodes=args.max_episodes,
                cache_mb=args.cache_mb,
                config=cfg,
            )
        except Exception as e:  # noqa: BLE001 - keep going across many datasets
            msg = str(e).replace("|", "/").replace("\n", " ")[:200]
            print(f"    failed: {msg}")
            summary.append({"source": src, "error": msg})
            continue
        d = r.to_dict()
        write_json(r, out / "reports" / f"{safe}.json")
        write_html(r, out / "reports" / f"{safe}.html")
        print(
            f"    score {r.scores.dataset_score:.1f} ({r.scores.grade}), "
            f"{r.scores.episodes_with_errors}/{len(r.dataset.episodes)} episodes with errors"
        )
        d.pop("findings", None)
        d.pop("episode_scores", None)
        summary.append(d)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(audit_markdown(summary), encoding="utf-8")
    print(f"\nAudit written to {out}/summary.md (per-dataset reports in {out}/reports/)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _cmd_check(args)
    return _cmd_audit(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
