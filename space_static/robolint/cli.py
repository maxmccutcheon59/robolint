"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .api import lint
from .dataset import DatasetError
from .findings import Severity
from .report import audit_markdown, write_exclude, write_html, write_json, write_text


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--videos", action="store_true", help="also check camera streams (downloads videos; needs PyAV)")
    p.add_argument("--revision", help="Hub revision (branch, tag, or commit)")
    p.add_argument("--cache-dir", help="where to cache Hub downloads")
    p.add_argument("--max-episodes", type=int, help="only check the first N episodes (quick look)")


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
    c.add_argument("--exclude-out", type=Path, help="write flagged episode indices, one per line")
    c.add_argument(
        "--exclude-level",
        choices=["error", "warning"],
        default="error",
        help="minimum severity for --exclude-out (default: error)",
    )
    c.add_argument("--fail-under", type=float, help="exit 1 if the score is below this (for CI)")
    c.add_argument("--no-color", action="store_true")

    a = sub.add_parser("audit", help="check many datasets and write a summary table")
    a.add_argument("sources", nargs="*", help="local folders or Hub repo ids")
    a.add_argument("--from-file", type=Path, help="file with one dataset per line")
    _common(a)
    a.add_argument("--out", type=Path, default=Path("robolint-audit"), help="output folder (default: robolint-audit)")
    return parser


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        result = lint(
            args.source,
            videos=args.videos,
            revision=args.revision,
            cache_dir=args.cache_dir,
            max_episodes=args.max_episodes,
        )
    except DatasetError as e:
        print(f"robolint: {e}", file=sys.stderr)
        return 2
    write_text(result, color=False if args.no_color else None)
    if args.json:
        write_json(result, args.json)
        print(f"  JSON report: {args.json}")
    if args.html:
        write_html(result, args.html)
        print(f"  HTML report: {args.html}")
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
