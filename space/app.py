"""robolint on Hugging Face Spaces: check any public LeRobot dataset in the browser."""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import gradio as gr

from robolint import lint
from robolint.dataset import REPO_ID_RE, DatasetError
from robolint.report import HELP, write_html

MAX_EPISODES = 300  # keeps checks fast on free hardware; the CLI has no limit

EXAMPLES = [
    "lerobot/svla_so100_stacking",
    "lerobot/svla_so100_sorting",
    "lerobot/pusht",
    "lerobot/svla_so101_pickplace",
]


def _summary(r) -> str:
    sc, ds = r.scores, r.dataset
    n = len(ds.episodes)
    lines = [
        f"### {html.escape(r.source)}",
        f"**Score {sc.dataset_score:.1f}/100 (grade {sc.grade})** · {n:,} episodes checked · {ds.total_frames:,} frames"
        + (f" · first {n} of {ds.info.get('total_episodes', '?')} episodes" if ds.partial else ""),
        "",
        f"Clean: {sc.clean_episodes} · With warnings: {sc.episodes_with_warnings} · With errors: {sc.episodes_with_errors}",
        "",
    ]
    rows = r.code_counts()
    if rows:
        lines += ["| Severity | Issue | Affected | What it means |", "|---|---|---|---|"]
        for code, sev, k in rows:
            scope = "dataset" if all(f.episode is None for f in r.findings if f.code == code) else f"{k} ep."
            lines.append(f"| {sev.label} | `{code}` | {scope} | {html.escape(HELP.get(code, ''))} |")
    else:
        lines.append("No issues found.")
    bad = r.bad_episodes()
    if bad:
        shown = ", ".join(map(str, bad[:40])) + (" …" if len(bad) > 40 else "")
        lines += ["", f"**Episodes with errors:** {shown}"]
    return "\n".join(lines)


def check(repo_id: str, quick: bool):
    repo_id = (repo_id or "").strip()
    # Only public Hub repo ids: never let a request point the checker at the server's filesystem.
    if not REPO_ID_RE.match(repo_id) or Path(repo_id).exists():
        return "Enter a Hugging Face dataset id like `lerobot/pusht`.", ""
    try:
        r = lint(repo_id, max_episodes=MAX_EPISODES if quick else None)
    except DatasetError as e:
        return f"Could not check `{html.escape(repo_id)}`: {html.escape(str(e))}", ""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.html"
        write_html(r, out)
        page = out.read_text(encoding="utf-8")
    frame = (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" style="width:100%;height:720px;border:0" sandbox></iframe>'
    )
    return _summary(r), frame


with gr.Blocks(title="robolint: robot dataset quality checker") as demo:
    gr.Markdown(
        "# robolint\n"
        "Find dropped frames, dead episodes, teleop glitches, and broken metadata in any public "
        "[LeRobot](https://github.com/huggingface/lerobot) dataset before you train on it.  \n"
        "[GitHub](https://github.com/maxmccutcheon59/robolint) · "
        "[Audit of 18 public datasets](https://github.com/maxmccutcheon59/robolint/blob/main/AUDIT.md) · "
        "camera checks and unlimited datasets: install the CLI from GitHub"
    )
    with gr.Row():
        repo = gr.Textbox(label="Dataset id", placeholder="lerobot/pusht", scale=4)
        quick = gr.Checkbox(label=f"Quick mode (first {MAX_EPISODES} episodes)", value=True, scale=1)
    run = gr.Button("Check dataset", variant="primary")
    gr.Examples(EXAMPLES, inputs=repo)
    summary = gr.Markdown()
    report = gr.HTML()
    run.click(check, inputs=[repo, quick], outputs=[summary, report])
    repo.submit(check, inputs=[repo, quick], outputs=[summary, report])

if __name__ == "__main__":
    demo.launch()
