"""Live demo: upload an .mp4 from the camera, get the detected events back.

    python demo/app.py            # http://127.0.0.1:7860

The same pipeline as the submission (solution.py / src/), run through
src/analyze.py: the clip is downscaled, Part A (events) and Part B (causal
accident risk) run, and an annotated video is rendered. On Hugging Face
ZeroGPU the detector runs inside a @spaces.GPU call (the GPU exists only
there); if it is not available (quota, error) the same step runs on the CPU.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

try:                        # Hugging Face ZeroGPU: GPU только внутри @spaces.GPU
    import spaces           # импортируется до torch — пакет его патчит
except ImportError:
    spaces = None
if spaces is not None:
    os.environ.setdefault("WIUT_DEVICE", "cpu")   # при импорте CUDA ещё нет

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

import solution  # noqa: E402,F401  (seeds, warm-up, CLASSES)
from src import analyze, track  # noqa: E402
from src.render import EVENT_COLORS  # noqa: E402

MAX_FILE = os.environ.get("DEMO_MAX_FILE", "3gb")
GPU_MAX_SEC = 120
INTRO = f"""
# Traffic events at the intersection — live demo

Upload an **.mp4 from this camera** (up to **{analyze.MAX_DURATION_SEC / 60:.1f} min**, up to **{MAX_FILE}**; any
resolution — 4K is downscaled to {analyze.DEMO_WIDTH} px at {analyze.DEMO_FPS:g} fps for the demo).
The page returns every detected event as `[start, end, class]`, a timeline, the accident-risk curve
(Part B, causal) and the video annotated with our own tooling.

The detector runs on a shared GPU when one is available and falls back to a **CPU** otherwise
(a 2-minute 4K clip then takes several minutes). Progress is shown below. It is the same code
as the submission (the submission uses the full 4K frame). Videos from a different camera are
processed too, but the scene zones will not fit them.
"""


def _gpu_seconds(path, info):
    """Сколько GPU-времени заказать у ZeroGPU: по длине ролика, не больше лимита."""
    return int(min(GPU_MAX_SEC, 30 + 0.6 * info["duration"]))


if spaces is not None:
    @spaces.GPU(duration=_gpu_seconds)
    def compute_on_gpu(path, info):
        track.set_device(0)
        return analyze.compute(path)


def _hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def timeline(result):
    events = result["events"] + result["diagnostics"]
    classes = sorted({e[2] for e in events})
    fig = go.Figure()
    for s, e, label in events:
        fig.add_trace(go.Bar(x=[e - s], base=[s], y=[label], orientation="h",
                             marker_color=_hex(EVENT_COLORS.get(label, (160, 160, 160))),
                             hovertemplate=f"{label}<br>%{{base:.1f}}–{e:.1f} s<extra></extra>",
                             showlegend=False))
    risk = result["risk"][::3]
    fig.add_trace(go.Scatter(x=[t for t, _ in risk], y=[s for _, s in risk], name="risk (Part B)",
                             yaxis="y2", line=dict(color="#e0a000", width=1.5),
                             hovertemplate="t=%{x:.1f} s<br>risk %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0.5, yref="y2", line_dash="dot", line_color="#e0a000", opacity=0.5)
    fig.update_layout(
        height=160 + 28 * max(len(classes), 1), margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(title="seconds", range=[0, result["info"]["duration"]]),
        yaxis=dict(categoryorder="array", categoryarray=classes[::-1] or ["—"]),
        yaxis2=dict(overlaying="y", side="right", range=[0, 1], title="risk", showgrid=False),
        barmode="overlay", title="Events and accident risk")
    return fig


def summary(result):
    a = result.get("align") or {}
    if a.get("status") == "ok":
        align = (f"aligned to the reference view ({a.get('model')}, reference `{a.get('ref')}`): "
                 f"shift {a.get('dx')}×{a.get('dy')} px, rotation {a.get('rot_deg')}°, scale {a.get('scale')}")
    else:
        align = f"**not aligned** ({a.get('status')}) — zones used as drawn; is this the same camera?"
    light = result.get("light")
    light = ", ".join(f"{k} {v:.0%}" for k, v in sorted(light.items())) if light else "not used"
    t = result["timings_sec"]
    device = "CPU" if result.get("device") == "cpu" else "GPU"
    counts = {}
    for _, _, label in result["events"]:
        counts[label] = counts.get(label, 0) + 1
    found = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no events"
    return (f"**{len(result['events'])} events** — {found}\n\n"
            f"- Scene zones: {align}\n- Traffic light states: {light}\n"
            f"- Alarms (risk ≥ 0.5): {sum(1 for _, s in result['risk'] if s >= 0.5)} frames\n"
            f"- Time: {t['total']:.0f} s, detector on {device} (prepare {t['transcode']:.0f}, "
            f"Part A {t['part_a']:.0f}, Part B {t['part_b']:.0f}, render {t['render']:.0f})\n"
            + (f"- {result['note']}\n" if result.get("note") else "")
            + (f"- Diagnostics (not submitted): curb mounts {len(result['diagnostics'])}\n"
               if result["diagnostics"] else ""))


def run(video, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload an .mp4 first.")
    work = tempfile.mkdtemp(prefix="demo_")
    say = lambda f, msg: progress(f, desc=msg)  # noqa: E731
    t0 = time.perf_counter()
    try:
        path, info = analyze.prepare(video, work, say)
        t_prep = time.perf_counter() - t0
        computed, note = None, None
        if spaces is not None:
            say(0.3, "Part A + Part B on a GPU")
            try:
                computed = compute_on_gpu(str(path), info)
            except Exception as exc:  # noqa: BLE001 — квота ZeroGPU и т.п.: считаем на CPU
                note = f"GPU was not available ({type(exc).__name__}: {str(exc)[:120]}); computed on the CPU"
                print(f"[demo] {note}")
                track.set_device("cpu")
        if computed is None:
            computed = analyze.compute(path, say)
        result = analyze.finish(video, path, info, work, computed, say)
    except analyze.VideoError as exc:
        raise gr.Error(str(exc)) from exc
    result["timings_sec"].update(transcode=round(t_prep, 1), total=round(time.perf_counter() - t0, 1))
    result["note"] = note
    rows = [[round(s, 2), round(e, 2), round(e - s, 2), label] for s, e, label in result["events"]]
    return (result["annotated"], timeline(result), rows or [[None, None, None, "no events"]],
            result["json"], summary(result))


with gr.Blocks(title="Traffic events — live demo") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Video(label="Upload .mp4", sources=["upload"])
            go_btn = gr.Button("Analyze", variant="primary")
            info = gr.Markdown()
        with gr.Column(scale=2):
            out_video = gr.Video(label="Annotated video", interactive=False)
    plot = gr.Plot(label="Timeline")
    with gr.Row():
        table = gr.Dataframe(headers=["start, s", "end, s", "length, s", "class"], label="Events",
                             interactive=False, wrap=True)
        out_json = gr.File(label="events.json")
    go_btn.click(run, inputs=inp, outputs=[out_video, plot, table, out_json, info])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=os.environ.get("DEMO_HOST", "127.0.0.1"), max_file_size=MAX_FILE)
