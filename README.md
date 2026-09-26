# WIUT Hackathon 2026 — CV track: SoWeNeedAName

Traffic event detection (Part A) and causal accident anticipation (Part B) for a
fixed 4K CCTV camera. Detector + tracker + rules on trajectories and a
hand-drawn scene layout.

## 1. Install and run

Python ≥ 3.10 (tested on 3.11/3.12), one NVIDIA GPU (T4-class or newer).

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

`predictions_samples.json` is exactly the output of this commit on the four sample
videos (reproduce: `python run_submission.py --videos samples --out predictions_samples.json
--team SoWeNeedAName`; two runs give identical events and risk, see §4).

**Weights** (`weights/yolo11s.pt`, `weights/yolo11n.pt`, ~25 MB total) are committed
to the repository, so no download is needed. If they are missing,
`bash weights/download.sh` fetches them once (internet required) and checks
their sha256. The evaluation run itself is fully offline (`solution.py` sets
`YOLO_OFFLINE=1`; `lap` for ByteTrack is in requirements, so nothing is
installed at run time).

**Environment notes**

* `torch==2.8.0` from PyPI is the CUDA 12.8 build: supports T4 (sm_75) through
  RTX 50xx (sm_120) and NVIDIA drivers ≥ 525. If CUDA is unusable, the code
  logs a warning and falls back to CPU instead of crashing.
* `opencv-python` (required by ultralytics) needs libGL on a headless server:
  `apt-get install -y libgl1 libglib2.0-0`.
* Docker alternative (installs these libraries itself):
  ```bash
  docker build -t team .
  docker run --rm --gpus all -v /data/test:/data/test -v "$PWD/out":/out team \
      python run_submission.py --videos /data/test --out /out/predictions.json
  ```
* Windows laptop with an RTX GPU (development): `powershell -ExecutionPolicy Bypass -File setup.ps1`
  (installs the same torch 2.8.0 from the cu128 index). Linux/macOS: `bash setup.sh`.

## 2. Approach

```
                     ┌──────────── one decode pass per video (Part A) ────────────┐
 video ─► zone ─►    │ every 3rd frame, top 18% cropped                           │
       alignment     │   YOLO11s (COCO, fp16, 1280) ─► ByteTrack (long buffer)    │─► tracks
     (src/align.py)  │   traffic-light state (lit section: top/middle/bottom)     │─► light timeline
                     │   obstacle / fire scanners (MOG2, HSV)                     │─► candidates
                     └────────────────────────────────────────────────────────────┘
 tracks ─► stitch fragments ─► per-class rules on trajectories + zones ─► per-class segment
           post-processing (merge gaps, drop blips) ─► [[start, end, label], ...]

 Part B (separate causal pass, frames streamed by the harness):
 frame ─► YOLO11n + ByteTrack every 2nd frame (background thread) ─► ground points
          ─► per pair: closest approach (t*, d_min) + required deceleration
          v_c²/2·gap, discounted if already braking ─► median over 0.6 s ─► risk
```

* **Zone alignment (adaptive to camera shifts).** The camera is "fixed", but
  between recordings it moves: relative to C3896 (where the zones were drawn),
  C3902 is shifted and rotated by 1° with a 1.5 % zoom and a slight change of
  perspective; inside one clip it sways by ~±10 px. Zones are drawn once and
  mapped to each video by a **homography**: CLAHE → SIFT on 1280-px frames →
  RANSAC on the static background, with a fallback to similarity when there
  are too few correspondences and a plausibility check (≤ 25 % shift, ≤ 10°,
  ±35 % zoom, no degenerate perspective) that rejects mismatches instead of
  drawing zones 2000 px off. A **bank of reference frames** (day C3896,
  sunset C3902, dusk C3905, each with its stored transform to the main one)
  lets a dark video match a dark reference. Three frames per video (one for
  clips < 60 s) are combined through the median of projected control points;
  a large spread is logged as "camera moved during the clip".
  `tools/align_stress.py` perturbs real frames (shift 500×250 px, rotation 8°,
  zoom 0.8–1.25, tilt up to 8 %, 30 % of the frame occluded, blur, noise,
  darker/brighter) with each video's own reference removed from the bank:
  60/60 aligned, max error 1 px on the zone corners. Similarity alone missed
  tilted views by 49–262 px. If the traffic light still reads confidently in
  < 30 % of frames, it is not trusted: no red_light/stop_line for that video.
* **Kinematics.** Zone membership uses the bottom-centre of the box (ground
  contact point). Speeds are measured over a 0.6 s window in "body lengths per
  second" (normalised by the box diagonal) to compensate perspective.
* **Traffic light.** Classified by *which section is lit*, not by colour
  counts, so sunset tint and tail-lights behind the signal do not flip it;
  sticky hold with 2-sample confirmation; a frame is skipped when a vehicle
  closer to the camera covers the signal, and green → red without yellow is
  accepted only if it holds 3 s (an occluded green otherwise reads as a dim
  unlit red lens). The lit section must be 35 levels
  brighter than the next one; the absolute floor is only 40, because in
  direct sun a lit lamp reads V≈45–115 (255 at dusk). All four samples show
  the same cycle: red ≈37 s → green ≈37 s → yellow 3 s.
* **Tracker threshold.** The detector runs at conf 0.1 and ByteTrack does the
  filtering: new tracks need 0.6, weak detections only extend existing tracks
  through occlusions. At conf 0.35 a courier moped split into three tracks
  with a 2 s hole; median track length went from 7.5 s to 9.7 s.
  Two-wheelers remain the weak spot: the COCO-trained detector often misses
  motorcycles and mopeds on this view (far away, or next to cars and buses), so
  their boxes flicker or vanish for a few frames and a violation by a two-wheeler
  can be missed. Fine-tuning on this camera is the planned fix.
* **Rules** (src/rules.py): congestion (a queue that keeps standing through
  ≥ 15 s of green, or a jam on the junction longer than one signal phase; a
  normal red-light queue is not congestion), stopped_vehicle (excluding waiting at
  the signal and cars packed in a queue or jam; any lone stop on the junction
  counts), jaywalking (crossings and pedestrian islands excluded), red_light,
  stop_line (a stop of ≥ 1.5 s on red past the stop line: the band before the far
  zebra or the zebra itself, in the lanes of the queue; not for vehicles that then
  enter the junction on red), illegal_turn (the
  forbidden route: from the main road deep into the junction, a U-shaped turn
  there and back onto the lower end of the near crossing) and experimental
  rules for the other classes. Driving onto a pedestrian island is detected
  as `curb_mount`: there is no official class for it, so it is shown in the
  visualisations only and never written to predictions. People whose box sits inside
  a vehicle box (riders, bus passengers) are not pedestrians.
* **Robustness.** Weights, CUDA and the zone reference are warmed up when
  `solution.py` is imported (the harness does that before starting the per-video
  clock); each rule runs only if its class is submitted and inside its own
  `try`; zones are rescaled for non-4K input; short clips align on one frame.
* **Submitted classes** are `solution.CLASSES`. A predicted class that is absent
  from the test set counts as F1 = 0 in the macro average, so experimental rules
  (`EXPERIMENTAL_CLASSES`) are computed but not submitted until they pass our
  dev-set ablation (`tools/dev_loop.py --ablate`).
* **Part B** is causal: it never opens the video and does not use Part A output.
  It only reads a shared wall clock (`src/budget.py`) to stay inside the time budget.
  A pair (at least one vehicle) raises an alarm only if it is on a collision
  course (bottom-centre ground points, closest approach < 0.3 box diagonals
  within 3.5 s) **and** stopping is already hard (required deceleration
  v_c²/(2·gap) above 1–3 diagonals/s²). The first version used closest approach
  alone and scored ≥ 0.5 on 57–71 % of all sample frames: a car rolling up to a
  stopped queue is "about to collide" at constant velocity. Now 0.1 % of frames,
  2 alarms in 18 min of samples (`tools/risk_replay.py score`); in the harness run
  that wrote `predictions_samples.json` no frame reaches 0.5 (peak 0.498). A capped
  sub-threshold term (≤ 0.3) for any collision course ranks the 1–5 s before
  contact higher for AP without creating alarms. Synthetic scenarios
  (`tests/test_risk.py`: T-bone, rear-end, pedestrian; queue braking, parallel
  lanes, oncoming pass) pin the behaviour. `tools/risk_scenarios.py` measures it
  on 15 crash and 9 safe scenarios with box jitter and missed detections, the way
  evaluate.py counts alarms: 40 % of crashes alarmed, ~0.1 s before contact, no
  false alarms; rear-end crashes at city speed are missed. Looser thresholds catch
  up to 93 % but fire 15–43 times on the 18 min of samples (a dense queue looks
  like a crash in image space), so the conservative setting is kept: with rare
  accidents, false alarms cost more alarm-F1 than the extra hits bring.
* **Speed.** Decoding 4K H.264 on the CPU is the main cost (≈0.7× video duration
  per pass, and there are two passes: ours in Part A and the harness's in Part B).
  Both parts overlap decoding with GPU inference: Part A reads frames in a
  separate thread, Part B runs detection of frame k in a worker thread while
  the harness decodes the next frames and collects it at the next inference
  step, so scores lag by one step (2 frames). Order is fixed, results do not
  depend on thread timing.

**Learned vs rule-based.** Learned: only the object detectors (YOLO11s/n,
COCO-pretrained by Ultralytics, not fine-tuned). Everything else — tracking
association, zone alignment, light state, all event rules, risk score — is
rule-based.

## 3. Data, models and licences

| Item | Used for | Licence |
|---|---|---|
| YOLO11s, YOLO11n weights — Ultralytics | detection | AGPL-3.0 |
| COCO 2017 (Ultralytics' pre-training of the weights above) | — (not used by us directly) | CC BY 4.0 |
| ByteTrack (implementation in ultralytics) | tracking | MIT (original) / AGPL-3.0 (ultralytics) |
| Our own labels of the sample videos (`labels/`, in progress) | dev set, threshold tuning | ours |

No other datasets. No hosted models or paid APIs at any stage of inference.
The repository is licensed under **AGPL-3.0**, as required by Ultralytics YOLO.

## 4. Determinism

* Seeds fixed in `solution.py` (python `random`, numpy, torch);
  `cudnn.benchmark = False`, `cudnn.deterministic = True`; OpenCV RANSAC seed
  fixed (`cv2.setRNGSeed(0)`). ByteTrack and all rules are deterministic.
  `python tools/presubmit.py --determinism samples/<video>` runs the harness
  twice and compares the outputs.
* Threads (frame reader in Part A, detector worker in Part B) only overlap
  work; frames are consumed in a fixed order, so outputs are the same as a
  sequential run (checked: identical events on C3905).
* **Only time-dependent behaviour:** emergency time guards. If the tracker
  pass is projected above 1.5× the video duration, the frame stride is
  increased; Part B increases its stride if it would miss the 3× deadline.
  They exist so that a slow machine produces a slightly coarser result
  instead of an empty video. Every activation is printed in the log; on
  T4-class hardware with our measured runtime they do not fire.
* fp16 GPU inference may differ from fp32/CPU at floating-point level.

## 5. Team

Team **SoWeNeedAName**:

| Member | Role | What they did | Links |
|---|---|---|---|
| Shaxzod Kalandarov | Captain · ML pipeline and Part B | Detection and tracking: YOLO11s / YOLO11n with ByteTrack, frame sampling, track stitching; Part B accident-risk estimator: closest approach, required deceleration, calibration on normal traffic; Runtime and the submission: reader and detector threads, time guards, determinism, run_submission.py compatibility; Repository, weights, Dockerfile and README; team coordination | [GitHub](https://github.com/Roicalus) · [LinkedIn](https://www.linkedin.com/in/shakhzod-kalandarov-71a9293b9/) |
| Aleksandr Polyakov | Scene understanding and event rules | Scene zones of the junction and their alignment to every recording (CLAHE + SIFT homography, reference bank); Traffic-light reader: lit-section classification, occlusion handling, phase smoothing; Event rules for the six submitted classes and segment post-processing; Unit tests for alignment, rules and risk; ablations on the sample videos | [GitHub](https://github.com/justm1x) · [LinkedIn](https://www.linkedin.com/in/aleksandr-polyakov-07142b43a) |
| Yana Semianiuta | Website, live demo and visual analysis | Team website: design, results pages, dashboard, interactive timelines and charts; Live demo: Hugging Face Space (Gradio, ZeroGPU with CPU fallback) and its connection to the site; Annotated video rendering and the site export of samples, heatmaps and trajectories; EDA of the sample videos and the technical report | [GitHub](https://github.com/yanasemianiuta) · [LinkedIn](https://www.linkedin.com/in/yana-semianiuta) |

## 6. Repository layout

```
solution.py              interface: CLASSES, detect_events, RiskEstimator (thin wrapper over src/)
run_submission.py        organizers' harness (unchanged)
evaluate.py              organizers' metric (unchanged)
requirements.txt         pinned dependencies        Dockerfile  alternative environment
zones.json               hand-drawn scene zones (reference-frame pixels)
zones_ref*.jpg, .json    reference frames for alignment: main (day), sunset, dusk
predictions_samples.json our output on the sample videos (this exact commit)
weights/                 yolo11s.pt, yolo11n.pt, download.sh (sha256-checked)
src/
  pipeline.py   Part A: extract (one decode pass) -> infer (rules); dev cache
  align.py      per-video zone alignment (homography, reference bank)
  track.py      YOLO + ByteTrack, reader thread, stride guard
  stitch_tracks.py  track fragment stitching      light_state.py  traffic-light state
  rules.py      event rules                       obstacle_fire.py  obstacle / fire scanners
  postprocess.py  per-class segment post-processing
  risk.py       Part B: RiskScorer (numpy) + RiskEstimator (detector, budget)
  budget.py     shared time budget
  render.py     annotated video (zones, boxes, violators, light, timeline, risk)
  analyze.py    one clip end to end for the live demo
demo/           app.py (Gradio live demo), requirements.txt (demo only)
tools/          define_zones, make_zone_ref, check_alignment, align_stress, light_check,
                label_tool, build_ground_truth, dev_loop, risk_replay, visualize_debug,
                build_space, check_env, presubmit
tests/          synthetic tests, no GPU needed: python -m pytest tests -q
docs/           camera_own.md (scene notes), CHANGES.md
```

## 7. Live demo

`demo/app.py` — upload an .mp4 (≤ 2.5 min), get the events, a timeline with the risk
curve, `events.json` and an annotated video. Same code as the submission
(`src/analyze.py`); on a CPU host the clip is first downscaled to 1920 px / 15 fps
(`DEMO_WIDTH`, `DEMO_FPS`, `DEMO_IMGSZ` override this), while the rules keep working in
reference-frame pixels.

```bash
pip install -r demo/requirements.txt
python demo/app.py                              # http://127.0.0.1:7860
python tools/build_space.py --out ../space      # folder for a Hugging Face Space
```

## 8. Development loop

```bash
python tools/label_tool.py --video samples/C3896.MP4          # own labels -> labels/
python tools/build_ground_truth.py --labels labels --out my_labels.json
python tools/dev_loop.py --videos samples --gt my_labels.json --ablate   # cached, seconds per run
python tools/visualize_debug.py --video samples/C3902.MP4 --predictions predictions_samples.json
python tools/align_stress.py                                  # alignment vs synthetic camera shifts
python tools/risk_replay.py dump  --videos samples            # Part B detections -> cache/risk (once)
python tools/risk_replay.py score --videos samples --set MEDIAN_K=7   # alarms/min in seconds
python tools/presubmit.py                                     # before every tag
```

## 9. Attribution

Ultralytics YOLO and its ByteTrack implementation (AGPL-3.0). All other code in
this repository was written by the team (with AI coding assistance, allowed by
the rules for code, not for inference).
