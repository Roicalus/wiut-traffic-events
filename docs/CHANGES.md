# What changed while building the base (relative to the files we were given)

Critical (decides whether a video is scored at all):
- track.py: `persist=False` on the first frame of every video. The model is
  cached across videos, and with `persist=True` the ByteTrack state always
  leaked from one clip into the next.
- rules.py: accident/near_miss iterated over ALL object pairs in a clip
  (O(N²)): 128 s for this function alone on a 340 s / 1500-track synthetic.
  Candidate pairs are now taken per frame (numpy); the whole compute_events
  takes 19 s on the same synthetic.
- The traffic light is read during the tracker pass (LightScanner), not in a
  separate pass: one fewer full 4K decode per video.
- Time guards: the tracker raises the stride if the projected pass time is
  > 1.5x the duration; RiskEstimator knows the video deadline (src/budget.py)
  and thins out inference. Otherwise a budget overrun zeroes Part A as well.
- requirements.txt: ultralytics, torch, lap (without lap, ultralytics tries
  to pip-install it during the run, which fails offline).

Quality:
- rules.annotate: zones by the bottom of the box (ground point), speed over a
  0.6 s window. The old per-frame speed tore stationary cars apart on box
  jitter (test_stopped_vehicle_single_car fails on the old rules.py).
- accident: one of the pair must be moving within 1.5 s before contact;
  otherwise neighbours in a queue with overlapping boxes count as a crash.
- near_miss from a single braking object is disabled (NEAR_MISS_FROM_BRAKING).
- solid_line_crossing: hysteresis at the line + projection onto the segment.
- Part B rewritten: closest point of approach instead of centre convergence,
  pedestrian-pedestrian pairs excluded, braking capped at 0.35, median over 3.
- src/postprocess.py: per-class merging / minimum duration.
- CLASSES = 5 base classes; the rest are in EXPERIMENTAL_CLASSES.

Tooling: extract/infer pipeline + cache, tools/dev_loop.py (evaluation and
per-class ablation in seconds), tests/ (10 synthetic tests, no GPU).

## After the first real run (debug videos C3896/C3902)

- The camera shifts between recordings (~80x45 px in 4K): on C3902 light_roi
  sat next to the traffic light -> false red_light/stop_line. New:
  src/align.py aligns the zones to each video via a reference frame
  (zones_ref.jpg, ORB + RANSAC). tools/make_zone_ref.py, tools/check_alignment.py.
- Traffic light: classifier by the position of the lit section
  (top/middle/bottom) instead of colour counting, so it does not depend on
  sunset or brake lights behind the frame. tools/light_check.py is a grid of
  crops for visual checking.
- wrong_way: reference direction per zone, and only in one-way zones
  (WRONG_WAY_ZONES); area excluded.
- stopped_vehicle: stopping at the stop line/crossing and moving off right
  after green = waiting for the signal, not an event.
- track.py: the time guard ignores GPU warm-up (first frames).
- define_zones.py --load: add/replace individual zones.
- dev_loop.py --list: events of a class with timestamps.

## After a full run on all 4 samples (samples/, 29.97 fps, 4K H.264)

- Zone alignment: ORB -> CLAHE + SIFT, median over three frames of the clip.
  ORB found no matches at sunset/dusk (C3902, C3905: 16-21 inliers), and
  there the camera is shifted by 140x77 px and rotated by 1°.
- Time: was 2.08-2.23x the duration (limit 3x). The main cost is 4K decoding
  on the CPU (0.7x per pass, two passes). Part A reads frames in a separate
  thread; Part B runs detection in the background while the harness decodes
  the next frames. C3905: 2.17x -> 1.81x, events identical.
- Part B: risk >= 0.5 on 57-71% of frames (approaching a queue = "collision
  at constant speed"). Now: ground points, required deceleration, braking
  taken into account, boxes cut off by the frame edge filtered out, median
  over 0.6 s: 0.1% of frames, 2 alarms in 18 min. RiskScorer is separated
  from the detector, tools/risk_replay.py calibrates on the detection cache
  in seconds, tests/test_risk.py has synthetic scenarios.

## After reviewing the debug videos (tools/visualize_debug.py)

- Traffic light: in daytime sun the lit lamp is dim (V 45-115 vs 255 at
  sunset); the LAMP_MIN_V=150 threshold made C3896 36% and C3897 100%
  "unknown" -> red_light/stop_line did not work there at all. Now the
  difference from the neighbouring section decides (LAMP_MIN_V=40): unknown
  0-2%, on all 4 clips one cycle of red ~37 s / green ~37 s / yellow 3 s. A
  red_light appeared on C3896 at 79-80 s (running a red, spotted by eye).
- Part A tracker: conf 0.35 -> 0.1. Weak detections never reached ByteTrack,
  so it could not extend tracks with them: a courier on a moped (C3896,
  20-27 s) broke into 3 tracks with a 2 s gap. Median track length 7.5 -> 9.7 s.
- jaywalking: sidewalk islands sidewalk_island_1/2 added to zones.json
  (inside the crossroad polygon); people walking across them from one zebra
  to the other are no longer violators. What remains is a real "desire path"
  over the asphalt between crossings (jaywalking by definition).
- stopped_vehicle: "part of a jam" now = at least 2 more cars standing close
  by (<= 1.5 diagonals), not "there is a congestion segment somewhere in the
  frame". A lone car in the middle of the square is no longer lost because
  of the queue on the main road.
- tools/light_state.py removed: an outdated, unused copy of
  src/light_state.py.
- visualize_debug.py: timeline of all events and risk under the frame, the
  object pair producing the risk, box highlighting only for submitted
  classes (--all-classes for all), zones without fill.

## Second pass over the debug videos

- Traffic light: false red on C3896 at 189.8-192.5 s. A bus roof covered the
  lit green section, and the unlit red lens in daylight has brightness ~50
  (like a dim lit red). (1) LightScanner skips a frame if the ROI is covered
  by the box of a vehicle closer to the camera (box bottom 60+ px below the
  frame); (2) smooth_states: green -> red without yellow is accepted only if
  it holds for 3 s.
- Verified from the data that light_roi is the main-road queue signal: 433
  of 452 entries from queue_zone onto the far crossing happen on its green.
  The pedestrian light on the pole on the left is in phase with it (a
  different crossing).
- stopped_vehicle: on the intersection beyond the light (crossroad*) the
  "moved off on green" exception does not apply; a lone car in the middle of
  the square is an event (C3897 0-27 s, 210-250 s). Cluster = >= 2
  neighbours close by (<= 1.5 diagonals of the smaller box) for >= 50% of
  the stop time, so a dense jam on the square (C3896 40-110 s) is
  congestion, not 10 events.
- illegal_turn included in the submission. Added cutting across a sidewalk
  island (detect_island_cut): a point at 30% of box height above the bottom
  (wheels; the box bottom is the bumper corner on the asphalt); the event is
  extended to the whole turn (_turn_span). C3905 99.3-101.7 s, C3896
  23.6-25.3 s (moped).

## Third pass over the debug videos

- illegal_turn: instead of "a 50-150 degree turn inside illegal_turn_zone"
  (1 hit, 2 false positives: cars detouring from the square) a route-based
  rule: from the main road (queue_zone/stop_line/crossing_far) across the
  square to the lower end of the near crossing (new zone illegal_turn_exit).
  Real U-turns ran below the old polygon and sharper than 150 degrees. The
  span runs from the heading deviating from the approach heading
  (> 20 degrees, no earlier than 6 s before the exit) to the exit.
  illegal_turn_zone removed.
- Cutting across an island is not counted for motorcycles and bicycles.
- jaywalking: a person whose box is >= 60% inside a vehicle box is a driver
  or passenger (C3902, 145 s: a motorcyclist).
- red_light outranks stop_line: an object that ran a red does not also get
  "stopped past the stop line" (C3902, 97 s). The visualiser shows all
  active violations of an object, coloured by the most important one.
- compute_events() is now compute_events_debug() + merging: a single
  implementation of the rules for both the submission and the visualisation.

## Fourth pass: U-turn, island mounting, whole-system audit

- illegal_turn: added a mandatory U-turn apex (illegal_turn_apex) and an
  exit no later than 3 s after it. Without it, cars driving left along the
  bottom edge of the frame from the right edge fired (the tracker had
  stitched them to a car from the main road): C3896 628, C3897 602, C3902
  x3. Exactly the ones confirmed by eye remain: C3896 45 (49.6 s) and 830 (284 s).
- Mounting a sidewalk island is a separate event, curb_mount. There is no
  official class, and custom ids are forbidden in the submission (the
  harness drops them), so it is diagnostics only: visible in the
  visualisation, not written to predictions.json
  (solution.DIAGNOSTIC_CLASSES).
- congestion per the task definition: a queue at a signal is not a jam, it
  clears on green (clusters of 4+ stationary cars in queue_zone on the
  samples are red 63-90% of the time). A queue is a jam if it stands >= 15 s
  on green; the intersection beyond the light is, if a dense mass stands
  longer than a phase (>= 40 s). The queue and the square are counted separately.
- Audit (independent code review + edge-case clips through the harness):
  * warm-up on importing solution.py (weights, CUDA, first inference,
    reference-frame features) is outside the video budget. 3-second clip:
    12 s against a 9 s budget (empty video) -> 5.5 s;
  * zone alignment on clips < 60 s uses a single frame (seeking in 4K is
    ~2.5 s per frame);
  * rules are computed only for submitted classes, each in its own
    safeguard: an experimental rule crashing on an unfamiliar scene does not
    wipe the video's other classes; the obstacle/fire scanner is created
    only if its classes are needed;
  * concurrent_runs: on equal times the interval start goes first, so a
    "relay hand-off" between cars no longer broke congestion; the duration
    threshold is applied after merging;
  * traffic light: with no confident readings for > 45 s the colour resets
    to unknown;
  * zones are scaled to the video resolution if it is not 4K (and on failed
    alignment);
  * track stitching does not join objects that exist in the same frame;
    the traffic light ROI is clipped to the frame edge; _mark_riders is vectorised;
  * stdout/stderr with errors="replace": Russian-language logs no longer
    crash detect_events on a machine with a non-UTF-8 console;
  * restored weights/download.sh (Dockerfile and setup.sh referenced it,
    the file was missing, docker build failed), with a sha256 check;
  * removed outdated descriptions and dead code; logs/ in .dockerignore.

## Robustness to camera displacement

- Stress test of the old alignment: estimates were accurate, but the
  plausibility filter (8% / 3° / ±5%) rejected 372 of 432 synthetic
  displacements, leaving zones unaligned. Camera tilt (perspective) cannot
  be described by shift+rotation+scale at all: 49-262 px error.
- src/align.py rewritten: homography (fallback to similarity), a bank of
  reference frames (day/sunset/dusk: tools/make_zone_ref.py --extra), median
  of projections of a control-point grid over 3 frames, a warning on camera
  shift mid-clip, plausibility check 25% / 10° / ±35% / degenerate
  perspective.
- tools/align_stress.py: honest mode (the same video's reference frame is
  removed from the bank), 15 kinds of displacement x 4 videos: 60/60,
  error <= 1 px.
- The traffic light is not trusted if confident readings are < 30% (ROI not
  on the light): red_light/stop_line are not emitted for such a video.

## stop_line per the task definition

- Before: stop_line only when stopping in a narrow strip before the far
  zebra, and red_light if a car entered the zebra on red, even if it then
  stopped there. Result on the samples: 0 stop_line.
- Zone past_stop_line (zones.json): from the stop line to the far edge of
  the zebra, only across the width of the queue lanes (on the right of the
  zebra is another flow with another signal). stop_line = stopping >= 1.5 s
  in it on red, before green.
- red_light is not counted if the car stood past the line and entered the
  intersection already on green (C3902, 1:38: a courier waits ahead of the
  queue, pedestrians cross the zebra, leaves together with the queue).
- Samples: C3905 +stop_line 1:17.7–1:54.6 (car with its front on the zebra
  for 37 s), C3902 red_light 1:36.9 -> stop_line 1:38.7–1:57.2. Tests:
  stopping on the zebra, waiting and leaving on green, running a red without stopping.

## Class switch within a track

- ByteTrack in ultralytics does not distinguish classes: a pedestrian's ID
  passed to a car that covered their box (C3897 #829, 5:01.9). Rules took
  the class from the first record, and the car became a "pedestrian on the
  roadway".
- stitch_tracks.split_class_switches: a track that changes family (person /
  vehicle) is split into two parts with different IDs. The samples have 31
  such tracks; 5 false jaywalking events went away (all were cars or a
  truck at the moment of the event), one got shorter.
- Part B: on a family switch the track history starts over (otherwise the
  "person to car" speed is a false risk spike).
