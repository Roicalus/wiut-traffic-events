# camera_own.md — the scene from sample frames (C3896/97/902/905)

The organisers did not provide a `camera.md`; the scene layout below was
made by hand from frames. Positions are approximate; refine coordinates
against the actual footage when annotating (pixels at the source resolution
3840x2160).

## General layout

- Shot from a high point (pole/bracket), a single view, no camera motion.
- The main multi-lane road runs across the far part of the frame, at least
  3 lanes each way, split by a yellow centre line, with a sidewalk and trees
  behind it.
- The main flow of cars in the frame moves **from the far part of the frame
  towards the camera** (top to bottom/diagonally) and runs into a queue in
  front of the pedestrian crossing.
- In front of the camera is a wide paved square/intersection with triangular
  and diamond-shaped curbed islands (pink paving).
- Two crossings: the "far" one (a zebra across the main road, near the upper
  third of the frame) and the "near" one (diagonal, crossing the square
  itself at the bottom of the frame).
- A U-shaped signal gantry stands right in front of the far crossing; the
  stop line roughly coincides with where the queue of cars stops.
- A blue round sign with a right arrow on one of the islands: mandatory
  right turn for some of the lanes.
- City buses periodically pass along the far lanes of the main road (routes
  seen: 60, 14).

## Directions of travel

- All cars seen on the far lanes travel in one consistent direction
  (towards the camera / across the crossing); no driving into the oncoming
  lane was seen, so `wrong_way` is unlikely in these samples, but check on
  the other videos.
- After the crossing, part of the flow turns right (per the sign), part
  apparently continues straight across the square.

## What actually happens in the samples

- **Queues/jams**: in daytime and sunset frames 5-8+ cars regularly stand in
  several rows in front of the crossing, a candidate for `congestion` and/or
  `stopped_vehicle` (look at the stop duration: >=10 s not in a signal
  queue is `stopped_vehicle`; cars queuing at the light go to
  `congestion`, not `stopped_vehicle`).
- **Pedestrians**: mass crossing on the zebra; one frame has a large group
  plus a single pedestrian not strictly within the stripes (possible
  `jaywalking`, verify on the video, not on a single frame).
- **Light vehicles**: in the dusk frame a moped/e-scooter courier rides
  across the crossing; in the sunset frame a cyclist at the curb. They
  should be classified as vehicles for tracking (not pedestrians), even if
  YOLO/COCO confuses them.
- **Traffic light**: the signal gantry is in the frame, but the signal
  colour itself could not be read from the still frames (too small/not
  visible from this angle in the screenshots); the video needs to be viewed
  frame by frame near the gantry to see how reliably red/green can be
  detected.
- **Time of day**: the samples cover day, sunset (hard shadows, backlight)
  and dusk (headlights/side lights on); lighting will strongly affect
  detection, especially at sunset and dusk.

## Next steps

- Verify these observations on the full videos (not on a single frame),
  especially the jaywalking candidate and the traffic light colour.
- If lane/stop line/crossing coordinates are needed for the rules, annotate
  them by hand on one frame of each video (polygons/lines in pixels) and
  store them in `src/scene_layout.py` or a json keyed by video_id (the view
  is the same for all samples, so the annotation is shared).
