"""
rules.py — event rules on top of stitched tracks (track.py -> stitch_tracks.py),
scene zones (zones.json, aligned with the video in align.py) and the traffic-light state
(light_state.py). Called from pipeline.infer(); the entry points are
compute_events() (submission) / compute_events_debug() (the same, with object ids).
The submitted classes are solution.CLASSES; the other rules are experimental.

CLI main() is for debugging only, on saved track JSON:
    python src/rules.py --tracks src/tracks/C3896.stitched.json --zones zones.json \
        --light src/light/C3896.json --out predictions/C3896.json

Thresholds were tuned on the samples (see docs/CHANGES.md). Speed is normalised by the
object's box diagonal (a rough perspective compensation: for distant cars the same
pixels/s mean a much higher real speed than for near ones), so the
thresholds are "body lengths per second", not px/s. Calibrate against your own
annotations:
    python evaluate.py --pred predictions.json --gt ground_truth.json --per-video
"""
import argparse
import bisect
import json
from pathlib import Path

import cv2
import numpy as np

VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck
PERSON_CLASS = 0


# ---------------------------------------------------------------- helpers
def load_tracks(path):
    data = json.loads(Path(path).read_text())
    return data["meta"], data["tracks"]


def load_zones(path):
    raw = json.loads(Path(path).read_text())
    return {name: np.array(pts, dtype=np.float32) for name, pts in raw.items()}


def in_zone(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0


def group_by_object(records):
    groups = {}
    for r in records:
        oid = r.get("stitched_id", r["track_id"])
        groups.setdefault(oid, []).append(r)
    for recs in groups.values():
        recs.sort(key=lambda r: r["t_sec"])
    return groups


# The point at which an object "stands" in a zone. Zones are drawn on the asphalt, while
# the box centre of a car/person hangs above the ground and, in the camera perspective, is
# shifted "further" from the real position — for a pedestrian on the sidewalk the torso
# centre may already fall into roadway. The bottom-centre of the box (wheels/feet) is the
# correct projection onto the road plane. "center" is the old behaviour.
ZONE_ANCHOR = "bottom"
# Speed is computed from the displacement over a window, not between adjacent samples:
# with stride=3 (0.12 s) a 2-3 px box jitter on a distant car already gives
# ~0.3 "body lengths/s" — exactly the stop threshold — so a standing car "moves",
# chopping stopped_vehicle/congestion into pieces.
SPEED_WINDOW_SEC = 0.6


def _anchor(r):
    cx = (r["x1"] + r["x2"]) / 2.0
    if ZONE_ANCHOR == "bottom":
        return (cx, r["y2"])
    return (cx, (r["y1"] + r["y2"]) / 2.0)


def annotate(recs, zones, speed_window=SPEED_WINDOW_SEC):
    """Object trajectory: for each sample
      t      — time;
      c      — box centre (for pairwise distances between objects);
      g      — ground anchor point (ZONE_ANCHOR) — zones and lines are tested against it;
      speed, vx, vy — speed in "body lengths per second" (normalisation
               by the box diagonal = rough perspective compensation),
               from the centre displacement over the speed_window;
      diag   — box diagonal in px (for pairwise normalisation);
      zones  — set of zones that contain g;
      cls    — COCO class.
    """
    samples = []
    ts, cs = [], []
    j = 0
    for r in recs:
        c = ((r["x1"] + r["x2"]) / 2.0, (r["y1"] + r["y2"]) / 2.0)
        g = _anchor(r)
        diag = max(((r["x2"] - r["x1"]) ** 2 + (r["y2"] - r["y1"]) ** 2) ** 0.5, 1.0)
        t = r["t_sec"]
        ts.append(t)
        cs.append(c)
        # the latest sample that is older than t by at least speed_window
        while j + 1 < len(ts) - 1 and t - ts[j + 1] >= speed_window:
            j += 1
        speed = vx = vy = 0.0
        if len(ts) > 1:
            k = j if t - ts[j] >= speed_window else 0
            dt = t - ts[k]
            if dt > 0:
                vx = (c[0] - cs[k][0]) / diag / dt
                vy = (c[1] - cs[k][1]) / diag / dt
                speed = (vx ** 2 + vy ** 2) ** 0.5
        zones_here = {name for name, poly in zones.items()
                      if len(poly) >= 3 and in_zone(g, poly)}
        samples.append({"t": t, "c": c, "g": g, "speed": speed, "vx": vx, "vy": vy,
                        "diag": diag, "zones": zones_here, "cls": r["cls"]})
    return samples


def sample_runs(samples, predicate, max_gap):
    """Intervals (start, end) where predicate(sample) holds on
    consecutive samples; breaks the run if the gap between samples is >
    max_gap (the object dropped out of tracking — no guarantee the state
    persisted all that time)."""
    runs, start, prev_t = [], None, None
    for s in samples:
        ok = predicate(s)
        if prev_t is not None and s["t"] - prev_t > max_gap:
            if start is not None:
                runs.append((start, prev_t))
            start = s["t"] if ok else None
        elif ok and start is None:
            start = s["t"]
        elif not ok and start is not None:
            runs.append((start, prev_t))
            start = None
        prev_t = s["t"]
    if start is not None:
        runs.append((start, prev_t))
    return runs


def merge_intervals(intervals, gap=0.0):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def concurrent_runs(intervals, thresh):
    """Time intervals where >= thresh intervals are active at once
    (sweep line over +1/-1 start/end events)."""
    if not intervals:
        return []
    # at equal t, +1 goes first: a "relay handover" (one car leaves, another
    # stops in the same sample) and zero-length intervals do not drop the counter
    events = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals],
                    key=lambda x: (x[0], -x[1]))
    count, start, runs = 0, None, []
    for t, delta in events:
        count += delta
        if count >= thresh and start is None:
            start = t
        elif count < thresh and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, events[-1][0]))
    return runs


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def first_entry_time(samples, zone_name):
    was_in = False
    for s in samples:
        now_in = zone_name in s["zones"]
        if now_in and not was_in:
            return s["t"]
        was_in = now_in
    return None


def _angle_between(vx1, vy1, vx2, vy2):
    """Angle between two vectors in degrees [0, 180]. None if either
    vector is zero (the object is standing — direction undefined)."""
    n1 = (vx1 ** 2 + vy1 ** 2) ** 0.5
    n2 = (vx2 ** 2 + vy2 ** 2) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = max(-1.0, min(1.0, (vx1 * vx2 + vy1 * vy2) / (n1 * n2)))
    return np.degrees(np.arccos(cos_a))


def _line_side(point, line):
    """Sign of the cross product — which side of line (2 points) point
    lies on. line is [[x1,y1],[x2,y2]] (as define_zones.py draws it
    for solid_line*: a LINE, not a polygon, so pointPolygonTest does not
    apply — a line has no "inside", only two sides)."""
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _closest_approach(samples_a, samples_b, max_dt=0.6):
    """Aligns two trajectories in time (different objects are sampled
    on the same grid of video frames, but may drop out of tracking
    at different moments — so we look for the nearest sample in time rather
    than require an exact index match).

    Returns a list of (t, dist_norm, closing) sorted by t:
      dist_norm — distance between centroids / mean diagonal of the pair
                  ("body lengths", the same normalisation as speed);
      closing   — closing speed (dist_norm decreasing -> closing > 0),
                  None for the pair's very first point (no previous one)."""
    ts_b = [s["t"] for s in samples_b]
    aligned = []
    for sa in samples_a:
        i = bisect.bisect_left(ts_b, sa["t"])
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(samples_b):
                sb = samples_b[j]
                dt = abs(sb["t"] - sa["t"])
                if dt <= max_dt and (best is None or dt < best[0]):
                    best = (dt, sb)
        if best is None:
            continue
        _, sb = best
        t = (sa["t"] + sb["t"]) / 2.0
        avg_diag = max((sa["diag"] + sb["diag"]) / 2.0, 1.0)
        dist = ((sa["c"][0] - sb["c"][0]) ** 2 + (sa["c"][1] - sb["c"][1]) ** 2) ** 0.5
        aligned.append((t, dist / avg_diag))

    aligned.sort(key=lambda x: x[0])
    out = []
    prev = None
    for t, dist_norm in aligned:
        closing = None
        if prev is not None:
            pt, pd = prev
            dt = t - pt
            if dt > 1e-6:
                closing = (pd - dist_norm) / dt
        out.append((t, dist_norm, closing))
        prev = (t, dist_norm)
    return out


class LightState:
    def __init__(self, samples):
        """samples: list of [t_sec, state], as returned by
        light_state.run_light_state() (or read from its JSON)."""
        samples = samples or []
        self.ts = [t for t, _ in samples]
        self.states = [s for _, s in samples]

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text()) if path else {"samples": []}
        return cls(data["samples"])

    def at(self, t):
        if not self.ts:
            return "unknown"
        i = bisect.bisect_right(self.ts, t) - 1
        i = max(0, min(i, len(self.states) - 1))
        return self.states[i]

    def next_after(self, t, state):
        for tt, s in zip(self.ts, self.states):
            if tt > t and s == state:
                return tt
        return None


# ---------------------------------------------------------------- rules
GREEN_STAND_SEC = 15.0     # queue stands on green longer than this — it did not clear within the phase
JUNCTION_JAM_SEC = 40.0    # dense mass on the junction longer than a light phase (~37 s)


def _stationary_clusters(vehicle_objs, zones, min_count, merge_gap, min_duration,
                         speed_thresh, max_gap):
    """Intervals when >= min_count cars stand in zones at the same time."""
    intervals = []
    for samples in vehicle_objs.values():
        intervals += sample_runs(
            samples, lambda x: x["speed"] < speed_thresh and bool(zones & x["zones"]), max_gap)
    active = merge_intervals(concurrent_runs(intervals, min_count), gap=merge_gap)
    return [iv for iv in active if iv[1] - iv[0] >= min_duration]   # after merging


def _green_time(light, s, e):
    """How many seconds of [s, e] the light was green (from the light samples)."""
    if light is None or not light.ts:
        return 0.0
    lo, hi = bisect.bisect_left(light.ts, s), bisect.bisect_right(light.ts, e)
    ts, st = light.ts[lo:hi], light.states[lo:hi]
    return sum(t1 - t0 for t0, t1, x in zip(ts, ts[1:], st) if x == "green")


def detect_congestion(vehicle_objs, signal_zones=("queue_zone",), junction_zones=(), light=None,
                      min_count=4, min_duration=5.0, merge_gap=3.0, speed_thresh=0.3, max_gap=2.0,
                      green_stand_sec=GREEN_STAND_SEC, junction_jam_sec=JUNCTION_JAM_SEC):
    """Congestion (task definition: traffic stands or crawls in all lanes
    of a direction; from "the queue stopped" to "the queue cleared").

    A queue at a signal is not congestion: it clears on every green. On the
    samples, clusters of 4+ stopped cars in queue_zone fall on red
    (green is 10-37% of their time: that is the discharge delay), and the old rule
    "4+ cars standing" turned every red phase into congestion.
      * queue before the light (signal_zones): congestion only if 4+
        cars stood for >= green_stand_sec ON GREEN. Without a light —
        as on the junction, by duration;
      * the junction itself past the light (junction_zones, crossroad*): congestion
        if a dense mass stands longer than a phase (>= junction_jam_sec) —
        C3896, 40-108 s.
    Clusters are counted per zone group separately: the queue and the square are
    different places, and "2 standing there + 2 here" is not congestion."""
    signal_zones, junction_zones = set(signal_zones), set(junction_zones)
    args = (min_count, merge_gap, min_duration, speed_thresh, max_gap)
    jams = []
    for s, e in _stationary_clusters(vehicle_objs, signal_zones, *args):
        if (light is not None and light.ts and _green_time(light, s, e) >= green_stand_sec) or \
                ((light is None or not light.ts) and e - s >= junction_jam_sec):
            jams.append((s, e))
    if junction_zones:
        jams += [(s, e) for s, e in _stationary_clusters(vehicle_objs, junction_zones, *args)
                 if e - s >= junction_jam_sec]
    return [[round(s, 2), round(e, 2), "congestion"] for s, e in merge_intervals(sorted(jams))]


SIGNAL_WAIT_ZONES = ("queue_zone", "stop_line", "crossing_far")
GREEN_RELEASE_SEC = 8.0


def _released_by_green(light, t_stop, t_move, window=GREEN_RELEASE_SEC):
    """The car moved off within window s after the switch to green,
    having stood mostly on red -> this is waiting for the signal."""
    if light is None or not light.ts:
        return False
    t_green = light.next_after(t_stop, "green")
    if t_green is None or not (t_green - 1.0 <= t_move <= t_green + window):
        return False
    return light.at(t_stop + 0.5) in ("red", "yellow")


def _stationary_runs(vehicle_objs, speed_thresh, max_gap):
    """oid -> [(start, end, centre px, diagonal px)] — the object's stops."""
    out = {}
    for oid, samples in vehicle_objs.items():
        runs = []
        for s, e in sample_runs(samples, lambda x: x["speed"] < speed_thresh, max_gap):
            inside = [x for x in samples if s - 1e-6 <= x["t"] <= e + 1e-6]
            c = np.median([x["c"] for x in inside], axis=0)
            runs.append((s, e, c, float(np.median([x["diag"] for x in inside]))))
        out[oid] = runs
    return out


def _time_with_at_least(intervals, k):
    """Total time during which >= k intervals are open at once."""
    edges = sorted([(a, 1) for a, b in intervals if b > a] + [(b, -1) for a, b in intervals if b > a])
    total, depth, prev = 0.0, 0, None
    for t, d in edges:
        if depth >= k and prev is not None:
            total += t - prev
        depth += d
        prev = t
    return total


def detect_stopped_vehicle(vehicle_objs, road_zones=None, junction_zones=(),
                            speed_thresh=0.3, min_duration=10.0, queue_min_duration=75.0,
                            max_gap=2.0, cluster_min_others=2, cluster_radius=1.5,
                            cluster_overlap_frac=0.5, on_road_frac=0.5, light=None,
                            return_ids=False):
    """Stands >= 10 s (or >= queue_min_duration if it stands in queue_zone — that is
    ordinary waiting for the light cycle, not an event, unless it drags on) and is
    NOT part of a jam or a queue.

    Part of a queue, jam or row of parked cars — if for >= cluster_overlap_frac
    of the stop time >= cluster_min_others cars stand RIGHT NEXT to it at the same
    time (in a jam neighbours come and go — we count time, not
    neighbours that stood through the whole stop). Right next to — no further than
    cluster_radius diagonals of the SMALLER of the two boxes (a car close to the camera
    has a huge diagonal, and with the larger one the queue on the
    main road 650 px away became "neighbours"): a queue is a chain of cars bumper to
    bumper. Previously the car was discarded by an overlap with ANY congestion segment
    anywhere in the frame, and a single car that stood 36 s in the middle of the square
    (C3897, 210-246 s) was lost because a queue was waiting on red on the main road
    at that time.
    A "same zone" check is no better: the square zone is half the frame, and there is
    almost always someone waiting to exit.

    junction_zones — the junction itself PAST the light (crossroad*). For a car
    that stood there, the "waited for green" exemption does not apply: you may not stand
    in the middle of a junction, even if you move off with the phase (C3897:
    0-27 s and 210-250 s — single cars moved off 1-5 s after
    green). The "stands in a cluster" exemption remains: a dense mass of cars
    on the square is congestion (C3896, 40-70 s), not a dozen stopped_vehicle.

    Previously there was no separate threshold for queue_zone: a car
    that waited out one normal red cycle (15-40 s) in the queue without
    reaching the critical mass for congestion (see detect_congestion,
    min_count) was falsely counted as stopped_vehicle. The
    queue_min_duration threshold (60-90 s per the task) filters this out.

    road_zones: set of zone names that are physically the
    carriageway (roadway*, crossroad*, queue_zone, stop_line, crossing_*, see
    _scene_zone_sets). If given and fewer than on_road_frac (by default
    half) of the stop run's samples fall into these zones, the event is NOT
    written: most likely this is a car standing on the shoulder/in a parking spot/outside
    the annotated road, not one that "stopped in the middle of the carriageway"."""
    junction_zones = set(junction_zones)
    stationary = _stationary_runs(vehicle_objs, speed_thresh, max_gap)
    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        for s, e, c, diag in stationary[oid]:
            dur = e - s
            run_samples = [x for x in samples if s - 1e-6 <= x["t"] <= e + 1e-6]
            if road_zones:
                frac_on_road = (sum(1 for x in run_samples if road_zones & x["zones"])
                                  / max(len(run_samples), 1))
                if frac_on_road < on_road_frac:
                    continue
            # The first car in the queue stands not in queue_zone but on the stop line or
            # right on the crossing — that is also waiting for the signal, not stopped_vehicle.
            frac_in_queue = (sum(1 for x in run_samples if x["zones"] & set(SIGNAL_WAIT_ZONES))
                              / max(len(run_samples), 1))
            required = queue_min_duration if frac_in_queue > 0.5 else min_duration
            if dur < required:
                continue
            frac_junction = (sum(1 for x in run_samples if x["zones"] & junction_zones)
                             / max(len(run_samples), 1))
            on_junction = frac_junction > 0.5 and frac_in_queue <= 0.5
            if not on_junction and _released_by_green(light, s, e):
                continue
            near = [(max(s, os_), min(e, oe)) for other, runs in stationary.items() if other != oid
                    for os_, oe, oc, od in runs
                    if os_ < e and oe > s and np.hypot(*(oc - c)) <= cluster_radius * min(diag, od)]
            if _time_with_at_least(near, cluster_min_others) >= cluster_overlap_frac * dur:
                continue
            events.append([round(s, 2), round(e, 2), "stopped_vehicle"])
            ids.append(oid)
    return (events, ids) if return_ids else events


def _zones_by_prefix(zones, prefixes):
    """Zone names that equal one of prefixes exactly or start
    with 'prefix_'. This lets one physical meaning ("carriageway", "legitimate
    crossing") be split into several polygons to fit an awkward scene
    shape (roadway, roadway_before_queue, crossroad, crossroad_2 — all
    "carriageway"; crossing_far, crossing_near — all "crossing"),
    instead of one self-intersecting polygon, which cv2 cannot handle."""
    names = set()
    for name in zones:
        for p in prefixes:
            if name == p or name.startswith(p + "_"):
                names.add(name)
                break
    return names


RIDER_MIN_SPEED = 1.2     # box diagonals/s; walking: median 0.4, 99th percentile of jaywalkers 2.8
RIDER_MAX_ASPECT = 2.2    # box height / width; a standing or walking person is ~2.5-3.5
RIDER_IOA = 0.6   # fraction of the person's box inside the vehicle's box: riding on/in it


def _mark_riders(groups, person_objs):
    """sample["in_vehicle"] for people whose box in this frame lies almost entirely
    inside a vehicle's box: motorcyclist, cyclist, bus passenger at the window
    (C3902, 145 s: a motorcyclist was flagged as jaywalking)."""
    boxes = lambda rs: np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rs], np.float32)
    vehicles, persons = {}, {}
    for oid, recs in groups.items():
        target = vehicles if recs[0]["cls"] in VEHICLE_CLASSES else persons if oid in person_objs else None
        if target is not None:
            for k, r in enumerate(recs):
                target.setdefault(r["frame"], []).append((r, oid, k))
    for smp_list in person_objs.values():
        for smp in smp_list:
            smp["in_vehicle"] = False
    for frame, plist in persons.items():
        vlist = vehicles.get(frame)
        if not vlist:
            continue
        P, V = boxes([p[0] for p in plist]), boxes([v[0] for v in vlist])
        iw = np.clip(np.minimum(P[:, None, 2], V[None, :, 2]) - np.maximum(P[:, None, 0], V[None, :, 0]), 0, None)
        ih = np.clip(np.minimum(P[:, None, 3], V[None, :, 3]) - np.maximum(P[:, None, 1], V[None, :, 1]), 0, None)
        area = np.maximum((P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1]), 1.0)
        rider = ((iw * ih) >= RIDER_IOA * area[:, None]).any(axis=1)
        for (_r, oid, k), is_rider in zip(plist, rider):
            person_objs[oid][k]["in_vehicle"] = bool(is_rider)
    for oid, smp_list in person_objs.items():         # box shape, for riders without a vehicle box
        for smp, r in zip(smp_list, groups[oid]):
            smp["aspect"] = (r["y2"] - r["y1"]) / max(r["x2"] - r["x1"], 1.0)


def detect_jaywalking(person_objs, roadway_zones, crossing_zones=(), min_duration=1.0,
                       max_gap=1.0, return_ids=False):
    """Pedestrian on the carriageway OUTSIDE a legitimate crossing.

    roadway_zones/crossing_zones — sets of zone names (see _zones_by_prefix).
    crossing_zones are always excluded: in real annotations the
    "carriageway" and "crossing" zones almost always overlap slightly at the
    border (as here — crossroad_2 extends onto crossing_near), and
    without an explicit exclusion a person walking ALONG the zebra in the overlap
    would also be counted as jaywalking."""
    crossing_zones = set(crossing_zones)
    events, ids = [], []
    for oid, samples in person_objs.items():
        runs = sample_runs(
            samples,
            lambda s: (bool(roadway_zones & s["zones"]) and not (crossing_zones & s["zones"])
                       and not s.get("in_vehicle")),
            max_gap,
        )
        for s, e in runs:
            if e - s < min_duration:
                continue
            # The detector often misses a scooter/moped and sees only its rider (C3897, 3:54;
            # C3905, 0:37): no vehicle box to sit in. On the road a rider moves fast and the box
            # is squat (legs on the footboard, the box covers the scooter); a pedestrian's is tall.
            run = [x for x in samples if s <= x["t"] <= e]
            if (np.median([x["speed"] for x in run]) >= RIDER_MIN_SPEED
                    and np.median([x.get("aspect", 3.0) for x in run]) < RIDER_MAX_ASPECT):
                continue
            events.append([round(s, 2), round(e, 2), "jaywalking"])
            ids.append(oid)
    return (events, ids) if return_ids else events


STOP_LINE_MIN_STOP_SEC = 1.5   # shorter is not a stop, just slowing down


def _waited_then_entered_on_green(samples, t_from, wait_zone, junction_zones, speed_thresh=0.3,
                                  min_stop=STOP_LINE_MIN_STOP_SEC, light=None):
    """True if after t_from the car stood >= min_stop in wait_zone (past the
    stop line) and entered the junction no longer on red: this is stop_line, not
    running a red light."""
    stopped_since, waited = None, False
    for s in samples:
        if s["t"] < t_from:
            continue
        if s["zones"] & junction_zones and wait_zone not in s["zones"]:
            return waited and light.at(s["t"]) != "red"
        if wait_zone in s["zones"] and s["speed"] < speed_thresh:
            stopped_since = s["t"] if stopped_since is None else stopped_since
            waited = waited or s["t"] - stopped_since >= min_stop
        else:
            stopped_since = None
    return False


def detect_red_light(vehicle_objs, light, gate_zone="stop_line", exit_zone="crossing_far",
                      wait_zone=None, junction_zones=frozenset(), return_ids=False):
    """The car actually runs a red light: enters gate_zone (a wide
    queue zone before the crossing; entering it is fine even for a "proper" stop) on
    red AND REACHES exit_zone while the light is STILL red.

    Previously the light was checked only once — at the moment of entering gate_zone —
    and the event was written even if the car: (a) never reached
    exit_zone during the track (just stood in the queue), or (b) reached
    exit_zone already on green, having honestly waited out the light cycle. Both
    cases produced mass false positives on almost every car in the
    queue, because our gate_zone is wide (the whole queue front), not a
    thin line."""
    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        t_cross = first_entry_time(samples, gate_zone)
        if t_cross is None or light.at(t_cross) != "red":
            continue
        t_end, was_in_exit, violated = None, False, False
        for s in samples:
            if s["t"] < t_cross:
                continue
            if exit_zone in s["zones"]:
                if not was_in_exit:
                    # the moment of actually entering exit_zone — the light must
                    # be red right now, not only at t_cross
                    violated = light.at(s["t"]) == "red"
                was_in_exit = True
            elif was_in_exit:
                t_end = s["t"]
                break
        if not was_in_exit or not violated:
            continue  # did not reach exit_zone, or reached it already on green
        if wait_zone and _waited_then_entered_on_green(samples, t_cross, wait_zone, junction_zones, light=light):
            continue  # stopped past the line and went on green — that is stop_line
        if t_end is None:
            t_end = samples[-1]["t"]
        if t_end > t_cross:
            events.append([round(t_cross, 2), round(t_end, 2), "red_light"])
            ids.append(oid)
    return (events, ids) if return_ids else events


def detect_stop_line(vehicle_objs, light, zone="past_stop_line", speed_thresh=0.3, max_gap=2.0,
                     min_stop=STOP_LINE_MIN_STOP_SEC, return_ids=False):
    """stop_line per the task definition: the car STOPPED past the stop line on
    red without entering the junction; the event ends when the light turns green.

    zone — the area past the stop line: from the line to the far edge of the zebra,
    only across the width of our queue's lanes (past_stop_line). Previously only the
    strip up to the zebra counted, and a car that stopped with its front on the zebra
    (C3905, 1:18, 37 s on red) fell into no zone. One event per car per red phase;
    whoever then runs the red is red_light (filter in compute_events_debug)."""
    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        runs = sample_runs(samples, lambda s: s["speed"] < speed_thresh and zone in s["zones"], max_gap)
        last_end = None
        for s, e in runs:
            if e - s < min_stop or light.at(s) != "red":
                continue
            if last_end is not None and s < last_end:
                continue  # same red phase: the car crept forward a bit and stopped again
            t_green = light.next_after(s, "green")
            t_end = t_green if t_green is not None else e
            events.append([round(s, 2), round(t_end, 2), "stop_line"])
            ids.append(oid)
            last_end = t_end
    return (events, ids) if return_ids else events


# Zones with a SINGLE direction of travel, where wrong_way makes sense at all.
# The square/junction is not included: traffic legitimately goes in different directions there.
WRONG_WAY_ZONES = ("roadway", "roadway_before_queue")
WRONG_WAY_MIN_SAMPLES = 200
WRONG_WAY_MIN_CONCENTRATION = 0.6


def detect_wrong_way(vehicle_objs, roadway_zones, min_speed=0.15, angle_thresh=140.0,
                      min_duration=1.0, max_gap=1.5, return_ids=False):
    """A car drives against the prevailing flow on the carriageway.

    FIRST version, not calibrated. Instead of a hard-coded direction
    vector (which would have to be tuned by hand to the view of this
    particular camera), the reference flow direction is computed FROM THE
    VIDEO ITSELF: the circular mean of the velocity vectors of all cars on
    the carriageway in the same clip. This is more robust than a rough manual
    estimate of the direction from camera_own.md and works for a hidden test
    on the same camera without re-annotation. Trade-off: if in a clip really
    almost everyone drives "the wrong way" (the flow itself was reversed, e.g.
    roadworks/closure), the reference shifts and wrong_way does not fire —
    nothing like this was seen in the organisers' samples (camera_own.md).

    angle_thresh=140° is large on purpose: ordinary manoeuvres (lane change,
    turning on the square) change heading by 30-90°, oncoming traffic is
    ~180°. The threshold is closer to 180 than to 90 so a turn is not taken for wrong_way."""
    # The reference is computed SEPARATELY for each zone and only where the flow
    # is one-directional. Several roads meet on the square (crossroad*):
    # a single averaged "reference" flagged the ordinary flow from the
    # right-hand road as oncoming (visible in the debug videos C3896/C3902).
    zones_used = [z for z in WRONG_WAY_ZONES if z in roadway_zones]
    if not zones_used:
        return ([], []) if return_ids else []

    refs = {}
    for zname in zones_used:
        sx = sy = 0.0
        n = 0
        for samples in vehicle_objs.values():
            for s in samples:
                if s["speed"] >= min_speed and zname in s["zones"]:
                    sx += s["vx"] / max(s["speed"], 1e-6)
                    sy += s["vy"] / max(s["speed"], 1e-6)
                    n += 1
        if n < WRONG_WAY_MIN_SAMPLES:
            continue
        concentration = (sx ** 2 + sy ** 2) ** 0.5 / n   # 1 = all in one direction
        if concentration >= WRONG_WAY_MIN_CONCENTRATION:
            refs[zname] = (sx, sy)
        else:
            print(f"wrong_way: zone {zname} skipped — flow is not one-directional "
                  f"(concentration {concentration:.2f})")
    if not refs:
        return ([], []) if return_ids else []

    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        def against_flow(s):
            if s["speed"] < min_speed:
                return False
            for zname, (rx, ry) in refs.items():
                if zname in s["zones"]:
                    ang = _angle_between(s["vx"], s["vy"], rx, ry)
                    return ang is not None and ang >= angle_thresh
            return False

        for s, e in sample_runs(samples, against_flow, max_gap):
            if e - s >= min_duration:
                events.append([round(s, 2), round(e, 2), "wrong_way"])
                ids.append(oid)
    return (events, ids) if return_ids else events


def _heading_series(samples, min_speed):
    """[(t, heading_deg)] only for samples where the object is actually moving
    (otherwise heading is atan2(0,0) noise)."""
    out = []
    for s in samples:
        if s["speed"] >= min_speed:
            out.append((s["t"], float(np.degrees(np.arctan2(s["vy"], s["vx"])))))
    return out


def detect_illegal_u_turn(vehicle_objs, roadway_zones, min_speed=0.12, window_sec=4.0,
                           turn_angle_thresh=120.0, max_gap=2.0, return_ids=False):
    """180° U-turn on the carriageway. FIRST version: there is no "U-turn allowed
    here" annotation anywhere in zones.json/camera_own.md, so ANY detected
    U-turn on the carriageway (roadway*/crossroad*) counts as illegal.
    If the hidden test has a place where U-turns are allowed, this will
    give FPs — a separate exclusion zone will be needed, modelled on
    illegal_turn_exit.

    We detect not the "U-turn" geometrically (there is no lane annotation for
    the two sides) but the FACT of heading reversal: heading at t2 differs from
    heading at t1 by >= turn_angle_thresh, the window t2-t1 <= window_sec (a U-turn
    is a manoeuvre over several seconds, not instantaneous), and throughout the
    interval the object stays on the carriageway (otherwise it could be, for example,
    leaving the frame and reappearing with a different heading — a different
    physical meaning)."""
    if not roadway_zones:
        return ([], []) if return_ids else []

    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        heading = _heading_series(samples, min_speed)
        # sample index (by time) -> whether the object is on the carriageway
        on_road_at = {s["t"]: bool(roadway_zones & s["zones"]) for s in samples}
        ts_all = sorted(on_road_at)

        candidates = []
        for i in range(len(heading)):
            t1, h1 = heading[i]
            for j in range(i + 1, len(heading)):
                t2, h2 = heading[j]
                dt = t2 - t1
                if dt > window_sec:
                    break
                if dt < window_sec * 0.35:
                    continue  # too fast for a real U-turn — more likely heading noise
                diff = abs(h1 - h2)
                diff = min(diff, 360.0 - diff)
                if diff >= turn_angle_thresh:
                    # the whole range [t1, t2] must be on the carriageway
                    span = [t for t in ts_all if t1 - 1e-6 <= t <= t2 + 1e-6]
                    if span and all(on_road_at[t] for t in span):
                        candidates.append((t1, t2))
        merged = merge_intervals(candidates, gap=max_gap)
        for s, e in merged:
            events.append([round(s, 2), round(e, 2), "illegal_u_turn"])
            ids.append(oid)
    return (events, ids) if return_ids else events


TWO_WHEELERS = {1, 3}         # bicycle, motorcycle
# Mounting a sidewalk island. There is no official class for it (adding our own
# ids is not allowed — the harness drops them), so this is a DIAGNOSTIC event:
# visible in the visualisation, not written to predictions.json (solution.DIAGNOSTIC_CLASSES).
CURB_MOUNT = "curb_mount"
ISLAND_FOOTPRINT_FRAC = 0.3   # fraction of box height above its bottom: middle of the wheelbase
TURN_RATE_DEG_S = 12.0        # heading changes faster — the car is turning
TURN_PAD_MIN, TURN_PAD_MAX = 1.0, 4.0


def _turn_span(samples, t0, t1, min_speed=0.3):
    """Turn bounds around [t0, t1]: as long as heading changes faster than
    TURN_RATE_DEG_S, but no less than TURN_PAD_MIN and no more than TURN_PAD_MAX s
    on each side. In the annotations a turn spans the whole manoeuvre, while
    mounting the island is only its middle (0.4 s): without widening, IoU < 0.3."""
    hs = _heading_series(samples, min_speed)
    if len(hs) < 3:
        return t0 - TURN_PAD_MIN, t1 + TURN_PAD_MIN
    ts = np.array([t for t, _ in hs])
    hd = np.unwrap(np.radians([h for _, h in hs]))
    rate = np.degrees(np.abs(np.gradient(hd, ts)))
    start, end = t0 - TURN_PAD_MIN, t1 + TURN_PAD_MIN
    for k in range(int(np.searchsorted(ts, t0)) - 1, -1, -1):
        if rate[k] < TURN_RATE_DEG_S or t0 - ts[k] > TURN_PAD_MAX:
            break
        start = min(start, ts[k])
    for k in range(int(np.searchsorted(ts, t1)), len(ts)):
        if rate[k] < TURN_RATE_DEG_S or ts[k] - t1 > TURN_PAD_MAX:
            break
        end = max(end, ts[k])
    return float(max(start, samples[0]["t"])), float(min(end, samples[-1]["t"]))


ROUTE_TURN_DEV_DEG = 20.0   # heading deviated from the approach heading -> the turn has started
ROUTE_TURN_MIN_SPEED = 0.07  # slower than this, heading is noisy (standing before the turn)
ROUTE_TURN_MAX_SEC = 6.0     # earlier than this is lane shuffling in the jam on the square, not the turn


def _route_turn_span(samples, t_origin, t_junction, t_exit):
    """Turn of a car that came from the main road: start — the first sample after
    entering the square where heading deviated from the approach heading by more than
    ROUTE_TURN_DEV_DEG; end — entry into the exit, extended while heading is still
    changing. The turn often starts almost from standstill (C3896, 45: stood
    40-48 s, turned 49.6-55.9 s) — hence the low speed threshold."""
    hs = _heading_series(samples, ROUTE_TURN_MIN_SPEED)
    approach = [h for t, h in hs if t_origin <= t <= t_junction]
    if not approach:
        return _turn_span(samples, t_exit, t_exit)
    ref = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(approach))))))
    dev = lambda h: abs((h - ref + 180.0) % 360.0 - 180.0)
    lo = max(t_junction, t_exit - ROUTE_TURN_MAX_SEC)
    start = next((t for t, h in hs if lo <= t <= t_exit and dev(h) > ROUTE_TURN_DEV_DEG),
                 t_exit - TURN_PAD_MIN)
    _, end = _turn_span(samples, t_exit, t_exit)
    return float(max(start, samples[0]["t"])), float(end)


def detect_curb_mount(vehicle_objs, zones, min_speed=0.3, min_duration=0.3, max_gap=1.0):
    """A car mounts a sidewalk island (sidewalk*) while moving.
    The bottom of the box of a car driving diagonally is the bumper corner nearest
    to the camera, and it stays on the asphalt (C3905, 100 s: 0 of 7 frames on
    the island); we test the point ISLAND_FOOTPRINT_FRAC of the box height above
    the bottom — that is where the wheels are. Returns ([events], [oid])."""
    islands = [np.asarray(zones[n], np.float32) for n in _zones_by_prefix(zones, ("sidewalk",))]
    events, ids = [], []
    if not islands:
        return events, ids

    def on_island(x):
        (cx, cy), (_, gy) = x["c"], x["g"]
        foot = (cx, gy - ISLAND_FOOTPRINT_FRAC * 2.0 * (gy - cy))   # h = 2 * (bottom - centre)
        return any(in_zone(foot, poly) for poly in islands)

    for oid, samples in vehicle_objs.items():
        if samples[0]["cls"] in TWO_WHEELERS:
            continue  # a moped/bicycle legitimately rides onto the island edge (C3896, 24 s)
        runs = sample_runs(samples, lambda x: x["speed"] >= min_speed and on_island(x), max_gap)
        for s, e in runs:
            if e - s >= min_duration:
                s, e = _turn_span(samples, s, e)
                events.append([round(s, 2), round(e, 2), CURB_MOUNT])
                ids.append(oid)
    return events, ids


ILLEGAL_TURN_ORIGIN = ("queue_zone", "stop_line", "crossing_far")
APEX_TO_EXIT_MAX_SEC = 3.0


def detect_illegal_turn(vehicle_objs, zones, origin_zones=ILLEGAL_TURN_ORIGIN, return_ids=False):
    """A turn in a prohibited direction — by ROUTE, not by where the turn happens.

    At this junction the prohibited manoeuvre is: arrive from the main road (queue
    -> far crossing), go deep into the square, turn around there (zone
    illegal_turn_apex*) and leave back-left through the lower end of the near
    crossing (zone illegal_turn_exit*). The allowed turn into the same street is
    immediately right, onto the upper end of the crossing. Verified by eye on the
    samples: C3896 45 (49.6 s) and 830 (284 s) are violations. Without the U-turn apex
    the rule caught cars that simply drive left along the bottom edge of the
    frame from the right edge (C3896 628, C3897 602, C3902 x3: the tracker merged them
    with a car from the main road), and cars that entered the square from the right
    (C3905 457). The old rule "a 50-150 degree turn inside a
    polygon" caught detours and missed these U-turns.

    The segment is the turn itself (_route_turn_span): from the moment heading
    deviated from the approach heading until entering illegal_turn_exit (+ extended
    while heading is still changing). Mounting the island is separate: detect_curb_mount."""
    events, ids = [], []
    exit_zones = _zones_by_prefix(zones, ("illegal_turn_exit",))
    apex_zones = _zones_by_prefix(zones, ("illegal_turn_apex",))
    junction = _zones_by_prefix(zones, ("crossroad",))
    origin = set(origin_zones)
    if not exit_zones or not apex_zones:
        return (events, ids) if return_ids else events
    for oid, samples in vehicle_objs.items():
        t_origin = next((x["t"] for x in samples if x["zones"] & origin), None)
        if t_origin is None:
            continue
        after = [x for x in samples if x["t"] > t_origin]
        t_apex = next((x["t"] for x in after if x["zones"] & apex_zones), None)
        if t_apex is None:
            continue
        t_exit = next((x["t"] for x in after if x["t"] > t_apex and x["zones"] & exit_zones), None)
        # a U-turn is a continuous manoeuvre: from the apex straight into the exit (on the
        # samples 0.3-1 s). 6-8 s means the tracker merged the tracks of two different cars (C3902)
        if t_exit is not None and not any(
                x["zones"] & apex_zones for x in after if t_exit - APEX_TO_EXIT_MAX_SEC <= x["t"] < t_exit):
            continue
        if t_exit is None or not any(x["zones"] & junction for x in after if x["t"] < t_exit):
            continue
        t_junction = next(x["t"] for x in after if x["zones"] & junction)
        s, e = _route_turn_span(samples, t_origin, t_junction, t_exit)
        events.append([round(s, 2), round(e, 2), "illegal_turn"])
        ids.append(oid)
    return (events, ids) if return_ids else events


def detect_solid_line_crossing(vehicle_objs, zones, settle_sec=1.5, max_gap=2.0,
                                margin_norm=0.25, return_ids=False):
    """Crossing a solid line — requires LINE zones 'solid_line'/
    'solid_line_*' (exactly 2 points each, see define_zones.py, key 7).
    Without such a line in zones.json it finds nothing and does not crash.

    Unlike polygon zones, a line has no "inside" — cv2.pointPolygonTest
    does not apply, so the side is given by the sign of the cross product
    (_line_side). A crossing is a sign change between consecutive samples
    of one object, but with hysteresis: a side counts only if the
    anchor point is further from the line than margin_norm box diagonals —
    otherwise a car driving ALONG the line "crosses" it dozens of times from
    box jitter. end = start + settle_sec (a fixed margin for the object
    to be "fully in the new lane" — a second to a second and a half after crossing
    in this scene matches the scale of a car on the carriageway; fine-tuning
    without annotated machinery is impossible, this is a rough but safe
    estimate of the event duration)."""
    lines = {name: pts for name, pts in zones.items()
             if name == "solid_line" or name.startswith("solid_line_")}
    if not lines:
        return ([], []) if return_ids else []

    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        for line in lines.values():
            (x1, y1), (x2, y2) = line[0], line[1]
            seg_len = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
            prev_side, prev_t = None, None
            for s_ in samples:
                if prev_t is not None and s_["t"] - prev_t > max_gap:
                    prev_side = None  # the track was interrupted — do not trust a "crossing" across the gap
                prev_t = s_["t"]
                # projection onto the segment: there is no crossing beyond the line's ends
                px, py = s_["g"]
                u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg_len ** 2
                if not 0.0 <= u <= 1.0:
                    continue
                dist = _line_side(s_["g"], line) / seg_len  # signed distance, px
                if abs(dist) < margin_norm * s_["diag"]:
                    continue  # in the "dead zone" near the line — do not update the side
                side = dist > 0
                if prev_side is not None and side != prev_side:
                    t_cross = s_["t"]
                    events.append([round(t_cross, 2), round(t_cross + settle_sec, 2),
                                   "solid_line_crossing"])
                    ids.append(oid)
                prev_side = side
    return (events, ids) if return_ids else events


def detect_failure_to_yield(vehicle_objs, person_objs, crossing_zones, max_gap=2.0,
                             ped_merge_gap=0.5, min_duration=0.2, return_ids=False):
    """A car drives through a crossing while a pedestrian is on it (or stepping
    onto it). Computed PER crossing separately (crossing_far and
    crossing_near are different physical crossings; we do not mix a pedestrian on
    one with a car on the other)."""
    if not crossing_zones:
        return ([], []) if return_ids else []

    events, ids = [], []
    for zone in crossing_zones:
        ped_runs = []
        for samples in person_objs.values():
            ped_runs += sample_runs(samples, lambda s, z=zone: z in s["zones"], max_gap)
        ped_intervals = merge_intervals(ped_runs, gap=ped_merge_gap)
        if not ped_intervals:
            continue

        for oid, samples in vehicle_objs.items():
            veh_runs = sample_runs(samples, lambda s, z=zone: z in s["zones"], max_gap)
            for s, e in veh_runs:
                if e - s < min_duration:
                    continue
                if any(overlap((s, e), pi) > 0 for pi in ped_intervals):
                    events.append([round(s, 2), round(e, 2), "failure_to_yield"])
                    ids.append(oid)
    return (events, ids) if return_ids else events


# Thresholds for accident/near_miss — the same units ("body lengths" and
# seconds) that were already calibrated (informally, by eye) for the TTC heuristic of
# RiskEstimator in solution.py; here it is NOT causal (the whole clip at once), which
# lets us look both backward and forward from the moment of closest approach —
# in particular, to tell an accident (after the approach the objects STOP
# and stay stopped) from ordinary dense but safe traffic (bboxes can also
# overlap because of camera perspective, especially in a queue — the separate
# STOP feature is needed precisely to cut this source of FP).
ACCIDENT_DIST_NORM = 0.35     # "body lengths" — the boxes actually overlap
DANGER_DIST_NORM = 1.6        # closer than this is already a dangerous approach
TTC_DANGER_NONCAUSAL = 1.5    # s
STILL_SPEED = 0.12
MOVING_SPEED = 0.25
STILL_HOLD_SEC = 1.5          # how long it must stand to count as a "stop caused by a crash"
BRAKE_DROP_RATIO = 0.55       # relative speed drop for "hard braking"
PRE_CONTACT_SEC = 1.5         # window BEFORE contact in which one of the pair must actually be moving
# near_miss (b) — a single hard braking with no second participant.
# DISABLED: every car braking before a queue on red produces
# such an event — on this camera that is hundreds of FP per clip.
NEAR_MISS_FROM_BRAKING = False
PAIR_PREFILTER_NORM = 2.0     # pairs that never came closer than this are not considered


def _candidate_pairs(all_objs, thresh=PAIR_PREFILTER_NORM):
    """Pairs of objects that were closer than thresh "body lengths" in at least
    one frame. All objects are sampled on the same frame grid, so it is
    enough to compare objects within each frame (numpy). Without this,
    enumerating all pairs over the whole clip is O(N^2) over thousands of tracks
    and tens of minutes per video."""
    by_t = {}
    for key, samples in all_objs.items():
        for s in samples:
            by_t.setdefault(s["t"], []).append((key, s["c"][0], s["c"][1], s["diag"]))
    pairs = set()
    for items in by_t.values():
        if len(items) < 2:
            continue
        keys = [it[0] for it in items]
        arr = np.array([it[1:] for it in items], dtype=np.float64)
        dx = arr[:, 0][:, None] - arr[:, 0][None, :]
        dy = arr[:, 1][:, None] - arr[:, 1][None, :]
        avg_diag = (arr[:, 2][:, None] + arr[:, 2][None, :]) / 2.0
        close = np.sqrt(dx ** 2 + dy ** 2) / avg_diag < thresh
        ii, jj = np.nonzero(np.triu(close, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            a, b = keys[i], keys[j]
            if a[0] == "p" and b[0] == "p":
                continue
            pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def _was_moving_before(samples, t0, window=PRE_CONTACT_SEC, min_speed=MOVING_SPEED):
    return any(t0 - window <= s["t"] <= t0 and s["speed"] >= min_speed for s in samples)


def _object_still_run_after(samples, t0, still_speed=STILL_SPEED, max_gap=2.0):
    """Returns (t_end_of_stop) if the object, starting at roughly t0,
    continuously (allowing for max_gap) keeps speed < still_speed for at least
    STILL_HOLD_SEC seconds; otherwise None. t_end is the moment the object
    started moving AGAIN (or the last sample of the track if it never moved)."""
    still = [s for s in samples if s["t"] >= t0 - 1e-6]
    if not still:
        return None
    runs = sample_runs(still, lambda s: s["speed"] < still_speed, max_gap)
    for s, e in runs:
        if s <= t0 + 1.0 and e - s >= STILL_HOLD_SEC:  # the stop began shortly after t0
            return e
    return None


def _brake_events(samples, window_sec=2.0, min_speed=MOVING_SPEED, max_gap=1.5):
    """Hard-braking intervals without regard to another object: speed
    drops from >= min_speed to < min_speed*(1-BRAKE_DROP_RATIO) within
    window_sec. A cheap single-object signal (like _max_brake_ratio in
    RiskEstimator, but non-causal and on an explicit sliding window rather than
    a fixed history of RISK_HISTORY samples)."""
    events = []
    n = len(samples)
    for i in range(n):
        if samples[i]["speed"] < min_speed:
            continue
        for j in range(i + 1, n):
            dt = samples[j]["t"] - samples[i]["t"]
            if dt > window_sec:
                break
            if samples[j]["speed"] < min_speed * (1 - BRAKE_DROP_RATIO):
                events.append((samples[i]["t"], samples[j]["t"]))
                break
    return merge_intervals(events, gap=max_gap)


def detect_accidents_and_near_misses(vehicle_objs, person_objs, max_dt=0.6, return_ids=False):
    """Accidents and near misses — the only two classes not tied to the
    camera zones (pure track kinematics), but also the only ones with no
    ready-made training signal at all — this is a first heuristic
    attempt that needs calibration on real annotations.

    accident: a pair of objects (car-car or car-pedestrian) closes in
    to dist_norm < ACCIDENT_DIST_NORM ("the boxes effectively coincided") AND
    after that at least one of them stops (speed < STILL_SPEED)
    for at least STILL_HOLD_SEC in a row. The "stop after
    contact" requirement is an attempt to cut false positives from plain visual
    box overlap due to perspective (cars in different lanes/at different
    scene depths whose boxes intersect in the 2D frame but physically
    do not touch) — if after the "contact" both carry on
    driving as if nothing happened, it is most likely not a crash but a perspective
    tracking artefact.

    near_miss: either (a) the pair closed to DANGER_DIST_NORM with a closing speed
    high enough for TTC < TTC_DANGER_NONCAUSAL, but WITHOUT contact
    (dist_norm never dropped below ACCIDENT_DIST_NORM) and then separated,
    or (b) a single object braked hard (_brake_events) while not
    already covered by a counted accident. (a) is more specific and more likely
    accurate; (b) is a rough single-object signal (hard braking also happens without
    a collision threat, e.g. before an ordinary red) and gives more
    FP — when calibrating with evaluate.py --per-video, look here
    first if near_miss fires falsely too often; if
    needed, (b) can simply be switched off, keeping only (a)."""
    all_objs = {}
    for oid, samples in vehicle_objs.items():
        all_objs[("v", oid)] = samples
    for oid, samples in person_objs.items():
        all_objs[("p", oid)] = samples
    accidents, accident_pairs = [], []
    near_miss_candidates = []

    for ka, kb in _candidate_pairs(all_objs):
        sa, sb = all_objs[ka], all_objs[kb]
        approach = _closest_approach(sa, sb, max_dt=max_dt)
        if not approach:
            continue

        contact_runs = sample_runs(
            [{"t": t, "_ok": d < ACCIDENT_DIST_NORM} for t, d, _ in approach],
            lambda x: x["_ok"], max_gap=max_dt * 2,
        )
        had_accident = False
        for s, e in contact_runs:
            t_end_a = _object_still_run_after(sa, s)
            t_end_b = _object_still_run_after(sb, s)
            t_end = max(t_end_a or 0.0, t_end_b or 0.0)
            if t_end_a is None and t_end_b is None:
                continue  # there was contact, but both kept driving — probably a perspective artefact
            if not (_was_moving_before(sa, s) or _was_moving_before(sb, s)):
                continue  # both were standing before the "contact" — queue neighbours whose boxes
                          # overlapped in perspective, not a crash
            accidents.append([round(s, 2), round(max(t_end, e), 2), "accident"])
            accident_pairs.append((ka[1] if ka[0] == "v" else kb[1]))
            had_accident = True
        if had_accident:
            continue  # this pair is already "used up" by accident — do not duplicate it as near_miss

        min_dist, min_t = None, None
        for t, d, closing in approach:
            if d < DANGER_DIST_NORM and (min_dist is None or d < min_dist):
                min_dist, min_t = d, t
        if min_dist is not None and min_t is not None:
            # look for the moment around the minimum where TTC was dangerous
            danger_t = None
            for t, d, closing in approach:
                if closing and closing > 1e-6 and d / closing < TTC_DANGER_NONCAUSAL \
                        and abs(t - min_t) <= 3.0:
                    danger_t = t if danger_t is None else min(danger_t, t)
            if danger_t is not None:
                # near_miss window: from the dangerous approach until they separate past DANGER_DIST_NORM
                end_t = min_t
                for t, d, _ in approach:
                    if t >= min_t and d >= DANGER_DIST_NORM:
                        end_t = t
                        break
                else:
                    end_t = approach[-1][0]
                if end_t > danger_t:
                    near_miss_candidates.append((danger_t, end_t,
                                                  ka[1] if ka[0] == "v" else kb[1]))

    # (b) hard braking not tied to a specific other car — only
    # for objects not already flagged in accident above in the same interval
    accident_intervals_by_obj = {}
    for (s, e, _), oid in zip(accidents, accident_pairs):
        accident_intervals_by_obj.setdefault(oid, []).append((s, e))
    for oid, samples in (vehicle_objs.items() if NEAR_MISS_FROM_BRAKING else ()):
        busy = merge_intervals(accident_intervals_by_obj.get(oid, []))
        for s, e in _brake_events(samples):
            if not any(overlap((s, e), b) > 0 for b in busy):
                near_miss_candidates.append((s, e, oid))

    # merge near_miss candidates per object (same car, close intervals)
    by_obj: dict = {}
    for s, e, oid in near_miss_candidates:
        by_obj.setdefault(oid, []).append((s, e))
    near_misses, nm_ids = [], []
    for oid, intervals in by_obj.items():
        for s, e in merge_intervals(intervals, gap=1.0):
            near_misses.append([round(s, 2), round(e, 2), "near_miss"])
            nm_ids.append(oid)

    if return_ids:
        return accidents, accident_pairs, near_misses, nm_ids
    return accidents, near_misses


def merge_same_class_overlaps(events):
    """Overlapping segments of one class are ONE event per the task FAQ
    ("two events of one class at the same time -> one segment
    covering both"), not a reason to lose one of them.

    Previously this was drop_same_class_overlaps, copying the behaviour of
    run_submission.py (on an overlap within a class it keeps the earlier
    segment and drops the rest as FP/duplicate) — as a safety net on the
    harness side this is reasonable, but if WE filter this way before
    submitting, we lose recall every time two DIFFERENT cars
    stand/run a red at the same time: one of two real events
    silently disappears instead of being merged into a covering segment."""
    by_label: dict[str, list[tuple[float, float]]] = {}
    for s, e, lbl in events:
        by_label.setdefault(lbl, []).append((s, e))
    out = []
    for lbl, intervals in by_label.items():
        for s, e in merge_intervals(intervals, gap=0.0):
            out.append([round(s, 2), round(e, 2), lbl])
    return sorted(out)


def compute_events(records, zones, light_samples=None, classes=None):
    """Single entry point without file I/O — called by pipeline.infer().

    Args:
        records: list of track records (already with stitched_id, see
            stitch_tracks.stitch_records()).
        zones: dict {zone_name: np.ndarray[[x,y],...]} — as from load_zones().
            The "carriageway" zones for jaywalking are collected by naming
            convention: 'roadway' and anything starting with 'roadway_' or
            'crossroad' (roadway, roadway_before_queue, crossroad,
            crossroad_2, ...). Legitimate crossing zones ('crossing_far',
            'crossing_near', any 'crossing_*') are automatically excluded
            from the carriageway, even if they intersect geometrically.
        light_samples: list of [t_sec, state] from light_state.run_light_state(),
            or None if light_roi is not annotated / we chose not to compute
            red_light/stop_line.
        classes: which classes to compute (None — all). Rules for the other classes
            are not run at all: on a busy 10-minute video the
            experimental rules cost tens of seconds of the budget.

    Returns:
        List of [start_sec, end_sec, label] events, already without overlaps
        within a class.
    """
    events = compute_events_debug(records, zones, light_samples, classes)
    return merge_same_class_overlaps([e[:3] for e in events])


def _scene_zone_sets(zones):
    """Semantic groups of scene zones (shared by compute_events and _debug):
      roadway    — carriageway (roadway*, crossroad*);
      ped_safe   — where a pedestrian is allowed: crossings (crossing*) and sidewalk
                   islands (sidewalk*: they lie inside the crossroad polygon, and
                   people walking from one zebra to another across an island
                   produced most of the false jaywalking);
      road       — where a standing car obstructs traffic (stopped_vehicle):
                   carriageway + crossings + queue + stop line."""
    roadway = _zones_by_prefix(zones, ("roadway", "crossroad"))
    crossing = _zones_by_prefix(zones, ("crossing",))
    ped_safe = crossing | _zones_by_prefix(zones, ("sidewalk",))
    road = roadway | crossing | ({"queue_zone", "stop_line"} & set(zones))
    return roadway, ped_safe, road


def compute_events_debug(records, zones, light_samples=None, classes=None):
    """All rules, with the offending object attached. compute_events() is the same
    plus merge_same_class_overlaps (a single implementation: the submission and the
    visualisation cannot diverge).

    Differences from compute_events():
      - events do NOT go through merge_same_class_overlaps: two different
        violators overlapping in time remain two separate
        records rather than one covering segment — otherwise the
        link to a specific object is lost;
      - each record is [start, end, label, object_id], where object_id is the
        stitched_id of the offender (see stitch_tracks.py). For "congestion"
        object_id is always None: this event is about a cluster of several
        cars at once, not about one — use zones['queue_zone'] +
        the box coordinates at that moment to highlight the whole cluster.

    Returns a list of [start, end, label, object_id_or_None], sorted
    by start time.
    """
    groups = group_by_object(records)
    vehicle_objs = {oid: annotate(recs, zones) for oid, recs in groups.items()
                    if recs[0]["cls"] in VEHICLE_CLASSES}
    person_objs = {oid: annotate(recs, zones) for oid, recs in groups.items()
                   if recs[0]["cls"] == PERSON_CLASS}
    _mark_riders(groups, person_objs)

    roadway_zones, ped_safe_zones, road_zones = _scene_zone_sets(zones)
    crossing_zones = _zones_by_prefix(zones, ("crossing",))

    light = LightState(light_samples) if light_samples else None
    want = (lambda *labels: True) if classes is None else (lambda *labels: bool(set(labels) & set(classes)))
    out = []

    def run(label, fn, per_object=True):
        """One rule: only if its class is requested, and with its own
        safety net — a crash of an experimental rule on an unfamiliar scene
        must not wipe out the video's other classes."""
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001 — log and continue
            print(f"[rules] {label} failed: {exc!r}")
            return [], []
        events, ids = res if per_object else (res, [None] * len(res))
        out.extend([s, e, lbl, oid] for (s, e, lbl), oid in zip(events, ids))
        return events, ids

    if want("congestion"):
        run("congestion", lambda: detect_congestion(
            vehicle_objs, signal_zones={"queue_zone"} & set(zones) or {"queue_zone"},
            junction_zones=_zones_by_prefix(zones, ("crossroad",)), light=light), per_object=False)
    if want("stopped_vehicle"):
        run("stopped_vehicle", lambda: detect_stopped_vehicle(
            vehicle_objs, road_zones=road_zones, junction_zones=_zones_by_prefix(zones, ("crossroad",)),
            light=light, return_ids=True))
    if want("jaywalking"):
        run("jaywalking", lambda: detect_jaywalking(person_objs, roadway_zones, ped_safe_zones,
                                                    return_ids=True))

    if light is not None and "stop_line" in zones and want("red_light", "stop_line"):
        stop_zone = "past_stop_line" if "past_stop_line" in zones else "stop_line"
        red, red_ids = [], []
        if "crossing_far" in zones:
            red, red_ids = run("red_light", lambda: detect_red_light(
                vehicle_objs, light, wait_zone=stop_zone, junction_zones=_zones_by_prefix(zones, ("crossroad",)),
                return_ids=True))
        if want("stop_line"):
            # stop_line — "stopped past the stop line WITHOUT entering the junction"; whoever
            # then ran the red is red_light (C3902, 97 s: a motorcycle got both)
            red_set = set(red_ids)

            def stop_line():
                events, ids = detect_stop_line(vehicle_objs, light, zone=stop_zone, return_ids=True)
                keep = [k for k, oid in enumerate(ids) if oid not in red_set]
                return [events[k] for k in keep], [ids[k] for k in keep]
            run("stop_line", stop_line)
        if not want("red_light"):
            out[:] = [ev for ev in out if ev[2] != "red_light"]

    if want("wrong_way"):
        run("wrong_way", lambda: detect_wrong_way(vehicle_objs, roadway_zones, return_ids=True))
    if want("illegal_u_turn"):
        run("illegal_u_turn", lambda: detect_illegal_u_turn(vehicle_objs, roadway_zones, return_ids=True))
    if want("illegal_turn"):
        run("illegal_turn", lambda: detect_illegal_turn(vehicle_objs, zones, return_ids=True))
    if want(CURB_MOUNT):
        run(CURB_MOUNT, lambda: detect_curb_mount(vehicle_objs, zones))
    if want("solid_line_crossing"):
        run("solid_line_crossing", lambda: detect_solid_line_crossing(vehicle_objs, zones, return_ids=True))
    if want("failure_to_yield"):
        run("failure_to_yield", lambda: detect_failure_to_yield(vehicle_objs, person_objs, crossing_zones,
                                                                return_ids=True))
    if want("accident", "near_miss"):
        def accidents():
            acc, _acc_ids, nm, nm_ids = detect_accidents_and_near_misses(
                vehicle_objs, person_objs, return_ids=True)
            # accident is about TWO objects at once; the object_id of one "offender"
            # would be misleading when highlighting, hence None (as for congestion)
            return acc + nm, [None] * len(acc) + list(nm_ids)
        run("accident/near_miss", accidents)

    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------- main (dev only)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="*.stitched.json from stitch_tracks.py")
    ap.add_argument("--zones", required=True)
    ap.add_argument("--light", default=None, help="*.json from light_state.py (for red_light/stop_line)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta, records = load_tracks(args.tracks)
    zones = load_zones(args.zones)
    light_samples = None
    if args.light:
        light_samples = json.loads(Path(args.light).read_text())["samples"]

    events = compute_events(records, zones, light_samples)

    fps = meta.get("fps", 25.0)
    duration = meta.get("n_frames_total", 0) / fps if fps else 0.0

    out_path = Path(args.out) if args.out else Path("predictions") / (Path(meta["video"]).stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"video": meta["video"], "duration": round(duration, 2), "fps": fps, "events": events}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    by_class = {}
    for _, _, lbl in events:
        by_class[lbl] = by_class.get(lbl, 0) + 1
    print(f"{meta['video']}: {len(events)} events -> {out_path}  {by_class}")


if __name__ == "__main__":
    main()