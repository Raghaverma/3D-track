#!/usr/bin/env python3
"""
Combine TWO models from delivery_bundle to mark release/pitching/impact
on the ball itself:

  1. The event-timing model (model/, via cricket_infer.py) -- decides WHEN
     each event happens (timestamp + frame number). It has no idea where
     the ball is in the frame.
  2. The ball-tracker model (cricket_ball_tracker_repaired/, via
     track_ball.py) -- a trained per-frame ball detector (HRNet-W18,
     epoch 9, val recall ~74% @ threshold 0.10). It has no idea which
     frames are "events".

This script runs both, then for each event looks up the ball's tracked
pixel position at that event's frame and draws a color-coded ring right on
the ball (green=release, yellow=pitching, red=impact). The ball trail is
drawn in the broadcast style: a blue shaded tube that is thick and
round-ended at release and tapers toward the bounce, a sharp V at the
bounce (pitching point), then a thinner, lighter-blue segment to the ball.
It grows with the ball and stays on screen until the next delivery / end of
the clip. A corner panel still lists timestamp/frame/speed as plain text,
same as app.py.

Usage:
    python3 app1.py
    python3 app1.py input.mp4 output.mp4
    python3 app1.py input.mp4 output.mp4 --bundle-dir /path/to/delivery_bundle
    python3 app1.py input.mp4 output.mp4 --device cpu --ball-device cpu
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_venv_dir = os.path.join(HERE, 'myenv')
_venv_python = os.path.join(_venv_dir, 'bin', 'python3')
if (os.path.isfile(_venv_python)
        and os.path.realpath(sys.prefix) != os.path.realpath(_venv_dir)):
    os.execv(_venv_python, [_venv_python] + sys.argv)

import argparse
import json
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

DEFAULT_BUNDLE_DIR = HERE
BALL_TRACKER_DIR = os.path.join(DEFAULT_BUNDLE_DIR, 'cricket_ball_tracker_repaired')
VIDEOS_DIR = os.path.join(DEFAULT_BUNDLE_DIR, 'videos')
OUTPUT_DIR = os.path.join(DEFAULT_BUNDLE_DIR, 'output')
VIDEO_EXTS = ('.mp4', '.mov', '.m4v', '.avi', '.mkv')

# ============================================================
# To test a specific video by just running `python3 app1.py` with no
# arguments, put its filename here (looked up in VIDEOS_DIR, i.e.
# delivery_bundle/videos/). Leave it as None to auto-pick the most
# recently added clip in that folder instead.
#
INPUT_VIDEO_NAME = "83_kmph.mp4"
# ============================================================

# ============================================================
# Batch sweep: set either (or both) of these to run every video x
# confidence combination in one `python3 app1.py`, no CLI args needed.
# Each combo gets its own output file, named with its confidence so you
# can compare them side by side:
#   output/<video>_conf<value>_tracked.mp4
#   output/<video>_conf<value>_timestamps.json
#
# Leave both as None for the normal single-video behavior above
# (INPUT_VIDEO_NAME + track_ball.py's own default confidence/--ball-confidence).
#
# Example:
#   BATCH_VIDEOS = ["60_kmph.mp4", "84_kmph.mp4", "89_kmph.mp4"]
#   BATCH_BALL_CONFIDENCES = [0.05, 0.10, 0.20, 0.30]
# runs all 3 videos at all 4 confidence thresholds -- 12 renders total.
#
BATCH_VIDEOS = None                 # e.g. ["60_kmph.mp4", "84_kmph.mp4"]; None = [INPUT_VIDEO_NAME]
BATCH_BALL_CONFIDENCES = [0.65]    # e.g. [0.05, 0.10, 0.20, 0.30]; None = [--ball-confidence]
# ============================================================


def _newest_video(*dirs):
    best_path, best_mtime = None, -1.0
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith(VIDEO_EXTS):
                p = os.path.join(d, name)
                mtime = os.path.getmtime(p)
                if mtime > best_mtime:
                    best_path, best_mtime = p, mtime
    return best_path


def default_output_for(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(OUTPUT_DIR, '%s_tracked.mp4' % stem)


def batch_output_for(video_path, ball_confidence):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    conf_tag = 'default' if ball_confidence is None else ('%.3f' % ball_confidence)
    return os.path.join(OUTPUT_DIR, '%s_conf%s_tracked.mp4' % (stem, conf_tag))


def _resolve_named_video(name):
    """Look up INPUT_VIDEO_NAME: as an explicit path if it has one, else by
    filename in VIDEOS_DIR (the usual place), OUTPUT_DIR, then the bundle
    root -- in that order. Returns the resolved path, or None if the name
    isn't found anywhere."""
    if not name:
        return None
    if os.path.isabs(name) or os.path.dirname(name):
        return name if os.path.isfile(name) else None
    for d in (VIDEOS_DIR, OUTPUT_DIR, DEFAULT_BUNDLE_DIR):
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate):
            return candidate
    return None


if INPUT_VIDEO_NAME:
    DEFAULT_INPUT_VIDEO = _resolve_named_video(INPUT_VIDEO_NAME)
else:
    DEFAULT_INPUT_VIDEO = _newest_video(VIDEOS_DIR, DEFAULT_BUNDLE_DIR)
# Deliberately not raising here: --help and an explicit CLI video argument
# must keep working even when INPUT_VIDEO_NAME is misspelled. main() checks
# this once it knows whether the caller actually needed the default.

EVENT_ORDER = ('release', 'pitching', 'impact')
EVENT_COLOR = {
    'release':  (60, 220, 110),
    'pitching': (255, 210, 60),
    'impact':   (255, 70, 70),
}

MARKER_SECONDS = 1.0        # how long the ring stays on the ball per event
BALL_SEARCH_FRAMES = 6      # if the exact event frame has no ball detection,
                            # look this many frames either side for one

PANEL_MARGIN = 20
PANEL_LINE_H = 46
PANEL_FONT_SIZE = 30
SPEED_COLOR = (255, 255, 255)

DEFAULT_PITCH_DISTANCE_M = 16
PITCH_DISTANCES_FILE = os.path.join(VIDEOS_DIR, 'pitch_distances.json')

# ---- ball trail style (blue shaded tube, matches the reference) ----
TRAIL_ALPHA = 250

# Tube widths are proportional to the trail's OWN pre-bounce arc length (the
# reference's proportions, measured from the original screenshot), so the
# tube keeps the same slender look whatever the camera distance / zoom is.
# A fixed fraction of the frame width cannot do that: it makes a long trail
# look thin and a short or close-up one look like a fat stub.
TRAIL_START_PER_LEN = 0.046      # width at the release end
TRAIL_PITCH_PER_LEN = 0.0269     # width where the pre-bounce tube reaches the bounce
TRAIL_POST_START_PER_LEN = 0.0203  # post-bounce arm, right after the bounce
TRAIL_POST_END_PER_LEN = 0.0169    # ... tapering to this at the ball
# Safety clamp for the release-end width, as a fraction of frame width; the
# other widths are scaled by the same factor so the proportions never change.
TRAIL_START_MIN_FRAC = 0.007
TRAIL_START_MAX_FRAC = 0.021
TRAIL_TAPER_POWER = 1.25         # >1 keeps the tube thick for longer before it tapers (reference look)
TRAIL_SS = 3                     # supersampling factor for anti-aliasing
TRAIL_SAMPLES_PER_SEG = 10       # catmull-rom samples per raw-point segment
TRAIL_LIGHT = (0.30, -0.95)      # light direction (image coords: mostly from above)

# Shaded-tube layers: (colour, width fraction). Drawn dark -> light, each
# narrower layer is pushed toward the lit side of the tube, which gives the
# dark-navy-underside -> light-blue-top gradient seen in the reference.
TRAIL_LAYERS_PRE = (
    ((14, 36, 100), 1.00),
    ((26, 58, 142), 0.78),
    ((42, 78, 170), 0.55),
    ((58, 94, 184), 0.32),
)
# the thin segment after the bounce is a lighter periwinkle blue
TRAIL_LAYERS_POST = (
    ((62, 90, 178), 1.00),
    ((84, 112, 202), 0.70),
    ((110, 136, 216), 0.42),
)


# -------------------------------------------------------
# Fonts / PNG rendering
# -------------------------------------------------------

def find_font(size):
    for path in (
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_ball_marker_png(label, out_path):
    """A colored ring + word label, anchored so the ring's centre lands
    exactly on the ball's detected pixel position. Returns (anchor_x, anchor_y)."""
    color = EVENT_COLOR[label]
    W, H = 260, 90
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = 45, H // 2
    outer_r = 24
    draw.ellipse((cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r),
                 outline=color + (255,), width=5)

    font = find_font(28)
    word = label.upper()
    bbox = draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 12, 7
    pill_x0 = cx + outer_r + 10
    pill_y0 = cy - (th // 2) - pad_y
    pill_x1 = pill_x0 + tw + 2 * pad_x
    pill_y1 = pill_y0 + th + 2 * pad_y
    draw.rounded_rectangle((pill_x0, pill_y0, pill_x1, pill_y1),
                           radius=9, fill=color + (230,))
    draw.text((pill_x0 + pad_x - bbox[0], pill_y0 + pad_y - bbox[1]),
              word, fill=(15, 15, 15), font=font)

    img.save(out_path)
    return (cx, cy)


def _catmull_rom(xy, samples_per_segment=10):
    """Smooth curve through the (already-smoothed) points via Catmull-Rom
    spline interpolation, so the trail reads as one fluid arc instead of
    straight jagged segments between noisy per-frame detections."""
    n = len(xy)
    if n < 3:
        return xy
    padded = [xy[0]] + list(xy) + [xy[-1]]
    curve = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for s in range(samples_per_segment):
            t = s / samples_per_segment
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            curve.append((x, y))
    curve.append(xy[-1])
    return curve


# ---- ball trail (blue shaded tube + sharp bounce V) ----
def _smooth_xy(xy, window=3):
    """Moving-average smoothing; first and last points are kept exact so the
    trail starts at the release and ends exactly on the ball / bounce."""
    n = len(xy)
    if n <= 2:
        return list(xy)
    out = [xy[0]]
    half = window // 2
    for i in range(1, n - 1):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = xy[lo:hi]
        out.append((sum(p[0] for p in chunk) / len(chunk),
                    sum(p[1] for p in chunk) / len(chunk)))
    out.append(xy[-1])
    return out


def _cumulative_lengths(curve):
    cum = [0.0]
    for i in range(1, len(curve)):
        x0, y0 = curve[i - 1]
        x1, y1 = curve[i]
        cum.append(cum[-1] + ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    return cum


def _smooth_curve(xy):
    """Smoothed Catmull-Rom curve through the COMPLETE point list of one
    segment (see build_trail_geometry)."""
    return _catmull_rom(_smooth_xy(xy), samples_per_segment=TRAIL_SAMPLES_PER_SEG)


def _taper_widths(curve, w_start, w_end):
    """One width per curve sample, tapering from w_start to w_end over the
    curve's actual arc length (eased by TRAIL_TAPER_POWER so the tube stays
    thick near the start, like the reference)."""
    cum = _cumulative_lengths(curve)
    total = cum[-1] or 1.0
    return [w_start + (w_end - w_start) * ((c / total) ** TRAIL_TAPER_POWER) for c in cum]


def _curve_with_widths(xy, w_start, w_end):
    """Curve + tapered widths for one segment (kept for compatibility)."""
    curve = _smooth_curve(xy)
    return curve, _taper_widths(curve, w_start, w_end)


def reveal_curve(curve, widths, n_total_pts, n_revealed_pts):
    """Truncates a full precomputed curve to the prefix produced by the
    first n_revealed_pts of n_total_pts raw points."""
    if n_revealed_pts <= 0 or not curve:
        return [], []
    if n_revealed_pts >= n_total_pts:
        return curve, widths
    cutoff = min(len(curve), (n_revealed_pts - 1) * TRAIL_SAMPLES_PER_SEG + 1)
    return curve[:cutoff], widths[:cutoff]


def _draw_tube(draw, curve, widths, scale, ox, oy, layers, round_start=True):
    """Shaded variable-width tube. `layers` is ((colour, width_frac), ...)
    ordered dark -> light. Every layer is a full tube along the same curve
    but narrower and shifted toward the lit side (TRAIL_LIGHT), so the tube
    reads as a rounded 3-D noodle: dark underside, light top edge. A circle
    at every sample plus a quad between neighbours keeps it gap-free even
    where it is thin; the circles also give the rounded end caps."""
    n = len(curve)
    if n == 0:
        return
    lx, ly = TRAIL_LIGHT
    ll = (lx * lx + ly * ly) ** 0.5
    lx, ly = lx / ll, ly / ll

    # per-sample unit normal (central difference of the curve)
    normals = []
    for i in range(n):
        a = curve[max(0, i - 1)]
        b = curve[min(n - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        ln = (tx * tx + ty * ty) ** 0.5 or 1.0
        normals.append((-ty / ln, tx / ln))

    for color, frac in layers:
        fill = color + (TRAIL_ALPHA,)
        pts, rad = [], []
        for i in range(n):
            w = widths[i]
            nx, ny = normals[i]
            # offset toward the light: n * (n . L) keeps the shift continuous
            # (it fades to 0 where the tube runs parallel to the light) and
            # does not depend on which way the normal happens to point
            k = nx * lx + ny * ly
            off = (1.0 - frac) * w / 2.0 * k
            pts.append(((curve[i][0] + nx * off - ox) * scale,
                        (curve[i][1] + ny * off - oy) * scale))
            rad.append(max(0.5, w * frac * scale / 2.0))
        for i in range(n):
            x, y = pts[i]
            r = rad[i]
            if i > 0 or round_start:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
            if i == 0:
                continue
            x0, y0 = pts[i - 1]
            r0 = rad[i - 1]
            dx, dy = x - x0, y - y0
            ln = (dx * dx + dy * dy) ** 0.5
            if ln < 1e-6:
                continue
            px, py = -dy / ln, dx / ln
            draw.polygon([(x0 + px * r0, y0 + py * r0), (x + px * r, y + py * r),
                          (x - px * r, y - py * r), (x0 - px * r0, y0 - py * r0)],
                         fill=fill)


def build_trail_geometry(points, vertex_frame, frame_wh):
    """Call ONCE per delivery, with every ball-tracker point the trail will
    ever have (release through the last tracked frame). Splits it into a
    pre-bounce and post-bounce segment at vertex_frame (if any) and builds
    one fixed Catmull-Rom tube (curve + arc-length-tapered widths) for
    each. The geometry never changes once built -- per-frame rendering only
    reveals a growing PREFIX of it (see reveal_curve()).

    Widths come from the pre-bounce arc length (TRAIL_*_PER_LEN), so the
    tube has the reference's proportions at any scale: thick round release
    end -> thinner at the bounce -> a slender, thinner arm after it.

    points: [(frame, x, y), ...] sorted by frame, for the WHOLE delivery.
    vertex_frame: frame of the pitching (bounce) point, or None."""
    W, _H = frame_wh

    if vertex_frame is not None and points[-1][0] > vertex_frame:
        pre = [(x, y) for f, x, y in points if f <= vertex_frame]
        post = [(x, y) for f, x, y in points if f >= vertex_frame]
    else:
        pre = [(x, y) for _f, x, y in points]
        post = []

    pre_curve = _smooth_curve(pre) if len(pre) >= 2 else []
    post_curve = _smooth_curve(post) if len(post) >= 2 else []

    ref_len = (_cumulative_lengths(pre_curve)[-1] if pre_curve else 0.0) or 1.0
    w_start = TRAIL_START_PER_LEN * ref_len
    clamped = min(max(w_start, TRAIL_START_MIN_FRAC * W), TRAIL_START_MAX_FRAC * W)
    k = clamped / w_start                        # same factor for every width
    w_start = max(2.2, clamped)
    w_pitch = max(1.4, TRAIL_PITCH_PER_LEN * ref_len * k)
    w_post0 = max(1.2, TRAIL_POST_START_PER_LEN * ref_len * k)
    w_post1 = max(1.0, TRAIL_POST_END_PER_LEN * ref_len * k)

    return {
        'pre_curve': pre_curve,
        'pre_widths': _taper_widths(pre_curve, w_start, w_pitch) if pre_curve else [],
        'n_pre_pts': len(pre),
        'post_curve': post_curve,
        'post_widths': _taper_widths(post_curve, w_post0, w_post1) if post_curve else [],
        'n_post_pts': len(post),
        'w_start': w_start,
    }


def render_trail_png(geom, n_revealed, frame_wh, out_path):
    """Draws the trail up to the first n_revealed of the ORIGINAL combined
    (pre+post, bounce point shared by both) raw ball-tracker points, using
    the fixed geometry from build_trail_geometry(). Reference style: a blue
    shaded tube, thick and round-ended at release, tapering toward the
    bounce; a sharp V at the bounce; then a thinner, lighter-blue segment
    to the ball."""
    W, H = frame_wh
    n_pre = geom['n_pre_pts']
    reveal_pre = min(n_revealed, n_pre)
    reveal_post = (max(0, n_revealed - n_pre + 1) if geom['post_curve'] else 0)

    pre_curve, pre_widths = reveal_curve(geom['pre_curve'], geom['pre_widths'],
                                         n_pre, reveal_pre)
    post_curve, post_widths = reveal_curve(geom['post_curve'], geom['post_widths'],
                                           geom['n_post_pts'], reveal_post)

    segs = []
    if len(pre_curve) >= 2:
        segs.append((pre_curve, pre_widths, TRAIL_LAYERS_PRE))
    if len(post_curve) >= 2:
        segs.append((post_curve, post_widths, TRAIL_LAYERS_POST))

    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    if segs:
        allx = [p[0] for c, _w, _l in segs for p in c]
        ally = [p[1] for c, _w, _l in segs for p in c]
        pad = geom['w_start'] + 4
        x0, y0 = max(0, int(min(allx) - pad)), max(0, int(min(ally) - pad))
        x1, y1 = min(W, int(max(allx) + pad)), min(H, int(max(ally) + pad))
        bw, bh = x1 - x0, y1 - y0
        S = TRAIL_SS
        big = Image.new('RGBA', (bw * S, bh * S), (0, 0, 0, 0))
        draw = ImageDraw.Draw(big)
        for curve, widths, layers in segs:
            _draw_tube(draw, curve, widths, S, x0, y0, layers, round_start=True)
        small = big.resize((bw, bh), Image.LANCZOS)
        layer.paste(small, (x0, y0))
    layer.save(out_path)


def event_line(label, time_s, frame_no):
    text = '%-9s t=%.3fs   frame %d' % (label.upper(), time_s, frame_no)
    return (text, EVENT_COLOR[label])


def speed_line(speed):
    text = ('SPEED     ~%.1f km/h (%.1f mph)'
            % (speed['speed_kmh'], speed['speed_mph']))
    return (text, SPEED_COLOR)


def make_event_panel_png(lines, out_path, width=520):
    height = PANEL_MARGIN * 2 + PANEL_LINE_H * max(1, len(lines))
    img = Image.new('RGBA', (width, height), (12, 12, 12, 200))
    draw = ImageDraw.Draw(img)
    font = find_font(PANEL_FONT_SIZE)
    y = PANEL_MARGIN
    for text, color in lines:
        draw.text((PANEL_MARGIN, y), text, fill=color, font=font)
        y += PANEL_LINE_H
    img.save(out_path)
    return height


# -------------------------------------------------------
# Speed
# -------------------------------------------------------

def load_pitch_distances():
    if not os.path.isfile(PITCH_DISTANCES_FILE):
        return {}
    try:
        with open(PITCH_DISTANCES_FILE) as fp:
            data = json.load(fp)
        return {k: float(v) for k, v in data.items()}
    except (ValueError, TypeError, OSError) as exc:
        print('[app1] WARNING: could not read %s (%s); ignoring it'
              % (PITCH_DISTANCES_FILE, exc))
        return {}


def resolve_pitch_distance(video_path, cli_value, distances_map):
    if cli_value is not None:
        return cli_value, 'command line'
    name = os.path.basename(video_path)
    if name in distances_map:
        return distances_map[name], PITCH_DISTANCES_FILE
    stem = os.path.splitext(name)[0]
    if stem in distances_map:
        return distances_map[stem], PITCH_DISTANCES_FILE
    return DEFAULT_PITCH_DISTANCE_M, 'built-in default (no entry for %s in %s)' % (
        name, PITCH_DISTANCES_FILE)


def compute_speed(release_ev, impact_ev, distance_m):
    if not release_ev or not impact_ev:
        return None
    dt = impact_ev['time'] - release_ev['time']
    if dt <= 0:
        return None
    mps = distance_m / dt
    return {
        'distance_m': distance_m,
        'time_of_flight_s': dt,
        'speed_mps': mps,
        'speed_kmh': mps * 3.6,
        'speed_mph': mps * 2.2369362921,
    }


# -------------------------------------------------------
# Model 1: event timing (cricket_infer.py)
# -------------------------------------------------------

def run_event_inference(video, bundle_dir, device, out_json, extra_args):
    infer_script = os.path.join(bundle_dir, 'cricket_infer.py')
    if not os.path.isfile(infer_script):
        raise SystemExit('cricket_infer.py not found under %s' % bundle_dir)
    cmd = [sys.executable, infer_script, os.path.abspath(video),
           '--out', out_json, '--device', device] + extra_args
    print('[app1] running event model: %s' % ' '.join(cmd))
    subprocess.run(cmd, cwd=bundle_dir, check=True)
    with open(out_json) as fp:
        return json.load(fp)


def probe_video(video):
    out = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'json', video])
    s = json.loads(out)['streams'][0]
    W = int(s['width'])
    H = int(s['height'])
    fr = s.get('r_frame_rate', '30/1')
    try:
        num, den = fr.split('/')
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0
    dur = float(s.get('duration') or 0.0)
    if dur <= 0:
        out2 = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'json', video])
        dur = float(json.loads(out2)['format']['duration'])
    return W, H, fps, dur


# -------------------------------------------------------
# Model 2: ball tracking (cricket_ball_tracker_repaired/track_ball.py)
# -------------------------------------------------------

def run_ball_tracking(video, tracker_dir, device, out_dir,
                      threshold=None, visibility_threshold=None):
    """Runs the trained ball tracker over the whole video and returns
    {frame_no: {x, y, confidence, presence, visible, time_s}}.

    threshold: min per-frame confidence for the tracker to count a ball
    as present at all (track_ball.py's own default: 0.10).
    visibility_threshold: presence-probability cutoff for marking a frame
    "visible" (track_ball.py's own default: 0.5). Both are forwarded
    as-is when given; leave them None to use track_ball.py's defaults."""
    script = os.path.join(tracker_dir, 'track_ball.py')
    if not os.path.isfile(script):
        raise SystemExit('track_ball.py not found under %s' % tracker_dir)
    cmd = [sys.executable, script, '--input', os.path.abspath(video),
           '--out-dir', out_dir, '--device', device]
    if threshold is not None:
        cmd += ['--threshold', str(threshold)]
    if visibility_threshold is not None:
        cmd += ['--visibility-threshold', str(visibility_threshold)]
    print('[app1] running ball tracker: %s' % ' '.join(cmd))
    subprocess.run(cmd, cwd=tracker_dir, check=True)

    stem = os.path.splitext(os.path.basename(video))[0]
    json_path = os.path.join(out_dir, stem + '.json')
    with open(json_path) as fp:
        records = json.load(fp)
    return {r['frame']: r for r in records}


def find_ball_near(ball_by_frame, target_frame, search):
    """Nearest visible ball detection to target_frame, searching outward
    frame by frame up to `search` frames either side. Returns (x, y,
    frame_used) or None."""
    rec = ball_by_frame.get(target_frame)
    if rec and rec.get('visible'):
        return rec['x'], rec['y'], target_frame
    for delta in range(1, search + 1):
        for f in (target_frame - delta, target_frame + delta):
            rec = ball_by_frame.get(f)
            if rec and rec.get('visible'):
                return rec['x'], rec['y'], f
    return None


# -------------------------------------------------------
# Timestamp reporting (same shape as app.py)
# -------------------------------------------------------

def timestamps_summary(deliveries, pitch_distance_m):
    lines = []
    for d in deliveries:
        lines.append('Delivery %d (score=%.3f, complete=%s)'
                      % (d['index'], d.get('score', 0), d.get('complete')))
        for label in EVENT_ORDER:
            ev = d.get(label)
            if ev:
                lines.append(
                    '    %-9s t=%7.3fs   frame %-5d src_frame %-5d confidence %.3f'
                    % (label.upper() + ':', ev['time'], ev['frame'],
                       ev.get('src_frame', ev['frame']), ev.get('score', 0)))
            else:
                lines.append('    %-9s -- not detected' % (label.upper() + ':'))
        speed = compute_speed(d.get('release'), d.get('impact'), pitch_distance_m)
        if speed:
            lines.append(
                '    SPEED:    ~%.1f km/h (%.1f mph) over %.2fs, assuming %.2fm '
                '(release->impact, average -- not a radar reading)'
                % (speed['speed_kmh'], speed['speed_mph'],
                   speed['time_of_flight_s'], speed['distance_m']))
    return '\n'.join(lines)


def timestamps_json(deliveries, pitch_distance_m, ball_positions_by_delivery):
    out = []
    for d in deliveries:
        entry = {
            'delivery': d['index'],
            'score': d.get('score', 0),
            'complete': d.get('complete'),
        }
        for label in EVENT_ORDER:
            ev = d.get(label)
            entry[label] = None if not ev else {
                'time_seconds': ev['time'],
                'frame': ev['frame'],
                'src_frame': ev.get('src_frame', ev['frame']),
                'confidence': ev.get('score', 0),
                'ball_xy': ball_positions_by_delivery.get(d['index'], {}).get(label),
            }
        entry['speed'] = compute_speed(d.get('release'), d.get('impact'),
                                       pitch_distance_m)
        out.append(entry)
    return out


# -------------------------------------------------------
# Filtergraph
# -------------------------------------------------------

def enable_expr(windows):
    return '+'.join('between(t,%.3f,%.3f)' % (s, e) for s, e in windows)


def build_filter(overlay_layers):
    parts = []
    cur = '[0:v]'
    for tag, idx, windows, position_expr in overlay_layers:
        expr = enable_expr(windows)
        nxt = '[o_%s]' % tag
        parts.append(
            "%s[%d:v]overlay=%s:enable='%s'%s"
            % (cur, idx, position_expr, expr, nxt))
        cur = nxt
    return ';'.join(parts), cur


# -------------------------------------------------------
# main
# -------------------------------------------------------

def process_one(video, out_video, args, ball_confidence, ball_visibility):
    """Runs the full two-model pipeline once for a single
    (video, output path, ball-tracker confidence) combination."""
    if not os.path.isfile(video):
        print('[app1] SKIPPING: no such video: %s' % video)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)

    frame_w, frame_h, src_fps, dur_s = probe_video(video)
    print('[app1] === %s (ball-confidence=%s) -> %s ==='
          % (os.path.basename(video), ball_confidence, out_video))
    print('[app1] source: %dx%d @ %.3f fps, %.2fs' % (frame_w, frame_h, src_fps, dur_s))

    pitch_distance_m, distance_src = resolve_pitch_distance(
        video, args.pitch_distance_m, load_pitch_distances())
    print('[app1] pitch distance: %.2fm (source: %s)' % (pitch_distance_m, distance_src))

    with tempfile.TemporaryDirectory(prefix='app1_') as tmp:
        # --- Model 1: event timing ---
        json_path = args.json_out or os.path.join(tmp, 'events.json')
        extra = []
        for a in args.infer_arg:
            if a.startswith('--') and '=' in a:
                extra.extend(a.split('=', 1))
            else:
                extra.append(a)
        result = run_event_inference(video, args.bundle_dir, args.device,
                                     json_path, extra)

        all_deliveries = result.get('deliveries', [])
        deliveries = [d for d in all_deliveries
                      if d.get('score', 0) >= args.min_delivery_score]
        dropped = len(all_deliveries) - len(deliveries)
        print('[app1] %d deliveries decoded, %d kept (score >= %.2f)%s'
              % (len(all_deliveries), len(deliveries), args.min_delivery_score,
                 '' if not dropped else ' -- %d dropped as low-confidence noise' % dropped))

        # --- Model 2: ball tracking (whole clip, once) ---
        ball_dir = os.path.join(tmp, 'ball')
        ball_by_frame = run_ball_tracking(video, args.ball_tracker_dir,
                                          args.ball_device, ball_dir,
                                          threshold=ball_confidence,
                                          visibility_threshold=ball_visibility)
        n_visible = sum(1 for r in ball_by_frame.values() if r.get('visible'))
        print('[app1] ball tracker: %d/%d frames with a visible ball'
              % (n_visible, len(ball_by_frame)))

        # keep the per-frame ball positions next to the output (ball3d.py reads this)
        ball_json_path = os.path.splitext(out_video)[0] + '_ball.json'
        with open(ball_json_path, 'w') as fp:
            json.dump([ball_by_frame[f] for f in sorted(ball_by_frame)], fp)
        print('[app1] wrote %s' % ball_json_path)

        # --- Resolve each event's pixel position from the ball tracker ---
        ball_positions_by_delivery = {}
        for d in deliveries:
            per_label = {}
            for label in EVENT_ORDER:
                ev = d.get(label)
                if not ev:
                    continue
                target = ev.get('src_frame', ev['frame'])
                hit = find_ball_near(ball_by_frame, target, args.ball_search_frames)
                if hit is None:
                    print('[app1]   delivery %d %-8s frame %d: no ball detection '
                          'within %d frames' % (d['index'], label, target,
                                                 args.ball_search_frames))
                    continue
                x, y, used_frame = hit
                per_label[label] = (x, y)
                note = '' if used_frame == target else ' (nearest at frame %d)' % used_frame
                print('[app1]   delivery %d %-8s frame %d -> ball at (%.1f, %.1f)%s'
                      % (d['index'], label, target, x, y, note))
            ball_positions_by_delivery[d['index']] = per_label

        print('\n[app1] TIMESTAMPS:')
        print(timestamps_summary(deliveries, pitch_distance_m))
        print()

        timestamps_path = os.path.splitext(out_video)[0] + '_timestamps.json'
        with open(timestamps_path, 'w') as fp:
            json.dump(timestamps_json(deliveries, pitch_distance_m,
                                      ball_positions_by_delivery), fp, indent=2)
        print('[app1] wrote %s' % timestamps_path)

        # --- Build overlays: trail, per-event ball ring, corner panel ---
        extra_ffmpeg_inputs = []
        overlay_layers = []
        input_idx = 1

        if not args.no_trail:
            with_release = sorted((d for d in deliveries if d.get('release')),
                                  key=lambda d: d['release']['time'])
            for k, d in enumerate(with_release):
                release_ev = d['release']
                end_ev = d.get('impact') or d.get('pitching')
                if not end_ev:
                    continue
                f0 = release_ev.get('src_frame', release_ev['frame'])
                f1 = end_ev.get('src_frame', end_ev['frame'])
                pts = sorted(
                    (f, r['x'], r['y'], r.get('time_s', r['frame'] / src_fps))
                    for f, r in ball_by_frame.items()
                    if f0 <= f <= f1 and r.get('visible'))
                if len(pts) < 2:
                    continue

                # Bounce point = the tracked ball position closest to the
                # pitching frame. The trail turns a sharp V here instead of
                # being smoothed through it.
                vertex_f = None
                pitch_ev = d.get('pitching')
                if pitch_ev:
                    pitch_f = pitch_ev.get('src_frame', pitch_ev['frame'])
                    vertex_f = min((p[0] for p in pts),
                                   key=lambda f: abs(f - pitch_f))

                # Build the tube geometry ONCE from the complete trajectory,
                # then just reveal a growing prefix of it frame by frame --
                # the already-drawn part never re-tapers or wobbles.
                full_points = [(f, x, y) for f, x, y, _t in pts]
                geom = build_trail_geometry(full_points, vertex_f, (frame_w, frame_h))

                # Trail grows one detected point at a time, in step with the
                # ball, then stays up until the next delivery (or end of clip).
                trail_end = (with_release[k + 1]['release']['time']
                             if k + 1 < len(with_release) else dur_s)
                for i in range(1, len(pts)):
                    start = pts[i][3]
                    end = pts[i + 1][3] if i + 1 < len(pts) else trail_end
                    png_path = os.path.join(tmp, 'trail_%d_%d.png' % (d['index'], i))
                    render_trail_png(geom, i + 1, (frame_w, frame_h), png_path)
                    extra_ffmpeg_inputs += ['-i', png_path]
                    overlay_layers.append(
                        ('trail_%d_%d' % (d['index'], i), input_idx,
                         [(start, end)], 'x=0:y=0'))
                    input_idx += 1

        for d in deliveries:
            events_sorted = sorted(
                ((label, d[label]) for label in EVENT_ORDER if d.get(label)),
                key=lambda pair: pair[1]['time'])
            per_label_xy = ball_positions_by_delivery.get(d['index'], {})
            for i, (label, ev) in enumerate(events_sorted):
                xy = per_label_xy.get(label)
                if xy is None:
                    continue
                x, y = xy
                next_t = events_sorted[i + 1][1]['time'] if i + 1 < len(events_sorted) else None
                end = ev['time'] + MARKER_SECONDS
                if next_t is not None:
                    end = min(end, next_t)
                png_path = os.path.join(tmp, 'mark_%d_%s.png' % (d['index'], label))
                anchor_x, anchor_y = make_ball_marker_png(label, png_path)
                extra_ffmpeg_inputs += ['-i', png_path]
                overlay_layers.append(
                    ('mark_%d_%s' % (d['index'], label), input_idx,
                     [(ev['time'], end)],
                     'x=%.1f:y=%.1f' % (x - anchor_x, y - anchor_y)))
                input_idx += 1

        for d in deliveries:
            events_sorted = sorted(
                ((label, d[label]) for label in EVENT_ORDER if d.get(label)),
                key=lambda pair: pair[1]['time'])
            release_ev = d.get('release')
            cumulative = []
            for i, (label, ev) in enumerate(events_sorted):
                cumulative.append(event_line(label, ev['time'],
                                             ev.get('src_frame', ev['frame'])))
                if label == 'impact':
                    speed = compute_speed(release_ev, ev, pitch_distance_m)
                    if speed:
                        cumulative.append(speed_line(speed))
                start = ev['time']
                end = events_sorted[i + 1][1]['time'] if i + 1 < len(events_sorted) else dur_s
                png_path = os.path.join(tmp, 'panel_%d_%d.png' % (d['index'], i))
                panel_h = make_event_panel_png(cumulative, png_path)
                extra_ffmpeg_inputs += ['-i', png_path]
                overlay_layers.append(
                    ('panel_%d_%d' % (d['index'], i), input_idx,
                     [(start, end)], 'x=20:y=H-%d-20' % panel_h))
                input_idx += 1

        if not overlay_layers:
            print('[app1] no events to show; copying video through unchanged')
            subprocess.run(['ffmpeg', '-y', '-i', video,
                             '-c', 'copy', out_video], check=True)
        else:
            filter_complex, final_label = build_filter(overlay_layers)
            cmd = ['ffmpeg', '-y', '-i', video] + extra_ffmpeg_inputs + [
                '-filter_complex', filter_complex,
                '-map', final_label, '-map', '0:a?',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'copy',
                out_video,
            ]
            print('[app1] running ffmpeg with %d overlay inputs' % len(overlay_layers))
            subprocess.run(cmd, check=True)

        print('[app1] wrote %s' % out_video)

    if not args.no_play and sys.platform == 'darwin' and _which('open'):
        print('[app1] opening %s' % out_video)
        subprocess.run(['open', out_video])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video', nargs='?', default=None,
                     help='input video (default: INPUT_VIDEO_NAME / BATCH_VIDEOS '
                          'set at the top of app1.py, or the newest clip in %s)'
                          % VIDEOS_DIR)
    ap.add_argument('out_video', nargs='?', default=None,
                     help='annotated output video path (default: auto, under %s; '
                          'ignored in batch mode)' % OUTPUT_DIR)
    ap.add_argument('--bundle-dir', default=DEFAULT_BUNDLE_DIR,
                     help='path to the delivery_bundle folder (default: %(default)s)')
    ap.add_argument('--ball-tracker-dir', default=BALL_TRACKER_DIR,
                     help='path to cricket_ball_tracker_repaired (default: %(default)s)')
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'),
                     help='device for the event-timing model')
    ap.add_argument('--ball-device', default='auto',
                     help='device for the ball tracker: auto, cpu, cuda, or cuda:N')
    ap.add_argument('--ball-confidence', type=float, default=None,
                     help='min per-frame confidence for the ball tracker to count '
                          'a detection at all (track_ball.py --threshold, default 0.10). '
                          'Used only outside batch mode -- set BATCH_BALL_CONFIDENCES '
                          'at the top of the file to sweep several values at once. '
                          'Raise this to drop more false-positive "ball" detections; '
                          'lower it to recover more real-but-faint ones.')
    ap.add_argument('--ball-visibility', type=float, default=None,
                     help='presence-probability cutoff for marking a frame "visible" '
                          '(track_ball.py --visibility-threshold, default 0.5)')
    ap.add_argument('--json-out', default=None,
                     help='also save the raw cricket_infer.py JSON here')
    ap.add_argument('--infer-arg', action='append', default=[],
                     help='extra arg to pass through to cricket_infer.py (repeatable)')
    ap.add_argument('--min-delivery-score', type=float, default=0.3,
                     help='drop decoded deliveries below this mean class score '
                          '(default: %(default)s)')
    ap.add_argument('--ball-search-frames', type=int, default=BALL_SEARCH_FRAMES,
                     help='if the ball tracker has no detection on the exact '
                          "event frame, search this many frames either side "
                          '(default: %(default)s)')
    ap.add_argument('--pitch-distance-m', type=float, default=None,
                     help='distance between the two popping creases, in meters. '
                          'Overrides any entry in %s (default: filename lookup, '
                          'falling back to %.2f)' % (PITCH_DISTANCES_FILE,
                                                     DEFAULT_PITCH_DISTANCE_M))
    ap.add_argument('--no-trail', action='store_true',
                     help="don't draw the continuous ball trail")
    ap.add_argument('--no-play', action='store_true',
                     help="don't auto-open the finished video for playback")
    args = ap.parse_args()

    if not os.path.isdir(args.ball_tracker_dir):
        raise SystemExit('cricket_ball_tracker_repaired not found at %s'
                          % args.ball_tracker_dir)
    for tool in ('ffmpeg', 'ffprobe'):
        if not _which(tool):
            raise SystemExit('%s is not on PATH' % tool)

    explicit_video = args.video is not None
    batch_mode = not explicit_video and (BATCH_VIDEOS or BATCH_BALL_CONFIDENCES)

    if not batch_mode:
        # --- normal single-run path (unchanged CLI behavior) ---
        video = args.video or DEFAULT_INPUT_VIDEO
        if not video:
            if INPUT_VIDEO_NAME:
                raise SystemExit(
                    'INPUT_VIDEO_NAME=%r (set at the top of app1.py) was not '
                    'found in %s, %s, or %s'
                    % (INPUT_VIDEO_NAME, VIDEOS_DIR, OUTPUT_DIR, DEFAULT_BUNDLE_DIR))
            raise SystemExit('no video given, and no clip found under %s or %s'
                              % (VIDEOS_DIR, DEFAULT_BUNDLE_DIR))
        if not os.path.isfile(video):
            raise SystemExit('no such video: %s' % video)
        out_video = args.out_video or default_output_for(video)
        process_one(video, out_video, args, args.ball_confidence, args.ball_visibility)
        return

    # --- batch sweep: every (video, confidence) combination, no CLI args needed ---
    names = BATCH_VIDEOS or [INPUT_VIDEO_NAME]
    videos = []
    for name in names:
        resolved = _resolve_named_video(name)
        if resolved is None:
            print('[app1] SKIPPING %r: not found in %s, %s, or %s'
                  % (name, VIDEOS_DIR, OUTPUT_DIR, DEFAULT_BUNDLE_DIR))
            continue
        videos.append(resolved)
    if not videos:
        raise SystemExit('none of BATCH_VIDEOS=%r were found' % (names,))

    confidences = BATCH_BALL_CONFIDENCES or [args.ball_confidence]
    combos = [(v, c) for v in videos for c in confidences]
    print('[app1] batch sweep: %d video(s) x %d confidence(s) = %d run(s)'
          % (len(videos), len(confidences), len(combos)))

    for video, ball_confidence in combos:
        out_video = batch_output_for(video, ball_confidence)
        process_one(video, out_video, args, ball_confidence, args.ball_visibility)


def _which(name):
    from shutil import which
    return which(name)


if __name__ == '__main__':
    main()
