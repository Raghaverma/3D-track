#!/usr/bin/env python3
"""
check_overlay.py -- draw the fitted 3D ball track on the video so you can CHECK it.

Put this file next to ball3d.py (it imports it) and run, for example:

    python check_overlay.py --video out/83_kmph_tracked.mp4 --track track3d_83.json \
        --ball out/83_kmph_tracked_ball.json --calib calib_83.json --out 83_check.mp4

What you see on every frame
---------------------------
  * the 3D track projected onto the image (blue tube; thick near the camera, thin far away),
    growing with the ball
  * the pitch and both sets of stumps projected from the CALIBRATION -- they must sit on the real
    pitch/stumps, otherwise the camera is wrong and so is the whole track
  * the ball tracker's 2D detections: GREEN = agrees with the 3D track, RED = far from it (rejected
    as an outlier by the fit); a yellow line joins a detection to where the 3D track says the ball is
  * a ring where the 3D fit puts the ball right now (it should sit on the real ball)
  * release / bounce / impact markers (projected 3D points)
  * a mini map (top view + side view) showing the same track in metres, with the ball moving along it
  * a readout: time, frame, X/Y/Z in metres and speed in km/h

It also prints a short report (pixel error of the track against the detections, the frames where
they disagree most) and saves a contact sheet PNG with the key frames.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ball3d as b3                                             # noqa: E402

BLUE_PRE = (200, 110, 40)        # BGR
BLUE_POST = (235, 160, 110)


def put_text(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def put_text_fit(img, text, x, y, max_w, scale=0.55, color=(255, 255, 255)):
    """Like put_text but shrinks the text until it fits in max_w pixels (narrow phone videos)."""
    while scale > 0.25 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_w:
        scale -= 0.03
    put_text(img, text, (x, y), scale, color)


class MiniMap:
    """Top view + side view of the pitch in metres, drawn with OpenCV (no matplotlib needed)."""

    def __init__(self, track, width_px):
        self.pit = track["pitch"]
        L = self.pit["len"]
        self.pad = 8
        self.x0, self.x1 = -1.8, L + 1.8
        self.s = (width_px - 2 * self.pad) / (self.x1 - self.x0)
        self.Wp = int(width_px)
        self.top_h = int(5.0 * self.s) + 2 * self.pad + 14
        self.side_h = int(2.8 * self.s) + 2 * self.pad + 14
        self.H = self.top_h + self.side_h + 4
        self.T = np.array(track["t"])
        self.P = np.stack([track["x"], track["y"], track["z"]], 1)
        self.ib = track["ib"] if track["ib"] >= 0 else len(self.T) - 1
        self.cam = track["camera"]["center"]
        self.base = self._static()

    def tx(self, X):
        return int(round(self.pad + (X - self.x0) * self.s))

    def ty_top(self, Y):                               # + Y = left from the bowler's end -> up on screen
        return int(round(self.pad + 14 + 2.5 * self.s - Y * self.s))

    def ty_side(self, Z):
        return int(round(self.top_h + 4 + self.pad + 14 + 2.6 * self.s - Z * self.s))

    def _static(self):
        img = np.zeros((self.H, self.Wp, 3), np.uint8)
        img[:] = (25, 25, 25)
        p, L = self.pit, self.pit["len"]
        put_text(img, "top view (X along pitch, bowler at left)", (self.pad, 12), 0.38)
        put_text(img, "side view", (self.pad, self.top_h + 14), 0.38)
        # top view: pitch strip, creases, stumps
        cv2.rectangle(img, (self.tx(0), self.ty_top(p["half_w"])), (self.tx(L), self.ty_top(-p["half_w"])), (95, 140, 170), -1)
        for X in (p["popping"], L - p["popping"], 0.0, L):
            cv2.line(img, (self.tx(X), self.ty_top(p["return_half"])), (self.tx(X), self.ty_top(-p["return_half"])), (235, 235, 235), 1)
        for X in (0.0, L):
            cv2.line(img, (self.tx(X), self.ty_top(p["stump_dy"])), (self.tx(X), self.ty_top(-p["stump_dy"])), (40, 150, 245), 3)
        cv2.rectangle(img, (self.tx(0), self.ty_top(p["half_w"])), (self.tx(L), self.ty_top(-p["half_w"])), (160, 160, 160), 1)
        # side view: ground + stumps
        cv2.line(img, (self.tx(self.x0), self.ty_side(0)), (self.tx(self.x1), self.ty_side(0)), (70, 140, 70), 2)
        for X in (0.0, L):
            cv2.line(img, (self.tx(X), self.ty_side(0)), (self.tx(X), self.ty_side(p["stump_h"])), (40, 150, 245), 3)
        # camera marker
        cx, cy, cz = self.cam
        if self.x0 <= cx <= self.x1:
            cv2.rectangle(img, (self.tx(cx) - 3, self.ty_top(cy) - 3), (self.tx(cx) + 3, self.ty_top(cy) + 3), (255, 255, 255), -1)
            put_text(img, "cam", (self.tx(cx) - 8, self.ty_top(cy) - 6), 0.33)
            cv2.circle(img, (self.tx(cx), self.ty_side(cz)), 3, (255, 255, 255), -1)
        # faint full track
        self._poly(img, self.P, self.ib, faint=True)
        return img

    def _poly(self, img, pts, ib, faint=False, n=None):
        n = len(pts) if n is None else n
        if n < 2:
            return
        f = 0.35 if faint else 1.0
        for view in ("top", "side"):
            xy = [(self.tx(p[0]), self.ty_top(p[1]) if view == "top" else self.ty_side(p[2])) for p in pts[:n]]
            pre, post = xy[:min(n, ib + 1)], xy[ib + 1:n] if n > ib + 1 else []
            c1 = tuple(int(c * f + 25 * (1 - f)) for c in BLUE_PRE)
            c2 = tuple(int(c * f + 25 * (1 - f)) for c in BLUE_POST)
            if len(pre) >= 2:
                cv2.polylines(img, [np.array(pre, np.int32)], False, c1, 2, cv2.LINE_AA)
            if len(post) >= 2:
                cv2.polylines(img, [np.array([xy[ib]] + post, np.int32)], False, c2, 2, cv2.LINE_AA)

    def render(self, now, pos, events):
        img = self.base.copy()
        n = int(np.searchsorted(self.T, now, side="right"))
        self._poly(img, self.P, self.ib, faint=False, n=n)
        for k, col in (("release", (80, 220, 110)), ("bounce", (60, 210, 255)), ("impact", (70, 70, 255))):
            if k in events and now >= events[k][0]:
                p = events[k][1]
                cv2.circle(img, (self.tx(p[0]), self.ty_top(p[1])), 4, col, -1)
                cv2.circle(img, (self.tx(p[0]), self.ty_side(p[2])), 4, col, -1)
        if pos is not None:
            cv2.circle(img, (self.tx(pos[0]), self.ty_top(pos[1])), 4, (0, 0, 255), -1)
            cv2.circle(img, (self.tx(pos[0]), self.ty_side(pos[2])), 4, (0, 0, 255), -1)
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the video to draw on (the original or your app1 *_tracked.mp4)")
    ap.add_argument("--track", required=True, help="track3d.json from `ball3d.py fit`")
    ap.add_argument("--ball", help="ball tracker json (<out>_ball.json) to show/compare the 2D detections")
    ap.add_argument("--calib", help="calib.json, to draw the stumps with the right heights (plastic base etc.)")
    ap.add_argument("--out", default="track_check.mp4")
    ap.add_argument("--inset", default="top-left", choices=("top-left", "top-right", "bottom-left", "bottom-right", "off"))
    ap.add_argument("--no-pitch", action="store_true", help="do not draw the projected pitch / stumps")
    ap.add_argument("--tube-diameter-m", type=float, default=0.085)
    ap.add_argument("--err-thresh", type=float, default=8.0, help="detections farther than this many px from the 3D track are drawn red")
    ap.add_argument("--trim", action="store_true", help="only write the part around the delivery")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds kept before release / after impact with --trim")
    ap.add_argument("--no-sheet", action="store_true", help="do not write the contact-sheet PNG")
    a = ap.parse_args()

    track = json.load(open(a.track))
    cam = b3.Camera.from_dict(track["camera"])
    T = np.array(track["t"])
    P = np.stack([track["x"], track["y"], track["z"]], 1)
    pit = track["pitch"]
    M = track["metrics"]
    ib = track["ib"] if track["ib"] >= 0 else len(T) - 1
    events = {k: (M[k]["t"], M[k]["pos"]) for k in ("release", "bounce", "impact") if k in M}

    geom = None
    if a.calib:
        geom = json.load(open(a.calib)).get("geom")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or track.get("fps") or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = W / float(cam.W)
    if abs(H / float(cam.H) - sc) > 0.02 * sc:
        print("[warn] video is %dx%d but the calibration was for %dx%d (different aspect ratio): the overlay will be misplaced"
              % (W, H, cam.W, cam.H))
    elif abs(sc - 1.0) > 1e-6:
        print("[info] video is %dx%d, calibration was %dx%d: scaling coordinates by %.3f" % (W, H, cam.W, cam.H, sc))

    # ---- 3D track in pixels
    uv, depth = cam.project(P)
    uv = uv * sc
    widths = np.maximum(cam.fx * sc * a.tube_diameter_m / np.maximum(depth, 0.5), 2.0)

    def pos_at(t):
        t = min(max(t, T[0]), T[-1])
        return np.array([np.interp(t, T, P[:, k]) for k in range(3)])

    def px_at(t):
        q, d = cam.project(pos_at(t))
        return q[0] * sc, float(d[0])

    def speed_kmh(t, h=1.0 / 120):
        return float(np.linalg.norm(pos_at(t + h) - pos_at(t - h)) / (2 * h) * 3.6)

    # ---- 2D detections and how well the 3D track explains them
    dets = []
    if a.ball:
        for f, r in sorted(b3.load_ball_track(a.ball).items()):
            if not r.get("visible", True):
                continue
            t = float(r["time_s"]) if r.get("time_s") is not None else f / fps
            if T[0] - 1.5 / fps <= t <= T[-1] + 1.5 / fps:
                u, v = float(r["x"]) * sc, float(r["y"]) * sc
                pu, _ = px_at(t)
                dets.append({"t": t, "frame": f, "u": u, "v": v, "err": math.hypot(u - pu[0], v - pu[1])})
    if dets:
        errs = np.array([d["err"] for d in dets])
        good = errs <= a.err_thresh
        print("\n[check] %d ball detections inside the track's time window" % len(dets))
        print("        median error %.1f px | inliers (<= %.0f px): %d (RMS %.1f px) | far off: %d"
              % (np.median(errs), a.err_thresh, good.sum(), math.sqrt(np.mean(errs[good] ** 2)) if good.any() else float("nan"), (~good).sum()))
        worst = sorted(dets, key=lambda d: -d["err"])[:3]
        print("        frames where the 3D track and the detection disagree most: " +
              ", ".join("frame %d (%.0f px)" % (d["frame"], d["err"]) for d in worst))

    # ---- mini map
    mm = None
    if a.inset != "off":
        mm = MiniMap(track, int(max(260, 0.40 * W)))

    t_lo = (min(T[0], events.get("release", (T[0],))[0]) - a.pad) if a.trim else -1e9
    t_hi = (T[-1] + a.pad) if a.trim else 1e9
    tmp = a.out + ".tmp.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # key frames for the contact sheet
    key_t = []
    if "release" in events:
        key_t = [events["release"][0]]
        if "bounce" in events:
            key_t += [0.5 * (events["release"][0] + events["bounce"][0]), events["bounce"][0]]
            if "impact" in events:
                key_t += [0.5 * (events["bounce"][0] + events["impact"][0]), events["impact"][0]]
        key_t.append(T[-1] + 0.1)
    key_idx = {int(round(t * fps)): None for t in key_t}

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = idx / fps
        if now < t_lo or now > t_hi:
            idx += 1
            continue

        # projected pitch + stumps (calibration check)
        if not a.no_pitch:
            def poly(pts3, color, th=1):
                q, d = cam.project(np.array(pts3, float))
                if (d > 0.1).all():
                    cv2.polylines(frame, [(q * sc).astype(np.int32).reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)
            L = pit["len"]
            for X, end in ((0.0, "BOWL"), (L, "BAT")):
                g = (geom or {}).get(end, {})
                z0, z1 = g.get("bottom_z", 0.0), g.get("top_z", pit["stump_h"])
                for y in (-pit["stump_dy"], 0.0, pit["stump_dy"]):
                    poly([(X, y, z0), (X, y, z1)], (0, 200, 255), 2)
            for X in (pit["popping"], L - pit["popping"], 0.0, L):
                poly([(X, -pit["return_half"], 0), (X, pit["return_half"], 0)], (255, 255, 255), 1)
            for y in (-pit["half_w"], pit["half_w"]):
                poly([(0, y, 0), (L, y, 0)], (0, 255, 0), 1)

        # 3D tube, growing with the ball
        n = int(np.searchsorted(T, now, side="right"))
        segs = []
        if n >= 2:
            n_pre = min(n, ib + 1)
            segs.append((uv[:n_pre], widths[:n_pre], b3.LAYERS_PRE))
            if n > ib + 1:
                segs.append((uv[ib + 1:n], widths[ib + 1:n] * 0.8, b3.LAYERS_POST))
        layer = b3.render_track_layer((W, H), segs)
        if layer is not None:
            al = layer[:, :, 3:4]
            frame = (frame.astype(np.float32) * (1 - al) + layer[:, :, :3][:, :, ::-1] * 255.0 * al).astype(np.uint8)

        # 2D detections so far (green = explained by the 3D track, red = far off)
        for d in dets:
            if d["t"] <= now + 0.5 / fps:
                col = (0, 220, 0) if d["err"] <= a.err_thresh else (0, 0, 255)
                cv2.circle(frame, (int(round(d["u"])), int(round(d["v"]))), 4, col, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(round(d["u"])), int(round(d["v"]))), 5, (0, 0, 0), 1, cv2.LINE_AA)
            if abs(d["t"] - now) <= 0.5 / fps:                     # the detection of THIS frame -> line to the 3D position
                pu, _ = px_at(d["t"])
                cv2.line(frame, (int(round(d["u"])), int(round(d["v"]))), (int(round(pu[0])), int(round(pu[1]))), (0, 255, 255), 1, cv2.LINE_AA)

        # where the 3D fit puts the ball right now (ring of the real ball size)
        cur = None
        if T[0] <= now <= T[-1]:
            cur = pos_at(now)
            q, dpt = px_at(now)
            r = max(5, int(round(cam.fx * sc * b3.BALL_R / max(dpt, 0.5))) + 3)
            cv2.circle(frame, (int(round(q[0])), int(round(q[1]))), r, (255, 0, 255), 2, cv2.LINE_AA)

        # event markers
        for k, col in (("release", (80, 220, 110)), ("bounce", (60, 210, 255)), ("impact", (70, 70, 255))):
            if k in events and now >= events[k][0] - 1e-6:
                q, _ = cam.project(np.array(events[k][1], float))
                x, y = int(round(q[0, 0] * sc)), int(round(q[0, 1] * sc))
                cv2.circle(frame, (x, y), 16, col, 2, cv2.LINE_AA)
                put_text(frame, k, (x + 20, y + 5), 0.5, col)

        # mini map
        if mm is not None:
            im = mm.render(now, cur, events)
            h, w = im.shape[:2]
            ox = 8 if "left" in a.inset else W - w - 8
            oy = 8 if "top" in a.inset else H - h - 8
            roi = frame[oy:oy + h, ox:ox + w]
            if roi.shape[:2] == im.shape[:2]:
                frame[oy:oy + h, ox:ox + w] = cv2.addWeighted(im, 0.85, roi, 0.15, 0)

        # readout
        if cur is not None:
            txt = "t=%.2fs  frame %d | X %.1f  Y %+.2f  Z %.2f m | %.0f km/h" % (now, idx, cur[0], cur[1], cur[2], speed_kmh(now))
        else:
            txt = "t=%.2fs  frame %d" % (now, idx)
        put_text_fit(frame, txt, 10, H - 12, W - 20, 0.55)
        if dets:
            put_text_fit(frame, "green dot = detection explained by 3D track | red = far off | pink ring = 3D position",
                         10, H - 34, W - 20, 0.4, (220, 220, 220))

        wr.write(frame)
        if idx in key_idx:
            key_idx[idx] = frame.copy()
        idx += 1
    cap.release()
    wr.release()

    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", a.video, "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "copy", "-shortest", a.out], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, a.out)
        print("[warn] ffmpeg not found: wrote mp4v (may not play in browsers)")
    print("[check] wrote %s" % a.out)

    if not a.no_sheet:
        tiles = [f for f in key_idx.values() if f is not None]
        if tiles:
            tw = 480
            tiles = [cv2.resize(t, (tw, int(t.shape[0] * tw / t.shape[1]))) for t in tiles]
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
            sheet = a.out.rsplit(".", 1)[0] + "_sheet.png"
            cv2.imwrite(sheet, np.vstack(rows))
            print("[check] key frames (release, bounce, impact ...) -> %s" % sheet)


if __name__ == "__main__":
    main()