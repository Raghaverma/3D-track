#!/usr/bin/env python3
"""
ball3d_manual.py -- same as ball3d.py, but the camera is calibrated by CLICKING every stump
base and top yourself (no colour detection).

Everything after calibration (fit / overlay / viewer / plot / demo) is ball3d.py unchanged, so the
calib file written here works with either script.

Calibrate (12 clicks: both wickets, bases then tops, each left -> middle -> right ON SCREEN):
  python ball3d_manual.py calibrate 83_kmph.mp4 --focal-35mm 26 --out calib_83.json

  The NEAR wicket is clicked first, then the FAR one. A magnifier (bottom-right of the window)
  follows the cursor so you can place the tiny far wicket precisely.
  keys:  s = skip this point   u = undo   ENTER (or q) = finish

Then, as before:
  python ball3d_manual.py fit --calib calib_83.json --ball out/83_kmph_tracked_ball.json \
        --events out/83_kmph_tracked_timestamps.json --video 83_kmph.mp4 --out track3d_83.json
  python ball3d_manual.py viewer --track track3d_83.json --out viewer_83.html

Options for calibrate: --frame N, --fx / --hfov-deg / --focal-35mm, --refine-focal,
--camera-at bowler|batter (which end the camera stands behind; default bowler),
--base-height-m (stumps standing on a plastic base: height of the point where the stumps enter the
base, applied to the near wicket; --far-base-height-m for the far one, default = same).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ball3d as b3                                             # noqa: E402

LOUPE_SRC = 24          # half-size (px of the real image) of the magnified area
LOUPE_ZOOM = 8


def click_plan(camera_at):
    """[(point name, prompt)] -- near wicket first. Sides are as seen ON SCREEN (left stump = leftmost)."""
    near, far = ("BOWL", "BAT") if camera_at == "bowler" else ("BAT", "BOWL")
    # Looking down the pitch from the bowler's end, screen-left = +Y = 'left'. From the batter's end it flips.
    side = {"leftmost": "left", "middle": "mid", "rightmost": "right"}
    if camera_at != "bowler":
        side = {"leftmost": "right", "middle": "mid", "rightmost": "left"}
    plan = []
    for end, label in ((near, "NEAR"), (far, "FAR")):
        for part in ("base", "top"):
            for scr in ("leftmost", "middle", "rightmost"):
                what = ("where the stump meets the ground/base" if part == "base" else "very TOP of the stump")
                plan.append(("%s_%s_%s" % (end, side[scr], part), "%s wicket, %s stump: %s" % (label, scr, what)))
    return plan


def calibrate_manual(video, frame_idx, fx, size, out_path, camera_at="bowler", refine_focal=False,
                     base_h=0.0, far_base_h=None):
    img = b3.read_frame(video, frame_idx)
    H, W = img.shape[:2]
    s = min(1.0, 1000.0 / max(W, H))
    base_img = cv2.resize(img, (int(W * s), int(H * s)))
    plan = click_plan(camera_at)
    names = [n for n, _ in plan]
    clicks, k, mouse = {}, [0], [0, 0]
    win = "calibrate by hand (click each point; s = skip, u = undo, ENTER = finish)"

    def loupe(disp):
        """magnified view of the real image around the cursor, bottom-right corner"""
        cx, cy = int(mouse[0] / s), int(mouse[1] / s)
        x0, y0 = max(0, min(W - 2 * LOUPE_SRC, cx - LOUPE_SRC)), max(0, min(H - 2 * LOUPE_SRC, cy - LOUPE_SRC))
        crop = img[y0:y0 + 2 * LOUPE_SRC, x0:x0 + 2 * LOUPE_SRC]
        if crop.size == 0:
            return
        z = cv2.resize(crop, None, fx=LOUPE_ZOOM, fy=LOUPE_ZOOM, interpolation=cv2.INTER_NEAREST)
        px, py = int((cx - x0 + 0.5) * LOUPE_ZOOM), int((cy - y0 + 0.5) * LOUPE_ZOOM)
        cv2.line(z, (px, 0), (px, z.shape[0]), (0, 0, 255), 1)
        cv2.line(z, (0, py), (z.shape[1], py), (0, 0, 255), 1)
        h, w = z.shape[:2]
        if h < disp.shape[0] and w < disp.shape[1]:
            disp[disp.shape[0] - h:, disp.shape[1] - w:] = z
            cv2.rectangle(disp, (disp.shape[1] - w, disp.shape[0] - h), (disp.shape[1] - 1, disp.shape[0] - 1), (255, 255, 255), 1)

    def render():
        disp = base_img.copy()
        for n, (u, v) in clicks.items():
            cv2.circle(disp, (int(u * s), int(v * s)), 3, (0, 255, 0), -1)
            cv2.putText(disp, n.replace("_", " "), (int(u * s) + 5, int(v * s) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        for end in ("BOWL", "BAT"):                                   # live wicket bounding boxes
            pts = [uv for n, uv in clicks.items() if n.startswith(end + "_")]
            if len(pts) >= 2:
                a = np.array(pts) * s
                cv2.rectangle(disp, tuple(a.min(0).astype(int)), tuple(a.max(0).astype(int)), (255, 0, 255), 1)
        msg = "done - press ENTER to finish" if k[0] >= len(names) else "[%d/%d] %s" % (k[0] + 1, len(names), plan[k[0]][1])
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(disp, msg, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        loupe(disp)
        cv2.imshow(win, disp)

    def on_mouse(ev, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if ev == cv2.EVENT_LBUTTONDOWN and k[0] < len(names):
            n = names[k[0]]
            clicks[n] = (x / s, y / s)
            print("[click %d/%d] %-18s pixel (%7.1f, %7.1f)  ->  world (%.3f, %+.3f, %.3f)" % (
                k[0] + 1, len(names), n, x / s, y / s, *b3.world_xyz(n)))
            k[0] += 1
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            render()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    render()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, ord("q")):
            break
        if key == ord("s") and k[0] < len(names):
            k[0] += 1; render()
        if key == ord("u") and k[0] > 0:
            k[0] -= 1; clicks.pop(names[k[0]], None); render()
    cv2.destroyAllWindows()

    if len(clicks) < 4:
        raise SystemExit("need at least 4 clicked points, got %d" % len(clicks))

    geom = None
    if base_h or far_base_h:
        near, far = ("BOWL", "BAT") if camera_at == "bowler" else ("BAT", "BOWL")
        fb = base_h if far_base_h is None else far_base_h
        geom = {near: {"bottom_z": float(base_h or 0.0), "top_z": float(base_h or 0.0) + b3.STUMP_H},
                far: {"bottom_z": float(fb or 0.0), "top_z": float(fb or 0.0) + b3.STUMP_H}}
    cam = b3.save_calibration(img, clicks, fx, (W, H), out_path, refine_focal, geom=geom)

    check = out_path.rsplit(".", 1)[0] + "_check.png"
    try:                                                              # show the result for confirmation
        im = cv2.imread(check)
        sc = min(1.0, 1200.0 / max(im.shape[:2]))
        cv2.imshow("check: drawn stumps/pitch must sit on the real ones (any key to close)", cv2.resize(im, None, fx=sc, fy=sc))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("[calib] could not open a window (%s); open %s yourself" % (e, check))
    return cam


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        ap = argparse.ArgumentParser(prog="ball3d_manual.py calibrate", description="calibrate by clicking every stump base and top")
        ap.add_argument("cmd")
        ap.add_argument("video", help="video (or a still image)")
        ap.add_argument("--frame", type=int, default=0)
        ap.add_argument("--fx", type=float); ap.add_argument("--hfov-deg", type=float); ap.add_argument("--focal-35mm", type=float)
        ap.add_argument("--refine-focal", action="store_true", help="also solve the focal length (needs 8+ well-spread points)")
        ap.add_argument("--camera-at", choices=("bowler", "batter"), default="bowler")
        ap.add_argument("--base-height-m", type=float, default=0.0, help="height of the near stumps' bottom point above the ground (plastic base)")
        ap.add_argument("--far-base-height-m", type=float, help="same for the far wicket (default: same as near)")
        ap.add_argument("--out", default="calib.json")
        a = ap.parse_args()
        img = b3.read_frame(a.video, a.frame)
        size = (img.shape[1], img.shape[0])
        fx, how = b3.focal_length_px(a.fx, a.hfov_deg, a.focal_35mm, size)
        print("[calib] focal length %.1f px (%s), frame %dx%d" % (fx, how, *size))
        calibrate_manual(a.video, a.frame, fx, size, a.out, a.camera_at, a.refine_focal, a.base_height_m, a.far_base_height_m)
    else:
        b3.main()                                   # fit / overlay / viewer / plot / demo / ... exactly as ball3d.py


if __name__ == "__main__":
    main()
