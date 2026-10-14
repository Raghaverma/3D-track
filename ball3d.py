#!/usr/bin/env python3
"""
ball3d.py -- reconstruct a cricket ball's 3D track from ONE camera, then show it
(a) as a perspective-correct 3D tube drawn onto the video, and
(b) in an interactive 3D world (three.js viewer, one self-contained .html).

How the depth is recovered
--------------------------
One camera only gives a *ray* per detection, not a 3D point. Two things pin the
depth down:
  1. CALIBRATION  -- the stumps/pitch have standard sizes. Both wickets are found
     automatically by colour (or you can click them); solvePnP then gives
     the camera position + orientation in a pitch-fixed world frame.
  2. PHYSICS      -- the ball flies under gravity + air drag, bounces on the
     ground plane (z = 0) and flies on. We fit that physical trajectory
     (release position/velocity + bounce restitution) so that its projection
     matches your per-frame 2D ball detections.

World frame (metres):  X along the pitch, 0 = bowler's stumps -> 20.12 = batter's
stumps;  Y across the pitch, + = LEFT as seen from behind the bowler's end looking
down the pitch;  Z up, 0 = ground.   (x forward, y left, z up -- right-handed.)

Commands
--------
  calibrate  camera pose -> calib.json. Finds both wickets BY COLOUR automatically (no clicking);
             falls back to a click window only if detection fails, or with --manual
  calibrate-boxes  same, but from two wicket bounding boxes you supply -- no clicking
  calibrate-auto   finds both wickets by itself (colour + shape rules, no model) and calibrates
  calibrate-xywh   same, from the two (x,y,w,h) wicket boxes your own colour detector produced
  fit        2D detections + event times + calib -> track3d.json (3D track, speeds, ...)
  overlay    draw the 3D track on the video (tube thickness follows perspective)
  viewer     write an interactive 3D viewer (.html) -- includes the two light-red popping-crease
             planes with the release / impact / crossing heights marked on them
  plot       side / top view PNG with the numbers
  demo       synthetic end-to-end example (no data needed) -- run this first

Typical flow with your existing pipeline (app1.py now also writes <out>_ball.json):
  python ball3d.py calibrate clip.mp4 --focal-35mm 26 --out calib.json      # no clicking
  python ball3d.py fit --calib calib.json --ball out/clip_ball.json \
        --events out/clip_timestamps.json --video clip.mp4 --out track3d.json
  python ball3d.py overlay --video clip.mp4 --track track3d.json --out clip_3d.mp4
  python ball3d.py viewer  --track track3d.json --out viewer.html

Needs: numpy, scipy, opencv-python, pillow  (matplotlib only for `plot`; ffmpeg optional)
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
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# World model (metres)
# ----------------------------------------------------------------------
G = 9.81
PITCH_LEN = 16.0          # stump line to stump line
STUMP_H = 0.711            # 28 in, height of the stumps above the ground (or above their base)
WICKET_W = 0.2286          # 9 in: width of the wicket measured OUTER edge to OUTER edge
STUMP_RADIUS = 0.01825     # stump diameter ~ 3.65 cm
STUMP_DY = (WICKET_W - 2 * STUMP_RADIUS) / 2.0   # centre-to-centre of adjacent stumps (~9.6 cm)
POPPING = 1.22             # popping crease distance in front of the stumps
RETURN_HALF = 1.31         # half-length of the popping crease marking
PITCH_HALF_W = 1.525       # half the pitch width (3.05 m)
BALL_R = 0.036             # ball radius
DRAG_K = 0.0065            # 1/m  =  0.5 * rho * Cd * A / m for a cricket ball (~35 m/s -> ~8 m/s^2)
DIAG_35MM_MM = math.hypot(36.0, 24.0)


def _wicket(prefix, x):
    d = {}
    for nm, y in (("left", +STUMP_DY), ("mid", 0.0), ("right", -STUMP_DY)):
        d["%s_%s_base" % (prefix, nm)] = (x, y, 0.0)
        d["%s_%s_top" % (prefix, nm)] = (x, y, STUMP_H)
    return d


# Points you can click. left/right = as seen from behind the BOWLER's end looking
# down the pitch (so for a camera behind the bowling stumps, left = image-left).
CALIB_POINTS = {}
CALIB_POINTS.update(_wicket("BOWL", 0.0))
CALIB_POINTS.update(_wicket("BAT", PITCH_LEN))
CALIB_POINTS.update({
    "BOWL_crease_left":  (POPPING, +RETURN_HALF, 0.0),
    "BOWL_crease_right": (POPPING, -RETURN_HALF, 0.0),
    "BAT_crease_left":   (PITCH_LEN - POPPING, +RETURN_HALF, 0.0),
    "BAT_crease_right":  (PITCH_LEN - POPPING, -RETURN_HALF, 0.0),
})
# Corners of a wicket's bounding box. A box around three stumps spans the OUTER stump edges
# (+-WICKET_W/2) and runs from the ground to the top of the stumps. These are
# the points you get "for free" from a stump detector -- no clicking needed.
BOX_HALF_W = WICKET_W / 2.0
for _end, _x in (("BOWL", 0.0), ("BAT", PITCH_LEN)):
    CALIB_POINTS["%s_box_left_base" % _end] = (_x, +BOX_HALF_W, 0.0)
    CALIB_POINTS["%s_box_right_base" % _end] = (_x, -BOX_HALF_W, 0.0)
    CALIB_POINTS["%s_box_left_top" % _end] = (_x, +BOX_HALF_W, STUMP_H)
    CALIB_POINTS["%s_box_right_top" % _end] = (_x, -BOX_HALF_W, STUMP_H)
BOX_NAMES = ["%s_box_%s_%s" % (e, s_, p) for e in ("BOWL", "BAT") for p in ("base", "top") for s_ in ("left", "right")]

CALIB_ORDER = (
    ["BOWL_mid_base", "BOWL_mid_top", "BAT_mid_base", "BAT_mid_top"] +                     # 4 = bare minimum
    ["BOWL_%s_%s" % (s_, p) for p in ("base", "top") for s_ in ("left", "right")] +      # then the outer stumps
    ["BAT_%s_%s" % (s_, p) for p in ("base", "top") for s_ in ("left", "right")] +
    ["BOWL_crease_left", "BOWL_crease_right", "BAT_crease_left", "BAT_crease_right"])    # optional, adds little


def world_xyz(name, geom=None):
    """World coordinates of a named calibration point. `geom` optionally overrides the heights
    per wicket end: {"BOWL": {"top_z": .., "bottom_z": ..}, "BAT": {...}} -- needed when the stumps
    stand on a plastic base (bottom_z = base height) or are not standard height."""
    x, y, z = CALIB_POINTS[name]
    if geom:
        g = geom.get(name.split("_")[0])
        if g:
            if name.endswith("_top"):
                z = g["top_z"]
            elif name.endswith("_base"):
                z = g["bottom_z"]
    return (x, y, z)


def describe_point(name):
    if "_box_" in name:
        end, _, side, part = name.split("_")
        return "%s wicket box, %s %s corner" % ("BOWLER's" if end == "BOWL" else "BATTER's", side, "bottom" if part == "base" else "top")
    if name.endswith("_base") or name.endswith("_top"):
        end, side, part = name.split("_")
        return "%s-end stump, %s one, %s" % (
            "BOWLER's" if end == "BOWL" else "BATTER's", side if side != "mid" else "middle",
            "where it meets the ground" if part == "base" else "very top of the stump")
    end, _, side = name.split("_")
    return "%s popping-crease line, %s end (looking from the bowler's end)" % (
        "bowler's" if end == "BOWL" else "batter's", side)


# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------
def focal_length_px(fx=None, hfov_deg=None, focal_35mm=None, size=(1080, 1920)):
    W, H = size
    if fx:
        return float(fx), "--fx"
    if hfov_deg:
        return (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0), "--hfov-deg"
    if focal_35mm:
        return focal_35mm * math.hypot(W, H) / DIAG_35MM_MM, "--focal-35mm"
    print("[warn] no focal length given; assuming a 26 mm-equivalent phone camera. "
          "Depth accuracy depends on this -- pass --fx / --hfov-deg / --focal-35mm.")
    return 26.0 * math.hypot(W, H) / DIAG_35MM_MM, "ASSUMED 26mm-equivalent"


class Camera:
    """Pinhole camera, square pixels, principal point at the image centre."""

    def __init__(self, fx, size, rvec, tvec):
        self.fx = float(fx)
        self.W, self.H = int(size[0]), int(size[1])
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.rvec = np.asarray(rvec, float).reshape(3)
        self.tvec = np.asarray(tvec, float).reshape(3)
        self.R = cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self):
        return -self.R.T @ self.tvec

    def to_cam(self, P):
        return np.atleast_2d(P) @ self.R.T + self.tvec

    def project(self, P):
        """world (N,3) -> pixel (N,2), depth along the optical axis (N,)"""
        Pc = self.to_cam(P)
        z = Pc[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        uv = np.stack([self.fx * Pc[:, 0] / zs + self.cx, self.fx * Pc[:, 1] / zs + self.cy], 1)
        return uv, z

    def ray(self, u, v):
        d = self.R.T @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fx, 1.0])
        return self.center, d / np.linalg.norm(d)

    def intersect_plane_z(self, u, v, z0):
        c, d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            return None
        s = (z0 - c[2]) / d[2]
        return c + s * d if s > 0 else None

    def to_dict(self):
        return {"fx": self.fx, "size": [self.W, self.H], "rvec": self.rvec.tolist(),
                "tvec": self.tvec.tolist(), "center": self.center.tolist()}

    @staticmethod
    def from_dict(d):
        return Camera(d["fx"], d["size"], d["rvec"], d["tvec"])


def wicket_box_points(bowler_box, batter_box, camera_at="bowler"):
    """Turn two wicket bounding boxes (x1, y1, x2, y2 in pixels, e.g. from your stump detector)
    into named calibration points.

    camera_at = "bowler": the camera is behind the bowler's end looking down the pitch (as in
    your clips): image-left = +Y at BOTH wickets.  camera_at = "batter": the camera is behind the
    batter looking back, so image-left = -Y.  Pass the box of the BOWLER's-end wicket as
    bowler_box and the BATTER's-end one as batter_box, whichever is nearer the camera."""
    flip = (camera_at == "batter")
    pts = {}
    for end, box in (("BOWL", bowler_box), ("BAT", batter_box)):
        if box is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        xl, xr = min(x1, x2), max(x1, x2)
        yt, yb = min(y1, y2), max(y1, y2)
        left, right = ("right", "left") if flip else ("left", "right")   # which world side is image-left
        pts["%s_box_%s_base" % (end, left)] = (xl, yb)
        pts["%s_box_%s_base" % (end, right)] = (xr, yb)
        pts["%s_box_%s_top" % (end, left)] = (xl, yt)
        pts["%s_box_%s_top" % (end, right)] = (xr, yt)
    return pts


def solve_camera(points_uv, fx, size, refine_focal=False, verbose=True, geom=None, sigmas=None):
    """points_uv: {name: (u, v)} using the names in CALIB_POINTS. Returns (Camera, info).
    geom: per-wicket height override (see world_xyz).  sigmas: {name: pixel uncertainty}; when given,
    the pose is refined by a weighted robust least squares so uncertain points count less."""
    names = [n for n in points_uv if n in CALIB_POINTS]
    if len(names) < 4:
        raise SystemExit("need at least 4 calibration points (6+ incl. some tops recommended), got %d" % len(names))
    obj = np.array([world_xyz(n, geom) for n in names], np.float64)
    img = np.array([points_uv[n] for n in names], np.float64)
    W, H = size
    ends = {n.split("_")[0] for n in names if n.startswith(("BOWL", "BAT"))}
    if len(ends) < 2:
        print("[warn] only ONE wicket supplied: the camera's direction along the pitch is then very poorly "
              "determined and the 3D track will be unreliable (tested: ~6 m bounce error). Use both wickets.")
    elif len(names) < 8 and not sigmas:
        print("[warn] only %d calibration points: it works, but 8+ (e.g. both wicket boxes) is noticeably more accurate" % len(names))
    K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]])

    coplanar = np.ptp(obj[:, 2]) < 1e-9
    if coplanar:
        flag = cv2.SOLVEPNP_IPPE
    else:
        flag = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
    if not ok:
        raise SystemExit("solvePnP failed -- check the clicked points")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    rvec, tvec = np.asarray(rvec, float).ravel(), np.asarray(tvec, float).ravel()

    wts = np.array([1.0 / float((sigmas or {}).get(n, 1.0)) for n in names])
    fx_used = float(fx)
    if sigmas or refine_focal:
        free_f = bool(refine_focal and len(names) >= 8)

        def res(p):
            cam_ = Camera(p[6] if free_f else fx, size, p[:3], p[3:6])
            uv_, _ = cam_.project(obj)
            return ((uv_ - img) * wts[:, None]).ravel()
        x0 = np.concatenate([rvec, tvec] + ([[fx]] if free_f else []))
        kw = {}
        if free_f:
            kw["bounds"] = (np.concatenate([np.full(6, -np.inf), [0.7 * fx]]), np.concatenate([np.full(6, np.inf), [1.4 * fx]]))
        sol = least_squares(res, x0, loss="soft_l1", f_scale=2.0, **kw)
        rvec, tvec = sol.x[:3], sol.x[3:6]
        if free_f:
            fx_used = float(sol.x[6])
            if verbose:
                print("[calib] focal length refined: %.1f -> %.1f px (only trust this with many well-spread points)" % (fx, fx_used))

    cam = Camera(fx_used, size, rvec, tvec)
    uv, depth = cam.project(obj)
    err = np.linalg.norm(uv - img, axis=1)
    info = {"rms_px": float(np.sqrt(np.mean(err ** 2))), "max_px": float(err.max()),
            "per_point_px": {n: float(e) for n, e in zip(names, err)},
            "min_depth_m": float(depth.min()), "n_points": len(names)}
    if verbose:
        print("[calib] %d points, reprojection RMS %.2f px (max %.2f px)" % (len(names), info["rms_px"], info["max_px"]))
        worst = sorted(info["per_point_px"].items(), key=lambda kv: -kv[1])[:3]
        print("        worst points: " + ", ".join("%s %.1fpx" % kv for kv in worst))
        print("        camera centre (m): X=%.2f  Y=%.2f  Z=%.2f   (X<0 means behind the bowler's stumps)" % tuple(cam.center))
    if depth.min() <= 0:
        print("[warn] some calibration points are behind the camera -- left/right or bowler/batter end probably swapped")
    if info["rms_px"] > 4:
        print("[warn] high reprojection error: re-click the worst points, or check the focal length")
    return cam, info


# ----------------------------------------------------------------------
# Ball physics
# ----------------------------------------------------------------------
def _acc(vx, vy, vz, k):
    sp = math.sqrt(vx * vx + vy * vy + vz * vz)
    return -k * sp * vx, -k * sp * vy, -G - k * sp * vz


def _rk4(x, y, z, vx, vy, vz, h, k):
    a1 = _acc(vx, vy, vz, k)
    v2 = (vx + 0.5 * h * a1[0], vy + 0.5 * h * a1[1], vz + 0.5 * h * a1[2])
    a2 = _acc(*v2, k)
    v3 = (vx + 0.5 * h * a2[0], vy + 0.5 * h * a2[1], vz + 0.5 * h * a2[2])
    a3 = _acc(*v3, k)
    v4 = (vx + h * a3[0], vy + h * a3[1], vz + h * a3[2])
    a4 = _acc(*v4, k)
    nx = x + h / 6.0 * (vx + 2 * v2[0] + 2 * v3[0] + v4[0])
    ny = y + h / 6.0 * (vy + 2 * v2[1] + 2 * v3[1] + v4[1])
    nz = z + h / 6.0 * (vz + 2 * v2[2] + 2 * v3[2] + v4[2])
    nvx = vx + h / 6.0 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
    nvy = vy + h / 6.0 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
    nvz = vz + h / 6.0 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
    return nx, ny, nz, nvx, nvy, nvz


def simulate(theta, t0, t_end, drag=DRAG_K, dt=1.0 / 240):
    """theta = (x0,y0,z0, vx,vy,vz, e, mu). Gravity + quadratic drag, one bounce on z=0:
    vertical speed *= -e, horizontal speed *= mu. Returns dict with T (n,), P (n,3),
    V (n,3), t_bounce (or None)."""
    x, y, z, vx, vy, vz, e, mu = theta
    t = t0
    T, P, V = [t], [(x, y, z)], [(vx, vy, vz)]
    bounced, t_b = False, None
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        nx, ny, nz, nvx, nvy, nvz = _rk4(x, y, z, vx, vy, vz, h, drag)
        if (not bounced) and nz <= 0.0 and vz < 0.0:
            f = z / (z - nz) if z != nz else 1.0
            h1 = f * h
            bx, by, bz, bvx, bvy, bvz = _rk4(x, y, z, vx, vy, vz, h1, drag)
            t_b = t + h1
            T.append(t_b); P.append((bx, by, 0.0)); V.append((bvx, bvy, bvz))
            avx, avy, avz = mu * bvx, mu * bvy, -e * bvz
            T.append(t_b + 1e-9); P.append((bx, by, 0.0)); V.append((avx, avy, avz))
            bounced = True
            h2 = h - h1
            if h2 > 1e-9:
                nx, ny, nz, nvx, nvy, nvz = _rk4(bx, by, 0.0, avx, avy, avz, h2, drag)
            else:
                nx, ny, nz, nvx, nvy, nvz = bx, by, 0.0, avx, avy, avz
        t += h
        x, y, z, vx, vy, vz = nx, ny, nz, nvx, nvy, nvz
        T.append(t); P.append((x, y, z)); V.append((vx, vy, vz))
    return {"T": np.array(T), "P": np.array(P), "V": np.array(V), "t_bounce": t_b}


def _interp(sim, t, key="P"):
    A = sim[key]
    return np.stack([np.interp(t, sim["T"], A[:, k]) for k in range(3)], axis=1)


# default priors: weak, only there to stop the problem drifting when few detections exist
PRIORS = {"z0": (2.10, 0.45), "x0": (0.70, 1.20), "y0": (0.0, 0.70), "e": (0.55, 0.18), "mu": (0.75, 0.18)}
BOUNDS_LO = np.array([-3.0, -2.5, 1.0, 5.0, -18.0, -35.0, 0.15, 0.30])
BOUNDS_HI = np.array([6.0, 2.5, 3.3, 65.0, 18.0, 12.0, 0.95, 1.00])


def _initial_guesses(cam, dets, t_r, t_b):
    dets = np.asarray(dets)
    # ground point under the detection closest to the bounce time
    ib = int(np.argmin(np.abs(dets[:, 0] - t_b)))
    Pb = cam.intersect_plane_z(dets[ib, 1], dets[ib, 2], 0.0)
    if Pb is None:
        Pb = np.array([10.0, 0.0, 0.0])
    ir = int(np.argmin(np.abs(dets[:, 0] - t_r)))
    T = max(t_b - t_r, 0.08)
    starts = []
    for z0, dx in ((2.1, 0.0), (1.85, -0.5), (2.35, 0.5), (2.1, 1.0), (2.1, -0.9)):
        P0 = cam.intersect_plane_z(dets[ir, 1], dets[ir, 2], z0)
        r0 = np.array([P0[0], P0[1], z0]) if P0 is not None else np.array([0.7 + dx, 0.0, z0])
        r0[0] = float(np.clip(r0[0] + dx * 0.3, -2.5, 5.0))
        v0 = (Pb - r0 - 0.5 * np.array([0, 0, -G]) * T * T) / T
        v0[:2] *= 1.04                                   # a little more to pay for drag
        th = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], 0.55, 0.75])
        starts.append(np.clip(th, BOUNDS_LO + 1e-6, BOUNDS_HI - 1e-6))
    return starts


def fit_trajectory(cam, dets, t_release, t_bounce, t_end, drag=DRAG_K, sigma_px=2.0,
                   sigma_tb=1.0 / 30.0, priors=None, verbose=True, ended_by_impact=True):
    """dets: [(t, u, v), ...] ball detections (pixels) between release and end."""
    pri = dict(PRIORS)
    if priors:
        pri.update(priors)
    dets = np.asarray(dets, float)
    if len(dets) < 4:
        raise SystemExit("only %d usable ball detections -- need at least 4 to fit a trajectory" % len(dets))
    det_t, det_uv = dets[:, 0], dets[:, 1:3]

    def residuals(th):
        sim = simulate(th, t_release, t_end, drag)
        P = _interp(sim, det_t)
        uv, depth = cam.project(P)
        r_px = ((uv - det_uv) / sigma_px)
        r_px[depth < 0.3] = 50.0
        tb = sim["t_bounce"]
        if tb is None:
            r_tb = 10.0 + sim["P"][-1, 2] / 0.05
        else:
            r_tb = (tb - t_bounce) / sigma_tb
        # the ball was stopped by bat/pad at the impact event, so it cannot be far past the stumps
        x_end = sim["P"][-1, 0]
        r_end = max(0.0, x_end - (PITCH_LEN + 0.4)) / 0.25 if ended_by_impact else 0.0
        r_pr = [r_end, (th[2] - pri["z0"][0]) / pri["z0"][1], (th[0] - pri["x0"][0]) / pri["x0"][1],
                (th[1] - pri["y0"][0]) / pri["y0"][1], (th[6] - pri["e"][0]) / pri["e"][1],
                (th[7] - pri["mu"][0]) / pri["mu"][1]]
        return np.concatenate([r_px.ravel(), [r_tb], r_pr])

    best = None
    for th0 in _initial_guesses(cam, dets, t_release, t_bounce):
        try:
            sol = least_squares(residuals, th0, bounds=(BOUNDS_LO, BOUNDS_HI), loss="soft_l1",
                                f_scale=1.5, x_scale=[1, 1, 0.5, 5, 2, 3, 0.1, 0.1],
                                diff_step=1e-4, max_nfev=150)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise SystemExit("trajectory fit failed to converge")
    th = best.x
    sim = simulate(th, t_release, t_end, drag)
    uv, _ = cam.project(_interp(sim, det_t))
    err = np.linalg.norm(uv - det_uv, axis=1)
    inl = err < 4.0 * sigma_px                         # robust loss ignores outliers, so report inliers
    info = {"theta": th.tolist(), "cost": float(best.cost), "n_det": int(len(dets)),
            "n_outliers": int((~inl).sum()),
            "rms_px": float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("nan"),
            "median_px": float(np.median(err)), "per_det_px": err.tolist(),
            "t_bounce_fit": sim["t_bounce"], "warnings": []}
    if sim["t_bounce"] is None:
        info["warnings"].append("fitted ball never reaches the ground before the end time")
    elif abs(sim["t_bounce"] - t_bounce) > 2.5 / 60:
        info["warnings"].append("fitted bounce time differs from the event time by %.0f ms"
                                % (1000 * abs(sim["t_bounce"] - t_bounce)))
    if not np.isfinite(info["rms_px"]) or info["rms_px"] > 2.5 * sigma_px:
        info["warnings"].append("high reprojection error (%.1f px on inliers): check calibration / detections" % info["rms_px"])
    if info["n_outliers"] > max(2, len(dets) // 4):
        info["warnings"].append("%d of %d detections were rejected as outliers" % (info["n_outliers"], len(dets)))
    if len(dets) < 6:
        info["warnings"].append("few detections (%d): depth is mostly driven by the priors and the bounce point" % len(dets))
    near = [i for i, e in enumerate(th) if abs(th[i] - BOUNDS_LO[i]) < 1e-3 or abs(th[i] - BOUNDS_HI[i]) < 1e-3]
    if near:
        info["warnings"].append("parameter(s) hit their bounds (index %s): fit may be unreliable" % near)
    if verbose:
        print("[fit] %d detections (%d rejected as outliers), inlier reprojection RMS %.2f px (median %.2f px)"
              % (len(dets), info["n_outliers"], info["rms_px"], info["median_px"]))
    return th, sim, info


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------
KMH = 3.6


def summarize(theta, t_release, t_impact, t_end, drag=DRAG_K):
    sim = simulate(theta, t_release, t_end + 0.7, drag)
    T, P, V = sim["T"], sim["P"], sim["V"]
    tb = sim["t_bounce"]
    out = {"release": {"t": t_release, "pos": P[0].tolist(), "speed_kmh": float(np.linalg.norm(V[0]) * KMH)}}
    if tb is not None:
        i0 = int(np.argmin(np.abs(T - tb)))
        vpre, vpost = V[i0], V[min(i0 + 1, len(V) - 1)]
        hpre, hpost = math.hypot(vpre[0], vpre[1]), math.hypot(vpost[0], vpost[1])
        out["bounce"] = {
            "t": tb, "pos": P[i0].tolist(),
            "speed_before_kmh": float(np.linalg.norm(vpre) * KMH),
            "speed_after_kmh": float(np.linalg.norm(vpost) * KMH),
            "angle_before_deg": float(math.degrees(math.atan2(-vpre[2], hpre))),
            "angle_after_deg": float(math.degrees(math.atan2(vpost[2], hpost))),
            "distance_from_batter_stumps_m": float(PITCH_LEN - P[i0][0]),
            "distance_from_bowler_stumps_m": float(P[i0][0]),
            "line_y_m": float(P[i0][1])}
    ti = min(max(t_impact, t_release), t_end)
    pi = _interp(sim, np.array([ti]))[0]
    vi = _interp(sim, np.array([ti]), "V")[0]
    out["impact"] = {"t": ti, "pos": pi.tolist(), "speed_kmh": float(np.linalg.norm(vi) * KMH)}
    # where the ball would reach the batter's stump line if nothing hit it
    after = (T > (tb or t_release)) & (P[:, 0] >= PITCH_LEN)
    if after.any():
        j = int(np.argmax(after))
        k = max(j - 1, 0)
        f = (PITCH_LEN - P[k, 0]) / (P[j, 0] - P[k, 0]) if P[j, 0] != P[k, 0] else 0.0
        y_at, z_at = P[k, 1] + f * (P[j, 1] - P[k, 1]), P[k, 2] + f * (P[j, 2] - P[k, 2])
        out["at_stumps_projected"] = {
            "y_m": float(y_at), "height_m": float(z_at),
            "would_hit_stumps": bool(abs(y_at) <= BOX_HALF_W + BALL_R and 0 <= z_at <= STUMP_H + BALL_R),
            "note": "straight-line physics projection to the stump line; ignores bat/pad and swing/spin after the bounce"}
    # mean speed release -> bounce along the actual path, for a sanity check against the app's speed
    out["theta"] = {"x0": theta[0], "y0": theta[1], "z0": theta[2], "vx": theta[3], "vy": theta[4],
                    "vz": theta[5], "restitution_e": theta[6], "horizontal_retention_mu": theta[7], "drag_k": drag}
    return out


def track_to_json(cam, theta, info, fps, t_release, t_bounce_event, t_impact, t_end, drag, meta=None):
    sim = simulate(theta, t_release, t_end, drag)
    T, P = sim["T"], sim["P"]
    ib = int(np.argmin(np.abs(T - sim["t_bounce"]))) if sim["t_bounce"] is not None else -1
    data = {
        "version": 1, "fps": fps, "camera": cam.to_dict(),
        "t": [round(float(v), 6) for v in T],
        "x": [round(float(v), 4) for v in P[:, 0]],
        "y": [round(float(v), 4) for v in P[:, 1]],
        "z": [round(float(v), 4) for v in P[:, 2]],
        "ib": ib, "t_bounce": sim["t_bounce"],
        "events_in": {"release": t_release, "bounce": t_bounce_event, "impact": t_impact},
        "metrics": summarize(theta, t_release, t_impact, t_end, drag),
        "fit": {k: v for k, v in info.items() if k != "theta"},
        "pitch": {"len": PITCH_LEN, "stump_h": STUMP_H, "stump_dy": STUMP_DY, "popping": POPPING,
                  "return_half": RETURN_HALF, "half_w": PITCH_HALF_W, "ball_r": BALL_R},
        "meta": meta or {}}
    return data


def print_metrics(m):
    r = m["release"]
    print("\n==== 3D TRACK ====")
    print("release : X=%.2f  Y=%+.2f  height %.2f m   speed %.1f km/h" % (r["pos"][0], r["pos"][1], r["pos"][2], r["speed_kmh"]))
    if "bounce" in m:
        b = m["bounce"]
        print("bounce  : X=%.2f  Y=%+.2f   %.2f m from the batter's stumps | speed %.1f -> %.1f km/h | angle in %.1f deg, out %.1f deg"
              % (b["pos"][0], b["pos"][1], b["distance_from_batter_stumps_m"], b["speed_before_kmh"],
                 b["speed_after_kmh"], b["angle_before_deg"], b["angle_after_deg"]))
    i = m["impact"]
    print("impact  : X=%.2f  Y=%+.2f  height %.2f m   speed %.1f km/h" % (i["pos"][0], i["pos"][1], i["pos"][2], i["speed_kmh"]))
    if "at_stumps_projected" in m:
        s = m["at_stumps_projected"]
        print("at the stumps (projected): Y=%+.2f  height %.2f m  -> %s" % (
            s["y_m"], s["height_m"], "would hit" if s["would_hit_stumps"] else "would miss"))


# ----------------------------------------------------------------------
# Reading your pipeline's files
# ----------------------------------------------------------------------
def load_ball_track(path):
    data = json.load(open(path))
    recs = data if isinstance(data, list) else (data.get("frames") or list(data.values()))
    return {int(r["frame"]): r for r in recs}


def load_events(path, delivery=None):
    """Accepts app1's *_timestamps.json (list) or cricket_infer's raw JSON ({'deliveries': [...]})."""
    data = json.load(open(path))
    if isinstance(data, dict):
        data = data.get("deliveries", [data])
    if not data:
        raise SystemExit("no deliveries in %s" % path)
    if delivery is None:
        entry = data[0]
    else:
        match = [d for d in data if d.get("delivery", d.get("index")) == delivery]
        if not match:
            raise SystemExit("delivery %s not found (have %s)" % (delivery, [d.get("delivery", d.get("index")) for d in data]))
        entry = match[0]

    def get(label):
        ev = entry.get(label)
        if not ev:
            return None
        t = ev.get("time_seconds", ev.get("time"))
        return {"t": float(t), "frame": ev.get("src_frame", ev.get("frame"))}
    return {"release": get("release"), "pitching": get("pitching"), "impact": get("impact")}


def detections_between(ball_by_frame, fps, t_lo, t_hi):
    dets = []
    for f in sorted(ball_by_frame):
        r = ball_by_frame[f]
        if not r.get("visible", True):
            continue
        t = float(r["time_s"]) if r.get("time_s") is not None else f / fps
        if t_lo <= t <= t_hi:
            dets.append((t, float(r["x"]), float(r["y"])))
    return dets


# ----------------------------------------------------------------------
# Drawing: perspective-correct shaded tube (same look as app1.py's 2D trail)
# ----------------------------------------------------------------------
TUBE_LIGHT = (0.30, -0.95)
LAYERS_PRE = (((14, 36, 100), 1.00), ((26, 58, 142), 0.78), ((42, 78, 170), 0.55), ((58, 94, 184), 0.32))
LAYERS_POST = (((62, 90, 178), 1.00), ((84, 112, 202), 0.70), ((110, 136, 216), 0.42))


def _draw_tube(draw, pts, widths, scale, ox, oy, layers, alpha=250):
    n = len(pts)
    if n == 0:
        return
    lx, ly = TUBE_LIGHT
    ll = math.hypot(lx, ly)
    lx, ly = lx / ll, ly / ll
    normals = []
    for i in range(n):
        a, b = pts[max(0, i - 1)], pts[min(n - 1, i + 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(tx, ty) or 1.0
        normals.append((-ty / ln, tx / ln))
    for color, frac in layers:
        fill = tuple(color) + (alpha,)
        P, R = [], []
        for i in range(n):
            nx, ny = normals[i]
            off = (1.0 - frac) * widths[i] / 2.0 * (nx * lx + ny * ly)
            P.append(((pts[i][0] + nx * off - ox) * scale, (pts[i][1] + ny * off - oy) * scale))
            R.append(max(0.5, widths[i] * frac * scale / 2.0))
        for i in range(n):
            x, y = P[i]
            r = R[i]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
            if i == 0:
                continue
            x0, y0 = P[i - 1]
            r0 = R[i - 1]
            dx, dy = x - x0, y - y0
            ln = math.hypot(dx, dy)
            if ln < 1e-6:
                continue
            px, py = -dy / ln, dx / ln
            draw.polygon([(x0 + px * r0, y0 + py * r0), (x + px * r, y + py * r),
                          (x - px * r, y - py * r), (x0 - px * r0, y0 - py * r0)], fill=fill)


def render_track_layer(size, segments, ss=3):
    """segments: [(pts (N,2), widths (N,), layers)] -> RGBA float array (H,W,4) 0..1 premultiplied-free."""
    from PIL import Image, ImageDraw
    W, H = size
    segs = [s for s in segments if len(s[0]) >= 2]
    if not segs:
        return None
    allp = np.concatenate([s[0] for s in segs])
    pad = float(max(s[1].max() for s in segs)) + 4
    x0, y0 = max(0, int(allp[:, 0].min() - pad)), max(0, int(allp[:, 1].min() - pad))
    x1, y1 = min(W, int(allp[:, 0].max() + pad)), min(H, int(allp[:, 1].max() + pad))
    if x1 <= x0 or y1 <= y0:
        return None
    big = Image.new("RGBA", ((x1 - x0) * ss, (y1 - y0) * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    for pts, widths, layers in segs:
        _draw_tube(d, [tuple(p) for p in pts], list(widths), ss, x0, y0, layers)
    small = big.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    layer = np.zeros((H, W, 4), np.float32)
    layer[y0:y1, x0:x1] = np.asarray(small, np.float32) / 255.0
    return layer


def overlay_video(video, track, out_path, tube_diameter_m=0.085, min_width_px=2.0, hold_end=True):
    cam = Camera.from_dict(track["camera"])
    T = np.array(track["t"])
    P = np.stack([track["x"], track["y"], track["z"]], 1)
    uv, depth = cam.project(P)
    widths = np.maximum(cam.fx * tube_diameter_m / np.maximum(depth, 0.5), min_width_px)
    ib = track["ib"] if track["ib"] >= 0 else len(T) - 1

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % video)
    fps = cap.get(cv2.CAP_PROP_FPS) or track.get("fps") or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != (cam.W, cam.H):
        print("[warn] video is %dx%d but the calibration was for %dx%d -- the overlay will be misplaced" % (W, H, cam.W, cam.H))
    tmp = out_path + ".tmp.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = idx / fps
        n = int(np.searchsorted(T, now, side="right"))
        if not hold_end:
            n = min(n, len(T))
        segs = []
        if n >= 2:
            n_pre = min(n, ib + 1)
            segs.append((uv[:n_pre], widths[:n_pre], LAYERS_PRE))
            if n > ib + 1:
                segs.append((uv[ib + 1:n], widths[ib + 1:n] * 0.8, LAYERS_POST))
        layer = render_track_layer((W, H), segs)
        if layer is not None:
            a = layer[:, :, 3:4]
            rgb = layer[:, :, :3][:, :, ::-1] * 255.0          # RGB -> BGR
            frame = (frame.astype(np.float32) * (1 - a) + rgb * a).astype(np.uint8)
        wr.write(frame)
        idx += 1
    cap.release()
    wr.release()
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", video, "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "copy", out_path]
        subprocess.run(cmd, check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)
        print("[warn] ffmpeg not found: wrote mp4v (may not play in browsers)")
    print("[overlay] wrote %s (%d frames)" % (out_path, idx))


# ----------------------------------------------------------------------
# Static plot
# ----------------------------------------------------------------------
def plot_track(track, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y, z = np.array(track["x"]), np.array(track["y"]), np.array(track["z"])
    ib = track["ib"]
    m = track["metrics"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 1]})
    a = ax[0]
    a.axhline(0, color="#3b7a3b", lw=3)
    for X in (0, PITCH_LEN):
        a.plot([X, X], [0, STUMP_H], color="#f2b134", lw=6)
    a.plot(x[:ib + 1], z[:ib + 1], color="#1f3f9c", lw=3)
    a.plot(x[ib + 1:], z[ib + 1:], color="#6d8be0", lw=3)
    for k, c in (("release", "#2ecc71"), ("bounce", "#f1c40f"), ("impact", "#e74c3c")):
        if k in m:
            p = m[k]["pos"]
            a.scatter([p[0]], [p[2]], c=c, s=70, zorder=5, label=k)
    a.set_xlim(-1.5, PITCH_LEN + 1.5); a.set_ylim(-0.1, max(2.6, z.max() + 0.3))
    a.set_aspect("equal"); a.set_xlabel("distance along pitch X (m)"); a.set_ylabel("height (m)")
    a.set_title("side view"); a.legend(loc="upper right")
    b = ax[1]
    b.add_patch(plt.Rectangle((0, -PITCH_HALF_W), PITCH_LEN, 2 * PITCH_HALF_W, color="#d8c690", alpha=0.7))
    for X in (0, PITCH_LEN):
        b.plot([X, X], [-STUMP_DY, STUMP_DY], color="#f2b134", lw=6)
    for X in (POPPING, PITCH_LEN - POPPING):
        b.plot([X, X], [-RETURN_HALF, RETURN_HALF], color="white", lw=1.5)
    b.plot(x[:ib + 1], y[:ib + 1], color="#1f3f9c", lw=3)
    b.plot(x[ib + 1:], y[ib + 1:], color="#6d8be0", lw=3)
    cx, cy, _ = track["camera"]["center"]
    b.scatter([cx], [cy], marker="s", c="k", s=40); b.annotate("camera", (cx, cy), textcoords="offset points", xytext=(4, 6))
    b.set_xlim(-6, PITCH_LEN + 1.5); b.set_ylim(-2.2, 2.2); b.set_aspect("equal")
    b.set_xlabel("X (m)"); b.set_ylabel("Y (m, + = left from bowler's end)"); b.set_title("top view")
    r = m["release"]
    txt = "release %.0f km/h" % r["speed_kmh"]
    if "bounce" in m:
        txt += " | bounce %.2f m from batter's stumps, %.0f km/h" % (m["bounce"]["distance_from_batter_stumps_m"], m["bounce"]["speed_before_kmh"])
    fig.suptitle(txt, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print("[plot] wrote", out_path)


# ----------------------------------------------------------------------
# Interactive viewer (three.js r128, no other dependency)
# ----------------------------------------------------------------------
THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"

VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ball track - 3D</title>
<style>
html,body{margin:0;height:100%;background:#0b1220;color:#e8eefc;font-family:-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
canvas{display:block;width:100%;height:100%}
#ui{position:absolute;left:12px;top:12px;display:flex;flex-direction:column;gap:8px;width:300px;max-width:calc(100vw - 24px)}
.card{background:rgba(10,18,36,.80);border:1px solid rgba(255,255,255,.14);border-radius:12px;padding:10px 12px;font-size:13px;line-height:1.5}
.card b{font-size:14px}
button,select{background:#1d2c52;color:#e8eefc;border:1px solid rgba(255,255,255,.2);border-radius:8px;padding:5px 9px;font-size:12px;cursor:pointer}
button:hover{background:#2a3f78}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;align-items:center}
input[type=range]{flex:1;min-width:120px}
table{border-collapse:collapse;width:100%}td{padding:1px 0}td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.lab{position:absolute;transform:translate(-50%,-135%);font-size:11px;padding:2px 6px;border-radius:6px;background:rgba(0,0,0,.65);pointer-events:none;white-space:nowrap}
</style></head><body>
<canvas id="c"></canvas><div id="labels"></div>
<div id="ui">
  <div class="card"><b>Ball track &middot; 3D</b><table id="stats"></table></div>
  <div class="card"><div class="row" id="views"></div>
    <div class="row"><label><input type="checkbox" id="grow" checked> trail grows with the ball</label></div></div>
  <div class="card"><div class="row"><button id="play">&#9654; Play</button>
    <select id="rate"><option value="0.1">0.1x</option><option value="0.25" selected>0.25x</option><option value="0.5">0.5x</option><option value="1">1x</option></select>
    <span id="clock" style="margin-left:auto;font-variant-numeric:tabular-nums"></span></div>
    <div class="row"><input type="range" id="scrub" min="0" max="1000" value="0"></div>
    <div class="row" style="font-size:11px;opacity:.7">drag = rotate &middot; wheel = zoom &middot; right-drag = pan</div></div>
</div>
<script src="__THREE__"></script>
<script>
const D = __DATA__;
const w2t = (x,y,z) => new THREE.Vector3(x, z, -y);          // world (x fwd, y left, z up) -> three (x, y up, z)
const N = D.t.length, PIT = D.pitch;
const PTS = D.t.map((_,i)=>w2t(D.x[i],D.y[i],D.z[i]));
const IB = D.ib >= 0 ? D.ib : N-1;
const T0 = D.t[0], T1 = D.t[N-1], TB = D.t[IB];

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 2));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x9ec7ee); scene.fog = new THREE.Fog(0x9ec7ee, 60, 220);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
scene.add(new THREE.HemisphereLight(0xffffff, 0x4a6a3a, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 0.45); sun.position.set(-15, 30, 10); scene.add(sun);

// ground, pitch strip, creases
const grass = new THREE.Mesh(new THREE.PlaneGeometry(400,400), new THREE.MeshStandardMaterial({color:0x2f6f35, roughness:1}));
grass.rotation.x = -Math.PI/2; grass.position.set(PIT.len/2, -0.002, 0); scene.add(grass);
const strip = new THREE.Mesh(new THREE.PlaneGeometry(PIT.len+3.0, PIT.half_w*2), new THREE.MeshStandardMaterial({color:0xa8946a, roughness:1}));
strip.rotation.x = -Math.PI/2; strip.position.set(PIT.len/2, 0.001, 0); scene.add(strip);
function line(x0,z0,x1,z1,w){ const L=Math.hypot(x1-x0,z1-z0);
  const m=new THREE.Mesh(new THREE.PlaneGeometry(L,w), new THREE.MeshBasicMaterial({color:0xffffff}));
  m.rotation.x=-Math.PI/2; m.rotation.z=-Math.atan2(z1-z0,x1-x0); m.position.set((x0+x1)/2,0.004,(z0+z1)/2); scene.add(m); }
[0, PIT.len].forEach(X=>line(X,-PIT.return_half,X,PIT.return_half,0.05));                        // bowling creases (through the stumps)
[PIT.popping, PIT.len-PIT.popping].forEach(X=>line(X,-PIT.return_half,X,PIT.return_half,0.05));  // popping creases
[ -PIT.return_half, PIT.return_half ].forEach(Z=>{ line(0,Z,PIT.popping+0.0,Z,0.05); line(PIT.len-PIT.popping,Z,PIT.len,Z,0.05); });

// stumps
function wicket(X){ const g=new THREE.Group();
  [-1,0,1].forEach(k=>{ const s=new THREE.Mesh(new THREE.CylinderGeometry(0.0175,0.0175,PIT.stump_h,14), new THREE.MeshStandardMaterial({color:0xf2b134, roughness:.5}));
    s.position.set(X, PIT.stump_h/2, k*PIT.stump_dy); g.add(s); });
  [-0.5,0.5].forEach(k=>{ const b=new THREE.Mesh(new THREE.CylinderGeometry(0.008,0.008,PIT.stump_dy*0.95,8), new THREE.MeshStandardMaterial({color:0xfff2c2}));
    b.rotation.x=Math.PI/2; b.position.set(X, PIT.stump_h+0.01, k*PIT.stump_dy); g.add(b); });
  scene.add(g); }
wicket(0); wicket(PIT.len);

const markers = [];
// camera marker
if (D.camera && D.camera.center){ const c=D.camera.center; const cm=new THREE.Mesh(new THREE.BoxGeometry(0.12,0.07,0.05), new THREE.MeshStandardMaterial({color:0x555555}));
  cm.position.copy(w2t(c[0],c[1],c[2])); scene.add(cm);
  const el=document.createElement('div'); el.className='lab'; el.textContent='camera'; document.getElementById('labels').appendChild(el); markers.push({pos:cm.position,el}); }

// ball + markers
const ball = new THREE.Mesh(new THREE.SphereGeometry(PIT.ball_r*1.6,24,16), new THREE.MeshStandardMaterial({color:0xc1121f, roughness:.4}));
scene.add(ball);
const M = D.metrics;
function marker(key,color,text){ if(!M[key]) return; const p=M[key].pos;
  const m=new THREE.Mesh(new THREE.SphereGeometry(0.045,16,12), new THREE.MeshBasicMaterial({color})); m.position.copy(w2t(p[0],p[1],p[2])); scene.add(m);
  const el=document.createElement('div'); el.className='lab'; el.textContent=text; document.getElementById('labels').appendChild(el); markers.push({pos:m.position,el}); }
marker('release',0x2ecc71,'release'); marker('bounce',0xf1c40f,'bounce'); marker('impact',0xe74c3c,'impact');

// trail (rebuilt from the samples up to "now" so it always ends exactly at the ball)
const trail = new THREE.Group(); scene.add(trail);
function clearTrail(){ while(trail.children.length){ const o=trail.children.pop(); o.geometry.dispose(); o.material.dispose(); } }
function tube(points,color){
  const pts=[points[0]]; for(let i=1;i<points.length;i++){ if(points[i].distanceTo(pts[pts.length-1])>1e-3) pts.push(points[i]); }
  if(pts.length<2) return;
  const curve=new THREE.CatmullRomCurve3(pts,false,'centripetal');
  const mesh=new THREE.Mesh(new THREE.TubeGeometry(curve,Math.max(8,pts.length*2),0.028,10,false), new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:0.35,roughness:.35,metalness:.05}));
  trail.add(mesh);
  const cap=new THREE.Mesh(new THREE.SphereGeometry(0.028,10,8), mesh.material.clone()); cap.position.copy(pts[0]); trail.add(cap); }
function posAt(t){ t=Math.min(Math.max(t,T0),T1); let i=0; while(i<N-2 && D.t[i+1]<t) i++;
  const a=D.t[i], b=D.t[i+1], f=b>a?(t-a)/(b-a):0; return PTS[i].clone().lerp(PTS[i+1],Math.min(Math.max(f,0),1)); }
function buildTrail(t){ clearTrail();
  const grow=document.getElementById('grow').checked; const tt=grow?t:T1;
  const pre=[], post=[];
  for(let i=0;i<N;i++){ if(D.t[i]>tt) break; (i<=IB?pre:post).push(PTS[i]); }
  const cur=posAt(tt);
  if(tt<=TB){ pre.push(cur); } else { if(!post.length) post.push(PTS[IB]); post.push(cur); if(tt<T1+1) {} }
  if(pre.length>1) tube(pre,0x2447b8);
  if(post.length>1) tube(post,0x7a97e6); }

// camera controls (orbit / zoom / pan)
const target=new THREE.Vector3(PIT.len*0.45,0.8,0); let yaw=0,pitch=0.3,dist=20;
function setView(camWorld,tgtWorld,fov){ camera.fov=fov||50; camera.updateProjectionMatrix(); const c=w2t(...camWorld), t=w2t(...tgtWorld); target.copy(t); const d=c.clone().sub(t);
  dist=d.length(); pitch=Math.asin(d.y/dist); yaw=Math.atan2(d.z,d.x); applyCam(); }
function applyCam(){ camera.position.set(target.x+dist*Math.cos(pitch)*Math.cos(yaw), target.y+dist*Math.sin(pitch), target.z+dist*Math.cos(pitch)*Math.sin(yaw)); camera.lookAt(target); }
const camC = D.camera && D.camera.center ? D.camera.center : [-3,0,1.4];
const VIEWS = {
  'Camera view':   ()=>{ const f=(D.camera&&D.camera.forward)||[1,0,0]; setView(camC,[camC[0]+f[0]*15,camC[1]+f[1]*15,camC[2]+f[2]*15], (D.camera&&D.camera.vfov_deg)||50); },
  'Bowler end':    ()=>setView([-4.5,0.0,1.7],[PIT.len*0.65,0,0.5]),
  'Side':          ()=>setView([PIT.len*0.5,-15,1.3],[PIT.len*0.5,0,1.0]),
  'Top':           ()=>setView([PIT.len*0.5,-0.5,15.5],[PIT.len*0.5,0,0]),   // bowler at the left, left side of the pitch at the top
  'Batter end':    ()=>setView([PIT.len+4.5,0.4,1.7],[PIT.len*0.3,0,0.5]),
  'Follow ball':   ()=>{ follow=true; }
};
let follow=false;
const vb=document.getElementById('views');
Object.keys(VIEWS).forEach(k=>{ const b=document.createElement('button'); b.textContent=k; b.onclick=()=>{ follow=false; VIEWS[k](); }; vb.appendChild(b); });
let drag=null;
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{ drag={x:e.clientX,y:e.clientY,pan:(e.button===2||e.shiftKey)}; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup',()=>drag=null);
canvas.addEventListener('pointermove',e=>{ if(!drag) return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if(drag.pan){ const right=new THREE.Vector3().setFromMatrixColumn(camera.matrix,0), up=new THREE.Vector3().setFromMatrixColumn(camera.matrix,1);
    target.addScaledVector(right,-dx*dist*0.0016).addScaledVector(up,dy*dist*0.0016); }
  else { yaw-=dx*0.006; pitch=Math.min(Math.max(pitch+dy*0.006,-0.05),1.55); }
  applyCam(); });
canvas.addEventListener('wheel',e=>{ e.preventDefault(); dist=Math.min(Math.max(dist*Math.exp(e.deltaY*0.001),1.5),150); applyCam(); },{passive:false});

// stats panel
const km=v=>v.toFixed(1);
let rows=[['release speed', km(M.release.speed_kmh)+' km/h'],['release height', M.release.pos[2].toFixed(2)+' m']];
if(M.bounce){ rows.push(['bounce from batter stumps', M.bounce.distance_from_batter_stumps_m.toFixed(2)+' m'],['bounce line (Y)', (M.bounce.line_y_m>=0?'+':'')+M.bounce.line_y_m.toFixed(2)+' m'],
  ['speed before / after bounce', km(M.bounce.speed_before_kmh)+' / '+km(M.bounce.speed_after_kmh)],['bounce angle in / out', M.bounce.angle_before_deg.toFixed(1)+'\u00b0 / '+M.bounce.angle_after_deg.toFixed(1)+'\u00b0']); }
rows.push(['impact speed', km(M.impact.speed_kmh)+' km/h']);
if(M.at_stumps_projected) rows.push(['projected at stumps', (M.at_stumps_projected.would_hit_stumps?'HITTING':'missing')+' (h '+M.at_stumps_projected.height_m.toFixed(2)+' m)']);
rows.push(['fit error', D.fit.rms_px.toFixed(1)+' px rms']);
document.getElementById('stats').innerHTML=rows.map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');

// ===== crease planes (light red, transparent) + release / impact / crossing heights =====
(function(){
  const XB = PIT.popping, XT = PIT.len - PIT.popping;                  // bowling / batting popping crease (world X)
  const WID = PIT.half_w*2, HGT = Math.max(3.0, Math.max.apply(null, D.z) + 0.6);
  const RED = 0xff5a5a, RED2 = 0xff2b2b;
  const group = new THREE.Group(); scene.add(group);
  const labels = [];

  function label(p3, html, color, dy){
    const el = document.createElement('div'); el.className = 'lab'; el.innerHTML = html;
    el.style.borderLeft = '3px solid ' + color; el.style.textAlign = 'left';
    document.getElementById('labels').appendChild(el); labels.push({pos: p3.clone(), el, dy: dy || 0});
  }
  function dot(p3, color, r){
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({color, depthTest:false}));
    m.position.copy(p3); m.renderOrder = 10; group.add(m);
  }
  function dropLine(p3){
    const g = new THREE.BufferGeometry().setFromPoints([p3, new THREE.Vector3(p3.x, 0.004, p3.z)]);
    group.add(new THREE.Line(g, new THREE.LineBasicMaterial({color:0xffffff, transparent:true, opacity:0.9})));
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.085, 24), new THREE.MeshBasicMaterial({color:0xffffff, side:THREE.DoubleSide, transparent:true, opacity:0.95}));
    ring.rotation.x = -Math.PI/2; ring.position.set(p3.x, 0.008, p3.z); group.add(ring);
  }
  function dashed(a, b, color){
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const l = new THREE.Line(g, new THREE.LineDashedMaterial({color, dashSize:0.12, gapSize:0.08}));
    l.computeLineDistances(); group.add(l);
  }

  // ---- the two planes
  const planeMat = new THREE.MeshBasicMaterial({color:RED, transparent:true, opacity:0.22, side:THREE.DoubleSide, depthWrite:false});
  function crease(X, name){
    const pl = new THREE.Mesh(new THREE.PlaneGeometry(WID, HGT), planeMat);
    pl.rotation.y = Math.PI/2; pl.position.set(X, HGT/2, 0); group.add(pl);          // spans the pitch width and up in the air
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(pl.geometry), new THREE.LineBasicMaterial({color:RED2}));
    edges.rotation.copy(pl.rotation); edges.position.copy(pl.position); group.add(edges);
    const band = new THREE.Mesh(new THREE.PlaneGeometry(0.12, WID), new THREE.MeshBasicMaterial({color:RED2, transparent:true, opacity:0.8, depthWrite:false}));
    band.rotation.x = -Math.PI/2; band.position.set(X, 0.006, 0); group.add(band);   // red strip on the ground = the crease line
    label(new THREE.Vector3(X, HGT, 0), name, '#ff5a5a', 0);
  }
  crease(XB, 'bowling crease plane');
  crease(XT, 'batting crease plane');

  // ---- where the path passes through a plane (null if it never does)
  function crossing(X){
    for(let i = 0; i < N-1; i++){
      const a = D.x[i], b = D.x[i+1];
      if(a !== b && (a-X)*(b-X) <= 0){
        const f = (X-a)/(b-a);
        return {y: D.y[i]+f*(D.y[i+1]-D.y[i]), z: D.z[i]+f*(D.z[i+1]-D.z[i])};
      }
    }
    return null;
  }

  // ---- release on the bowling plane, impact on the batting plane
  function pointOnPlane(key, X, color, hex, name){
    if(!M[key]) return;
    const p = M[key].pos, real = w2t(p[0], p[1], p[2]), onPlane = w2t(X, p[1], p[2]);
    dot(onPlane, hex, 0.07); dropLine(onPlane);
    let extra = '';
    if(Math.abs(p[0]-X) > 0.05){
      dashed(real, onPlane, hex);                                                      // joins the real point to its spot on the plane
      extra = ' <span style="opacity:.7">(real X=' + p[0].toFixed(1) + ' m)</span>';
    }
    label(onPlane, name + ' &middot; <b>' + p[2].toFixed(2) + ' m</b>' + extra, color, 0);
  }
  pointOnPlane('release', XB, '#2ecc71', 0x2ecc71, 'release');
  pointOnPlane('impact',  XT, '#ff6b6b', 0xe74c3c, 'impact');

  // ---- where the ball really passes through each plane
  [[XB, 'bowling'], [XT, 'batting']].forEach(function(pair){
    const X = pair[0], nm = pair[1], c = crossing(X);
    if(!c) return;
    const key = (X === XB) ? 'release' : 'impact';
    if(M[key] && Math.abs(M[key].pos[0]-X) <= 0.05) return;                         // same spot as the release/impact marker
    const p = w2t(X, c.y, c.z);
    dot(p, 0xffffff, 0.055); dropLine(p);
    label(p, 'passes ' + nm + ' crease &middot; <b>' + c.z.toFixed(2) + ' m</b>', '#ffffff', 40);
  });

  // ---- on/off switch + label placement (own loop, so it can hide the labels too)
  const row = document.createElement('div'); row.className = 'row';
  row.innerHTML = '<label><input type="checkbox" id="planes" checked> crease planes + heights</label>';
  document.getElementById('grow').closest('.card').appendChild(row);
  document.getElementById('planes').onchange = function(e){ group.visible = e.target.checked; };
  function placeLabels(){
    const on = document.getElementById('planes').checked;
    labels.forEach(function(m){
      const v = m.pos.clone().project(camera);
      m.el.style.display = (on && v.z < 1) ? 'block' : 'none';
      m.el.style.left = ((v.x*0.5+0.5)*innerWidth) + 'px';
      m.el.style.top  = ((-v.y*0.5+0.5)*innerHeight + m.dy) + 'px';
    });
    requestAnimationFrame(placeLabels);
  }
  requestAnimationFrame(placeLabels);
})();

// playback
let tNow=T0, playing=false, last=performance.now(), lastIdx=-1;
const scrub=document.getElementById('scrub'), clock=document.getElementById('clock');
document.getElementById('play').onclick=()=>{ playing=!playing; if(playing&&tNow>=T1-1e-3) tNow=T0; document.getElementById('play').innerHTML=playing?'&#10074;&#10074; Pause':'&#9654; Play'; };
scrub.oninput=()=>{ tNow=T0+(T1-T0)*scrub.value/1000; };
document.getElementById('grow').onchange=()=>{ lastIdx=-1; };
function resize(){ const w=innerWidth,h=innerHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
addEventListener('resize',resize); resize(); VIEWS['Camera view']();
tNow=T1;          // start showing the full track; press Play to animate

function frame(now){
  const dt=(now-last)/1000; last=now;
  if(playing){ tNow+=dt*parseFloat(document.getElementById('rate').value); if(tNow>=T1){ tNow=T1; playing=false; document.getElementById('play').innerHTML='&#9654; Play'; } }
  scrub.value=Math.round(1000*(tNow-T0)/(T1-T0)); clock.textContent=((tNow-T0)*1000).toFixed(0)+' ms';
  const idx=Math.floor((tNow-T0)*240)+(document.getElementById('grow').checked?0:-1000000);
  if(idx!==lastIdx){ buildTrail(tNow); lastIdx=idx; }
  ball.position.copy(posAt(tNow));
  if(follow){ target.lerp(ball.position,0.15); dist=Math.max(dist,6); applyCam(); }
  renderer.render(scene,camera);
  markers.forEach(m=>{ const v=m.pos.clone().project(camera); const vis=v.z<1; m.el.style.display=vis?'block':'none';
    m.el.style.left=((v.x*0.5+0.5)*innerWidth)+'px'; m.el.style.top=((-v.y*0.5+0.5)*innerHeight)+'px'; });
  requestAnimationFrame(frame);
}
buildTrail(T1); requestAnimationFrame(frame);
window.__ready = true;
</script></body></html>
"""


def write_viewer(track, out_path, three_src=THREE_CDN):
    html = VIEWER_HTML.replace("__THREE__", three_src).replace("__DATA__", json.dumps(track))
    with open(out_path, "w") as fp:
        fp.write(html)
    print("[viewer] wrote %s (open it in a browser; three.js loads from %s)" % (out_path, three_src))


# ----------------------------------------------------------------------
# Calibration GUI
# ----------------------------------------------------------------------
def read_frame(video, idx):
    if video.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        img = cv2.imread(video)
        if img is None:
            raise SystemExit("cannot read image %s" % video)
        return img
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read frame %d" % idx)
    return frame


def draw_pitch_check(img, cam, geom=None):
    out = img.copy()
    def poly(pts, color, th=2):
        uv, d = cam.project(np.array(pts, float))
        if (d > 0.1).all():
            cv2.polylines(out, [uv.astype(np.int32).reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)
    for X, end in ((0.0, "BOWL"), (PITCH_LEN, "BAT")):
        g = (geom or {}).get(end, {})
        z0, z1 = g.get("bottom_z", 0.0), g.get("top_z", STUMP_H)
        for y in (-STUMP_DY, 0.0, STUMP_DY):
            poly([(X, y, z0), (X, y, z1)], (0, 200, 255), 2)
    for X in (POPPING, PITCH_LEN - POPPING, 0.0, PITCH_LEN):
        poly([(X, -RETURN_HALF, 0), (X, RETURN_HALF, 0)], (255, 255, 255), 1)
    poly([(0, -PITCH_HALF_W, 0), (PITCH_LEN, -PITCH_HALF_W, 0)], (0, 255, 0), 1)
    poly([(0, PITCH_HALF_W, 0), (PITCH_LEN, PITCH_HALF_W, 0)], (0, 255, 0), 1)
    return out


def calibrate_gui(video, frame_idx, fx, size, out_path, refine_focal=False):
    img = read_frame(video, frame_idx)
    H, W = img.shape[:2]
    s = min(1.0, 1000.0 / max(W, H))
    names = list(CALIB_ORDER)
    clicks, k = {}, [0]
    win = "calibrate (click the points; s = skip, u = undo, ENTER = finish)"

    def render():
        disp = cv2.resize(img, (int(W * s), int(H * s)))
        for n, (u, v) in clicks.items():
            cv2.circle(disp, (int(u * s), int(v * s)), 4, (0, 255, 0), -1)
            cv2.putText(disp, n.replace("_", " "), (int(u * s) + 5, int(v * s) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        msg = "done - ENTER to finish" if k[0] >= len(names) else "[%d/%d] %s" % (k[0] + 1, len(names), describe_point(names[k[0]]))
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(disp, msg, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, disp)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and k[0] < len(names):
            clicks[names[k[0]]] = (x / s, y / s)
            print("[click %d/%d] %-18s pixel (%7.1f, %7.1f)  ->  world (%.3f, %+.3f, %.3f)" % (
                k[0] + 1, len(names), names[k[0]], x / s, y / s, *world_xyz(names[k[0]])))
            k[0] += 1
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
    save_calibration(img, clicks, fx, (W, H), out_path, refine_focal)


def print_point_table(clicks, geom=None):
    names = [n for n in CALIB_ORDER if n in clicks] + [n for n in clicks if n not in CALIB_ORDER and n in CALIB_POINTS]
    w = max([len(n) for n in names] + [13])
    print("\n%-*s  %-19s  %s" % (w, "Clicked point", "Pixel (u, v)", "Known world position (m)  (X, Y, Z)"))
    for n in names:
        u, v = clicks[n]
        X, Y, Z = world_xyz(n, geom)
        print("%-*s  (%7.1f, %7.1f)   (%.3f, %+.3f, %.3f)" % (w, n, u, v, X, Y, Z))
    print()


def save_calibration(img, clicks, fx, size, out_path, refine_focal=False, geom=None, sigmas=None, extra=None):
    print_point_table(clicks, geom)
    cam, info = solve_camera(clicks, fx, size, refine_focal, geom=geom, sigmas=sigmas)
    data = {"image_size": list(size), "fx_given": fx, "points": {n: list(map(float, uv)) for n, uv in clicks.items()},
            "camera": cam.to_dict(), "rms_px": info["rms_px"], "geom": geom, "sigmas": sigmas}
    if extra:
        data.update(extra)
    with open(out_path, "w") as fp:
        json.dump(data, fp, indent=2)
    check = out_path.rsplit(".", 1)[0] + "_check.png"
    if img is not None:
        cv2.imwrite(check, draw_pitch_check(img, cam, geom))
        print("[calib] wrote %s and %s  <- open the PNG: the drawn pitch/stumps should sit on the real ones" % (out_path, check))
    return cam


def calibrate_from_boxes(bowler_box, batter_box, fx, size, out_path, camera_at="bowler", frame=None, wicket_height_m=None):
    """Calibration from two wicket bounding boxes (x1,y1,x2,y2 px), e.g. from a detector.
    Boxes must span ground -> top of the stumps and the full width of the three stumps.
    wicket_height_m: that ground->top height if it is not the standard 0.711 m (stumps on a base)."""
    pts = wicket_box_points(bowler_box, batter_box, camera_at)
    geom = None
    if wicket_height_m:
        geom = {e: {"top_z": float(wicket_height_m), "bottom_z": 0.0} for e in ("BOWL", "BAT")}
    return save_calibration(frame, pts, fx, size, out_path, geom=geom)


def load_camera(calib_path):
    d = json.load(open(calib_path))
    if "camera" in d:
        return Camera.from_dict(d["camera"])
    cam, _ = solve_camera({k: tuple(v) for k, v in d["points"].items()}, d["fx"], d["image_size"])
    return cam


# ----------------------------------------------------------------------
# Synthetic demo (also the self-test)
# ----------------------------------------------------------------------
def demo_scene(rng, noise_px=1.5, click_px=1.2, fps=60.0, size=(540, 960), missing=0.3, outliers=2,
               cam_pos=(-2.3, 0.10, 1.15), look_at=(14.0, 0.0, 0.55), fx_rel=0.90, truth_theta=None):
    W, H = size
    fx = fx_rel * W
    # build a look-at camera
    C = np.array(cam_pos, float)
    z_axis = np.array(look_at, float) - C
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, [0, 0, 1.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis])
    cam = Camera(fx, size, cv2.Rodrigues(R)[0].ravel(), (-R @ C))

    th = np.array(truth_theta if truth_theta is not None else [0.6, 0.85, 2.10, 26.0, -1.5, -1.9, 0.55, 0.78])
    t_r = 20.0 / fps
    sim = simulate(th, t_r, t_r + 0.9)
    tb = sim["t_bounce"]
    i_imp = int(np.argmax((sim["T"] > tb) & (sim["P"][:, 0] >= PITCH_LEN - 0.9)))
    t_imp = float(sim["T"][i_imp])
    t_end = t_imp
    # calibration clicks (all stump points + crease corners)
    clicks = {}
    for n in CALIB_ORDER:
        uv, d = cam.project(np.array([CALIB_POINTS[n]]))
        clicks[n] = tuple((uv[0] + rng.normal(0, click_px, 2)).tolist())
    # ball detections, per frame
    ball = []
    f0, f1 = int(t_r * fps) - 1, int(t_end * fps) + 2
    for f in range(f0, f1 + 1):
        t = f / fps
        if t < t_r or t > t_end:
            continue
        P = _interp(sim, np.array([t]))
        uv, _ = cam.project(P)
        vis = rng.random() > missing
        u, v = (uv[0] + rng.normal(0, noise_px, 2)) if vis else (0.0, 0.0)
        ball.append({"frame": f, "x": float(u), "y": float(v), "visible": bool(vis), "time_s": t})
    for _ in range(outliers):                       # confident false detections far from the ball
        r = ball[int(rng.integers(1, len(ball) - 1))]
        r["x"], r["y"], r["visible"] = float(rng.uniform(0, W)), float(rng.uniform(0, H * 0.6)), True
    ev_noise = lambda: float(rng.integers(-1, 2)) / fps
    events = [{"delivery": 1, "score": 0.9, "complete": True,
               "release": {"time_seconds": t_r + ev_noise(), "frame": int(round(t_r * fps)), "src_frame": int(round(t_r * fps))},
               "pitching": {"time_seconds": tb + ev_noise(), "frame": int(round(tb * fps)), "src_frame": int(round(tb * fps))},
               "impact": {"time_seconds": t_imp, "frame": int(round(t_imp * fps)), "src_frame": int(round(t_imp * fps))}}]
    return {"cam": cam, "clicks": clicks, "ball": ball, "events": events, "theta": th, "sim": sim,
            "t_r": t_r, "t_b": tb, "t_imp": t_imp, "fps": fps, "size": size, "fx": fx}


def write_demo_video(scene, path):
    cam, W, H, fps = scene["cam"], scene["size"][0], scene["size"][1], scene["fps"]
    n = int((scene["t_imp"] + 0.4) * fps)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    far = cam.project(np.array([[2000.0, 0.0, 0.0]]))[0][0, 1]       # horizon row
    for f in range(n):
        img = np.zeros((H, W, 3), np.uint8)
        hrow = int(np.clip(far, 0, H))
        img[:hrow] = (205, 175, 140)                   # sky (BGR)
        img[hrow:] = (70, 130, 70)                     # grass
        pitch = [(-1.5, -PITCH_HALF_W, 0), (PITCH_LEN + 1.5, -PITCH_HALF_W, 0), (PITCH_LEN + 1.5, PITCH_HALF_W, 0), (-1.5, PITCH_HALF_W, 0)]
        uv, d = cam.project(np.array(pitch, float))
        cv2.fillPoly(img, [uv.astype(np.int32).reshape(-1, 1, 2)], (120, 170, 200))
        img = draw_pitch_check(img, cam)
        t = f / fps
        P = _interp(scene["sim"], np.array([min(max(t, scene["t_r"]), scene["t_imp"])]))
        uv, d = cam.project(P)
        if scene["t_r"] <= t <= scene["t_imp"]:
            cv2.circle(img, (int(uv[0, 0]), int(uv[0, 1])), 4, (30, 30, 220), -1, cv2.LINE_AA)
        wr.write(img)
    wr.release()
    if shutil.which("ffmpeg"):
        tmp = path + ".tmp.mp4"
        os.replace(path, tmp)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True)
        os.remove(tmp)


# ----------------------------------------------------------------------
# Automatic wicket detection (classical computer vision -- no trained model)
# ----------------------------------------------------------------------
# Plastic/net stumps are strongly coloured, so a colour mask + shape rules finds them:
#   1. median of several frames  -> static background (players and ball vanish)
#   2. HSV colour mask -> connected blobs
#   3. NEAR wicket = the biggest tall blob; split into 3 stumps + base from its row/column profiles
#   4. FAR wicket  = a small, tall, thin blob above it; its top/bottom edges are refined to sub-pixel
#                    with half-maximum crossings (it is only ~10 x 40 px, so every pixel matters)
STUMP_HUES = {"yellow": (14, 36), "orange": (4, 20)}      # OpenCV hue range (0..179)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def median_frame(source, n=15):
    """Static background of a fixed camera: per-pixel median of n frames spread over the clip."""
    if source.lower().endswith(IMAGE_EXTS):
        img = cv2.imread(source)
        if img is None:
            raise SystemExit("cannot read image %s" % source)
        return img
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), max(1, n)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("could not read any frame from %s" % source)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def stump_mask(img, hue, s_min=100, v_min=110):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (int(hue[0]), int(s_min), int(v_min)), (int(hue[1]), 255, 255)) > 0


def soft_score(img, hue, s_min, v_min):
    """0..1 'how stump-coloured is this pixel' (used for sub-pixel edges of the blurry far wicket)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gate = ((h >= hue[0] - 2) & (h <= hue[1] + 2)).astype(np.float32)
    return gate * np.clip((sat - 0.4 * s_min) / (0.6 * s_min), 0, 1) * np.clip((val - 0.6 * v_min) / (0.4 * v_min), 0, 1)


def _runs(flags, min_len=1, merge_gap=1):
    runs, st = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and st is None:
            st = i
        if (not f) and st is not None:
            runs.append([st, i - 1]); st = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] + 1 >= min_len]


def analyse_near_wicket(sub, x0, y0):
    """sub: boolean mask of the near wicket's blob (stumps + base). Returns the 3 stump centre
    lines (top and bottom points, image coordinates), where the base starts, and a few sizes."""
    h, w = sub.shape
    widths = sub.sum(1).astype(float)
    mid = float(np.median(widths[int(0.2 * h):int(0.6 * h)])) or 1.0
    base_top = None
    for r in range(int(0.6 * h), h):                       # first row from which the set is much wider = the base
        if widths[r] > 1.6 * mid and (widths[r:] > 1.4 * mid).mean() > 0.9:
            base_top = r
            break
    stump_end = base_top if base_top is not None else h
    r0, r1 = int(0.15 * stump_end), max(int(0.85 * stump_end), int(0.15 * stump_end) + 2)
    runs = _runs(sub[r0:r1].mean(0) > 0.55, min_len=3, merge_gap=1)
    out = {"ok": False, "n_runs": len(runs), "stumps": [], "bottom": float(y0 + h),
           "base_top": None if base_top is None else float(y0 + base_top),
           "x_min": float(x0 + (runs[0][0] if runs else 0)), "x_max": float(x0 + (runs[-1][1] + 1 if runs else w)),
           "y_top": float(y0 + int(np.argmax(widths > 0.3 * mid)))}
    if len(runs) != 3:
        return out
    for (a, b) in runs:
        cs = sub[:stump_end, a:b + 1]
        top_i = int(np.argmax(cs.mean(1) > 0.5))           # first row where this stump is at least half filled
        rows = np.arange(top_i, stump_end)
        cx = np.array([np.nonzero(cs[r])[0].mean() + a + 0.5 for r in rows])
        k, c = np.polyfit(rows, cx, 1)                     # centre line x = k*y + c (follows any lean / camera roll)
        out["stumps"].append({"x_top": float(x0 + k * top_i + c), "y_top": float(y0 + top_i),
                              "x_bot": float(x0 + k * stump_end + c), "y_bot": float(y0 + stump_end), "width_px": int(b - a + 1)})
    out["ok"] = True
    return out


def _half_max_top(p, thr):
    i = int(np.argmax(p >= thr))
    if i == 0 or p[i] == p[i - 1]:
        return i
    return (i - 1 + 0.5) + (thr - p[i - 1]) / (p[i] - p[i - 1])


def _half_max_bottom(p, thr):
    j = len(p) - 1 - int(np.argmax(p[::-1] >= thr))
    if j >= len(p) - 1 or p[j] == p[j + 1]:
        return j + 1.0
    return (j + 0.5) + (p[j] - thr) / (p[j] - p[j + 1])


def refine_thin_wicket(img, bbox, hue, s_min, v_min):
    """Sub-pixel top/bottom/centre of a small blurry wicket inside bbox=(x, y, w, h)."""
    x, y, w, h = bbox
    H, W = img.shape[:2]
    X0, X1, Y0, Y1 = max(0, x - 3), min(W, x + w + 3), max(0, y - 14), min(H, y + h + 10)
    sc = soft_score(img[Y0:Y1, X0:X1], hue, 0.55 * s_min, 0.7 * v_min)     # more tolerant: far stumps are pale/blurred
    rowp = sc.sum(1)
    rows = np.nonzero(rowp > 0.25 * rowp.max())[0]
    r_lo, r_hi = int(rows.min()), int(rows.max())
    span = r_hi - r_lo + 1
    mid_plateau = float(np.median(rowp[r_lo + int(0.1 * span): r_lo + int(0.5 * span) + 1]))      # the 3 stumps
    top = _half_max_top(rowp, 0.5 * mid_plateau)
    # the base is the widest part: take its peak in the lower half, then walk DOWN to the first half-maximum
    # crossing (so a faint tail below it -- a shoe, a shadow -- cannot stretch the bottom edge)
    smooth = np.convolve(rowp, [0.25, 0.5, 0.25], mode="same")
    lo_half = r_lo + span // 2
    i_pk = lo_half + int(np.argmax(smooth[lo_half:r_hi + 1]))
    thr_b = 0.5 * float(smooth[i_pk])
    j = i_pk
    while j + 1 < len(rowp) and rowp[j + 1] >= thr_b:
        j += 1
    bottom = (j + 0.5) + (rowp[j] - thr_b) / max(rowp[j] - rowp[min(j + 1, len(rowp) - 1)], 1e-6)
    r_a, r_b = int(top) + 2, max(int(bottom) - 5, int(top) + 4)
    band = sc[r_a:r_b]
    xs = np.arange(sc.shape[1]) + 0.5
    xc = float((band.sum(0) * xs).sum() / max(band.sum(), 1e-6))
    return {"top_y": float(Y0 + top), "bottom_y": float(Y0 + bottom), "xc": float(X0 + xc), "height_px": float(bottom - top)}


def detect_wickets(img, color="yellow", hsv=None, near_box=None, far_box=None, s_min=100, v_min=110):
    """Find the near and far wicket. hsv = (hue_lo, hue_hi, s_min, v_min) overrides the colour preset.
    near_box / far_box = (x1, y1, x2, y2) override the automatic search for that wicket."""
    hue = STUMP_HUES[color]
    if hsv:
        hue, s_min, v_min = (hsv[0], hsv[1]), hsv[2], hsv[3]
    H, W = img.shape[:2]
    mask = stump_mask(img, hue, s_min, v_min)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(closed, 8)
    comps = [{"label": i, "x": int(st[i][0]), "y": int(st[i][1]), "w": int(st[i][2]), "h": int(st[i][3]),
              "area": int(st[i][4]), "cx": float(cen[i][0]), "cy": float(cen[i][1])}
             for i in range(1, n) if st[i][4] >= 20]
    comps.sort(key=lambda c: -c["area"])
    det = {"mask": mask, "hue": hue, "s_min": s_min, "v_min": v_min, "notes": [], "n_blobs": len(comps)}

    # ---- near wicket
    if near_box:
        x1, y1, x2, y2 = [int(round(v)) for v in near_box]
        sub = closed[y1:y2, x1:x2] > 0
        near = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0, "area": int(sub.sum())}
    else:
        tall = [c for c in comps if 1.2 <= c["h"] / c["w"] <= 7.0]
        if not tall:
            raise SystemExit("no wicket-like %s blob found (%d candidate blobs). Check --stump-color / --hsv, or pass --near-box"
                             % (color, len(comps)))
        near = tall[0]
        sub = lab[near["y"]:near["y"] + near["h"], near["x"]:near["x"] + near["w"]] == near["label"]
    det["near"] = {"bbox": (near["x"], near["y"], near["w"], near["h"]), **analyse_near_wicket(sub, near["x"], near["y"])}
    if not det["near"]["ok"]:
        det["notes"].append("near wicket: found %d stump columns instead of 3 -- falling back to the blob's box corners"
                            % det["near"]["n_runs"])

    # ---- far wicket
    if far_box:
        x1, y1, x2, y2 = [int(round(v)) for v in far_box]
        fb = (x1, y1, x2 - x1, y2 - y1)
    else:
        cands = [c for c in comps if c is not near and 0.015 * near["h"] <= c["h"] <= 0.5 * near["h"]
                 and c["area"] <= 0.3 * near["area"] and c["h"] / c["w"] >= 1.3 and c["cy"] < near["cy"]]
        if not cands:
            raise SystemExit("could not find the FAR wicket (small tall blob above the near wicket). It may be hidden or a different "
                             "colour: pass --far-box x1,y1,x2,y2, or try --hsv / --stump-color")
        cands.sort(key=lambda c: -c["area"] * math.exp(-((c["cx"] - near["cx"]) / (0.6 * W)) ** 2))
        far = cands[0]
        fb = (far["x"], far["y"], far["w"], far["h"])
        if len(cands) > 1:
            det["notes"].append("far wicket: %d candidate blobs, picked the biggest/most central -- check wickets_debug.png" % len(cands))
    det["far"] = {"bbox": fb, **refine_thin_wicket(img, fb, hue, s_min, v_min)}
    ratio = det["far"]["height_px"] / max(near["h"], 1)
    if not 0.02 < ratio < 0.4:
        det["notes"].append("far/near wicket height ratio %.3f looks unusual" % ratio)
    return det


def box_points_xywh(box, end, camera_at="bowler", sigma=2.5, dense_sigma=4.0):
    """Every calibration point implied by one wicket bounding box (x, y, w, h) in pixels:
         * the 4 corners  (outer stump edges x ground/top)
         * the 3 stump centre-lines' top and bottom points -- positioned inside the box using the real
           wicket proportions (stump centres sit at 8% / 50% / 92% of the box width)
    The box must cover the three stumps only (not a wider base) from their bottom to their top.
    Returns ({name: (u, v)}, {name: sigma_px}). The 6 stump points repeat information already in the
    corners, so they get a larger sigma (they count less) instead of being allowed to double-count."""
    x, y, w, h = [float(v) for v in box]
    flip = (camera_at == "batter")
    left, right = ("right", "left") if flip else ("left", "right")          # world side of image-left / image-right
    pts, sig = {}, {}
    for nm, uv in (("%s_box_%s_top" % (end, left), (x, y)), ("%s_box_%s_top" % (end, right), (x + w, y)),
                   ("%s_box_%s_base" % (end, left), (x, y + h)), ("%s_box_%s_base" % (end, right), (x + w, y + h))):
        pts[nm] = uv; sig[nm] = sigma
    fr = STUMP_RADIUS / WICKET_W
    for side, fx_ in ((left, fr), ("mid", 0.5), (right, 1.0 - fr)):
        for part, yy in (("top", y), ("base", y + h)):
            nm = "%s_%s_%s" % (end, side, part)
            pts[nm] = (x + fx_ * w, yy); sig[nm] = dense_sigma
    return pts, sig


def calibrate_from_xywh(near_box, far_box, f_px, size, out_path="calib.json", camera_at="bowler",
                        near_height_m=STUMP_H, far_height_m=None, near_bottom_m=0.0, frame=None, refine_focal=False):
    """Calibrate from the two wicket boxes (x, y, w, h) that a colour detector gives you -- no clicking.
      near_box / far_box : the wicket NEARER to / FARTHER from the camera.
      near_height_m      : height above the ground of the TOP of the boxed stumps (0.711 for plain stumps;
                           0.711 + base height if they stand on a base).
      near_bottom_m      : height above the ground of the BOTTOM of the near box (0 if the box reaches the
                           ground; the base height if the box stops where the stumps enter the base).
      far_height_m       : same as near_height_m for the far wicket (default: same as near).
    The far wicket's box is taken to reach the ground (its base cannot be resolved at that size)."""
    near_end, far_end = ("BOWL", "BAT") if camera_at == "bowler" else ("BAT", "BOWL")
    far_height_m = near_height_m if far_height_m is None else far_height_m
    p1, s1 = box_points_xywh(near_box, near_end, camera_at, sigma=2.0, dense_sigma=3.5)
    p2, s2 = box_points_xywh(far_box, far_end, camera_at, sigma=3.0, dense_sigma=4.5)
    pts, sig = {**p1, **p2}, {**s1, **s2}
    geom = {near_end: {"top_z": float(near_height_m), "bottom_z": float(near_bottom_m)},
            far_end: {"top_z": float(far_height_m), "bottom_z": 0.0}}
    return save_calibration(frame, pts, f_px, tuple(size), out_path, refine_focal, geom=geom, sigmas=sig,
                            extra={"from_boxes_xywh": {"near": list(map(float, near_box)), "far": list(map(float, far_box)),
                                                       "camera_at": camera_at}})


def wickets_to_points(det, camera_at="bowler", stump_h=STUMP_H, base_h=None, far_base_h=None):
    """Detected wickets -> (points_uv, geom, sigmas, info) for solve_camera."""
    near_end, far_end = ("BOWL", "BAT") if camera_at == "bowler" else ("BAT", "BOWL")
    sides = ("left", "mid", "right") if camera_at == "bowler" else ("right", "mid", "left")   # image order -> world side
    nr, fr = det["near"], det["far"]
    info = {}
    # height of the plastic base under the near stumps: measured from pixels unless given
    if nr["ok"]:
        tops = [s_["y_top"] for s_ in nr["stumps"]]
        stump_px = (nr["base_top"] if nr["base_top"] is not None else nr["bottom"]) - float(np.median(tops))
        if base_h is None:
            base_px = (nr["bottom"] - nr["base_top"]) if nr["base_top"] is not None else 0.0
            base_h = float(np.clip(stump_h * base_px / max(stump_px, 1.0), 0.0, 0.4))
            info["base_height_source"] = "estimated from pixels; this tends to OVERestimate (the base's top face is counted) -- measure it and pass --base-height-m"
        else:
            info["base_height_source"] = "given"
    elif base_h is None:
        base_h = 0.0
        info["base_height_source"] = "assumed 0 (stumps not resolved)"
    if far_base_h is None:
        far_base_h = base_h
    info.update({"base_h_near_m": base_h, "base_h_far_m": far_base_h})
    pts, sig = {}, {}
    geom = {near_end: {"top_z": base_h + stump_h, "bottom_z": base_h},
            far_end: {"top_z": far_base_h + stump_h, "bottom_z": 0.0}}
    if nr["ok"]:
        for side, s_ in zip(sides, nr["stumps"]):
            pts["%s_%s_top" % (near_end, side)] = (s_["x_top"], s_["y_top"]); sig["%s_%s_top" % (near_end, side)] = 3.0   # tops differ on toy sets
            pts["%s_%s_base" % (near_end, side)] = (s_["x_bot"], s_["y_bot"]); sig["%s_%s_base" % (near_end, side)] = 1.2
    else:                                                  # fallback: outer box of the stump part
        xl, xr, yt, yb = nr["x_min"], nr["x_max"], nr["y_top"], (nr["base_top"] if nr["base_top"] is not None else nr["bottom"])
        left, right = ("left", "right") if camera_at == "bowler" else ("right", "left")
        for nm, uv in (("box_%s_top" % left, (xl, yt)), ("box_%s_top" % right, (xr, yt)),
                       ("box_%s_base" % left, (xl, yb)), ("box_%s_base" % right, (xr, yb))):
            pts["%s_%s" % (near_end, nm)] = uv; sig["%s_%s" % (near_end, nm)] = 3.0
    pts["%s_mid_top" % far_end] = (fr["xc"], fr["top_y"]); sig["%s_mid_top" % far_end] = 1.8
    pts["%s_mid_base" % far_end] = (fr["xc"], fr["bottom_y"]); sig["%s_mid_base" % far_end] = 1.8
    return pts, geom, sig, info


def show_image_window(title, path, max_side=1200):
    """Pop the saved debug image up so the user can confirm the detection; any key closes it."""
    try:
        im = cv2.imread(path)
        sc = min(1.0, max_side / max(im.shape[:2]))
        cv2.imshow(title, cv2.resize(im, None, fx=sc, fy=sc))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("[wickets] could not open a window (%s); open %s yourself" % (e, path))


# Paste this over the existing draw_wicket_debug() in ball3d.py (same place, same name).
# Optional 2nd step: in calibrate_auto(), change the call to
#     draw_wicket_debug(img, det, pts, prefix + "_wickets.png", info)
# so the picture also prints the base height that was used.

def draw_wicket_debug(img, det, pts, path, info=None):
    """Debug picture of what was detected and which points go to the calibration.
       GREEN box   = the whole yellow object found (stumps + plastic base)
       CYAN box    = the three stumps only (this is what is measured)
       ORANGE band = the plastic base;  its bottom edge = the ground (height 0)
       RED dots    = the points used: stump TOPS and the point where each stump ENTERS THE BASE
                     (that is the stump's bottom, 0.711 m below its top -- NOT the bottom of the base)"""
    out = img.copy()
    tint = np.zeros_like(out)
    tint[det["mask"]] = (0, 255, 255)
    out = cv2.addWeighted(out, 0.7, tint, 0.3, 0)
    nr, fr = det["near"], det["far"]

    x, y, w, h = nr["bbox"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.putText(out, "NEAR: whole object (stumps + base)", (x, max(y - 26, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    if nr.get("stumps"):
        top = int(round(min(s_["y_top"] for s_ in nr["stumps"])))
        bot = int(round(nr["base_top"] if nr["base_top"] is not None else nr["bottom"]))
        x0, x1 = int(round(nr["x_min"])), int(round(nr["x_max"]))
        cv2.rectangle(out, (x0, top), (x1, bot), (255, 255, 0), 2)                       # stumps only
        cv2.putText(out, "stumps only (measured)", (x1 + 6, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        if nr["base_top"] is not None:
            ov = out.copy()
            cv2.rectangle(ov, (x, int(nr["base_top"])), (x + w, int(nr["bottom"])), (0, 140, 255), -1)
            out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)                                 # base block, shaded orange
            gy = int(round(nr["bottom"]))
            cv2.line(out, (x - 25, gy), (x + w + 25, gy), (0, 140, 255), 2)
            cv2.putText(out, "ground (z=0)", (x + w + 28, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
            cv2.putText(out, "plastic base", (x + w + 6, int(nr["base_top"]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    fx_, fy_, fw_, fh_ = fr["bbox"]
    cv2.rectangle(out, (fx_ - 1, int(fr["top_y"])), (fx_ + fw_ + 1, int(round(fr["bottom_y"]))), (255, 0, 255), 1)
    cv2.putText(out, "FAR wicket", (fx_ - 20, max(int(fr["top_y"]) - 18, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    for n, (u, v) in pts.items():
        cv2.circle(out, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
        cv2.circle(out, (int(round(u)), int(round(v))), 4, (255, 255, 255), 1)
    # label only the middle stump (and the far wicket) so the text never overlaps
    for n, (u, v) in pts.items():
        if "_mid_" in n:
            end, _, rest = n.partition("_")
            lab = "top of stump" if rest.endswith("_top") else ("stump bottom (enters base)" if nr.get("base_top") is not None and end in ("BOWL", "BAT") and abs(v - (nr["base_top"] or -1)) < 6 else "bottom")
            if end == ("BAT" if "BAT" in n else "BOWL") and abs(u - fr["xc"]) < 3:
                lab = "far top" if rest.endswith("_top") else "far bottom"
            cv2.putText(out, lab, (int(u) + 7, int(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(out, lab, (int(u) + 7, int(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    lines = ["red dots = stump TOPS and where the stumps ENTER THE BASE (stump bottoms)",
             "box bottom (orange line) is the BASE's bottom = ground; it is not a calibration point"]
    if info:
        lines.append("base height used: %.3f m  (%s)" % (info["base_h_near_m"], "given" if info.get("base_height_source") == "given" else "estimated from pixels"))
    for i, t in enumerate(lines):
        cv2.putText(out, t, (8, 16 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, t, (8, 16 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(path, out)


def detected_boxes_xywh(det, base_h, far_base_h):
    """The two wicket boxes (x, y, w, h) implied by a detection: near box = the 3 stumps (outer edges, tops to
    where they enter the base); far box = its blob, narrowed to the stump width using the near wicket's
    stump-to-base width ratio (the far base cannot be separated from its stumps)."""
    nr, fr = det["near"], det["far"]
    if nr["ok"]:
        tops = float(np.median([s_["y_top"] for s_ in nr["stumps"]]))
        bot = nr["base_top"] if nr["base_top"] is not None else nr["bottom"]
    else:
        tops, bot = nr["y_top"], (nr["base_top"] if nr["base_top"] is not None else nr["bottom"])
    near = (nr["x_min"], tops, nr["x_max"] - nr["x_min"], bot - tops)
    nx, ny, nw, nh = nr["bbox"]
    ratio = float(np.clip((nr["x_max"] - nr["x_min"]) / max(nw, 1), 0.5, 1.0)) if nr["base_top"] is not None else 1.0
    fx_, fy_, fw_, fh_ = fr["bbox"]
    far_w = max(fw_ * ratio, 2.0)
    far = (fr["xc"] - far_w / 2.0, fr["top_y"], far_w, fr["bottom_y"] - fr["top_y"])
    return near, far


def calibrate_auto(source, fx_spec, out_path, color="yellow", hsv=None, camera_at="bowler", frames=15,
                   stump_h=STUMP_H, base_h=None, far_base_h=None, near_box=None, far_box=None, refine_focal=False,
                   boxes_only=False):
    img = median_frame(source, frames)
    H, W = img.shape[:2]
    fx, how = focal_length_px(*fx_spec, size=(W, H))
    print("[calib] focal length %.1f px (%s), image %dx%d" % (fx, how, W, H))
    det = detect_wickets(img, color, hsv, near_box, far_box)
    pts, geom, sig, info = wickets_to_points(det, camera_at, stump_h, base_h, far_base_h)
    nr, fr = det["near"], det["far"]
    print("[wickets] near: %s, 3 stumps %s | far: top y=%.1f bottom y=%.1f (%.1f px tall) at x=%.1f"
          % ("bbox %s" % (nr["bbox"],), "found" if nr["ok"] else "NOT resolved", fr["top_y"], fr["bottom_y"], fr["height_px"], fr["xc"]))
    print("[wickets] base height under the near stumps: %.3f m (%s); far wicket base %.3f m"
          % (info["base_h_near_m"], info.get("base_height_source", ""), info["base_h_far_m"]))
    for note in det["notes"]:
        print("[warn]", note)
    prefix = out_path.rsplit(".", 1)[0]
    if boxes_only:                                       # derive every point from the two boxes instead
        nb, fb = detected_boxes_xywh(det, info["base_h_near_m"], info["base_h_far_m"])
        near_end, far_end = ("BOWL", "BAT") if camera_at == "bowler" else ("BAT", "BOWL")
        p1, s1 = box_points_xywh(nb, near_end, camera_at, sigma=2.0, dense_sigma=3.5)
        p2, s2 = box_points_xywh(fb, far_end, camera_at, sigma=3.0, dense_sigma=4.5)
        pts, sig = {**p1, **p2}, {**s1, **s2}
        print("[wickets] boxes-only mode: near box %s, far box %s -> %d points" %
              (tuple(round(v, 1) for v in nb), tuple(round(v, 1) for v in fb), len(pts)))
    draw_wicket_debug(img, det, pts, prefix + "_wickets.png")
    print("[wickets] debug image: %s_wickets.png" % prefix)
    show_image_window("wicket detection: NEAR box = green, FAR box = magenta, red dots = points used (press any key to continue)",
                      prefix + "_wickets.png")
    return save_calibration(img, pts, fx, (W, H), out_path, refine_focal, geom=geom, sigmas=sig,
                            extra={"auto_detect": {"color": color, "camera_at": camera_at, "boxes_only": boxes_only, **info}})


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_fit(cam, ball_by_frame, events, fps, out_path, drag=DRAG_K, sigma_px=2.0, meta=None, verbose=True):
    rel, pit, imp = events["release"], events["pitching"], events["impact"]
    if not rel or not pit:
        raise SystemExit("need at least the release and pitching events for this delivery")
    t_r, t_b = rel["t"], pit["t"]
    t_i = imp["t"] if imp else None
    last_det = max((r.get("time_s", f / fps) for f, r in ball_by_frame.items() if r.get("visible", True)), default=t_b)
    t_end = t_i if t_i else min(last_det, t_b + 0.35)
    dets = detections_between(ball_by_frame, fps, t_r - 1.0 / fps, t_end + 1.0 / fps)
    if verbose:
        print("[fit] release %.3fs  bounce %.3fs  impact %s  | %d detections in that window"
              % (t_r, t_b, "%.3fs" % t_i if t_i else "n/a", len(dets)))
    th, sim, info = fit_trajectory(cam, dets, t_r, t_b, t_end, drag, sigma_px, verbose=verbose,
                                   ended_by_impact=bool(t_i))
    data = track_to_json(cam, th, info, fps, t_r, t_b, t_i or t_end, t_end, drag, meta)
    with open(out_path, "w") as fp:
        json.dump(data, fp)
    if verbose:
        print_metrics(data["metrics"])
        for w in info["warnings"]:
            print("[warn]", w)
        print("[fit] wrote", out_path)
    return data


def cmd_demo(a):
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    sc = demo_scene(rng, noise_px=a.noise_px, outliers=a.outliers)
    P = lambda n: os.path.join(a.out_dir, n)
    write_demo_video(sc, P("demo.mp4"))
    json.dump(sc["ball"], open(P("demo_ball.json"), "w"))
    json.dump(sc["events"], open(P("demo_timestamps.json"), "w"), indent=1)
    cam = save_calibration(None, sc["clicks"], sc["fx"], sc["size"], P("calib.json"))
    # ground truth for comparison
    truth = summarize(sc["theta"], sc["t_r"], sc["t_imp"], sc["t_imp"])
    json.dump(truth, open(P("demo_truth_metrics.json"), "w"), indent=1)
    print("\n[demo] truth: release %.1f km/h, bounce at X=%.2f (%.2f m from batter), impact %.1f km/h"
          % (truth["release"]["speed_kmh"], truth["bounce"]["pos"][0], truth["bounce"]["distance_from_batter_stumps_m"], truth["impact"]["speed_kmh"]))
    ball = load_ball_track(P("demo_ball.json"))
    ev = load_events(P("demo_timestamps.json"), 1)
    run_fit(cam, ball, ev, sc["fps"], P("track3d.json"))
    print("\n[demo] files written to %s/ -- now try:" % a.out_dir)
    print("   python ball3d.py viewer  --track %s --out %s" % (P("track3d.json"), P("viewer.html")))
    print("   python ball3d.py overlay --video %s --track %s --out %s" % (P("demo.mp4"), P("track3d.json"), P("demo_3d.mp4")))
    print("   python ball3d.py plot    --track %s --out %s" % (P("track3d.json"), P("summary.png")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def auto_opts(p):
        p.add_argument("--frames", type=int, default=15, help="frames for the median background (players/ball vanish)")
        p.add_argument("--stump-color", choices=sorted(STUMP_HUES), default="yellow")
        p.add_argument("--hsv", help="custom colour: hue_lo,hue_hi,sat_min,val_min (OpenCV scale: hue 0-179)")
        p.add_argument("--camera-at", choices=("bowler", "batter"), default="bowler")
        p.add_argument("--stump-height-m", type=float, default=STUMP_H, help="height of the stumps above their base (default 0.711)")
        p.add_argument("--base-height-m", help="height of the plastic base under the NEAR stumps, in metres (measure it once with a ruler). 'auto' (default) estimates it from pixels but overestimates")
        p.add_argument("--far-base-height-m", help="base height under the FAR stumps (default: same as near)")
        p.add_argument("--near-box", help="x1,y1,x2,y2 override for the near wicket"); p.add_argument("--far-box", help="x1,y1,x2,y2 override for the far wicket")
        p.add_argument("--boxes-only", action="store_true",
                       help="derive ALL points from the two detected wicket boxes (corners + stump lines) instead of measuring each stump")

    c = sub.add_parser("calibrate", help="calibrate the camera: finds both wickets by colour automatically (no clicking); --manual to click instead")
    c.add_argument("video", help="video (or a still image)"); c.add_argument("--frame", type=int, default=0, help="frame to click on (manual mode only)")
    c.add_argument("--fx", type=float); c.add_argument("--hfov-deg", type=float); c.add_argument("--focal-35mm", type=float)
    c.add_argument("--refine-focal", action="store_true", help="also solve the focal length (needs 8+ well-spread points)")
    c.add_argument("--manual", action="store_true", help="skip the automatic detection and click the stump points yourself")
    c.add_argument("--no-fallback", action="store_true", help="if automatic detection fails, stop with an error instead of opening the click window")
    auto_opts(c)
    c.add_argument("--out", default="calib.json")

    ca = sub.add_parser("calibrate-auto", help="same as `calibrate` but never falls back to clicking")
    ca.add_argument("source", help="video (median of several frames is used) or a single image")
    ca.add_argument("--fx", type=float); ca.add_argument("--hfov-deg", type=float); ca.add_argument("--focal-35mm", type=float)
    ca.add_argument("--refine-focal", action="store_true"); ca.add_argument("--out", default="calib.json")
    auto_opts(ca)

    cx = sub.add_parser("calibrate-xywh", help="calibrate from the two wicket boxes (x,y,w,h) your own colour detector produced")
    cx.add_argument("--near", required=True, help="x,y,w,h of the wicket NEARER the camera"); cx.add_argument("--far", required=True, help="x,y,w,h of the FARTHER wicket")
    cx.add_argument("--camera-at", choices=("bowler", "batter"), default="bowler")
    cx.add_argument("--size", help="WxH of the image the boxes are in"); cx.add_argument("--video", help="or a video, to read the size / save a check image")
    cx.add_argument("--frame", type=int, default=0)
    cx.add_argument("--near-height-m", type=float, default=STUMP_H, help="ground-to-TOP height of the boxed near stumps (0.711 + base height if they stand on a base)")
    cx.add_argument("--near-bottom-m", type=float, default=0.0, help="height of the near box's BOTTOM edge above the ground (base height if the box stops at the base)")
    cx.add_argument("--far-height-m", type=float, help="ground-to-top height of the far wicket (default: same as near)")
    cx.add_argument("--fx", type=float); cx.add_argument("--hfov-deg", type=float); cx.add_argument("--focal-35mm", type=float)
    cx.add_argument("--refine-focal", action="store_true"); cx.add_argument("--out", default="calib.json")

    cb = sub.add_parser("calibrate-boxes", help="calibrate from two wicket bounding boxes (no clicking)")
    cb.add_argument("--bowler-box", required=True, help="x1,y1,x2,y2 of the BOWLER's-end wicket (pixels)")
    cb.add_argument("--batter-box", required=True, help="x1,y1,x2,y2 of the BATTER's-end wicket (pixels)")
    cb.add_argument("--camera-at", choices=("bowler", "batter"), default="bowler",
                    help="which end the camera stands behind (default bowler: image-left = +Y at both wickets)")
    cb.add_argument("--wicket-height-m", type=float, help="ground-to-top height of the wicket box if not 0.711 (e.g. stumps on a base)")
    cb.add_argument("--video", help="video (gives the image size and a check image); or use --size")
    cb.add_argument("--frame", type=int, default=0); cb.add_argument("--size", help="WxH, if no --video")
    cb.add_argument("--fx", type=float); cb.add_argument("--hfov-deg", type=float); cb.add_argument("--focal-35mm", type=float)
    cb.add_argument("--out", default="calib.json")

    f = sub.add_parser("fit", help="fit the 3D trajectory -> track3d.json")
    f.add_argument("--calib", required=True); f.add_argument("--ball", required=True, help="ball tracker json (frame, x, y, visible, time_s)")
    f.add_argument("--events", required=True, help="app1 *_timestamps.json"); f.add_argument("--delivery", type=int)
    f.add_argument("--video", help="to read fps"); f.add_argument("--fps", type=float)
    f.add_argument("--drag", type=float, default=DRAG_K); f.add_argument("--sigma-px", type=float, default=2.0)
    f.add_argument("--out", default="track3d.json")

    o = sub.add_parser("overlay", help="draw the 3D track onto the video")
    o.add_argument("--video", required=True); o.add_argument("--track", required=True); o.add_argument("--out", required=True)
    o.add_argument("--tube-diameter-m", type=float, default=0.085)

    v = sub.add_parser("viewer", help="write the interactive 3D viewer (with the popping-crease planes)")
    v.add_argument("--track", required=True); v.add_argument("--out", default="viewer.html")
    v.add_argument("--three-src", default=THREE_CDN, help="URL/path of three.min.js (r128) for offline use")

    p = sub.add_parser("plot", help="side/top view png")
    p.add_argument("--track", required=True); p.add_argument("--out", default="summary.png")

    d = sub.add_parser("demo", help="synthetic end-to-end example")
    d.add_argument("--out-dir", default="demo3d"); d.add_argument("--seed", type=int, default=1)
    d.add_argument("--noise-px", type=float, default=1.5); d.add_argument("--outliers", type=int, default=2)

    a = ap.parse_args()
    num = lambda t: None if t in (None, "auto") else float(t)
    box_ = lambda t: None if not t else [float(v) for v in t.split(",")]
    if a.cmd == "calibrate":
        if not a.manual:
            try:
                hsv_ = None if not a.hsv else [float(v) for v in a.hsv.split(",")]
                calibrate_auto(a.video, (a.fx, a.hfov_deg, a.focal_35mm), a.out, a.stump_color, hsv_, a.camera_at, a.frames,
                               a.stump_height_m, num(a.base_height_m), num(a.far_base_height_m), box_(a.near_box), box_(a.far_box),
                               a.refine_focal, a.boxes_only)
                return
            except (SystemExit, Exception) as ex:
                if a.no_fallback:
                    raise
                print("\n[calib] automatic wicket detection failed: %s" % (ex if str(ex) else type(ex).__name__))
                print("[calib] falling back to clicking the points by hand (use --stump-color / --hsv / --near-box / --far-box to fix the detector)\n")
        img = read_frame(a.video, a.frame)
        size = (img.shape[1], img.shape[0])
        fx, how = focal_length_px(a.fx, a.hfov_deg, a.focal_35mm, size)
        print("[calib] focal length %.1f px (%s), frame %dx%d" % (fx, how, *size))
        calibrate_gui(a.video, a.frame, fx, size, a.out, a.refine_focal)
    elif a.cmd == "calibrate-auto":
        hsv = None if not a.hsv else [float(v) for v in a.hsv.split(",")]
        calibrate_auto(a.source, (a.fx, a.hfov_deg, a.focal_35mm), a.out, a.stump_color, hsv, a.camera_at, a.frames,
                       a.stump_height_m, num(a.base_height_m), num(a.far_base_height_m), box_(a.near_box), box_(a.far_box),
                       a.refine_focal, a.boxes_only)
    elif a.cmd == "calibrate-xywh":
        frame = None
        if a.video:
            frame = read_frame(a.video, a.frame); size = (frame.shape[1], frame.shape[0])
        elif a.size:
            size = tuple(int(v) for v in a.size.lower().split("x"))
        else:
            raise SystemExit("give --size WxH or --video")
        fx, how = focal_length_px(a.fx, a.hfov_deg, a.focal_35mm, size)
        print("[calib] focal length %.1f px (%s), image %dx%d" % (fx, how, *size))
        xywh = lambda t: [float(v) for v in t.split(",")]
        calibrate_from_xywh(xywh(a.near), xywh(a.far), fx, size, a.out, a.camera_at, a.near_height_m, a.far_height_m,
                            a.near_bottom_m, frame, a.refine_focal)
    elif a.cmd == "calibrate-boxes":
        frame = None
        if a.video:
            frame = read_frame(a.video, a.frame)
            size = (frame.shape[1], frame.shape[0])
        elif a.size:
            size = tuple(int(v) for v in a.size.lower().split("x"))
        else:
            raise SystemExit("give --video or --size WxH")
        fx, how = focal_length_px(a.fx, a.hfov_deg, a.focal_35mm, size)
        box = lambda t: [float(v) for v in t.split(",")]
        print("[calib] focal length %.1f px (%s), frame %dx%d" % (fx, how, *size))
        calibrate_from_boxes(box(a.bowler_box), box(a.batter_box), fx, size, a.out, a.camera_at, frame, a.wicket_height_m)
    elif a.cmd == "fit":
        cam = load_camera(a.calib)
        fps = a.fps
        if fps is None and a.video:
            fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS)
        if not fps:
            raise SystemExit("give --fps or --video so frame numbers can be turned into seconds")
        run_fit(cam, load_ball_track(a.ball), load_events(a.events, a.delivery), fps, a.out, a.drag, a.sigma_px)
    elif a.cmd == "overlay":
        overlay_video(a.video, json.load(open(a.track)), a.out, a.tube_diameter_m)
    elif a.cmd == "viewer":
        write_viewer(json.load(open(a.track)), a.out, a.three_src)
    elif a.cmd == "plot":
        plot_track(json.load(open(a.track)), a.out)
    elif a.cmd == "demo":
        cmd_demo(a)


if __name__ == "__main__":
    main()