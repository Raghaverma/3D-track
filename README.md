# Takneek-3D-Ball-Track

Reconstruct a cricket ball's **3D trajectory from a single camera** and view it as an interactive 3D scene, a tube drawn on the video, or a side/top plot.

## Folder contents

| Path | What it is |
|---|---|
| `app1.py` | Runs the two ML models on a video: event timing (release / pitching / impact) + per-frame ball tracker. Writes `<out>_ball.json`, `<out>_timestamps.json` and an annotated video. |
| `ball3d.py` | Calibration (automatic wicket detection by colour, with a click fallback), 3D fit, overlay, viewer and plot (subcommands below). |
| `ball3d_manual.py` | Same as `ball3d.py`, but `calibrate` is **always manual**: you click all 12 stump bases and tops. Everything else is `ball3d.py` unchanged. |
| `check_overlay.py` | Draws the fitted 3D track, projected pitch/stumps and the ball detections on the video so you can check the fit (`--video --track --ball --calib --out`). |
| `cricket_infer.py`, `spot/`, `model/` | Event-timing model (E2E-Spot) and its weights. Called by `app1.py`. |
| `cricket_ball_tracker_repaired/` | Ball detector (HRNet-W18) + `track_ball.py`. Called by `app1.py`. |
| `videos/` | Input clips and `pitch_distances.json` (release-to-stump distance in metres per clip, used for the speed estimate). |
| `out/` | Outputs of `app1.py`. |
| `demo3d/` | Output of the synthetic demo (`ball3d.py demo`). |
| `ball3d_demo_viewer.html` | A pre-made example viewer. |
| `myenv/` | Virtual environment (`app1.py` auto-switches to it if present). |

## Setup

```bash
cd "/Users/takneekmacmini/Documents/3d ball track/Takneek-3D-Ball-Track"   # run everything from this folder
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
brew install ffmpeg          # if not installed
```

Known gotcha: keep `numpy>=2`. Installing `numpy<2` breaks scipy (`AttributeError: module 'numpy' has no attribute 'long'`).

Quick check with no data needed:

```bash
python ball3d.py demo --out-dir demo3d
python ball3d.py viewer --track demo3d/track3d.json --out demo3d/viewer.html && open demo3d/viewer.html
```

## How it works

One camera gives only a ray per detection, not a 3D point. Depth is recovered with:

1. **Calibration**: stumps and pitch have standard sizes. You click stump points on one frame; `solvePnP` gives the camera pose in a pitch-fixed world frame.
2. **Physics**: the ball flies under gravity + air drag, bounces on the ground (z = 0) and continues. The release position/velocity and bounce restitution are fitted so the projected path matches the 2D detections.

World frame (metres): X along the pitch (0 = bowler's stumps, 20.12 = batter's stumps), Y across (+ = left as seen from behind the bowler), Z up.

## Pipeline: video to 3D track

Run every command from the `Takneek-3D-Ball-Track` folder. Example for `83_kmph.mp4` (the repo contains no videos: copy your clip into this folder first).

```bash
cd "/Users/takneekmacmini/Documents/3d ball track/Takneek-3D-Ball-Track"
source myenv/bin/activate
mkdir -p out

# 1. Detect the ball and the delivery events
python app1.py 83_kmph.mp4 out/83_kmph_tracked.mp4 --device cpu --ball-device cpu --no-play
#    -> out/83_kmph_tracked_ball.json, out/83_kmph_tracked_timestamps.json, out/83_kmph_tracked.mp4

# 2. Calibrate the camera. Automatic: finds both (yellow) wickets itself and shows the detected boxes
python ball3d.py calibrate 83_kmph.mp4 --focal-35mm 26 --out calib_83.json
#    Or by hand: you click every stump base and top (see "Calibration clicks" below)
python ball3d_manual.py calibrate 83_kmph.mp4 --focal-35mm 26 --out calib_83.json
#    -> calib_83.json, calib_83_check.png (the drawn stumps/pitch must sit on the real ones),
#       and, in automatic mode, calib_83_wickets.png (the detected wicket boxes)

# 3. Fit the 3D trajectory
python ball3d.py fit --calib calib_83.json \
    --ball out/83_kmph_tracked_ball.json \
    --events out/83_kmph_tracked_timestamps.json \
    --video 83_kmph.mp4 --out track3d_83.json

# 4. See it
python ball3d.py viewer  --track track3d_83.json --out viewer_83.html && open viewer_83.html
python ball3d.py overlay --video 83_kmph.mp4 --track track3d_83.json --out 83_kmph_3d.mp4
python ball3d.py plot    --track track3d_83.json --out summary_83.png
```

Notes:
- `--device cpu --ball-device cpu` is the safe choice on a Mac. Drop them to auto-select.
- If a clip has several deliveries, add `--delivery N` to `fit`.
- Each video needs its own calibration unless the camera has not moved.
- `--focal-35mm 26` is a guess for a phone main camera. Use the real value (or `--fx` / `--hfov-deg`) for better depth accuracy.

## Viewing the 3D track

- **Interactive viewer** (`viewer_*.html`): open in a browser. Drag to rotate, scroll to zoom. Shows the pitch, stumps and the ball path with release, bounce and impact marked. It loads three.js from a CDN, so it needs internet; for offline use pass `--three-src /path/to/three.min.js` to `viewer`.
- **Overlay video** (`*_3d.mp4`): the 3D tube drawn onto the original video, thickness following perspective.
- **Plot** (`summary_*.png`): side and top views with the numbers.

## Calibration

### Automatic (`ball3d.py calibrate`)
Finds both wickets by colour (default yellow; `--stump-color`, `--hsv` for others), prints the table of pixel positions vs known world positions, and pops up a window with the detected boxes (green = near wicket, magenta = far wicket, red dots = points used). Press any key to continue. The same picture is saved as `calib_*_wickets.png`.
Useful options: `--base-height-m` (height of the plastic base under the near stumps; measure with a ruler, the pixel estimate is usually too high), `--near-box` / `--far-box` (override a wrong detection), `--boxes-only`, `--manual` (click instead). If detection fails it falls back to clicking unless you pass `--no-fallback`.
The far wicket is small and often behind the batter, so check its reprojection error; if it is high (> ~3 px) use the manual script.

### Manual (`ball3d_manual.py calibrate`)
```bash
python ball3d_manual.py calibrate 83_kmph.mp4 --focal-35mm 26 --out calib_83.json
```
You click 12 points: the **near** wicket first, then the **far** one. For each wicket click the three **bases** (where the stump meets the ground/base), then the three **tops**, each in the order leftmost, middle, rightmost **on screen**. The title bar shows `[n/12] <what to click>`, and every click prints its world coordinates in the terminal.

- A magnifier in the bottom-right corner follows the cursor, so you can place the tiny far wicket precisely.
- Magenta boxes grow around each wicket as you click, so you can see what you have marked.
- Keys: `s` skip a point you cannot see, `u` undo, `Enter` (or `q`) finish.
- At the end it prints the table, the reprojection error and a window with the drawn pitch/stumps (`calib_*_check.png`) for confirmation.
- Options: `--frame`, `--fx` / `--hfov-deg` / `--focal-35mm`, `--refine-focal`, `--camera-at bowler|batter`, `--base-height-m` / `--far-base-height-m` (stumps on a plastic base).
- You need at least 4 points, not all on one line; both wickets (bases and tops) is best.
- All other commands work the same through this script: `python ball3d_manual.py fit ...`, `viewer ...`, etc. The calibration file works with either script.

How to read the check: the reprojection RMS is the pixel error between your clicks and where the solved camera puts those points. Below about 3 px is good; high values mean a mis-clicked point or a wrong focal length / base height.

## `ball3d.py` subcommands

(`ball3d_manual.py` accepts the same ones; only its `calibrate` differs.)

| Command | Purpose |
|---|---|
| `calibrate` | Detect both wickets by colour (falls back to clicking), writes `calib.json` (camera pose) plus check images. Options: `--manual`, `--no-fallback`, `--frame`, `--fx`, `--hfov-deg`, `--focal-35mm`, `--refine-focal`, `--base-height-m`, `--stump-color`, `--near-box`, `--far-box`, `--out`. |
| `calibrate-auto`, `calibrate-xywh`, `calibrate-boxes` | Other no-click calibrations (detector only; from your own wicket boxes). |
| `fit` | 2D detections + events + calibration, writes `track3d.json` (3D track, speeds, bounce, etc). Options: `--delivery`, `--fps`, `--drag`, `--sigma-px`. |
| `overlay` | Draw the 3D track on the video. Option: `--tube-diameter-m`. |
| `viewer` | Write the interactive three.js `.html`. |
| `plot` | Side/top PNG with the numbers. |
| `demo` | Synthetic end-to-end example. |

## `app1.py` options

```
python app1.py [input.mp4] [output.mp4] [--device auto|cuda|cpu] [--ball-device auto|cuda|cpu]
               [--ball-confidence X] [--ball-visibility X] [--min-delivery-score 0.3]
               [--pitch-distance-m M] [--no-trail] [--no-play]
```

- With no arguments it uses `INPUT_VIDEO_NAME` at the top of `app1.py` (looked up in `videos/`).
- `--bundle-dir` defaults to this folder.
- The speed it prints is an average release-to-impact estimate using the pitch distance (from `videos/pitch_distances.json`, default 16 m). It is **not** a radar reading. Add `"<clip>.mp4": <metres>` to that file, or use `--pitch-distance-m`, for correct speeds.

## Troubleshooting

| Problem | Fix |
|---|---|
| `no such video: X.mp4` | Videos are not part of the repo. Copy the clip into the `Takneek-3D-Ball-Track` folder (or `videos/`) and run from that folder. |
| `cricket_infer.py not found under ...` | `--bundle-dir` must point at the folder containing `cricket_infer.py` (default: this folder). |
| `module 'numpy' has no attribute 'long'` | numpy was downgraded below 2. Run `pip install -U "numpy>=2" scipy opencv-python`. |
| Event marks are off the ball / speed looks wrong | Check `out/*_tracked.mp4`; set the right pitch distance. A bad 2D track gives a bad 3D fit. |
| Drawn pitch in `calib_*_check.png` does not match / high reprojection error | Check `calib_*_wickets.png`. Re-calibrate with `ball3d_manual.py` (careful clicks, use the magnifier), set `--base-height-m`, and use a correct focal length. |
