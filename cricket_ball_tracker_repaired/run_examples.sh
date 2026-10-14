#!/usr/bin/env bash
# Worked examples. Not a test -- edit the paths and run the ones you need.
set -euo pipefail
cd "$(dirname "$0")"

# 0. Does this bundle work on this machine at all?
python selftest.py

# 1. One video -> out/clip.json (+ overlay to eyeball it)
python track_ball.py --input /path/to/clip.mp4 --overlay

# 2. A folder of videos, recursively, JSON only, skipping ones already done
python track_ball.py --input /path/to/clips --recursive --out-dir results

# 3. An annotation-style frame directory at 50 fps, emitting Label.csv columns
python track_ball.py --input /path/to/frames_dir --fps 50 --csv

# 4. Higher precision over recall, plus the trajectory gate for clean tracks
python track_ball.py --input /path/to/clip.mp4 \
    --visibility-threshold 0.7 --threshold 0.2 --trajectory --overlay

# 5. CPU box, first 200 frames only
python track_ball.py --input /path/to/clip.mp4 --device cpu --max-frames 200
