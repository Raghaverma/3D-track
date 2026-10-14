#!/usr/bin/env python3
"""Verify this bundle can load its checkpoint and track a synthetic clip.

Run this first on a new machine: it needs no real video and no GPU, so a
failure here is an environment problem (missing timm, wrong torch, unreadable
checkpoint) rather than a data problem.

    python selftest.py            # ~20s on CPU

The synthetic clip is a white disc on grey moving in a straight line. The model
was never trained on it, so the check is that inference runs end to end and the
coordinates stay inside the frame -- not that it finds the fake ball.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import track_ball


FRAMES, WIDTH, HEIGHT = 24, 1280, 720


def write_synthetic_clip(destination: Path) -> None:
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit("FAIL: OpenCV cannot write mp4v; install opencv-python with codec support")
    for index in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
        centre = (120 + index * 40, 300 + index * 6)
        cv2.circle(frame, centre, 7, (245, 245, 245), -1, lineType=cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def main() -> int:
    checkpoint = track_ball.DEFAULT_CHECKPOINT
    if not checkpoint.exists():
        print(f"FAIL: no checkpoint at {checkpoint}")
        return 1
    print(f"checkpoint {checkpoint} ({checkpoint.stat().st_size / 1e6:.1f} MB)")
    with tempfile.TemporaryDirectory() as workspace:
        clip = Path(workspace) / "synthetic.mp4"
        write_synthetic_clip(clip)
        out_dir = Path(workspace) / "out"
        code = track_ball.main([
            "--input", str(clip), "--out-dir", str(out_dir), "--quiet", "--overlay",
        ])
        if code != 0:
            print("FAIL: track_ball reported a failed clip")
            return 1
        import json

        records = json.loads((out_dir / "synthetic.json").read_text())
        if len(records) != FRAMES:
            print(f"FAIL: expected {FRAMES} records, got {len(records)}")
            return 1
        if not all(0 <= r["x"] <= WIDTH and 0 <= r["y"] <= HEIGHT for r in records):
            print("FAIL: coordinates outside the source frame; check the scaling")
            return 1
        if not (out_dir / "synthetic_tracked.mp4").exists():
            print("FAIL: overlay video was not written")
            return 1
        keys = {"frame", "time_s", "x", "y", "confidence", "presence", "visible"}
        if not keys <= set(records[0]):
            print(f"FAIL: record is missing keys: {sorted(keys - set(records[0]))}")
            return 1
    print(f"PASS: {FRAMES} frames tracked, JSON + overlay written, coordinates in range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
