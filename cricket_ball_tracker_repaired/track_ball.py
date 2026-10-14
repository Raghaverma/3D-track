#!/usr/bin/env python3
"""Standalone cricket ball-tracker inference.

Runs one 13-channel ``StackedHRNetBallTracker`` checkpoint over videos or
image-sequence directories and writes per-frame ball positions.

Self-contained: with no ``--checkpoint`` it loads ``model/serving.pt`` next to
this file, and it imports only the inference half of ``temporal_ball_tracker``
(model, runtime, metrics, trajectory) which ships in this same folder.

    python track_ball.py --input clip.mp4                       # -> out/clip.json
    python track_ball.py --input clips/ --overlay --csv         # every video below clips/
    python track_ball.py --input frames_dir/ --fps 50           # image sequence

Inference is causal and streaming: frame N's answer uses frames N-2..N only, so
a long video costs bounded memory and the result is identical to what a live
feed would produce.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # importable from anywhere, including a copied folder
    sys.path.insert(0, str(_HERE))

from temporal_ball_tracker.model import build_tracker
from temporal_ball_tracker.runtime import StreamingTracker
from temporal_ball_tracker.trajectory import Detection, TrajectoryFilter

DEFAULT_CHECKPOINT = _HERE / "model" / "serving.pt"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track the cricket ball in videos or frame directories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, action="append", metavar="PATH",
        help="video file, image-sequence directory, or a directory of either; repeatable",
    )
    parser.add_argument("--out-dir", "-o", default="out", help="directory for JSON/CSV/overlay outputs")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="tracker checkpoint (.pt)")
    parser.add_argument(
        "--weights", choices=("ema", "raw"), default="ema",
        help="ema = the validation-selected averaged weights; raw = the same step's unaveraged weights",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16"), default="fp32",
        help="fp16 requires CUDA and measured slower than fp32 on an L4 at this input "
             "size (batch 1, 512x288 is launch-latency bound); measure before enabling",
    )
    parser.add_argument("--height", type=int, help="model input height (default: the checkpoint's)")
    parser.add_argument("--width", type=int, help="model input width (default: the checkpoint's)")
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="minimum peak score for a detection. Calibrated for this checkpoint on the "
             "val split (see README): the visibility head is the dominant gate, so this "
             "only trims the weak tail. 0.35, the default for the older focal-loss "
             "checkpoints, costs 2.4pp of recall here",
    )
    parser.add_argument(
        "--visibility-threshold", type=float, default=0.5,
        help="minimum P(ball present) from the visibility head; frames below it emit no detection",
    )
    parser.add_argument(
        "--trajectory", action="store_true",
        help="add the constant-velocity Kalman gate: detections inconsistent with the "
             "track become abstentions instead of jumps",
    )
    parser.add_argument("--gate-radius", type=float, default=40.0, help="--trajectory: base gate radius, model pixels")
    parser.add_argument("--max-coast", type=int, default=4, help="--trajectory: frames a track survives without a detection")
    parser.add_argument("--start-confidence", type=float, default=0.2, help="--trajectory: score needed to start a track")
    parser.add_argument("--overlay", action="store_true", help="also write <stem>_tracked.mp4 with the ball marked")
    parser.add_argument("--csv", action="store_true", help="also write <stem>.csv (TrackNet Label.csv column order)")
    parser.add_argument("--fps", type=float, help="frame rate for image sequences and overlay output")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames per clip (smoke tests)")
    parser.add_argument("--recursive", action="store_true", help="search input directories for videos recursively")
    parser.add_argument("--overwrite", action="store_true", help="re-run clips whose JSON already exists")
    parser.add_argument("--quiet", action="store_true", help="only print the final summary line")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
class FrameSource:
    """Uniform iterator over a video file or a directory of numbered frames."""

    def __init__(self, path: Path, fps_override: Optional[float] = None):
        self.path = path
        self.is_video = path.is_file()
        if self.is_video:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open video: {path}")
            self._capture = capture
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            self.fps = fps_override or capture.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self.images = sorted(
                child for child in path.iterdir()
                if child.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self.images:
                raise RuntimeError(f"no image frames in directory: {path}")
            probe = cv2.imread(str(self.images[0]))
            if probe is None:
                raise RuntimeError(f"could not read frame: {self.images[0]}")
            self.height, self.width = probe.shape[:2]
            self.count = len(self.images)
            self.fps = fps_override or 30.0

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.is_video:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break
                yield frame
            self._capture.release()
        else:
            for image_path in self.images:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise RuntimeError(f"could not read frame: {image_path}")
                yield frame

    def names(self) -> Optional[List[str]]:
        """Source frame filenames, for image sequences only."""
        return None if self.is_video else [image.name for image in self.images]


def discover_clips(inputs: List[str], recursive: bool) -> List[Path]:
    """Expand the --input list into the clips to run, in a stable order.

    A directory is one clip when it holds image frames, and a container of
    clips when it holds videos; that is how annotation exports and plain video
    folders both work without a flag.
    """

    clips: List[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"input does not exist: {path}")
        if path.is_file():
            clips.append(path)
            continue
        has_images = any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir() if child.is_file())
        if has_images:
            clips.append(path)
            continue
        pattern = "**/*" if recursive else "*"
        videos = sorted(
            child for child in path.glob(pattern)
            if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES
        )
        if not videos and recursive:
            # A tree of per-clip frame directories, e.g. a dataset split.
            videos = sorted(
                child for child in path.glob("**/")
                if child.is_dir() and any(g.suffix.lower() in IMAGE_SUFFIXES for g in child.iterdir() if g.is_file())
            )
        if not videos:
            raise FileNotFoundError(f"no videos or frame directories found under {path}")
        clips.extend(videos)
    # Preserve order, drop duplicates.
    seen, unique = set(), []
    for clip in clips:
        resolved = clip.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(clip)
    return unique


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, dict, torch.device]:
    """Build the tracker described by the checkpoint and load its weights."""

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}\n"
            f"Pass --checkpoint, or place one at {DEFAULT_CHECKPOINT}."
        )
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if args.precision == "fp16" and device.type != "cuda":
        raise SystemExit("--precision fp16 requires CUDA; use fp32 on CPU")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config") or {})
    sidecar = checkpoint_path.parent / "config.json"
    if "model" not in config and sidecar.exists():
        config = json.loads(sidecar.read_text())
    model_config = config.get("model")
    if model_config is None:
        raise SystemExit(
            f"{checkpoint_path} carries no model config; it is not a tracker checkpoint "
            "produced by this project"
        )
    model_config = dict(model_config)
    # The checkpoint holds the backbone weights; never re-download ImageNet ones.
    model_config["pretrained_backbone"] = False
    key = "raw_model" if args.weights == "raw" else "model"
    if key not in checkpoint:
        available = sorted(k for k in checkpoint if k in ("model", "raw_model"))
        raise SystemExit(f"checkpoint has no {key!r} weights; available: {available}")
    model = build_tracker(model_config).to(device)
    model.load_state_dict(checkpoint[key])
    model.eval()
    del checkpoint
    return model, config, device


# --------------------------------------------------------------------------- #
# One clip
# --------------------------------------------------------------------------- #
def track_clip(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    source: FrameSource,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Stream one clip and return per-frame records in *source* pixels."""

    tracker = StreamingTracker(
        model, config, device,
        image_size=(args.height, args.width),
        precision=args.precision,
        threshold=args.threshold,
        visibility_threshold=args.visibility_threshold,
    )
    gate = TrajectoryFilter(
        gate_radius=args.gate_radius,
        max_coast=args.max_coast,
        min_start_confidence=args.start_confidence,
    ) if args.trajectory else None
    scale_x = source.width / args.width
    scale_y = source.height / args.height
    names = source.names()
    records: List[Dict[str, object]] = []
    for index, frame in enumerate(source):
        if args.max_frames is not None and index >= args.max_frames:
            break
        result = tracker.predict(frame)
        x, y = result["x"], result["y"]
        visible = bool(result["visible"])
        if gate is not None:
            # Gate in model pixels, where the radii above are calibrated.
            point = gate.update(Detection(x, y, result["confidence"]))
            x, y, visible = point.x, point.y, point.visible
        record = {
            "frame": index,
            "time_s": round(index / source.fps, 6) if source.fps else None,
            "x": x * scale_x,
            "y": y * scale_y,
            "confidence": result["confidence"],
            "presence": result["presence"],
            "visible": visible,
        }
        if names is not None and index < len(names):
            record["file_name"] = names[index]
        records.append(record)
    return records


def write_overlay(source_path: Path, records: List[Dict[str, object]], destination: Path, fps: float) -> None:
    """Re-read the clip and draw each accepted detection."""

    source = FrameSource(source_path, fps_override=fps)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (source.width, source.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create overlay video: {destination}")
    try:
        for index, frame in enumerate(source):
            if index >= len(records):
                break
            record = records[index]
            if record["visible"]:
                point = (round(float(record["x"])), round(float(record["y"])))
                cv2.circle(frame, point, 9, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    frame, f"ball {float(record['confidence']):.2f}",
                    (point[0] + 12, point[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1, cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()


def write_csv(records: List[Dict[str, object]], destination: Path) -> None:
    """Write labels in the TrackNet ``Label.csv`` column order."""

    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file name", "visibility", "x-coordinate", "y-coordinate", "status"])
        for record in records:
            visible = bool(record["visible"])
            writer.writerow([
                record.get("file_name", f"{record['frame']:04d}.jpg"),
                int(visible),
                round(float(record["x"]), 2) if visible else -1,
                round(float(record["y"]), 2) if visible else -1,
                0,
            ])


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    clips = discover_clips(args.input, args.recursive)
    model, config, device = load_model(args)
    args.height = args.height or int(config.get("height", 288))
    args.width = args.width or int(config.get("width", 512))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.quiet:
        print(
            f"{len(clips)} clip(s) | {Path(args.checkpoint).name} ({args.weights}) | {device} "
            f"{args.precision} | {args.width}x{args.height} | threshold {args.threshold} "
            f"presence {args.visibility_threshold}"
            + (" | trajectory gate" if args.trajectory else "")
        )
    total_frames, total_visible, total_seconds, failures = 0, 0, 0.0, 0
    for position, clip in enumerate(clips, start=1):
        json_path = out_dir / f"{clip.stem}.json"
        if json_path.exists() and not args.overwrite:
            if not args.quiet:
                print(f"[{position}/{len(clips)}] {clip.name}: skipped (exists; --overwrite to redo)")
            continue
        try:
            source = FrameSource(clip, fps_override=args.fps)
            started = time.perf_counter()
            records = track_clip(model, config, device, source, args)
            elapsed = time.perf_counter() - started
        except Exception as error:  # one bad clip must not lose a batch run
            failures += 1
            print(f"[{position}/{len(clips)}] {clip.name}: FAILED {type(error).__name__}: {error}", file=sys.stderr)
            continue
        json_path.write_text(json.dumps(records, indent=2) + "\n")
        (out_dir / f"{clip.stem}.meta.json").write_text(json.dumps({
            "source": str(clip.resolve()),
            "frames": len(records),
            "visible": sum(1 for r in records if r["visible"]),
            "source_size": [source.width, source.height],
            "model_size": [args.width, args.height],
            "fps": source.fps,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "weights": args.weights,
            "threshold": args.threshold,
            "visibility_threshold": args.visibility_threshold,
            "trajectory_gate": bool(args.trajectory),
            "seconds": round(elapsed, 3),
        }, indent=2) + "\n")
        if args.csv:
            write_csv(records, out_dir / f"{clip.stem}.csv")
        if args.overlay:
            write_overlay(clip, records, out_dir / f"{clip.stem}_tracked.mp4", source.fps)
        visible = sum(1 for record in records if record["visible"])
        total_frames += len(records)
        total_visible += visible
        total_seconds += elapsed
        if not args.quiet:
            rate = len(records) / elapsed if elapsed else 0.0
            print(
                f"[{position}/{len(clips)}] {clip.name}: {len(records)} frames, "
                f"{visible} with a ball ({visible / max(len(records), 1):.0%}), {rate:.1f} fps"
            )
    share = total_visible / total_frames if total_frames else 0.0
    rate = total_frames / total_seconds if total_seconds else 0.0
    print(
        f"done: {total_frames} frames, {total_visible} detections ({share:.1%}), "
        f"{rate:.1f} fps, {failures} failed clip(s) -> {out_dir}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
