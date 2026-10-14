# Cricket ball tracker — serving bundle

Self-contained inference for the **`all_annotations_13ch_repaired`** checkpoint:
a 13-channel `StackedHRNetBallTracker` (HRNet-W18, 3 stacked RGB frames + 4
grayscale motion differences) trained on every finished clip in every
annotation project — 2,090 clips / 432k frames across 20 projects.

Copy or unzip this folder anywhere. Nothing outside it is needed except the
Python packages in `requirements.txt`.

```bash
python selftest.py                                  # check the environment first
python track_ball.py --input clip.mp4               # -> out/clip.json
python track_ball.py --input clips/ --overlay --csv  # every video in a folder
```

## What is in here

| Path | What it is |
|---|---|
| `track_ball.py` | the inference CLI — videos, frame directories, or folders of either |
| `selftest.py` | tracks a synthetic clip; run it first on a new machine |
| `run_examples.sh` | worked command lines for the common cases |
| `model/serving.pt` | epoch 9 EMA weights, 44.8 MB (optimizer/scheduler/raw copy stripped) |
| `model/config.json` | the training config, for reference — `serving.pt` carries its own copy |
| `temporal_ball_tracker/` | the inference half of the training package: `model`, `runtime`, `metrics`, `data` (channel schemes), `motion` (peak decode), `trajectory` |

Regenerate the stripped checkpoint from a training run with
`tools/export_serving_checkpoint.py` in the parent project.

## Usage

```
--input PATH            video file, image-sequence directory, or a directory of
                        either; repeatable. A directory holding images is one
                        clip; a directory holding videos is a batch.
--out-dir DIR           where JSON/CSV/overlay land (default: out)
--overlay               also write <stem>_tracked.mp4 with the ball circled
--csv                   also write <stem>.csv in TrackNet Label.csv column order
--fps N                 frame rate for image sequences (default 30) and overlays
--recursive             search input directories for videos recursively
--device cpu|cuda|cuda:N   default: cuda when available
--precision fp16        CUDA only; measured *slower* than fp32 here (see Speed)
--weights ema|raw       ema (default) is the weight copy validation selected;
                        `raw` needs an unstripped checkpoint (this bundle ships
                        EMA only) passed via --checkpoint
--threshold F           peak-score gate (default 0.10, calibrated — see below)
--visibility-threshold F   presence gate (default 0.5)
--trajectory            constant-velocity Kalman gate; inconsistent detections
                        become abstentions instead of jumps
--max-frames N          stop early, for smoke tests
--overwrite             re-run clips whose JSON already exists
```

Inference is **causal and streaming**: frame N uses frames N-2..N only, so memory
is bounded regardless of clip length and a file gives the same answer a live feed
would. One `StreamingTracker` is built per clip, so temporal state never leaks
across clip boundaries.

## Output

`out/<stem>.json` is a flat list, one record per frame, coordinates in **source
video pixels**:

```json
{"frame": 52, "time_s": 1.04, "x": 457.4, "y": 274.1,
 "confidence": 0.69, "presence": 0.998, "visible": true}
```

- `visible` — the frame's verdict after both gates (and the trajectory gate if on).
  This is the field to read; a consumer that ignores it will use coasted or
  rejected positions.
- `confidence` — heatmap peak score, 0 when the presence gate rejects the frame.
- `presence` — P(ball present) from the visibility head.
- `file_name` — present for image sequences, so results rejoin the source frames.

`out/<stem>.meta.json` records the source, sizes, fps, checkpoint, gates and
wall-clock for the run.

Frames are resized to 512x288 **without letterboxing**, exactly as in training
(the dataset is 16:9 at 1024x576). A 4:3 or portrait source is squeezed, not
padded; results on one will be worse than the numbers below.

## Operating point

Measured with `tools/calibrate_threshold.py` on 7,403 frames from 40 val clips
(2,303 labelled visible), through this same streaming path:

```
 thresh  accepted   recall  precision  fp_rate  px_err
   0.00    28.6%    74.6%      81.2%     6.7%    0.48
   0.10    27.7%    74.1%      83.3%     5.9%    0.47   <- default
   0.20    26.7%    73.0%      85.1%     5.2%    0.46
   0.35    25.7%    71.7%      86.8%     4.6%    0.45
   0.50    23.7%    68.6%      90.0%     3.3%    0.44
```

recall = labelled-visible frames localised within 8 px (512x288), px_err = median
error on accepted labelled frames.

Two things follow, and they matter when comparing against older checkpoints:

1. **The visibility head is the real gate, not the peak score.** Median presence
   is 0.998 on labelled-visible frames and 0.009 elsewhere. At `--threshold 0`
   only 28.6% of frames are accepted, all of it the presence head's doing. Raise
   `--visibility-threshold` (0.7 → precision 86.4%, recall 72.3%) to trade recall
   for precision; the peak threshold only trims a weak tail.
2. **0.35 is the wrong default for this checkpoint.** The penalty-reduced focal
   objective (`OBJECTIVE_REPAIR.md`) produces sharp low-amplitude peaks instead
   of the saturated discs the old focal loss rewarded, so the threshold that
   suited `stacked_cricket_13ch` costs 2.4 pp of recall here for precision the
   presence gate already provides.

**Localisation is not the failure mode.** Median error on accepted frames is
0.47 px and p90 is 1.5 px at 512x288 — when it finds the ball it is exact.
The 26% it misses is mode selection (it locks onto something else) or frames the
detector never sees, as measured in `OBJECTIVE_REPAIR.md` §1.

`fp_rate` is **pessimistic by construction**: 45% of the labelled span in the
training data carries no label at all, so many "false positives" on unlabelled
frames are correct detections the labels cannot score. Use it to compare
thresholds, never as an absolute error rate.

## Speed

L4 (shared with other services), 1920x1080 source, batch of one, end to end
including video decode and JSON:

| device | precision | fps | detections |
|---|---|---|---|
| CUDA | fp32 | 18.5–23.5 | baseline |
| CUDA | fp16 | 15.0–15.6 | identical (217/217 on the bench clip) |
| CPU | fp32 | 4.6 | — |

**fp16 is consistently slower here, not faster** — a 512x288 input at batch 1 is
launch-latency bound, not arithmetic bound, and autocast adds per-op casts. It
is kept for parity with the training path and for GPUs where it does pay;
measure on yours before enabling it. The fp32 spread is GPU contention, not clip
content.

## Requirements

`pip install -r requirements.txt`. Verified on torch 2.1.2+cu118 / timm 1.0.27 /
OpenCV 4.5.4 / numpy 1.26.4 (the `mmpose_prod` conda env). `timm` is required —
the HRNet-W18 backbone is built through it — but no weights are downloaded at
inference time: `pretrained_backbone` is forced off because the checkpoint
already carries the backbone.

## Provenance

- Training run: `updated_ball_tracker/checkpoints/all_annotations_13ch_repaired/`
  (`.log` beside it), 12 epochs configured, best at **epoch 9**, val within-2 0.840.
- Recipe: `train_repaired_objective.sh` — penalty-reduced focal loss + peak
  ranking loss + heatmap prior 0.01, with train-only label repair
  (`--gap-fill 4 --ignore-gaps --ignore-edge-frames 15`).
- Why that recipe: `OBJECTIVE_REPAIR.md`. Dataset: `ALL_ANNOTATIONS.md`.
- The unstripped checkpoint (with `raw_model` and optimizer state) stays in the
  training directory; this bundle ships the EMA weights only.
