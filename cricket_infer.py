#!/usr/bin/env python3
"""Cricket delivery event spotting -- standalone, streaming, GPU-optimised.

    video -> ffmpeg (square frames at the model's own fps, piped as raw RGB)
          -> overlapping clips through E2E-Spot, scores averaged per frame
          -> per-class score-peak NMS
          -> delivery decoding under the release -> pitching -> impact order

The model emits three classes -- `release` (ball leaves the hand), `pitching`
(ball touches the pitch) and `impact` (ball meets bat or pad).  The last stage
is what separates this from three independent detectors: a delivery is a rigid
sequence with gaps this dataset measures at ~12 and ~5 frames (30 fps), so
decoding under that constraint recovers events whose own peak was too weak to
survive a threshold and rejects peaks that belong to no plausible delivery.

Two properties of the input are a hard contract with training and are read
from the model's own config.json / priors.json rather than from constants:
30 fps analysis rate, and a 256 px square with the *whole* frame squashed into
it (never centre-cropped).  Overriding either silently degrades the model
rather than adapting it.

Run it on one video, a directory, or a glob:

    python cricket_infer.py match.mp4
    python cricket_infer.py clips/ --out-dir out --csv
"""

import argparse
import bisect
import csv
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPOT_DIR = os.path.join(HERE, 'spot')
DEFAULT_MODEL_DIR = os.path.join(HERE, 'model')
VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm', '.mpg', '.mpeg',
              '.ts', '.m2ts')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The video dataset pads the head of every video with this many empty frames
# before the first clip, so the first real frame is never at a clip boundary.
# It is a property of the trained pipeline, not a tunable.
PAD_LEN = 5

# Gap bounds in frames at the analysis fps, used only when priors.json is
# missing.  The shipped model carries its own measured distribution.
FALLBACK_GAPS = {
    'release_to_pitching': {'min': 5, 'max': 60},
    'pitching_to_impact': {'min': 1, 'max': 40},
}


def log(msg, end='\n'):
    sys.stderr.write(msg + end)
    sys.stderr.flush()


class Progress:
    """Atomically rewrites a JSON file, for running this as a subprocess."""

    def __init__(self, path):
        self.path = path

    def update(self, stage, pct, msg=''):
        if not self.path:
            return
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fp:
            json.dump({'stage': stage, 'pct': round(pct, 1), 'msg': msg,
                       'updated': time.time()}, fp)
        os.replace(tmp, self.path)


# ---------------------------------------------------------------- video I/O

def probe_video(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate',
         '-show_entries', 'format=duration',
         '-of', 'json', path],
        capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    if not info.get('streams'):
        raise ValueError('%s has no video stream' % path)
    stream = info['streams'][0]

    def parse_rate(s):
        if not s or s == '0/0':
            return None
        num, _, den = s.partition('/')
        den = float(den) if den else 1.0
        return float(num) / den if den else None

    fps = parse_rate(stream.get('avg_frame_rate')) \
        or parse_rate(stream.get('r_frame_rate'))
    if not fps or fps <= 0 or fps > 1000:
        raise ValueError('Could not determine a sane fps for %s' % path)
    return {
        'width': int(stream['width']),
        'height': int(stream['height']),
        'fps': fps,
        'duration': float(info.get('format', {}).get('duration', 0) or 0),
    }


def ffmpeg_filter(frame_size, target_fps):
    """The one filter chain both pipelines use.

    Must match the dataset builder exactly: fps normalisation first, then the
    whole frame squashed to a square -- never a centre crop, because a release
    happens anywhere across the width of the frame.
    """
    vf = 'scale=%d:%d' % (frame_size, frame_size)
    if target_fps:
        vf = 'fps=%s,' % target_fps + vf
    return vf


def _ffmpeg_pipe(video_path, frame_size, target_fps, threads, tail_args):
    """ffmpeg decoding to the analysis geometry, writing to stdout."""
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin']
    if threads:
        cmd += ['-threads', str(threads)]
    cmd += ['-i', video_path, '-vf', ffmpeg_filter(frame_size, target_fps)]
    cmd += tail_args + ['-']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=1 << 20)


def _finish(proc, drained):
    """Reap ffmpeg; a non-zero exit only matters if we read to the end."""
    if not drained and proc.poll() is None:
        proc.terminate()                    # the consumer gave up early
    proc.stdout.close()
    err = proc.stderr.read().decode('utf-8', 'replace')
    proc.stderr.close()
    rc = proc.wait()
    if drained and rc:
        raise RuntimeError('ffmpeg failed (%s): %s' % (rc, err[-2000:]))


def mjpeg_frames(video_path, frame_size, target_fps, threads=0, quality=2):
    """Yield CxHxW uint8 frames, JPEG-encoded by ffmpeg and decoded in memory.

    This is the default because it is the only fast path that is *identical*
    to what the model was trained and validated on.  The training frames were
    written to disk as quality-2 JPEGs, and a JPEG round-trip is not a no-op:
    against frames piped as raw RGB it shifts every pixel by ~1% (mean |d|
    2.5/255 on test footage), which is enough to move a marginal peak's score
    by 0.2 and change which candidate the delivery decoder picks.  Encoding in
    ffmpeg and decoding here reproduces the training input byte for byte while
    keeping the frames off disk and out of a DataLoader.
    """
    import torch
    import torchvision
    proc = _ffmpeg_pipe(video_path, frame_size, target_fps, threads,
                        ['-qscale:v', str(quality), '-f', 'image2pipe',
                         '-vcodec', 'mjpeg'])
    buf = b''
    drained = False
    try:
        while True:
            chunk = proc.stdout.read(1 << 18)
            if not chunk:
                drained = True
                break
            buf += chunk
            # An MJPEG stream is JPEGs back to back.  Inside entropy-coded
            # data every 0xFF is stuffed (0xFF 0x00) or a restart marker, so
            # 0xFFD9 appears only as a real end-of-image.
            while True:
                end = buf.find(b'\xff\xd9')
                if end < 0:
                    break
                frame, buf = buf[:end + 2], buf[end + 2:]
                yield torchvision.io.decode_jpeg(
                    torch.frombuffer(bytearray(frame), dtype=torch.uint8))
        while True:
            end = buf.find(b'\xff\xd9')
            if end < 0:
                break
            frame, buf = buf[:end + 2], buf[end + 2:]
            yield torchvision.io.decode_jpeg(
                torch.frombuffer(bytearray(frame), dtype=torch.uint8))
    finally:
        _finish(proc, drained)


def raw_frames(video_path, frame_size, target_fps, threads=0):
    """Yield CxHxW uint8 frames as raw RGB, with no JPEG round-trip.

    Faster than `mjpeg_frames` by the cost of an encode and a decode, but the
    frames are *cleaner* than the ones the model was trained on -- see the
    note there.  Offered for throughput on footage where you have checked that
    the difference does not matter.
    """
    import numpy as np
    import torch
    nbytes = frame_size * frame_size * 3
    proc = _ffmpeg_pipe(video_path, frame_size, target_fps, threads,
                        ['-f', 'rawvideo', '-pix_fmt', 'rgb24'])
    drained = False
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                drained = True
                break
            arr = np.frombuffer(buf, np.uint8).reshape(
                frame_size, frame_size, 3)
            yield torch.from_numpy(arr.copy()).permute(2, 0, 1)
    finally:
        _finish(proc, drained)


def extract_jpegs(video_path, frame_dir, frame_size, target_fps):
    """The original on-disk pipeline, kept for parity checking."""
    os.makedirs(frame_dir, exist_ok=True)
    proc = subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
         '-i', video_path, '-vf', ffmpeg_filter(frame_size, target_fps),
         '-qscale:v', '2', '-start_number', '0',
         os.path.join(frame_dir, '%06d.jpg')],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError('ffmpeg failed: %s' % proc.stderr[-2000:])
    num_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if num_frames == 0:
        raise RuntimeError('No frames extracted from %s' % video_path)
    return num_frames


def snapshot(video_path, out_path, seconds):
    """A full-resolution JPEG of the source video at an event instant."""
    subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
         '-ss', '%.3f' % max(seconds, 0), '-i', video_path,
         '-frames:v', '1', '-qscale:v', '2', out_path],
        check=True, capture_output=True)


# ------------------------------------------------------------------- model

def checkpoint_epoch(ckpt_name):
    """Epoch number out of 'checkpoint_023.pt', or None if it is not one."""
    stem = os.path.splitext(os.path.basename(ckpt_name))[0]
    _, _, digits = stem.rpartition('_')
    return int(digits) if digits.isdigit() else None


def pick_checkpoint(model_dir, epoch=None):
    """The promoted checkpoint, not the last one.

    Training keeps every epoch, so the highest-numbered file is merely where
    training stopped -- usually past the point where validation mAP peaked.
    A promoted model directory holds the one chosen epoch, so the single-file
    case is the normal one; loss.json settles it otherwise.
    """
    ckpts = sorted(f for f in os.listdir(model_dir)
                   if f.startswith('checkpoint_') and f.endswith('.pt'))
    if not ckpts:
        raise RuntimeError('No checkpoint_*.pt in %s' % model_dir)
    if epoch is not None:
        name = 'checkpoint_{:03d}.pt'.format(epoch)
        if name not in ckpts:
            raise RuntimeError('%s not in %s' % (name, model_dir))
        return name
    if len(ckpts) == 1:
        return ckpts[0]
    loss_path = os.path.join(model_dir, 'loss.json')
    if os.path.exists(loss_path):
        try:
            with open(loss_path) as fp:
                losses = json.load(fp)
            scored = [l for l in losses if l.get('val_mAP')]
            if scored:
                best = max(scored, key=lambda l: l['val_mAP'])
                name = 'checkpoint_{:03d}.pt'.format(best['epoch'])
                if name in ckpts:
                    return name
        except (OSError, ValueError, KeyError):
            pass
    return ckpts[-1]


def model_name(model_dir):
    """The training run's name, so an output JSON says which model made it.

    The bundle ships the model under a generic directory, so the run name in
    promoted.json is the only thing that identifies the weights.
    """
    promoted = os.path.join(model_dir, 'promoted.json')
    if os.path.exists(promoted):
        try:
            with open(promoted) as fp:
                run_dir = json.load(fp).get('run_dir')
            if run_dir:
                return os.path.basename(os.path.normpath(run_dir))
        except (OSError, ValueError):
            pass
    return os.path.basename(os.path.normpath(model_dir))


def load_priors(model_dir):
    path = os.path.join(model_dir, 'priors.json')
    if not os.path.exists(path):
        return {}
    with open(path) as fp:
        return json.load(fp)


def resolve_classes(config, priors, spot_dir):
    """{class name: index}, indices starting at 1 as E2E-Spot expects.

    Prefer priors.json, which travels with the model directory, so a bundle of
    checkpoint + spot/ works with no dataset tree; fall back to the class.txt
    that training read.
    """
    names = priors.get('classes')
    if not names:
        class_txt = os.path.join(spot_dir, 'data', config.get('dataset', ''),
                                 'class.txt')
        if not os.path.exists(class_txt):
            raise RuntimeError('No class list: priors.json has no "classes" '
                               'and %s is missing' % class_txt)
        with open(class_txt) as fp:
            names = [line.strip() for line in fp if line.strip()]
    n_expected = config.get('num_classes')
    if n_expected and len(names) != n_expected:
        raise RuntimeError('config says %d classes, class list has %d: %s'
                           % (n_expected, len(names), names))
    return {name: i + 1 for i, name in enumerate(names)}


def load_model(model_dir, spot_dir, device, epoch=None, compile_model=False):
    sys.path.insert(0, spot_dir)
    # Every parameter comes from the checkpoint, so skip the ImageNet download
    # the backbone constructor would otherwise do -- this is what lets the
    # bundle run on a machine with no internet.
    os.environ.setdefault('E2E_SPOT_PRETRAINED', '0')
    import torch
    from train_e2e import E2EModel

    with open(os.path.join(model_dir, 'config.json')) as fp:
        config = json.load(fp)
    priors = load_priors(model_dir)
    classes = resolve_classes(config, priors, spot_dir)

    if device == 'cuda':
        # Ampere and later: TF32 matmuls cost nothing measurable in accuracy
        # here and the GRU head is matmul-bound.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = E2EModel(
        len(classes) + 1, config['feature_arch'], config['temporal_arch'],
        clip_len=config['clip_len'], modality=config['modality'],
        device=device, multi_gpu=False)
    ckpt = pick_checkpoint(model_dir, epoch)
    path = os.path.join(model_dir, ckpt)
    try:
        # The checkpoint is a plain state_dict, so weights_only is safe -- and
        # on torch >= 2.6 it is the default anyway.
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:       # torch < 1.13 has no weights_only
        state = torch.load(path, map_location=device)
    model.load(state)
    model._model.eval()
    if compile_model:
        try:
            model._model = torch.compile(model._model)
        except Exception as exc:                # noqa: BLE001
            log('torch.compile unavailable (%s); running eager' % exc)
    return model, classes, config, priors, ckpt


# --------------------------------------------------------- clip production
#
# A clip is `clip_len` consecutive analysis frames; clips start every
# `clip_len - overlap_len` frames beginning at -PAD_LEN, exactly as the
# training-time video dataset enumerated them.  Frames before the start and
# past the end of the video are zeros *after* normalisation, which is what the
# dataset's padding produced, so a clip at a boundary sees what the model saw
# in validation.

def clip_starts_after(num_frames, clip_len, stride, overlap_len, first):
    """Remaining clip starts once the video length is known."""
    starts = []
    start = first
    limit = max(0, num_frames - overlap_len)
    while start < limit:
        starts.append(start)
        start += stride
    return starts


def _assemble(buf, start, clip_len, frame_size, torch):
    """One clip as (uint8 [T,C,H,W], float32 valid-mask [T]).

    Frames outside the video stay zero and are masked *after* normalisation,
    which is exactly what the training-time dataset's padding produced.
    """
    arr = torch.zeros((clip_len, 3, frame_size, frame_size),
                      dtype=torch.uint8)
    mask = torch.zeros(clip_len, dtype=torch.float32)
    for t in range(clip_len):
        frame = buf.get(start + t)
        if frame is not None:
            arr[t] = frame
            mask[t] = 1.0
    return arr, mask


def iter_clip_batches(frames, clip_len, stride, overlap_len, frame_size,
                      batch_size):
    """Stream clips out of a frame iterator, in batches, in constant memory.

    Only the frames a pending clip still needs are held, so a two-hour match
    costs the same RAM as a five-second clip.  Yields
    (starts, uint8 [B,T,H,W,3], mask [B,T]) and finally the frame count.
    """
    import torch
    buf = {}
    clips, masks, starts = [], [], []
    next_start = -PAD_LEN
    count = 0

    def flush():
        batch = (list(starts), torch.stack(clips), torch.stack(masks))
        del clips[:], masks[:], starts[:]
        return batch

    for frame in frames:
        buf[count] = frame
        count += 1
        while next_start + clip_len <= count:
            arr, mask = _assemble(buf, next_start, clip_len, frame_size,
                                  torch)
            clips.append(arr)
            masks.append(mask)
            starts.append(next_start)
            next_start += stride
            for key in [k for k in buf if k < next_start]:
                del buf[key]
            if len(clips) == batch_size:
                yield flush()
    if count == 0:
        raise RuntimeError('no frames decoded')
    for start in clip_starts_after(count, clip_len, stride, overlap_len,
                                   next_start):
        arr, mask = _assemble(buf, start, clip_len, frame_size, torch)
        clips.append(arr)
        masks.append(mask)
        starts.append(start)
        if len(clips) == batch_size:
            yield flush()
    if clips:
        yield flush()
    yield count


# --------------------------------------------------------------- inference

def _crop_bounds(frame_size, crop_dim):
    """Centre-crop offsets, matching torchvision's rounding."""
    if not crop_dim or crop_dim >= frame_size:
        return 0, frame_size
    top = int(round((frame_size - crop_dim) / 2.0))
    return top, crop_dim


def run_inference_stream(model, classes, config, video_path, frame_size,
                         target_fps, device, batch_size, overlap_len,
                         progress, expected_frames, quiet, ffmpeg_threads,
                         pipeline='mjpeg'):
    """Per-frame class scores, averaged over overlapping clips.

    The decode runs in a producer thread and the GPU consumes batches off a
    queue, so ffmpeg and the model overlap instead of taking turns.  Frames
    cross the bus as uint8 -- a quarter of the bytes of float32 -- and are
    scaled and normalised on the device.
    """
    import numpy as np
    import torch

    clip_len = config['clip_len']
    stride = clip_len - overlap_len
    crop_dim = config.get('crop_dim') or frame_size
    top, side = _crop_bounds(frame_size, crop_dim)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
    use_cuda = device == 'cuda'

    work = queue.Queue(maxsize=3)
    state = {'frames': 0, 'error': None, 'decode': 0.0}

    def produce():
        started = time.time()
        try:
            producer = raw_frames if pipeline == 'raw' else mjpeg_frames
            frames = producer(video_path, frame_size, target_fps,
                              ffmpeg_threads)
            for item in iter_clip_batches(frames, clip_len, stride,
                                          overlap_len, frame_size,
                                          batch_size):
                if isinstance(item, int):
                    state['frames'] = item
                    break
                work.put(item)
        except Exception as exc:                # noqa: BLE001
            state['error'] = exc
        finally:
            state['decode'] = time.time() - started
            work.put(None)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()

    # The frame count is only known when the decode finishes, so accumulate
    # into a growing list of blocks and lay them into one array at the end.
    num_classes = len(classes) + 1
    chunks = []
    infer_seconds = 0.0
    done_clips = 0
    expected_clips = max(1, int(expected_frames / max(stride, 1)) + 1)
    while True:
        item = work.get()
        if item is None:
            break
        starts, clips_u8, masks = item
        started = time.time()
        if use_cuda:
            clips_u8 = clips_u8.pin_memory()
            masks = masks.pin_memory()
        x = clips_u8.to(device, non_blocking=use_cuda)
        if side != frame_size:
            x = x[:, :, :, top:top + side, top:top + side]
        # uint8 crosses the bus (a quarter of the bytes) and is scaled and
        # normalised here, in the same order the training dataset did it.
        x = x.float().div_(255.0).sub_(mean).div_(std)
        m = masks.to(device, non_blocking=use_cuda)
        x = x.mul_(m[:, :, None, None, None])
        _, scores = model.predict(x, use_amp=use_cuda)
        infer_seconds += time.time() - started
        for i, start in enumerate(starts):
            chunks.append((start, scores[i]))
        done_clips += len(starts)
        progress.update('detecting',
                        min(done_clips / expected_clips * 100, 99.0),
                        'clip %d' % done_clips)
        if not quiet:
            log('\r  clips %d' % done_clips, end='')
    thread.join()
    if not quiet:
        log('')
    if state['error']:
        raise state['error']

    num_frames = state['frames']
    scores = np.zeros((num_frames, num_classes), np.float32)
    support = np.zeros(num_frames, np.int32)
    for start, pred in chunks:
        if start < 0:                       # left-padded clip at the head
            pred = pred[-start:, :]
            start = 0
        end = min(start + pred.shape[0], num_frames)
        pred = pred[:end - start, :]
        scores[start:end, :] += pred
        support[start:end] += 1
    if num_frames == 0 or support.min() <= 0:
        raise RuntimeError('some frames received no prediction')
    scores /= support[:, None]
    return scores, num_frames, state['decode'], infer_seconds


def run_inference_jpeg(model, classes, config, video_path, frame_size,
                       target_fps, device, batch_size, overlap_len, workers,
                       progress, quiet):
    """The original pipeline: JPEGs on disk read back by the vendored dataset.

    Kept because it is the reference the shipped accuracy was measured on --
    `--pipeline jpeg` re-runs it so the streaming path can be checked against
    it on your own footage.
    """
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset.frame import ActionSpotVideoDataset

    work_dir = tempfile.mkdtemp(prefix='cricket_frames_')
    frame_dir = os.path.join(work_dir, 'video')
    decode_started = time.time()
    try:
        num_frames = extract_jpegs(video_path, frame_dir, frame_size,
                                   target_fps)
        decode_seconds = time.time() - decode_started
        stub = [{'video': 'video', 'num_frames': num_frames,
                 'fps': target_fps, 'width': 0, 'height': 0,
                 'events': [], 'num_events': 0}]
        stub_path = os.path.join(work_dir, 'stub.json')
        with open(stub_path, 'w') as fp:
            json.dump(stub, fp)
        dataset = ActionSpotVideoDataset(
            classes, stub_path, work_dir, config['modality'],
            config['clip_len'], overlap_len=overlap_len,
            crop_dim=config['crop_dim'])
        scores = np.zeros((num_frames, len(classes) + 1), np.float32)
        support = np.zeros(num_frames, np.int32)
        loader = DataLoader(dataset, num_workers=workers,
                            pin_memory=(device == 'cuda'),
                            batch_size=batch_size)
        started = time.time()
        for batch_idx, clip in enumerate(loader):
            _, batch_scores = model.predict(clip['frame'],
                                            use_amp=(device == 'cuda'))
            for i in range(clip['frame'].shape[0]):
                pred = batch_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred = pred[-start:, :]
                    start = 0
                end = min(start + pred.shape[0], num_frames)
                pred = pred[:end - start, :]
                scores[start:end, :] += pred
                support[start:end] += 1
            progress.update('detecting',
                            (batch_idx + 1) / max(len(loader), 1) * 100,
                            'clip batch %d' % (batch_idx + 1))
            if not quiet:
                log('\r  clip batch %d / %d' % (batch_idx + 1, len(loader)),
                    end='')
        if not quiet:
            log('')
        infer_seconds = time.time() - started
        if support.min() <= 0:
            raise RuntimeError('some frames received no prediction')
        scores /= support[:, None]
        return scores, num_frames, decode_seconds, infer_seconds
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ------------------------------------------------------------ peak picking

def find_peaks(scores, classes, min_score, nms_window):
    """Per-class local maxima above min_score, NMS'd within nms_window."""
    import numpy as np
    inv = {v: k for k, v in classes.items()}
    argmax = np.argmax(scores, axis=1)

    events = []
    for cls_idx in range(1, scores.shape[1]):
        s = scores[:, cls_idx]
        # Sentinel -1 outside the video, so the first and last frames can peak.
        padded = np.concatenate(([-1.0], s, [-1.0])).astype(np.float32)
        is_peak = (s >= min_score) & (s >= padded[:-2]) & (s >= padded[2:])
        idx = np.flatnonzero(is_peak)
        peaks = sorted(((float(s[i]), int(i)) for i in idx),
                       key=lambda p: (-p[0], p[1]))
        kept = []
        for score, frame in peaks:
            if all(abs(frame - k) > nms_window for k in kept):
                kept.append(frame)
                events.append({
                    'frame': frame,
                    'label': inv[cls_idx],
                    'score': round(score, 4),
                    'argmax': bool(argmax[frame] == cls_idx),
                })
    events.sort(key=lambda e: (e['frame'], e['label']))
    return events


# -------------------------------------------------------- delivery decoding

def load_gap_bounds(priors, fps, slack):
    """Gap bounds in frames, rescaled if inference fps differs from training.

    Bounds come from the labelled distribution rather than a guess, but from
    its 5th/95th percentiles rather than its extremes: one mis-annotated clip
    (release and pitching a frame apart, which no 20 m pitch allows) would
    otherwise widen the bound enough to let a spurious peak outscore the true
    one.  --gap-slack widens both ends for unusual footage.
    """
    bounds = {k: dict(v) for k, v in FALLBACK_GAPS.items()}
    source = 'fallback'
    if priors:
        prior_fps = priors.get('fps') or fps
        rescale = fps / prior_fps if prior_fps else 1.0
        for name, stats in (priors.get('gaps_frames') or {}).items():
            if not stats or name not in bounds:
                continue
            lo = max(1, int(stats['p05'] * rescale * (1 - slack)))
            hi = max(lo + 1, int(round(stats['p95'] * rescale * (1 + slack))))
            bounds[name] = {'min': lo, 'max': hi,
                            'from_p05_p95': [stats['p05'], stats['p95']],
                            'observed_range': [stats['min'], stats['max']]}
        source = 'priors.json'
    return bounds, source


def _delivery_start(delivery):
    """The earliest event a delivery actually has.

    Not always the release: a delivery whose release was never visible starts
    at its pitching.
    """
    return delivery['release'] or delivery['pitching'] or delivery['impact']


def _finalise_delivery(delivery, fps, src_fps):
    """Everything derived from the event frames, in one place."""
    start_f = _delivery_start(delivery)['frame']
    impact_f = delivery['impact']['frame']
    delivery['start_time'] = round(start_f / fps, 3)
    delivery['end_time'] = round(impact_f / fps, 3)
    delivery['start_src_frame'] = int(round(start_f * src_fps / fps))
    delivery['end_src_frame'] = int(round(impact_f * src_fps / fps))
    gaps = {}
    if delivery['release'] and delivery['pitching']:
        gaps['release_to_pitching'] = (delivery['pitching']['frame']
                                       - delivery['release']['frame'])
    elif delivery['release']:
        gaps['release_to_impact'] = impact_f - delivery['release']['frame']
    if delivery['pitching']:
        gaps['pitching_to_impact'] = impact_f - delivery['pitching']['frame']
    delivery['flight_frames'] = gaps


def decode_deliveries(events, fps, src_fps, bounds, allow_missing_pitching,
                      min_delivery_score, allow_missing_release=True):
    """Group peaks into non-overlapping release -> pitching -> impact triples.

    Greedy over the strongest candidate delivery first: each peak is consumed
    by at most one delivery, and deliveries may not interleave.

    Two of the three may be missing, each at a score penalty and each flagged
    `"complete": false`:

    * **pitching**, because a full toss never pitches.
    * **release**, because plenty of cameras never see it.  Behind-the-arm
      footage puts the bowler's own body across the release, so the ball
      simply appears in flight, and a model trained on such footage is right
      to stay quiet there.  Requiring a release then threw the whole delivery
      away even with a 0.84 pitching and a 0.65 impact correctly ordered,
      which is a worse answer than an incomplete one.
    """
    by_label = {'release': [], 'pitching': [], 'impact': []}
    for event in events:
        if event['label'] in by_label:
            by_label[event['label']].append(event)

    rp = bounds['release_to_pitching']
    pi = bounds['pitching_to_impact']
    # A pitchless delivery still has to reach the bat in a plausible time.
    ri_max = rp['max'] + pi['max']
    ri_min = max(1, rp['min'])

    # Frame-sorted, so each release only looks at the impacts and pitchings
    # inside its own window.  A full match produces thousands of peaks per
    # class, where pairing everything with everything would not scale.
    for events_of_class in by_label.values():
        events_of_class.sort(key=lambda e: e['frame'])
    impact_frames = [e['frame'] for e in by_label['impact']]
    pitching_frames = [e['frame'] for e in by_label['pitching']]

    candidates = []
    for release in by_label['release']:
        lo = bisect.bisect_left(impact_frames, release['frame'] + ri_min)
        hi = bisect.bisect_right(impact_frames, release['frame'] + ri_max)
        for impact in by_label['impact'][lo:hi]:
            p_lo = bisect.bisect_left(pitching_frames,
                                      release['frame'] + rp['min'])
            p_hi = bisect.bisect_right(pitching_frames,
                                       release['frame'] + rp['max'])
            middle = [p for p in by_label['pitching'][p_lo:p_hi]
                      if pi['min'] <= impact['frame'] - p['frame'] <= pi['max']]
            if middle:
                pitching = max(middle, key=lambda p: p['score'])
                score = (release['score'] + pitching['score']
                         + impact['score']) / 3.0
            elif allow_missing_pitching:
                pitching = None
                # Two of three classes agreeing is weaker evidence than three.
                score = (release['score'] + impact['score']) / 3.0
            else:
                continue
            candidates.append((score, release, pitching, impact))

    # Releaseless deliveries: a pitching -> impact pair inside the same gap
    # bound.  Scored over three classes like everything else, so a complete
    # triple always outranks one of these and consumes the peaks first.
    #
    # The pitching must be the model's top class on its frame.  Without that
    # gate this path doubled the spurious deliveries on the test split (14 ->
    # 31) while recovering nothing there -- those clips are all front-on, so
    # every release is visible and every releaseless candidate is noise.  The
    # gate keeps the recovery on behind-the-arm footage, where the pitching
    # peak is unambiguous, and drops the rest.
    if allow_missing_release:
        for pitching in by_label['pitching']:
            if not pitching.get('argmax'):
                continue
            lo = bisect.bisect_left(impact_frames,
                                    pitching['frame'] + pi['min'])
            hi = bisect.bisect_right(impact_frames,
                                     pitching['frame'] + pi['max'])
            for impact in by_label['impact'][lo:hi]:
                candidates.append(((pitching['score'] + impact['score']) / 3.0,
                                   None, pitching, impact))

    candidates.sort(key=lambda c: (-c[0], (c[1] or c[2])['frame']))

    used = set()
    spans = []
    deliveries = []
    for score, release, pitching, impact in candidates:
        if score < min_delivery_score:
            continue
        parts = [p for p in (release, pitching, impact) if p]
        keys = {(p['label'], p['frame']) for p in parts}
        if keys & used:
            continue
        span = (parts[0]['frame'], impact['frame'])
        if any(span[0] <= s[1] and s[0] <= span[1] for s in spans):
            continue
        used |= keys
        spans.append(span)
        deliveries.append({
            'score': round(score, 4),
            'release': release,
            'pitching': pitching,
            'impact': impact,
            'complete': release is not None and pitching is not None,
        })

    deliveries.sort(key=lambda d: _delivery_start(d)['frame'])
    for i, delivery in enumerate(deliveries):
        delivery['index'] = i
        _finalise_delivery(delivery, fps, src_fps)
    return deliveries


# ------------------------------------------------------------------ driver

def spot_video(video_path, model, classes, config, priors, ckpt, args,
               analysis_fps, frame_size, device, progress):
    meta = probe_video(video_path)
    started = time.time()
    overlap_len = args.overlap if args.overlap >= 0 \
        else config['clip_len'] // 2
    expected_frames = max(int(meta['duration'] * (analysis_fps
                                                  or meta['fps'])), 1)
    progress.update('detecting', 0, os.path.basename(video_path))
    if args.pipeline == 'jpeg':
        scores, num_frames, decode_s, infer_s = run_inference_jpeg(
            model, classes, config, video_path, frame_size, analysis_fps,
            device, args.batch_size, overlap_len, args.workers, progress,
            args.quiet)
    else:
        scores, num_frames, decode_s, infer_s = run_inference_stream(
            model, classes, config, video_path, frame_size, analysis_fps,
            device, args.batch_size, overlap_len, progress, expected_frames,
            args.quiet, args.ffmpeg_threads, args.pipeline)

    fps = analysis_fps or meta['fps']
    progress.update('postprocess', 0)
    events = find_peaks(scores, classes, args.min_score, args.nms_window)
    bounds, bounds_from = load_gap_bounds(priors, fps, args.gap_slack)
    # A delivery is a release -> pitching -> impact triple.  A single-class
    # model has no sequence to constrain, so its peaks are the answer.
    can_decode = {'release', 'pitching', 'impact'} <= set(classes)
    deliveries = decode_deliveries(
        events, fps, meta['fps'], bounds, not args.no_partial_deliveries,
        args.min_delivery_score,
        not args.no_missing_release) if can_decode else []

    # Times and source-frame numbers, so a caller can seek the original video
    # without knowing the analysis rate.
    for event in events:
        event['time'] = round(event['frame'] / fps, 3)
        event['src_frame'] = int(round(event['frame'] * meta['fps'] / fps))

    elapsed = time.time() - started
    return {
        'version': 1,
        'video': {
            'path': os.path.abspath(video_path),
            'filename': os.path.basename(video_path),
            'fps': meta['fps'],
            'width': meta['width'],
            'height': meta['height'],
            'duration': round(meta['duration'], 3),
        },
        'analysis': {
            'fps': fps,
            'num_frames': num_frames,
            'frame_size': frame_size,
            'crop_dim': config.get('crop_dim'),
            'device': device,
            'pipeline': args.pipeline,
            'batch_size': args.batch_size,
            'clip_len': config['clip_len'],
            'overlap_len': overlap_len,
            'model': model_name(args.model_dir),
            'checkpoint': ckpt,
            'epoch': checkpoint_epoch(ckpt),
            'feature_arch': config.get('feature_arch'),
            'temporal_arch': config.get('temporal_arch'),
            'gap_bounds': bounds if can_decode else None,
            'gap_bounds_from': bounds_from if can_decode else None,
            'delivery_decode': can_decode,
            'nms_window': args.nms_window,
            'min_score': args.min_score,
            'min_delivery_score': args.min_delivery_score,
            'seconds': round(elapsed, 1),
            'decode_seconds': round(decode_s, 1),
            'infer_seconds': round(infer_s, 1),
            'frames_per_second': round(num_frames / elapsed, 1) if elapsed
            else None,
            'realtime_factor': round(meta['duration'] / elapsed, 2)
            if elapsed else None,
        },
        'classes': sorted(classes, key=classes.get),
        'events': events,
        'deliveries': deliveries,
    }


def write_csv(path, result):
    with open(path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['video', 'delivery', 'label', 'score',
                         'analysis_frame', 'time_s', 'source_frame'])
        for delivery in result['deliveries']:
            for label in ('release', 'pitching', 'impact'):
                event = delivery.get(label)
                if event:
                    writer.writerow([result['video']['filename'],
                                     delivery['index'], label, event['score'],
                                     event['frame'], event['time'],
                                     event['src_frame']])
        if not result['deliveries']:
            for event in result['events']:
                writer.writerow([result['video']['filename'], '',
                                 event['label'], event['score'],
                                 event['frame'], event['time'],
                                 event['src_frame']])


def describe(result):
    lines = []
    name = result['video']['filename']
    for delivery in result['deliveries']:
        parts = []
        for label in ('release', 'pitching', 'impact'):
            event = delivery.get(label)
            parts.append('%s %s' % (
                label[:4],
                '-' if not event else '%d (%.2fs, %.2f)'
                % (event['frame'], event['time'], event['score'])))
        lines.append('%s\tdelivery %d\t%s\t%s'
                     % (name, delivery['index'], '\t'.join(parts),
                        'complete' if delivery['complete'] else 'partial'))
    if not result['deliveries']:
        for event in result['events']:
            lines.append('%s\tpeak\t%s\tframe %d\t%.2fs\tsrc %d\t%.3f'
                         % (name, event['label'], event['frame'],
                            event['time'], event['src_frame'],
                            event['score']))
    return lines


def collect_videos(paths):
    found = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(VIDEO_EXTS):
                    found.append(os.path.join(path, name))
        elif any(ch in path for ch in '*?['):
            found.extend(sorted(glob.glob(path)))
        else:
            found.append(path)
    missing = [p for p in found if not os.path.isfile(p)]
    if missing:
        raise SystemExit('not a file: %s' % ', '.join(missing))
    if not found:
        raise SystemExit('no videos found')
    return found


def get_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('videos', nargs='+',
                    help='video files, directories, or globs')
    ap.add_argument('--model-dir', dest='model_dir', default=DEFAULT_MODEL_DIR,
                    help='promoted model directory (default: %(default)s)')
    ap.add_argument('--spot-dir', dest='spot_dir', default=DEFAULT_SPOT_DIR,
                    help='the vendored E2E-Spot package')
    ap.add_argument('--out-dir', dest='out_dir', default=None,
                    help='write <video>.events.json here '
                         '(default: alongside each video)')
    ap.add_argument('--out', default=None,
                    help='single output JSON path; with several videos it '
                         'holds a list of results')
    ap.add_argument('--csv', action='store_true',
                    help='also write a flat CSV next to the JSON')
    ap.add_argument('--snapshots', action='store_true',
                    help='save a full-resolution JPEG at each decoded event')
    ap.add_argument('--device', default='auto',
                    choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--epoch', type=int, default=None,
                    help='force a checkpoint epoch instead of the promoted '
                         'one')
    ap.add_argument('--pipeline', default='mjpeg',
                    choices=['mjpeg', 'raw', 'jpeg'],
                    help='mjpeg: JPEG frames piped from ffmpeg and decoded '
                         'in memory (default; same input the model was '
                         'trained on, no disk); raw: uncompressed frames, '
                         'faster but cleaner than training; jpeg: the '
                         'original on-disk pipeline, for parity checking')
    ap.add_argument('--target-fps', dest='target_fps', type=float,
                    default=-1.0,
                    help="analysis frame rate; it is a contract with "
                         "training, so the default (-1) reads it from the "
                         "model's priors.json (30 fps for this model)")
    ap.add_argument('--frame-size', dest='frame_size', type=int, default=0,
                    help='square frame size; 0 reads it from the model config')
    ap.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                    help='clips per forward pass.  1 is both the fastest and '
                         'the smallest: a clip is already 100 frames, so the '
                         'GPU is saturated at one clip, and larger batches '
                         'only add memory pressure (measured: 1840 frames/s '
                         'and 1.0 GB at 1, 1460 and 7.8 GB at 8)')
    ap.add_argument('--overlap', type=int, default=-1,
                    help='clip overlap in frames; -1 means half a clip (the '
                         'validated setting).  0 halves the compute at a '
                         'small accuracy cost')
    ap.add_argument('--workers', type=int, default=4,
                    help='dataloader workers, --pipeline jpeg only')
    ap.add_argument('--ffmpeg-threads', dest='ffmpeg_threads', type=int,
                    default=0, help='0 lets ffmpeg choose')
    ap.add_argument('--compile', dest='compile_model', action='store_true',
                    help='torch.compile the backbone (slow first video, '
                         'faster afterwards)')
    ap.add_argument('--nms-window', dest='nms_window', type=int, default=2,
                    help='per-class peak suppression window, in frames')
    ap.add_argument('--min-score', dest='min_score', type=float, default=0.02,
                    help='keep peaks above this; deliberately permissive '
                         'because the delivery decoder, not the threshold, '
                         'is what rejects noise')
    ap.add_argument('--min-delivery-score', dest='min_delivery_score',
                    type=float, default=0.05,
                    help='mean class score a decoded delivery must reach')
    ap.add_argument('--gap-slack', dest='gap_slack', type=float, default=0.25,
                    help='widen the measured gap bounds by this fraction')
    ap.add_argument('--no-partial-deliveries', dest='no_partial_deliveries',
                    action='store_true',
                    help='require a pitching event in every delivery')
    ap.add_argument('--no-missing-release', dest='no_missing_release',
                    action='store_true',
                    help='require a release event in every delivery')
    ap.add_argument('--progress-file', dest='progress_file', default=None,
                    help='JSON progress file, for running this as a '
                         'subprocess')
    ap.add_argument('--quiet', action='store_true')
    return ap.parse_args()


def main():
    args = get_args()
    for tool in ('ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise SystemExit('%s is not on PATH' % tool)
    if not os.path.isdir(args.model_dir):
        raise SystemExit('no model directory: %s' % args.model_dir)
    if not os.path.isfile(os.path.join(args.spot_dir, 'train_e2e.py')):
        raise SystemExit('no E2E-Spot package at %s (pass --spot-dir)'
                         % args.spot_dir)

    videos = collect_videos(args.videos)
    progress = Progress(args.progress_file)
    progress.update('loading_model', 0)

    sys.path.insert(0, args.spot_dir)
    import torch
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda but torch reports no CUDA')

    model, classes, config, priors, ckpt = load_model(
        args.model_dir, args.spot_dir, device, args.epoch, args.compile_model)

    # Training squashed each frame to a square and then cropped; reproduce the
    # square exactly and let the centre crop match crop_dim.
    frame_size = args.frame_size or config.get('frame_size') \
        or config.get('crop_dim') or 256
    if args.target_fps < 0:
        analysis_fps = float(priors.get('fps') or 0) or 30.0
        fps_from = 'priors.json' if priors.get('fps') else 'fallback'
    else:
        analysis_fps = args.target_fps if args.target_fps > 0 else 0
        fps_from = 'command line'
    if not args.quiet:
        log('model %s (%s, epoch %s) on %s'
            % (model_name(args.model_dir), ckpt, checkpoint_epoch(ckpt),
               device))
        log('classes %s | %g fps (%s) | %dpx square | %s pipeline'
            % (sorted(classes, key=classes.get), analysis_fps, fps_from,
               frame_size, args.pipeline))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for i, video in enumerate(videos):
        if not args.quiet:
            log('[%d/%d] %s' % (i + 1, len(videos), video))
        result = spot_video(video, model, classes, config, priors, ckpt, args,
                            analysis_fps, frame_size, device, progress)
        results.append(result)

        stem = os.path.splitext(os.path.basename(video))[0]
        out_root = args.out_dir or os.path.dirname(os.path.abspath(video))
        if not args.out:
            with open(os.path.join(out_root, stem + '.events.json'), 'w') as fp:
                json.dump(result, fp, indent=2)
        if args.csv:
            write_csv(os.path.join(out_root, stem + '.events.csv'), result)
        if args.snapshots:
            for delivery in result['deliveries']:
                for label in ('release', 'pitching', 'impact'):
                    event = delivery.get(label)
                    if event:
                        snapshot(video, os.path.join(
                            out_root, '%s_d%02d_%s_%06d.jpg'
                            % (stem, delivery['index'], label,
                               event['frame'])), event['time'])
        for line in describe(result):
            print(line)
        if not args.quiet:
            analysis = result['analysis']
            log('  %d deliveries, %d peaks in %.1fs '
                '(%.0f frames/s, %.1fx realtime; decode %.1fs, gpu %.1fs)'
                % (len(result['deliveries']), len(result['events']),
                   analysis['seconds'], analysis['frames_per_second'] or 0,
                   analysis['realtime_factor'] or 0,
                   analysis['decode_seconds'], analysis['infer_seconds']))

    if args.out:
        with open(args.out, 'w') as fp:
            json.dump(results if len(results) > 1 else results[0], fp,
                      indent=2)
        if not args.quiet:
            log('wrote %s' % args.out)
    progress.update('done', 100, '%d videos' % len(results))


if __name__ == '__main__':
    main()
