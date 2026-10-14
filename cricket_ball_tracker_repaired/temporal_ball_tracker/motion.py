"""Learned ball-motion model: selects the true ball among per-frame candidates.

The heatmap detector is deliberately good at appearance, which makes it
vulnerable to ball look-alikes. This module adds a second *trainable* stage
that understands how a ball moves instead of what it looks like: per frame the
detector proposes K candidate positions, and a small transformer receives only
those ``(x, y)`` coordinates over a window of frames. Appearance and detector
confidence are deliberately excluded from its input, so the only signal it can
learn is motion. Ball dynamics are learned from real ground-truth trajectories
rather than hand-coded (compare ``trajectory.py``'s fixed constant-velocity
gate); distractor dynamics are procedural models of the failure modes that
fool an appearance detector. Each frame is classified as one of its candidates
or as an explicit "no ball" token, so abstention is learned, not thresholded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import discover_tracknet_sequences


@torch.no_grad()
def topk_peaks(logits: Tensor, k: int, nms_radius: int = 2) -> Tuple[Tensor, Tensor]:
    """Extract the K strongest non-overlapping peaks from ``[B, 1, H, W]`` logits.

    Local maxima are isolated with max-pool non-maximum suppression (the
    CenterNet decoding trick) so the K candidates are distinct blobs rather
    than K pixels of one blob. Returns pixel positions ``[B, K, 2]`` as
    ``(x, y)`` and sigmoid scores ``[B, K]``, strongest first.
    """

    probabilities = torch.sigmoid(logits[:, 0])
    kernel = 2 * nms_radius + 1
    pooled = F.max_pool2d(probabilities.unsqueeze(1), kernel, stride=1, padding=nms_radius).squeeze(1)
    suppressed = torch.where(pooled == probabilities, probabilities, torch.zeros_like(probabilities))
    batch, height, width = suppressed.shape
    scores, flat = suppressed.flatten(1).topk(k, dim=-1)
    x = (flat % width).float()
    y = torch.div(flat, width, rounding_mode="floor").float()
    return torch.stack((x, y), dim=-1), scores


class TrackSelector(nn.Module):
    """Transformer that picks the ball candidate (or abstains) for every frame.

    Tokens are one per candidate per frame plus one learned "no ball" token per
    frame; candidates within a frame share the frame's time embedding and carry
    no slot embedding, so the model is permutation-invariant over candidate
    order and must associate candidates across frames purely by their motion.
    The output is a ``[B, T, K + 1]`` logit tensor scored per frame; index K is
    the abstention class.
    """

    def __init__(
        self,
        window: int = 15,
        candidates: int = 6,
        dim: int = 96,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.window = window
        self.candidates = candidates
        self.embed = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time_embedding = nn.Parameter(torch.randn(window, dim) * 0.02)
        self.null_token = nn.Parameter(torch.randn(dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.score = nn.Linear(dim, 1)

    def forward(self, positions: Tensor) -> Tensor:
        if positions.ndim != 4 or positions.size(-1) != 2:
            raise ValueError("expected positions shaped [batch, time, candidates, 2]")
        batch, steps, k, _ = positions.shape
        if steps != self.window or k != self.candidates:
            raise ValueError(f"expected window {self.window} x {self.candidates} candidates, received {steps} x {k}")
        tokens = self.embed(positions - 0.5)  # centre normalised coordinates
        null = self.null_token.expand(batch, steps, 1, -1)
        tokens = torch.cat((tokens, null), dim=2) + self.time_embedding[None, :, None, :]
        encoded = self.encoder(tokens.reshape(batch, steps * (k + 1), -1))
        return self.score(encoded).reshape(batch, steps, k + 1)


def build_selector(config: dict) -> TrackSelector:
    """Instantiate a ``TrackSelector`` from a checkpoint's selector config."""

    return TrackSelector(**config)


class SyntheticCandidateWindows(Dataset):
    """Candidate-selection training windows built from real ball trajectories.

    Every window contains the ground-truth ball track of a real rally segment
    (with simulated localisation noise and dropped detections) mixed with
    procedural distractors modelling how false positives behave over time:

    * ``static``  - fixed point with sub-pixel jitter (court marks, logos)
    * ``walk``    - slow random walk (people, shadows)
    * ``mover``   - smooth constant-velocity path with acceleration noise
      (limbs, birds), covering the same speed range as real ball motion
    * ``teleport``- independent random position per frame (per-frame detector
      errors with no temporal consistency)

    Candidate order is reshuffled every frame, so slot identity carries no
    information and cross-frame association must come from positions alone.
    Targets are the ball's slot index per frame, or K (abstain) when the ball
    is occluded or its detection was dropped. Horizontal mirroring and time
    reversal augment trajectory diversity; both preserve ballistic dynamics.
    """

    def __init__(
        self,
        root: str,
        window: int = 15,
        candidates: int = 6,
        drop_probability: float = 0.08,
        noise_pixels: float = 1.5,
        augment: bool = True,
    ):
        if candidates < 2:
            raise ValueError("candidates must be at least 2 (ball + one distractor)")
        self.window = window
        self.candidates = candidates
        self.drop_probability = drop_probability
        self.noise_pixels = noise_pixels
        self.augment = augment
        self.tracks: List[np.ndarray] = []
        self.samples: List[Tuple[int, int]] = []
        for sequence in discover_tracknet_sequences(root):
            image = cv2.imread(str(sequence.image_paths[0]))
            if image is None or len(sequence.labels) < window:
                continue
            source_height, source_width = image.shape[:2]
            track = np.array(
                [
                    (label.x / source_width, label.y / source_height, float(label.visible))
                    for label in sequence.labels
                ],
                dtype=np.float32,
            )
            track_index = len(self.tracks)
            self.tracks.append(track)
            self.samples.extend((track_index, start) for start in range(len(track) - window + 1))
        if not self.samples:
            raise ValueError(f"no sequences of at least {window} frames under {root}")
        # Normalised jitter scale: pixels are defined at the detector's 512-wide
        # input, matching the localisation error the selector will see live.
        self._noise = noise_pixels / 512.0

    def __len__(self) -> int:
        return len(self.samples)

    def _distractor(self, rng: np.random.Generator) -> np.ndarray:
        kind = rng.choice(4, p=(0.30, 0.25, 0.30, 0.15))
        length = self.window
        if kind == 0:  # static + jitter
            centre = rng.uniform(0.02, 0.98, size=2)
            return centre + rng.normal(0.0, rng.uniform(0.0005, 0.004), size=(length, 2))
        if kind == 1:  # random walk
            steps = rng.normal(0.0, rng.uniform(0.001, 0.006), size=(length, 2))
            return np.clip(rng.uniform(0.05, 0.95, size=2) + np.cumsum(steps, axis=0), 0.0, 1.0)
        if kind == 2:  # smooth mover, ball-like speeds
            angle = rng.uniform(0.0, 2.0 * np.pi)
            speed = rng.uniform(0.002, 0.06)
            velocity = np.array((np.cos(angle), np.sin(angle))) * speed
            steps = velocity + rng.normal(0.0, 0.001, size=(length, 2))
            return np.clip(rng.uniform(0.1, 0.9, size=2) + np.cumsum(steps, axis=0), 0.0, 1.0)
        return rng.uniform(0.0, 1.0, size=(length, 2))  # per-frame teleporter

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        rng = np.random.default_rng((torch.initial_seed() % (2**63), index))
        track_index, start = self.samples[index]
        segment = self.tracks[track_index][start : start + self.window].copy()
        if self.augment:
            if rng.random() < 0.5:
                segment = segment[::-1].copy()  # time reversal keeps ballistic motion valid
            if rng.random() < 0.5:
                segment[:, 0] = 1.0 - segment[:, 0]
            shift = rng.uniform(-0.05, 0.05, size=2)
            segment[:, :2] = segment[:, :2] + shift

        ball = segment[:, :2] + rng.normal(0.0, self._noise, size=(self.window, 2))
        has_ball = segment[:, 2] > 0.5
        has_ball &= rng.random(self.window) >= self.drop_probability  # simulate detector misses
        # Occluded or dropped frames still fill the slot with a junk candidate,
        # because the live detector always returns K peaks.
        junk = rng.uniform(0.0, 1.0, size=(self.window, 2))
        ball = np.where(has_ball[:, None], ball, junk)

        paths = [ball] + [self._distractor(rng) for _ in range(self.candidates - 1)]
        positions = np.stack(paths, axis=1).astype(np.float32)  # [T, K, 2]
        positions = np.clip(positions, 0.0, 1.0)

        targets = np.full(self.window, self.candidates, dtype=np.int64)  # default: abstain
        order = np.argsort(rng.random((self.window, self.candidates)), axis=1)
        positions = np.take_along_axis(positions, order[..., None], axis=1)
        ball_slot = np.argmin(order, axis=1)  # where slot 0 (the ball) landed
        targets[has_ball] = ball_slot[has_ball]
        return {
            "positions": torch.from_numpy(positions),
            "targets": torch.from_numpy(targets),
        }


class HarvestedCandidateWindows(Dataset):
    """Selection windows harvested from a real detector (``harvest_candidates``).

    Unlike ``SyntheticCandidateWindows`` the distractors here are the
    detector's actual secondary peaks, so the selector trains on the true
    joint distribution of ball motion and false-positive dynamics. The target
    for each frame is the candidate nearest the ground truth within
    ``radius_pixels`` (at the detector's input scale), else abstain. With
    ``augment`` the ball candidate is randomly dropped (replaced by a junk
    position) so the model also learns to abstain inside otherwise-real
    distractor fields; mirroring, time reversal, and a global shift preserve
    dynamics while expanding coverage.
    """

    def __init__(
        self,
        directory: str,
        window: int = 15,
        candidates: int = 6,
        radius_pixels: float = 8.0,
        drop_probability: float = 0.15,
        augment: bool = True,
        image_size: Tuple[int, int] = (288, 512),
    ):
        self.window = window
        self.candidates = candidates
        self.drop_probability = drop_probability
        self.augment = augment
        self._scale = np.array((image_size[1], image_size[0]), dtype=np.float32)  # (width, height)
        self._radius = radius_pixels
        self.sequences: List[Dict[str, np.ndarray]] = []
        self.samples: List[Tuple[int, int]] = []
        for path in sorted(Path(directory).glob("*.npz")):
            data = np.load(path)
            candidates_array = data["candidates"].astype(np.float32)
            if candidates_array.shape[1] != candidates:
                raise ValueError(f"{path} holds {candidates_array.shape[1]} candidates, expected {candidates}")
            if len(candidates_array) < window:
                continue
            record = {
                "candidates": candidates_array,
                "gt": data["gt"].astype(np.float32),
                "visible": data["visible"].astype(bool),
            }
            sequence_index = len(self.sequences)
            self.sequences.append(record)
            self.samples.extend((sequence_index, start) for start in range(len(candidates_array) - window + 1))
        if not self.samples:
            raise ValueError(f"no harvested sequences of at least {window} frames under {directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        rng = np.random.default_rng((torch.initial_seed() % (2**63), index))
        sequence_index, start = self.samples[index]
        record = self.sequences[sequence_index]
        positions = record["candidates"][start : start + self.window].copy()  # [T, K, 2]
        gt = record["gt"][start : start + self.window].copy()
        visible = record["visible"][start : start + self.window].copy()

        if self.augment:
            if rng.random() < 0.5:  # time reversal keeps ballistic motion valid
                positions, gt, visible = positions[::-1].copy(), gt[::-1].copy(), visible[::-1].copy()
            if rng.random() < 0.5:
                positions[..., 0] = 1.0 - positions[..., 0]
                gt[:, 0] = 1.0 - gt[:, 0]
            shift = rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
            positions = positions + shift
            gt = gt + shift

        # Label: nearest candidate within the pixel radius of the ground truth.
        distances = np.linalg.norm((positions - gt[:, None, :]) * self._scale, axis=-1)  # [T, K]
        nearest = distances.argmin(axis=1)
        matched = visible & (distances[np.arange(self.window), nearest] <= self._radius)
        targets = np.where(matched, nearest, self.candidates).astype(np.int64)

        if self.augment and self.drop_probability > 0:
            drop = matched & (rng.random(self.window) < self.drop_probability)
            if drop.any():
                frames = np.flatnonzero(drop)
                positions[frames, targets[frames]] = rng.uniform(0.0, 1.0, size=(len(frames), 2))
                targets[frames] = self.candidates
        positions = np.clip(positions, 0.0, 1.0)
        return {
            "positions": torch.from_numpy(positions.astype(np.float32)),
            "targets": torch.from_numpy(targets),
        }
