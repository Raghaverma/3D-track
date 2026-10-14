"""Causal, bounded-window inference using the checkpoint's training layout.

Create one StreamingTracker per video/stream and call reset() at clip boundaries.
Feature caching is only valid for an eval-mode shared encoder with fixed weights.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from typing import Tuple

import cv2
import numpy as np
import torch

from .data import CHANNEL_SCHEMES
from .metrics import decode_heatmaps, decode_softmax_null
from .model import HRNetConvLSTMBallTracker
from .motion import topk_peaks

_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def fp16_autocast(device: torch.device, enabled: bool):
    """Autocast context for fp16, or no context at all when fp32 is wanted.

    ``torch.autocast(device_type="cpu", dtype=torch.float16)`` raises on
    *construction*, before ``enabled=False`` can spare it, so a disabled
    fp16 autocast is not a no-op on CPU. Returning ``nullcontext()`` is both
    the fix and what fp32 means.
    """

    return torch.autocast(device_type=device.type, dtype=torch.float16) if enabled else nullcontext()


def prepare_frame(frame: np.ndarray, size: Tuple[int, int], motion: bool = False):
    """Resize once; normalize RGB and optionally retain raw grayscale motion."""
    height, width = size
    rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height),
                     interpolation=cv2.INTER_LINEAR)
    gray = (torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0)
            if motion else None)
    normalized = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
    return tensor, gray


class StreamingTracker:
    """One result per BGR frame, with coordinates in model input pixels.

    Null-head confidence is P(ball present), matching training's null gate;
    older heatmap models retain their sigmoid peak confidence. FP16 is opt-in
    on CUDA, and probability/coordinate decoding always runs in FP32.
    """

    def __init__(self, model, config: dict, device: torch.device,
                 image_size=None, precision="fp32", cache_features=True,
                 threshold=0.35, visibility_threshold=0.5, use_offsets=True):
        if model.training:
            raise ValueError("StreamingTracker requires model.eval()")
        self.model = model
        self.device = torch.device(device)
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision must be fp32 or fp16")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 inference requires CUDA; use fp32 on CPU")
        self.precision = precision
        self.image_size = tuple(image_size or (config.get("height", 288), config.get("width", 512)))
        if min(self.image_size) <= 0:
            raise ValueError("height and width must be positive")
        self.threshold, self.visibility_threshold = threshold, visibility_threshold
        self.use_offsets = use_offsets
        scheme_name = config.get("channel_scheme", "default")
        if scheme_name != "default" and scheme_name not in CHANNEL_SCHEMES:
            raise ValueError(f"unknown channel scheme: {scheme_name}")
        self.scheme = CHANNEL_SCHEMES.get(scheme_name)
        self.window_length = int(config.get("sequence_length", model.sequence_length))
        self.motion_channels = int(getattr(model, "motion_channels", 0))
        if self.scheme:
            indices = self.scheme["rgb"] + [i for pair in self.scheme["diffs"] for i in pair]
            if max(indices) >= self.window_length:
                raise ValueError("channel scheme references frames outside the configured window")
            if len(self.scheme["rgb"]) != model.sequence_length or len(self.scheme["diffs"]) != self.motion_channels:
                raise ValueError("channel scheme does not match the model's RGB/motion channels")
        elif self.window_length != model.sequence_length or self.motion_channels not in (0, self.window_length - 1):
            raise ValueError("checkpoint input layout is inconsistent; supply the training channel_scheme")
        if isinstance(model, HRNetConvLSTMBallTracker) and model.target_index % model.sequence_length != model.sequence_length - 1:
            raise ValueError("streaming inference requires a newest-frame target_index")
        self.cache_features = cache_features and hasattr(model, "forward_features")
        self.has_null = getattr(model, "null_head", None) is not None
        self.reset()

    def reset(self):
        """Forget all temporal history before a new video or a camera cut."""
        self.frames = deque(maxlen=self.window_length)
        self.grays = deque(maxlen=self.window_length)
        self.features = deque(maxlen=self.model.sequence_length)

    @staticmethod
    def _append(window, tensor):
        if not window:
            window.extend([tensor] * (window.maxlen - 1))
        window.append(tensor)

    @torch.inference_mode()
    def predict(self, frame: np.ndarray, top_k: int = 0) -> dict:
        if self.model.training:
            raise ValueError("cached inference cannot run in training mode")
        tensor, gray = prepare_frame(frame, self.image_size, bool(self.motion_channels))
        tensor = tensor.to(self.device)
        with fp16_autocast(self.device, self.precision == "fp16"):
            if self.cache_features:
                feature = self.model.encode(tensor[None, None])[0, 0]
                self._append(self.features, feature)
                output = self.model.forward_features(torch.stack(tuple(self.features))[None], self.image_size)
            else:
                self._append(self.frames, tensor)
                rgb_indices = self.scheme["rgb"] if self.scheme else range(self.window_length)
                frames = torch.stack([self.frames[i] for i in rgb_indices])[None]
                motion = None
                if gray is not None:
                    self._append(self.grays, gray.to(self.device))
                    pairs = (self.scheme["diffs"] if self.scheme else
                             [(i, i - 1) for i in range(1, self.window_length)])
                    motion = torch.stack([(self.grays[a] - self.grays[b]).abs() for a, b in pairs])[None]
                if self.has_null:
                    output = self.model(frames, motion, return_null=True)
                elif isinstance(self.model, HRNetConvLSTMBallTracker):
                    output = (self.model(frames), None)
                else:
                    output = self.model(frames, motion, return_visibility=True)
        logits = output[0][:, -1:].float()
        if self.has_null:
            offsets = output[2] if self.use_offsets else None
            positions, _, absent = decode_softmax_null(
                logits, output[1].float(), offsets, stride=self.model.stride, topk=max(1, top_k))
            coordinates = positions[:, 0]
            presence = 1.0 - absent
            confidence = presence
            candidates = positions[0] if top_k else coordinates.new_empty((0, 2))
        else:
            coordinates, confidence = decode_heatmaps(logits)
            presence = output[1].float().sigmoid().reshape(-1) if output[1] is not None else torch.ones_like(confidence)
            candidates = topk_peaks(logits, top_k)[0][0] if top_k else coordinates.new_empty((0, 2))
        # One device-to-host synchronization for the compact result, never a full heatmap.
        packed = torch.cat((coordinates[0], confidence[:1], presence[:1], candidates.flatten())).cpu().numpy()
        x, y, score, present = map(float, packed[:4])
        visible = present >= self.visibility_threshold and score >= self.threshold
        return {"x": x, "y": y, "confidence": score if present >= self.visibility_threshold else 0.0,
                "presence": present, "visible": visible, "candidates": packed[4:].reshape(-1, 2)}
