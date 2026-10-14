"""Spatial-temporal heatmap model for small-ball tracking.

The network deliberately keeps the spatial encoder and temporal model separate:
every RGB frame passes through *the same* HRNet-style encoder, and only then is
the sequence passed to ConvLSTM.  This avoids treating time as colour channels.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    """A 3x3 convolution followed by batch normalisation and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    """Small residual block used by the high-resolution branches."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.conv2(self.conv1(x)))


class HRNetStage(nn.Module):
    """A two-resolution HRNet stage with bidirectional feature fusion."""

    def __init__(self, high_channels: int, low_channels: int, blocks: int = 2):
        super().__init__()
        self.high = nn.Sequential(*[BasicBlock(high_channels) for _ in range(blocks)])
        self.low = nn.Sequential(*[BasicBlock(low_channels) for _ in range(blocks)])
        self.high_to_low = ConvNormAct(high_channels, low_channels, stride=2)
        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, 1, bias=False),
            nn.BatchNorm2d(high_channels),
        )
        self.high_out = BasicBlock(high_channels)
        self.low_out = BasicBlock(low_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high: Tensor, low: Tensor) -> Tuple[Tensor, Tensor]:
        high = self.high(high)
        low = self.low(low)
        fused_high = high + F.interpolate(
            self.low_to_high(low), size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        fused_low = low + self.high_to_low(high)
        return self.high_out(self.relu(fused_high)), self.low_out(self.relu(fused_low))


class SharedHRNetEncoder(nn.Module):
    """Compact HRNet-style feature extractor shared by all sequence frames.

    It retains a 1/2-resolution stream throughout the network while fusing a
    1/4-resolution semantic stream at each stage. Keeping this branch at half
    resolution is important because a broadcast tennis ball is only a few input
    pixels across. Output is ``C x H/2 x W/2`` by default.
    """

    def __init__(
        self,
        feature_channels: int = 48,
        low_channels: int | None = None,
        output_stride: int = 2,
        dropout: float = 0.10,
        in_channels: int = 3,
    ):
        super().__init__()
        if output_stride not in (2, 4):
            raise ValueError("output_stride must be 2 or 4")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        low_channels = low_channels if low_channels is not None else feature_channels * 3 // 2
        self.output_stride = output_stride
        stem_channels = feature_channels // 2
        first_stride = 1 if output_stride == 2 else 2
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, stride=first_stride),
            ConvNormAct(stem_channels, feature_channels, stride=2),
            BasicBlock(feature_channels),
        )
        self.to_low = ConvNormAct(feature_channels, low_channels, stride=2)
        self.stage1 = HRNetStage(feature_channels, low_channels)
        self.stage2 = HRNetStage(feature_channels, low_channels)
        self.project = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, frame: Tensor) -> Tensor:
        high = self.stem(frame)
        low = self.to_low(high)
        high, low = self.stage1(high, low)
        high, _ = self.stage2(high, low)
        return self.project(high)


# Backbone id -> timm model id. Every entry must expose a stride-4 pyramid
# level; all three HRNet variants publish an identical one (index 1, 128
# channels), so swapping between them changes cost but not the feature contract
# the heads consume.
#
# The list is short on purpose. HRNet is the only common family that keeps a
# *wide* high-resolution branch: it fuses the semantic streams back into the
# 1/4-scale branch at every stage, so stride 4 carries 128 channels. Ordinary
# classification backbones spend their depth at stride 16/32 and expose only
# 24-64 channels at stride 4 -- too thin for a ball that is 3-6 pixels across.
# Adding one of those here would need an FPN to rebuild high-resolution width,
# which is a different encoder, not a drop-in.
TIMM_BACKBONES: Dict[str, str] = {
    "hrnet_w18": "hrnet_w18.ms_in1k",
    "hrnet_w18_small": "hrnet_w18_small.gluon_in1k",
    "hrnet_w18_small_v2": "hrnet_w18_small_v2.gluon_in1k",
}

# Indexes in the timm feature pyramid, shared by the family: level 1 is the
# stride-4 high-resolution branch (128 channels), level 0 the stride-2 stem
# output (64 channels).
_HRNET_STRIDE4_INDEX = 1
_HRNET_STRIDE2_INDEX = 0


class TimmHRNetW18Encoder(nn.Module):
    """ImageNet-pretrained standard HRNet feature encoder.

    The selected feature map is HRNet's high-resolution (1/4 input stride)
    branch. It is projected to the feature width used by the heads, while all
    backbone stages retain their pretrained weights.

    ``model_name`` picks a width/depth point from ``TIMM_BACKBONES``. All of
    them emit 128 channels at stride 4, so they are interchangeable without
    touching the heads; ``hrnet_w18_small`` is ~4x cheaper than ``hrnet_w18``
    for the same output shape.

    ``hires_lateral`` adds the pyramid's **stride-2** level (64 channels, the
    stem output) as an FPN-style lateral, so the encoder emits stride 2 instead
    of stride 4. For a ball 2-3 px across that resolution is the whole game: at
    stride 4 the object is a quarter of one feature cell, and everything below
    cell resolution has to be reinvented by a 49k-parameter transposed-conv
    head. The lateral is shallow (one pretrained conv) but high-frequency, which
    is exactly what the head cannot synthesise. Cost is one 1x1 conv plus a
    bilinear upsample; the deep semantic path is unchanged.
    """

    output_stride = 4

    def __init__(
        self,
        feature_channels: int = 48,
        pretrained: bool = False,
        dropout: float = 0.10,
        in_channels: int = 3,
        model_name: str = "hrnet_w18",
        hires_lateral: bool = False,
    ):
        super().__init__()
        if model_name not in TIMM_BACKBONES:
            raise ValueError(
                f"unknown timm backbone {model_name!r}; expected one of {sorted(TIMM_BACKBONES)}"
            )
        try:
            import timm
        except ImportError as error:  # pragma: no cover - dependency error is user-environment specific
            raise RuntimeError("HRNet requires timm; install the project requirements") from error
        # ``in_chans != 3`` (e.g. a 15+4 channel stacked window) keeps the
        # ImageNet weights: timm inflates the pretrained RGB stem across the
        # extra channels, so only the first convolution is adapted.
        self.model_name = model_name
        self.hires_lateral = hires_lateral
        out_indices = (
            (_HRNET_STRIDE2_INDEX, _HRNET_STRIDE4_INDEX) if hires_lateral
            else (_HRNET_STRIDE4_INDEX,)
        )
        self.backbone = timm.create_model(
            TIMM_BACKBONES[model_name],
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
        )
        info = self.backbone.feature_info
        channels = info.channels()
        reductions = info.reduction()
        deep_position = 1 if hires_lateral else 0
        if reductions[deep_position] != 4:
            raise RuntimeError(f"unexpected {model_name} high-resolution feature stride")
        self.project = nn.Sequential(
            nn.Conv2d(channels[deep_position], feature_channels, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        if hires_lateral:
            if reductions[0] != 2:
                raise RuntimeError(f"unexpected {model_name} stem feature stride")
            self.output_stride = 2
            self.lateral = nn.Sequential(
                nn.Conv2d(channels[0], feature_channels, 1, bias=False),
                nn.BatchNorm2d(feature_channels),
            )
            # A 3x3 after the sum, the standard FPN anti-aliasing conv: without
            # it the upsampled deep map and the sharp stem map stay visibly
            # superimposed rather than fused.
            self.fuse = nn.Sequential(
                ConvNormAct(feature_channels, feature_channels),
                nn.Dropout2d(dropout),
            )

    def forward(self, frame: Tensor) -> Tensor:
        features = self.backbone(frame)
        if not self.hires_lateral:
            return self.project(features[0])
        stem, deep = features
        deep = self.project(deep)
        deep = F.interpolate(deep, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(deep + self.lateral(stem))


class ConvLSTMCell(nn.Module):
    """A convolutional LSTM cell that preserves the feature-map layout."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding
        )
        # Gates are ordered (input, forget, output, cell). Biasing the forget
        # gate towards 1 lets the cell retain state early in training, which
        # noticeably speeds up convergence of the temporal model.
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        if state is None:
            h = x.new_zeros((x.size(0), self.hidden_channels, *x.shape[-2:]))
            c = x.new_zeros((x.size(0), self.hidden_channels, *x.shape[-2:]))
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """Single-layer ConvLSTM which returns one state for every input frame."""

    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)

    def forward(self, sequence: Tensor) -> Tensor:
        # sequence: [batch, frames, channels, height, width]
        state: Optional[Tuple[Tensor, Tensor]] = None
        outputs = []
        for frame_features in sequence.unbind(dim=1):
            h, c = self.cell(frame_features, state)
            state = (h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class HeatmapHead(nn.Module):
    """Upsamples features to ``out_channels`` input-resolution logit heatmaps.

    ``out_channels`` is one per supervised frame: it is 1 for last-frame-only
    training and equals the window length under multi-frame supervision.

    ``prior_probability`` sets the output bias so the untrained heatmap reads
    that probability everywhere instead of 0.5 -- the RetinaNet/CenterNet prior
    trick. It is *required* by any loss that sums over pixels and normalises by
    the number of positives (see ``PenaltyReducedFocalLoss``): with a zero bias
    the 147,456 background pixels each contribute ~0.17 at initialisation, which
    lands as ~1e5 of loss on the batch's handful of positives. It also removes
    the degenerate early descent direction of a pixel-mean loss, which is to
    raise the whole map before carving it back.
    """

    def __init__(
        self,
        in_channels: int,
        upsample_stages: int,
        out_channels: int = 1,
        prior_probability: float | None = None,
    ):
        super().__init__()
        mid_channels = max(in_channels // 2, 16)
        layers = [
            ConvNormAct(in_channels, in_channels),
        ]
        channels = in_channels
        for _ in range(upsample_stages):
            layers.extend(
                [
                    nn.ConvTranspose2d(channels, mid_channels, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = mid_channels
        output = nn.Conv2d(channels, out_channels, 1)
        if prior_probability is not None:
            if not 0.0 < prior_probability < 1.0:
                raise ValueError("prior_probability must be in (0, 1)")
            nn.init.constant_(
                output.bias, math.log(prior_probability / (1.0 - prior_probability))
            )
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, output_size: Tuple[int, int]) -> Tensor:
        logits = self.layers(x)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


class HRNetConvLSTMBallTracker(nn.Module):
    """Predict a ball heatmap from a window of RGB frames.

    Inputs have shape ``[B, 5, 3, H, W]`` by default. ``target_index=-1``
    predicts the newest frame, which supports streaming inference without future
    frames. For offline, centred windows use ``target_index=2``.
    """

    def __init__(
        self,
        sequence_length: int = 5,
        feature_channels: int = 48,
        hidden_channels: int = 48,
        low_channels: int | None = None,
        encoder_stride: int = 2,
        encoder_dropout: float = 0.10,
        backbone: str = "compact",
        pretrained_backbone: bool = False,
        target_index: int = -1,
        heatmap_prior: float | None = None,
    ):
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if not -sequence_length <= target_index < sequence_length:
            raise ValueError("target_index is outside the input sequence")
        self.sequence_length = sequence_length
        self.target_index = target_index
        self.backbone_name = backbone
        if backbone == "compact":
            self.encoder = SharedHRNetEncoder(
                feature_channels=feature_channels,
                low_channels=low_channels,
                output_stride=encoder_stride,
                dropout=encoder_dropout,
            )
            actual_stride = encoder_stride
        elif backbone in TIMM_BACKBONES:
            if encoder_stride != 4:
                raise ValueError("standard HRNet exposes its high-resolution branch at stride 4")
            self.encoder = TimmHRNetW18Encoder(
                feature_channels=feature_channels,
                pretrained=pretrained_backbone,
                dropout=encoder_dropout,
                model_name=backbone,
            )
            actual_stride = self.encoder.output_stride
        else:
            raise ValueError(
                f"backbone must be 'compact' or one of {sorted(TIMM_BACKBONES)}, got {backbone!r}"
            )
        self.temporal = ConvLSTM(feature_channels, hidden_channels)
        self.head = HeatmapHead(
            hidden_channels,
            upsample_stages=int(math.log2(actual_stride)),
            prior_probability=heatmap_prior,
        )

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.size(2) != 3:
            raise ValueError("expected frames shaped [batch, time, 3, height, width]")
        if frames.size(1) != self.sequence_length:
            raise ValueError(f"expected {self.sequence_length} frames, received {frames.size(1)}")
        batch, steps, channels, height, width = frames.shape
        features = self.encoder(frames.reshape(batch * steps, channels, height, width))
        features = features.reshape(batch, steps, *features.shape[1:])
        temporal_features = self.temporal(features)
        return self.head(temporal_features[:, self.target_index], (height, width))


class _VisibilityHead(nn.Module):
    """Ball-presence classifier over pooled encoder features.

    Uses concatenated global average- and max-pooling. Max-pooling is the load-
    bearing part: the ball occupies ~0.1% of pixels, so average-pooling washes
    its feature response out, while max-pooling preserves "is there a strong
    ball-like response anywhere" -- exactly the presence signal. Outputs one
    logit per sample (newest frame).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(channels * 2, 1)

    def forward(self, features: Tensor) -> Tensor:
        pooled = torch.cat(
            [self.avg_pool(features).flatten(1), self.max_pool(features).flatten(1)], dim=1
        )
        return self.classifier(pooled).reshape(-1)


class StackedHRNetBallTracker(nn.Module):
    """Early-fusion ball tracker: 5 stacked RGB frames -> HRNet -> heatmap.

    The window is concatenated on the channel axis (``5 x 3 = 15`` input
    channels), so the very first convolution fuses appearance and motion. This
    is the TrackNet/WASB-style design: temporal reasoning costs one wider stem
    instead of a recurrent loop, which trains and infers in a single parallel
    pass. The heatmap targets the final (newest) frame of the window, matching
    ``TrackNetWindowDataset``.

    Two options break the appearance-only shortcut a bare stacked model tends
    to learn. ``motion_channels`` appends per-step frame-difference maps to the
    input so the first convolution receives explicit velocity. ``predict_frames``
    emits one heatmap per input frame (multi-frame supervision); the newest
    frame's heatmap is always the last output channel, so downstream decoding is
    unchanged. Both default off for backward compatibility.
    """

    def __init__(
        self,
        sequence_length: int = 5,
        feature_channels: int = 48,
        low_channels: int | None = None,
        encoder_stride: int = 2,
        encoder_dropout: float = 0.10,
        motion_channels: int = 0,
        predict_frames: int = 1,
        backbone: str = "compact",
        pretrained_backbone: bool = False,
        predict_visibility: bool = False,
        heatmap_prior: float | None = None,
        hires_lateral: bool = False,
        predict_null: bool = False,
        predict_offsets: bool = False,
    ):
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if motion_channels < 0:
            raise ValueError("motion_channels must be non-negative")
        if predict_frames < 1:
            raise ValueError("predict_frames must be at least 1")
        self.sequence_length = sequence_length
        self.motion_channels = motion_channels
        self.predict_frames = predict_frames
        in_channels = 3 * sequence_length + motion_channels
        if backbone in TIMM_BACKBONES:
            # Standard ImageNet HRNet with the stem inflated to the stacked
            # window. Its high-resolution branch is stride 4; ``hires_lateral``
            # fuses the stride-2 stem level back in and moves the output to
            # stride 2, which is the only way to run this backbone at stride 2.
            expected_stride = 2 if hires_lateral else TimmHRNetW18Encoder.output_stride
            if encoder_stride != expected_stride:
                raise ValueError(
                    f"{backbone} emits stride {expected_stride}"
                    f"{' with --hires-lateral' if hires_lateral else ''}; "
                    f"use --encoder-stride {expected_stride}"
                )
            self.encoder: nn.Module = TimmHRNetW18Encoder(
                feature_channels=feature_channels,
                pretrained=pretrained_backbone,
                dropout=encoder_dropout,
                in_channels=in_channels,
                model_name=backbone,
                hires_lateral=hires_lateral,
            )
        elif backbone == "compact":
            self.encoder = SharedHRNetEncoder(
                feature_channels=feature_channels,
                low_channels=low_channels,
                output_stride=encoder_stride,
                dropout=encoder_dropout,
                in_channels=in_channels,
            )
        else:
            raise ValueError(f"unknown stacked backbone: {backbone!r}")
        self.head = HeatmapHead(
            feature_channels,
            upsample_stages=int(math.log2(encoder_stride)),
            out_channels=predict_frames,
            prior_probability=heatmap_prior,
        )
        # Dedicated presence head: the heatmap alone cannot represent "no ball"
        # (decoding always argmaxes a location), so this outputs an explicit
        # P(ball visible) for the newest frame, pooled from the same features.
        self.visibility_head = _VisibilityHead(feature_channels) if predict_visibility else None
        # ``predict_null`` swaps that pooled presence logit for one that competes
        # inside a softmax over ``H*W + 1`` (see SpatialSoftmaxNullLoss), and
        # ``predict_offsets`` adds the sub-cell regression that keeps localisation
        # from being capped by the decode grid. Both were built for the
        # shared-encoder architecture, but the measurements say the *head* is
        # where the headroom is: on the epoch-9 stacked model, post-pitch
        # failures are 10.0% gated out and 10.7% outranked against only 1.8%
        # never seen, so the encoder is not the binding constraint. Putting them
        # here gets that headroom at this architecture's cost -- 81.9 frames/s
        # against the shared encoder's 32.7 unstreamed.
        self.stride = encoder_stride
        self.null_head = (
            nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                          nn.Linear(feature_channels, 1))
            if predict_null else None
        )
        self.offset_head = nn.Conv2d(feature_channels, 2, 1) if predict_offsets else None

    def forward(
        self, frames: Tensor, motion: Optional[Tensor] = None, return_visibility: bool = False,
        return_null: bool = False,
    ):
        """Return heatmap logits. With ``return_visibility`` also return the
        newest-frame presence logit (``None`` if the model has no visibility
        head), as ``(heatmap, visibility)``. With ``return_null`` return
        ``(heatmap, null_logit, offsets)`` for the softmax-with-null objective.
        The default single-tensor return keeps every existing caller
        unchanged."""
        if frames.ndim != 5 or frames.size(2) != 3:
            raise ValueError("expected frames shaped [batch, time, 3, height, width]")
        if frames.size(1) != self.sequence_length:
            raise ValueError(f"expected {self.sequence_length} frames, received {frames.size(1)}")
        batch, steps, channels, height, width = frames.shape
        stacked = frames.reshape(batch, steps * channels, height, width)
        if self.motion_channels:
            if motion is None:
                raise ValueError("model expects motion channels but none were provided")
            if motion.size(1) != self.motion_channels:
                raise ValueError(
                    f"expected {self.motion_channels} motion channels, received {motion.size(1)}"
                )
            stacked = torch.cat((stacked, motion), dim=1)
        elif motion is not None:
            raise ValueError("motion channels supplied to a model configured without them")
        features = self.encoder(stacked)
        heatmap = self.head(features, (height, width))
        if return_null:
            null_logit = self.null_head(features).reshape(-1) if self.null_head is not None else None
            offsets = self.offset_head(features) if self.offset_head is not None else None
            return heatmap, null_logit, offsets
        if return_visibility:
            visibility = self.visibility_head(features) if self.visibility_head is not None else None
            # Presence is the combined probability of all newest-frame spatial
            # classes in the joint spatial/null softmax.
            if visibility is None and self.null_head is not None:
                visibility = (
                    torch.logsumexp(heatmap[:, -1].flatten(1), dim=1)
                    - self.null_head(features).reshape(-1)
                )
            return heatmap, visibility
        return heatmap


def temporal_shift(sequence: Tensor, fold_div: int = 8) -> Tensor:
    """TSM channel shift along time: a temporal receptive field for zero FLOPs.

    ``sequence`` is ``[batch, time, channels, height, width]``. The first
    ``C/fold_div`` channels are shifted one step towards the past and the next
    ``C/fold_div`` one step towards the future, with zero padding at the ends;
    the remaining channels are untouched. A plain 2-D convolution applied after
    this sees neighbouring frames, so no Conv3d is needed -- which also keeps the
    graph exportable to the NPU toolchains (see cricket_events/npu).
    """

    batch, steps, channels, height, width = sequence.shape
    fold = channels // fold_div
    if fold == 0 or steps < 2:
        return sequence
    shifted = torch.zeros_like(sequence)
    shifted[:, :-1, :fold] = sequence[:, 1:, :fold]            # from the future
    shifted[:, 1:, fold : 2 * fold] = sequence[:, :-1, fold : 2 * fold]  # from the past
    shifted[:, :, 2 * fold :] = sequence[:, :, 2 * fold :]
    return shifted


class TemporalShiftConv(nn.Module):
    """A shift followed by one shared 2-D convolution, applied per timestep."""

    def __init__(self, channels: int, fold_div: int = 8):
        super().__init__()
        self.fold_div = fold_div
        self.conv = ConvNormAct(channels, channels)

    def forward(self, sequence: Tensor) -> Tensor:
        shifted = temporal_shift(sequence, self.fold_div)
        batch, steps, channels, height, width = shifted.shape
        out = self.conv(shifted.reshape(batch * steps, channels, height, width))
        return out.reshape(batch, steps, channels, height, width)


class SharedEncoderMotionTracker(nn.Module):
    """Shared encoder, *signed* feature-space motion, softmax head with a null class.

    Every frame goes through the same encoder once, and motion is formed in
    feature space as signed differences ``F(t) - F(t-1)`` rather than in pixel
    space as ``|f(t) - f(t-1)|``. Four measured reasons, all from the 2026-09-02
    audit (see OBJECTIVE_REPAIR.md):

    * **Signed, learned differences.** A pixel ``abs`` difference fires on any
      luminance change and discards direction, so the bat, pads, batsman and
      shadows compete with the ball on equal terms. Post-bounce that is the
      dominant failure -- 12.0% of labelled frames there have the ball present in
      the top-8 peaks but outranked by a distractor, against 2.8% where the
      detector does not respond at all. A signed difference of *encoder features*
      can represent "a small round high-contrast blob translated this way".
    * **Appearance stays first-class.** The motion features are fused with
      ``F(t)`` rather than gating it, because the slow post-bounce ball is the
      sharpest, highest-contrast appearance in the delivery (diff SNR 10.12 at
      the ball, against 3.40 in the motion-blurred flight).
    * **Differences are taken on the high-resolution branch.** A ball stepping
      1.3 px does not leave a stride-4 cell, so a difference there is zero by
      construction. Use ``hires_lateral`` (stride 2) with this model.
    * **One encoder pass per frame.** Early channel-stacking re-encodes each
      frame once per window it appears in -- 3x redundant at inference. Here the
      per-frame features can be cached and the model streams.

    The head predicts a softmax over ``H*W + 1``: every location competes with
    every other and with an explicit "no ball" class, so presence is calibrated
    and spatially attributable instead of pooled out of the feature map. The old
    pooled visibility head was measured *blocking 75.7% of the frames where the
    baseline had already found the ball*.
    """

    def __init__(
        self,
        sequence_length: int = 3,
        feature_channels: int = 48,
        low_channels: int | None = None,
        encoder_stride: int = 2,
        encoder_dropout: float = 0.10,
        backbone: str = "compact",
        pretrained_backbone: bool = False,
        hires_lateral: bool = False,
        temporal_blocks: int = 2,
        shift_fold_div: int = 8,
        predict_offsets: bool = True,
        heatmap_prior: float | None = None,
    ):
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.sequence_length = sequence_length
        self.predict_offsets = predict_offsets
        if backbone in TIMM_BACKBONES:
            expected = 2 if hires_lateral else TimmHRNetW18Encoder.output_stride
            if encoder_stride != expected:
                raise ValueError(
                    f"{backbone} emits stride {expected}"
                    f"{' with hires_lateral' if hires_lateral else ''}"
                )
            self.encoder: nn.Module = TimmHRNetW18Encoder(
                feature_channels=feature_channels,
                pretrained=pretrained_backbone,
                dropout=encoder_dropout,
                in_channels=3,                     # one frame at a time, not a stack
                model_name=backbone,
                hires_lateral=hires_lateral,
            )
            self.stride = self.encoder.output_stride
        elif backbone == "compact":
            self.encoder = SharedHRNetEncoder(
                feature_channels=feature_channels,
                low_channels=low_channels,
                output_stride=encoder_stride,
                dropout=encoder_dropout,
                in_channels=3,
            )
            self.stride = encoder_stride
        else:
            raise ValueError(f"unknown backbone: {backbone!r}")

        # Motion branch over the T-1 signed differences.
        self.motion_blocks = nn.Sequential(
            *[TemporalShiftConv(feature_channels, shift_fold_div) for _ in range(temporal_blocks)]
        )
        self.motion_pool = nn.Sequential(
            nn.Conv2d(feature_channels * (sequence_length - 1), feature_channels, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        # Fuse motion with the target frame's own appearance.
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            ConvNormAct(feature_channels, feature_channels),
        )
        self.head = HeatmapHead(
            feature_channels,
            upsample_stages=int(math.log2(self.stride)),
            out_channels=1,
            prior_probability=heatmap_prior,
        )
        # Sub-cell offset at the feature grid, and the null ("no ball") logit.
        self.offset_head = (
            nn.Conv2d(feature_channels, 2, 1) if predict_offsets else None
        )
        self.null_head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(feature_channels, 1)
        )

    def encode(self, frames: Tensor) -> Tensor:
        """Encode ``[batch, time, 3, H, W]`` to ``[batch, time, C, h, w]``.

        Any positive number of frames is accepted, including one newly arrived
        frame. Features may be cached across windows when the encoder is in
        evaluation mode and its weights and input preprocessing stay fixed.
        """

        if frames.ndim != 5 or frames.size(2) != 3:
            raise ValueError("expected frames shaped [batch, time, 3, height, width]")
        if frames.size(1) < 1:
            raise ValueError("expected at least one frame to encode")
        batch, steps, channels, height, width = frames.shape
        features = self.encoder(frames.reshape(batch * steps, channels, height, width))
        return features.reshape(batch, steps, *features.shape[1:])

    def forward(self, frames: Tensor, motion: Optional[Tensor] = None,
                return_visibility: bool = False, return_null: bool = True):
        """``frames`` is ``[batch, time, 3, H, W]``. ``motion`` is accepted and
        ignored so this model is a drop-in for callers written against the
        stacked one -- motion is computed internally, in feature space."""

        if frames.ndim != 5 or frames.size(2) != 3:
            raise ValueError("expected frames shaped [batch, time, 3, height, width]")
        if frames.size(1) != self.sequence_length:
            raise ValueError(f"expected {self.sequence_length} frames, received {frames.size(1)}")
        return self.forward_features(
            self.encode(frames), frames.shape[-2:],
            return_visibility=return_visibility, return_null=return_null,
        )

    def forward_features(
        self,
        features: Tensor,
        output_size: Tuple[int, int],
        return_visibility: bool = False,
        return_null: bool = True,
    ):
        """Predict from an encoded window shaped ``[batch, time, C, h, w]``.

        ``time`` must equal ``sequence_length`` and the newest frame must be
        last. ``output_size`` is the original input's ``(height, width)``; it
        cannot always be recovered from the downsampled feature dimensions.
        In evaluation mode, reusing per-frame features preserves full-window
        predictions while avoiding repeated encoder work. Returns the same
        outputs as :meth:`forward`; ``return_null`` remains accepted for caller
        compatibility, with the default three-item result always including it.
        """
        if features.ndim != 5:
            raise ValueError("expected features shaped [batch, time, channels, height, width]")
        if features.size(1) != self.sequence_length:
            raise ValueError(f"expected {self.sequence_length} feature frames, received {features.size(1)}")
        # Signed feature differences: D_k = F(t_k) - F(t_k-1).
        differences = features[:, 1:] - features[:, :-1]
        differences = self.motion_blocks(differences)
        batch = differences.size(0)
        motion_features = self.motion_pool(
            differences.reshape(batch, -1, *differences.shape[-2:])
        )
        fused = self.fuse(torch.cat((features[:, -1], motion_features), dim=1))
        logits = self.head(fused, output_size)
        null_logit = self.null_head(fused).reshape(-1)
        offsets = self.offset_head(fused) if self.offset_head is not None else None
        if return_visibility:
            # sigmoid(presence_logit) equals 1 - P(null) under the joint
            # spatial/null softmax, including its spatial normalizer.
            presence_logit = torch.logsumexp(logits.flatten(1), dim=1) - null_logit
            return logits, presence_logit
        return logits, null_logit, offsets


def build_tracker(model_config: dict) -> nn.Module:
    """Instantiate the tracker described by a checkpoint's model config.

    ``arch`` selects between the recurrent baseline (``convlstm``, the default
    for older checkpoints without the key), the early-fusion ``stacked`` model
    and ``shared_motion`` (shared encoder, signed feature-space motion, TSM,
    softmax-with-null head). Keys that do not apply to the selected architecture are dropped so
    one config dict can round-trip through either model.
    """

    config = dict(model_config)
    arch = config.pop("arch", "convlstm")
    if arch == "shared_motion":
        for key in ("hidden_channels", "target_index", "motion_channels",
                    "predict_frames", "predict_visibility"):
            config.pop(key, None)
        if "pretrained_backbone" in config:
            config["pretrained_backbone"] = False
        return SharedEncoderMotionTracker(**config)
    if arch == "stacked":
        for key in ("hidden_channels", "target_index", "temporal_blocks", "shift_fold_div"):
            config.pop(key, None)
        # The checkpoint's weights supersede any ImageNet init at reload time.
        if "pretrained_backbone" in config:
            config["pretrained_backbone"] = False
        return StackedHRNetBallTracker(**config)
    if arch == "convlstm":
        return HRNetConvLSTMBallTracker(**config)
    raise ValueError(f"unknown tracker architecture: {arch!r}")
