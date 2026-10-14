"""Heatmap losses and decoding helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


class FocalHeatmapLoss(torch.nn.Module):
    """Class-balanced focal BCE for very sparse Gaussian ball heatmaps.

    A 512x288 target has one ball centre among 147,456 pixels. Ordinary focal
    BCE can minimise its average loss simply by predicting background
    everywhere. ``positive_weight`` balances the integrated Gaussian target
    against those background pixels, while focal modulation still downweights
    already-easy negatives. The foreground/background balance is handled solely
    by ``positive_weight``; ``alpha`` defaults to 0.5 (neutral) to avoid
    weighting the foreground twice, which otherwise encourages over-confident
    blobs and a bias towards common ball locations.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, positive_weight: float = 1200.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("positive_weight", torch.tensor(positive_weight))

    def forward(self, logits: Tensor, targets: Tensor, weight: Tensor | None = None) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.positive_weight.to(dtype=logits.dtype)
        )
        probabilities = torch.sigmoid(logits)
        probability_of_target = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha * (1 - probability_of_target).pow(self.gamma) * loss
        if weight is not None:
            # Mean over the *supervised* pixels, so masking a region out does not
            # quietly shrink the loss (and with it the learning rate).
            weight = weight.to(dtype=loss.dtype)
            return (loss * weight).sum() / weight.sum().clamp(min=1.0)
        return loss.mean()


class PenaltyReducedFocalLoss(torch.nn.Module):
    """CornerNet/CenterNet heatmap loss, normalised per positive.

    The difference from ``FocalHeatmapLoss`` is what the Gaussian *means*. There,
    the soft target is fed to a weighted BCE, so every pixel with target mass is
    pulled towards ``pos_weight * t / (pos_weight * t + 1 - t)``: at
    ``pos_weight=300`` and sigma 1.8 that is >= 0.87 out to 5 px and 0.54 at 6 px,
    so the loss's own minimiser is a ~12 px saturated disc rather than a peak.
    Here the positive term applies only at the peak pixel, and the Gaussian only
    *reduces* the penalty on the negatives around it -- near misses are forgiven
    without being rewarded, and the optimum is a sharp peak again.

    Normalising by the number of positives (rather than by all 147,456 pixels)
    keeps the gradient scale independent of image size and of how much of the
    frame is masked out, and removes the "raise the whole map" descent direction
    that a pixel-mean loss offers early in training.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, epsilon: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, logits: Tensor, targets: Tensor, weight: Tensor | None = None) -> Tensor:
        probabilities = torch.sigmoid(logits.float()).clamp(self.epsilon, 1.0 - self.epsilon)
        # The ball centre is sub-pixel, so the Gaussian's own peak value is < 1
        # whenever the label falls between pixels: the positive is "the strongest
        # target pixel", not "target == 1".
        peak = targets.amax(dim=(-2, -1), keepdim=True)
        positive = (targets >= peak - 1e-6) & (peak > 0.5)
        positive_loss = -(1 - probabilities).pow(self.alpha) * torch.log(probabilities)
        negative_loss = (
            -(1 - targets).pow(self.beta)
            * probabilities.pow(self.alpha)
            * torch.log(1 - probabilities)
        )
        loss = torch.where(positive, positive_loss, negative_loss)
        if weight is not None:
            weight = weight.to(dtype=loss.dtype)
            loss = loss * weight
            positive = positive & (weight > 0)
        return loss.sum() / positive.sum().clamp(min=1)


class PeakRankingLoss(torch.nn.Module):
    """Margin loss on the ball's peak against its strongest competitor.

    A per-pixel heatmap loss is blind to how a wrong peak is ranked: measured on
    this model's own loss, adding a fully confident phantom ball elsewhere in the
    frame costs 1/1588th of what missing the real ball costs, so nothing in
    training ever asks the ball to *win*. That shows up as a 16 pp gap between
    top-1 (71.4%) and top-8 (87.6%) peak recall -- frames where the detector
    already responds to the ball but responds harder to something else.

    This term is exactly that constraint: the logit at the labelled ball must
    beat the strongest logit outside ``radius`` by ``margin``. It applies only to
    frames with a visible ball; ball-absent frames are already handled by the
    heatmap loss, whose target for them is uniformly zero.
    """

    def __init__(self, margin: float = 2.0, radius: float = 8.0):
        super().__init__()
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.margin = margin
        self.radius = radius

    def forward(self, logits: Tensor, positions: Tensor, visible: Tensor) -> Tensor:
        heat = logits[:, 0].float()
        batch, height, width = heat.shape
        visible = visible.reshape(-1).bool()
        if not bool(visible.any()):
            return _connected_zero(logits)
        heat = heat[visible]
        centres = positions[visible]
        rows = torch.arange(height, device=heat.device, dtype=heat.dtype)
        columns = torch.arange(width, device=heat.device, dtype=heat.dtype)
        dy = (rows[None, :] - centres[:, 1:2]).square()
        dx = (columns[None, :] - centres[:, 0:1]).square()
        distance = dy[:, :, None] + dx[:, None, :]
        outside = distance > self.radius * self.radius
        # -inf where the ball is, so the max is over competitors only. A ball at
        # the image edge can leave no pixel outside the radius; guard with a
        # finite floor rather than producing NaN.
        competitor = torch.where(outside, heat, torch.full_like(heat, float("-inf")))
        strongest = competitor.flatten(1).amax(dim=-1)
        strongest = torch.where(torch.isfinite(strongest), strongest, torch.zeros_like(strongest))
        x = centres[:, 0].round().long().clamp(0, width - 1)
        y = centres[:, 1].round().long().clamp(0, height - 1)
        at_ball = heat[torch.arange(heat.size(0), device=heat.device), y, x]
        return F.relu(self.margin + strongest - at_ball).mean()


def windowed_soft_argmax(logits: Tensor, window: int = 9) -> Tensor:
    """Differentiable expected ``(x, y)`` inside a window centred on each peak.

    A *global* soft-argmax over a 512x288 map is easily biased by residual
    background mass or a second look-alike blob, so the expectation is instead
    restricted to a ``window x window`` region around the argmax peak. The
    window is shifted (not shrunk) at image borders. Gradients flow through the
    local softmax weights, making this usable as a DSNT-style training signal.
    """

    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd number >= 3")
    heat = logits[:, 0]
    batch, height, width = heat.shape
    flat_index = heat.flatten(1).argmax(dim=-1)
    peak_x = flat_index % width
    peak_y = torch.div(flat_index, width, rounding_mode="floor")
    radius = window // 2
    x0 = (peak_x - radius).clamp(0, max(width - window, 0))
    y0 = (peak_y - radius).clamp(0, max(height - window, 0))
    offsets = torch.arange(window, device=logits.device)
    xs = x0[:, None] + offsets[None]  # [batch, window]
    ys = y0[:, None] + offsets[None]
    rows = torch.arange(batch, device=logits.device)[:, None, None]
    patch = heat[rows, ys[:, :, None], xs[:, None, :]]  # [batch, window(y), window(x)]
    weights = torch.softmax(patch.flatten(1), dim=-1).reshape(batch, window, window)
    expected_x = (weights.sum(dim=1) * xs.float()).sum(dim=-1)
    expected_y = (weights.sum(dim=2) * ys.float()).sum(dim=-1)
    return torch.stack((expected_x, expected_y), dim=1)


class CoordinateRefinementLoss(torch.nn.Module):
    """Windowed DSNT coordinate loss complementing the focal heatmap loss.

    The focal loss owns gross localisation and the visible/occluded decision;
    this term only sharpens the peak so its local expectation lands on the
    exact sub-pixel ball centre. Samples whose ground truth is outside the
    peak's window are skipped: pulling the expectation towards an unreachable
    target would only distort the peak the heatmap loss is still moving.
    """

    def __init__(self, window: int = 9):
        super().__init__()
        if window % 2 == 0 or window < 3:
            raise ValueError("window must be an odd number >= 3")
        self.window = window

    def forward(self, logits: Tensor, positions: Tensor, visible: Tensor) -> Tensor:
        heat = logits[:, 0]
        batch, height, width = heat.shape
        flat_index = heat.flatten(1).argmax(dim=-1)
        peak_x = (flat_index % width).float()
        peak_y = torch.div(flat_index, width, rounding_mode="floor").float()
        radius = self.window // 2
        near = (
            visible.bool()
            & ((peak_x - positions[:, 0]).abs() <= radius)
            & ((peak_y - positions[:, 1]).abs() <= radius)
        )
        if not bool(near.any()):
            return _connected_zero(logits)
        coordinates = windowed_soft_argmax(logits, self.window)
        return F.smooth_l1_loss(coordinates[near] / radius, positions[near] / radius)


# Logit magnitude beyond which a softmax decision cannot change but fp16
# overflow can turn the loss into NaN. See SpatialSoftmaxNullLoss.forward.
_LOGIT_CLAMP = 50.0


def _connected_zero(tensor: Tensor) -> Tensor:
    """Zero that keeps the autograd graph, and stays zero if ``tensor`` has infs.

    Every loss here needs a "nothing to supervise in this batch" return: no
    visible ball, or every ball masked out by label repair. It has to stay
    attached to the graph, or ``loss.backward()`` raises when it is the only
    term. The obvious ``tensor.sum() * 0.0`` is wrong: under fp16 autocast an
    activation can reach inf, and ``inf * 0`` is **NaN**, which then poisons the
    running loss for the rest of the epoch -- observed as ``train_loss=nan``
    while every metric kept improving, i.e. the model was fine and the number
    was not.
    """

    # One element, in fp32. Summing the whole tensor is what the first version of
    # this did, and on a [4, 2, 144, 256] fp16 offset map that sum reaches ~1e5,
    # past fp16's 65504 -> inf -> inf * 0 = NaN. Exactly the failure it was
    # written to prevent, arrived at from the other direction.
    return torch.nan_to_num(
        tensor.reshape(-1)[:1].float(), nan=0.0, posinf=0.0, neginf=0.0
    ).sum() * 0.0


class SpatialSoftmaxNullLoss(torch.nn.Module):
    """Cross-entropy over ``H*W`` locations plus one explicit "no ball" class.

    Every per-pixel loss in this file shares a structural blind spot: pixels are
    independent, so nothing states that there is *exactly one* ball and that it
    must outrank everything else. Measured on this model's own focal loss, adding
    a fully confident phantom ball elsewhere in the frame costs 1/1588th of what
    missing the real one costs, and the result is a 7-16 pp gap between top-1 and
    top-8 peak recall -- frames where the detector responds to the ball but
    responds harder to something else.

    A softmax makes that constraint structural instead of an added penalty: the
    locations compete for one unit of probability mass, and the extra class
    absorbs "no ball", so presence comes out calibrated and spatially
    attributable rather than pooled off the feature map. (The pooled visibility
    head it replaces was measured blocking 75.7% of the frames where the
    baseline had already found the ball.)

    The target is the single nearest pixel, with ``label_smoothing`` spread over
    the rest; sub-pixel accuracy is the offset head's job, not this term's.
    ``weight`` is the per-pixel loss mask from label repair -- a sample is
    dropped when its ball sits under a masked region, because the mask means "we
    do not know where the ball is here", which is not the same as "no ball".
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: Tensor,
        null_logit: Tensor,
        positions: Tensor,
        visible: Tensor,
        weight: Tensor | None = None,
    ) -> Tensor:
        heat = logits[:, 0].float()
        batch, height, width = heat.shape
        visible = visible.reshape(-1).bool()
        x = positions[:, 0].round().long().clamp(0, width - 1)
        y = positions[:, 1].round().long().clamp(0, height - 1)
        target = torch.where(visible, y * width + x, torch.full_like(x, height * width))
        keep = torch.ones_like(visible)
        if weight is not None:
            mask = weight.reshape(batch, -1).float()
            keep = mask.amax(dim=-1) > 0                     # fully masked -> drop
            at_ball = mask.gather(1, (y * width + x).unsqueeze(1)).reshape(-1)
            # A ball sitting under an ignore region: we do not know where it is.
            keep = keep & torch.where(visible, at_ball > 0, torch.ones_like(at_ball, dtype=torch.bool))
            # And an *unlabelled* frame carrying any ignore region cannot be
            # taught as the null class either. This is where a whole-frame label
            # differs from a per-pixel one: the sigmoid losses use the mask to
            # avoid supervising the ball's own location as background, but "no
            # ball anywhere" is a single label for the frame, so masking pixels
            # does nothing to it. Left in, the null class would be trained on
            # 84,440 windows where the ball is visible but unlabelled -- and
            # those concentrate in pitch->impact, which has 46% label gaps and is
            # already the weakest phase at 76.2% recall. Only a *fully*
            # unmasked unlabelled frame is a genuine negative.
            fully_supervised = mask.amin(dim=-1) > 0
            keep = keep & (visible | fully_supervised)
        if not bool(keep.any()):
            return _connected_zero(logits)
        scores = torch.cat((heat.reshape(batch, -1), null_logit.reshape(batch, 1).float()), dim=1)
        # Clamp before the softmax. The forward pass runs under fp16 autocast, so
        # a logit can reach inf on a bad step; casting to fp32 preserves the inf
        # and a softmax over 147,457 classes then returns NaN, which poisons the
        # loss for the rest of the epoch. Observed at lr 5e-4 on a fresh head.
        # A finite-but-huge logit is the same decision, and stays differentiable.
        scores = torch.nan_to_num(scores, nan=0.0, posinf=_LOGIT_CLAMP, neginf=-_LOGIT_CLAMP)
        scores = scores.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        return F.cross_entropy(
            scores[keep], target[keep], label_smoothing=self.label_smoothing
        )


class SubCellOffsetLoss(torch.nn.Module):
    """L1 on the sub-cell offset predicted at the ball's own feature cell.

    The heatmap is decoded on a grid; this recovers the fraction of a cell the
    ball sits away from that grid point, so the coarse classification above does
    not cap localisation. Only visible frames contribute.
    """

    def forward(self, offsets: Tensor, positions: Tensor, visible: Tensor, stride: int) -> Tensor:
        visible = visible.reshape(-1).bool()
        if not bool(visible.any()):
            return _connected_zero(offsets)
        offsets = offsets.float()
        cells = positions[visible] / stride
        indices = cells.floor().long()
        fractional = cells - indices.float()
        height, width = offsets.shape[-2:]
        cx = indices[:, 0].clamp(0, width - 1)
        cy = indices[:, 1].clamp(0, height - 1)
        rows = torch.arange(offsets.size(0), device=offsets.device)[visible]
        predicted = offsets[rows, :, cy, cx]
        return F.l1_loss(predicted, fractional)


@torch.no_grad()
def decode_softmax_null(
    logits: Tensor,
    null_logit: Tensor,
    offsets: Tensor | None = None,
    stride: int = 2,
    topk: int = 1,
    nms_radius: int = 3,
):
    """Decode the softmax-with-null head.

    Returns ``(positions, scores, absent_probability)``: the top-K locations with
    their share of the softmax mass, refined by the offset head when present, and
    the probability the frame has no ball at all. Unlike ``decode_heatmaps``
    these scores are *comparable across frames* -- they are one distribution over
    "where, or nowhere" rather than an independent per-pixel sigmoid.
    """

    heat = logits[:, 0].float()
    batch, height, width = heat.shape
    scores = torch.cat((heat.reshape(batch, -1), null_logit.reshape(batch, 1).float()), dim=1)
    scores = torch.nan_to_num(scores, nan=0.0, posinf=_LOGIT_CLAMP, neginf=-_LOGIT_CLAMP)
    probabilities = torch.softmax(scores.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP), dim=-1)
    absent = probabilities[:, -1]
    spatial = probabilities[:, :-1].reshape(batch, 1, height, width)
    kernel = 2 * nms_radius + 1
    pooled = F.max_pool2d(spatial, kernel, stride=1, padding=nms_radius)
    suppressed = torch.where(pooled == spatial, spatial, torch.zeros_like(spatial))
    values, flat = suppressed.reshape(batch, -1).topk(topk, dim=-1)
    x = (flat % width).float()
    y = torch.div(flat, width, rounding_mode="floor").float()
    if offsets is not None:
        cx = (x / stride).long().clamp(0, offsets.size(-1) - 1)
        cy = (y / stride).long().clamp(0, offsets.size(-2) - 1)
        rows = torch.arange(batch, device=heat.device)[:, None].expand_as(cx)
        delta = offsets.float()[rows, :, cy, cx]          # [batch, topk, 2]
        x = x + delta[..., 0] * stride
        y = y + delta[..., 1] * stride
    return torch.stack((x, y), dim=-1), values, absent


@torch.no_grad()
def decode_heatmaps(logits: Tensor, refine: bool = True) -> Tuple[Tensor, Tensor]:
    """Return sub-pixel ``(x, y)`` coordinates and confidence for every heatmap.

    The integer argmax peak is refined by fitting a 1-D parabola to the peak and
    its two neighbours along each axis (the standard quadratic sub-pixel
    estimator). This recovers accuracy lost to the discrete heatmap grid without
    changing the peak itself when neighbours are symmetric.
    """

    probabilities = torch.sigmoid(logits)[:, 0]  # [batch, height, width]
    batch, height, width = probabilities.shape
    confidence, flat_index = probabilities.flatten(1).max(dim=-1)
    x = (flat_index % width).long()
    y = torch.div(flat_index, width, rounding_mode="floor").long()
    x_f, y_f = x.float(), y.float()
    if refine:
        rows = torch.arange(batch, device=probabilities.device)
        centre = probabilities[rows, y, x]
        left = probabilities[rows, y, (x - 1).clamp_(0, width - 1)]
        right = probabilities[rows, y, (x + 1).clamp_(0, width - 1)]
        up = probabilities[rows, (y - 1).clamp_(0, height - 1), x]
        down = probabilities[rows, (y + 1).clamp_(0, height - 1), x]
        denom_x = left - 2 * centre + right
        denom_y = up - 2 * centre + down
        offset_x = torch.where(denom_x.abs() > 1e-6, 0.5 * (left - right) / denom_x, torch.zeros_like(denom_x))
        offset_y = torch.where(denom_y.abs() > 1e-6, 0.5 * (up - down) / denom_y, torch.zeros_like(denom_y))
        x_f = x_f + offset_x.clamp_(-0.5, 0.5)
        y_f = y_f + offset_y.clamp_(-0.5, 0.5)
    return torch.stack((x_f, y_f), dim=1), confidence


@torch.no_grad()
def batch_metrics(logits: Tensor, positions: Tensor, visible: Tensor, threshold: float = 8.0) -> Dict[str, float]:
    """Compute visible-frame localisation error and detection rate."""

    predicted_positions, confidences = decode_heatmaps(logits)
    visible = visible.bool()
    if not bool(visible.any()):
        return {"pixel_error": 0.0, "within_threshold": 0.0, "mean_confidence": float(confidences.mean())}
    distances = torch.linalg.vector_norm(predicted_positions[visible] - positions[visible], dim=1)
    return {
        "pixel_error": float(distances.mean()),
        "within_threshold": float((distances <= threshold).float().mean()),
        "mean_confidence": float(confidences.mean()),
    }
