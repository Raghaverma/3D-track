"""TrackNet-style image/CSV dataset reader and heatmap target generation."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Border fill for the geometric augmentation; see ``pad_mode``.
_PAD_MODES = {
    "reflect": cv2.BORDER_REFLECT_101,
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
}


@dataclass(frozen=True)
class BallLabel:
    frame_id: int
    x: float
    y: float
    visible: bool


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    image_paths: Tuple[Path, ...]
    labels: Tuple[BallLabel, ...]


def _normalise(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _as_number(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return default


def _frame_number(value: str, default: int) -> int:
    """Accept either a numeric frame id or a TrackNet filename such as 0007.jpg."""

    digits = "".join(character for character in value if character.isdigit())
    return int(digits) if digits else default


def _field(row: Dict[str, str], candidates: Iterable[str], default: str = "") -> str:
    normalised = {_normalise(key): value for key, value in row.items() if key is not None}
    for candidate in candidates:
        if candidate in normalised:
            return normalised[candidate]
    return default


def read_tracknet_labels(path: Path) -> List[BallLabel]:
    """Read a TrackNet CSV using common names for frame, visibility, x and y.

    The canonical dataset has one row per frame. Both ``Visibility`` / ``V``
    and ``X`` / ``Y`` column conventions are accepted. Rows without a visible
    ball are retained as negative heatmap examples.
    """

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} does not contain a header row")
        labels = []
        for row_index, row in enumerate(reader):
            frame_value = _field(row, ("frame", "frameid", "index", "id", "filename", "file"))
            frame = _frame_number(frame_value, row_index)
            x = _as_number(_field(row, ("x", "ballx", "xcoordinate")))
            y = _as_number(_field(row, ("y", "bally", "ycoordinate")))
            visibility = _field(row, ("visibility", "visible", "v", "status"), "1")
            visible = _as_number(visibility, 1.0) > 0 and x >= 0 and y >= 0
            labels.append(BallLabel(frame, x, y, visible))
    if not labels:
        raise ValueError(f"{path} has no annotations")
    return labels


def _image_frame_id(path: Path, fallback: int) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else fallback


def _find_images(annotation_path: Path) -> List[Path]:
    """Locate image frames beside a Label.csv, including a ``frames`` folder."""

    parent = annotation_path.parent
    roots = (parent, parent / "frames", parent / "images", parent / "img")
    candidates = []
    for root in roots:
        if root.is_dir():
            candidates.extend(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if candidates:
            break
    return sorted(candidates, key=lambda path: (_image_frame_id(path, 0), path.name))


def discover_tracknet_sequences(root: str | Path) -> List[SequenceRecord]:
    """Discover every labelled clip below ``root``.

    This keeps the project independent of a particular download wrapper: place
    any TrackNet split under ``data/tracknet/{train,val}`` and retain its
    ``Label.csv`` files. Clips whose labels or images cannot be read are
    rejected early with a useful error.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root_path}")
    # ``Path.rglob`` intentionally skips symlinked directories. Following them
    # here permits space-efficient match-level train/validation splits.
    annotation_files = sorted(
        Path(directory) / filename
        for directory, _, filenames in os.walk(root_path, followlinks=True)
        for filename in filenames
        if Path(filename).suffix.lower() == ".csv"
        and _normalise(Path(filename).stem) in {"label", "labels", "annotation", "annotations"}
    )
    if not annotation_files:
        raise FileNotFoundError(f"no Label.csv, labels.csv, or annotation.csv found in {root_path}")
    sequences = []
    for annotation_path in annotation_files:
        labels = read_tracknet_labels(annotation_path)
        images = _find_images(annotation_path)
        if not images:
            raise FileNotFoundError(f"no image frames found next to {annotation_path}")
        by_frame = {label.frame_id: label for label in labels}
        ordered_labels = []
        for index, image_path in enumerate(images):
            label = by_frame.get(_image_frame_id(image_path, index))
            # Some releases number image files from 0/1 differently from CSV.
            ordered_labels.append(label if label is not None else labels[min(index, len(labels) - 1)])
        sequences.append(
            SequenceRecord(str(annotation_path.parent.relative_to(root_path)), tuple(images), tuple(ordered_labels))
        )
    return sequences


_AXIS_CACHE: Dict[Tuple[int, int], Tuple[Tensor, Tensor]] = {}


def _pixel_axes(height: int, width: int) -> Tuple[Tensor, Tensor]:
    """Cached ``(rows, columns)`` coordinate vectors for one image size."""

    axes = _AXIS_CACHE.get((height, width))
    if axes is None:
        axes = (torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32))
        _AXIS_CACHE[(height, width)] = axes
    return axes


def make_gaussian_heatmap(height: int, width: int, x: float, y: float, sigma: float) -> Tensor:
    """Create a unit-peak Gaussian target, clipped naturally by image bounds.

    An isotropic Gaussian is separable -- exp(-(dx^2 + dy^2)/2s^2) is
    exp(-dx^2/2s^2) * exp(-dy^2/2s^2) -- so this builds two 1-D vectors and takes
    their outer product instead of running meshgrid, two squares, a sum and an
    exp over the whole H*W grid. Same values, but the transcendental is
    evaluated H+W times rather than H*W: at 288x512 that is 800 exps rather than
    147,456. Building the target was ~15% of the dataloader's per-window cost,
    and the loader is what bounds training throughput.
    """

    rows, columns = _pixel_axes(height, width)
    scale = -1.0 / (2 * sigma * sigma)
    row_weights = torch.exp((rows - y).square() * scale)
    column_weights = torch.exp((columns - x).square() * scale)
    return torch.outer(row_weights, column_weights).unsqueeze(0)


# Explicit input-channel schemes over a `sequence_length`-frame window.
# ``rgb`` lists which window positions are emitted as RGB (0 = oldest, -1/4 =
# newest = the localisation target). ``diffs`` are (a, b) pairs meaning the
# grayscale motion channel ``|frame[a] - frame[b]|``. The default scheme is the
# historical one (all frames RGB + consecutive diffs) and is applied in-code, so
# only non-default schemes need an entry here.
#
# ``rgb3newest_motion13``: the 3 NEWEST frames as RGB (so the target frame is in
# RGB and stays precisely localisable) + 4 motion diffs -- the two dropped older
# frames' motion (|f2-f1|, |f3-f2|), the most-recent transition into the target
# frame (|f5-f4|, the ball's current velocity), and a whole-window diff
# (|f5-f1|). 3*3 + 4 = 13 input channels.
# ``rgb3newest_pyramid13``: the same 13 channels, but the four diffs form a
# temporal *pyramid* over a 9-frame window -- baselines of 1, 2, 4 and 8 frames
# against the target frame -- instead of three 1-frame diffs and one 4-frame.
#
# The reason is measured. A frame difference only separates the ball from itself
# once the ball has moved further than its own diameter: at 5 px across and
# 10 px/frame (the pre-bounce flight) a 1-frame diff gives two clean blobs, but
# at 1.3 px/frame it gives one blob overlapping itself, i.e. almost no signal.
# And image-space speed is exactly what collapses after the bounce, because the
# ball is then travelling towards the camera: median step 9.9 px/frame before
# pitching against 3.5 px/frame after, 1.3 px/frame on the slowest post-bounce
# frames. Recall follows it -- 95.2% in the flight, 73.0% after the bounce, and
# within the post-bounce phase alone 81.9% on frames stepping >= 3 px against
# **65.1%** on the slow ones. The post-bounce segment is the one that carries
# deviation and the impact point, so that is the expensive place to be blind.
#
# A pyramid covers both regimes with the same channel budget: the 8-frame
# baseline moves a 1.3 px/frame ball ~10 px (clear of itself) while the 1-frame
# baseline still resolves a 10 px/frame ball without smearing it.
CHANNEL_SCHEMES: Dict[str, Dict[str, list]] = {
    "rgb3newest_motion13": {
        "rgb": [2, 3, 4],
        "diffs": [(1, 0), (2, 1), (4, 3), (4, 0)],
    },
    "rgb3newest_pyramid13": {
        "rgb": [6, 7, 8],
        "diffs": [(8, 7), (8, 6), (8, 4), (8, 0)],
    },
}


class TrackNetWindowDataset(Dataset[Dict[str, Tensor]]):
    """Returns 5-frame RGB windows and the target heatmap for the final frame."""

    def __init__(
        self,
        root: str | Path,
        sequence_length: int = 5,
        image_size: Tuple[int, int] = (288, 512),
        sigma: float = 2.5,
        augment: bool = False,
        motion: bool = False,
        predict_all_frames: bool = False,
        channel_scheme: str = "default",
        gap_fill: int = 0,
        ignore_gaps: bool = False,
        ignore_edge_frames: int = 0,
        pad_mode: str = "reflect",
        drop_unsupervised: bool = False,
        drop_masked_negatives: bool = False,
    ):
        self.sequence_length = sequence_length
        self.image_size = image_size  # (height, width)
        self.sigma = sigma
        self.augment = augment
        # ``motion`` appends |frame_t - frame_{t-1}| grayscale channels so the
        # first convolution receives explicit velocity, not just appearance.
        # ``predict_all_frames`` supervises a heatmap per frame instead of only
        # the newest one, forcing the internal representation to carry the whole
        # moving track rather than a single-frame appearance match.
        self.motion = motion
        self.predict_all_frames = predict_all_frames
        # ``channel_scheme`` selects a fixed RGB/motion-diff layout (see
        # CHANNEL_SCHEMES). "default" keeps the historical all-RGB behaviour.
        self.channel_scheme = channel_scheme
        self._scheme = None if channel_scheme == "default" else CHANNEL_SCHEMES[channel_scheme]
        if self._scheme is not None:
            max_index = max(max(self._scheme["rgb"]), max(i for pair in self._scheme["diffs"] for i in pair))
            if max_index >= sequence_length:
                raise ValueError(
                    f"channel_scheme {channel_scheme!r} references frame {max_index} "
                    f"but sequence_length is {sequence_length}"
                )
            if predict_all_frames:
                raise ValueError("predict_all_frames is not supported with a fixed channel_scheme")
        # Label-repair options. The annotation tool labels a *delivery span* and
        # leaves 45% of the frames inside it unlabelled (plus the run-up before
        # release), so a plain reading of the CSV supervises a visible ball as
        # background on ~54% of the windows. These three options stop that
        # without discarding the genuinely useful negatives:
        #
        # ``gap_fill``           - unlabelled frames whose surrounding labels are
        #                          <= N frames apart get the linearly interpolated
        #                          position as a real positive (a short gap is
        #                          almost always the annotator skipping, and the
        #                          interpolation error over <= 4 frames is small).
        # ``ignore_gaps``        - longer interior gaps are *excluded* from the
        #                          loss in a disc around the interpolated path
        #                          rather than trained as background; the rest of
        #                          the frame stays a negative.
        # ``ignore_edge_frames`` - the N frames on either side of the labelled
        #                          span are dropped from supervision entirely
        #                          (whole-frame weight 0). That is the in-hand
        #                          ball before release and the ball flying away
        #                          after impact: present, unlabelled, unlocatable.
        #
        # All three are training-set transforms; leave them off for validation so
        # val metrics stay comparable with earlier runs.
        if gap_fill < 0 or ignore_edge_frames < 0:
            raise ValueError("gap_fill and ignore_edge_frames must be non-negative")
        # How the geometric augmentation fills the area outside the source frame.
        # "reflect" (the historical default) mirrors the frame edge, which for a
        # detection task with exactly one target is a labelling error: at the 45%
        # of samples with scale < 1 the padding band is a mirror of the edge, so
        # a ball within ~64 px of an edge (11.1% of labels) gets an unlabelled
        # twin in the band -- ~5% of positive windows carry an injected hard
        # negative, in a loss that already penalises a miss 1588x more than a
        # phantom. "constant" pads black instead; "replicate" smears the edge.
        if pad_mode not in _PAD_MODES:
            raise ValueError(f"pad_mode must be one of {sorted(_PAD_MODES)}")
        self.pad_mode = pad_mode
        self._border_mode = _PAD_MODES[pad_mode]
        self.gap_fill = gap_fill
        self.ignore_gaps = ignore_gaps
        self.ignore_edge_frames = ignore_edge_frames
        self._repair = bool(gap_fill or ignore_gaps or ignore_edge_frames)
        # ``ignore_edge_frames`` emits windows whose loss weight is zero
        # everywhere. Those contribute *exactly* nothing to either heatmap loss
        # (masked sum / positives, with no positives) and nothing to the softmax
        # loss (dropped by its keep mask), yet each still costs three JPEG
        # decodes, a forward and a backward. Dropping them from the sample list
        # is mathematically identical training for free -- measured at 34,537 of
        # 366,406 windows on all-annotations, i.e. 9.4% of every epoch.
        self.drop_unsupervised = drop_unsupervised
        # ``drop_masked_negatives`` additionally drops *unlabelled* frames that
        # carry an ignore region. They are waste only for a whole-frame objective:
        # the sigmoid losses still learn the background outside the mask, but a
        # softmax over "location or nowhere" cannot use a partially-masked frame
        # at all (see SpatialSoftmaxNullLoss), so it drops them -- and then
        # sampling them costs three decodes, a forward and a backward for
        # nothing. 84,440 of 366,406 windows on all-annotations.
        self.drop_masked_negatives = drop_masked_negatives
        if self._repair and predict_all_frames:
            raise ValueError("label repair emits one weight map for the target frame only")
        self.sequences = discover_tracknet_sequences(root)
        self.samples: List[Tuple[int, int]] = [
            (sequence_index, frame_index)
            for sequence_index, sequence in enumerate(self.sequences)
            for frame_index in range(len(sequence.image_paths))
        ]
        # Frame indexes carrying a visible label, per sequence, for the repair
        # options above. Cheap: labels are already in memory.
        self._labelled_frames: List[np.ndarray] = [
            np.array([index for index, label in enumerate(sequence.labels) if label.visible], dtype=np.int64)
            for sequence in self.sequences
        ] if self._repair else []
        if self._repair and (drop_unsupervised or drop_masked_negatives):
            useless = {"blank"} if drop_unsupervised else set()
            if drop_masked_negatives:
                useless.add("ignore")
            kept = [
                (sequence_index, frame)
                for sequence_index, frame in self.samples
                if self.sequences[sequence_index].labels[frame].visible
                or self._label_context(sequence_index, frame)[0] not in useless
            ]
            self.dropped_unsupervised = len(self.samples) - len(kept)
            self.samples = kept
        else:
            self.dropped_unsupervised = 0

    def __len__(self) -> int:
        return len(self.samples)

    # JPEG frames are 1080p but the model trains at ~512x288; decoding at half
    # resolution straight from the JPEG (IMREAD_REDUCED_COLOR_2) is ~2-4x cheaper
    # on CPU and lossless for our purpose since we downscale anyway. Label coords
    # stay in original-pixel space, so __getitem__ multiplies the loaded shape
    # back up by this factor when scaling labels.
    _DECODE_REDUCTION = 2

    def _read_frame(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
        if image is None:
            raise ValueError(f"could not decode image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _random_uniform(low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high).item())

    @staticmethod
    def _motion_blur_kernel(length: int, angle_degrees: float) -> np.ndarray:
        """A normalised line kernel that smears along one direction."""

        kernel = np.zeros((length, length), dtype=np.float32)
        centre = (length - 1) / 2
        radians = math.radians(angle_degrees)
        dx, dy = math.cos(radians), math.sin(radians)
        point_a = (int(round(centre - dx * centre)), int(round(centre - dy * centre)))
        point_b = (int(round(centre + dx * centre)), int(round(centre + dy * centre)))
        cv2.line(kernel, point_a, point_b, 1.0, 1)
        return kernel / max(kernel.sum(), 1.0)

    def _augment_window(
        self, images: List[np.ndarray], coords: List[Tuple[float, float, bool]]
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float, bool]]]:
        """Apply one geometric transform and one appearance transform to a clip.

        Geometry and appearance must be shared by every frame; otherwise the
        temporal model learns artificial motion or flicker. The same affine
        matrix is applied to each frame's labelled coordinate, and a ball
        shifted outside the image becomes invisible. The wide scale, hue,
        grayscale, and directional motion-blur ranges target scale,
        ball-colour, and shutter-blur robustness respectively.

        ``coords`` holds one ``(x, y, visible)`` per frame (all equal to the
        target label in single-frame mode). The transformed coordinates are
        returned in the same order.
        """

        height, width = self.image_size
        angle = self._random_uniform(-3.0, 3.0)
        scale = self._random_uniform(0.75, 1.30)
        tx = self._random_uniform(-0.06 * width, 0.06 * width)
        ty = self._random_uniform(-0.05 * height, 0.05 * height)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[:, 2] += (tx, ty)
        images = [
            cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=self._border_mode)
            for image in images
        ]
        transformed: List[Tuple[float, float, bool]] = []
        for x, y, visible in coords:
            if visible:
                x, y = (matrix @ np.array((x, y, 1.0), dtype=np.float32)).tolist()
                visible = 0 <= x < width and 0 <= y < height
            transformed.append((x, y, visible))
        coords = transformed

        if bool(torch.rand(()) < 0.5):
            images = [np.ascontiguousarray(image[:, ::-1]) for image in images]
            coords = [
                ((width - 1 - x if visible else x), y, visible) for x, y, visible in coords
            ]

        contrast = self._random_uniform(0.75, 1.25)
        brightness = self._random_uniform(-0.14, 0.14)
        saturation = self._random_uniform(0.75, 1.25)
        # Wide hue rotation (~±90°) prevents the network from keying on one
        # ball colour; occasional grayscale removes colour entirely.
        hue_shift = round(self._random_uniform(-45, 45))
        grayscale = bool(torch.rand(()) < 0.20)
        blur_kernel = 3 if bool(torch.rand(()) < 0.25) else 0
        motion_kernel = None
        if bool(torch.rand(()) < 0.30):
            motion_length = 3 + 2 * int(self._random_uniform(1.0, 5.0))  # odd, 5..13
            motion_kernel = self._motion_blur_kernel(motion_length, self._random_uniform(0.0, 180.0))
        augmented = []
        for image in images:
            image = np.clip(image.astype(np.float32) / 255.0 * contrast + brightness, 0.0, 1.0)
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 180
            hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * saturation, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            if grayscale:
                image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            if motion_kernel is not None:
                image = cv2.filter2D(image, -1, motion_kernel)
            if blur_kernel:
                image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
            augmented.append(image)
        return augmented, coords

    def _scaled_label(self, label: BallLabel, source_width: int, source_height: int) -> Tuple[float, float, bool]:
        height, width = self.image_size
        if not label.visible:
            return -1.0, -1.0, False
        return label.x * width / source_width, label.y * height / source_height, True

    # Interpolating across a gap of ``g`` frames is trustworthy for small ``g``
    # and useless for large ones (measured: the ball's own p50 step is 1 px but
    # p90 is 16 px, and detections inside gaps > 4 frames sit a median 43 px from
    # the straight line). So a long gap is masked along the whole *chord* between
    # its surrounding labels -- the ball is somewhere on that path -- with a
    # margin for how far it bows off the straight line. A capsule covers the real
    # uncertainty for roughly the area a disc big enough to do so would need.
    _IGNORE_MARGIN_BASE = 8.0
    _IGNORE_MARGIN_PER_FRAME = 2.0
    _IGNORE_MARGIN_MAX = 48.0
    # A filled label is an interpolation, not an observation, so its target peak
    # is deliberately blunter than a hand-placed one.
    _FILL_SIGMA_SCALE = 1.5

    def _label_context(self, sequence_index: int, target_frame: int):
        """Classify an *unlabelled* target frame against its clip's label span.

        Returns ``(kind, points, margin)`` where ``kind`` is one of ``"fill"``
        (interpolate a positive at ``points[0]``), ``"ignore"`` (mask the chord
        ``points[0]``-``points[1]`` out of the loss), ``"blank"`` (drop the whole
        frame from supervision) or ``"negative"`` (a real ball-absent frame,
        supervise as background). Points are in *source* pixel space.
        """

        labelled = self._labelled_frames[sequence_index]
        if labelled.size == 0:
            return "negative", (), 0.0
        first, last = int(labelled[0]), int(labelled[-1])
        if target_frame < first or target_frame > last:
            distance = first - target_frame if target_frame < first else target_frame - last
            if self.ignore_edge_frames and distance <= self.ignore_edge_frames:
                return "blank", (), 0.0
            return "negative", (), 0.0
        position = int(np.searchsorted(labelled, target_frame))
        previous, following = int(labelled[position - 1]), int(labelled[position])
        span = following - previous
        labels = self.sequences[sequence_index].labels
        start, end = labels[previous], labels[following]
        if self.gap_fill and span <= self.gap_fill + 1:
            ratio = (target_frame - previous) / span
            point = (start.x + (end.x - start.x) * ratio, start.y + (end.y - start.y) * ratio)
            return "fill", (point,), 0.0
        if self.ignore_gaps:
            margin = min(
                self._IGNORE_MARGIN_BASE + self._IGNORE_MARGIN_PER_FRAME * span,
                self._IGNORE_MARGIN_MAX,
            )
            return "ignore", ((start.x, start.y), (end.x, end.y)), margin
        return "negative", (), 0.0

    def _ignore_capsule(self, points: Sequence[Tuple[float, float]], margin: float) -> Tensor:
        """A ``[1, H, W]`` loss weight of ones, zeroed within ``margin`` of a path.

        ``points`` holds the chord's two endpoints, or one point when the other
        was pushed out of frame by augmentation.
        """

        height, width = self.image_size
        rows, columns = _pixel_axes(height, width)
        dy = rows.unsqueeze(1) - points[0][1]
        dx = columns.unsqueeze(0) - points[0][0]
        if len(points) == 2:
            along_x = points[1][0] - points[0][0]
            along_y = points[1][1] - points[0][1]
            length = along_x * along_x + along_y * along_y
            if length > 1e-6:
                # Project every pixel onto the chord and clamp to the segment, so
                # the masked region is a capsule around the whole path rather
                # than a disc at one interpolated guess along it.
                ratio = ((dx * along_x + dy * along_y) / length).clamp(0.0, 1.0)
                dx = dx - ratio * along_x
                dy = dy - ratio * along_y
        distance = dx.square() + dy.square()
        return (distance > margin * margin).float().unsqueeze(0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        sequence_index, target_frame = self.samples[index]
        sequence = self.sequences[sequence_index]
        frame_indexes = [max(0, target_frame - offset) for offset in range(self.sequence_length - 1, -1, -1)]
        source_images = [self._read_frame(sequence.image_paths[frame_index]) for frame_index in frame_indexes]
        # Labels are in original-pixel space; recover original dims from the
        # (reduced) decoded shape so scaling stays correct.
        source_height, source_width = (dim * self._DECODE_REDUCTION for dim in source_images[-1].shape[:2])
        height, width = self.image_size
        images = [cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) for image in source_images]
        # One (x, y, visible) per input frame; only the newest is ever used as
        # the localisation target unless ``predict_all_frames`` is set.
        coords = [
            self._scaled_label(sequence.labels[frame_index], source_width, source_height)
            for frame_index in frame_indexes
        ]
        # Label repair, decided on the *source* labels and then carried through
        # augmentation with the rest of the coordinates so a filled positive or
        # an ignore disc lands where the warped ball actually is.
        target_sigma = self.sigma
        blank_frame = False
        ignore_points: List[Tuple[float, float]] = []
        ignore_margin = 0.0
        if self._repair and not coords[-1][2]:
            kind, points, margin = self._label_context(sequence_index, target_frame)
            scaled = [(x * width / source_width, y * height / source_height) for x, y in points]
            if kind == "blank":
                blank_frame = True
            elif kind == "fill":
                coords[-1] = (scaled[0][0], scaled[0][1], True)
                target_sigma = self.sigma * self._FILL_SIGMA_SCALE
            elif kind == "ignore":
                ignore_points = scaled
                ignore_margin = margin
                coords.extend((x, y, True) for x, y in scaled)
        if self.augment:
            images, coords = self._augment_window(images, coords)
        if ignore_points:
            # The chord endpoints rode through augmentation with the labels; keep
            # whichever are still inside the frame.
            transformed = [coords.pop() for _ in range(len(ignore_points))][::-1]
            ignore_points = [(x, y) for x, y, inside in transformed if inside]

        # Grayscale in [0, 1] before ImageNet normalisation, kept for the
        # motion channels which are computed on this scale (a diff, mean ~0,
        # so no ImageNet normalisation is applied to it).
        grays = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for image in images]
        frames = []
        for image in images:
            image = image.astype(np.float32) / 255.0
            image = (image - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / np.array(
                (0.229, 0.224, 0.225), dtype=np.float32
            )
            frames.append(torch.from_numpy(image.transpose(2, 0, 1)))

        def build_target(coord: Tuple[float, float, bool], sigma: float | None = None) -> Tensor:
            x, y, visible = coord
            if visible:
                return make_gaussian_heatmap(height, width, x, y, sigma or self.sigma)
            return torch.zeros((1, height, width), dtype=torch.float32)

        if self.predict_all_frames:
            heatmap = torch.cat([build_target(coord) for coord in coords], dim=0)  # [T, H, W]
        else:
            heatmap = build_target(coords[-1], target_sigma)  # [1, H, W]

        target_x, target_y, target_visible = coords[-1]
        position = (
            torch.tensor((target_x, target_y), dtype=torch.float32)
            if target_visible
            else torch.tensor((-1.0, -1.0), dtype=torch.float32)
        )
        if self._scheme is not None:
            # Emit only the scheme's RGB positions; the target (coords[-1], the
            # newest frame) is among them so heatmap/position above are correct.
            frames = [frames[i] for i in self._scheme["rgb"]]
        sample = {
            "frames": torch.stack(frames),
            "heatmap": heatmap,
            "position": position,
            "visible": torch.tensor(target_visible, dtype=torch.bool),
        }
        if self._repair:
            # Per-pixel loss weight. Emitted whenever repair is enabled -- even
            # when it is all ones -- so every batch collates to the same keys.
            if blank_frame:
                sample["weight"] = torch.zeros((1, height, width), dtype=torch.float32)
            elif ignore_points:
                sample["weight"] = self._ignore_capsule(ignore_points, ignore_margin)
            else:
                sample["weight"] = torch.ones((1, height, width), dtype=torch.float32)
        if self._scheme is not None:
            diffs = [np.abs(grays[a] - grays[b]) for (a, b) in self._scheme["diffs"]]
            sample["motion"] = torch.from_numpy(np.stack(diffs, axis=0))  # [D, H, W]
        elif self.motion:
            diffs = [np.abs(grays[i] - grays[i - 1]) for i in range(1, len(grays))]
            sample["motion"] = torch.from_numpy(np.stack(diffs, axis=0))  # [T-1, H, W]
        return sample
