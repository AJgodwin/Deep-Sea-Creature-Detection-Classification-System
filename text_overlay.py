"""Detect burned-in telemetry text so detections drawn on it can be suppressed.

Submersible footage carries an overlay burned into the pixels by the vehicle's
video system: timestamp, depth, heading, and a copyright watermark. The detector
was never trained on text, but glyph strokes are small, bright, high-contrast and
compact, which is exactly the low-level signature it learned to fire on. On JEDI
frames it draws boxes around digit groups, and the classifier, having no way to
abstain, names them as organisms.

The question is asked per detection rather than per frame. Scanning a whole image
for text forces a precision/recall trade-off that is not worth paying: seabed
texture throws up glyph-like speckle everywhere, and a false text region in some
corner is harmless unless a detection happens to sit on it. Scoring only the
boxes the detector proposed keeps the question narrow, and lets each box be
resampled to a common resolution first, which is what makes the thresholds
transferable between a 480x360 frame and a 1080p one.

Within a box the test is that text is *regular* in ways a seabed is not: glyph
strokes of one pen width, characters of one height, sitting on a shared baseline.
A single bright compact blob is never text — an organism can look like one. Four
of them marching along a baseline at one height is not something sediment does.

Public entry points: :func:`text_score` for one box, :func:`filter_text_detections`
for a detection list, and :func:`find_text_regions` for whole-image use.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Component shape bounds. Loose on purpose: discrimination comes from the
# regularity tests below, not from these.
MAX_ASPECT = 3.2           # width / height of a single glyph
MIN_FILL = 0.10            # component area / its bounding-box area
MAX_FILL = 0.98

# Chaining tolerances, relative to glyph height so they scale with the frame.
# Deliberately tight: a loose chain lets one texture blob attach to the end of a
# genuine glyph run and destroy its alignment statistics.
BASELINE_TOL = 0.16
HEIGHT_RATIO = 0.62
MAX_GAP = 1.3
MIN_RUN = 4                # glyphs needed before a run counts as text

# Regularity thresholds, measured by comparing overlay runs against texture
# chains on JEDI and aquarium frames rather than chosen a priori. Genuine
# overlay lines sit at baseline residual <= 0.07 and height CV <= 0.27; the
# tightest texture chain observed was 0.12 / 0.16.
MAX_BASELINE_RESIDUAL = 0.10
MAX_HEIGHT_CV = 0.30
MAX_STROKE_CV = 0.45

# Boxes are resampled to this height before analysis so one set of pixel
# thresholds applies regardless of source resolution.
TARGET_CROP_H = 140
MAX_UPSCALE = 8.0

# Fraction of a detection box that must be text before it is suppressed.
# Measured separation is wide: hand-marked overlay boxes score 0.41-0.96, while
# 321 detector boxes collected over JEDI frames and every non-text region tested
# score 0.00. The threshold sits in that gap, nearer the text side so a box that
# merely clips an overlay — the JAMSTEC watermark inside a box drawn around
# hardware, which scores 0.00 — cannot be lost. Suppressing a real organism
# would be a worse error than keeping a text one.
SUPPRESS_COVERAGE = 0.35


def _glyph_mask(gray: np.ndarray) -> np.ndarray:
    """Isolate thin bright strokes, which is what overlay glyphs are.

    A top-hat keeps structures narrower than the kernel and brighter than their
    surroundings, so it responds to strokes while discarding the broad luminance
    gradients of a lit seabed.
    """
    h = gray.shape[0]
    k = max(3, int(round(h * 0.35)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Overlay strokes sit far above the texture response. Otsu alone would
    # happily threshold noise on a frame containing no text, so an absolute
    # floor is applied as well.
    thr, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (tophat >= max(thr, 30)).astype(np.uint8) * 255


def _glyph_components(mask: np.ndarray, min_h: float,
                      max_h: float) -> list[tuple[int, ...]]:
    """Return components whose shape is compatible with a character."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, cv2.CC_STAT_LEFT]),
                            int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]),
                            int(stats[i, cv2.CC_STAT_HEIGHT]),
                            int(stats[i, cv2.CC_STAT_AREA]))
        if w == 0 or h == 0 or not (min_h <= h <= max_h):
            continue
        if w / h > MAX_ASPECT:
            continue
        if not (MIN_FILL <= area / float(w * h) <= MAX_FILL):
            continue
        out.append((x, y, w, h, area))
    return out


def _runs(comps: list[tuple[int, ...]]) -> list[list[tuple[int, ...]]]:
    """Chain glyph-shaped components into baseline-aligned horizontal runs."""
    runs: list[list[tuple[int, ...]]] = []
    for c in sorted(comps, key=lambda c: c[0]):
        x, y, w, h = c[0], c[1], c[2], c[3]
        baseline = y + h
        for run in runs:
            p = run[-1]
            ph, p_base = p[3], p[1] + p[3]
            same_line = abs(baseline - p_base) <= BASELINE_TOL * max(h, ph)
            same_size = min(h, ph) / max(h, ph) >= HEIGHT_RATIO
            gap = x - (p[0] + p[2])
            if same_line and same_size and -0.5 * max(h, ph) <= gap <= MAX_GAP * max(h, ph):
                run.append(c)
                break
        else:
            runs.append([c])
    return [r for r in runs if len(r) >= MIN_RUN]


def _stroke_cv(mask: np.ndarray, run: list[tuple[int, ...]]) -> float:
    """Coefficient of variation of stroke width across a run.

    Glyphs of one overlay font share a pen width; a chain of sediment blobs does
    not. This catches runs that pass the alignment test by coincidence.
    """
    widths = []
    for x, y, w, h, _ in run:
        sub = mask[y:y + h, x:x + w]
        if sub.size == 0:
            continue
        d = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
        v = d[d > 0]
        if v.size:
            widths.append(2.0 * float(v.mean()))
    if len(widths) < 2:
        return 1.0
    a = np.asarray(widths)
    return float(a.std() / max(a.mean(), 1e-6))


def _is_text_run(mask: np.ndarray, run: list[tuple[int, ...]]) -> bool:
    """Accept a run only if it is regular the way printed text is regular."""
    heights = np.array([c[3] for c in run], float)
    baselines = np.array([c[1] + c[3] for c in run], float)
    med_h = float(np.median(heights))
    if med_h <= 0:
        return False
    if baselines.std() / med_h > MAX_BASELINE_RESIDUAL:
        return False
    if heights.std() / max(heights.mean(), 1e-6) > MAX_HEIGHT_CV:
        return False
    return _stroke_cv(mask, run) <= MAX_STROKE_CV


def _text_runs(bgr: np.ndarray, min_h: float, max_h: float) -> list[list[tuple[int, ...]]]:
    """Validated glyph runs in an image region."""
    if bgr is None or bgr.size == 0:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    mask = _glyph_mask(gray)
    comps = _glyph_components(mask, min_h, max_h)
    return [r for r in _runs(comps) if _is_text_run(mask, r)]


def _run_bbox(run: list[tuple[int, ...]]) -> list[float]:
    return [float(min(c[0] for c in run)), float(min(c[1] for c in run)),
            float(max(c[0] + c[2] for c in run)), float(max(c[1] + c[3] for c in run))]


def text_score(image: np.ndarray, bbox: list[float],
               pad: float = 0.15) -> float:
    """Fraction of a detection box occupied by burned-in text.

    The box is padded, resampled to a fixed height, and searched for validated
    glyph runs; the score is the share of the *original* box area those runs
    cover. Padding matters: a box drawn tightly around two digits shows the rest
    of its line once padded, and a four-glyph run is far more distinctive than
    two isolated blobs.
    """
    if image is None or image.size == 0:
        return 0.0
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return 0.0

    px, py = bw * pad, bh * pad
    cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    cx2, cy2 = int(min(W, x2 + px)), int(min(H, y2 + py))
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 4:
        return 0.0

    s = float(np.clip(TARGET_CROP_H / crop.shape[0], 1.0, MAX_UPSCALE))
    up = (cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
          if s > 1.0 else crop)

    # Glyphs may fill most of a tight crop, so the height band is expressed
    # against the crop rather than against the frame.
    runs = _text_runs(up, min_h=max(5.0, up.shape[0] * 0.06),
                      max_h=up.shape[0] * 0.85)
    if not runs:
        return 0.0

    # Accumulate on a raster of the original box so overlapping runs cannot
    # push the fraction above 1.
    grid = np.zeros((max(1, int(bh)), max(1, int(bw))), np.uint8)
    for run in runs:
        rx1, ry1, rx2, ry2 = _run_bbox(run)
        # upscaled crop -> full image -> box-local
        gx1 = rx1 / s + cx1 - x1
        gy1 = ry1 / s + cy1 - y1
        gx2 = rx2 / s + cx1 - x1
        gy2 = ry2 / s + cy1 - y1
        ix1, iy1 = int(max(0, gx1)), int(max(0, gy1))
        ix2, iy2 = int(min(bw, np.ceil(gx2))), int(min(bh, np.ceil(gy2)))
        if ix2 > ix1 and iy2 > iy1:
            grid[iy1:iy2, ix1:ix2] = 1
    return float(grid.mean())


def find_text_regions(image: np.ndarray, min_h_frac: float = 0.008,
                      max_h_frac: float = 0.14) -> list[dict[str, Any]]:
    """Locate overlay text across a whole frame.

    Used for reporting and for the figures; the suppression path uses
    :func:`text_score` instead, which is scoped to proposed boxes.
    """
    if image is None or image.size == 0:
        return []
    h = image.shape[0]
    runs = _text_runs(image, min_h=h * min_h_frac, max_h=h * max_h_frac)
    return [{"bbox": _run_bbox(r), "count": len(r)} for r in runs]


def filter_text_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    threshold: float = SUPPRESS_COVERAGE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split detections into those to keep and those sitting on overlay text.

    Returns ``(kept, suppressed)``. Each suppressed detection carries
    ``text_coverage`` so the decision stays auditable in the trace rather than
    boxes silently disappearing.
    """
    kept, dropped = [], []
    for det in detections:
        cov = text_score(image, det["bbox"])
        if cov >= threshold:
            dropped.append(dict(det, text_coverage=round(cov, 3)))
        else:
            kept.append(det)
    return kept, dropped
