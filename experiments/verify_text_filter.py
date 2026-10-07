"""Invariant checks for the burned-in overlay-text filter.

The filter exists because the detector boxes video telemetry — timestamps, depth
readouts, the copyright watermark — and the classifier, unable to abstain, names
those boxes as organisms. Frame 03 of the demo set produced "fish" at 61%
confidence on the digits "02/10/".

Two invariants matter, and they pull against each other:

  1. Boxes drawn on overlay text are suppressed.
  2. No box drawn on anything else is suppressed. Losing a real organism is a
     worse failure than keeping a text box, so the threshold sits nearer the
     text end of the measured gap.

Run after any change to text_overlay.py or to the enhancement step, since
enhancement is what makes overlay glyphs detectable in the first place.

Usage:
    python experiments/verify_text_filter.py
"""

from __future__ import annotations

import glob
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ensemble  # noqa: E402
import text_overlay as T  # noqa: E402
from agent_controller import assess_image_quality, enhance_image  # noqa: E402
from pipeline import EnsemblePipeline  # noqa: E402

DEMO = ROOT / "demo_images"
JEDI = ROOT / "JediOrganismDetectionDataset"
SAMPLE = 200

# Hand-marked regions, read off the frames at 2x magnification.
TEXT_BOXES = [
    ("07_crowded_30_detections.jpg", (33, 10, 140, 36), "timestamp block"),
    ("07_crowded_30_detections.jpg", (85, 12, 140, 24), "one timestamp line"),
    ("07_crowded_30_detections.jpg", (380, 252, 478, 270), "(C) JAMSTEC"),
    ("06_correct_fish_for_contrast.jpg", (50, 27, 162, 45), "timestamp line"),
    ("06_correct_fish_for_contrast.jpg", (379, 342, 480, 360), "watermark"),
    ("01_hardware_called_puffin_99pct.jpg", (48, 27, 132, 56), "timestamp block"),
]
NON_TEXT_BOXES = [
    ("01_hardware_called_puffin_99pct.jpg", (200, 127, 255, 184), "the anemone"),
    ("01_hardware_called_puffin_99pct.jpg", (310, 220, 480, 360),
     "hardware, watermark inside the box"),
    ("07_crowded_30_detections.jpg", (350, 130, 430, 200), "bare seabed"),
    ("06_correct_fish_for_contrast.jpg", (256, 126, 311, 169), "a real fish"),
]

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def detect(pipe, path):
    """Boxes as production sees them: quality check, enhancement, detect, fuse."""
    im = cv2.imread(str(path))
    if im is None:
        return None, []
    q = assess_image_quality(im)
    enh = im
    if q["needs_enhancement"] or q["is_blurry"]:
        enh, _ = enhance_image(im, q)
    ra, rb = ensemble.run_dual_detection(enh, pipe.model_a, pipe.model_b)
    return enh, ensemble.fuse_detections(ra, rb, enh.shape[:2])


print("1. Marked overlay-text regions score above the suppression threshold")
for name, box, note in TEXT_BOXES:
    img = cv2.imread(str(DEMO / name))
    s = T.text_score(img, list(box))
    check(s >= T.SUPPRESS_COVERAGE, f"{note:34s} score={s:.2f} >= {T.SUPPRESS_COVERAGE}")

print("\n2. Organisms, hardware and seabed score below it")
for name, box, note in NON_TEXT_BOXES:
    img = cv2.imread(str(DEMO / name))
    s = T.text_score(img, list(box))
    check(s < T.SUPPRESS_COVERAGE, f"{note:34s} score={s:.2f} <  {T.SUPPRESS_COVERAGE}")

print("\n3. The reported failure no longer reaches the output")
pipe = EnsemblePipeline()
enh, fused = detect(pipe, DEMO / "03_seacucumber_called_fish_100pct.jpg")
kept, dropped = T.filter_text_detections(enh, fused)
check(len(dropped) == 1, f"exactly one box suppressed on frame 03 (got {len(dropped)})")
check(len(kept) == len(fused) - len(dropped), "kept + suppressed accounts for every box")
if dropped:
    d = dropped[0]
    on_text = 90 <= d["bbox"][0] <= 105 and 35 <= d["bbox"][1] <= 45
    check(on_text, f"the suppressed box is the timestamp at {[round(v) for v in d['bbox']]}")
check(any(190 <= k["bbox"][0] <= 205 for k in kept), "the real organism survives")

print("\n4. Aquarium frames carry no overlay, so nothing may be suppressed")
for name in ["11_aquarium_sharks.jpg", "12_aquarium_fish_school_44.jpg",
             "13_aquarium_sharks_closeup.jpg", "14_sample_fish.jpg",
             "15_sample_shark.jpg"]:
    enh, fused = detect(pipe, DEMO / name)
    _, dropped = T.filter_text_detections(enh, fused)
    check(not dropped, f"{name:38s} {len(fused):3d} boxes, {len(dropped)} suppressed")

print(f"\n5. Suppression stays rare across {SAMPLE} JEDI frames")
random.seed(1)
paths: list[str] = []
for split in ("coco_test_data", "coco_valid_data", "coco_train_data"):
    paths += sorted(glob.glob(str(JEDI / split / "images" / "*")))
random.shuffle(paths)

n_boxes = n_dropped = 0
for p in paths[:SAMPLE]:
    enh, fused = detect(pipe, p)
    if enh is None:
        continue
    _, dropped = T.filter_text_detections(enh, fused)
    n_boxes += len(fused)
    n_dropped += len(dropped)
rate = n_dropped / max(n_boxes, 1)
print(f"  {n_boxes} boxes, {n_dropped} suppressed ({rate:.1%})")
check(rate <= 0.03, f"suppression rate {rate:.1%} <= 3% (a higher rate means the "
                    f"filter is eating organisms)")

print("\n" + ("ALL CHECKS PASSED" if not failures
             else f"{len(failures)} CHECK(S) FAILED:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
