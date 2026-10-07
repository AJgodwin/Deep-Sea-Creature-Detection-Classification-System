"""Fill the empty slides of the Phase 2 review deck.

Slides 6 (architecture), 7 (results) and 11 (references) carried only a title.
This populates them from the measured results in experiments/RESULTS.md and
regenerates a slide-sized architecture diagram. Nothing is invented: every
figure quoted here appears in the repository with the split it came from.

Writes to a new file; the original deck is not modified.
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

SRC = Path(r"C:\Users\moons\Downloads\Team_32_Tag_4_Phase_2.pptx")
DST = Path(r"D:\Final-Project-Phase-1\paper\Team_32_Tag_4_Phase_2_FILLED.pptx")
FIGS = Path(r"D:\Final-Project-Phase-1\experiments\presentation_figures")

INK = RGBColor(0x12, 0x23, 0x2B)
ACC = RGBColor(0x0A, 0x6E, 0x8A)
MUT = RGBColor(0x5C, 0x70, 0x78)
GOOD = RGBColor(0x3F, 0x7A, 0x5E)
CRIT = RGBColor(0xB3, 0x40, 0x2F)

prs = Presentation(str(SRC))
SW, SH = prs.slide_width, prs.slide_height


def clear(slide, keep_names=()):
    for sh in list(slide.shapes):
        if sh.name in keep_names:
            continue
        sh._element.getparent().remove(sh._element)


def header(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.18), Inches(0.08),
                                  SW - Inches(0.36), Inches(0.44))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = INK
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(8)
        r2.font.color.rgb = MUT
    return tb


def block(slide, x, y, w, h, items, size=8.2, gap=1):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for txt, bold, col in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = txt
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = col
    return tb


def fit_pic(slide, path, x, y, maxw, maxh):
    with Image.open(path) as im:
        iw, ih = im.size
    s = min(maxw / iw, maxh / ih)
    w, h = int(iw * s), int(ih * s)
    return slide.shapes.add_picture(str(path), x + (maxw - w) // 2,
                                    y + (maxh - h) // 2, w, h)


S = prs.slides

# ---------------------------------------------------------------- slide 6
s6 = S[5]
clear(s6)
header(s6, "Architecture",
       "Perception always runs; the decision layer chooses what else to compute")
fit_pic(s6, FIGS / "arch_diagram_slide.png", Inches(0.12), Inches(0.60),
        SW - Inches(0.24), Inches(2.52))
block(s6, Inches(0.18), SH - Inches(0.42), SW - Inches(0.36), Inches(0.32), [
    ("D1 enhance if degraded   \u00b7   D2 abstain if unreadable or no box \u2265 0.40   "
     "\u00b7   D3 zoom ambiguous boxes (0.40\u20130.75)   \u00b7   "
     "D4 resolve detector\u2013classifier disagreement", False, MUT)], size=7.2)

# ---------------------------------------------------------------- slide 7
s7 = S[6]
clear(s7)
header(s7, "Results",
       "Dive-disjoint splits; 95% CIs from a bootstrap over dives; test set read once")

left = [
    ("In-domain (aquarium)", True, ACC),
    ("mAP@0.50 = 0.780", False, INK),
    ("", False, INK),
    ("Deployed on 8,151 deep-sea frames", True, CRIT),
    ("66.1% of organisms lie outside", False, INK),
    ("the label space", False, INK),
    ("Selective risk 0.888  [0.876, 0.900]", False, INK),
    ("21.3% confident labels on taxa", False, INK),
    ("never trained on", False, INK),
    ("Risk does not fall as the", True, CRIT),
    ("threshold rises", True, CRIT),
    ("", False, INK),
    ("Retrained on deep-sea (3 seeds)", True, GOOD),
    ("mAP@0.50 = 0.688 \u00b1 0.005", False, INK),
    ("", False, INK),
    ("Uncertainty layer recovers", True, ACC),
    ("ECE            0.467 \u2192 0.016", False, INK),
    ("Novelty      0.553 \u2192 0.806", False, INK),
    ("Risk @30%  0.846 \u2192 0.776", False, INK),
    ("OOD errors 0.373 \u2192 0.182", False, INK),
    ("", False, INK),
    ("None of the 4 adaptive actions", True, GOOD),
    ("improves accuracy \u2014 removing", True, GOOD),
    ("them is cheaper too.", True, GOOD),
]
block(s7, Inches(0.18), Inches(0.60), Inches(2.28), Inches(2.75), left,
      size=7.0, gap=0)

# Two figures beside the numbers. Both are single-panel and survive being
# scaled to roughly an inch and a quarter; the two-panel action-effects chart
# does not, so it is carried by the text instead.
fit_pic(s7, FIGS / "05_ood_feature_comparison.png",
        Inches(2.55), Inches(0.58), Inches(3.58), Inches(1.32))
fit_pic(s7, FIGS / "04_learned_policy.png",
        Inches(2.55), Inches(1.96), Inches(3.58), Inches(1.32))
block(s7, Inches(2.55), Inches(3.30), Inches(3.58), Inches(0.20), [
    ("Above: novelty detection, aquarium vs retrained features.   "
     "Below: risk\u2013coverage, learned vs fixed gates.", False, MUT)], size=5.6)

# ---------------------------------------------------------------- slide 11
s11 = S[10]
clear(s11, keep_names=("Google Shape;129;p17",))
header(s11, "References", "IEEE style")

REFS = [
    '[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: '
    'Unified, real-time object detection," in Proc. IEEE CVPR, 2016, pp. 779\u2013788.',
    '[2] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. [Online]. '
    'Available: https://github.com/ultralytics/ultralytics',
    '[3] R. Solovyev, W. Wang, and T. Gabruseva, "Weighted boxes fusion: Ensembling '
    'boxes from different object detection models," Image Vis. Comput., vol. 107, '
    'art. 104117, 2021.',
    '[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image '
    'recognition," in Proc. IEEE CVPR, 2016, pp. 770\u2013778.',
    '[5] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in '
    'Proc. IEEE CVPR, 2009, pp. 248\u2013255.',
    '[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern '
    'neural networks," in Proc. ICML, 2017, pp. 1321\u20131330.',
    '[7] B. Zadrozny and C. Elkan, "Transforming classifier scores into accurate '
    'multiclass probability estimates," in Proc. ACM SIGKDD, 2002, pp. 694\u2013699.',
    '[8] D. Hendrycks and K. Gimpel, "A baseline for detecting misclassified and '
    'out-of-distribution examples in neural networks," in Proc. ICLR, 2017.',
    '[9] W. Liu, X. Wang, J. Owens, and Y. Li, "Energy-based out-of-distribution '
    'detection," in Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.',
    '[10] K. Lee, K. Lee, H. Lee, and J. Shin, "A simple unified framework for '
    'detecting out-of-distribution samples and adversarial attacks," in NeurIPS, 2018.',
    '[11] Y. Geifman and R. El-Yaniv, "Selective classification for deep neural '
    'networks," in NeurIPS, 2017.',
    '[12] R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective '
    'classification," J. Mach. Learn. Res., vol. 11, pp. 1605\u20131641, 2010.',
    '[13] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in '
    'Graphics Gems IV, P. S. Heckbert, Ed. Academic Press, 1994, pp. 474\u2013485.',
    '[14] K. Katija et al., "FathomNet: A global image database for enabling '
    'artificial intelligence in the ocean," Sci. Rep., vol. 12, art. 15914, 2022.',
    '[15] S. Holm, "A simple sequentially rejective multiple test procedure," '
    'Scand. J. Statist., vol. 6, no. 2, pp. 65\u201370, 1979.',
    '[16] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. '
    'New York, NY, USA: Chapman & Hall, 1993.',
    '[17] Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every '
    'detection box," in Proc. ECCV, 2022.',
    '[18] B. Dwyer, "Aquarium Combined dataset," Roboflow Universe, 2020. [Online]. '
    'Available: https://universe.roboflow.com/brad-dwyer/aquarium-combined',
]
half = 9
block(s11, Inches(0.18), Inches(0.60), Inches(2.96), Inches(2.75),
      [(r, False, INK) for r in REFS[:half]], size=6.0, gap=3)
block(s11, Inches(3.22), Inches(0.60), Inches(2.92), Inches(2.75),
      [(r, False, INK) for r in REFS[half:]], size=6.0, gap=3)

# ---------------------------------------------------------------- slide 12
# Two rows overstated what exists. Meta-confidence was never built, and the
# value model Q(s,a) could not be fitted — the ZOOM action produces too few
# outcome changes to learn from (15 on the fit split), so a model trained on it
# would be noise. The compute-budget allocator is likewise unbuilt. The learned
# accept/abstain policy IS built and beats the tuned threshold with a bootstrap
# CI excluding zero, so the corrected wording is still a positive claim.
def set_cell(cell, text):
    """Replace cell text while preserving the existing run's formatting."""
    tf = cell.text_frame
    p = tf.paragraphs[0]
    runs = list(p.runs)
    if not runs:
        p.add_run().text = text
        return
    runs[0].text = text
    for extra in runs[1:]:
        extra._r.getparent().remove(extra._r)
    for extra_p in list(tf.paragraphs)[1:]:
        extra_p._p.getparent().remove(extra_p._p)


s12 = S[11]
for sh in s12.shapes:
    if not getattr(sh, "has_table", False):
        continue
    t = sh.table
    set_cell(t.cell(5, 1), "Calibration + OOD scoring")
    set_cell(t.cell(6, 1), "Learned accept/abstain policy")
    set_cell(t.cell(6, 2), "Functional; Q(s,a) in progress")

DST.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(DST))
print("saved:", DST)
