# An Agentic Multi-Model Deep Learning Framework for Underwater Creature Detection and Classification

**Structured answers to the paper template.** Scope: still-image pipeline only.
Video extension is deliberately excluded and reserved for a separate paper.

**Status conventions used throughout.** Every quantitative claim is one of:

- **[M]** measured in this work, with the split and interval stated
- **[M-dev]** measured on a development split only; not confirmed on held-out test
- **[P]** pending — designed but not yet executed; stated as such, never estimated

No number in this document is inferred, rounded from memory, or carried over
from another study.

---

# 1. ABSTRACT

## Draft abstract (single paragraph)

Automated recognition of marine organisms from underwater imagery is a
prerequisite for scaling ecological survey work beyond what expert annotators
can process, but a recognition system is only useful to a survey if the rate at
which it is wrong is known and bounded. Underwater imagery is degraded by
wavelength-dependent attenuation, turbidity, backscatter, non-uniform artificial
illumination and motion blur, and the organisms themselves are frequently
occluded, cryptic, and distributed over a wide range of apparent scales. Single
model detection pipelines respond to this by emitting a class label and a
confidence score for every region they propose, with no mechanism for declining
to answer and no mechanism for recognising an organism outside their label
space; the resulting errors are indistinguishable downstream from valid
observations. This work proposes an agentic multi-model framework in which
perception is separated from the decision to commit to an answer. Two YOLOv8
detectors (Nano and Small) are trained under different random seeds and
different augmentation regimes so that their failure modes decorrelate, and
their proposals are combined by Weighted Boxes Fusion rather than
Non-Maximum Suppression, so that agreement between independently trained models
is preserved as evidence rather than discarded. Every fused region is
re-examined by an independent ResNet18 classifier, and detector and classifier
confidence are combined under a fixed weighting with an agreement bonus. A
rule-based agentic controller governs four runtime decisions: whether the image
requires enhancement, whether the evidence is sufficient to proceed at all,
whether an individual detection is ambiguous enough to warrant re-detection at
higher effective resolution, and how to resolve detector–classifier
disagreement through a three-opinion majority vote. Every branch taken is
emitted as a machine-readable decision trace. The system is trained and
validated on the Aquarium Combined dataset (638 images, seven taxa) and is then
evaluated, unmodified, on 8,151 frames of real deep-sea ROV imagery in which
66.1% of annotated organisms fall outside its label space. In-domain the
pipeline attains mAP@50 of 0.780; under domain shift its selective risk is
0.888 (95% CI [0.876, 0.900]) and 21.3% of accepted detections are confident
species labels assigned to organisms it was never trained on. Critically, the
error rate does not fall as the confidence threshold is raised, establishing
that the confidence score carries almost no information about correctness in
this regime. Forced-counterfactual measurement of each discretionary action
shows that none of the four adaptive behaviours improves accuracy, and that
conditional enhancement measurably degrades it. Isotonic calibration reduces
expected calibration error from 0.467 to 0.016, and out-of-distribution scoring
computed from the classifier's penultimate representation attains AUROC 0.806
for unseen taxa where every confidence-derived score is at chance, demonstrating
that the information required for novelty rejection is present in the network
and is destroyed by the classification head. A learned state-conditional
accept/abstain policy reduces selective risk from 0.846 to 0.776 at matched
coverage and independently assigns novelty distance approximately five times the
weight of calibrated confidence. The contribution is therefore twofold: an
agentic architecture in which every adaptive action is instrumented and
falsifiable, and the empirical finding that hand-specified adaptive rules in
such a pipeline cannot be assumed to help and, when measured, do not.

## Question-by-question basis for the abstract

**What is deep-sea creature detection and classification?**
It is the joint task of localising marine organisms within underwater imagery
(detection: predicting bounding-box coordinates for each organism present) and
assigning each localised organism a taxonomic label (classification), typically
to genus or morphotype level, on imagery acquired by remotely operated or
autonomous underwater vehicles.

**Why is automatic aquatic species recognition important?**
Benthic and midwater survey programmes generate imagery at a rate that
substantially exceeds expert annotation capacity. Manual annotation is the
principal bottleneck in converting recorded transects into occurrence records
usable for biodiversity assessment, habitat mapping and long-term monitoring.
Automation is attractive only if its error characteristics are quantified,
because an unflagged false occurrence record propagates into downstream
ecological analysis and is not distinguishable from a verified observation.

**What challenges exist in underwater image analysis?**
Wavelength-dependent attenuation (red light is absorbed within the first few
metres, producing a progressive blue–green cast with depth), turbidity and
suspended particulate matter causing backscatter and contrast loss, non-uniform
artificial illumination from vehicle-mounted lighting, motion blur from platform
movement, occlusion by substrate and conspecifics, cryptic coloration, and wide
variation in apparent object scale arising from uncontrolled camera-to-subject
distance.

**What problems exist in current single-model detection or classification
systems?**
A single detector supplies one opinion per region with no independent check.
Its confidence score is an uncalibrated softmax-derived quantity and is not a
probability of correctness. The label space is closed, so any organism outside
it is necessarily assigned an incorrect in-vocabulary label rather than being
rejected. Control flow is fixed: the same computation is applied to a clear
image of a single large organism and to a degraded image of a crowded scene.

**What is the main research gap?**
Adaptive or "agentic" processing has been proposed for vision pipelines, but the
adaptive actions are specified by hand and their benefit is assumed. We are not
aware of prior underwater recognition work that measures each discretionary
action against the counterfactual of not performing it, under a matched compute
accounting, on a leakage-free split.

**What new system do you propose?**
A dual-detector, fusion-based, independently-classified pipeline under a
rule-based agentic controller, fully instrumented so that every decision and its
cost is recorded, together with a calibrated uncertainty layer and an
out-of-distribution scorer that permit principled abstention.

**Why do you use two YOLOv8 detection models?**
Not for capacity but for error decorrelation. Agreement between two models
trained under different seeds and different augmentation regimes constitutes
evidence that a single confident model cannot provide. Two models sharing a
failure mode agree on their errors and their agreement is uninformative;
inducing diversity is therefore the design objective.

**Why do you use ResNet18 for classification?**
Detection and fine-grained recognition are different problems. The detector
optimises localisation jointly with a coarse class assignment over the full
frame; a dedicated classifier operating on a cropped region at fixed resolution
can allocate its capacity to appearance discrimination. ResNet18 is selected as
the smallest residual architecture that retains ImageNet-pretrained features
adequate for transfer at this dataset scale, which matters because the
classification training set is small and imbalanced.

**What is the purpose of Weighted Boxes Fusion?**
Non-Maximum Suppression resolves overlapping proposals by discarding all but the
highest-scoring box, which destroys the agreement signal the ensemble exists to
produce. WBF instead forms a confidence-weighted average of the overlapping
coordinates and combines their scores, so that a region proposed by both models
is reinforced and a region proposed by one is down-weighted rather than deleted.

**Why is an agentic rule-based controller required?**
Because the appropriate amount of computation is a function of the image. A
confident detection of a large, well-lit organism requires nothing further; an
ambiguous detection in a degraded frame may justify additional inference. A
fixed pipeline cannot express that distinction. The controller makes the
allocation explicit and, being explicit, makes it measurable.

**How does image-quality assessment improve the system?**
It provides the state on which the first and second decisions are conditioned:
Laplacian variance, mean intensity and intensity standard deviation are computed
before any network is invoked, at a measured mean cost of 7.10 ms, and determine
whether enhancement is applied and whether the image is admissible at all. **[M]**

**What is the purpose of the zoom-and-recheck mechanism?**
To re-examine detections whose confidence places them in an ambiguous band by
re-running detection on a padded crop, which presents the region to the network
at higher effective resolution. Its measured effect is reported in §4.7 and is
not positive.

**How does the majority voting mechanism resolve model disagreement?**
When detector and classifier disagree, a third opinion is obtained by
re-classifying a more loosely cropped region, and the modal class of the three
opinions is taken, defaulting to the detector class under a three-way tie. Its
measured effect relative to simply trusting the classifier is reported in §4.8
and is not statistically distinguishable from zero.

**What dataset is used for training and evaluation?**
Aquarium Combined (Roboflow, `brad-dwyer/aquarium-combined` v2, CC BY 4.0; 638
images, seven classes) for training and in-domain validation, and the Jedi
Organism Detection Dataset (8,151 images, 3,152 dives, 15,621 annotations, 20
classes) for out-of-domain and open-set evaluation. **[M]**

**What aquatic species are considered?**
In-domain: fish, jellyfish, penguin, puffin, shark, starfish, stingray. Out of
domain, the evaluation label space comprises 20 deep-sea classes of which four
map onto the training taxonomy and 16 do not.

**What are the main experimental results?**
In-domain mAP@50 0.780. Out-of-domain selective risk 0.888 [0.876, 0.900] with
21.3% of accepted detections confidently mislabelling organisms outside the
label space. Calibration error 0.467 → 0.016. Novelty AUROC 0.806 from
representation-space scoring against ~0.47 from confidence. Learned gate reduces
selective risk 0.846 → 0.776 at matched coverage. None of four discretionary
actions improves accuracy. **[M]**, **[M-dev]** as indicated in §4.

**How does the proposed system perform compared with individual models?**
The full ablation ladder specified in §4.5 has not yet been executed **[P]**. The
component-level counterfactual measurements in §4.6–§4.8 are complete and
constitute the substantive comparison currently available.

**What is the main advantage of the proposed method?**
That every adaptive decision is instrumented, costed and falsifiable, which
converts architectural assumptions into testable claims — and, in this
instance, falsifies several of them.

---

# 2. INTRODUCTION

## 2.1 Background

**What is underwater computer vision?**
The application of image-analysis methods to imagery formed in an aquatic
medium, in which the optical channel itself is a significant and
depth-dependent source of degradation, distinguishing it from terrestrial
computer vision where the medium is approximately transparent.

**What is aquatic species detection?**
Prediction of bounding-box coordinates enclosing each organism present in a
frame, together with an objectness or class score, without prior knowledge of
the number of organisms.

**What is aquatic species classification?**
Assignment of a taxonomic label to a region already known to contain an
organism.

**What is the difference between object detection and image classification?**
Classification maps an image to a single label over a fixed set. Detection maps
an image to a variable-length set of localised labelled regions, and must
therefore additionally solve localisation and instance separation. In this work
the two are deliberately separated into distinct networks so that each may be
optimised, and evaluated, independently.

**Why is automatic marine-species recognition important?**
See §1. The operative constraint is not throughput alone but trustworthiness:
an automated pipeline whose error rate is unquantified cannot be admitted into
a survey workflow regardless of throughput.

**Where can deep-sea creature detection systems be applied?**
Benthic habitat mapping, biodiversity and abundance estimation, environmental
impact assessment around seabed infrastructure, long-term monitoring of
protected areas, and on-vehicle triage to prioritise which imagery is
transmitted or retained under bandwidth or storage constraints.

**Why is deep learning suitable for underwater image analysis?**
Because the discriminative features of marine organisms under variable optical
conditions are not readily expressible as hand-designed descriptors, and
convolutional feature hierarchies transfer effectively from large terrestrial
corpora to this domain given limited in-domain data. This work also documents
the limits of that transfer.

**What types of aquatic species are considered in this research?**
Seven aquarium taxa in domain (§4.1), and 20 deep-sea classes for open-set
evaluation spanning fish, cnidarians, echinoderms, crustaceans, molluscs,
poriferans and non-biological submersible hardware.

## 2.2 Problem

**Why is underwater creature detection difficult?**
Because both the imaging channel and the subject are adverse. The channel
attenuates, scatters and colour-shifts; the subject is frequently cryptic,
occluded, deformable and observed at uncontrolled scale.

**How does poor underwater illumination affect detection?**
Vehicle-mounted lighting produces a non-uniform field with a bright centre and
strong falloff, so the same organism has different appearance statistics
depending on its position in frame. Measured on controlled degradation, reducing
image intensity to 25% of nominal raised selective risk from 0.364 to 0.489
while the abstention rate rose only to 10.0%, indicating that the pipeline
largely fails to recognise the condition. **[M]**

**How does turbidity affect image quality?**
Suspended particulates scatter light into the optical path, reducing contrast
and introducing spurious high-frequency structure that is a plausible source of
the 41.9% background false-positive rate observed under domain shift. **[M]**

**Why is low contrast a major problem?**
It compresses the intensity range separating organism from substrate. Measured
at 25% of nominal contrast, selective risk rose from 0.364 to 0.533 with the
abstention rate at 11.1%. **[M]**

**How does blur affect object detection?**
It attenuates the high-frequency content on which edge- and texture-sensitive
filters depend. This is the one degradation the pipeline explicitly tests for,
and it is correspondingly the one it handles well: under severe Gaussian blur
the abstention rate rose from 1.6% to 83.2% while selective risk remained
approximately unchanged (0.364 → 0.393), i.e. the system declined to answer
rather than answering incorrectly. **[M]**

**How does occlusion make species recognition difficult?**
Partial occlusion removes discriminative regions and biases the fused box toward
the visible fraction, which then propagates a mis-scaled crop to the classifier.

**Why does camouflage create difficulties for detection?**
Cryptic coloration minimises the very contrast the detector relies upon;
crinoids in this dataset are a concrete instance, attaining AP@50 of only 0.223
after on-domain retraining, against 0.800 for sea-cucumber at fewer training
annotations. **[M]**

**How do changes in object size affect YOLO detection?**
Detection quality degrades for objects occupying few pixels after the network's
input rescaling. This work observes the converse effect directly: re-detecting
the full frame at 1280 px rather than the trained 640 px produced substantially
more boxes but fewer correct ones (−0.0453 correct detections per image, 95% CI
[−0.0577, −0.0328]), consistent with a train/test resolution mismatch. **[M-dev]**

**Why can visually similar aquatic species be difficult to classify?**
Because interclass appearance distance may be smaller than intraclass variation
induced by pose, occlusion and optical condition. This is also why near-OOD
detection is harder than far-OOD in this work: distinguishing an unseen
invertebrate from a known invertebrate (AUROC 0.806) is a harder problem than
distinguishing submersible hardware from any organism (0.816). **[M-dev]**

**Underwater challenges considered.** Low illumination, turbidity, blur, low
contrast, occlusion, camouflage, object-scale variation, background complexity
and colour distortion. Seven of these are evaluated quantitatively under
controlled degradation in §4.10; colour distortion is found to be the condition
the system is least able to recognise, which is significant because it is the
condition most characteristic of increasing depth.

## 2.3 Existing Methods

**What traditional computer vision methods have been used for aquatic species
detection?** Background subtraction and frame differencing for moving-subject
video, thresholding and morphological segmentation, hand-designed descriptors
(SIFT, SURF, HOG) with classical classifiers, and template matching. These
degrade sharply under the illumination and contrast variation characteristic of
the domain.

**What deep learning methods are commonly used for underwater object
detection?** Single-stage detectors of the YOLO and SSD families, two-stage
detectors of the R-CNN family, and more recently transformer-based detectors
such as DETR and RT-DETR.

**Why are YOLO-based models widely used?**
Favourable accuracy-to-latency ratio, single-pass inference suitable for
embedded and on-vehicle deployment, and mature tooling for transfer learning at
small dataset scale.

**What classification networks are commonly used for species recognition?**
Residual networks (ResNet family), densely connected networks, EfficientNet, and
increasingly Vision Transformers where sufficient data is available.

**What are the limitations of using a single YOLO model?**
One opinion per region, no independent verification, an uncalibrated confidence
score, and a closed label space with no rejection option.

**What are the limitations of using only a classifier?**
A classifier presupposes localisation and cannot enumerate instances, so it
cannot be applied to a scene containing an unknown number of organisms.

**Why can detector confidence alone be unreliable?**
This work measures the failure directly. Under domain shift the mean reported
confidence was 0.558 while the empirical accuracy of the same detections was
0.091 — an overconfidence of +0.467 in expectation — and expected calibration
error was 0.4669. More consequentially, confidence carried no usable information
about novelty: AUROC for separating unseen from known taxa was 0.4693 using
maximum softmax probability, 0.4356 using detector confidence and 0.4383 using
the fused score, all at or below chance. **[M-dev]**

**Why is model fusion useful in underwater object detection?**
Because independently trained models make partly independent errors, so
concordance is informative. The value of fusion is contingent on that
independence actually obtaining, which is why the two detectors here differ in
seed and augmentation regime rather than in capacity alone.

**What are the limitations of conventional Non-Maximum Suppression?**
NMS is a selection operator: among mutually overlapping proposals it retains one
and discards the remainder. Consequently the information that two independent
models proposed the same region is destroyed at precisely the point where it is
most useful, and a marginally lower-scoring but better-localised box is lost.

## 2.4 Research Gap

**What is missing in existing underwater creature detection systems?**
An explicit, instrumented and falsifiable account of the decision layer.
Pipelines commonly include conditional enhancement, re-detection or ensemble
tie-breaking, but report only end-to-end metrics, from which the contribution of
any individual adaptive rule cannot be recovered.

**Do existing systems dynamically change their processing according to image
quality?** Some apply enhancement conditionally, but we are not aware of work
that quantifies the effect of that conditioning against the counterfactual. The
present work does, and finds the effect negative (§4.6).

**Do existing systems combine multiple heterogeneous detection models?**
Ensembling is common. Deliberate induction of diversity through differing seeds
and augmentation regimes, with the fusion operator chosen to preserve rather
than discard agreement, is less so.

**Is Weighted Boxes Fusion widely integrated with independent species
classifiers?** WBF is established in detection ensembling literature. Its
composition with a separate classification stage whose disagreement with the
detector is itself treated as an actionable signal is the specific configuration
examined here.

**Do existing systems reanalyse uncertain detections automatically?**
Mechanisms of this kind exist. What is absent is measurement of whether the
reanalysis changes outcomes: in this work the deployed rule fired on 641
detections, consumed 18.9 s, and altered the outcome of nine. **[M-dev]**

**How do existing systems handle disagreement between detection and
classification models?** Typically by fixed precedence or by score comparison.
The three-opinion vote used here is a more elaborate alternative and is measured
against the two trivial precedence policies in §4.8.

**Do existing systems provide an explainable runtime decision trace?**
Explainability work in this domain concentrates on saliency and attribution over
model outputs. A trace of the *control-flow* decisions — which branches
executed, why, and at what cost — is the form of transparency required to audit
an adaptive pipeline, and is what this system emits.

**Why is an adaptive or agentic processing framework required?**
Because the marginal value of additional computation varies per detection, so a
uniform allocation is necessarily inefficient. The framework's value, however,
is contingent on the adaptive actions being beneficial, which this work is the
first to test in this setting — with substantially negative results that
motivate replacing hand-specified rules with a learned policy.

## 2.5 Proposed Idea

**What is the proposed system?**
A five-stage instrumented pipeline: quality assessment and conditional
enhancement; dual YOLOv8 detection; Weighted Boxes Fusion; independent ResNet18
classification of each fused region; and confidence fusion with
disagreement resolution — all under a rule-based controller that emits a
decision trace, and extended by a calibration and out-of-distribution layer
supporting principled abstention.

**Why are YOLOv8-Nano and YOLOv8-Small combined?**
To obtain decorrelated errors. They differ in capacity, in random seed (42 and
123) and in augmentation regime, the Small variant additionally employing mixup
and copy-paste.

**What is the role of ResNet18?**
To supply a second, architecturally and functionally independent opinion on the
taxonomic identity of each fused region, and thereby to make detector–classifier
disagreement available as a signal.

**What is the role of Weighted Boxes Fusion?**
To combine the two detectors' proposals in a manner that preserves agreement as
evidence, using a confidence-weighted coordinate average with model weights
[2, 1] at an IoU threshold of 0.55.

**What is the role of the agentic controller?**
To select, per image and per detection, which discretionary computations to
perform, and to record those selections.

**How does the system handle poor-quality images?**
Through conditional enhancement (CLAHE on the L channel; sharpening when
Laplacian variance is below 100) and, below a Laplacian variance of 5.0, through
outright refusal, in which classification is never invoked and a review flag is
returned in place of predictions.

**How does the system handle uncertain detections?**
Detections with confidence in [0.40, 0.75) trigger re-detection on a 15%-padded
crop, with the result adopted only if it overlaps the original at IoU > 0.4 and
scores strictly higher.

**How does the system handle model disagreement?**
Through the three-opinion majority vote described in §3.13.

**What is the main novelty of the proposed work?**
Two claims. Architecturally, an agentic recognition pipeline in which every
discretionary action is instrumented, costed and independently falsifiable.
Empirically, the demonstration by forced counterfactual that the hand-specified
adaptive rules in such a pipeline do not improve accuracy — and that the
information required for novelty rejection, absent from the confidence score, is
recoverable from the classifier's internal representation.

**What are the major contributions of this research?**

1. An agentic multi-model underwater recognition architecture with an
   explicit decision trace and per-action cost accounting.
2. A leakage-free evaluation protocol for the deep-sea corpus, following an
   audit that found 57.3% of the supplied test partition shared a dive with
   training images. **[M]**
3. Quantification of out-of-domain failure: selective risk 0.888, with the
   out-of-distribution error rate *increasing* as the confidence threshold is
   raised. **[M]**
4. A forced-counterfactual measurement protocol for discretionary actions, and
   the finding that none of four such actions improves accuracy. **[M-dev]**
5. Demonstration that novelty information survives in the penultimate
   representation (AUROC 0.806) while being absent from every confidence-derived
   score (≈0.47), localising the deficiency to the classification head.
   **[M-dev]**
6. A learned state-conditional accept/abstain policy that reduces selective risk
   at matched coverage and independently recovers the same conclusion, weighting
   novelty distance approximately five times calibrated confidence. **[M-dev]**

---

# 3. PROPOSED WORK

## 3.1 Overall System

**Input.** A single RGB still image, accepted as JPEG, PNG or WebP, to a maximum of 20 MB.

**Main stages.** (i) image-quality assessment; (ii) conditional enhancement; (iii) parallel detection by YOLOv8-Nano and YOLOv8-Small; (iv) Weighted Boxes Fusion; (v) review gate; (vi) zoom-and-recheck on ambiguous detections; (vii) ResNet18 classification of each region; (viii) detector–classifier agreement resolution by three-opinion majority vote; (ix) combined confidence scoring; (x) annotation and emission of the decision trace.

**Detection.** Both detectors run on the same (possibly enhanced) image at a confidence threshold of 0.25 and an intra-model NMS IoU of 0.45.

**Classification.** Each fused box is cropped with 5% padding, resized to 224×224, normalised with ImageNet statistics, and passed to ResNet18.

**Combination of predictions.** Bounding boxes are combined by WBF; class and confidence are combined by the weighted rule of §3.11 with an agreement bonus.

**Controller decision.** Four decisions are taken on measured state: image quality statistics (D1, D2), per-detection fused confidence (D3), and detector–classifier label equality (D4).

**Final output.** An annotated image, a list of detections each carrying bounding box, detector class and confidence, classifier class and confidence, combined confidence and an agreement flag, a summary block containing object count and per-stage timings, and the decision trace.

**Pipeline.** Input → Quality Assessment → Conditional Enhancement → YOLOv8-Nano + YOLOv8-Small → Weighted Boxes Fusion → Review Gate → Zoom & Recheck → ResNet18 → Majority Voting → Confidence Scoring → Final Detection + Decision Trace.

## 3.2 Image Quality Assessment

**Why required.** It supplies the state variable on which the enhancement and admissibility decisions are conditioned, and must be cheap enough to precede network invocation. Measured mean cost 7.10 ms (median 3.05, p95 10.26, max 91.00) against 88.97 ms for detection. **[M]**

**Parameters measured.** Blur (variance of the Laplacian response), brightness (mean grayscale intensity), contrast (standard deviation of grayscale intensity), and frame resolution.

**How blur is measured, and why Laplacian variance.** The image is converted to grayscale and convolved with a Laplacian kernel; the variance of the response is taken. The Laplacian is a second-derivative operator and therefore responds to the high-spatial-frequency content that optical blur attenuates first, so the variance of its response falls monotonically with defocus. It requires one convolution and one variance computation, hence its suitability as a pre-network gate.

**How brightness is calculated.** Arithmetic mean of grayscale intensity over all pixels.

**How contrast is measured.** Standard deviation of grayscale intensity, i.e. RMS contrast.

**Is resolution checked.** Yes, recorded as (width, height) in the quality state and emitted in the trace, though no threshold is currently applied to it.

**Thresholds.** Blur: sharpening below 100; admissibility below 5.0. Brightness: below 50 or above 200 triggers CLAHE. Contrast: below 30 triggers CLAHE.

**How an unreadable image is identified, and why rejected.** Laplacian variance below 5.0 is treated as unreadable. The justification is evidential rather than performance-driven: below that threshold the image does not contain sufficient high-frequency structure for any detection to be trustworthy, so the appropriate response is to decline rather than to emit predictions that cannot be verified. Classification is never invoked and a review flag is returned. This is the only terminal decision in the pipeline.

## 3.3 Conditional Image Enhancement

**Why required.** To recover contrast lost to scattering and to compensate non-uniform illumination before detection.

**Is it applied to every image.** No. The conditionality is the design point, and §4.6 shows it remains the correct design even though the transform itself does not help.

**How the decision is made.** CLAHE when mean brightness is below 50 or above 200, or contrast is below 30; sharpening when Laplacian variance is below 100. The two are independent and may both fire.

**What is CLAHE.** Contrast Limited Adaptive Histogram Equalisation: histogram equalisation applied over local tiles rather than globally, with a clip limit bounding the amplification of any single histogram bin, thereby limiting noise amplification in near-uniform regions.

**Why CLAHE for underwater images.** Underwater degradation is spatially non-uniform — a vehicle-lit frame is bright at the centre and dark at the periphery — so a global transform cannot correct both regions simultaneously.

**Why LAB before CLAHE, and which channel.** LAB separates lightness (L) from chromaticity (a, b). Equalising L alone adjusts luminance while leaving hue relationships intact; equalising in RGB would alter colour balance and introduce chromatic artefacts.

**When and how sharpening is applied.** When Laplacian variance is below 100, by convolution with a 3×3 high-pass kernel.

**How enhancement affects detection.** Measured, and negatively: see §4.6. The mechanism is retained in the architecture and reported honestly rather than removed, because its measurement is a principal result of this work.

**Why conditional enhancement is better than unconditional.** Because enhancement is neither free nor universally beneficial. The measurements in §4.6 strengthen this argument rather than weakening it: since the transform degrades accuracy on average, applying it universally would be worse than applying it selectively — though the measured trigger selects, if anything, the images on which it is most harmful.

## 3.4 YOLOv8-Nano Object Detection

**What is YOLOv8, and YOLOv8-Nano.** YOLOv8 is a single-stage anchor-free detector with a CSP-derived backbone, a PAN-FPN neck for multi-scale feature aggregation, and a decoupled detection head. Nano is its smallest variant, approximately 3.2 M parameters and 8.1 GFLOPs at 640 px.

**Why selected, and its advantages.** Low latency and memory footprint, suitable for on-vehicle deployment, and — for the ensemble — a capacity and inductive bias sufficiently different from the Small variant to contribute error diversity.

**Training dataset.** Aquarium Combined (§4.1).

**Augmentation.** Mosaic only. The restriction is deliberate: the Small variant receives a strictly richer regime so the two models differ in more than size.

**Random seed.** 42.

**Epochs.** 50, early-stopping patience 15.

**Output.** Bounding boxes in image coordinates, a class index and a confidence per box, filtered at 0.25 with intra-model NMS IoU 0.45.

**Information per detection.** Box coordinates (x1, y1, x2, y2), class index, class name and confidence.

## 3.5 YOLOv8-Small Object Detection

**What is YOLOv8-Small.** The next variant up, approximately 11.2 M parameters and 28.6 GFLOPs at 640 px.

**Why used with Nano, and how it differs.** It differs in capacity, seed and augmentation. The purpose is decorrelation of errors, not accuracy alone.

**Augmentation.** Mosaic, plus mixup at 0.15 and copy-paste at 0.1.

**Why mosaic, mixup and copy-paste.** Mosaic composites four images, exposing the model to unusual scale and context combinations. Mixup blends image pairs and their labels, discouraging over-confident decision boundaries. Copy-paste transplants instances between images, increasing instance density and occlusion diversity — directly relevant to crowded benthic scenes.

**Random seed.** 123.

**Why different seeds.** Seed governs weight initialisation, data ordering and augmentation sampling. Distinct seeds are a necessary, though not sufficient, condition for the models to reach different minima and therefore make partly independent errors.

**How differing training creates prediction diversity.** Different initialisation and different augmentation distributions induce different decision boundaries, so the models fail on different subsets of the input distribution. The ensemble's value is a monotone function of that non-overlap.

**Output.** Identical in form to §3.4.

## 3.6 Dual-Model Detection

**Why two detectors.** So that agreement carries information. A single confident model provides no means of distinguishing a well-supported detection from a confident error.

**Do both process the same image.** Yes — the same (possibly enhanced) frame under identical thresholds, so any difference in output is attributable to the models rather than to their inputs.

**Why model diversity matters.** If the two models shared a failure mode they would agree on their errors and their concordance would be uninformative; the ensemble would then add cost without adding evidence.

**If the models produce different boxes.** WBF forms a confidence-weighted average of the overlapping coordinates, generally better localised than either input.

**If one detects and the other does not.** The isolated proposal survives into the fused set at a reduced score, since only one model supports it. It is down-weighted rather than deleted — precisely the behaviour NMS cannot express.

**Transfer to fusion.** Both raw result sets, with the frame dimensions required for coordinate normalisation, are passed to the fusion stage.

**Does it improve reliability.** In principle yes; a controlled ablation isolating the ensemble's contribution is specified in §4.5 and has not been executed **[P]**.

## 3.7 Weighted Boxes Fusion

**What it is.** An ensembling method that clusters overlapping proposals across models and replaces each cluster with a single box whose coordinates are the confidence-weighted mean of the cluster and whose score reflects the number and confidence of contributing models.

**Why used.** To retain the agreement signal that motivates the ensemble.

**Difference from NMS.** NMS selects and discards; WBF averages and reweights. Under NMS a cluster contributes exactly one original box; under WBF it contributes a new box informed by all members.

**Input.** The two detectors' proposal sets with coordinates normalised to [0, 1].

**Model weights and rationale.** [2, 1], favouring Nano. The weighting is inherited from the frozen baseline configuration; it is not the outcome of a sweep, and a sensitivity analysis over this parameter is outstanding **[P]**.

**IoU threshold.** 0.55 for cluster membership.

**How overlapping boxes are identified.** Pairwise IoU against existing cluster representatives; a proposal exceeding 0.55 joins that cluster, otherwise it seeds a new one.

**How coordinates and confidences are combined.** Coordinates: mean weighted by the product of model weight and box confidence. Confidence: weighted mean of the cluster's scores (`conf_type = avg`), scaled by cluster support, so a box found by both models scores above one found by a single model at equal raw confidence.

**Isolated detections.** Retained at reduced score, subject to a skip threshold of 0.001.

**Output.** A single fused proposal set in absolute image coordinates.

## 3.8 Detection Review Gate

**Why required.** To separate the question *is there sufficient evidence to analyse this image at all?* from *which detections should be reported?* Only the former is terminal.

**Minimum confidence.** 0.40 on the fused score.

**Why 0.40.** It is the operating point of the frozen baseline. It is acknowledged as a hand-set constant, and the calibration work of §4.9 together with the learned gate of §4.11 exist precisely because a constant cannot condition on image state.

**When no detection reaches 0.40.** The run terminates. Classification is not invoked, no boxes are returned, and a review flag with the measured reason is emitted.

**Why classification is skipped.** Because classifying regions the detector does not support would produce labels with no evidential basis, and their presence in the output would imply a confidence the system does not have.

**When at least one reliable detection exists.** Processing continues, and *all* detections are carried forward, not only those above threshold.

**Why weaker boxes are retained.** The gate decides whether the image is worth analysing, not which detections survive. Filtering here would truncate the confidence distribution and make the risk–coverage analysis of §4.9 impossible, since the low-confidence tail is required to trace the curve.

## 3.9 Zoom-and-Recheck

**What it is.** Re-detection of an ambiguous region on a padded crop, presenting that region to the network at higher effective resolution.

**Why required, as designed.** A small or low-contrast organism may occupy too few pixels after input rescaling for confident detection; re-detection on a crop increases its apparent size.

**Which detections are ambiguous, and the activation range.** Fused confidence in [0.40, 0.75) — above the review threshold but below the level treated as confident.

**Padding.** 15% of box width and height on each axis.

**Why 15%.** To include immediate context, which aids both localisation and subsequent classification, without admitting a neighbouring instance. The value is hand-set and not the result of a sweep **[P]**.

**Are both detectors rerun, and is WBF reapplied.** Yes to both: the crop is processed by the identical detection and fusion path, so the action is a genuine re-execution rather than a different operator.

**Mapping back.** Crop-local coordinates are offset by the crop origin to return to frame coordinates.

**IoU and replacement condition.** IoU is computed between the original box and each remapped candidate; a candidate is eligible at IoU > 0.4, and the original is replaced only if the candidate additionally scores strictly higher.

**Why the new confidence must be higher.** So the action can fail safely: a re-check that finds nothing better leaves the detection untouched, bounding the harm the action can do. §4.7 shows the bound is not zero, since a replaced box may be worse by criteria the guard does not measure.

**How it improves uncertain detections.** As designed, by resolution. As measured, it does not: see §4.7.

## 3.10 ResNet18 Classification

**What is ResNet18.** An 18-layer convolutional network with residual connections, permitting gradient flow through depth and hence stable training.

**Why used here.** Its capacity is matched to the available classification data (4,180 crops, severely imbalanced) and ImageNet-pretrained features transfer adequately. A larger backbone would overfit at this scale.

**Input.** A cropped region, resized to 224×224 and normalised with ImageNet channel statistics.

**How crops are generated, and why padding.** The fused box is expanded by a padding fraction and clipped to frame bounds. Padding is applied because a tight box frequently truncates discriminative extremities — fins, arms, tentacles — and because it supplies local context.

**Padding percentage.** 5% on the primary classification pass; 15% on the tie-break pass of §3.13.

**Pretraining.** Yes, ImageNet-1k.

**Final layer.** The 1000-way fully connected layer is replaced by a linear layer of width equal to the class count (7 in domain; 8 for the retrained deep-sea variant).

**Classes.** fish, jellyfish, penguin, puffin, shark, starfish, stingray.

**Output and confidence.** A softmax distribution over classes; the predicted class is the argmax and the classifier confidence is the corresponding probability. This is the quantity shown in §4.9 to be uninformative about novelty.

## 3.11 Combined Confidence Score

**Why combined.** Because detector and classifier assess different evidence — the detector, objectness and coarse class over the frame; the classifier, appearance within an isolated region — so their concordance is informative.

**Detector confidence.** The fused WBF score for the region.

**Classifier confidence.** The maximum softmax probability from ResNet18.

**Equation.**

    C = min[(0.6·c_d + 0.4·c_c) × (1 + B), 1.0]

where c_d is detector confidence, c_c classifier confidence, and B the agreement bonus.

**Why 0.6 for the detector.** The detector's score reflects both localisation and classification evidence, whereas the classifier is conditioned on a localisation already assumed correct; a mis-localised box yields a confident but meaningless classification. The weighting is a hand-set design constant, not a fitted parameter.

**Why 0.4 for the classifier.** It is the complement, reflecting a deliberately secondary but non-trivial contribution.

**Agreement bonus.** B = 0.10 when detector and classifier predict the same class; B = 0 otherwise.

**Why useful.** It expresses that concordance between independent models is itself evidence, beyond the magnitude of either score.

**Why capped at 1.0.** To preserve the interpretation of C as bounded and comparable across detections; the bonus would otherwise permit values above unity for already-confident agreeing detections.

**Visual representation.** Green at C ≥ 0.75, amber at C ≥ 0.50, orange below.

**Measured status.** §4.9 shows C is severely miscalibrated under domain shift (ECE 0.4669) and that the weighting was never empirically optimised. The calibration layer of §4.9 is the response.

## 3.12 Detector–Classifier Agreement

**How determined.** Equality of the predicted class labels.

**Both predict the same species.** The agreement bonus applies and no further computation is performed.

**Effect on combined confidence.** Multiplication by 1.10, capped at 1.0.

**Different species.** The three-opinion vote of §3.13 is invoked.

**Why disagreement should be analysed.** Because it localises uncertainty to a specific detection at negligible cost, and identifies exactly the cases where a single-model pipeline would have committed silently. Disagreement occurred on 42.1% of detections under domain shift. **[M-dev]**

## 3.13 Three-Opinion Majority Vote

**Why required, as designed.** To arbitrate between two conflicting opinions using a third that is neither.

**Activation.** Only on detector–classifier disagreement.

**The three opinions.** (i) the detector's class; (ii) the classifier's class on the 5% crop; (iii) the classifier's class on a 15% crop.

**Why 5% and 15%.** The two crops present different context-to-subject ratios, so the second classification is not a deterministic repetition of the first. This is a genuine limitation of the design: opinions (ii) and (iii) originate from the *same network* and are therefore correlated in a way that (i) is not.

**How the majority is determined.** The modal class of the three.

**When two agree.** That class is adopted, and the confidence of the agreeing source is used in the combination.

**When all three differ.** No majority exists; the detector class is adopted as fallback and no agreement bonus is applied. This occurred on 8.2% of firings. **[M-dev]**

**Why the detector is the fallback.** Because it alone incorporates localisation evidence, and because a deterministic fallback is required for reproducibility. §4.8 shows this fallback is the weakest of the three available policies, so the deadlock case actively steers the outcome toward the worst option.

**How it reduces classification errors.** As designed, by majority. As measured, it improves on the detector-only policy but is not distinguishable from simply adopting the classifier's class: see §4.8.

## 3.14 Decision Trace

**What it is.** An ordered, human-readable record of every control-flow decision taken during a single inference, together with the measured quantities on which each was conditioned.

**Why important.** Because in an adaptive pipeline the output alone is insufficient to reconstruct how it was produced. Two identical predictions may arise from entirely different execution paths, and without a trace the two are indistinguishable — which makes the adaptive layer unauditable, and made the counterfactual study of §4.6–§4.8 necessary in the first place.

**What is recorded.** Quality metrics and the enhancement decision; the review-gate outcome and its reason; each zoom-and-recheck invocation with its accept/reject result and justification; each detector–classifier disagreement with all three opinions; each majority-vote outcome including deadlocks; and the final label and confidence per object. Enhancement, review-gate, zoom, disagreement and voting decisions are all emitted.

**How it improves transparency.** It converts the controller from an opaque component into an inspectable one, and makes per-action cost attribution possible.

**How traces help analyse incorrect predictions.** They permit errors to be attributed to a stage. An error following a rejected zoom is a different failure from one following an accepted zoom that degraded the box, and only the trace distinguishes them. The forced-counterfactual protocol of §4.7 is a direct extension of this capability.

## 3.15 Final Detection and Annotation

**Final species label.** The classifier's class where the two agree; otherwise the majority-vote outcome of §3.13.

**Final confidence.** The equation of §3.11.

**Bounding box display.** Drawn in frame coordinates, colour-coded by confidence band.

**Information shown.** Species name and combined confidence per box, an object count, and per-stage timings.

**Colour semantics.** Green: C ≥ 0.75. Cyan/amber: C ≥ 0.50. Orange: C < 0.50.

**Species name and confidence displayed.** Yes, per object.

**Final output.** The annotated image (base64 PNG), the structured detection list, the summary block and the decision trace.

## 3.16 FastAPI Deployment

**Why FastAPI.** Native asynchronous support, automatic OpenAPI schema generation, and Pydantic request validation, permitting the pipeline to be exposed without a separate serving framework.

**Purpose of `/predict`.** Accepts a single image by multipart upload and returns the full structured result.

**Supported formats.** JPEG, PNG, WebP. HEIC is explicitly rejected with a conversion hint rather than failing opaquely.

**Maximum upload size.** 20 MB.

**Validation.** MIME type against the permitted set, then size, before any decoding is attempted.

**Execution.** Models are loaded once during application startup (measured 2,550 ms) and held in memory; requests reuse the loaded pipeline. **[M]**

**Returned information.** Annotated image, detections, summary with timing breakdown, and decision trace.

**Is the annotated image saved.** Yes, to an outputs directory, for audit.

**Are processing times recorded.** Yes, per stage, returned in the summary.

**Is the decision trace returned.** Yes.

**Browser interface.** A single-origin static front end presenting the architecture, the drop zone and one-click samples, the annotated result toggleable against the original, per-detection detector and classifier confidences with the agreement flag, the decision trace rendered as a timeline, and the evaluation figures.

---

# 4. RESULTS AND ANALYSIS

**A note on what is and is not reported.** Several subsections below report
*negative* results for components the architecture was designed around. These
are reported as measured. Where a question in the template presupposes a benefit
("does X improve Y?"), the answer given is the measured one, which in four cases
is no. This is the substantive contribution of the evaluation and is not
softened.

**Split discipline.** All development and analysis uses `J_cal` and `J_policy`
(Phase A) or `S_cal` and `S_policy` (Phase B). The held-out test partition
`J_test` has been read exactly once, to establish the frozen baseline in §4.3,
and not subsequently. `S_test` has not been read. Consequently no result marked
**[M-dev]** has been tuned against the data on which it will finally be judged.

## 4.1 Dataset

**Which dataset.** Two. **Aquarium Combined** (Roboflow
`brad-dwyer/aquarium-combined` v2, CC BY 4.0) for training and in-domain
validation. **Jedi Organism Detection Dataset** for out-of-domain and open-set
evaluation.

**Why Aquarium Combined.** It provides YOLO-format annotations across seven
aquatic taxa with sufficient instance counts for transfer learning at small
scale, in imagery that is optically benign — which is precisely what makes it
useful as the *in-domain* reference against which domain-shift failure is
measured.

**How many images.** Aquarium Combined: 638 (448 train, 190 validation), 4,180
annotated boxes. Jedi: 8,151 images across 3,152 dives, 15,621 annotations.
**[M]**

**How many classes.** Seven and twenty respectively.

**The seven species.** fish, jellyfish, penguin, puffin, shark, starfish,
stingray.

**Data division.** Aquarium Combined uses the supplied 448/190 train/validation
partition. For Jedi, the supplied partition was **discarded**: an audit
(`audit_group_leakage.py`) established that **57.3%** of its test images share a
dive with training images. Frames from a single dive share camera, illumination,
substrate and frequently the same individual organism seconds apart, so an
image-level partition leaks. All 8,151 images were pooled and re-partitioned by
**dive group**. Phase A uses three partitions (2,445 / 2,445 / 3,261 images);
Phase B uses five, proportioned over dives at 55 / 10 / 10 / 10 / 15 percent for
fitting, model selection, calibration, policy and test. **[M]**

**Are YOLO-format annotations available.** Yes for Aquarium Combined. Jedi is
supplied in COCO format and is converted (`coco_to_yolo.py`) with class indices
remapped to the Phase B label space.

**How classification crops are generated.** Each annotated box is expanded by 5%
padding, clipped to frame bounds, cropped and written to a class-directory tree
for `ImageFolder` loading. Boxes smaller than 8 px on either axis are discarded.

**Are all classes equally represented; is there imbalance.** No, and severely.
Crop counts: fish 2,668; jellyfish 694; penguin 516; shark 352; puffin 284;
stingray 184; starfish 116 — a ratio of approximately **23:1** between the most
and least represented class. **[M]** This imbalance is the most probable
explanation for the per-class variation in §4.4 and is a stated limitation.

## 4.2 Experimental Setup

**Language.** Python 3.12.10.

**Deep learning framework.** PyTorch 2.11.0 with CUDA 12.8.

**YOLOv8 implementation.** Ultralytics 8.4.138.

**ResNet18 framework.** torchvision 0.26.0.

**Hardware.** NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM, compute capability
12.0 (Blackwell, sm_120).

**GPU acceleration.** Yes; device selection is automatic and verified at
startup.

**YOLO image size.** 640 px.

**YOLO epochs.** 50 for the in-domain models, early-stopping patience 15. For
the deep-sea retraining of §4.3, 80 epochs with early stopping **disabled**
(patience set equal to epochs), because an earlier run showed early stopping
terminating one seed prematurely and collapsing its rare-class performance,
which inflated apparent seed variance.

**ResNet18 epochs.** 25.

**Optimizer.** Adam, initial learning rate 1×10³.

**Learning-rate schedule.** Step reduction to 1×10 at unfreezing.

**Initially frozen layers.** `conv1`, `bn1`, `layer1`, `layer2`.

**When unfrozen.** After 10 epochs, at which point all parameters are trained at
the reduced learning rate.

## 4.3 Detection Performance Evaluation

**Metrics.** Precision, recall, mAP@0.5, mAP@0.5:0.95, computed at IoU 0.50 as
the primary operating point.

**Precision.** TP / (TP + FP) — of the detections emitted, the fraction correct.

**Recall.** TP / (TP + FN) — of the objects present, the fraction found.

**mAP@0.5.** Mean over classes of average precision computed at an IoU matching
threshold of 0.50.

**mAP@0.5:0.95.** The mean of mAP computed at IoU thresholds from 0.50 to 0.95
in steps of 0.05, penalising imprecise localisation more heavily.

**Intersection over Union.** Area of intersection divided by area of union
between predicted and ground-truth boxes.

### In-domain (Aquarium Combined validation) **[M]**

| Metric | YOLOv8-Nano | YOLOv8-Small |
|---|---|---|
| mAP@0.5 | 0.750 | **0.780** |
| mAP@0.5:0.95 | 0.450 | **0.487** |
| Precision | 0.794 | **0.813** |
| Recall | 0.677 | **0.730** |
| Training time | 27 min | 48 min |

**Which detector performs best individually.** YOLOv8-Small on every metric,
as expected from capacity.

**Does WBF improve detection performance.** The isolated fused-versus-individual
comparison at equal evaluation protocol has **not** been executed **[P]**. It is
specified in §4.5 and is a material gap: the architecture's central ensembling
claim is currently supported by design argument rather than by measurement.

### Out-of-domain (Jedi `J_test`, single read) **[M]**

The frozen pipeline was run unmodified on 3,261 deep-sea frames producing 4,343
detections against 2,003 in-vocabulary ground-truth objects. 66.1% of annotated
organisms fall outside the seven-class label space.

| Outcome | Count | Share |
|---|---|---|
| TP — correct known class | 465 | 10.7% |
| CLS_ERR — matched, wrong known class | 263 | 6.1% |
| **OOD_ERR — matched an organism outside the label space** | **913** | **21.0%** |
| LOC_ERR — unmatched, IoU ∈ [0.10, 0.50) | 883 | 20.3% |
| BG_FP — unmatched, IoU < 0.10 | 1,819 | 41.9% |

At the deployed operating point (per-detection acceptance at C ≥ 0.40):

| Metric | Value | 95% CI (dive bootstrap, 1,000×) |
|---|---|---|
| Accepted detections | 3,946 | — |
| **Selective risk** | **0.888** | [0.876, 0.900] |
| Selective recall (GT-anchored) | 0.221 | [0.206, 0.264] |
| **OOD-error rate among accepted** | **0.213** | [0.197, 0.232] |
| AURC (risk vs recall) | 0.803 | [0.770, 0.836] |

Selective risk remains approximately 0.89 across the usable coverage range and
its floor at minimum coverage is approximately 0.65. The OOD-error panel
*increases* toward low coverage: the most confident deep-sea detections are
among the most likely to be confident misidentifications. **No threshold on this
confidence score yields a reliable system**, which is the central negative result
motivating §4.9.

### On-domain retraining (Phase B) **[M]**

Retrained on dive-disjoint deep-sea data over an 8-class known vocabulary with
11 taxa withheld as unseen:

| Seed | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| 0 | 0.6827 | 0.4741 |
| 1 | 0.6880 | 0.4783 |
| 2 | 0.6932 | 0.4772 |
| **Mean ± SD** | **0.6880 ± 0.0053** | **0.4765 ± 0.0022** |

Per-class AP@0.5 (seed 0): fish 0.883, crab 0.835, sea-cucumber 0.800, sponge
0.706, coral/sea-anemone 0.696, starfish 0.682, shrimp 0.637, **crinoid 0.223**.

The domain-shift failure is therefore specific to deployment outside the
training domain and is not a property of the architecture: trained on
representative data, the same detector attains 0.688 mAP@0.5 with a seed
standard deviation of 0.005.

## 4.4 Classification Performance

**Metrics.** Accuracy, macro-averaged precision, recall and F1, and the
confusion matrix.

**Definitions.** Accuracy is the fraction of crops assigned the correct label.
Precision, recall and F1 are computed per class and macro-averaged so that rare
classes are not dominated by frequent ones. F1 is the harmonic mean of precision
and recall.

**Overall accuracy.** For the Phase B (deep-sea, 8-class) classifier over three
seeds: validation accuracy 0.8285, 0.8418, 0.8418; **macro-recall 0.7623,
0.7521, 0.7671**. The gap between accuracy and macro-recall quantifies the
effect of class imbalance. **[M]**

**Per-species performance, most and least accurate species, confusion
structure.** Per-class precision/recall/F1 and the confusion matrix for the
*in-domain seven-class* classifier have not been tabulated in the form the
template requires **[P]**. The confusion matrix has been generated as a figure
but the numeric table is outstanding. This is a genuine reporting gap and should
be closed before submission.

**Why some species are confused.** Two mechanisms are expected and one is
observed. Expected: taxa sharing gross morphology under degraded optics; and
minority classes (starfish at 116 crops) receiving insufficient gradient signal.
Observed in the detection results: crinoid, a feathery low-contrast sessile
organism, attains AP@0.5 of only 0.223 against 0.800 for sea-cucumber on *fewer*
training annotations (226 versus 367), indicating that intrinsic visual
difficulty, not sample count alone, governs the failure.

## 4.5 Comparison of Individual and Combined Models

**Status: not executed [P].** The full ladder — YOLOv8-Nano alone; YOLOv8-Small
alone; both without fusion; dual + WBF; dual + WBF + ResNet18; the complete
agentic system — evaluated under one protocol on one split, reporting precision,
recall, mAP, classification accuracy, F1 and inference time per configuration,
has not been run.

This is the most significant outstanding item in the evaluation and is stated as
such rather than approximated. Individual detector numbers exist (§4.3) but were
produced under the detectors' own validation runs, not under the unified
per-detection protocol used elsewhere in this work, and are therefore **not**
directly comparable to the fused system.

**What can be answered now.** The component-level counterfactual measurements of
§4.6–§4.8 address the same underlying question — *does each component earn its
place?* — by a stronger method than an ablation ladder, since each action is
compared against the counterfactual of not performing it on the *same*
detections rather than across separately-trained configurations.

**Does model fusion improve reliability.** Unmeasured **[P]**.

**What additional computational cost is introduced.** Measured per stage in
§4.12.

## 4.6 Image Enhancement Analysis

**Method.** Rather than testing on curated degraded images, a controlled
protocol was used: 190 in-domain validation images were degraded synthetically
along seven axes at three severities each, with ground truth unchanged, and the
frozen pipeline run on every variant. Separately, ENHANCE was **forced** on every
image regardless of its trigger condition, and the whole-pipeline outcome
compared against not enhancing, paired by image and bootstrapped over dives.

**Performance before and after enhancement.** Paired dive bootstrap on correct
detections per image, enhancement minus no enhancement: **[M-dev]**

| | Frozen pipeline | Retrained pipeline |
|---|---|---|
| All images | −0.0052 [−0.0117, +0.0017] | **−0.0572 [−0.0909, −0.0291]** |
| Images the rule fires on | −0.0107 [−0.0241, +0.0025] | **−0.1069 [−0.1697, −0.0572]** |

**Does CLAHE improve detection confidence; does sharpening improve blurred-object
detection; does enhancement improve classification accuracy.** On aggregate, no.
On the retrained pipeline the effect is significantly negative, and the
degradation is approximately twice as large on the subset the rule itself
selects (−0.107 against −0.057). The quality heuristic is therefore not merely
failing to identify images that benefit — it is preferentially selecting images
that are harmed.

**Are there cases where enhancement reduces performance.** Yes; this is the
dominant case, which inverts the premise of the question.

**Which image-quality problem has the greatest effect.** From the controlled
degradation sweep, measured as selective risk at severity 3 against a clean
baseline of 0.364, with the abstention rate in parentheses: **[M]**

| Degradation | Selective risk | Abstention rate |
|---|---|---|
| Gaussian blur | 0.393 | 1.6% → **83.2%** |
| Motion blur | 0.319 | 1.6% → 26.8% |
| Noise | **0.681** | 1.6% → 30.5% |
| Compression | **0.610** | 1.6% → 3.7% |
| **Colour shift** | **0.537** | 1.6% → **2.1%** |
| Low contrast | 0.533 | 1.6% → 11.1% |
| Low illumination | 0.489 | 1.6% → 10.0% |

Noise produces the largest rise in error rate. **Colour shift and compression
are the most dangerous**, because error rises substantially while the abstention
rate remains at its clean-image level — the system does not detect that its
evidence has degraded. Gaussian blur is the sole condition handled correctly, and
is also the sole condition the controller explicitly tests for. The general
finding is that hand-specified quality rules detect exactly the degradations
their author anticipated and no others. Colour shift is of particular
significance because red attenuation with depth makes it the dominant distortion
in real underwater imagery, supplying a mechanistic partial explanation for the
§4.3 domain-shift failure.

*Note: motion blur's apparent risk decrease is a selection effect — abstention
removed the harder images, leaving an easier surviving population. Selective risk
must be read alongside its abstention rate.*

## 4.7 Zoom-and-Recheck Analysis

**Method.** The action was **forced** on every detection, not only those within
the rule's [0.40, 0.75) band, re-detected, re-classified and scored against
ground truth, producing the counterfactual the decision trace cannot supply.
Forcing exhaustively rather than sampling ε-greedily removes sampling noise from
the estimate, which matters because the effect is small. **[M-dev]**

| | Frozen pipeline | Retrained pipeline |
|---|---|---|
| Detections forced | 3,253 | 1,343 |
| A candidate box was found | 32.2% | 72.7% |
| …and confidence improved | 9.4% | 23.5% |
| **The deployed rule would fire on** | 19.7% | 24.9% |
| **The rule actually changed a box** | **0.3%** | **4.5%** |
| Correct without any zoom | 0.1085 | 0.5138 |
| Correct under the deployed rule | 0.1091 | 0.5101 |
| **Net effect** | **+0.00064** | **−0.00376** |
| 95% CI | [−0.00124, +0.00251] | [−0.01007, +0.00221] |
| Cost on the split | 18.9 s / 641 firings | 9.6 s / 335 firings |

**How many detections activate it.** 641 and 335 respectively.

**Average confidence before and after; how many boxes are replaced.** On the
frozen pipeline the rule replaced **nine** boxes across 641 firings.

**How many uncertain detections become correct.** Across all 3,253 forced
actions on the frozen pipeline, 37 detections changed outcome: 6 improved, 4
worsened, and 17 moved a box from one out-of-vocabulary organism to a
mislocalisation, exchanging one error for another.

**Does it improve localization; does it improve confidence.** Both confidence
intervals include zero. On the retrained pipeline the point estimate is
**negative**, and of 29 outcome changes, **11 were worse and 6 better**
(TP → CLS_ERR six times, TP → LOC_ERR five times). A more capable detector
produces better-localised boxes, which the re-crop is correspondingly more
likely to damage than improve.

**Additional inference time.** 29.5 ms per invocation (median 28.1 ms).

**In which situations is it most useful.** No such situation was identified. A
value model Q(s, ZOOM) was attempted and could not be fitted: 15 outcome changes
on the fit split is insufficient signal, and fitting would have produced
coefficients indistinguishable from noise. The action being near-inert is itself
the finding.

## 4.8 Majority Voting Analysis

**Method.** Two structural facts narrow the question. On agreement the vote
already holds a majority of two, so the third opinion provably cannot alter the
outcome. And the action never moves the box, so it can only convert TP ↔
CLS_ERR. The measurable question is therefore whether, on disagreements, a third
opinion outperforms simply trusting one of the two already held. Three policies
were scored against identical ground truth; only the first incurs additional
computation. **[M-dev]**

**How frequently do the models disagree.** 42.1% of detections on the frozen
pipeline (1,370 of 3,253); 31.0% on the retrained pipeline.

**Correctness on disagreements:**

| Policy | Frozen | Retrained | Extra compute? |
|---|---|---|---|
| **Rule (majority vote)** | 0.1066 | 0.2404 | yes |
| Classifier only | 0.1000 | 0.2236 | no |
| Detector only | 0.0555 | 0.1755 | no |

**Paired dive bootstrap, rule minus alternative, over all detections:**

| | Frozen | Retrained |
|---|---|---|
| vs detector only | **+0.02172 [+0.0137, +0.0311]** | **+0.02033 [+0.0024, +0.0384]** |
| vs classifier only | +0.00271 [−0.0006, +0.0065] | +0.00517 [−0.0062, +0.0161] |

**How many disagreements are correctly resolved.** The rule clearly outperforms
the detector-only policy on both pipelines — establishing that the independent
classification stage earns its place. It does **not** outperform simply adopting
the classifier's class: both intervals include zero. Over all detections the
vote purchases **+9 correct detections for 7.1 s** (frozen) and **+7 for 2.4 s**
(retrained), neither distinguishable from zero.

**How often do all three opinions disagree.** 8.2% of firings on the frozen
pipeline. In these the vote deadlocks and defaults to the detector — the weakest
of the three policies at 0.0555 — so the additional computation is not merely
wasted but actively steers the outcome toward the worst available option.

**Does majority voting improve classification accuracy / reduce false
classifications.** Relative to the detector-only policy, yes. Relative to the
free alternative of trusting the classifier, not measurably.

**Which species benefit most.** Per-species decomposition of the vote outcome
has not been computed **[P]**.

## 4.9 Combined Confidence Analysis

**Does combined confidence better represent prediction reliability.** Not as
originally specified. Fitted on `J_cal` and evaluated on `J_policy`, the raw
combined score has mean 0.5582 against an empirical accuracy of 0.0912 — an
overconfidence of **+0.467** — with expected calibration error 0.4669. **[M-dev]**

| Method | ECE | MCE | Brier | NLL | AUROC |
|---|---|---|---|---|---|
| Raw (uncalibrated) | 0.4669 | 0.7716 | 0.3076 | 0.8570 | 0.6414 |
| Temperature scaling | 0.4088 | 0.4088 | 0.2500 | 0.6931 | 0.6414 |
| Platt scaling | 0.0190 | 0.4106 | 0.0817 | 0.2991 | 0.6414 |
| **Isotonic regression** | **0.0160** | 0.1536 | **0.0810** | **0.2936** | 0.6469 |
| *Constant = base rate (trivial)* | *0.0140* | — | *0.0831* | *0.3065* | *n/a* |

Three findings. First, **temperature scaling degenerates**: it rescales logits
but has no bias term, so it cannot correct a level error of +0.467; the NLL
optimum drives T → ∞ and every prediction collapses to 0.5. Second, **the
headline ECE reduction is largely regression to the base rate** — a constant
predictor ignoring the input achieves ECE 0.0140, better than isotonic's 0.0160,
and isotonic's Brier skill score against that constant is only **+0.0254**. The
defensible claim is that calibration restores *honesty* while preserving
*ranking*, which the constant cannot do. Third, **AUROC is essentially
unchanged** (0.6414 → 0.6469): calibration repairs what a score means, never how
well it ranks.

**How detector confidence compares with classifier confidence.** As
out-of-distribution indicators, both are useless: AUROC 0.4356 (detector),
0.4693 (classifier maximum softmax), 0.4383 (fused) — all at or below chance,
i.e. unseen taxa receive marginally *higher* confidence than correct detections.
**[M-dev]**

**When both models strongly agree.** The agreement bonus applies. Since
agreement occurs on 57.9% of detections and the score remains miscalibrated, the
bonus amplifies an already-unreliable quantity.

**High detector, low classifier confidence, and the converse.** Both produce a
mid-range combined score and, if in [0.40, 0.75), trigger zoom-and-recheck —
whose measured value is null (§4.7).

**Does the 0.6/0.4 weighting improve performance; is the 10% bonus effective;
would different weights change results.** The weighting and bonus were never
empirically optimised; no sweep has been performed **[P]**. Indirect evidence
that the specification is suboptimal comes from the learned gate (§4.11), which
when free to choose its own weighting assigned **novelty distance a
standardised weight of −3.027 against +0.619 for calibrated confidence and
−0.027 for classifier confidence** — that is, it weighted novelty roughly five
times more heavily than confidence and effectively discarded classifier
confidence, a weighting the hand-specified rule cannot express.

## 4.10 Robustness Analysis

Seven of the ten listed degradations were evaluated under the controlled
protocol of §4.6; occlusion, scaling and rotation were not **[P]**. Results are
tabulated in §4.6.

**How detection and classification accuracy change.** Selective risk rises from
0.364 to between 0.393 and 0.681 depending on degradation and severity.

**How confidence changes.** Mean confidence falls under noise (0.711 → 0.522)
and compression (0.736 → 0.620) but only marginally under colour shift (0.725 →
0.686) — that is, confidence tracks *some* degradations and not others.

**Does the agentic controller activate corrective processing.** Only for blur.
Abstention rises to 83.2% under Gaussian blur and to 26.8% under motion blur,
but remains at 2.1% under colour shift and 3.7% under compression.

**Which degradation is most difficult.** Noise produces the highest error rate;
colour shift and compression are the most dangerous, because error rises without
any corresponding rise in abstention.

**Which component helps recover performance.** Only the blur gate, and only for
blur. No component was found to recover performance under any other degradation.

## 4.11 Ablation Study

**Component-removal ablations [P].** The systematic single-component removals
listed in the template — without WBF, without ResNet18, without the review gate,
without the agreement bonus, without the controller — have not been executed.

**What has been measured instead.** A forced-counterfactual study of every
discretionary action, on two pipelines, which addresses the same question by a
stronger method: each action is compared against not performing it on the *same*
detections, rather than across separately-trained configurations that differ in
more than one respect.

| Action | Effect on correctness | Verdict |
|---|---|---|
| ENHANCE | −0.0572 TP/image (retrained), CI excludes zero | **actively harmful** |
| ZOOM | +0.0006 / −0.0038, CIs include zero | inert; negative point estimate when retrained |
| RECLASSIFY (vote) | +0.0027 / +0.0052 vs classifier-only, CIs include zero | no better than a free alternative |
| FULL_IMAGE_REDETECT | −0.0453 / −0.0227, CIs exclude zero | harmful (see caveat) |

**Which component produces the largest performance improvement.** None of the
four discretionary actions produces any measurable improvement. **Removing the
discretionary layer entirely yields a system that is both more accurate and
cheaper.**

**Does combining all components give the best result.** On this evidence, no.
This inverts the template's presupposition and is the paper's principal
empirical claim about hand-specified adaptive control.

**Caveat on FULL_IMAGE_REDETECT.** This action is listed in the design
specification but **is not implemented in the pipeline** — the deployed system
has three discretionary actions, not four. Re-running the detector on identical
pixels is deterministic and would measure nothing, so the action was
operationalised as full-frame re-detection at 1280 px against the trained 640 px.
The detectors were trained at 640 px, so a train/test resolution mismatch is a
sufficient explanation for the negative result on its own. The finding is that
*re-detection at double resolution with a 640-trained model degrades accuracy*,
not that full-frame re-detection is unhelpful in principle; a variant using
multi-scale training or augmentation is untested.

**A positive result from the same framework.** A **learned state-conditional
accept/abstain policy**, fitted on nine features and requiring no exploration
data (the outcome of accepting is already recorded for every cached detection),
reduces selective risk from **0.8462 to 0.7756 at matched 30% coverage** (paired
dive bootstrap, Δ = +0.0708, 95% CI [+0.0493, +0.0932]) and from 0.8801 to
0.8384 at 50% coverage (Δ = +0.0419, CI [+0.0288, +0.0555]). It also operates
across the full coverage range, whereas the fixed two-dimensional gate is capped
at 14.7% by its hard mask. **[M-dev]** Note that the coverage advantage follows
structurally from using a score rather than a mask, independently of learning.

**Out-of-distribution detection.** Scoring computed from the classifier's
penultimate representation attains, out of sample: **[M-dev]**

| Score | Aquarium features | Retrained features |
|---|---|---|
| Energy, near-OOD (unseen taxa) | 0.5533 | **0.8059** |
| Energy, far-OOD (hardware) | 0.6414 | **0.8161** |
| Mahalanobis, near-OOD | 0.6504 | 0.7127 |
| Mahalanobis, far-OOD | **0.8163** | 0.7178 |

On-domain training relocates the novelty signal into the *logits*: the aquarium
classifier's logits span seven aquarium species and are meaningless on deep-sea
taxa, so only the feature geometry carried signal; the retrained classifier's
logits span the eight relevant taxa, so "activates none of these strongly"
becomes genuine evidence of novelty. Since every confidence-derived score is at
chance (§4.9), **the information required for novelty rejection is present in the
network and is destroyed by the classification head** — a localisable deficiency
rather than a general shortfall.

*Methodological note.* Mahalanobis scoring initially returned below-chance AUROC
(0.42) on retrained features while scoring 0.87 in-sample. Diagnosis established
this was not covariance conditioning — the covariance is full rank and added
shrinkage monotonically worsened the score — but feature magnitude: the retrained
network assigns smaller-magnitude features to novel objects, so any
magnitude-dependent distance inverts. L2 normalisation recovers the signal
(0.44 → 0.71). This is recorded because the uncorrected figure would have been
reported as a property of the features rather than of the estimator.

## 4.12 Runtime and Computational Analysis

Measured over 30 images at 8.13 objects per image (maximum 27), CPU: **[M]**

| Stage | Mean | Median | p95 | Max |
|---|---|---|---|---|
| Image-quality assessment | 7.10 ms | 3.05 | 10.26 | 91.00 |
| Dual detection (both models) | 88.97 ms | 84.72 | 103.53 | 153.47 |
| Weighted Boxes Fusion | 0.42 ms | 0.32 | — | — |
| Zoom-and-recheck (per invocation) | 29.5 ms | 28.1 | — | — |
| Majority vote (per invocation) | 5.2 ms | — | — | — |

Accounting residual 0.03 ms, confirming the stage decomposition is complete.
Full-pipeline GPU inference: **217.4 ms** per image. Model loading: 2,550 ms at
startup, amortised across requests.

**Which stage is most expensive.** Dual detection, at approximately 89 ms of a
~97 ms mandatory CPU path — roughly 92%.

**Is the system suitable for real-time applications.** At 217 ms per image on
the stated GPU it supports approximately 4.6 fps, adequate for frame-sampled
survey processing and for interactive use, but not for full-rate video without
the adaptive frame-sampling strategy reserved for the video paper.

**Discretionary cost.** The zoom action consumed 18.9 s across 641 firings and
the vote 7.1 s across 1,370 firings on a single evaluation split. Since §4.7 and
§4.8 establish neither improves accuracy, this is 26 s of recoverable compute
per ~1,500 images at no accuracy cost — satisfying the efficiency objective by
measurement rather than by policy optimisation.

## 4.13 Comparison with Existing Methods

**Status: not performed, deliberately [P].** No quantitative comparison against
published underwater detection methods is reported, for two reasons that are
stated rather than concealed.

First, **the supplied partition of the deep-sea corpus leaks**: 57.3% of its test
images share a dive with training images. Any figure computed on that partition
is inflated by an unquantified amount, so a comparison against it would be
misleading in the favourable direction.

Second, **the label spaces differ**. This work evaluates eight known taxa with
eleven deliberately withheld as unseen; a method reporting all twenty classes is
solving a closed-set problem and its numbers are not commensurable.

**What is compared instead.** The system is compared against its own frozen
baseline under an identical protocol, and each component against the
counterfactual of its own removal. This is a weaker claim about the literature
and a stronger claim about the system.

**Does the proposed system improve precision / recall / mAP / reduce false
positives.** Against its own frozen baseline under domain shift, the uncertainty
layer reduces selective risk at matched coverage (0.846 → 0.776) and more than
halves the rate of confident errors on unknown organisms (0.373 → 0.182).
**[M-dev]** Detection metrics against external methods: not measured **[P]**.

**Does independent ResNet18 classification improve species recognition.** Yes —
the majority-vote policy outperforms the detector-only policy by +0.0217 and
+0.0203 on the two pipelines, both intervals excluding zero (§4.8). This is the
clearest positive evidence for a component in the architecture.

**Does WBF reduce localization errors.** Unmeasured in isolation **[P]**.

**Which method achieves the best overall performance.** Within this work, the
retrained detector combined with the calibrated, novelty-aware learned gate.

---

# 5. CONCLUSION

## Draft conclusion (prose)

This work addressed the reliability of automated marine organism recognition
when such a system is deployed outside the conditions it was trained for. An
agentic multi-model pipeline was constructed — two YOLOv8 detectors trained
under differing seeds and augmentation regimes, combined by Weighted Boxes
Fusion, with every fused region independently re-examined by a ResNet18
classifier, under a rule-based controller governing image enhancement, an
abstention gate, ambiguous-detection re-inspection and detector–classifier
disagreement resolution, with every decision emitted as a machine-readable
trace. In its training domain the pipeline attains mAP@0.5 of 0.780. Evaluated
unmodified on 8,151 frames of deep-sea ROV imagery, in which 66.1% of annotated
organisms lie outside its label space, its selective risk is 0.888 and 21.3% of
accepted detections are confident species labels assigned to organisms it was
never trained on — and, critically, the error rate does not fall as the
confidence threshold is raised, establishing that the score carries almost no
information about correctness in this regime. Because every discretionary action
was instrumented, each could be measured against the counterfactual of not
performing it: none of the four improves accuracy, conditional enhancement
measurably degrades it, and removing the discretionary layer yields a system
both more accurate and cheaper. Isotonic calibration reduces expected
calibration error from 0.467 to 0.016 while preserving ranking, and
out-of-distribution scoring computed from the classifier's penultimate
representation attains AUROC 0.806 for unseen taxa where every confidence-derived
score is at chance — demonstrating that the information required for novelty
rejection is present in the network and destroyed by the classification head. A
learned state-conditional accept/abstain policy reduces selective risk from
0.846 to 0.776 at matched coverage and, when free to weight its own inputs,
assigns novelty distance approximately five times the weight of calibrated
confidence, independently recovering the same conclusion. The contribution is
therefore not a higher benchmark score but a demonstration that adaptive control
in such pipelines must be measured rather than assumed, together with the
instrumentation that makes such measurement possible.

## Question-by-question

**What problem was addressed.** Whether an automated underwater recognition
system can be trusted outside its training domain, and whether the adaptive
mechanisms commonly added to such systems in fact improve them.

**Why deep-sea creature detection and classification is important.** Because
survey imagery accumulates faster than expert annotation capacity, and because
an unflagged false occurrence record contaminates downstream ecological
analysis in a manner indistinguishable from a verified observation.

**Major challenges of underwater image recognition.** Wavelength-dependent
attenuation and colour distortion, turbidity and backscatter, non-uniform
artificial illumination, motion blur, occlusion, camouflage, uncontrolled
object scale, and — most consequentially for this work — the open-set condition
in which most encountered organisms lie outside any fixed label space.

**What system was proposed, and how it works.** See §3.1. Quality assessment and
conditional enhancement; dual YOLOv8 detection; Weighted Boxes Fusion; a review
gate that may terminate the run; zoom-and-recheck on ambiguous detections;
independent ResNet18 classification; three-opinion majority voting on
disagreement; combined confidence scoring; annotation and decision trace.

**Role of YOLOv8-Nano.** Low-cost detector contributing one of two decorrelated
opinions; trained at seed 42 with mosaic augmentation only.

**Role of YOLOv8-Small.** Higher-capacity detector contributing the second
opinion; trained at seed 123 with mosaic, mixup and copy-paste, so that its
errors differ from Nano's in kind and not merely in frequency.

**Why two detection models.** So that agreement carries information. A single
model, however confident, offers no means of distinguishing well-supported
detections from confident errors.

**Role of Weighted Boxes Fusion.** To combine proposals in a manner that
preserves agreement as evidence, rather than discarding it as
Non-Maximum Suppression does.

**Role of ResNet18.** To supply an architecturally independent second opinion on
taxonomic identity, and thereby to render detector–classifier disagreement
available as a signal. §4.8 provides the clearest positive evidence for any
component: the classification stage outperforms detector-only labelling by
+0.0217 and +0.0203 on the two pipelines, both intervals excluding zero.

**Role of image-quality assessment.** To supply, at 7.10 ms mean cost, the state
on which enhancement and admissibility are conditioned.

**Role of conditional enhancement.** As designed, to recover contrast and
compensate illumination. As measured, it degrades accuracy (§4.6), and does so
most on the images its own trigger selects.

**Role of the review gate.** To terminate processing when the image is
unreadable or no detection reaches the acceptance threshold. It is the only
terminal decision and the system's sole mechanism for declining to answer.

**Role of zoom-and-recheck.** As designed, to re-inspect ambiguous detections at
higher effective resolution. As measured, its effect is not distinguishable from
zero and is negative in point estimate on the retrained pipeline (§4.7).

**Role of three-opinion majority voting.** As designed, to arbitrate
disagreement. As measured, it improves upon the detector-only policy but not
upon simply adopting the classifier's label (§4.8).

**Purpose of combined confidence scoring.** To express that concordance between
independent models is evidence beyond either score alone. §4.9 shows the
resulting score to be severely miscalibrated and its weighting never empirically
optimised.

**Purpose of the decision trace.** To make an adaptive pipeline auditable. It is
also what made the counterfactual measurements of §4.6–§4.8 possible, and is
therefore the enabling condition of the paper's principal findings rather than
an interface convenience.

**Major experimental findings.**

1. In-domain mAP@0.5 0.780; out-of-domain selective risk 0.888 [0.876, 0.900],
   with 21.3% of accepted detections confidently mislabelling organisms outside
   the label space.
2. Selective risk does not decrease as the confidence threshold rises; no
   threshold yields a reliable system.
3. Every confidence-derived out-of-distribution score is at or below chance
   (AUROC 0.436–0.533).
4. Representation-space scoring attains AUROC 0.806 for unseen taxa, localising
   the deficiency to the classification head.
5. Isotonic calibration reduces ECE 0.467 → 0.016, though largely by regression
   to the base rate; its value is preserving ranking while restoring honesty.
6. A learned state-conditional gate reduces selective risk 0.846 → 0.776 at
   matched coverage, and independently weights novelty ≈5× confidence.
7. None of the four discretionary actions improves accuracy; enhancement is
   actively harmful.
8. On-domain retraining attains mAP@0.5 0.688 ± 0.005 over three seeds, showing
   the failure is a deployment condition and not an architectural defect.

**Does WBF improve detection performance.** Not measured in isolation **[P]**.
This is an acknowledged gap.

**Does ResNet18 improve species classification.** Yes (§4.8), with intervals
excluding zero on both pipelines.

**Does the agentic controller improve overall reliability.** As specified with
hand-written rules, no — §4.11. As replaced by a learned state-conditional
policy, yes — §4.11. This distinction is the paper's central claim.

**Is the proposed method better than individual YOLO models.** Not established
**[P]**; the controlled ladder of §4.5 has not been executed.

**Main contributions.**

1. An agentic underwater recognition architecture in which every discretionary
   action is instrumented, costed and independently falsifiable.
2. A leakage-free evaluation protocol for the deep-sea corpus, following an audit
   establishing 57.3% dive contamination in the supplied partition.
3. Quantification of open-set failure under domain shift, including the finding
   that out-of-distribution error rises with confidence.
4. A forced-counterfactual measurement protocol for discretionary actions, and
   the finding that none of four such actions improves accuracy.
5. Demonstration that novelty information survives in the penultimate
   representation while being absent from every confidence-derived score.
6. A learned accept/abstain policy that reduces selective risk at matched
   coverage and independently confirms the novelty finding.

**Limitations.**

1. The full ablation ladder comparing individual and fused configurations under
   one protocol has not been executed; the ensemble's contribution is therefore
   argued rather than measured.
2. All uncertainty, out-of-distribution and policy results are development-split
   results; none has been confirmed on held-out test data.
3. Residual selective risk remains high — the learned gate is wrong 77.6% of the
   time at 30% coverage. The system is improved, not reliable.
4. Near-OOD detection is not deployable at high recall: FPR@95TPR is 0.627, so
   catching 95% of novel taxa discards 63% of correct detections.
5. Calibration's headline ECE improvement is largely regression to the base rate;
   its Brier skill against a constant predictor is only +0.025.
6. Out-of-distribution sample sizes are small (44–212 detections per comparison
   after fit-eligibility filtering), so intervals are correspondingly wide.
7. `FULL_IMAGE_REDETECT` is specified but not implemented; the measured variant
   is an operationalisation confounded by training resolution.
8. Only the accept/abstain decision is learned; the remaining action space is
   still governed by hand-written thresholds, and the compute-budget allocator
   is unbuilt.
9. Class imbalance in the classification training data is severe (≈23:1), and
   crinoid remains weak at AP@0.5 0.223.
10. No quantitative comparison against published methods, for the reasons in
    §4.13.

**Does using multiple models increase inference time.** Yes. Dual detection is
approximately 89 ms of a ~97 ms mandatory CPU path. However, §4.7 and §4.8
establish that the *discretionary* actions consume 26 s per ~1,500 images
without measurable benefit, so the recoverable saving lies in the controller
rather than in the ensemble.

**Is the current dataset sufficiently large and diverse.** No. The training
corpus is 638 images across seven taxa with ≈23:1 class imbalance, and the
evaluation corpus, while substantially larger, is dominated by a small number of
frequent classes. The 66.1% out-of-vocabulary rate at evaluation is itself
evidence that the training taxonomy is inadequate for the deployment domain.

**How can the system be improved in the future.**

- **Larger YOLO models.** Yes. Only YOLOv8-Nano was retrained on-domain;
  YOLOv8-Small and a transformer-family detector are specified but untrained,
  the latter being impractical on the available 8 GB accelerator.
- **Vision Transformers.** A natural extension, particularly given the finding
  that novelty information resides in the representation: an architecture with
  different representational geometry may localise that signal differently.
- **Underwater image-enhancement deep networks.** Motivated directly by §4.6:
  since the hand-specified CLAHE-and-sharpen transform degrades accuracy, a
  learned enhancement optimised jointly with the detection objective is the
  principled replacement.
- **Extension from still images to video.** Reserved for a separate paper. The
  same decision framework applies with temporal evidence added, and the
  observation state is designed to accommodate track-level features without
  architectural change.
- **Temporal tracking.** As above; deferred.
- **Deployment on an AUV.** Feasible in principle at 217 ms per frame, and the
  measured per-action cost model is the prerequisite for a compute budget under
  the power constraints such a platform imposes.
- **Additional marine species.** Necessary. The measured 66.1%
  out-of-vocabulary rate is the strongest argument for taxonomic expansion, in
  combination with open-set methods, since no fixed taxonomy will be complete.
- **Replacing the rule-based controller with a learned agent.** This is the
  principal direction, and it is partially realised: the accept/abstain decision
  is already learned and outperforms the threshold it replaces. Extending the
  policy to select *which* evidence to acquire under an explicit per-image
  compute budget requires action-outcome data of the kind §4.7 and §4.8 now
  provide, and constitutes the immediate continuation of this work.

