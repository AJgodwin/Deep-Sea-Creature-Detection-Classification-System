# Panel Q&A — expected questions and defensible answers

Every number below appears in `experiments/RESULTS.md` with the split it came
from. Where an answer concedes something, it concedes it early and then says why
that is acceptable — conceding under pressure later reads far worse than
conceding first.

**One rule for all of these: lead with the number, then the reasoning.** Never
open with a justification; open with the measurement.

---

## A. The questions about the low numbers

### A1. "Your system is wrong 88.8% of the time. Is it broken?"

**Most likely question in the room. Do not get defensive.**

> No — that figure is the experiment, not a defect. We took a system that scores
> 0.780 mAP in the domain it was trained for and deployed it unchanged on
> deep-sea footage where 66.1% of the organisms are outside its seven-class
> vocabulary. For two-thirds of what it sees, no correct answer exists in its
> label space. The 88.8% is the measured cost of deploying an in-domain-validated
> model out of domain, and quantifying that was the point.
>
> When we retrain on representative deep-sea data, the same architecture reaches
> mAP 0.688 with a seed standard deviation of 0.005. So the failure is a
> deployment condition, not an architectural one.

**Expected follow-up: "Then why not just always retrain?"** → See A3.

### A2. "Why is your mAP lower than published underwater detection papers?"

> Three reasons, and they all favour our numbers being lower.
>
> First, our task is open-set: two-thirds of test organisms have no valid label.
> Benchmark papers report closed-set mAP where every object has a correct answer.
>
> Second, we audited the supplied data partition and found **57.3% of its test
> images share a dive with training images**. Frames from one dive share camera,
> lighting, substrate, often the same individual seconds apart. We discarded that
> partition and re-split by dive. Any number computed on the original split is
> inflated by an unknown amount.
>
> Third, our label space differs. We hold out 11 taxa as genuinely unseen; a
> paper reporting all 20 classes is solving an easier problem.
>
> Our numbers are lower partly *because* we removed the advantages.

### A3. "If retraining fixes it, why is any of this necessary?"

> Retraining raises accuracy on the classes you trained for. It does not address
> the open-set problem, and the ocean will always contain organisms outside any
> fixed taxonomy. Even after retraining, our held-out taxa are still
> unrecognisable to the model — knowing *when you are looking at one* is a
> separate capability from being accurate on the ones you know. That capability
> is what we built and measured.

### A4. "Coverage of 14.7% means you reject 85% of detections. That's unusable."

**Concede this immediately — it is true of the fixed gate.**

> You are right about the fixed gate, and that is why we replaced it. The hard
> mask caps coverage structurally. The learned policy outputs a score instead, so
> it operates across the full coverage range — at 30% coverage it reduces
> selective risk from 0.846 to 0.776, with a bootstrap confidence interval of
> [+0.049, +0.093] that excludes zero.
>
> I would still not call the system reliable. At 30% coverage it is wrong 77.6%
> of the time. It is measurably improved, not solved, and I would not claim
> otherwise.

---

## B. The questions about the negative findings

### B1. "You built four adaptive components and none of them works. Isn't that a failed design?"

**This is the question to be ready for. The answer is the paper's thesis.**

> It would be, if we had assumed they worked and shipped them. What we did was
> instrument every one so its effect could be measured against the counterfactual
> of not performing it, on the same detections.
>
> The result is that enhancement is actively harmful — −0.057 correct detections
> per image on the retrained model, confidence interval excluding zero — zoom is
> inert, and the majority vote is no better than simply trusting the classifier.
>
> Nobody had checked. These rules had been firing on intuition since the system
> was built. Removing them makes the system both more accurate and cheaper, which
> is an immediate practical result, and the durable one is the method: an action
> that cannot beat not doing it has no place in the pipeline.

### B2. "So why is the enhancement still in your architecture diagram?"

> Because the paper reports what the system does, and removing it retrospectively
> would hide the finding. It is drawn as a discretionary branch, which is exactly
> what it is. The measurement in §4.6 says whether that branch should fire, and
> the answer is currently no.

### B3. "Your enhancement makes things worse. That contradicts the underwater imaging literature."

**Strong question. The answer strengthens your position.**

> It is consistent with it, actually. The reason the learned-enhancement
> literature exists — Sea-thru and the networks that followed — is that naive
> histogram methods do not model underwater light transport. Our result is
> empirical confirmation on our data: a hand-specified CLAHE-and-sharpen
> transform degrades accuracy, and does most damage on precisely the images its
> own trigger selects, −0.107 against −0.057 overall.
>
> That is an argument for a learned enhancement optimised jointly with the
> detection objective, which is our stated next step.

### B4. "Isn't your calibration result just predicting the average?"

**They will ask this if they know calibration. Answer before they push.**

> Partly, and we measured exactly that. A constant predictor that ignores the
> input achieves ECE 0.0140, which is *better* than our isotonic fit at 0.0160.
> The Brier skill score of isotonic against that constant is only +0.025.
>
> The value is not the ECE headline. It is that calibration restores honesty
> while preserving ranking — AUROC 0.641 to 0.647 — which a constant cannot do at
> all. A constant is perfectly calibrated and completely useless for deciding
> which detection to trust.

### B5. "Why did the zoom action produce a negative result on the better model?"

> Because a better detector produces better-localised boxes, and a re-crop is
> then more likely to damage one than improve it. Of 29 outcome changes on the
> retrained model, 11 were worse and 6 better — six true positives became
> classification errors and five became localisation errors. The guard condition
> requires the new box to overlap at IoU > 0.4 and score higher, but that guard
> does not measure the criteria the box is actually judged on.

---

## C. The questions about rigour

### C1. "Why should we trust these numbers?"

> Three protocol commitments. Splits are by dive group, never by image, after we
> measured 57.3% contamination in the supplied partition. All confidence
> intervals come from a bootstrap that resamples whole dives, so correlated
> frames are never split across a resample. And the held-out test set has been
> read exactly once, to establish the baseline — every development result since
> has been produced on separate splits, so nothing has been tuned against the
> data it will finally be judged on.

### C2. "Your uncertainty results are on development splits, not test. Isn't that weak?"

**Concede the fact, defend the choice.**

> It is a limitation and it is stated as one. It is also deliberate: reading the
> test set during design would forfeit the pre-registration. The confirmatory
> comparison against the three pre-registered hypotheses runs once, on test,
> after the design is frozen. Reporting development numbers now and test numbers
> then is the discipline working as intended, not a shortcut.

### C3. "Why three seeds? Why bootstrap over dives rather than images?"

> Three seeds because detector training is stochastic and a single run cannot
> distinguish a real effect from initialisation luck — our first retraining
> attempt showed a spread of ±0.022 mAP which the corrected protocol reduced to
> ±0.005. Bootstrap over dives because frames within a dive are near-duplicates;
> resampling images would treat correlated observations as independent and
> produce intervals that are far too narrow.

### C4. "How do you know your OOD result isn't overfitting?"

> Because we caught ourselves doing exactly that and corrected it. An earlier
> version fitted the distance model on the same split it scored, giving AUROC
> 0.94. Refitting on one split and evaluating on another gave 0.67 — that gap is
> the honest measure of generalisation, and 0.673 is the number we report.
>
> We also found a case where the score fell below chance, at 0.42. Rather than
> report it, we diagnosed it: the retrained network assigns smaller-magnitude
> features to novel objects, so any distance using magnitude inverts. Normalising
> to the unit sphere recovers it to 0.71.

---

## D. The questions about contribution

### D1. "What is actually novel here? This is YOLO plus ResNet."

> YOLO decides what is in the image. This decides whether it has looked hard
> enough to answer at all, and whether the answer is worth the computation. Four
> specific claims:
>
> Decisions are made per organism, not per image — four organisms in one frame get
> four independent judgements. Every adaptive action is instrumented and costed,
> so it is falsifiable rather than assumed. The policy learns from measured
> action outcomes rather than imitating the existing rules, so it can beat its
> teacher. And the open-set evaluation is class-disjoint *and* dive-disjoint, so
> the novelty claim survives scrutiny.
>
> The empirical contribution is that hand-specified adaptive control, measured
> properly, does not help — which nobody in this field had checked.

### D2. "Where does your work sit relative to existing underwater detection systems?"

> Orthogonally rather than competitively. Existing work reports mAP on closed-set
> benchmarks. We report selective risk, calibration and novelty rejection under
> domain shift, and we measure whether each adaptive component earns its compute.
> We are not claiming a better detector; we are measuring an axis that benchmark
> mAP does not capture.

### D3. "Why no quantitative comparison with published methods?"

> Because it would mislead. Their splits leak, their label spaces differ, and
> their task is closed-set. We compare against our own frozen baseline under an
> identical protocol instead. If you want a benchmark comparison, the honest way
> to produce it is to evaluate on URPC or DUO under their standard protocol —
> that is a defined piece of work we have not done.

---

## E. The questions about status

### E1. "Slide 12 says the agentic layer is functional. Show us Q(s,a)."

**Answer this accurately — the corrected slide already says so.**

> The learned accept/abstain policy is functional and beats the tuned threshold
> with an interval excluding zero. The value model Q(s,a) for choosing *which*
> evidence to acquire is in progress, and the slide says so.
>
> We collected the action-outcome data it needs and could not fit it — the zoom
> action produces only 15 outcome changes on the fit split, and a model trained
> on that would be fitting noise. We chose not to report a model of noise. That
> the action is near-inert is itself the finding.

### E2. "How much of this is implemented versus planned?"

> Implemented and measured: the full perception stack, all four decision points,
> the decision trace, per-action cost measurement, the dive-disjoint splits, the
> calibration layer, the OOD scorer, the learned accept/abstain policy, and the
> forced-counterfactual measurement of every discretionary action.
>
> Not implemented: the meta-confidence model, the value model for evidence
> selection, the compute-budget allocator, and everything for video.

### E3. "Why is crinoid at 0.223 when everything else is above 0.6?"

> Two causes and we can separate them. It was data-starved — an earlier split gave
> it 127 training annotations and it scored 0.036; correcting the split to 226
> annotations raised it to 0.223, a sixfold improvement. But it is also
> intrinsically hard: sea-cucumber reaches 0.800 on *fewer* annotations, 170. A
> crinoid is thin, feathery and low-contrast against substrate, and probably
> loses its discriminative structure at 640-pixel input. Higher input resolution
> is the obvious next test.

### E4. "Your training set is 638 images. Is that enough?"

> No, and we say so. It is also severely imbalanced — 2,668 fish crops against 116
> starfish, roughly 23 to 1. That is a stated limitation. The 66.1%
> out-of-vocabulary rate at evaluation is itself evidence that the training
> taxonomy is inadequate for the deployment domain, which is an argument for
> open-set methods alongside taxonomic expansion, since no fixed taxonomy will
> ever be complete.

---

## F. The awkward ones

### F1. "It sounds like your main result is that your own system doesn't work."

> Our main result is that the *perception* stack works when trained on
> representative data, and that the *hand-written decision layer* on top of it
> does not — which we established by measurement rather than assuming either way.
> One of those is a component we replaced with a learned policy that demonstrably
> improves on it. I would rather present a measured negative than an unmeasured
> positive.

### F2. "What would change your conclusion?"

**A good-faith question. Answer it directly — hedging here looks evasive.**

> For the action findings: a larger sample of outcome changes. Zoom altered 9
> detections out of 641 firings, so the interval is wide and a bigger effect could
> hide inside it. For the OOD result: it is a development-split number on 44 to
> 212 detections per comparison, so the test-set confirmation could move it. And
> the re-detection result is confounded by training resolution — a model trained
> at 1280 might reverse it.

### F3. "Isn't this just an engineering project with statistics attached?"

> The statistics are what make it a research contribution rather than a system
> description. Without the counterfactual measurement we would have shipped four
> adaptive components that do not work, and reported them as features — which is
> what the existing literature does, because nobody measures them.

---

## Quick-reference numbers

| Quantity | Value |
|---|---|
| In-domain mAP@0.50 | 0.780 |
| Retrained deep-sea mAP@0.50 | 0.688 ± 0.005 (3 seeds) |
| Out-of-vocabulary rate at evaluation | 66.1% |
| Selective risk, frozen model | 0.888 [0.876, 0.900] |
| Confident labels on unknown taxa | 21.3% |
| Split contamination found | 57.3% |
| ECE, raw → isotonic | 0.467 → 0.016 |
| Brier skill vs constant predictor | +0.025 |
| Novelty AUROC, confidence-based | ≈ 0.47 (chance) |
| Novelty AUROC, representation-based | 0.806 |
| Risk at 30% coverage, baseline → learned | 0.846 → 0.776, CI [+0.049, +0.093] |
| Enhance effect | −0.057 TP/image, CI excludes 0 |
| Zoom effect | +0.0006 / −0.0038, CIs span 0 |
| Full-pipeline inference | 217 ms/image |
