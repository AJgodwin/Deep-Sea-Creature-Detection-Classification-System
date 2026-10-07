# Panel Review — Speech

**Target length: ~10 minutes** (≈1,500 words at a measured pace). Slide cues in
brackets. Bold lines are the ones to land — slow down and let them sit.

Delivery note: the honest negative results are your strongest material. Say them
at normal volume, without apology and without hedging. The moment you sound
defensive about them, the panel starts probing. Delivered flatly as findings,
they read as rigour.

---

## [SLIDE 1 — Title] ~20 seconds

Good morning. We're Team 32, and our project is deep-sea creature recognition
using an agentic AI system.

I'll cover four things: what problem we're solving, what we built, what we
measured — including some results that surprised us — and where the work goes
next.

---

## [SLIDES 2–3 — Introduction] ~1 minute 15

Marine biologists record thousands of hours of underwater video from robotic
submersibles. Somebody has to identify every organism in that footage, and that
is the bottleneck — there simply aren't enough trained taxonomists.

So automation is attractive. But there's a catch that shapes our entire project.

**An AI that is confidently wrong is worse than no AI at all.** If the system
labels an unknown creature as "fish" with ninety percent confidence, that wrong
record enters a scientific database, and nobody downstream can tell it apart from
a real observation.

So the question we care about isn't "how accurate is it?" It's **"does it know
when it doesn't know?"**

And underwater makes that hard. Red light disappears within a few metres, so
everything turns blue-green with depth. Water scatters light, so contrast is
poor. Submersible lights are uneven. The robot moves, so images blur. And the
organisms themselves are often camouflaged against the seabed.

---

## [SLIDE 4 — Objective] ~45 seconds

Our objective is a system that adapts how hard it looks at an image based on the
evidence in front of it — and that can decline to answer when the evidence isn't
good enough.

Concretely: detect the organisms, classify them, estimate how confident we should
actually be, recognise when we're looking at a species the system was never
trained on, and decide whether more computation is worth spending.

---

## [SLIDES 5–6 — Methodology and Architecture] ~2 minutes

Here's how it works. [SLIDE 6]

The top row is perception, and it always runs.

An image comes in. We first measure its quality — blur, brightness, contrast —
before loading any model, because that's cheap and it tells us what to do next.

Then **two** detectors run, not one. YOLOv8-nano and YOLOv8-small, trained with
different random seeds and different augmentation. That's deliberate: we want
them to fail on *different* images. If two independently trained models both find
something, that's real evidence. Two models that make identical mistakes tell you
nothing by agreeing.

Their outputs are merged by Weighted Boxes Fusion. The standard method, NMS, just
throws away the overlapping box. Ours averages them, so we keep the fact that
both models agreed.

Then a separate ResNet-18 classifier looks inside every box and independently
says what species it is. If the detector and the classifier disagree, that
disagreement is itself a signal.

Now the four decision points, marked D1 to D4.

**D1** — is the image usable, or does it need enhancement?
**D2** — is there enough evidence to analyse this at all? If not, we stop. The
system returns nothing and flags it for a human. That's the refusal gate.
**D3** — is this particular detection ambiguous? If so, zoom in and look again.
**D4** — do the detector and classifier agree on the species? If not, take a
third opinion and vote.

The bottom row is the decision layer, and this is the research contribution. It
takes calibrated uncertainty and a novelty score, and decides: accept, abstain,
or go get more evidence.

Every branch the system takes is written into a decision trace. That matters —
and I'll come back to why.

---

## [SLIDE 7 — Results] ~3 minutes 30

Now the results, and this is the substance of the review.

**First: in the aquarium, it works.** Trained and tested on aquarium imagery,
mAP at 0.5 is 0.78. Full pipeline runs in 217 milliseconds. If we stopped there,
this would look like a success.

**Then we took that exact system, unchanged, and ran it on 8,151 frames of real
deep-sea footage.**

*(pause)*

**It is wrong 88.8% of the time.**

Two-thirds of the organisms in that footage — 66.1 percent — are species outside
its vocabulary entirely. For those, no correct answer exists. And 21.3 percent of
the answers it commits to are confident species labels on creatures it has never
been trained on.

Now, the normal engineering response is: raise the confidence threshold, only
accept the answers it's sure about.

**We measured that. It doesn't work.** The error rate stays flat around 89
percent no matter how selective we are. And the out-of-distribution errors
actually get *worse* at higher confidence — the system's most confident deep-sea
detections are among its most likely to be confident misidentifications.

So the confidence number carries almost no information about correctness. That's
the finding that drives everything else.

**Second result: we made the confidence honest.** It was reporting 56 percent
confidence while being right 9 percent of the time. Isotonic calibration brings
expected calibration error from 0.467 down to 0.016.

I'll be straight about this one — a lot of that improvement is just pushing
everything down toward the true average. What matters is that it stays honest
*while still ranking* detections correctly, which a trivial average-predictor
cannot do.

**Third result, and this is the one I'd point you at.** [gesture to the left
figure]

We asked: can the system tell "I've never seen this creature" from "I know this
one"? Using its confidence score — it's at chance. A coin flip. Slightly worse
than a coin flip, actually.

But the information is still inside the network. When we look at the internal
representation rather than the final confidence score, and retrain on deep-sea
data, novelty detection goes from 0.55 to **0.81**.

**The information needed to recognise something new survives inside the model —
and is destroyed by the final layer that turns it into a species name.**

That's a precise diagnosis, not a complaint. It tells us where to fix it.

**Fourth: the system improves.** [gesture to the lower figure] We trained a model
to decide when to trust a detection. At the same number of answers given, it's
wrong less often — 84.6 down to 77.6 percent, with a confidence interval that
excludes zero.

And here's what makes that interesting. We let it work out for itself what to pay
attention to. It weighted the novelty signal about **five times** more heavily
than confidence — and ignored classifier confidence almost entirely. It found
from data exactly what our analysis had argued separately.

**Fifth, and this is the result that surprised us.**

The system has four adaptive behaviours — enhance the image, zoom in,
re-classify, re-detect. **Nobody had ever checked whether they help.** They'd
been firing on intuition since the system was built.

So we forced each one on every single detection, and scored it against simply not
doing it.

**None of them earns its cost.** Image enhancement is actively harmful — and does
the most damage on precisely the images its own rule selects. Zoom changes almost
nothing: it fired 641 times and altered the outcome of nine detections. The
majority vote is no better than just trusting the classifier.

Removing that entire layer makes the system **more accurate and cheaper**.

Now — every bar on that chart is negative, and I'd still call it our strongest
result. Because it says these rules were never validated, we validated them, and
the learned policy now has a measured baseline to beat instead of assumptions to
imitate. That's only possible because we instrumented every decision from the
start.

**And finally: trained on the right data, the perception works.** Retrained on
deep-sea imagery, mAP is 0.688 across three random seeds with a standard
deviation of 0.005. So the failure I opened with is a *deployment* condition, not
a broken architecture.

---

## [SLIDE 12 — Implementation progress] ~1 minute

On implementation status.

Fully working and measured: the detection ensemble, the fusion, the classifier,
all four decision points, the decision trace, the cost measurement, the
calibration layer, the out-of-distribution scorer, and the learned accept-or-
abstain policy.

In progress: the value model that decides *which* evidence to acquire. We
collected the data it needs and could not fit it — because, as I said, the zoom
action produces almost no outcome changes to learn from. We chose not to report a
model fitted on noise.

I'd rather tell you that than show you a number I don't believe.

---

## [Future work] ~1 minute 30

Three directions, and each one comes directly out of a measurement.

**One.** We showed the novelty signal lives in the network's internal
representation once it's trained on the right domain. So the next step is
training *for* that explicitly — an open-set objective — rather than reading it
out afterwards.

**Two.** We showed hand-written image enhancement makes things worse. The fix is
a learned enhancement that models how light actually behaves underwater, trained
together with the detector rather than bolted on in front of it.

**Three — video.** And there's a reason we expect video to succeed where some of
this didn't. Zooming in re-reads the same pixels; it adds computation but no new
information. **Another frame is genuinely new information** — a different angle, a
different moment. We also expect it to attack our largest error source directly:
42 percent of our detections are boxes drawn on empty water, and a false
detection doesn't persist across frames while a real organism does.

There's a trap we've already designed around. A novel organism gets *consistently*
misclassified in every frame. If we treat consistency as evidence of correctness,
video would give false confidence to exactly the errors we're trying to catch. So
the novelty score has to be computed per-organism and allowed to override
consistency.

---

## [SLIDE 13 — Close] ~30 seconds

To summarise.

We built an underwater recognition system, and then we measured it honestly — on
leak-free data splits, with confidence intervals, and with the test set read
exactly once.

It fails badly outside its training domain, and we can say precisely why. Its
confidence is meaningless, and we fixed that. It cannot recognise unfamiliar
creatures from its confidence score — but the information is in the network, and
we can get it out. And the clever adaptive parts don't earn their cost, which we
only know because we measured them.

**The contribution isn't a higher score. It's that every adaptive decision in
this system is instrumented, costed, and falsifiable — and when we tested them,
several turned out to be false.**

Thank you. Happy to take questions.

---

## Delivery checklist

- **Pause after "88.8% of the time."** Let it land. It's your best moment.
- **Don't apologise for the negative results.** Say them flatly. "Nobody had
  checked" is a strong line — don't rush it.
- **The five-times-more-heavily line** is your proof point. Slow down there.
- **If you're running short on time**, cut the calibration result (result two).
  It's the weakest and the most likely to invite the constant-predictor question.
- **If asked mid-talk about Q(s,a)**, answer it straight from slide 12 and move
  on. Don't improvise a defence.
