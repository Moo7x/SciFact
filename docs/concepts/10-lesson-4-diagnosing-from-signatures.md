# Lesson 4 — Diagnosing training from signatures, not single features

*2026-10-01. Debrief of the closing questions of `lessons/04_training_step.py`.*

## The real Stage 3 log

```
step        10      60     110     160     210     260     310     360
loss     1.078   1.045   0.995   0.880   0.837   0.782   0.860   0.754
grad      1.81    4.65    2.36    2.59    5.05    5.55    5.74   10.63
```

## Q1 — which bugs can be ruled out?

**Mounir's answer:** the gradient size goes quite high, so it could be the missing-`zero_grad` bug.

Noticing the rise was right. The conclusion was not, because a diagnosis matches the **whole
signature**, not one feature of it.

The missing-`zero_grad` signature, from the mystery run:

```
grad   2.26  15.93  42.85  81.84  99.30  105.43     <- within 40 steps
loss   falls, then STALLS around 0.46
```

Feature by feature against Stage 3:

| feature of the bug | Stage 3 | match? |
|---|---|---|
| grad rises **every** step | 1.81 -> 4.65 -> **2.36** (it went down) | no |
| grad reaches **hundreds** | after 360 steps, still ~10 | no; summing 360 gradients would give far more |
| loss **stalls** | keeps falling, 1.078 -> 0.754 | no |

**All four bugs are ruled out:**

| bug | its signature | Stage 3 |
|---|---|---|
| no `backward` | grad exactly 0 | grad 1.8-10.6 |
| no `step` | loss never falls | loss falls 30% |
| lr too high | loss rises above its start, hovers near 1.099 | falls steadily |
| no `zero_grad` | grad grows every step into the hundreds; loss stalls | non-monotonic, ~10 max; loss falling |

The code itself confirms it: `optimizer.zero_grad(set_to_none=True)` is in the loop. Reading the
code is a legitimate diagnostic step too, and often the fastest one.

### So why does the gradient rise late?

Three ordinary reasons:

1. **The two rows are not measured the same way.** The loss row averages 10 steps; the grad row
   is a single batch of 16. One batch is noisy, which is why it jumps around (4.65 -> 2.36).
2. **Confident mistakes produce big gradients.** Cross-entropy's gradient grows as the model gets
   more confident in a wrong answer. Late in training the model is more confident, so its wrong
   answers cost more. Class weighting amplifies this: a wrong CONTRADICT counts 2.64x.
3. **The logged number is measured before clipping.** `clip_grad_norm_(..., 1.0)` shrinks any
   gradient longer than 1.0 before the weights are updated. The step that logged 10.63 actually
   moved the weights by a gradient of length 1.0. Large logged gradients here are information,
   not damage.

**The rule:** "it goes up" is a feature. A diagnosis needs the signature: the shape, the scale,
and what the other signals do at the same time. Matching on one feature is how a correct run
gets "fixed" into a broken one.

## Q2 — could the Stage 3 log (loss + grad, no weight change) have missed a bug?

This question separates two different jobs:

- **Detecting** that something is wrong.
- **Identifying** which bug it is.

| bug | detected with loss + grad? | identifiable with loss + grad? |
|---|---|---|
| no `backward` | yes: grad is 0 | yes, unmistakably |
| no `zero_grad` | yes: grad explodes | yes |
| no `step` | yes: loss doesn't fall | **only barely** |
| lr too high | yes: loss doesn't fall | **only barely** |

The last two both look like "loss stuck near 1.1, grad a few units". The only difference in loss
and grad is that the high-learning-rate run first *rises* above its starting loss (1.147 ->
1.221) while the step-less run wobbles below it. On real, noisy batches that difference is easy
to miss. Weight change separates them with no ambiguity:

```
no step     weight moved   0.000  0.000  0.000 ...   nothing is applied
lr too high weight moved  16.6   45.0   93.3   ...   far too much is applied
```

**Answer:** no bug would have gone completely unnoticed, but two of them could have been
confused with each other, and they need opposite fixes: add the missing line in one case, lower
the learning rate in the other. A third signal turns "something is wrong" into "this is what is
wrong".

**Follow-up for Stage 3b:** the training script logs loss and gradient size only. It should also
log the size of each update, so this ambiguity cannot arise in a real run.
