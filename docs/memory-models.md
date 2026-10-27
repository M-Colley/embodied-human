# Memory models and current alternatives

Reviewed 26 October 2026. Yes, newer work exists. The package's `act_r`
name denotes a small reimplementation of selected memory equations, not
an installed ACT-R release. Upgrading ACT-R software would not change this
implementation automatically.

## What is implemented

`ActRMemory` has an immediate, capacity-limited focus and keyed declarative
memory. Repeated encodings add traces; displaced items use a power-law
base level, logistic activation noise, a retrieval threshold and an
activation-dependent latency. The focus has no time-based eviction, and
retrieving an item does not automatically rehearse it or move it into the
focus. Calls to `encode` model rehearsal explicitly.

The implementation has no content similarity, partial-match retrieval,
competitor intrusions, context binding, spreading activation, blending or
procedural production system. Re-encoding an existing key replaces its
value while retaining its presentation history. Consequently, a digit
task using unique position keys measures loss of accessibility, without
modeling confusion between similar digits or positions. The four-item
focus is a package modeling choice informed by Cowan's aggregate capacity
estimate, not a complete implementation of ACT-R's buffers.

Memory-span benchmarks that compare the focus against the source of its
capacity parameter are consistency checks. They do not validate the
unimplemented mechanisms above.

## Newer work that is useful here

| Work | What it offers | What integration would require |
| --- | --- | --- |
| [Official ACT-R reference manual, currently titled 7.30+](https://act-r.psy.cmu.edu/actr7.x/reference-manual.pdf) | A maintained architecture, with declarative-memory features beyond this baseline. Software release numbers and theoretical mechanisms are separate matters. | An optional runtime adapter and explicit mapping to ACT-R buffers and timing, or equation-level additions with independent tests. The software download page timed out during this review, so the exact newest release is not asserted. |
| [Stocco et al. (2024), *An Integrated Computational Framework for the Neurobiology of Memory Based on the ACT-R Declarative Memory System*](https://doi.org/10.1007/s42113-023-00189-y) | A recent account of contextual activation, partial matching and blending, plus extensions involving episodic context and emotion. This is a useful specification for expanding the current model. | Structured features and retrieval cues, candidate competition and tests of intrusion errors; additional parameters need their own provenance and calibration. |
| [Oberauer & Lin (2024), *An interference model for visual and verbal working memory*](https://doi.org/10.1037/xlm0001303) | A formal alternative covering visual-feature errors and verbal serial recall through imprecise content-context bindings, a focused item and interference. | A context/similarity interface and tasks that score nontarget intrusions, serial position and continuous errors. Treating it as a changed ACT-R decay constant would omit its central mechanisms. |
| [Hake et al. (2026), *Reliable detection and continuous monitoring of memory dysfunction in mild cognitive impairment and healthy aging through adaptive computational phenotyping*](https://doi.org/10.1371/journal.pdig.0001686) | A September 2026 application of an adaptive trace model: subsequent trace decay depends on activation at encoding, with individual forgetting estimates inferred from accuracy and response time. Useful for repeated learning and participant differences. | Trace-specific decay, learning histories and participant-level calibration. Its paired-associate assessment setting does not establish validity for driving or short-term dual-task recall. |
| [Stocco et al. (2024), *Fitting, Evaluating, and Comparing Cognitive Architecture Models Using Likelihood*](https://arxiv.org/abs/2410.18055) | Methods for estimating and comparing ACT-R models using accuracy and response times. Figure 7 uses `d = 0.5, s = 0.25` as an illustration. | A fitting dataset and separate held-out evaluation. The package's `noise = 0.25` is therefore labeled **assumed**, rather than claiming a fit to its own tasks. |

## Recommended next model

For the existing n-back and interrupted-task use cases, implement the
Oberauer–Lin interference model as an additional registered memory module
after defining cue/context representations. Preserve `act_r` as the
reproducible baseline. Compare both models on published serial-position,
intrusion and load effects with separate fitting and held-out conditions.
For long-term learning, use trace-specific adaptive decay as a separate
extension; it answers a different empirical question.

These are implementation recommendations, not claims that the newer
models have already been replicated here. No new model or fitted constant
has been added on the strength of a paper's publication date alone.

## Robustness fixes in this review

The existing baseline now rejects fractional or negative focus capacities,
nonfinite parameters, negative decay/noise and nonpositive latency factors.
It evaluates base-level activation in log space, computes the logistic
probability without exponential overflow, and combines the latency scale
in log space. Unrepresentably long retrieval times return infinity.

Times must be finite and cannot precede the latest encoding. This prevents
retrieving a future value from the current focus. Historical queries need
a separate snapshot, and `reset()` starts a new history. At zero trace
age, immediate focused retrieval remains available; the singular
power-law term is excluded from declarative activation until its age is
positive, preserving the baseline convention. Numeric defaults are
unchanged.
