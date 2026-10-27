# Optional memory and choice models

The registered `memory=pyactup` and `response=pyddm` implementations delegate computation to MIT licensed upstream packages. PyACTUp 2.2.5 was released in March 2026 and PyDDM 0.9.0 in September 2025. Their licences permit commercial use with their notices retained. The adapter settings and demonstration protocols are explicit assumptions. Human calibration remains pending. Source information is available in the [PyACTUp package](https://pypi.org/project/pyactup/), [PyACTUp licence](https://github.com/cmu-psych-fms/pyactup/blob/main/LICENSE), [PyDDM release notes](https://github.com/mwshinn/PyDDM/blob/master/CHANGELOG.md) and [PyDDM licence](https://github.com/mwshinn/PyDDM/blob/master/LICENSE.txt).

Install the selected extras with `python -m pip install 'embodied-human[memory,decision]'`. Importing the package, constructing either implementation and obtaining parameter provenance do not import the optional libraries. Native computations import them when needed. The original `act_r` memory and `twin` response remain the defaults, with their existing parameters and interpretation.

## PyACTUp memory

`PyACTUpMemory.encode(key, value, time)` and `retrieve(key, time)` support the existing Memory interface. Focused keys remain immediately available. A displaced key is retrieved by the actual PyACTUp declarative memory. Payloads can be arbitrary Python objects. `encode_chunk(key, value, time, features)` additionally binds hashable features to a key. Repeating a key rehearses its traces and can replace its payload; changing its feature bindings requires a new key, so distinct episodes cannot silently overwrite each other.

`retrieve_cued(cues, time, partial=True)` accesses the native declarative pool directly and returns the retrieved key, payload, feature bindings, noisy activation and latency. It does not use the immediate focus or rehearse the result. `similarity(attribute, function=None, weight=1.0)` enables partial matching for a feature. Without a function, native binary equality is used; a supplied function must be symmetric and return a similarity between zero and one. `blend(attribute, cues, time)` uses native activation weighted blending of a numeric feature. Similarity specifications survive reset; learned traces do not. Custom functions are executable experimental specifications and must be retained with the experiment source, not assumed to be encoded in a JSON configuration.

Model time is interpreted as seconds. Encodings must be nonnegative and chronological. Declarative queries must occur strictly after the latest encoding, matching PyACTUp's positive trace age requirement. Upstream retrieval includes activation equal to its threshold; this differs from the original baseline's strict comparison at the exact boundary. Noise, decay, focus capacity and the latency factor retain the existing source labels. Mismatch penalty and blending temperature are explicitly assumed settings, not measured human constants. The latency wrapper uses the existing `F exp(-activation)` relationship.

PyACTUp 2.2.5 calls Python's module level random generator in its constructor and retrieval tie selection. The adapter creates two function objects from the installed implementation with a copied globals dictionary and an independent `random.Random` binding. Their native code is unchanged. No upstream module is patched and no global random state is changed or restored. Activation noise uses the adapter's NumPy Generator through the native noise distribution hook. Real package tests check repeated seeds, global RNG preservation and independent retrieval ties in concurrent threads. This small compatibility mechanism is tested against the declared 2.2 series and should be reviewed when upgrading that dependency.

The following experiment contrasts exact context retrieval with partial context competition on the same encoding history. The later list deliberately repeats the payloads. A returned item from that later list is still an intrusion because both identity and content are scored. Each serial position is probed at the same final time, without rehearsal. The task respects immediate target focus access for both memory implementations and uses native cue competition only after that target leaves the focus. The explicit `retrieve_cued` API itself continues to access only declarative memory. These are task assumptions rather than a fitted theory of focus control or output interference.

```python
from embodied_human.memory_pyactup import PyACTUpMemory
from embodied_human.tasks.recall import run_serial_recall

for partial in (False, True):
    memory = PyACTUpMemory(
        capacity=0, noise=0, threshold=-5, mismatch_penalty=0.1
    )
    result = run_serial_recall(
        memory, [1, 2, 3], distractors=[[1, 2, 3]], partial=partial, seed=4
    )
    print(partial, result.summary())
```

With these illustrative settings, exact context retrieval recalls the target list and partial matching retrieves the later list. Tests also check the serial position accessibility gradient from native decay and threshold, plus activation, latency and blending against their equations. A successful keyed retrieval with the wrong payload is scored as a separate content error because its actual source identity is unknown. Native identities allow true intrusions to be distinguished from those errors. Comparisons that change the number of distractor lists also change target retention time; a causal claim about interference requires controlling that timing. These are implementation consistency checks. This backend is not an implementation or replication of the Oberauer and Lin interference model, whose context binding and updating mechanisms differ. No published human intrusion or recall curve has been fitted or claimed as replicated here.

## PyDDM choices and response times

`PyDDMResponse.solve_choice(drift, deadline)` calls the actual [PyDDM solver](https://pyddm.readthedocs.io/en/stable/quickstart.html) for a constant drift process between symmetric absorbing bounds. Positive drift favours the upper boundary. Drift and the correct boundary are experimental inputs; the adapter does not infer evidence from visual, auditory or tactile Signals. The returned distribution includes the joint density for each boundary, unconditional choice probabilities, deadline miss probability and the mean response time conditional on responding. `accuracy(correct, conditional=False)` makes the distinction between all trial accuracy and accuracy among completed responses explicit. The [native Solution API](https://pyddm.readthedocs.io/en/stable/apidoc/model.html) defines these probability and density operations.

The deadline includes the independently specified nondecision time. Native contaminant mixture probability is set to zero explicitly. Solver time step is reduced slightly when necessary so the final grid point reaches the requested deadline without increasing the configured maximum step. Densities integrate using PyDDM's discrete rule, `sum(density) * dt`. `sample_choices(distribution, count)` samples those joint grid masses with the module's NumPy Generator, producing times on the numerical grid rather than continuous timestamps. Nondecision shifts also follow the native grid resolution. A timeout has no boundary choice and no invented reaction time. `decide(drift, deadline)` is a convenience method that solves and samples one trial; repeated trials should solve once and sample the resulting distribution together.

```python
from embodied_human.decision import PyDDMResponse
from embodied_human.tasks.choice import ChoiceCondition, run_choice_task

response = PyDDMResponse()
rows = run_choice_task(
    response,
    [ChoiceCondition(1.0, 2.0, "upper"), ChoiceCondition(1.0, 0.1, "upper")],
    trials=10000,
    seed=4,
)
for row in rows:
    print(row)
```

The task reports observed accuracy, accuracy among responses, deadline misses and mean response time alongside the solver's predictions. Tests compare the constant drift solution with the analytical first passage probability `1 / (1 + exp(-2 * drift * bound / noise**2))` for a centred start and long deadline. They also check the corresponding mean time, drift reversal, unbiased symmetry, probability mass, finite deadlines and seeded sampling. These are solver consistency checks, not human validation. Bounds, diffusion noise, starting position and nondecision time need fitting on one dataset and separate held out evaluation for a target human choice task.

Inherited `respond(signals, load)` and `visual_latency(eccentricity)` still execute the original TWIN detection pathway, with its original parameter values. Choosing `response=pyddm` therefore adds an explicit choice experiment API; it does not change DRT or takeover predictions through an invented evidence mapping. The new PyDDM module does not modify `response.py` or the fitted central load constants.

The experiment runner exposes these demonstrations as `memory.serial_recall` and `response.choice`, separately from the existing published benchmark registry. The serial recall protocol repeats six target probes after a later list and records accuracy by serial position, failures, content errors and identity based intrusions. The choice protocol runs six explicit drift and deadline conditions and records observed outcomes beside native solver predictions. Its aggregate accuracy gives each condition equal weight, while aggregate mean response time weights conditions by the number of answered trials. At zero drift, the upper boundary is an assumed target label rather than a direction supported by evidence. Fixed task settings have their own Parameters dataclasses and assumed source labels, included in each result. A response implementation without the choice API is skipped with an explanation. These protocols produce unscored model demonstrations until independent human evaluation data are supplied; they do not increase the number of validated human results.
