# Local no-code module builder

Install the package, then launch the builder:

```bash
pip install -e .
python -m embodied_human.builder
```

Open the printed address, normally <http://127.0.0.1:8765>. The builder uses
Python's standard-library web server and the package's existing NumPy/SciPy
dependencies. It needs no JavaScript package installation, internet connection,
account or hosted service. Use `--port 8766` if the default port is occupied.

The library is generated from the live module registry. Drag an implementation
onto its category, click its library button, or use the category's selection
menu. Each composition contains one model per category, as in `HumanConfig`.
This editor does not create arbitrary signal connections or new model equations.

Select a category to inspect its parameters and baseline sources. Numbers,
booleans and strings have dedicated editors; structured parameters use JSON.
Changing a value labels it as a user override. Source labels describe the
original baseline value; they do not establish that a new value is measured or
validated. Reset individual parameters or return the entire composition to the
baseline. Optional implementations can require separately installed extras.

**Validate model** constructs every selected module with the given random seed.
It catches unknown categories, implementation names, parameters, invalid JSON
types and model constructor errors. It does not establish scientific validity
or replace model-specific domain checks and benchmarks.

**Run seeded preview** runs the package's delayed-digit-recall task with 24
digits at each level from 0-back to 5-back. Each level starts with the same
seeded streams and fresh memory state. Only the memory module influences this
demonstration; other selected modules are constructed but are not exercised by
the task. The displayed latencies are memory retrieval times, not full verbal
response times. These results are a demonstration, not a human-data benchmark.

**Export JSON** saves the complete composition, its overrides, format version
and seed. **Import JSON** validates a saved project before replacing the current
composition. Files contain data only; the importer never imports arbitrary
Python modules or evaluates code. Only already registered implementations can
be selected. Structured parameters such as fatigue joint rates are restored
to their required dataclass types.

**Export Python** saves a runnable script that builds the composition and prints
its delayed-recall preview. Run it with the package installed:

```bash
python human-model.py
python -m embodied_human.builder --run human-config.json
```

To use the saved composition elsewhere:

```python
import json
from pathlib import Path

from embodied_human import build_human
from embodied_human.builder import config_from_dict

project = json.loads(Path("human-config.json").read_text())
human = build_human(config_from_dict(project["config"]), seed=project["seed"])
```

The server binds only to `127.0.0.1`, serves only its bundled assets and API,
rejects foreign origins and hostnames, and limits JSON requests to 256 KiB.
It is a local research tool. Keep the terminal open while using it and press
Ctrl+C to stop. No files are written until a browser download is requested.
