# Contact sensing on the existing body

The optional `touch=contact_map` module reports physical contacts and spatial force maps on an existing MuJoCo body. It binds the caller's `MjModel` and `MjData` without rebuilding the body, changing collision geometry, changing muscle control, or advancing the engine. Locations use metres and forces use newtons when the model uses SI units. The usual vibration detection, intensity and latency methods retain the existing psychophysical baseline.

This module reuses a small attributed subset of [MIMo's sensor geometry](https://github.com/trieschlab/MIMo/blob/040b0ae4914cbfb26afdf830aa81775b90922f3f/mimoTouch/sensorpoints.py) and its [linear force scaling](https://github.com/trieschlab/MIMo/blob/040b0ae4914cbfb26afdf830aa81775b90922f3f/mimoTouch/touch.py), adapted from commit `040b0ae4914cbfb26afdf830aa81775b90922f3f`. The retained [MIT notice](../src/embodied_human/_vendor/MIMO_LICENSE.txt) includes the original authors and is included in both the source distribution and the wheel, including the wheel's licence directory. The package metadata therefore declares `Apache-2.0 AND MIT`; the package's original code remains under Apache 2.0 and the adapted subset retains MIT. The subset removes dependencies on the MIMo environment, Gymnasium, Trimesh, training libraries and plotting. It adds allocation limits, keeps geometric surface points when sampling is coarse, and normalizes scalar spatial weights so purely tangential forces are conserved.

The complete MIMo package is not installed. Its verified requirements pin MuJoCo 3.3.0 and SciPy 1.13.1, which conflict with this package's MuJoCo extras and SciPy requirement. Its NumPy 2.0.2 pin alone does not conflict with this package's NumPy lower bound. The adapted subset avoids those dependency pins. The contact module adds no core dependency and imports MuJoCo only when `bind` is called. Install `python -m pip install 'embodied-human[contact]'`, or use `python -m pip install -e '.[contact]'` from a source checkout.

For a running full body driver, attach the reader directly to its existing physical model and data:

```python
import embodied_human as eh

human = eh.build_human(
    eh.HumanConfig().replace("touch", "contact_map"), seed=42
)
touch = human.touch
touch.bind(driver.model, driver.data)

# Let the existing driver or engine advance as usual before reading contacts.
reading = touch.sense()
for body, forces in reading.sensor_forces.items():
    print(body, forces.sum(axis=0))
for contact in reading.contacts:
    print(contact.body_id, contact.position, contact.force)
```

With no selection, automatic sampling includes every body with contact enabled rigid geoms except the world body. This can include cockpit mechanisms as well as the human. Pass `bodies=[...]` containing the desired body names or IDs to restrict the map to human parts. Each body is considered separately; selecting a parent does not automatically select its descendants. The adapter supports boxes, spheres, ellipsoids, capsules and cylinders through adapted MIMo sampling and uses existing compiled mesh vertices for mesh geoms. Planes and height fields need an explicitly bounded sensor map.

Custom maps specify body frame coordinates and optional outward normals. Keys are body names or IDs and appear unchanged in the readout dictionaries. Supplied arrays are copied, validated and held read only inside the module. Each readout contains detached arrays so changing a result cannot affect the next reading.

```python
import numpy as np
from embodied_human.touch_contact import SensorMap

# Replace the example body key and coordinates with those of the actual model.
sensor_map = SensorMap(
    positions=np.array([[0.0, 0.0, 0.02], [0.01, 0.0, 0.02]]),
    normals=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
)
touch.bind(driver.model, driver.data, sensors={body_name: sensor_map})
```

`ContactReadout.contacts` contains one record per solved rigid contact and sensing side. A contact between two sensing bodies therefore appears twice, with equal and opposite world forces. Each record supplies the body and geom IDs, the contact midpoint in body coordinates, the force acting on that body in body coordinates, the intrinsic contact torque in the same frame, an outward contact normal, and the world contact midpoint and force. Contact position is MuJoCo's midpoint between geoms, not a reconstructed skin surface. Flexible contacts are outside this adapter's supported scope. See the official [contact structure](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjcontact) and [contact force API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce).

The `spread_linear` response gives each eligible sensor a weight that decreases linearly to zero at `spread_radius`, then normalizes the weights to conserve all three components of that body's contact force. The `nearest` response allocates the full vector to one sensor. When normals are supplied, sensors facing away from the contacting surface are excluded. When a linear map has no sensor within its radius, or no supplied normal faces the contact, the nearest sensor receives the force and the fallback is reported through `nearest_fallback` and `fallback_count`. These fallbacks preserve force but can indicate insufficient sampling coverage.

The reused subset does not include MIMo's full Trimesh interior sensor removal or mesh adjacency search. Distances are Euclidean, coincident points are deduplicated, and points inside overlapping geoms are not removed. Automatic mesh vertex maps have no inferred outward normals, so provide explicit normals when orientation filtering is required. Spatial redistribution conserves net force, not the moment of the distributed force about the body origin; intrinsic contact torque remains available in each contact record and is not converted into force sensor output.

`sensor_spacing`, `spread_radius`, `response` and `max_sensors_per_body` are explicitly assumed protocol or implementation choices, not fitted receptor density, skin stiffness or human discrimination thresholds. Their sources are exposed by `touch.params.provenance()`. The default spacing is 0.02 m, the radius is 0.04 m, and automatic allocation is limited to 20000 sensors per body. Human reset keeps the existing physical binding and does not reset engine state; after recompiling or replacing a model, bind its new model and data again.

Contact force and location do not determine skin indentation without a contact mechanics model. This adapter does not feed forces into TouchSim, infer afferent firing, or claim improved detection or takeover behaviour. Its conservation and frame checks are engineering verification, not fitted or held out human validity. The existing aggregate vibration comparisons remain consistency checks of their source parameters.
