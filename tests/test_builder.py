import dataclasses
import http.client
import json
import threading

import pytest

from embodied_human import HumanConfig, build_human
from embodied_human.builder import (
    catalog,
    config_from_dict,
    preview_project,
    python_source,
    validate_project,
)
from embodied_human.builder.__main__ import make_server
from embodied_human.fatigue import JointRates


def project(config=None, seed=23):
    return {"version": 1, "seed": seed, "config": config or HumanConfig().as_dict()}


def test_catalog_follows_registry_and_carries_parameter_sources():
    models = catalog()
    assert "seev" in models["modules"]["attention"]
    memory = models["modules"]["memory"]["act_r"]
    assert memory["parameters"]["capacity"] == 4
    assert next(
        source for source in memory["sources"]
        if source["parameter"].endswith(".capacity")
    )["citation"] == "Cowan (2001)"


def test_project_roundtrip_restores_structured_parameters():
    config = HumanConfig().replace(
        "fatigue", "three_compartment", joints={"hand": JointRates(0.01, 0.001)}
    )
    data = project()
    data["config"]["fatigue"]["params"] = dataclasses.asdict(
        build_human(config).modules["fatigue"].parameters
    )
    validated = validate_project(json.loads(json.dumps(data)))
    restored = config_from_dict(validated["project"]["config"])
    human = build_human(restored, seed=23)
    assert human.modules["fatigue"].params.joints["hand"] == JointRates(0.01, 0.001)
    assert human.modules["fatigue"].step(0.01, {"hand": 0.5})["hand"] == 1.0


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"version": 2, "seed": 0, "config": {}},
        {"version": True, "seed": 0, "config": {}},
        {"version": 1, "seed": True, "config": {}},
        {"version": 1, "seed": -1, "config": {}},
        {"version": 1, "seed": 0, "config": {"unknown": {}}},
        project({"memory": {"name": "__import__('os')", "params": {}}}),
        project({"memory": {"name": "act_r", "params": {"capacity": 2.5}}}),
        project({"memory": {"name": "act_r", "params": {"capacity": True}}}),
        project({"memory": {"name": "act_r", "params": {"noise": float("nan")}}}),
        project({"memory": {"name": "act_r", "params": {"arbitrary": 1}}}),
        project({"estimator": {"name": "lane_kalman", "params": {"initial_sd": [1]}}}),
    ],
)
def test_project_rejects_invalid_data(data):
    with pytest.raises((ValueError, TypeError)):
        validate_project(data)


def test_preview_is_seeded_and_uses_selected_memory():
    baseline = project()
    assert preview_project(baseline) == preview_project(baseline)
    assert preview_project(baseline)["rows"][2]["accuracy"] == 1.0
    displaced = project(HumanConfig().replace("memory", "act_r", capacity=0).as_dict())
    assert preview_project(displaced)["rows"][2]["accuracy"] < 1.0


def test_variable_length_annotated_tuple_can_select_afferent_subset():
    data = project({"touch": {"name": "touchsim", "params": {
        "afferent_classes": ["RA"],
    }}})
    result = validate_project(data)
    restored = config_from_dict(result["project"]["config"])
    assert build_human(restored).modules["touch"].params.afferent_classes == ("RA",)


def test_exported_python_executes_and_restores_overrides(capsys):
    data = project(HumanConfig().replace("memory", "act_r", capacity=2).as_dict())
    scope = {}
    exec(compile(python_source(data), "human-model.py", "exec"), scope)
    output = json.loads(capsys.readouterr().out)
    assert output == preview_project(data)
    assert scope["human"].modules["memory"].params.capacity == 2


@pytest.fixture
def builder_server():
    server = make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = (response.status, response.read(), response.getheaders())
    connection.close()
    return result


def test_http_serves_assets_and_runs_task(builder_server):
    status, body, headers = request(builder_server, "GET", "/")
    assert status == 200 and b"Build a simulated human" in body
    assert "Content-Security-Policy" in dict(headers)
    for path in ("/builder.js", "/builder.css", "/api/catalog"):
        assert request(builder_server, "GET", path)[0] == 200
    status, body, _ = request(
        builder_server, "POST", "/api/preview", json.dumps(project()),
        {"Content-Type": "application/json"},
    )
    assert status == 200
    assert json.loads(body) == preview_project(project())
    assert request(builder_server, "GET", "/../../pyproject.toml")[0] == 404


def test_http_rejects_cross_origin_and_malformed_imports(builder_server):
    headers = {"Content-Type": "application/json", "Origin": "https://example.org"}
    assert request(builder_server, "POST", "/api/validate", "{}", headers)[0] == 403
    assert request(
        builder_server, "GET", "/api/catalog", headers={"Host": "example.org"}
    )[0] == 403
    headers = {"Content-Type": "application/json"}
    status, body, _ = request(builder_server, "POST", "/api/validate", "{", headers)
    assert status == 400 and "error" in json.loads(body)
    assert request(
        builder_server, "POST", "/api/validate", "{}", {"Content-Type": "text/plain"}
    )[0] == 400
