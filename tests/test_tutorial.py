"""The module tutorial notebook runs from top to bottom."""

from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "examples" / "tutorial_modules.ipynb"


def test_module_tutorial_runs():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "error"
    ]
    assert not errors
