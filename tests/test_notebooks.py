"""Checks of the stored notebooks: all 36 are present, executed without errors, and listed in the book."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LECTURES = sorted((ROOT / "notebooks").glob("part*/*.ipynb"))
SOLUTIONS = sorted((ROOT / "solutions").glob("part*/*.ipynb"))


def _cells(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def test_there_are_18_lectures_and_18_solution_notebooks():
    assert [p.name[:2] for p in LECTURES] == [f"{n:02d}" for n in range(18)]
    assert [p.name[:2] for p in SOLUTIONS] == [f"{n:02d}" for n in range(18)]


@pytest.mark.parametrize("path", LECTURES + SOLUTIONS, ids=lambda p: p.stem[:12])
def test_notebook_is_stored_executed_and_free_of_errors(path):
    for cell in _cells(path):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is not None, "a code cell was not executed"
        for output in cell["outputs"]:
            assert output["output_type"] != "error", "a cell stored an error"
            assert output.get("name") != "stderr", "a cell wrote to stderr"


@pytest.mark.parametrize("path", LECTURES, ids=lambda p: p.stem[:12])
def test_lecture_has_two_figures_with_captions_and_three_exercises(path):
    cells = _cells(path)
    n_figures = sum("image/png" in output.get("data", {}) for cell in cells for output in cell.get("outputs", []))
    text = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown")
    number = int(path.name[:2])
    assert n_figures == 2
    assert f"**Figure {number}.1.**" in text and f"**Figure {number}.2.**" in text
    assert "## Exercises" in text and "\n3. **" in text


def test_every_notebook_is_in_the_table_of_contents():
    toc = (ROOT / "myst.yml").read_text(encoding="utf-8")
    for path in LECTURES + SOLUTIONS:
        assert path.relative_to(ROOT).as_posix() in toc


def test_notebook_code_parses_on_this_python():
    """Every code cell must be valid syntax on the interpreter that runs the tests (3.10 in the workflow)."""
    import ast

    for path in sorted(ROOT.glob("notebooks/*/*.ipynb")) + sorted(ROOT.glob("solutions/*/*.ipynb")):
        for index, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
            if cell["cell_type"] == "code":
                try:
                    ast.parse("".join(cell["source"]))
                except SyntaxError as error:  # pragma: no cover - reported as a failure
                    raise AssertionError(f"{path.name}, cell {index}: {error}") from error
