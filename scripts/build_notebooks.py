"""Build and execute the 18 lecture notebooks and the 18 solution notebooks.

The text and code of the lectures are kept as plain Python data in
``scripts/lectures_00_05.py``, ``lectures_06_11.py`` and ``lectures_12_17.py``; the
figure cells are in ``scripts/lecture_figures.py``.  This script assembles the
notebooks, runs each one in a fresh Jupyter kernel and stores it with its outputs.

Run from the repository root:

    python scripts/build_notebooks.py            # all 36 notebooks
    python scripts/build_notebooks.py 05 14      # lectures 05 and 14 only
"""

import hashlib
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lecture_figures import FIGURES  # noqa: E402
from lectures_00_05 import LECTURES_00_05  # noqa: E402
from lectures_06_11 import LECTURES_06_11  # noqa: E402
from lectures_12_17 import LECTURES_12_17  # noqa: E402

CURRICULUM = LECTURES_00_05 + LECTURES_06_11 + LECTURES_12_17

# folder -> (title of the part, lectures)
PARTS = {
    "part1_conservation": ("Part I. Conservation as a constraint", ["00", "01", "02"]),
    "part2_shape": ("Part II. Monotone, convex and robust fits", ["03", "04", "05"]),
    "part3_chemometrics": ("Part III. Chemometrics and process monitoring", ["06", "07", "08", "09"]),
    "part4_clustering": ("Part IV. Clustering and phases", ["10", "11", "12"]),
    "part5_ensembles": ("Part V. Trees and ensembles", ["13", "14", "15"]),
    "part6_guardrails": ("Part VI. Prediction intervals and a capstone", ["16", "17"]),
}
PART_OF = {num: folder for folder, (_, nums) in PARTS.items() for num in nums}

PREAMBLE_CODE = r'''
try:                      # use the installed package, or the copy in this repository
    import engml_2_applied
except ImportError:
    import pathlib, subprocess, sys
    for _root in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        if (_root / "engml_2_applied" / "__init__.py").exists():
            sys.path.insert(0, str(_root))
            break
    else:                 # on Google Colab: install from GitHub
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "git+https://github.com/ktwyw/engml-2-applied.git"], check=True)
    import engml_2_applied

import inspect
import numpy as np
import matplotlib.pyplot as plt
from engml_2_applied import (clustering_phases, constants, descriptors, dim_reduction, linear_physics,
                             trees_ensembles, uncertainty)
from engml_2_applied.plotting import panel_label, use_style

use_style()

def show_source(obj):
    """Return the source code of a library function or class."""
    return inspect.getsource(obj)

def heading(text):
    print(f"\n{text}\n" + "=" * len(text))
'''


def md_cell(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


def code_cell(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def figure_cells(num: str, after: str) -> list:
    """The figure cells (code, then caption) that follow the demonstration ``after``."""
    cells = []
    for k, (where, code, caption) in enumerate(FIGURES.get(num, []), start=1):
        if where == after:
            cells.append(code_cell(code))
            cells.append(md_cell(f"**Figure {int(num)}.{k}.** {caption}"))
    return cells


def save_nb(path: Path, cells: list) -> None:
    """Execute the cells in a fresh kernel, in the notebook's own folder, and store the result."""
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    for idx, cell in enumerate(nb.cells):      # stable cell ids and no machine-specific metadata
        cell["id"] = hashlib.sha256(f"{path.name}:{idx}".encode("utf-8")).hexdigest()[:8]
        cell["metadata"] = {}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, path)


SILENT_START = "import io, sys\n_stdout, sys.stdout = sys.stdout, io.StringIO()   # the demonstrations are repeated without their output\n"
SILENT_END = "\nsys.stdout = _stdout\nprint(\"Set-up complete: the variables of the lecture's demonstrations are defined.\")"


def lecture_cells(item: dict) -> list:
    num = item["num"]
    objs_md = "\n".join(f"{i}. {obj}" for i, obj in enumerate(item["objectives"], start=1))
    cells = [
        md_cell(f"# {item['title']}\n\n{item['hook']}"),
        md_cell(f"## Learning objectives\n\n{objs_md}\n\n**Time:** about 60 min"),
        code_cell(PREAMBLE_CODE),
        md_cell(item["theory_md"]),
    ]
    for k in (1, 2, 3, 4):
        cells.append(code_cell(item[f"demo{k}_code"]))
        cells.extend(figure_cells(num, f"demo{k}"))
        cells.append(md_cell(item[f"post{k}_md"]))
    cells += [
        code_cell(item["inside_code"]),
        md_cell(
            "## Exercises\n\n"
            f"1. **Verify a result.** {item['ex1_prompt']}\n"
            f"2. **Find how it fails.** {item['ex2_prompt']}\n"
            f"3. **Write it yourself.** {item['ex3_prompt']}\n\n"
            "The cell below works Exercise 1 as a first check; all three are solved in the solutions notebook of this lecture."
        ),
        code_cell(item["ex1_code"]),
    ]
    return cells


def solution_cells(item: dict) -> list:
    num, slug, part = item["num"], item["slug"], PART_OF[item["num"]]
    setup = "\n".join(item[k] for k in ("demo1_code", "demo2_code", "demo3_code", "demo4_code", "inside_code"))
    return [
        md_cell(
            f"# Solutions · {item['title']}\n\n"
            f"Worked solutions to the three exercises of "
            f"[Lecture {int(num)}](../../notebooks/{part}/{slug}.ipynb). Each solution is computed here and "
            f"checked with `assert` statements against an exact identity or an independent calculation. "
            f"The first code cell sets up the notebook and repeats the demonstrations of the lecture without "
            f"printing them, because the solutions use their variables."
        ),
        code_cell(PREAMBLE_CODE + "\n" + SILENT_START + setup + SILENT_END),
        md_cell(f"## Exercise 1\n\n**Problem.** {item['ex1_prompt']}"),
        code_cell(item["ex1_code"]),
        md_cell(f"**Discussion.** {item['ex1_exp']}\n\n---\n\n## Exercise 2\n\n**Problem.** {item['ex2_prompt']}"),
        code_cell(item["ex2_code"]),
        md_cell(f"**Discussion.** {item['ex2_exp']}\n\n---\n\n## Exercise 3\n\n**Problem.** {item['ex3_prompt']}"),
        code_cell(item["ex3_code"]),
        md_cell(item["ex3_takeaway"]),
    ]


def main(only: list) -> None:
    assert len(CURRICULUM) == 18, f"Expected 18 lectures, got {len(CURRICULUM)}"
    for item in CURRICULUM:
        num, part = item["num"], PART_OF[item["num"]]
        if only and num not in only:
            continue
        save_nb(REPO_ROOT / "notebooks" / part / f"{item['slug']}.ipynb", lecture_cells(item))
        save_nb(REPO_ROOT / "solutions" / part / f"{num}_solutions.ipynb", solution_cells(item))
        print(f"built lecture {num} and its solutions")


if __name__ == "__main__":
    main(sys.argv[1:])
