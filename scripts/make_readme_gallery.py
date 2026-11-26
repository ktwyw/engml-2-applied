"""Gallery of README.md: one figure from each lecture, taken from the executed notebooks.

The figures are not redrawn here: they are copied out of the notebooks, so the gallery
always shows what the lectures show.  Output: docs/images/gallery_NN.png, and the
figure of Lecture 3 that compares the monotone fit with the equal-area rule, docs/images/checks_03.png.

Run from the repository root:  python scripts/make_readme_gallery.py
"""
import base64
import glob
import json
import os

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# lecture number -> which figure of the notebook to use (0 = first)
choice = {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0, 10: 1, 11: 1, 12: 1, 13: 1, 14: 0, 15: 1, 16: 1, 17: 1}


def figures(lecture):
    path = glob.glob(os.path.join(root, "notebooks", "*", f"{lecture:02d}_*.ipynb"))[0]
    notebook = json.load(open(path, encoding="utf-8"))
    return [output["data"]["image/png"] for cell in notebook["cells"] for output in cell.get("outputs", [])
            if "image/png" in output.get("data", {})]


for lecture, index in choice.items():
    images = figures(lecture)
    out = os.path.join(root, "docs", "images", f"gallery_{lecture:02d}.png")
    with open(out, "wb") as handle:
        handle.write(base64.b64decode(images[index]))
    print(f"lecture {lecture:02d}: figure {index + 1} of {len(images)} -> {os.path.normpath(out)}")

with open(os.path.join(root, "docs", "images", "checks_03.png"), "wb") as handle:
    handle.write(base64.b64decode(figures(3)[1]))
print("lecture 03: second figure -> docs/images/checks_03.png")
