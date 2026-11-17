"""Assemble a flat arXiv upload folder: one .tex with inputs inlined, figures alongside.

    python scripts/make_arxiv_bundle.py   # writes paper/arxiv/
"""

import re
import shutil
from pathlib import Path

paper = Path(__file__).resolve().parent.parent / "paper"
out = paper / "arxiv"
shutil.rmtree(out, ignore_errors=True)
out.mkdir()

tex = (paper / "scenecast.tex").read_text()
tex = re.sub(r"\\input\{(.+?)\}", lambda m: (paper / m[1]).read_text().strip(), tex)
tex = tex.replace("\\graphicspath{{../assets/}}\n", "")
tex = "\n".join(line for line in tex.splitlines() if not line.lstrip().startswith("%")) + "\n"
(out / "scenecast.tex").write_text(tex)
shutil.copy(paper / "references.bib", out)
for name in re.findall(r"\\includegraphics\[.*?\]\{(.+?)\}", tex):
    shutil.copy(paper.parent / "assets" / name, out)
print(sorted(p.name for p in out.iterdir()))
