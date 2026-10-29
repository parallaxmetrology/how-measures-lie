"""Build the essay page from page_src.html and the computed figures.

    .venv/bin/python demos/make_figures.py      # figures + numbers
    .venv/bin/python build_page.py              # writes how_measures_lie.html here
    .venv/bin/python build_page.py --site       # also copies it into the site folder

Placeholders in page_src.html:
    {{fig:NAME}}              figures/NAME.png, embedded as a data URI
    {{n:KEY.FIELD:FORMAT}}    a value from figures/numbers.json, Python format spec optional
Any placeholder that cannot be filled stops the build.
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = Path.home() / "Documents" / "parallaxmetrology" / "how_measures_lie.html"


def build() -> str:
    src = (HERE / "page_src.html").read_text()
    numbers = json.loads((HERE / "figures" / "numbers.json").read_text())

    def fig(m):
        path = HERE / "figures" / f"{m.group(1)}.png"
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    def num(m):
        key, field = m.group(1).split(".")
        value = numbers[key][field]
        spec = m.group(2) or ""
        out = format(value, spec)
        return out.replace("-", "−") if spec and value < 0 else out

    out = re.sub(r"\{\{fig:([a-z_]+)\}\}", fig, src)
    out = re.sub(r"\{\{n:([a-z0-9_]+\.[a-z0-9_]+)(?::([^}]+))?\}\}", num, out)
    left = re.findall(r"\{\{[^}]*\}\}", out)
    if left:
        sys.exit(f"unfilled placeholders: {left}")
    return out


if __name__ == "__main__":
    page = build()
    local = HERE / "how_measures_lie.html"
    local.write_text(page)
    print(f"wrote {local.name} ({len(page) / 1e6:.2f} MB)")
    if "--site" in sys.argv:
        SITE.write_text(page)
        print(f"wrote {SITE}")
