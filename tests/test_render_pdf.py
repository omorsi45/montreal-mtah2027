"""The stdlib Markdown-to-HTML converter used for the technical document (synthetic text only)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "render_pdf", Path(__file__).resolve().parents[1] / "scripts" / "render_pdf.py"
)
render_pdf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render_pdf)


def test_inline_styles_and_escaping():
    out = render_pdf.inline("a **bold** and *it* with `x < y` and <https://e.org/a>")
    assert "<strong>bold</strong>" in out
    assert "<em>it</em>" in out
    assert "<code>x &lt; y</code>" in out
    assert '<a href="https://e.org/a">https://e.org/a</a>' in out


def test_blocks():
    md = "\n".join([
        "# Title", "", "para one", "continues", "", "## Sec", "",
        "- a", "- b `c`", "", "1. one", "2. two", "",
        "| h1 | h2 |", "|---|---|", "| 1 | **2** |", "",
        "```", "x < 1", "```",
    ])
    out = render_pdf.md_to_html(md)
    assert "<h1>Title</h1>" in out and "<h2>Sec</h2>" in out
    assert "<p>para one continues</p>" in out
    assert "<ul><li>a</li><li>b <code>c</code></li></ul>" in out
    assert "<ol><li>one</li><li>two</li></ol>" in out
    assert "<th>h1</th>" in out and "<td><strong>2</strong></td>" in out
    assert "<pre><code>x &lt; 1</code></pre>" in out
