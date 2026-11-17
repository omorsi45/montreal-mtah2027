"""Render docs/Montreal.md to docs/Montreal.pdf (stdlib only + headless Microsoft Edge).

1. A minimal Markdown-to-HTML converter (headings, paragraphs, bullet and numbered lists, tables,
   fenced code blocks, **bold**, *italics*, `inline code`, links) with simple print CSS writes
   docs/Montreal.html.
2. Headless Edge prints that file to docs/Montreal.pdf.

Usage: python scripts/render_pdf.py [input.md] [output.pdf]
The HTML and the PDF are build products and are not committed.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
MIN_PDF_BYTES = 20 * 1024

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #111; }
h1 { font-size: 19pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 14pt 0 4pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; break-after: avoid; }
p { margin: 4pt 0; }
ul, ol { margin: 4pt 0 4pt 18pt; padding: 0; }
li { margin: 1.5pt 0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; background: #f1f1f1;
       padding: 0 2pt; border-radius: 2pt; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 6pt 8pt; margin: 6pt 0;
      white-space: pre-wrap; word-break: break-all; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; }
table { border-collapse: collapse; margin: 6pt 0; width: 100%; break-inside: avoid; }
th, td { border: 1px solid #999; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #e8e8e8; }
a { color: #0b4fa8; text-decoration: none; }
"""


# --------------------------------------------------------------------------------------
# Markdown -> HTML
# --------------------------------------------------------------------------------------


def inline(text: str) -> str:
    """Convert inline Markdown (code, bold, italics, links, <autolinks>) of one text run."""
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        codes.append("<code>" + html.escape(m.group(1), quote=False) + "</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"&lt;(https?://[^\s&]+)&gt;", r'<a href="\1">\1</a>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()))


def md_to_html(md: str, title: str = "Montreal") -> str:
    """Convert the Markdown subset used by docs/Montreal.md into a complete HTML page."""
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []
    i = 0

    def flush_para() -> None:
        if para:
            out.append("<p>" + inline(" ".join(s.strip() for s in para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):  # fenced code block
            flush_para()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(block), quote=False) + "</code></pre>")
            continue

        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:  # heading
            flush_para()
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and _is_separator(lines[i + 1]):
            flush_para()  # table
            head = _cells(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            t = ["<table>", "<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead>",
                 "<tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t += ["</tbody>", "</table>"]
            out.append("\n".join(t))
            continue

        m = re.match(r"([-*]|\d+\.)\s+(.*)", stripped)
        if m:  # list (one level; continuation lines are indented)
            flush_para()
            ordered = m.group(1)[0].isdigit()
            items: list[str] = []
            while i < len(lines):
                cur = lines[i]
                mm = re.match(r"\s*([-*]|\d+\.)\s+(.*)", cur)
                if mm and (mm.group(1)[0].isdigit() == ordered):
                    items.append(mm.group(2))
                elif cur.strip() and cur.startswith((" ", "\t")) and items:
                    items[-1] += " " + cur.strip()
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(s)}</li>" for s in items) + f"</{tag}>")
            continue

        if not stripped:
            flush_para()
        else:
            para.append(stripped)
        i += 1
    flush_para()

    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>\n<body>\n"
        + "\n".join(out)
        + "\n</body></html>\n"
    )


# --------------------------------------------------------------------------------------
# HTML -> PDF with headless Edge
# --------------------------------------------------------------------------------------


def find_edge() -> str:
    env = os.environ.get("EDGE_PATH")
    for cand in ([env] if env else []) + EDGE_CANDIDATES:
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError("Microsoft Edge not found; set EDGE_PATH to msedge.exe")


def render(md_path: Path, pdf_path: Path) -> Path:
    html_path = md_path.with_suffix(".html")
    html_path.write_text(md_to_html(md_path.read_text(encoding="utf-8")), encoding="utf-8")
    if pdf_path.exists():
        pdf_path.unlink()
    cmd = [
        find_edge(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}", html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    if not pdf_path.exists() or pdf_path.stat().st_size <= MIN_PDF_BYTES:
        raise RuntimeError(f"PDF missing or smaller than {MIN_PDF_BYTES} bytes: {pdf_path}")
    return pdf_path


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    md_path = Path(args[0]) if args else REPO / "docs" / "Montreal.md"
    pdf_path = Path(args[1]) if len(args) > 1 else md_path.with_suffix(".pdf")
    render(md_path, pdf_path)
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
