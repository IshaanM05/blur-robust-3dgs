"""Turn docs/report.md into docs/report.pdf (markdown -> HTML -> LibreOffice).

The few LaTeX expressions in the report are rewritten as readable text with sub- and superscripts, and the figures the
report links to are placed in an appendix, because a PDF cannot follow relative links.
Usage: python scripts/build_report_pdf.py
"""

import re
import subprocess
import tempfile
from pathlib import Path

import markdown
from PIL import Image

DOCS = Path("docs").resolve()
FIGURE_LINK = re.compile(r"\[([^\]]+)\]\((figures/[^)]+\.png)\)")

CSS = """
body { font-family: 'Liberation Serif', 'Times New Roman', serif; font-size: 11pt; line-height: 1.35; }
h1 { font-size: 18pt; } h2 { font-size: 14pt; margin-top: 14pt; } h3 { font-size: 12pt; }
table { border-collapse: collapse; margin: 6pt 0; } td, th { padding: 3pt 6pt; font-size: 10pt; }
code { font-family: 'Liberation Mono', monospace; font-size: 9.5pt; }
p.eq { text-align: center; font-style: italic; margin: 8pt 0; }
"""

MACROS = [
    (r"\mathcal{L}", "L"),
    (r"\lVert", "‖"),
    (r"\rVert", "‖"),
    (r"\mathrm{SSIM}", "SSIM"),
    (r"\mathrm{NSR}", "NSR"),
    (r"\hat X", "X̂"),
    (r"\bar H", "H̄"),
    (r"\,", " "),
    ("*", "∗"),
]


def math_to_text(expr):
    for old, new in MACROS:
        expr = expr.replace(old, new)
    expr = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", expr)
    expr = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", expr)
    expr = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", expr)
    expr = re.sub(r"\^([A-Za-z0-9])", r"<sup>\1</sup>", expr)
    return expr


def convert_math(text):
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: f'\n\n<p class="eq">{math_to_text(m.group(1))}</p>\n\n', text, flags=re.S)
    return re.sub(r"\$([^$\n]+)\$", lambda m: f"<i>{math_to_text(m.group(1))}</i>", text)


def main():
    text = (DOCS / "report.md").read_text()
    figures = []
    for _, path in FIGURE_LINK.findall(text):
        if path not in figures:
            figures.append(path)
    text = FIGURE_LINK.sub(lambda m: f"{m.group(1)} (Figure {figures.index(m.group(2)) + 1})", text)
    html = markdown.markdown(convert_math(text), extensions=["tables", "fenced_code"])
    html = html.replace("<table>", '<table border="1" cellspacing="0" cellpadding="3">')
    appendix = ["<h2>Appendix: figures</h2>"]
    for i, path in enumerate(figures, 1):
        appendix.append(f"<p><b>Figure {i}.</b> <code>{Path(path).name}</code></p>")
        with Image.open(DOCS / path) as im:
            width = 620
            height = round(width * im.height / im.width)
        appendix.append(f'<p><img src="{DOCS / path}" width="{width}" height="{height}"></p>')
    page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}{''.join(appendix)}</body></html>"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.html"
        src.write_text(page)
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf:writer_web_pdf_Export", "--outdir", tmp, str(src)],
            check=True,
            capture_output=True,
        )
        (DOCS / "report.pdf").write_bytes((Path(tmp) / "report.pdf").read_bytes())
    size_kb = (DOCS / "report.pdf").stat().st_size // 1024
    print(f"saved docs/report.pdf ({size_kb} KB, {len(figures)} figures in the appendix)")


if __name__ == "__main__":
    main()
