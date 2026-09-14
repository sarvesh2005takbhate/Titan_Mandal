"""
generate_pdf.py — Compile output/report.md and its figures into output/report.pdf (ReportLab).

Supports the small Markdown subset the report uses: headings, bullets, bold/italic,
inline code, links, tables, images (alt text becomes the caption) and a
`<!-- pagebreak -->` marker.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

PRIMARY = colors.HexColor("#1b365d")
SECONDARY = colors.HexColor("#2c3e50")
MUTED = colors.HexColor("#7f8c8d")
RULE = colors.HexColor("#bdc3c7")
MARGIN = 54
CONTENT_WIDTH = letter[0] - 2 * MARGIN


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that draws 'Page X of Y' footers and a running header."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(len(self._saved_page_states))
            super().showPage()
        super().save()

    def _draw_chrome(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8.5)
        self.setFillColor(MUTED)
        self.setStrokeColor(RULE)
        self.setLineWidth(0.5)
        if self._pageNumber > 1:
            self.drawString(MARGIN, 750, "Titan Mandal — Opinion Network Formation")
            self.line(MARGIN, 744, letter[0] - MARGIN, 744)
        self.line(MARGIN, 48, letter[0] - MARGIN, 48)
        self.drawString(MARGIN, 36, "DPCN Assignment 1")
        self.drawRightString(letter[0] - MARGIN, 36, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def _styles():
    base = getSampleStyleSheet()

    def style(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    ink = colors.HexColor("#222222")
    return {
        "title": style("DocTitle", "Title", fontName="Helvetica-Bold", fontSize=20, leading=24,
                       textColor=PRIMARY, alignment=0, spaceAfter=6),
        "h1": style("H1", "Heading1", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=PRIMARY,
                    spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "h2": style("H2", "Heading2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=SECONDARY,
                    spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "body": style("Body", fontName="Helvetica", fontSize=9.3, leading=12.4, textColor=ink, spaceAfter=4),
        "bullet": style("Bullet", fontName="Helvetica", fontSize=9.3, leading=12.4, leftIndent=12,
                        bulletIndent=2, textColor=ink, spaceAfter=2),
        "caption": style("Caption", fontName="Helvetica-Oblique", fontSize=8.3, leading=11,
                         textColor=colors.HexColor("#555555"), alignment=1, spaceBefore=2, spaceAfter=7),
        "th": style("TH", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white),
        "td": style("TD", fontName="Helvetica", fontSize=8, leading=10, textColor=ink),
    }


def _inline(text: str) -> str:
    """Escape XML, then convert Markdown emphasis, code and links to ReportLab markup."""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"<link href='\2' color='#1b365d'><u>\1</u></link>", text)
    return text


def _column_widths(header, body, min_width=88):
    """Widths proportional to content length. Columns that would shrink below `min_width`
    (or below their natural width, if smaller) are pinned first; the rest share the remainder."""
    lengths = [max(len(r[i]) if i < len(r) else 0 for r in [header] + body) for i in range(len(header))]
    want = [min(max(n, 5), 250) * 5.0 + 12 for n in lengths]
    if sum(want) <= CONTENT_WIDTH:
        return want
    widths, free, budget = list(want), list(range(len(want))), CONTENT_WIDTH
    while True:
        scale = budget / sum(want[i] for i in free)
        pinned = [i for i in free if want[i] * scale < min(want[i], min_width)]
        if not pinned:
            break
        for i in pinned:
            widths[i] = min(want[i], min_width)
            budget -= widths[i]
            free.remove(i)
    for i in free:
        widths[i] = want[i] * scale
    return widths


def _table(lines, st):
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in lines]
    header, body = rows[0], rows[2:]
    widths = _column_widths(header, body)

    data = [[Paragraph(_inline(h), st["th"]) for h in header]]
    data += [[Paragraph(_inline(c), st["td"]) for c in r] for r in body]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dcdde1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f7f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [Spacer(1, 3), t, Spacer(1, 8)]


def _image(path: Path, caption: str, st, max_height=270):
    """Image scaled to fit the text width and max height, preserving aspect ratio."""
    w, h = ImageReader(str(path)).getSize()
    scale = min(CONTENT_WIDTH / w, max_height / h)
    return KeepTogether([Spacer(1, 4), Image(str(path), width=w * scale, height=h * scale),
                         Paragraph(_inline(caption), st["caption"])])


def build_pdf(md_path: str | Path = "output/report.md", pdf_path: str | Path = "output/report.pdf"):
    md_path, pdf_path = Path(md_path), Path(pdf_path)
    st = _styles()
    story, table_buf = [], []

    for line in md_path.read_text(encoding="utf-8").split("\n") + [""]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_buf.append(stripped)
            continue
        if table_buf:
            story += _table(table_buf, st)
            table_buf = []
        if not stripped:
            continue
        if stripped == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif stripped in ("---", "***"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=4, spaceAfter=6))
        elif stripped.startswith("### "):
            story.append(Paragraph(_inline(stripped[4:]), st["h2"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(_inline(stripped[3:]), st["h1"]))
        elif stripped.startswith("# "):
            story.append(Paragraph(_inline(stripped[2:]), st["title"]))
        elif m := re.match(r"!\[(.*?)\]\((.*?)\)", stripped):
            caption, rel = m.groups()
            path = md_path.parent / rel
            story.append(_image(path, caption, st) if path.exists()
                         else Paragraph(f"[Image missing: {escape(rel)}]", st["caption"]))
        elif stripped.startswith(("- ", "* ")):
            story.append(Paragraph(_inline(stripped[2:]), st["bullet"], bulletText="•"))
        else:
            story.append(Paragraph(_inline(stripped), st["body"]))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN + 6, bottomMargin=MARGIN + 6,
                            title="Opinion Network Formation — Titan Mandal")
    doc.build(story, canvasmaker=NumberedCanvas)


if __name__ == "__main__":
    build_pdf()
