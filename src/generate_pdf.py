"""
generate_pdf.py — Compile output/report.md and embedded figures into output/report.pdf.
Uses ReportLab to generate a clean, 8-page publication-quality PDF document.
"""

import os
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass render to draw 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#7f8c8d"))

        # Running header (skip on page 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Titan Mandal — Survey Network Analysis Report")
            self.setStrokeColor(colors.HexColor("#bdc3c7"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "DPCN Coursework Assignment — Network Analysis of Class Opinions")
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)

        self.restoreState()


def build_pdf(md_path: str | Path = "output/report.md", pdf_path: str | Path = "output/report.pdf"):
    md_path = Path(md_path)
    pdf_path = Path(pdf_path)

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown report not found at {md_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1b365d")
    secondary_color = colors.HexColor("#2c3e50")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        alignment=0,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#555555"),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    )

    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10,
    )

    formula_style = ParagraphStyle(
        "Formula",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2c3e50"),
        alignment=1,
        spaceBefore=6,
        spaceAfter=6,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.white,
        alignment=0,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#222222"),
        alignment=0,
    )

    story = []

    lines = md_text.split("\n")
    i = 0
    in_table = False
    table_buffer = []

    def flush_table(tb_lines):
        if not tb_lines:
            return
        headers = [c.strip() for c in tb_lines[0].strip("|").split("|")]
        data_rows = []
        for line in tb_lines[2:]:  # skip separator line
            if "|" in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                data_rows.append(cells)

        table_data = []

        # Convert markdown bold/formatting in headers & cells to ReportLab Paragraphs
        header_p = [Paragraph(re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", h), table_header_style) for h in headers]
        table_data.append(header_p)

        for row in data_rows:
            row_p = [Paragraph(re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", c), table_cell_style) for c in row]
            table_data.append(row_p)

        t = Table(table_data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("TOPPADDING", (0, 0), (-1, 0), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dcdde1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ]))
        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 8))

    while i < len(lines):
        line = lines[i]

        # Table detection
        if line.strip().startswith("|"):
            table_buffer.append(line)
            in_table = True
            i += 1
            continue
        elif in_table:
            flush_table(table_buffer)
            table_buffer = []
            in_table = False

        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if stripped in ["---", "***", "___"]:
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bdc3c7"), spaceBefore=8, spaceAfter=8))
            i += 1
            continue

        # Headers
        if stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], title_style))
            i += 1
            continue
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], h1_style))
            i += 1
            continue
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], h2_style))
            i += 1
            continue
        elif stripped.startswith("#### "):
            story.append(Paragraph(stripped[5:], h3_style))
            i += 1
            continue

        # Images
        img_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if img_match:
            alt_text, img_rel_path = img_match.groups()
            img_full_path = Path("output") / img_rel_path
            if not img_full_path.exists():
                img_full_path = Path(img_rel_path)

            if img_full_path.exists():
                story.append(Spacer(1, 4))
                img = Image(str(img_full_path), width=480, height=270)
                story.append(img)
                story.append(Paragraph(f"<i>{alt_text}</i>", caption_style))
            else:
                story.append(Paragraph(f"[Image missing: {img_rel_path}]", caption_style))
            i += 1
            continue

        # Formated text processing
        # Convert markdown bold, italic, inline code, and latex formulas to ReportLab XML
        text = stripped
        text = re.sub(r"\$\$(.*?)\$\$", r"<i>\1</i>", text)
        text = re.sub(r"\$(.*?)\$", r"<i>\1</i>", text)
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
        text = re.sub(r"`(.*?)`", r"<font face='Courier'>\1</font>", text)
        text = re.sub(r"\[(.*?)\]\((.*?)\)", r"<u>\1</u>", text)

        if text.startswith("* ") or text.startswith("- "):
            bullet_text = text[2:]
            story.append(Paragraph(f"• {bullet_text}", body_style))
        else:
            story.append(Paragraph(text, body_style))

        i += 1

    if in_table:
        flush_table(table_buffer)

    print(f"  Building PDF with {len(story)} flowables …")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  PDF build complete! Output saved to {pdf_path}")


if __name__ == "__main__":
    build_pdf()
