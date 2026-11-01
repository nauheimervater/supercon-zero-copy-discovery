"""
Render paper/SuperCon_Zero_Copy_Discovery.md to PDF (simple Markdown subset) so that
the PDF can never diverge from the Markdown source.
"""

import os
import re
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "SuperCon_Zero_Copy_Discovery.md")
OUT = os.path.join(HERE, "SuperCon_Zero_Copy_Discovery.pdf")


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    text = re.sub(r"\$([^$]+)\$", lambda m: m.group(1).replace("\\approx", "≈").replace("\\min", "min")
                  .replace("_{", "").replace("}", "").replace("_", "").replace("\\", ""), text)
    return text


def main() -> None:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=10.5, leading=14, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold", fontSize=15, leading=19)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12.5, leading=16)
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Times-Bold", fontSize=11, leading=14)
    story, para = [], []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), body))
            para.clear()

    for line in open(SRC, encoding="utf-8").read().splitlines():
        s = line.rstrip()
        if not s.strip() or s.strip() == "---":
            flush()
            continue
        if s.startswith("### "):
            flush(); story.append(Paragraph(inline(s[4:]), h3))
        elif s.startswith("## "):
            flush(); story.append(Paragraph(inline(s[3:]), h2))
        elif s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), h1)); story.append(Spacer(1, 4))
        elif s.startswith("|"):
            flush()
            if re.match(r"^\|[\s\-:|]+\|$", s):
                continue
            cells = [inline(c.strip()) for c in s.strip("|").split("|")]
            story.append(Paragraph(" &nbsp;|&nbsp; ".join(cells), body))
        elif re.match(r"^(\d+\.|-)\s", s):
            flush(); story.append(Paragraph(inline(s), body))
        else:
            para.append(s.strip())
    flush()

    SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                      topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                      title="Toward Zero-Copy ML/DFT Materials Screening on Unified-Memory APUs (WIP)",
                      author="Thomas Nauheimer").build(story)
    print(f"Generated {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
