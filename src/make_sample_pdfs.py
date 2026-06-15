from __future__ import annotations

from pathlib import Path

from .utils import PDF_DIR


DOCS = {
    "nda_vendor_x.pdf": [
        "Vendor X NDA",
        [
            "Confidential Information means all non-public business, technical, and financial information disclosed by either party.",
            "The term of this agreement is two years from the effective date.",
            "The notice period for termination without cause is thirty (30) days written notice.",
            "No party may assign this agreement without prior written consent.",
            "This agreement is governed by the laws of the State of Delaware.",
        ],
    ],
    "msa_vendor_y.pdf": [
        "Vendor Y MSA",
        [
            "Services are provided on a subscription basis and invoiced monthly in advance.",
            "The limitation of liability shall not exceed INR 1 crore in aggregate.",
            "The vendor will maintain uptime of 99.5 percent excluding scheduled maintenance.",
            "Customer may terminate for material breach with a fifteen (15) day cure period.",
            "Disputes will be resolved by arbitration in Mumbai.",
        ],
    ],
    "policy_retention.pdf": [
        "Records Retention Policy",
        [
            "Employee records must be retained for seven years after employment ends.",
            "Finance records must be retained for eight years.",
            "Legal hold notices override ordinary deletion schedules.",
            "Access to archived files requires approval from the compliance team.",
            "Policy exceptions must be approved in writing by the General Counsel.",
        ],
    ],
}


def make_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, title)
    y -= 32
    c.setFont("Helvetica", 11)
    for i, para in enumerate(paragraphs, start=1):
        text = c.beginText(72, y)
        for line in wrap_text(para, 88):
            text.textLine(line)
            y -= 14
            if y < 96:
                c.drawText(text)
                c.showPage()
                y = height - 72
                c.setFont("Helvetica", 11)
                text = c.beginText(72, y)
        c.drawText(text)
        y -= 10
        if y < 96:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 11)
    c.save()


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines, line = [], []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return lines


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, paragraphs) in DOCS.items():
        make_pdf(PDF_DIR / filename, title, paragraphs)


if __name__ == "__main__":
    main()
