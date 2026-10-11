"""Build docs/RailSight_Report.pdf from report.html.

    uv run python docs/report/make_figures.py                                   # refresh the figures
    uv run --with pypdf --with reportlab python docs/report/build_report.py     # render the PDF

Renders the HTML with headless Microsoft Edge (or Chrome), then stamps a footer with page numbers on
every page except the cover and sets the PDF metadata.
"""

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "RailSight_Report.pdf"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            shutil.which("chromium") or "", shutil.which("google-chrome") or ""]


def render(html: Path, pdf: Path):
    exe = next((b for b in BROWSERS if b and Path(b).exists()), None)
    if exe is None:
        raise SystemExit("needs Microsoft Edge or Chrome for rendering")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", "--virtual-time-budget=15000",
                        f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True, timeout=180)


def footer(width, height, page, total):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))
    c.setFont("Helvetica", 7.5)
    c.setFillColor(HexColor("#8A93A6"))
    c.drawString(45, 26, "RailSight  ·  Vision-based railway hazard detection and collision risk")
    c.drawRightString(width - 45, 26, f"{page} / {total}")
    c.setStrokeColor(HexColor("#E3E7EE"))
    c.setLineWidth(0.5)
    c.line(45, 36, width - 45, 36)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.pdf"
        render(HERE / "report.html", raw)
        reader = PdfReader(str(raw))
        writer = PdfWriter()
        total = len(reader.pages)
        for i, page in enumerate(reader.pages):
            if i > 0:  # no footer on the cover
                w, h = float(page.mediabox.width), float(page.mediabox.height)
                page.merge_page(footer(w, h, i + 1, total))
            writer.add_page(page)
        writer.add_metadata({"/Title": "RailSight — Project Report",
                             "/Author": "Arnav Keshav Sinha, Faizaan Naiman Naveed Nasir",
                             "/Subject": "Railway hazard detection and collision-risk estimation",
                             "/Keywords": "railway safety, computer vision, segmentation, object detection"})
        with open(OUT, "wb") as f:
            writer.write(f)
    print(f"{total} pages -> {OUT} ({OUT.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
