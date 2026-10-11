# Documentation

| Document | What it is |
|---|---|
| **[RailSight_Report.pdf](RailSight_Report.pdf)** | The project report (14 pages): problem and statistics, system design, data, models, results, real-video evaluation, trial and error, lessons, tools and acknowledgements |
| [report_notes.md](report_notes.md) | Sourced statistics (NCRB, Ministry of Railways, Parliament replies, UIC/ORR), related systems and key findings used for the report |
| [report/](report) | Source of the PDF: `report.html`, figure script and build script |
| [assets/](assets) | README demo GIF and figures |

## Rebuilding the report

```bash
uv run python docs/report/make_figures.py                                  # figures from reports/ and outputs/
uv run --with pypdf --with reportlab python docs/report/build_report.py    # -> docs/RailSight_Report.pdf
```

The figures read the evaluation results in `reports/` and the rendered demo videos in `outputs/final/`
(`railsight video "data/videos/*.mp4" --out outputs/final`). Rendering uses headless Microsoft Edge or
Chrome.
