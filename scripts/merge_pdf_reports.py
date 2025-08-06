# Usage Example:
# python scripts/merge_control_reports.py
# This script filters experiments.csv for condition == 'control - 1st run',
# locates corresponding PDF reports in the export directory, and merges them
# into a single output PDF.

from pathlib import Path
import pandas as pd
from PyPDF2 import PdfMerger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_control_first_run_pdfs(
    csv_path: Path, export_dir: Path, output_pdf: Path
) -> None:
    """
    Merges PDFs for experiments with condition 'control - 1st run' listed in CSV.

    Args:
        csv_path (Path): Path to experiments.csv
        export_dir (Path): Directory containing individual report folders
        output_pdf (Path): Output path for merged PDF
    """
    df = pd.read_csv(csv_path)
    filtered_df = df[df["condition"] == "control - 1st run"]

    merger = PdfMerger()
    added_count = 0

    for _, row in filtered_df.iterrows():
        label = f"{str(row['date'])}_{row['image_sequence']}"
        if not label:
            continue

        pdf_path = export_dir / label / "report.pdf"
        if pdf_path.exists():
            logger.info(f"Adding: {pdf_path}")
            merger.append(str(pdf_path))
            added_count += 1
        else:
            logger.warning(f"Missing PDF: {pdf_path}")

    if added_count:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        merger.write(str(output_pdf))
        merger.close()
        logger.info(f"\n✅ Merged {added_count} PDFs into: {output_pdf}")
    else:
        logger.error("❌ No valid PDFs found. Nothing was merged.")

if __name__ == "__main__":
    merge_control_first_run_pdfs(
        csv_path=Path("D:\Mateo\Results\experiments.csv"),
        export_dir=Path("notebooks/export"),
        output_pdf=Path("notebooks/merged_controls_first_run.pdf")
    )