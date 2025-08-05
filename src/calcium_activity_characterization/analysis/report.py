"""
Usage:
    >>> from calcium_activity_characterization.analysis.report import export_current_notebook_to_pdf
    >>> export_current_notebook_to_pdf("my_notebook_name.ipynb")

Exports the specified Jupyter notebook to a PDF (markdown and outputs only) into:
notebooks/export/<notebook_name>.pdf

Note:
- This uses nbconvert with --no-input to hide code cells.
- Assumes the repo structure has a 'notebooks/export' folder.
"""

import subprocess
from pathlib import Path
from calcium_activity_characterization.logger import logger




def export_current_notebook_to_pdf(notebook_filename: str) -> None:
    """
    Export the given notebook name as a PDF (no code, only markdown and outputs),
    and save it under notebooks/export/<notebook_name>.pdf.

    Args:
        notebook_filename (str): Filename of the notebook (e.g., "analysis.ipynb")
    """
    try:
        notebook_path = Path("notebooks") / notebook_filename
        nb_name = notebook_path.stem
        export_dir = Path("notebooks") / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        pdf_output_path = export_dir / f"{nb_name}.pdf"

        logger.info(f"📓 Exporting notebook: {notebook_path}")
        logger.info(f"📄 Saving PDF to: {pdf_output_path}")

        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "pdf",
            "--no-input",
            "--output", pdf_output_path.stem,
            str(notebook_path)
        ], check=True)

        print(f"✅ PDF successfully exported to: {pdf_output_path.resolve()}")

    except Exception as e:
        print(f"❌ Failed to export notebook: {e}")
        logger.error(f"Notebook export error: {e}")