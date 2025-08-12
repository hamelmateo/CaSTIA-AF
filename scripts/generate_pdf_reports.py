import os
import argparse
from pathlib import Path
import papermill as pm
from nbconvert import PDFExporter

def execute_and_export(notebook_path: Path, output_path: Path, parameters: dict):
    """
    Execute a notebook with parameters using papermill and export to PDF.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    executed_notebook = output_path / "executed.ipynb"
    final_pdf = output_path / "report.pdf"

    # Run notebook with parameters
    pm.execute_notebook(
        str(notebook_path),
        str(executed_notebook),
        parameters=parameters
    )

    # Export to PDF
    pdf_exporter = PDFExporter()
    pdf_exporter.exclude_input = True
    body, _ = pdf_exporter.from_filename(str(executed_notebook))

    with open(final_pdf, "wb") as f:
        f.write(body)


    print(f"✅ PDF saved: {final_pdf}")

def find_all_is_folders(root_dir: Path) -> list[tuple[str, Path]]:
    """
    Recursively find all IS* folders under Output folders.
    
    Returns:
        List of tuples: (date_label, is_folder_path)
    """
    is_folders = []
    for date_folder in root_dir.iterdir():
        if not date_folder.is_dir():
            continue

        date_str = date_folder.name

        for subdir in date_folder.rglob("Output"):
            if not subdir.is_dir():
                continue

            for is_folder in subdir.iterdir():
                if is_folder.is_dir() and is_folder.name.startswith("IS"):
                    label = f"{date_str}_{is_folder.name}"
                    is_folders.append((label, is_folder.resolve()))

    return is_folders

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input-root", type=str, required=True)
    #parser.add_argument("--notebook", type=str, required=True)
    #parser.add_argument("--output-dir", type=str, required=True)
    #args = parser.parse_args()

    args = argparse.Namespace(
        input_root="D:/Mateo",
        notebook="notebooks/image_sequence_analysis.ipynb",
        output_dir="notebooks/export"
        )

    for label, dataset_path in find_all_is_folders(Path(args.input_root)):
        output_dir = Path(args.output_dir) / label
        report_file = output_dir / "report.pdf"

        if report_file.exists():
            print(f"⏭️ Skipping {label} — report already exists.")
            continue
            
        print(f"▶️ Running for: {label}")

        parameters = {
            "control_paths": {label: str(dataset_path)}
        }

        output_dir = Path(args.output_dir) / label
        execute_and_export(Path(args.notebook), output_dir, parameters)


if __name__ == "__main__":
    main()
