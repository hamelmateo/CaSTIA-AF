# scripts/generate_merged_datasets.py

from calcium_activity_characterization.config.datasets_metadata import get_all_image_sequences
from pathlib import Path
import pandas as pd
from calcium_activity_characterization.logger import logger


logging.basicConfig(level=logging.INFO)


def compile_dataset_metadata(
    image_sequences: list[dict],
    output_folder: Path
) -> None:
    """
    Compile the dataset-level metadata from all image sequences into a single CSV.

    Args:
        image_sequences (list[dict]): list of dataset metadata dictionaries.
        output_folder (Path): Path where the summary CSV will be saved.
    """
    try:
        # Add dataset label to each entry
        for entry in image_sequences:
            entry["dataset"] = f"{entry['date']}__{entry['image_sequence']}"

        df = pd.DataFrame(image_sequences)
        output_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_folder / "experiments.csv", index=False)
        logger.info("Saved experiments metadata to experiments.csv")
    except Exception as e:
        logger.error(f"Failed to save dataset metadata: {e}")

def merge_all_datasets(
    image_sequences: list[dict],
    output_folder: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge all peaks, cells, events, and communication datasets across image sequences,
    enrich with experiment metadata, and save them to CSV files.

    Args:
        image_sequences (list[dict]): list of dictionaries with keys:
            - path (str): Path to the folder containing datasets/peaks.csv etc.
            - date (str): Date of acquisition in YYYY-MM-DD format
            - image_sequence (str): ID of the replicate (e.g., IS2)
            - experiment_type (str): 'stimulated' or 'spontaneous'
            - condition (str): e.g., 'Control First Run' or 'ACH'
            - concentration (str or None): e.g., '10uM', 'None'
            - time (str or None): Time condition (e.g. '-300s', '+1800s', '2 days')
            - confluency (int): Cell seeding density

        output_folder (Path): Path to folder where merged CSVs will be saved.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Merged DataFrames: (peaks_all, cells_all, events_all, communication_all)
    """
    peaks_dfs = []
    cells_dfs = []
    events_dfs = []
    comm_dfs = []

    for entry in image_sequences:
        try:
            base_path = Path(entry["path"])
            label = f"{entry['date']}__{entry['image_sequence']}"

            for filetype, collector in zip(
                ["peaks", "cells", "events", "communications"],
                [peaks_dfs, cells_dfs, events_dfs, comm_dfs]
            ):
                file_path = base_path / "datasets" / f"{filetype}.csv"
                if not file_path.exists():
                    logger.warning(f"Missing file: {file_path}")
                    continue

                df = pd.read_csv(file_path)
                df["dataset"] = label
                df["date"] = entry["date"]
                df["image_sequence"] = entry["image_sequence"]
                df["experiment_type"] = entry["experiment_type"]
                df["condition"] = entry["condition"]
                df["concentration"] = entry["concentration"]
                df["time"] = entry["time"]
                df["confluency"] = entry["confluency"]
                collector.append(df)

        except Exception as e:
            logger.error(f"Failed to process dataset at {entry['path']}: {e}")

    # Concatenate and save
    output_folder.mkdir(parents=True, exist_ok=True)

    peaks_all = pd.concat(peaks_dfs, ignore_index=True)
    cells_all = pd.concat(cells_dfs, ignore_index=True)
    events_all = pd.concat(events_dfs, ignore_index=True)
    communications_all = pd.concat(comm_dfs, ignore_index=True)

    peaks_all.to_csv(output_folder / "merged_peaks.csv", index=False)
    cells_all.to_csv(output_folder / "merged_cells.csv", index=False)
    events_all.to_csv(output_folder / "merged_events.csv", index=False)
    communications_all.to_csv(output_folder / "merged_communications.csv", index=False)

    logger.info(f"Saved merged datasets to: {output_folder}")

    return peaks_all, cells_all, events_all, communications_all

def main():
    output_dir = Path("D:/Mateo/Results")
    image_sequences = get_all_image_sequences()
    compile_dataset_metadata(image_sequences, output_dir)
    merge_all_datasets(image_sequences, output_dir)

if __name__ == "__main__":
    main()
