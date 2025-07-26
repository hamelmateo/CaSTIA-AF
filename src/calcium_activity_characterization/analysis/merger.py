# merger.py
# Usage Example:
# >>> from calcium_activity_characterization.analysis.merger import merge_all_datasets
# >>> groups = merge_all_datasets(image_sequences, output_folder=Path("D:/Mateo/AllMerged/"))
# >>> peaks_all, cells_all, events_all = groups

from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def merge_all_datasets(
    image_sequences: List[Dict],
    output_folder: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge all peaks, cells, and events datasets across image sequences, 
    enrich with experiment metadata, and save them to CSV files.

    Args:
        image_sequences (List[Dict]): List of dictionaries with keys:
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
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Merged DataFrames: (peaks_all, cells_all, events_all)
    """
    peaks_dfs = []
    cells_dfs = []
    events_dfs = []

    for entry in image_sequences:
        try:
            base_path = Path(entry["path"])
            label = f"{entry['date']}__{entry['experiment_type']}__{entry['condition']}__{entry['concentration']}__{entry['time']}"

            for filetype, collector in zip(
                ["peaks", "cells", "events"], [peaks_dfs, cells_dfs, events_dfs]
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

    peaks_all.to_csv(output_folder / "merged_peaks.csv", index=False)
    cells_all.to_csv(output_folder / "merged_cells.csv", index=False)
    events_all.to_csv(output_folder / "merged_events.csv", index=False)

    logger.info(f"Saved merged datasets to: {output_folder}")

    return peaks_all, cells_all, events_all


# This script assumes image_sequences is passed from an external config or script.
# See the external script generate_merged_datasets.py to run this with actual data.
