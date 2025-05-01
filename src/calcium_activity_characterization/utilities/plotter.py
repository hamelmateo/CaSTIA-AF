import matplotlib.pyplot as plt
import logging
from data.cells import Cell
import pandas as pd


logger = logging.getLogger(__name__)

def show_cell_plot(cell: Cell) -> None:
    """
    Plot the intensity trace of a single cell.

    Args:
        cell (Cell): The cell object whose intensity trace will be plotted.
    """
    if not cell.raw_intensity_trace:
        logger.info(f"Cell {cell.label} has no intensity data to plot.")
        return

    try:
        fig, ax = plt.subplots()
        ax.plot(cell.raw_intensity_trace, label=f"Cell {cell.label} ({cell.centroid[1]}, {cell.centroid[0]})")
        ax.set_title(f"Intensity Profile for Cell {cell.label}")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Mean Intensity")
        ax.legend()
        ax.grid(True)
        plt.show(block=False)
    except Exception as e:
        logger.error(f"Failed to plot intensity profile for cell {cell.label}: {e}")



def plot_arcos_binarized_data(df: pd.DataFrame, track_id: int):
    """
    Plot raw intensity, detrended/rescaled intensity, and binarized data
    for a specific trackID.

    Args:
        df (pd.DataFrame): DataFrame containing the binarized data.
        track_id (int): Track identifier (cell label).
    """
    cell_data = df[df['trackID'] == track_id].sort_values('frame')

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(cell_data['frame'], cell_data['intensity'], 
            label='Raw Intensity', color='gray', linestyle='--', alpha=0.6)

    ax.plot(cell_data['frame'], cell_data['intensity.resc'], 
            label='Detrended & Rescaled Intensity', color='blue')

    # Plot binarized data (scaled for visibility)
    ax.step(cell_data['frame'], cell_data['intensity.bin'] * cell_data['intensity.resc'].max(), 
            label='Binarized Activity', color='red', linewidth=2, where='post')

    ax.set_xlabel('Frame (Time)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'ARCOS Binarization - Cell {track_id}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
