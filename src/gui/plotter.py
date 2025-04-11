import matplotlib.pyplot as plt
import logging
from src.core.cell import Cell

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
