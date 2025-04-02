import matplotlib.pyplot as plt
from src.core.cell import Cell

def show_cell_plot(cell: Cell):
    if len(cell.intensity_trace) == 0:
        print(f"[INFO] Cell {cell.label} has no intensity data to plot.")
        return

    fig, ax = plt.subplots()
    ax.plot(cell.intensity_trace, label=f"Cell {cell.label}")
    ax.set_title(f"Intensity Profile for Cell {cell.label}")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Mean Intensity")
    ax.legend()
    ax.grid(True)
    plt.show(block=False)