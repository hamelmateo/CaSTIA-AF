import matplotlib.pyplot as plt

def explore_processing_parameters(cell, sigmas, cutoffs, fs=1.0, order=2):
    """
    Explore combinations of Gaussian smoothing (sigma) and high-pass filtering (cutoff)
    and plot the resulting normalized traces for visual comparison. The raw trace is shown separately.

    Args:
        cell (Cell): Cell object with raw intensity_trace.
        sigmas (list[float]): List of sigma values for Gaussian smoothing.
        cutoffs (list[float]): List of cutoff frequencies for high-pass filter.
        fs (float): Sampling frequency in Hz (default 1.0).
        order (int): Filter order for high-pass (default 2).
    """
    raw = cell.intensity_trace
    n_rows = len(cutoffs)
    n_cols = len(sigmas)

    # Plot the raw trace separately
    plt.figure(figsize=(8, 3))
    plt.plot(raw, color='black')
    plt.title(f"Raw Intensity Trace - Cell {cell.label}")
    plt.xlabel("Timepoint")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the parameter sweep
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.5*n_rows), sharex=True, sharey=True)

    for i, cutoff in enumerate(cutoffs):
        for j, sigma in enumerate(sigmas):
            trace = cell.get_processed_trace(sigma=sigma, cutoff=cutoff, fs=fs, order=order)
            ax = axes[i][j] if n_rows > 1 else axes[j]
            ax.plot(trace, color='blue')
            ax.set_title(f"σ={sigma}, cutoff={cutoff}")
            ax.grid(True)

    fig.suptitle(f"Processed Trace Grid (Cell {cell.label})", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
