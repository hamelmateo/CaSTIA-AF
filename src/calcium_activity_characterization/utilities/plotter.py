import matplotlib.pyplot as plt
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional
import networkx as nx
from scipy.optimize import curve_fit


logger = logging.getLogger(__name__)

def plot_raster(
    output_path: Path,
    binary_traces: List[List[int]],
    cut_trace: int = 0
) -> None:
    """
    Plot a raster plot from a list of binary traces, with optional x-axis shift.

    Args:
        output_path (Path): File path to save the plot.
        binary_traces (List[List[int] | np.ndarray]): List of binary activity traces.
        cut_trace (int): Offset to add to x-axis labels for synchronization.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not binary_traces:
            logger.warning("No binary traces provided for raster plot.")
            return

        binarized_matrix = np.array(binary_traces)
        _, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(binarized_matrix, aspect='auto', cmap='Greys', interpolation='nearest')
        ax.set_title("Binarized Activity Raster Plot")
        ax.set_ylabel("Cell Index")
        ax.set_xlabel("Time")

        n_time = binarized_matrix.shape[1]
        ax.set_xticks(np.arange(0, n_time, max(1, n_time // 17)))
        ax.set_xticklabels([str(cut_trace + x) for x in ax.get_xticks().astype(int)])

        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.close()
        logger.info(f"Raster plot saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate raster plot: {e}")
        raise

def plot_raster_heatmap(
    output_path: Path,
    processed_traces: List[np.ndarray],
    clip_percentile: float = 98.0,
    cut_trace: int = 0
) -> None:
    """
    Plot a raster of processed calcium traces with a continuous colormap.

    Args:
        output_path (Path): Path to save the output figure.
        processed_traces (List[np.ndarray]): List of processed calcium intensity traces.
        clip_percentile (float): Percentile for upper clipping of intensities.
        cut_trace (int): Offset to add to the time axis (x-axis) for synchronization.

    Raises:
        ValueError: If no valid traces are provided.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not processed_traces:
            logger.warning("No processed traces provided for raster plot.")
            return

        trace_array = np.array(processed_traces)

        # Clip upper percentile to enhance contrast
        upper_clip = np.percentile(trace_array, clip_percentile)
        trace_array_clipped = np.clip(trace_array, a_min=None, a_max=upper_clip)

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.get_cmap("viridis")
        im = ax.imshow(trace_array_clipped, aspect='auto', cmap=cmap, interpolation='nearest')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Processed Trace Intensity")

        ax.set_title("Processed Trace Raster Heatmap")
        ax.set_ylabel("Cell Index")
        ax.set_xlabel("Time")

        # Add offset to x-axis if cut_trace > 0
        if cut_trace > 0:
            n_time = trace_array_clipped.shape[1]
            ax.set_xticks(np.arange(0, n_time, max(1, n_time // 17)))
            ax.set_xticklabels([str(cut_trace + x) for x in ax.get_xticks().astype(int)])

        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.close()
        logger.info(f"Processed trace raster plot saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate processed trace raster plot: {e}")
        raise

def convert_grayscale16_to_rgb(
    img16: np.ndarray
) -> np.ndarray:
    """
    Convert a 2D 16-bit (or float) image into an 8-bit RGB array.

    Usage example:
        >>> rgb = convert_grayscale16_to_rgb(raw16)

    Args:
        img16 (np.ndarray): 2D array of dtype uint16 or float.

    Returns:
        np.ndarray: H×W×3 uint8 RGB version of the input, linearly scaled.
    """
    if img16.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img16.shape}")

    # Force float for scaling
    data = img16.astype(np.float32)

    mn, mx = float(data.min()), float(data.max())
    logger.debug(f"Grayscale16 → float32: min={mn}, max={mx}")

    if mx <= mn:
        raise ValueError("Zero dynamic range: all pixels have the same value")

    # normalize to [0,1]
    norm = (data - mn) / (mx - mn)
    # scale to [0,255]
    gray8 = (norm * 255.0).round().astype(np.uint8)
    logger.debug(f"After scaling: dtype={gray8.dtype}, min={gray8.min()}, max={gray8.max()}")

    # Stack into RGB
    rgb = np.stack([gray8, gray8, gray8], axis=-1)
    return rgb

def render_cell_outline_overlay(
    background_image: np.ndarray,
    outline_mask: np.ndarray
) -> np.ndarray:
    """
    Create an RGB overlay image by converting a 16-bit grayscale background to RGB
    and overlaying a red outline where the mask is True.

    Args:
        background_image (np.ndarray):
            2D array (H×W) of dtype uint16 or float representing a grayscale image.
        outline_mask (np.ndarray):
            2D boolean array (H×W) where True indicates outline pixels.

    Returns:
        np.ndarray: H×W×3 uint8 RGB image with red outlines.

    Raises:
        ValueError: If shapes are incompatible or inputs invalid.
    """
    try:
        # Convert background to 8-bit RGB
        rgb = convert_grayscale16_to_rgb(background_image)

        # Ensure mask is boolean and shape matches
        mask = np.asarray(outline_mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"Outline mask shape {mask.shape} does not match "
                f"background image shape {rgb.shape[:2]}"
            )

        # Overlay red on masked pixels
        overlay = rgb.copy()
        overlay[mask, 0] = 255
        overlay[mask, 1] = 0
        overlay[mask, 2] = 0

        return overlay

    except Exception as e:
        logger.error(f"RenderCellOutlineOverlay failed: {e}")
        raise

def plot_minima_diagnostics(
    trace: np.ndarray,
    anchor_idx: List[int],
    inserted_idx: List[int],
    discarded1: List[int],
    discarded2: List[int],
    discarded3: List[int],
    output_dir: Path
) -> None:
    """
    Plot trace with all categories of local minima overlaid.

    Args:
        trace (np.ndarray): The smoothed intensity trace.
        anchor_idx (List[int]): Final retained anchor indices.
        inserted_idx (List[int]): Newly added anchors.
        discarded1 (List[int]): Discarded after shoulder rejection.
        discarded2 (List[int]): Discarded after angle filtering.
        discarded3 (List[int]): Reserved for future use.
        output_dir (Path): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(trace, label="Trace", lw=1.5)

    if anchor_idx:
        ax.scatter(anchor_idx, trace[anchor_idx], c="red", label="Final Anchors", s=15)
    if inserted_idx:
        ax.scatter(inserted_idx, trace[inserted_idx], c="orange", label="Inserted Anchors", s=15)
    if discarded1:
        ax.scatter(discarded1, trace[discarded1], c="gray", label="Shoulder Rejected", s=10, alpha=0.7)
    if discarded2:
        ax.scatter(discarded2, trace[discarded2], c="deepskyblue", label="Angle Rejected", s=10, alpha=0.5)
    if discarded3:
        ax.scatter(discarded3, trace[discarded3], c="lightgreen", label="Extra Discarded", s=10, alpha=0.5)

    ax.set_title("Local Minima Filtering Diagnostics")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    while (output_dir / f"local_minimas_filtering_{counter}.png").exists():
        counter += 1
    save_path = output_dir / f"local_minimas_filtering_{counter}.png"
    plt.savefig(save_path)
    plt.close()

def plot_final_baseline_fit(
    trace: np.ndarray,
    baseline: np.ndarray,
    anchor_idx: List[int],
    detrended: np.ndarray,
    label: str,
    output_dir: Path,
    model_name: str
) -> None:
    """
    Plot the raw trace, baseline, anchor points, and final detrended result.

    Args:
        trace (np.ndarray): Smoothed input trace.
        baseline (np.ndarray): Fitted baseline.
        anchor_idx (List[int]): Anchor indices used for baseline fit.
        detrended (np.ndarray): Detrended result.
        label (str): Label suffix for filename.
        output_dir (Path): Directory to save plot.
        model_name (str): Label for the model type.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(trace, label="Trace")
    axs[0].scatter(anchor_idx, trace[anchor_idx], c='red', label="Anchors", s=15)
    axs[0].legend()
    axs[0].set_title("Raw Trace with Anchor Points")

    axs[1].plot(trace, label="Trace")
    axs[1].plot(baseline, label=f"Baseline ({model_name})")
    axs[1].legend()
    axs[1].set_title("Fitted Baseline")

    axs[2].plot(detrended, label="Detrended Trace")
    axs[2].legend()
    axs[2].set_title("Detrended Output")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    while (output_dir / f"baseline_fit_{counter}.png").exists():
        counter += 1
    save_path = output_dir / f"baseline_fit_{counter}.png"
    plt.savefig(save_path)
    plt.close()


def plot_umap(
    embedding: np.ndarray,
    output_path: Path = None,
    title: str = "UMAP Projection",
    n_neighbors: int = None,
    min_dist: float = None,
    n_components: int = None,
    labels: np.ndarray = None
) -> None:
    """
    Plot the UMAP embedding.

    Args:
        embedding (np.ndarray): UMAP embedding (2D array).
        output_path (Path): Path to save the plot (optional).
        title (str): Title of the plot.
        n_neighbors (int): The size of the local neighborhood used for manifold approximation.
        min_dist (float): The minimum distance between points in the low-dimensional space.
        n_components (int): The number of dimensions for the UMAP embedding.
        labels (np.ndarray): Optional cluster labels for coloring points.
    """
    plt.figure(figsize=(8, 6))
    
    if labels is not None:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
        plt.colorbar(scatter, label="Cluster Labels")
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10, c='blue', alpha=0.6)

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)

    # Add UMAP parameters as text on the plot
    if n_neighbors is not None or min_dist is not None or n_components is not None:
        param_text = "\n".join([
            f"n_neighbors = {n_neighbors}" if n_neighbors is not None else "",
            f"min_dist = {min_dist}" if min_dist is not None else "",
            f"n_components = {n_components}" if n_components is not None else ""
        ]).strip()
        plt.gca().text(
            0.95, 0.95, param_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8)
        )

    plt.tight_layout()

    if output_path:
        np.save(output_path, embedding)
        logger.info(f"UMAP saved to {output_path}")
    else:
        plt.show()


def plot_event_graph(event_id: int, graph: nx.DiGraph, label_to_time: dict, save_path: Path = None):
    """
    Plot a DAG of cell-to-cell communications with activation times.

    Args:
        event_id (int): ID of the event (for title or filename).
        graph (nx.DiGraph): The DAG of communications.
        label_to_time (dict): Mapping from cell label to activation time.
        save_path (Path, optional): If provided, saves the plot to this path.
    """
    try:
        pos = nx.spring_layout(graph, seed=42)
        node_labels = {
            node: f"{node}\n{label_to_time.get(node, '?')}" for node in graph.nodes
        }
        times = [label_to_time.get(n, 0) for n in graph.nodes]

        plt.figure(figsize=(8, 6))
        nodes = nx.draw_networkx_nodes(graph, pos, node_color=times, cmap='viridis', node_size=500)
        edges = nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15)
        labels = nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(times), max(times)))
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label='Activation Time')


        plt.title(f"[Event {event_id}] DAG with Activation Times (No Root Found)")
        plt.axis('off')

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"🔍 Saved diagnostic plot to {save_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot event graph for Event {event_id}: {e}")


def plot_spatial_neighbor_graph(
    graph: nx.Graph,
    mask: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot or save a spatial neighbor graph from the population.

    Args:
        graph (nx.Graph): Graph with nodes that contain 'pos' (y, x) coordinates.
        mask (Optional[np.ndarray]): Optional grayscale image to show underneath the graph.
        output_path (Optional[Path]): If provided, saves the figure to this path. Else, shows interactively.
    """
    if not graph.nodes:
        raise ValueError("Graph has no nodes to plot.")

    pos = {node: (xy[1], xy[0]) for node, xy in nx.get_node_attributes(graph, "pos").items()}  # x=col, y=row

    plt.figure(figsize=(8, 8))
    if mask is not None:
        plt.imshow(mask, cmap="gray")

    nx.draw(graph, pos, node_size=30, node_color='red', edge_color='blue', with_labels=False)
    plt.axis("equal")
    plt.title("Spatial Neighbor Graph")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()



def plot_event_growth_curve(values: list[float], start: int, time_to_50: int, title: str, save_path: Path) -> None:

    # framewise_peaking_labels covers every frame from start to end
    frames = list(range(start, start + len(values)))

    # --- start the plot ---------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(frames, values, alpha=0.6)

    # --- fit a logistic (sigmoid) -----------------------------
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    try:
        # initial guesses: L≈max, k=1, midpoint≈middle frame
        p0 = [max(values), 1.0, start + len(values) / 2]
        popt, _ = curve_fit(logistic, frames, values, p0=p0, maxfev=10_000)
        L, _, _ = popt

        xs = np.linspace(frames[0], frames[-1], 200)
        ys = logistic(xs, *popt)
        ax.plot(xs, ys, color="orange", lw=2, label="Sigmoid fit")

        # draw the asymptote
        ax.hlines(L, frames[0], frames[-1],
                    colors="red", linestyles="--",
                    label=f"Asymptote = {L:.1f}")

    except Exception as e:
        logger.warning(f"Plot {title}: sigmoid fit failed: {e}")

    # --- mark time‐to‐50% -------------------------------------
    # _compute_time_and_peak_rate_at_50 returns (idx_offset, rate)
    t50 = time_to_50 + start
    if 0 <= time_to_50 < len(values):
        y50 = values[time_to_50]
        ax.scatter([t50], [y50],
                    color="green", marker="*", s=150,
                    label=f"50% @ t={time_to_50} (fr={t50})")
        ax.text(t50, y50,
                f"  t₅₀={time_to_50}",
                va="bottom", ha="left",
                color="green", fontsize=10)

    # --- finalize & save --------------------------------------
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cumulative number of cells peaking")
    ax.grid(True)
    ax.legend(loc="best")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_cell_connection_network(
    graph: nx.Graph,
    nuclei_mask: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot or save the interaction graph with weighted edges (weight ≥3) 
    overlayed on an optional nuclei segmentation mask.

    - Figure background is white.
    - Nuclei mask: all nonzero pixels shown as uniform gray on white.
    - Edges colored by weight with a colorbar.
    - Nodes plotted in red.

    Args:
        graph (nx.Graph): Interaction graph to plot.
        nuclei_mask (Optional[np.ndarray]): 2D array of integer labels
            (0 = background, >0 = cell). If provided, nonzero pixels
            will be shaded gray.
        output_path (Optional[Path]): Path to save the figure. If None,
            displays interactively.
    """
    try:
        if not graph.nodes:
            raise ValueError("Graph has no nodes to plot.")

        # 1) filter edges by weight ≥3
        filtered_edges = [
            (u, v) 
            for u, v, d in graph.edges(data=True) 
            if d.get('weight', 0) >= 3
        ]
        weights = [graph[u][v]['weight'] for u, v in filtered_edges]
        if not weights:
            logger.warning("No edges with weight ≥3 to display.")
            return

        # 2) Determine active vs inactive nodes
        active_nodes = {u for u, v in filtered_edges} | {v for u, v in filtered_edges}
        # build a color for each node: dark red if active, light red otherwise
        node_colors = [
            'red' if node in active_nodes else 'lightcoral'
            for node in graph.nodes()
        ]

        # normalize for colormap
        max_w = max(weights)
        norm_weights = [w / max_w for w in weights]

        # 2) white background
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('white')

        # Overlay nuclei mask as uniform gray on white background
        if nuclei_mask is not None:
            # Build a float image: 1.0 = white, 0.7 = dark gray for cells
            mask_img = np.ones_like(nuclei_mask, dtype=float)
            mask_img[nuclei_mask == 0] = 0.9
            mask_img[nuclei_mask > 0] = 0.7
            ax.imshow(mask_img, cmap='gray', vmin=0.0, vmax=1.0)

        # 4) get positions
        pos = {
            node: (xy[1], xy[0]) 
            for node, xy in nx.get_node_attributes(graph, 'pos').items()
        }

        # 5) draw edges
        nx.draw_networkx_edges(
            graph.edge_subgraph(filtered_edges),
            pos,
            edge_color=norm_weights,
            edge_cmap=plt.cm.Reds,
            edge_vmin=0,
            edge_vmax=1,
            width=2,
            ax=ax
        )

        # 7) Draw nodes with per-node color
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=8,
            node_color=node_colors,
            ax=ax
        )

        # 8) colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Reds, 
            norm=plt.Normalize(vmin=min(weights), vmax=max(weights))
        )
        sm.set_array(weights)
        cbar = fig.colorbar(
            sm, ax=ax, fraction=0.046, pad=0.04, 
            aspect=20, shrink=0.8
        )
        cbar.set_label("Edge Weight (Communication Count)")
        # keep ticks readable on white
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

        # 8) finalize
        ax.set_aspect('equal')
        ax.set_title(
            "Cells Connection Network (Weighted Edges, ≥3)", 
            color='black'
        )
        ax.axis('off')
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_path, dpi=300, 
                facecolor=fig.get_facecolor()
            )
            plt.close(fig)
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Failed to plot cell connection network: {e}")


def plot_metric_on_overlay(
    nuclei_mask: np.ndarray,
    cell_pixel_coords: dict[int, np.ndarray],
    metric_counts: dict[int, int],
    output_path: Path,
    title: str = "Mapping of Cell Metric",
    colorbar_label: str = "Metric Count",
    cmap: str = "Reds"
) -> None:
    """
    Plot the nuclei overlay colored by origin count per cell.

    Args:
        nuclei_mask (np.ndarray): Original nuclei image to display as background.
        cell_pixel_coords (dict[int, np.ndarray]): Mapping from cell label to pixel coordinates (Y, X).
        metric_counts (dict[int, int]): Mapping from cell label to number of origin events.
        output_path (Path): File path to save the plot.
        cmap (str): Matplotlib colormap to use.
    """
    overlay = np.zeros_like(nuclei_mask, dtype=np.float32)

    max_count = max(metric_counts.values()) if metric_counts else 1

    for label, coords in cell_pixel_coords.items():
        count = metric_counts.get(label, 0)
        norm_value = count #/ max_count
        for coord in coords:
            y, x = coord
            overlay[y, x] = norm_value

    plt.figure(figsize=(10, 10))
    plt.imshow(nuclei_mask, cmap="gray", alpha=0.6)
    im = plt.imshow(overlay, cmap=cmap, alpha=0.8, vmin=0, vmax=max_count)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, aspect=20, shrink=0.8)
    cbar.set_label(colorbar_label)
    plt.title(title)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"{title} figure saved at {output_path}")
