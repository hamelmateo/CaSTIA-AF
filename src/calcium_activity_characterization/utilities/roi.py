# utilities/roi.py

import logging
from typing import List, Tuple
import networkx as nx
from calcium_activity_characterization.data.cells import Cell

logger = logging.getLogger(__name__)

def filter_cells_and_graph(
    graph: nx.Graph,
    cells: List[Cell],
    roi_scale: float,
    img_shape: Tuple[int, int],
    border_margin: int
) -> Tuple[List[Cell], nx.Graph]:
    """
    Crop to a centered ROI at scale `roi_scale`, then discard any cell whose
    pixel footprint (from cell.pixel_coords) touches the outer `border_margin`
    of that ROI, and prune the neighbor graph.

    Steps:
      1. Compute centered ROI bounds from `roi_scale` and `img_shape`.
      2. Define a “safe” region by moving each ROI edge inwards by `border_margin`.
      3. For each cell, use `cell.pixel_coords` to compute its pixel-wise
         bounding box, and keep it only if *all* pixels lie within the safe region.
      4. Prune the neighbor-graph to only the surviving cells.

    Args:
        graph:         Full neighbor graph (nodes labeled by `cell.label`).
        cells:         List of all `Cell` objects, each with `.label` and
                       `.pixel_coords` an Iterable of (row, col) tuples.
        roi_scale:     Fraction of original height/width to keep (0 < scale ≤ 1).
        img_shape:     Tuple `(height, width)` of the Hoechst image.
        border_margin: Number of pixels inside the ROI border to exclude.

    Returns:
        filtered_cells: List of `Cell` whose entire pixel footprint is within
                        the safe ROI (after border_margin).
        pruned_graph:   `nx.Graph` induced on those cell labels.

    Raises:
        ValueError: If `roi_scale` not in (0,1].
    """
    if not (0 < roi_scale <= 1):
        raise ValueError(f"roi_scale must be in (0,1], got {roi_scale}")

    height, width = img_shape
    try:
        # 1) compute centered ROI
        crop_h = int(height * roi_scale)
        crop_w = int(width * roi_scale)
        start_h = (height - crop_h) // 2
        start_w = (width - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        # 2) define safe region
        safe_start_h = start_h + border_margin
        safe_end_h   = end_h   - border_margin
        safe_start_w = start_w + border_margin
        safe_end_w   = end_w   - border_margin

        filtered_cells: List[Cell] = []
        for cell in cells:
            coords = cell.pixel_coords

            # compute bounding box of the cell’s pixels
            ys = [r for r, _ in coords]
            xs = [c for _, c in coords]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)

            # 3) keep cell only if its entire footprint lies within the safe region
            if (
                min_y >= safe_start_h and max_y < safe_end_h and
                min_x >= safe_start_w and max_x < safe_end_w
            ):
                # 4) update coordinates relative to the cropped ROI
                cell.pixel_coords = [
                    (r - start_h, c - start_w) for (r, c) in coords
                ]
                cy, cx = cell.centroid
                cell.centroid = (cy - start_h, cx - start_w)
                filtered_cells.append(cell)

        # 4) prune the graph
        keep_labels = {c.label for c in filtered_cells}
        pruned_graph = graph.subgraph(keep_labels).copy()

        # 6) update pruned graph node attributes
        for cell in filtered_cells:
            label = cell.label
            if pruned_graph.has_node(label):
                pruned_graph.nodes[label]['pos'] = cell.centroid

        return filtered_cells, pruned_graph

    except Exception:
        logger.exception(
            "Error filtering cells and pruning graph "
            f"(roi_scale={roi_scale}, border_margin={border_margin})"
        )
        raise
