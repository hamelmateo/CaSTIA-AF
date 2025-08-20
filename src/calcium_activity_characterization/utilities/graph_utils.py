# Usage example:
# >>> G_top = filter_graph_by_edge_weight_percentile(G, percentile=90.0, preserve_nodes=True)
# >>> plot_cell_connection_network(G_top, nuclei_mask=mask, output_path=Path("conn_net_top10.png"))
#
# This keeps only the ~top 10% heaviest edges by 'weight' and then plots them.

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.spatial import Voronoi

from calcium_activity_characterization.logger import logger


def filter_graph_by_edge_weight_percentile(
    graph: nx.Graph,
    percentile: float,
    weight_attr: str = "weight",
    preserve_nodes: bool = True,
) -> nx.Graph:
    """
    Return a new graph with only edges whose weight is at/above the given percentile.

    The percentile is computed across **all** edges' `weight_attr` values in `graph`.
    Edges with weight >= threshold are kept; others are removed.

    Args:
        graph (nx.Graph): Input graph. Edge weights are read from `weight_attr`.
        percentile (float): Percentile in [0, 100]. Example: 90.0 keeps the top ~10%.
        weight_attr (str): Edge attribute name holding weights. Defaults to "weight".
        preserve_nodes (bool): If True, keep all original nodes (and attributes) even if
            they become isolated after filtering edges. If False, only nodes incident
            to retained edges remain.

    Returns:
        nx.Graph: A new graph with filtered edges. Graph, node, and edge attributes
        are preserved for the retained elements.

    Raises:
        ValueError: If `percentile` is outside [0, 100] or the input graph is None.

    Notes:
        - If the input graph has no edges, the function returns a copy of the input
          (with nodes only, if `preserve_nodes=True`) and logs a warning.
        - If all edges have identical weights, the threshold equals that weight,
          and all edges are kept.
    """
    try:
        if graph is None:
            raise ValueError("`graph` is None.")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"`percentile` must be in [0, 100], got {percentile}.")

        # Short-circuit: no edges to evaluate
        if graph.number_of_edges() == 0:
            logger.warning("filter_graph_by_edge_weight_percentile: graph has no edges.")
            return graph.copy()

        # Gather weights (defaulting missing to 0.0)
        all_edges = list(graph.edges(data=True))
        weights = np.asarray([float(d.get(weight_attr, 0.0)) for *_ , d in all_edges], dtype=float)

        # Compute percentile threshold (NumPy compatibility across versions)
        try:
            thr = float(np.percentile(weights, percentile, method="linear"))
        except TypeError:  # NumPy < 1.22
            thr = float(np.percentile(weights, percentile, interpolation="linear"))

        # Decide which edges to keep (inclusive)
        edges_to_keep = [(u, v) for u, v, d in all_edges if float(d.get(weight_attr, 0.0)) >= thr]

        if preserve_nodes:
            # Copy graph and drop edges below threshold; keep all nodes/attrs
            G_new = graph.copy()
            # Build a set for faster membership checks
            keep_set = { (u, v) if G_new.has_edge(u, v) else (v, u) for (u, v) in edges_to_keep }
            to_remove = []
            for u, v in G_new.edges():
                e = (u, v) if (u, v) in keep_set else (v, u)
                if e not in keep_set:
                    to_remove.append((u, v))
            if to_remove:
                G_new.remove_edges_from(to_remove)
        else:
            # Only nodes incident to the kept edges remain
            G_new = graph.edge_subgraph(edges_to_keep).copy()

        kept = len(edges_to_keep)
        total = graph.number_of_edges()
        logger.info(
            "filter_graph_by_edge_weight_percentile: percentile=%.1f, thr=%.6g, kept %d/%d edges (%.1f%%).",
            percentile, thr, kept, total, 100.0 * kept / max(1, total)
        )
        return G_new

    except Exception as e:
        # Fail safely: return a shallow copy of the original graph
        logger.error(f"filter_graph_by_edge_weight_percentile failed: {e}", exc_info=True)
        return graph.copy()


def finite_ridge_segments(vor: Voronoi, bbox: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    """
    Convert Voronoi ridges (including infinite ones) to finite line segments
    clipped to a bounding box.

    Args:
        vor (Voronoi): SciPy Voronoi object computed on (x, y) points.
        bbox (tuple[float, float, float, float]): (xmin, xmax, ymin, ymax) bounds.

    Returns:
        list[tuple[float, float, float, float]]: List of line segments (x0, y0, x1, y1).
    """
    xmin, xmax, ymin, ymax = bbox
    center = vor.points.mean(axis=0)
    radius = max(xmax - xmin, ymax - ymin) * 2.0  # generous extension

    segments: list[tuple[float, float, float, float]] = []

    for (p, q), (v0, v1) in zip(vor.ridge_points, vor.ridge_vertices):
        if v0 >= 0 and v1 >= 0:
            x0, y0 = vor.vertices[v0]
            x1, y1 = vor.vertices[v1]
            segments.append((x0, y0, x1, y1))
            continue

        # One vertex at infinity: extend the finite vertex in direction normal to the ridge
        if v0 == -1 or v1 == -1:
            finite_v = v1 if v0 == -1 else v0
            x_f, y_f = vor.vertices[finite_v]

            # Direction perpendicular to the line between the two points p and q
            px, py = vor.points[p]
            qx, qy = vor.points[q]
            dx, dy = qx - px, qy - py
            # Normal (rotate by 90°)
            nx_dir, ny_dir = -dy, dx

            # Pointing outward from center
            d_cx, d_cy = x_f - center[0], y_f - center[1]
            if (nx_dir * d_cx + ny_dir * d_cy) < 0:
                nx_dir, ny_dir = -nx_dir, -ny_dir

            # Build a long segment and clip later
            x0, y0 = x_f, y_f
            x1, y1 = x_f + nx_dir * radius, y_f + ny_dir * radius
            segments.append((x0, y0, x1, y1))

    # Clip each segment to bbox (Cohen–Sutherland style quick clip)
    def _clip_segment(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float] | None:
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

        def _code(x: float, y: float) -> int:
            c = INSIDE
            if x < xmin: c |= LEFT
            elif x > xmax: c |= RIGHT
            if y < ymin: c |= BOTTOM
            elif y > ymax: c |= TOP
            return c

        c0, c1 = _code(x0, y0), _code(x1, y1)
        while True:
            if not (c0 | c1):   # both inside
                return x0, y0, x1, y1
            if c0 & c1:         # trivially outside
                return None
            # choose an endpoint outside
            c_out = c0 or c1
            if c_out & TOP:
                x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
                y = ymax
            elif c_out & BOTTOM:
                x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
                y = ymin
            elif c_out & RIGHT:
                y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
                x = xmax
            else:  # LEFT
                y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
                x = xmin
            if c_out == c0:
                x0, y0 = x, y
                c0 = _code(x0, y0)
            else:
                x1, y1 = x, y
                c1 = _code(x1, y1)

    clipped: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in segments:
        seg = _clip_segment(x0, y0, x1, y1)
        if seg is not None:
            clipped.append(seg)

    return clipped