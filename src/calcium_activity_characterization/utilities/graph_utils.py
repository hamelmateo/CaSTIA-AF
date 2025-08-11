# Usage example:
# >>> G_top = filter_graph_by_edge_weight_percentile(G, percentile=90.0, preserve_nodes=True)
# >>> plot_cell_connection_network(G_top, nuclei_mask=mask, output_path=Path("conn_net_top10.png"))
#
# This keeps only the ~top 10% heaviest edges by 'weight' and then plots them.

from __future__ import annotations

import numpy as np
import networkx as nx

# Fallback logger if your module doesn't define one
try:
    logger  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    import logging
    logger = logging.getLogger("calcium")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


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
