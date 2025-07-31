#!/usr/bin/env python3
"""
Module: spatial.py

Utility function to construct an unbiased neighbor graph between cells
using Voronoi-based adjacency from centroids.

Usage Example:
    >>> from calcium_activity_characterization.utilities.spatial import build_spatial_neighbor_graph
    >>> graph = build_spatial_neighbor_graph(cells)

Requires:
    - Each Cell must have a unique .label and a .centroid (as tuple or np.ndarray)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from scipy.spatial import Voronoi
from calcium_activity_characterization.data.cells import Cell


def build_spatial_neighbor_graph(cells: List[Cell]) -> nx.Graph:
    """
    Build a Voronoi-based spatial neighbor graph from cell centroids.

    Args:
        cells (List[Cell]): List of Cell objects with centroid attributes.

    Returns:
        nx.Graph: Undirected graph with one node per cell, and edges for Voronoi neighbors.
    """
    if not cells:
        raise ValueError("Empty cell list passed to spatial neighbor graph builder.")

    label_to_centroid = {cell.label: tuple(cell.centroid) for cell in cells}
    labels = list(label_to_centroid.keys())
    centroids = np.array([label_to_centroid[label] for label in labels])

    if len(centroids) < 3:
        raise ValueError("At least 3 cells are needed to compute a Voronoi diagram.")

    vor = Voronoi(centroids)
    graph = nx.Graph()

    for label in labels:
        graph.add_node(label, pos=label_to_centroid[label])

    for i, j in vor.ridge_points:
        label_i = labels[i]
        label_j = labels[j]
        graph.add_edge(label_i, label_j, method="voronoi")

    return graph


def filter_graph_by_edge_length_mad(graph: nx.Graph, scale: float = 0.0) -> nx.Graph:
    """
    Remove outlier edges from the graph based on MAD (Median Absolute Deviation).

    Args:
        graph (nx.Graph): Input spatial graph with node 'pos' attributes.
        scale (float): Scaling factor for MAD thresholding (default 3.0).

    Returns:
        nx.Graph: A filtered copy of the graph without long edges.
    """
    lengths = []
    edge_data = []
    for u, v in graph.edges():
        p1 = np.array(graph.nodes[u]['pos'])
        p2 = np.array(graph.nodes[v]['pos'])
        dist = np.linalg.norm(p1 - p2)
        lengths.append(dist)
        edge_data.append((u, v, dist))

    if not lengths:
        return graph.copy()

    median = np.median(lengths)
    mad = np.median(np.abs(lengths - median))
    threshold = median + scale * mad

    filtered = nx.Graph()
    for u, data in graph.nodes(data=True):
        filtered.add_node(u, **data)

    for u, v, dist in edge_data:
        if dist <= threshold:
            attrs = graph.get_edge_data(u, v)
            filtered.add_edge(u, v, **attrs)

    return filtered
