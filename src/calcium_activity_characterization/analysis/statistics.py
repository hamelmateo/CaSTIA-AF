import numpy as np
from calcium_activity_characterization.logger import logger
import pandas as pd
from typing import Optional

def analyze_peak_intervals(peak_times: list[int]) -> tuple[list[int], float | None, float | None]:
    """
    Analyze periodicity of global (or trace-level) peak times.

    Computes inter-peak intervals, periodicity score (based on CV),
    and average peak frequency (events per frame).

    Args:
        peak_times (list[int]): Sorted list of peak times (in frames).

    Returns:
        tuple:
            - intervals (list[int]): list of inter-peak intervals.
            - periodicity_score (Optional[float]): [0, 1] score or None if too few events.
            - average_frequency (Optional[float]): Events per frame, or None if invalid.
    """
    if not peak_times or len(peak_times) < 2:
        return [], None, None

    peak_times = sorted(peak_times)
    intervals = np.diff(peak_times).tolist()

    if len(intervals) < 2:
        return intervals, None, None

    mean_ipi = np.mean(intervals)
    std_ipi = np.std(intervals)
    cv = std_ipi / mean_ipi if mean_ipi > 0 else None
    periodicity_score = 1 / (1 + cv) if cv is not None else None

    total_duration = peak_times[-1] - peak_times[0]
    average_frequency = len(intervals) / total_duration if total_duration > 0 else None

    return intervals, periodicity_score, average_frequency


def build_neighbor_pair_stats(
    *,
    cells_df: pd.DataFrame,
    comm_df: pd.DataFrame,
    dataset_col: str = "dataset",
    cell_id_col: str = "Cell ID",
    centroid_x_col: str = "Centroid X coordinate (um)",
    centroid_y_col: str = "Centroid Y coordinate (um)",
    neighbors_col: Optional[str] = "Neighbors (labels)",
    edges_df: Optional[pd.DataFrame] = None,
    edge_cols: tuple[str, str] = ("Cell ID", "Neighbor ID"),
) -> pd.DataFrame:
    """
    Build a per-dataset table of neighbor cell pairs with their centroid distance and
    the number of communications between the two cells.

    The function uses either:
      (A) a neighbors column in `cells_df` that contains a list/JSON list of neighbor labels, or
      (B) a separate long `edges_df` with one row per (cell_id, neighbor_id) pair.

    Communications are counted regardless of direction (origin↔cause), and pairs with
    zero communications are kept with count=0.

    Args:
        cells_df: DataFrame with at least [dataset_col, cell_id_col, centroid_x_col, centroid_y_col],
                  and optionally [neighbors_col] if `edges_df` is not provided.
        comm_df: DataFrame with communications; must include [dataset_col, "Origin cell ID", "Cause cell ID"].
        dataset_col: Column name identifying the dataset each row belongs to.
        cell_id_col: Column with integer cell labels/IDs.
        centroid_x_col: Column with centroid X coordinate (ideally in microns).
        centroid_y_col: Column with centroid Y coordinate (ideally in microns).
        neighbors_col: Column in `cells_df` that contains neighbors (list of ints). Used if `edges_df` is None.
        edges_df: Optional pre-built edge list DataFrame. If provided, must have at least [dataset_col, edge_cols[0], edge_cols[1]].
        edge_cols: Names of the two columns in `edges_df` giving the endpoints (defaults to ("Cell ID","Neighbor ID")).

    Returns:
        pd.DataFrame with columns:
            - dataset_col
            - Cell A (int)
            - Cell B (int)
            - distance_um (float)
            - n_communications (int)

        Each row represents an undirected neighbor pair (Cell A < Cell B) within a dataset.

    Raises:
        KeyError: if required columns are missing.
        ValueError: if inputs are empty or inconsistent.
    """
    try:
        # ---- Validate inputs
        if cells_df is None or cells_df.empty:
            raise ValueError("`cells_df` is empty.")
        if comm_df is None:
            raise ValueError("`comm_df` is None (pass an empty DataFrame if not available).")
        req_cells = {dataset_col, cell_id_col, centroid_x_col, centroid_y_col}
        missing_cells = req_cells - set(cells_df.columns)
        if missing_cells:
            raise KeyError(f"`cells_df` missing columns: {sorted(missing_cells)}")

        if edges_df is None and neighbors_col is None:
            raise ValueError("Provide either `edges_df` or `neighbors_col` in `cells_df`.")

        if edges_df is not None:
            req_edges = {dataset_col, edge_cols[0], edge_cols[1]}
            missing_edges = req_edges - set(edges_df.columns)
            if missing_edges:
                raise KeyError(f"`edges_df` missing columns: {sorted(missing_edges)}")

        if not comm_df.empty:
            req_comm = {dataset_col, "Origin cell ID", "Cause cell ID"}
            missing_comm = req_comm - set(comm_df.columns)
            if missing_comm:
                raise KeyError(f"`comm_df` missing columns: {sorted(missing_comm)}")

        # ---- Prepare centroids table (numeric)
        centroids = cells_df[[dataset_col, cell_id_col, centroid_x_col, centroid_y_col]].copy()
        centroids[centroid_x_col] = pd.to_numeric(centroids[centroid_x_col], errors="coerce")
        centroids[centroid_y_col] = pd.to_numeric(centroids[centroid_y_col], errors="coerce")
        centroids = centroids.dropna(subset=[centroid_x_col, centroid_y_col])

        # ---- Build neighbor pairs (undirected, per dataset)
        if edges_df is not None:
            pairs = edges_df[[dataset_col, edge_cols[0], edge_cols[1]]].copy()
            pairs.rename(columns={edge_cols[0]: "cell_u", edge_cols[1]: "cell_v"}, inplace=True)
        else:
            if neighbors_col not in cells_df.columns:
                raise KeyError(f"`neighbors_col='{neighbors_col}' not found in `cells_df`.")
            # explode neighbors
            work = cells_df[[dataset_col, cell_id_col, neighbors_col]].copy()
            # neighbors may be stored as JSON strings -> ensure list
            def _to_list(x):
                if isinstance(x, str):
                    try:
                        import json
                        val = json.loads(x)
                        return val if isinstance(val, list) else []
                    except Exception:
                        return []
                return list(x) if isinstance(x, (list, tuple, set, pd.Series, np.ndarray)) else []
            work["_nbrs"] = work[neighbors_col].apply(_to_list)
            pairs = work.explode("_nbrs", ignore_index=True)
            pairs = pairs.dropna(subset=["_nbrs"])
            pairs = pairs.rename(columns={cell_id_col: "cell_u"})
            pairs["cell_v"] = pd.to_numeric(pairs["_nbrs"], errors="coerce").astype("Int64")
            pairs = pairs.dropna(subset=["cell_v"])
            pairs["cell_v"] = pairs["cell_v"].astype(int)
            pairs = pairs[[dataset_col, "cell_u", "cell_v"]]

        # normalize undirected pair (Cell A < Cell B)
        pairs["Cell A"] = pairs[["cell_u", "cell_v"]].min(axis=1)
        pairs["Cell B"] = pairs[["cell_u", "cell_v"]].max(axis=1)
        pairs = pairs[[dataset_col, "Cell A", "Cell B"]].drop_duplicates()
        # drop self-pairs just in case
        pairs = pairs.loc[pairs["Cell A"] != pairs["Cell B"]].reset_index(drop=True)

        if pairs.empty:
            logger.warning("build_neighbor_pair_stats: no neighbor pairs found after normalization.")
            return pairs.assign(distance_um=np.nan, n_communications=0)

        # ---- Attach centroids for distance computation
        pairs = pairs.merge(
            centroids.rename(columns={
                cell_id_col: "Cell A",
                centroid_x_col: "x_a",
                centroid_y_col: "y_a",
            }),
            on=[dataset_col, "Cell A"], how="left"
        ).merge(
            centroids.rename(columns={
                cell_id_col: "Cell B",
                centroid_x_col: "x_b",
                centroid_y_col: "y_b",
            }),
            on=[dataset_col, "Cell B"], how="left"
        )

        # Drop pairs missing centroids (can happen if ROI filtered)
        before = len(pairs)
        pairs = pairs.dropna(subset=["x_a", "y_a", "x_b", "y_b"])
        dropped = before - len(pairs)
        if dropped:
            logger.info("build_neighbor_pair_stats: dropped %d pairs missing centroids.", dropped)

        # Compute Euclidean distance (um)
        pairs["distance_um"] = np.sqrt((pairs["x_a"] - pairs["x_b"])**2 + (pairs["y_a"] - pairs["y_b"])**2)

        # ---- Communications: count per unordered pair per dataset
        if comm_df.empty:
            comm_counts = pd.DataFrame(columns=[dataset_col, "Cell A", "Cell B", "n_communications"])
        else:
            comm = comm_df[[dataset_col, "Origin cell ID", "Cause cell ID"]].copy()
            comm["Cell A"] = comm[["Origin cell ID", "Cause cell ID"]].min(axis=1)
            comm["Cell B"] = comm[["Origin cell ID", "Cause cell ID"]].max(axis=1)
            comm_counts = (
                comm.groupby([dataset_col, "Cell A", "Cell B"])
                    .size()
                    .reset_index(name="n_communications")
            )

        # ---- Join pairs with counts (keep zero counts)
        out = pairs.merge(
            comm_counts,
            on=[dataset_col, "Cell A", "Cell B"],
            how="left"
        )
        out["n_communications"] = out["n_communications"].fillna(0).astype(int)

        # Final tidy columns
        out = out[[dataset_col, "Cell A", "Cell B", "distance_um", "n_communications"]].sort_values(
            by=[dataset_col, "Cell A", "Cell B"], kind="mergesort"
        ).reset_index(drop=True)

        logger.info(
            "build_neighbor_pair_stats: built %d pairs across %d datasets (mean distance=%.2f um)",
            len(out), out[dataset_col].nunique(), out["distance_um"].mean() if not out.empty else float("nan")
        )
        return out

    except Exception as exc:
        logger.exception("build_neighbor_pair_stats failed: %s", exc)
        raise