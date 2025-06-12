from typing import List, Optional, Tuple
from calcium_activity_characterization.data.cells import Cell


class Cluster:
    """
    Represents a group of temporally correlated peaks from different cells.

    Attributes:
        id (int): Unique identifier for the cluster.
        members (List[Tuple[Cell, int]]): List of (cell, peak_index) tuples.
        start_time (int): Start of the earliest peak.
        end_time (int): End of the latest peak.
        center_time (float): Mean of peak times.
    """
    def __init__(self, id: int, start_time: int, end_time: int):
        """
        Initialize a new cluster.
        Args:
            id (int): Unique identifier for the cluster.
            start_time (int): Start time of the cluster.
            end_time (int): End time of the cluster.
        """
        self.id = id
        self.members: List[Tuple[Cell, int]] = []

        self.start_time: int = start_time
        self.end_time: int = end_time
        self.center_time: Optional[float] = None
        self.metadata: dict[str, any] = {}

    def add(self, cell: Cell, peak_index: int):
        """
        Add a cell's peak to the cluster.

        Args:
            cell (Cell): The originating cell.
            peak_index (int): Index of the peak in cell.peaks.
        """
        peak = cell.trace.peaks[peak_index]

        self.members.append((cell, peak_index))

        peak.in_cluster = True
        peak.cluster_id = self.id

        self.center_time = sum(cell.trace.peaks[i].peak_time for cell, i in self.members) / len(self.members)

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return (
            f"Cluster(id={self.id}, n_peaks={len(self.members)}, "
            f"time=({self.start_time}-{self.end_time}))"
        )
