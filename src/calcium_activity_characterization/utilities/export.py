# population_metric_exporter.py
# Usage Example:
#         exporter = NormalizedDataExporter(population, output_dir=Path("Output/IS1/"))
#         exporter.export_all()
#         exporter.export_population_metrics()

import csv
import logging
from pathlib import Path
import json
import pickle
import numpy as np
from tqdm import tqdm

from calcium_activity_characterization.data.populations import Population

logger = logging.getLogger(__name__)



class NormalizedDataExporter:
    """
    Exports peaks, cells, events, and population metrics in a normalized, professional format.

    Attributes:
        population (Population): The analyzed population object.
        output_dir (Path): Directory to save exported files.
    """

    def __init__(self, population: Population, output_dir: Path) -> None:
        self.population = population
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cell_trace_dir = self.output_dir / "cell_traces"
        self.event_detail_dir = self.output_dir / "event_details"
        self.cell_trace_dir.mkdir(exist_ok=True)
        self.event_detail_dir.mkdir(exist_ok=True)

    def export_all(self) -> None:
        self.export_peaks()
        self.export_cells()
        self.export_events()

    def export_peaks(self) -> None:
        path = self.output_dir / "peaks.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "peak_id", "cell_id", "event_id", "start_time", "end_time", "duration",
                "rel_start_time", "rel_end_time", "rel_duration", "peak_time",
                "height", "prominence", "rel_height",
                "rise_time", "decay_time", "rel_rise_time", "rel_decay_time",
                "rel_symmetry_score", "scale_class", "in_event", "origin_type", "origin_label"
            ])
            writer.writeheader()
            for cell in tqdm(self.population.cells, desc="Exporting peaks", unit="cell"):
                for peak in cell.trace.peaks:
                    writer.writerow({
                        "peak_id": peak.id,
                        "cell_id": cell.label,
                        "event_id": peak.event_id,
                        "start_time": peak.start_time,
                        "end_time": peak.end_time,
                        "duration": peak.duration,
                        "rel_start_time": peak.rel_start_time,
                        "rel_end_time": peak.rel_end_time,
                        "rel_duration": peak.rel_duration,
                        "peak_time": peak.peak_time,
                        "height": peak.height,
                        "prominence": peak.prominence,
                        "rel_height": peak.rel_height,
                        "rise_time": peak.rise_time,
                        "decay_time": peak.decay_time,
                        "rel_rise_time": peak.rel_rise_time,
                        "rel_decay_time": peak.rel_decay_time,
                        "rel_symmetry_score": peak.rel_symmetry_score,
                        "scale_class": peak.scale_class,
                        "in_event": peak.in_event,
                        "origin_type": peak.origin_type,
                        "origin_label": peak.origin_label
                    })

    def export_cells(self) -> None:
        path = self.output_dir / "cells.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "cell_id", "centroid_x", "centroid_y", "is_valid",
                "num_peaks", "peak_frequency", "periodicity_score", "fraction_active_time"
            ])
            writer.writeheader()
            for cell in tqdm(self.population.cells, desc="Exporting cells", unit="cell"):
                # Save trace arrays externally
                np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_raw.npy", cell.trace.versions["raw"])
                np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_smoothed.npy", cell.trace.versions["smoothed"])
                np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_binary.npy", cell.trace.binary)

                writer.writerow({
                    "cell_id": cell.label,
                    "centroid_x": int(cell.centroid[1]),
                    "centroid_y": int(cell.centroid[0]),
                    "is_valid": cell.is_valid,
                    "num_peaks": len(cell.trace.peaks),
                    "peak_frequency": cell.trace.metadata.get("peak_frequency", 0),
                    "periodicity_score": cell.trace.metadata.get("periodicity_score", 0),
                    "fraction_active_time": cell.trace.metadata.get("fraction_active_time", 0)
                })

    def export_events(self) -> None:
        path = self.output_dir / "events.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "event_id", "event_type", "event_start_time", "event_end_time", "event_duration",
                "n_cells_involved", "dominant_direction_vector", "directional_propagation_speed",
                "dag_n_nodes", "dag_n_edges", "dag_n_roots", "dag_depth", "dag_width",
                "dag_avg_out_degree", "dag_avg_path_length",
                "communication_speed_mean", "communication_speed_std",
                "communication_time_mean", "communication_time_std",
                "elongation_score", "compactness_score", "global_propagation_speed"
            ])
            writer.writeheader()
            for event in tqdm(self.population.events, desc="Exporting events", unit="event"):
                is_seq = event.__class__.__name__ == "SequentialEvent"
                # Save graph and comms externally
                if is_seq:
                    if hasattr(event, "graph"):
                        with open(self.event_detail_dir / f"event_{event.id:04d}_graph.pkl", "wb") as g:
                            pickle.dump(event.graph, g)
                    if hasattr(event, "communications"):
                        with open(self.event_detail_dir / f"event_{event.id:04d}_communications.pkl", "wb") as c:
                            pickle.dump(event.communications, c)

                writer.writerow({
                    "event_id": event.id,
                    "event_type": event.__class__.__name__,
                    "event_start_time": event.event_start_time,
                    "event_end_time": event.event_end_time,
                    "event_duration": event.event_duration,
                    "n_cells_involved": event.n_cells_involved,
                    "dominant_direction_vector": str(event.dominant_direction_vector),
                    "directional_propagation_speed": event.directional_propagation_speed,
                    "dag_n_nodes": getattr(event, "dag_n_nodes", None),
                    "dag_n_edges": getattr(event, "dag_n_edges", None),
                    "dag_n_roots": getattr(event, "dag_n_roots", None),
                    "dag_depth": getattr(event, "dag_depth", None),
                    "dag_width": getattr(event, "dag_width", None),
                    "dag_avg_out_degree": getattr(event, "dag_avg_out_degree", None),
                    "dag_avg_path_length": getattr(event, "dag_avg_path_length", None),
                    "communication_speed_mean": getattr(event, "communication_speed_mean", None),
                    "communication_speed_std": getattr(event, "communication_speed_std", None),
                    "communication_time_mean": getattr(event, "communication_time_mean", None),
                    "communication_time_std": getattr(event, "communication_time_std", None),
                    "elongation_score": getattr(event, "elongation_score", None),
                    "compactness_score": getattr(event, "compactness_score", None),
                    "global_propagation_speed": getattr(event, "global_propagation_speed", None)
                })

    def export_population_metrics(self) -> None:
        try:
            path = self.output_dir / "population_metrics.json"
            with open(path, "w") as f:
                json.dump(self.population.metadata, f, indent=4)
            logger.info(f"Saved population-level metrics to {path}")
        except Exception as e:
            logger.error(f"Failed to export population metrics: {e}")
