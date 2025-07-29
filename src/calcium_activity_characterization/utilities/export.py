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

    def __init__(self, population: Population, output_dir: Path, cut_trace_num_points: int) -> None:
        self.population = population
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cell_trace_dir = self.output_dir / "cell_traces"
        self.event_detail_dir = self.output_dir / "event_details"
        self.cut_trace_num_points = cut_trace_num_points
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
                "fhw_start_time", "fhw_end_time", "fhw_duration", "peak_time", 
                "activation_start_time", "activation_end_time", "activation_duration", "communication_time",
                "height", "prominence", "fhw_height",
                "rise_time", "decay_time", "fhw_rise_time", "fhw_decay_time",
                "fhw_symmetry_score", "in_event", "origin_type"
            ])
            writer.writeheader()
            for cell in tqdm(self.population.cells, desc="Exporting peaks", unit="cell"):
                for peak in cell.trace.peaks:
                    writer.writerow({
                        "peak_id": peak.id,
                        "cell_id": cell.label,
                        "event_id": peak.event_id,
                        "start_time": peak.start_time + self.cut_trace_num_points,
                        "end_time": peak.end_time + self.cut_trace_num_points,
                        "duration": peak.duration,
                        "fhw_start_time": peak.fhw_start_time + self.cut_trace_num_points,
                        "fhw_end_time": peak.fhw_end_time + self.cut_trace_num_points,
                        "fhw_duration": peak.fhw_duration,
                        "peak_time": peak.peak_time + self.cut_trace_num_points,
                        "activation_start_time": peak.activation_start_time + self.cut_trace_num_points,
                        "activation_end_time": peak.activation_end_time + self.cut_trace_num_points,
                        "activation_duration": peak.activation_duration,
                        "communication_time": peak.communication_time + self.cut_trace_num_points,
                        "height": peak.height,
                        "prominence": peak.prominence,
                        "fhw_height": peak.fhw_height,
                        "rise_time": peak.rise_time,
                        "decay_time": peak.decay_time,
                        "fhw_rise_time": peak.fhw_rise_time,
                        "fhw_decay_time": peak.fhw_decay_time,
                        "fhw_symmetry_score": peak.fhw_symmetry_score,
                        "in_event": peak.in_event,
                        "origin_type": peak.origin_type
                    })

    def export_cells(self) -> None:
        path = self.output_dir / "cells.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "cell_id", "centroid_x", "centroid_y",
                "num_peaks", "is_active", "peak_frequency", "periodicity_score"
            ])
            writer.writeheader()
            for cell in tqdm(self.population.cells, desc="Exporting cells", unit="cell"):
                # Save trace arrays externally
                #np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_raw.npy", cell.trace.versions["raw"])
                #np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_smoothed.npy", cell.trace.versions["processed"])
                #np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_binary.npy", cell.trace.binary)

                writer.writerow({
                    "cell_id": cell.label,
                    "centroid_x": int(cell.centroid[1]),
                    "centroid_y": int(cell.centroid[0]),
                    "num_peaks": len(cell.trace.peaks),
                    "is_active": cell.is_active,
                    "peak_frequency": cell.trace.metadata.get("peak_frequency", 0),
                    "periodicity_score": cell.trace.metadata.get("periodicity_score", 0),
                })

    def export_events(self) -> None:
        path = self.output_dir / "events.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "event_id", "event_type", "event_start_time", "event_end_time", 
                "event_duration", "event_peak_time",
                "n_cells_involved", "dominant_direction_vector", "directional_propagation_speed",
                "growth_curve_mean", "growth_curve_std",
                "time_to_50", "peak_rate_at_50",
                "dag_n_nodes", "dag_n_edges", "dag_n_roots", "dag_depth", "dag_width",
                "dag_avg_out_degree", "dag_avg_path_length",
                "communication_speed_mean", "communication_speed_std",
                "communication_time_mean", "communication_time_std",
                "elongation_score", "radiality_score", "compactness_score"
            ])
            writer.writeheader()
            for event in tqdm(self.population.events, desc="Exporting events", unit="event"):
                is_seq = event.__class__.__name__ == "SequentialEvent"
                is_global = event.__class__.__name__ == "GlobalEvent"
                # Save graph and comms externally
                """
                if is_seq:
                    if hasattr(event, "graph"):
                        with open(self.event_detail_dir / f"event_{event.id:04d}_graph.pkl", "wb") as g:
                            pickle.dump(event.graph, g)
                    if hasattr(event, "communications"):
                        with open(self.event_detail_dir / f"event_{event.id:04d}_communications.pkl", "wb") as c:
                            pickle.dump(event.communications, c)
                """
                
                writer.writerow({
                    "event_id": event.id,
                    "event_type": event.__class__.__name__,
                    "event_start_time": event.event_start_time + self.cut_trace_num_points,
                    "event_end_time": event.event_end_time + self.cut_trace_num_points,
                    "event_duration": event.event_duration,
                    "event_peak_time": event.event_peak_time + self.cut_trace_num_points if is_global else None,
                    "n_cells_involved": event.n_cells_involved,
                    "dominant_direction_vector": str(event.dominant_direction_vector),
                    "directional_propagation_speed": event.directional_propagation_speed,
                    "growth_curve_mean": event.growth_curve_mean,
                    "growth_curve_std": event.growth_curve_std,
                    "time_to_50": event.time_to_50 if is_global else None,
                    "peak_rate_at_50": event.peak_rate_at_50 if is_global else None,
                    "dag_n_nodes": event.dag_metrics["n_nodes"] if is_seq else None,
                    "dag_n_edges": event.dag_metrics["n_edges"] if is_seq else None,
                    "dag_n_roots": event.dag_metrics["n_roots"] if is_seq else None,
                    "dag_depth": event.dag_metrics["depth"] if is_seq else None,
                    "dag_width": event.dag_metrics["width"] if is_seq else None,
                    "dag_avg_out_degree": event.dag_metrics["avg_out_degree"] if is_seq else None,
                    "dag_avg_path_length": event.dag_metrics["avg_path_length"] if is_seq else None,
                    "communication_speed_mean": event.communication_speed_mean if is_seq else None,
                    "communication_speed_std": event.communication_speed_std if is_seq else None,
                    "communication_time_mean": event.communication_time_mean if is_seq else None,
                    "communication_time_std": event.communication_time_std if is_seq else None,
                    "elongation_score": event.elongation_score if is_seq else None,
                    "radiality_score": event.radiality_score if is_seq else None,
                    "compactness_score": event.compactness_score if is_seq else None
                })

    def export_population_metrics(self) -> None:
        try:
            path = self.output_dir / "population_metrics.json"
            with open(path, "w") as f:
                json.dump(self.population.metadata, f, indent=4)
            logger.info(f"Saved population-level metrics to {path}")
        except Exception as e:
            logger.error(f"Failed to export population metrics: {e}")
