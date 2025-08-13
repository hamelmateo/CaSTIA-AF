# population_metric_exporter.py
# Usage Example:
#         exporter = NormalizedDataExporter(population, output_dir=Path("Output/IS1/"))
#         exporter.export_all()
#         exporter.export_population_metrics()

import os
import csv
from pathlib import Path
import pickle
import json
import numpy as np
from tqdm import tqdm
import tifffile
from PIL import Image
from matplotlib import pyplot as plt

from calcium_activity_characterization.data.populations import Population
from calcium_activity_characterization.config.structures import ExportConfig

from calcium_activity_characterization.logger import logger



class NormalizedDataExporter:
    """
    Exports peaks, cells, events, and population metrics in a normalized, professional format.

    Attributes:
        population (Population): The analyzed population object.
        output_dir (Path): Directory to save exported files.
    """

    def __init__(self, population: Population, output_dir: Path, config: ExportConfig, cut_trace_num_points: int) -> None:
        self.population = population
        self.output_dir = output_dir
        self.pixel_to_micron_x = config.spatial_calibration_params.pixel_to_micron_x
        self.pixel_to_micron_y = config.spatial_calibration_params.pixel_to_micron_y
        self.frame_rate = config.frame_rate
        self.pixel_per_frame_to_micron_per_second = self.pixel_to_micron_x * self.frame_rate
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
        self.export_communications()

    def export_peaks(self) -> None:
        path = self.output_dir / "peaks.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Peak ID", "Cell ID", "Event ID", "Baseline onset time (s)", "Baseline offset time (s)", "Baseline duration (s)",
                "FHW onset time (s)", "FHW offset time (s)", "FHW duration (s)", "Peak time (s)", 
                "Onset time (s)", "Offset time (s)", "Duration (s)", "Event reference time (s)",
                "Height (noise std units)", "Prominence (noise std units)", "FHW height (noise std units)",
                "Baseline rise time (s)", "Baseline decay time (s)", "FHW rise time (s)", "FHW decay time (s)",
                "FHW symmetry score", "In event", "Origin type"
            ])
            writer.writeheader()
            for cell in tqdm(self.population.cells, desc="Exporting peaks", unit="cell"):
                for peak in cell.trace.peaks:
                    writer.writerow({
                        "Peak ID": int(peak.id),
                        "Cell ID": int(cell.label),
                        "Event ID": int(peak.event_id) if peak.event_id is not None else None,
                        "Baseline onset time (s)": format((peak.start_time + self.cut_trace_num_points)/self.frame_rate, '.1f'),
                        "Baseline offset time (s)": format((peak.end_time + self.cut_trace_num_points)/self.frame_rate, '.1f'),
                        "Baseline duration (s)": format(peak.duration/self.frame_rate, '.1f'),
                        "FHW onset time (s)": format((peak.fhw_start_time + self.cut_trace_num_points)/self.frame_rate, '.1f'),
                        "FHW offset time (s)": format((peak.fhw_end_time + self.cut_trace_num_points)/self.frame_rate, '.1f'),
                        "FHW duration (s)": format(peak.fhw_duration/self.frame_rate, '.1f'),
                        "Peak time (s)": format(peak.peak_time/self.frame_rate, '.1f'),
                        "Onset time (s)": format(peak.activation_start_time/self.frame_rate, '.1f'),
                        "Offset time (s)": format(peak.activation_end_time/self.frame_rate, '.1f'),
                        "Duration (s)": format(peak.activation_duration/self.frame_rate, '.1f'),
                        "Event reference time (s)": format(peak.communication_time/self.frame_rate, '.1f'),
                        "Height (noise std units)": round(peak.height, 1),
                        "Prominence (noise std units)": round(peak.prominence, 1),
                        "FHW height (noise std units)": round(peak.fhw_height, 1),
                        "Baseline rise time (s)": format(peak.rise_time/self.frame_rate, '.1f'),
                        "Baseline decay time (s)": format(peak.decay_time/self.frame_rate, '.1f'),
                        "FHW rise time (s)": format(peak.fhw_rise_time/self.frame_rate, '.1f'),
                        "FHW decay time (s)": format(peak.fhw_decay_time/self.frame_rate, '.1f'),
                        "FHW symmetry score": round(peak.fhw_symmetry_score, 3),
                        "In event": peak.in_event,
                        "Origin type": peak.origin_type
                    })

    def export_cells(self) -> None:
        path = self.output_dir / "cells.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Cell ID", "Centroid X coordinate (um)", "Centroid Y coordinate (um)",
                "Number of peaks", "Is active", "Occurences in global events", "Occurences in global events as early peaker", 
                "Occurences in sequential events", "Occurences in sequential events as origin", 
                "Occurences in individual events", "Peak frequency (Hz)", "Periodicity score",
                "Neighbor count", "Neighbors (labels)"
            ])
            writer.writeheader()
            for cell in tqdm(self.population.cells, desc="Exporting cells", unit="cell"):
                # Save trace arrays externally
                #np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_raw.npy", cell.trace.versions["raw"])
                #np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_smoothed.npy", cell.trace.versions["processed"])
                #np.save(self.cell_trace_dir / f"cell_{cell.label:04d}_binary.npy", cell.trace.binary)

                # JSON-serialize list of ints; compact separators keep file small
                neighbors_json = json.dumps(
                    [int(n) for n in (cell.neighbors or [])],
                    ensure_ascii=False, separators=(",", ":")
                )

                writer.writerow({
                    "Cell ID": int(cell.label),
                    "Centroid X coordinate (um)": format(cell.centroid[1]*self.pixel_to_micron_x, '.2f'),
                    "Centroid Y coordinate (um)": format(cell.centroid[0]*self.pixel_to_micron_y, '.2f'),
                    "Number of peaks": len(cell.trace.peaks),
                    "Is active": bool(cell.is_active),
                    "Occurences in global events": int(cell.occurences_global_events),
                    "Occurences in global events as early peaker": int(cell.occurences_global_events_as_early_peaker),
                    "Occurences in sequential events": int(cell.occurences_sequential_events),
                    "Occurences in sequential events as origin": int(cell.occurences_sequential_events_as_origin),
                    "Occurences in individual events": int(cell.occurences_individual_events),
                    "Peak frequency (Hz)": format(cell.trace.metadata.get("peak_frequency", 0), '.2g'),
                    "Periodicity score": (format(score, '.2g') if (score := cell.trace.metadata.get("periodicity_score")) is not None else None),
                    "Neighbor count": int(len(cell.neighbors)),
                    "Neighbors (labels)": neighbors_json
                })

    def export_events(self) -> None:
        path = self.output_dir / "events.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Event ID", "Event type", "Event onset time (s)", "Event offset time (s)", 
                "Event duration (s)", "Event peak time (s)",
                "Number of cells involved", "Propagation direction vector", "Propagation speed (um/s)",
                "Growth curve mean", "Growth curve std",
                "Time to 50% (s)", "Normalized peak rate at 50% (% of peaks/s)",
                "DAG number of nodes", "DAG number of edges", "DAG number of roots", "DAG depth", "DAG width",
                "DAG average out degree", "DAG average path length",
                "Average communication speed (um/s)", "Standard deviation communication speed (um/s)",
                "Average communication time (s)", "Standard deviation communication time (s)",
                "Elongation score", "Radiality score", "Compactness score"
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
                    "Event ID": int(event.id),
                    "Event type": event.__class__.__name__,
                    "Event onset time (s)": format((event.event_start_time + self.cut_trace_num_points)/self.frame_rate, '.1f'),
                    "Event offset time (s)": format((event.event_end_time + self.cut_trace_num_points)/self.frame_rate, '.1f'),
                    "Event duration (s)": format(event.event_duration/self.frame_rate, '.1f'),
                    "Event peak time (s)": format((event.event_peak_time + self.cut_trace_num_points)/self.frame_rate, '.1f') if is_global else None,
                    "Number of cells involved": int(event.n_cells_involved),
                    "Propagation direction vector": str(event.dominant_direction_vector), #TODO convert to x and y
                    "Propagation speed (um/s)": format(event.directional_propagation_speed*self.pixel_per_frame_to_micron_per_second, '.2f') if is_global else None, # TODO convert units
                    "Growth curve mean": format(event.growth_curve_mean, '.2f'),
                    "Growth curve std": format(event.growth_curve_std, '.2f'),
                    "Time to 50% (s)": format(event.time_to_50/self.frame_rate, '.1f') if is_global else None,
                    "Normalized peak rate at 50% (% of peaks/s)": format(event.normalized_peak_rate_at_50*self.frame_rate, '.1f') if is_global else None,
                    "DAG number of nodes": int(event.dag_metrics["n_nodes"]) if is_seq else None,
                    "DAG number of edges": int(event.dag_metrics["n_edges"]) if is_seq else None,
                    "DAG number of roots": int(event.dag_metrics["n_roots"]) if is_seq else None,
                    "DAG depth": int(event.dag_metrics["depth"]) if is_seq else None,
                    "DAG width": int(event.dag_metrics["width"]) if is_seq else None,
                    "DAG average out degree": format(event.dag_metrics["avg_out_degree"], '.2f') if is_seq else None,
                    "DAG average path length": format(event.dag_metrics["avg_path_length"], '.2f') if is_seq else None,
                    "Average communication speed (um/s)": format(event.communication_speed_mean*self.pixel_per_frame_to_micron_per_second, '.2f') if is_seq else None, # TODO convert units
                    "Standard deviation communication speed (um/s)": format(event.communication_speed_std*self.pixel_per_frame_to_micron_per_second, '.2f') if is_seq else None, # TODO convert units
                    "Average communication time (s)": format(event.communication_time_mean/self.frame_rate, '.2f') if is_seq else None,
                    "Standard deviation communication time (s)": format(event.communication_time_std/self.frame_rate, '.2f') if is_seq else None,
                    "Elongation score": format(event.elongation_score, '.2f') if is_seq else None,
                    "Radiality score": format(event.radiality_score, '.2f') if is_seq else None,
                    "Compactness score": format(event.compactness_score, '.2f') if is_seq else None
                })

    def export_communications(self) -> None:
        path = self.output_dir / "communications.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Communication ID", "Event ID", "Origin cell ID", "Origin cell peak ID", "Cause cell ID", "Cause cell peak ID", "Start time (s)", "End time (s)",
                "Duration (s)", "Distance (um)", "Speed (um/s)", "Event time phase (fraction of event duration)", "Event recruitment phase (fraction of involved cells)"
            ])
            writer.writeheader()
            for event in tqdm(self.population.events, desc="Exporting communications", unit="communication"):
                if event.__class__.__name__ == "SequentialEvent":
                    for comm in event.communications:
                        writer.writerow({
                            "Communication ID": int(comm.id),
                            "Event ID": int(event.id),
                            "Origin cell ID": int(comm.origin[0]),
                            "Origin cell peak ID": int(comm.origin[1]),
                            "Cause cell ID": int(comm.cause[0]),
                            "Cause cell peak ID": int(comm.cause[1]),
                            "Start time (s)": format(comm.origin_start_time/self.frame_rate, '.1f'),
                            "End time (s)": format(comm.cause_start_time/self.frame_rate, '.1f'),
                            "Duration (s)": format((comm.duration)/self.frame_rate, '.1f'),
                            "Distance (um)": format(comm.distance * self.pixel_to_micron_x, '.2f'),
                            "Speed (um/s)": format(comm.speed * self.pixel_per_frame_to_micron_per_second, '.2f'),
                            "Event time phase (fraction of event duration)": format(comm.event_time_phase, '.2f') if comm.event_time_phase is not None else None,
                            "Event recruitment phase (fraction of involved cells)": format(comm.event_recruitment_phase, '.2f') if comm.event_recruitment_phase is not None else None
                        })

def save_tif_image(image: np.ndarray, file_path: str, photometric: str = "minisblack", imagej: bool = True) -> None:
    """
    Save a 2D image as a .tif file. Automatically creates the output directory if it does not exist.

    Args:
        image (np.ndarray): Image array to save.
        file_path (str): Full path (including filename) to save the image.
        photometric (str): Photometric interpretation for TIFF. Default is "minisblack".
        imagej (bool): Whether to save in ImageJ-compatible format. Default is True.
    """
    try:
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        tifffile.imwrite(file_path, image.astype(np.uint16), photometric=photometric, imagej=imagej)
        logger.info(f"Saved image to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save TIFF image to {file_path}: {e}")
        raise

def save_rgb_png_image(
    image: np.ndarray,
    filepath: Path | str
) -> None:
    """
    Save an RGB uint8 image as a PNG file.

    Args:
        image (np.ndarray): H×W×3 uint8 RGB array.
        filepath (Path | str): Path to the output PNG file.

    Raises:
        ValueError: If the image is not H×W×3 uint8.
        IOError: If saving the file fails.
    """
    try:
        # Validate image
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 RGB image of shape H×W×3, got shape {image.shape} and dtype {image.dtype}"
            )

        # Ensure output directory exists
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save using PIL
        img_pil = Image.fromarray(image, mode='RGB')
        img_pil.save(out_path, format='PNG')
        logger.info(f"Saved PNG image to {out_path}")

    except Exception as e:
        logger.error(f"Failed to save RGB PNG: {e}")
        raise

def save_rgb_tif_image(
    image: np.ndarray,
    filepath: Path | str,
    photometric: str = "rgb",
    imagej: bool = True
) -> None:
    """
    Save an RGB uint8 image as a TIFF file.

    Args:
        image (np.ndarray): H×W×3 uint8 RGB array.
        filepath (Path | str): Path to the output TIFF file (should end in .tif or .tiff).
        photometric (str): TIFF photometric interpretation (default: "rgb").
        imagej (bool): Whether to write in ImageJ-compatible TIFF format (default: True).

    Raises:
        ValueError: If the image is not H×W×3 uint8.
        IOError: If saving the file fails.
    """
    # Validate image
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 RGB image of shape H×W×3, got shape {image.shape} and dtype {image.dtype}"
        )

    # Ensure output directory exists
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write as TIFF
        tifffile.imwrite(
            str(out_path),
            image,
            photometric=photometric,
            imagej=imagej
        )
        logger.info(f"Saved TIFF image to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save RGB TIFF: {e}")
        raise

def save_pickle_file(obj: any, file_path: str) -> None:
    """
    Save an object to a pickle file. Creates directories if needed.

    Args:
        obj (any): The Python object to serialize.
        file_path (str): Full path to the pickle file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {file_path}: {e}")
        raise

def save_plot(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")