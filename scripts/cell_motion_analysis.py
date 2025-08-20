"""
cell_motion.py

Usage:
    Run this script to segment and overlay two Hoechst images per ISX folder for motion analysis.

    $ python cell_motion.py
"""

import sys
import csv
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage.segmentation import find_boundaries

import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

from calcium_activity_characterization.config.presets import GLOBAL_CONFIG
from calcium_activity_characterization.config.structures import ImageProcessingConfig, SegmentationConfig
from calcium_activity_characterization.preprocessing.image_processing import ImageProcessor
from calcium_activity_characterization.preprocessing.segmentation import segmented
from calcium_activity_characterization.io.export import save_tif_image
from calcium_activity_characterization.utilities.plotter import (
    plot_spatial_neighbor_graph
)
from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.populations import Population

def main() -> None:
    """
    Select one or more folders (date folders or ISX) and run segmentation for Hoechst image pairs.
    """
    app = QApplication(sys.argv)
    if GLOBAL_CONFIG.debug.debugging:
        print("[DEBUGGING MODE] Using test folder from config.")
        selected = [Path(GLOBAL_CONFIG.debug.debugging_file_path)]
    else:
        folder_dialog = QFileDialog()
        folder_dialog.setDirectory(str(GLOBAL_CONFIG.debug.harddrive_path))
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setWindowTitle("Select One or More Folders (Date folders or ISX)")

        view = folder_dialog.findChild(QWidget, "listView")
        if view:
            view.setSelectionMode(view.ExtendedSelection)
        f_tree_view = folder_dialog.findChild(QWidget, "treeView")
        if f_tree_view:
            f_tree_view.setSelectionMode(f_tree_view.ExtendedSelection)

        if not folder_dialog.exec_():
            print("No folder selected. Exiting.")
            return

        selected = [Path(folder_str) for folder_str in folder_dialog.selectedFiles()]

    all_isx_folders = []
    for folder in selected:
        if folder.name.startswith("IS"):
            all_isx_folders.append(folder)
        else:
            all_isx_folders.extend(find_isx_folders(folder))

    if not all_isx_folders:
        print("No ISX folders found in selected path(s). Exiting.")
        return

    for isx_path in sorted(all_isx_folders):
        print(f"Processing {isx_path}...")
        process_isx_folder(isx_path)

def find_isx_folders(folder: Path) -> list[Path]:
    """
    Recursively find all ISX folders under 'Data/' directories.
    Skips any folder paths that include 'Output'.
    """
    isx_folders = []
    for subpath in folder.rglob("*"):
        if "Output" in subpath.parts:
            continue  # skip anything in Output
        if subpath.is_dir() and subpath.name.startswith("IS"):
            isx_folders.append(subpath)
    return isx_folders

def load_config_from_output(isx_path: Path, filename: str, fallback: object) -> object:
    config_path = isx_path.parents[1] / "Output" / isx_path.name / "cell-mapping" / filename
    if config_path.exists():
        if filename == "hoechst_image_processing_config.json":
            return ImageProcessingConfig.from_json(config_path)
        elif filename == "segmentation_config.json":
            return SegmentationConfig.from_json(config_path)
    return fallback

def process_isx_folder(isx_path: Path) -> None:
    hoechst_dir = isx_path / "HOECHST"
    output_dir = isx_path.parents[1] / "Output" / isx_path.name / "cell-motion"
    output_dir.mkdir(parents=True, exist_ok=True)

    hoechst_images = sorted(hoechst_dir.glob("*.TIF"))
    if len(hoechst_images) < 2:
        print(f"[WARN] Less than 2 Hoechst images found in {hoechst_dir}")
        return

    img_t0, img_t1 = hoechst_images[:2]

    img_cfg: ImageProcessingConfig = load_config_from_output(isx_path, "hoechst_image_processing_config.json", GLOBAL_CONFIG.image_processing_hoechst)
    seg_cfg: SegmentationConfig = load_config_from_output(isx_path, "segmentation_config.json", GLOBAL_CONFIG.segmentation)

    segment_hoechst_pair(img_t0, img_t1, output_dir, img_cfg, seg_cfg)

def segment_hoechst_pair(
    path_t0: Path,
    path_t1: Path,
    output_dir: Path,
    img_cfg: ImageProcessingConfig,
    seg_cfg: SegmentationConfig
) -> None:
    processor = ImageProcessor(img_cfg)

    proc0 = processor.process_single_image(path_t0)
    proc1 = processor.process_single_image(path_t1)

    mask0 = segmented(proc0[np.newaxis, ...], seg_cfg)
    mask1 = segmented(proc1[np.newaxis, ...], seg_cfg)

    save_tif_image(file_path=output_dir / "nuclei_mask_full_t0.TIF", image=mask0)
    save_tif_image(file_path=output_dir / "nuclei_mask_full_t1.TIF", image=mask1)

    unfiltered_cells0 = Cell.from_segmentation_mask(mask0, GLOBAL_CONFIG.cell_filtering)
    cells0 = [cell for cell in unfiltered_cells0 if cell.is_valid]
    graph0 = Population.build_spatial_neighbor_graph(cells0)
    cropped_mask0 = processor._crop_image(mask0)
    save_tif_image(cropped_mask0, output_dir / "nuclei_mask_crop_t0.TIF")
    population0 = Population.from_roi_filtered(
        nuclei_mask=cropped_mask0,
        cells=cells0,
        graph=graph0,
        roi_scale=img_cfg.roi_scale,
        img_shape=mask0.shape,
        border_margin=GLOBAL_CONFIG.cell_filtering.border_margin
    )

    unfiltered_cells1 = Cell.from_segmentation_mask(mask1, GLOBAL_CONFIG.cell_filtering)
    cells1 = [cell for cell in unfiltered_cells1 if cell.is_valid]
    graph1 = Population.build_spatial_neighbor_graph(cells1)
    cropped_mask1 = processor._crop_image(mask1)
    save_tif_image(cropped_mask1, output_dir / "nuclei_mask_crop_t1.TIF")
    population1 = Population.from_roi_filtered(
        nuclei_mask=cropped_mask1,
        cells=cells1,
        graph=graph1,
        roi_scale=img_cfg.roi_scale,
        img_shape=mask1.shape,
        border_margin=GLOBAL_CONFIG.cell_filtering.border_margin
    )

    processor.config.pipeline.cropping = True
    cropped_hoechst0 = processor.process_single_image(path_t0)
    cropped_hoechst1 = processor.process_single_image(path_t1)

    pipeline = CalciumPipeline(GLOBAL_CONFIG)
    pipeline.save_cell_outline_overlay(output_dir / "cell_outline_overlay_t0.TIF", population0.cells, cropped_hoechst0)
    plot_spatial_neighbor_graph(population0.neighbor_graph, cropped_hoechst0, output_dir / "spatial_graph_t0.png")

    pipeline.save_cell_outline_overlay(output_dir / "cell_outline_overlay_t1.TIF", population1.cells, cropped_hoechst1)
    plot_spatial_neighbor_graph(population1.neighbor_graph, cropped_hoechst1, output_dir / "spatial_graph_t1.png")

    compute_motion_metrics(mask0, mask1, population0, population1, output_dir)
    create_cellmotion_overlay_from_cells(population0.cells, population1.cells, cropped_mask0.shape, output_dir / "cellmotion_comparison_overlay.png")

def compute_motion_metrics(
    mask0: np.ndarray,
    mask1: np.ndarray,
    pop0: Population,
    pop1: Population,
    output_dir: Path
) -> None:
    n0 = len(pop0.cells)
    n1 = len(pop1.cells)
    abs_diff = abs(n0 - n1)
    mean_n = (n0 + n1) / 2
    rel_diff = abs_diff / mean_n if mean_n > 0 else 0.0

    shape = mask0.shape
    total_pixels = shape[0] * shape[1]

    mask_t0 = np.zeros(shape, dtype=bool)
    mask_t1 = np.zeros(shape, dtype=bool)

    for cell in pop0.cells:
        mask_t0[tuple(cell.pixel_coords.T)] = True

    for cell in pop1.cells:
        mask_t1[tuple(cell.pixel_coords.T)] = True

    pixels_t0 = np.count_nonzero(mask_t0)
    pixels_t1 = np.count_nonzero(mask_t1)
    pixels_both = np.count_nonzero(mask_t0 & mask_t1)
    pixels_only_t0 = np.count_nonzero(mask_t0 & ~mask_t1)
    pixels_only_t1 = np.count_nonzero(mask_t1 & ~mask_t0)

    pixels_both_rel = pixels_both*2 / (pixels_t0 + pixels_t1)
    pixels_only_t0_rel = pixels_only_t0 / pixels_t0
    pixels_only_t1_rel = pixels_only_t1 / pixels_t1

    metrics = [
        ["n_cells_t0", n0],
        ["n_cells_t1", n1],
        ["cell_count_diff_abs", abs_diff],
        ["cell_count_diff_rel", round(rel_diff, 5)],
        ["total_pixels", total_pixels],
        ["pixels_t0", pixels_t0],
        ["pixels_t1", pixels_t1],
        ["pixels_both", pixels_both],
        ["pixels_both_rel", round(pixels_both_rel, 5)],
        ["pixels_only_t0", pixels_only_t0],
        ["pixels_only_t1", pixels_only_t1],
        ["pixels_only_t0_rel", round(pixels_only_t0_rel, 5)],
        ["pixels_only_t1_rel", round(pixels_only_t1_rel, 5)]
    ]

    csv_path = output_dir / "metrics_cellmotion.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerows(metrics)


def create_cellmotion_overlay_from_cells(
    cells_t0: list[Cell],
    cells_t1: list[Cell],
    image_shape: tuple[int, int],
    output_path: Path
) -> None:
    """
    Creates a visual overlay comparison using valid Cell objects from two timepoints.

    Args:
        cells_t0 (list[Cell]): Valid cells from timepoint 0.
        cells_t1 (list[Cell]): Valid cells from timepoint 1.
        image_shape (tuple): Shape of the image (H, W).
        output_path (Path): Path to save the comparison overlay.
    """
    canvas = np.ones((*image_shape, 3), dtype=np.uint8) * 255

    mask0 = np.zeros(image_shape, dtype=np.uint16)
    mask1 = np.zeros(image_shape, dtype=np.uint16)

    for cell in cells_t0:
        mask0[tuple(cell.pixel_coords.T)] = cell.label

    for cell in cells_t1:
        mask1[tuple(cell.pixel_coords.T)] = cell.label

    binary0 = mask0 > 0
    binary1 = mask1 > 0
    both = binary0 & binary1
    only0 = binary0 & ~binary1

    # Fill pixels
    canvas[only0] = [128, 128, 128]  # Gray for t=0 only
    canvas[both] = [128, 128, 128]   # Gray for overlapping

    # Add dark gray contours for t0
    boundary0 = find_boundaries(mask0, mode="inner")
    canvas[boundary0] = [80, 80, 80]

    fig, ax = plt.subplots(figsize=(image_shape[1] / 100, image_shape[0] / 100), dpi=300)
    ax.imshow(canvas)

    # Efficient mask for overlapping regions (combined before loops)
    mask_t0 = np.zeros(image_shape, dtype=bool)
    mask_t1 = np.zeros(image_shape, dtype=bool)
    for cell in cells_t0:
        mask_t0[tuple(cell.pixel_coords.T)] = True
    for cell in cells_t1:
        mask_t1[tuple(cell.pixel_coords.T)] = True

    overlap_mask = mask_t0 & mask_t1

    if np.any(overlap_mask):
        from skimage.measure import find_contours
        overlap_contours = find_contours(overlap_mask.astype(float), 0.5)
        for oc in overlap_contours:
            verts = np.flip(oc, axis=1)
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path
            path = Path(verts)
            patch = PathPatch(
                path,
                edgecolor="red",
                facecolor=[0.5, 0.5, 0.5],
                hatch="///",
                linewidth=0.5,
                zorder=3,
                label="Overlap"
            )
            ax.add_patch(patch)

    # Draw red dashed outlines (fixed to avoid doubles)
    drawn_legend = False
    for cell in cells_t1:
        mask = np.zeros(image_shape, dtype=bool)
        mask[tuple(cell.pixel_coords.T)] = True
        contours = find_contours(mask.astype(float), level=0.5)
        for contour in contours:
            polygon = Polygon(
                np.flip(contour, axis=1),
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.0,
                zorder=4,
                label="Cell t=1801" if not drawn_legend else None
            )
            ax.add_patch(polygon)

    ax.axis("off")
    plt.savefig(output_path, dpi=300, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()



if __name__ == "__main__":
    main()
