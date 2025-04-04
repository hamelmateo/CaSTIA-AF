import time
import logging
from typing import List

from src.io.loader import (
    load_existing_cells,
    load_existing_img,
    save_pickle_file,
)
from src.config.config import (
    ROI_SCALE,
    FITC_FILE_PATTERN,
    HOECHST_FILE_PATTERN,
    PADDING,
    ACTIVE_CELLS_FILE_PATH,
    CELLS_FILE_PATH,
    NUCLEI_MASK_PATH,
    OVERLAY_PATH,
    EXISTING_CELLS,
    EXISTING_MASK,
    SAVE_OVERLAY,
    HOECHST_IMG_PATH,
    FITC_IMG_PATH,
    PARALLELELIZE,
    EXISTING_INTENSITY_PROFILE,
)
from src.core.pipeline import (
    cells_segmentation,
    convert_mask_to_cells,
    get_cells_intensity_profiles,
    get_cells_intensity_profiles_parallelized,
)
from src.analysis.umap_analysis import (
    run_umap_on_cells,
    run_umap_with_clustering,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """
    Run the full calcium imaging analysis pipeline including:
    - Loading or segmenting nuclei
    - Extracting intensity traces
    - Performing UMAP projection and clustering
    """
    start = time.time()
    logger.info("Starting pipeline...")

    # Load cells if previously saved and EXISTING_CELLS is True
    if not EXISTING_CELLS or not CELLS_FILE_PATH.exists():
        # Load nuclei mask if it exists and EXISTING_MASK is True
        if not EXISTING_MASK or not NUCLEI_MASK_PATH.exists():
            try:
                nuclei_mask = cells_segmentation(
                    HOECHST_IMG_PATH,
                    ROI_SCALE,
                    HOECHST_FILE_PATTERN,
                    PADDING,
                    OVERLAY_PATH,
                    SAVE_OVERLAY,
                    NUCLEI_MASK_PATH,
                )
            except Exception as e:
                logger.error(f"Failed to perform segmentation: {e}")
                return
        else:
            try:
                nuclei_mask = load_existing_img(NUCLEI_MASK_PATH)
            except Exception as e:
                logger.error(f"Failed to load existing mask: {e}")
                return

        try:
            cells = convert_mask_to_cells(nuclei_mask)
            save_pickle_file(cells, CELLS_FILE_PATH)
        except Exception as e:
            logger.error(f"Error converting/saving cells: {e}")
            return
    else:
        cells = load_existing_cells(CELLS_FILE_PATH, EXISTING_CELLS)

    logger.info(f"Number of cells detected: {len(cells)}")

    active_cells = [cell for cell in cells if cell.is_valid]
    logger.info(f"Active cells: {len(active_cells)} / Total: {len(cells)}")

    if not EXISTING_INTENSITY_PROFILE or not ACTIVE_CELLS_FILE_PATH.exists():
        try:
            if not PARALLELELIZE:
                get_cells_intensity_profiles(
                    active_cells,
                    FITC_IMG_PATH,
                    ROI_SCALE,
                    FITC_FILE_PATTERN,
                    PADDING,
                )
            else:
                logger.debug("Running in parallelized mode for intensity profile computation...")
                get_cells_intensity_profiles_parallelized(
                    active_cells,
                    FITC_IMG_PATH,
                    FITC_FILE_PATTERN,
                    PADDING,
                    ROI_SCALE,
                )
            save_pickle_file(active_cells, ACTIVE_CELLS_FILE_PATH)
        except Exception as e:
            logger.error(f"Failed during intensity profiling: {e}")
            return
    else:
        active_cells = load_existing_cells(ACTIVE_CELLS_FILE_PATH, EXISTING_CELLS)

    # # logger.info("Running UMAP +
    # try:
    #     run_umap_with_clustering(
    #         active_cells,
    #         n_neighbors=5,
    #         min_dist=1,
    #         n_components=2,
    #         normalize=True,
    #         eps=0.5,
    #         min_samples=5,
    #     )
    # except Exception as e:
    #     logger.error(f"UMAP analysis failed: {e}")
    #     return

    logger.info("Pipeline completed successfully.")
    logger.info(f"Total time taken: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    run_pipeline()
