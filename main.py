from src.io.loader import load_existing_cells, load_existing_img, save_pickle_file
from src.config.config import ROI_SCALE, FITC_FILE_PATTERN, HOECHST_FILE_PATTERN, PADDING, CELLS_FILE_PATH, CELLS_TEMP_FILE_PATH, NUCLEI_MASK_PATH, OVERLAY_PATH, EXISTING_CELLS, EXISTING_MASK, SAVE_OVERLAY, HOECHST_IMG_PATH, FITC_IMG_PATH, PARALLELELIZE, EXISTING_INTENSITY_PROFILE
from src.core.pipeline import cells_segmentation
from src.core.pipeline import convert_mask_to_cells, get_cells_intensity_profiles, get_cells_intensity_profiles_parallelized

import numpy as np
import time
from pathlib import Path



if __name__ == "__main__":


    start = time.time()
    print("[INFO] Starting pipeline...")
    
    # Load cells if previously saved and EXISTING_CELLS is True
    if not EXISTING_CELLS or not CELLS_TEMP_FILE_PATH.exists():
        
        # Load nuclei mask if it exists and EXISTING_MASK is True
        if not EXISTING_MASK or not NUCLEI_MASK_PATH.exists():
            nuclei_mask = cells_segmentation(HOECHST_IMG_PATH, ROI_SCALE, HOECHST_FILE_PATTERN, PADDING, OVERLAY_PATH, SAVE_OVERLAY, NUCLEI_MASK_PATH)
        else:
            nuclei_mask = load_existing_img(NUCLEI_MASK_PATH)

        # Convert from labeled mask to list of Cell objects
        cells = convert_mask_to_cells(nuclei_mask)
        save_pickle_file(cells, CELLS_TEMP_FILE_PATH)
    else:
        cells = load_existing_cells(CELLS_TEMP_FILE_PATH, EXISTING_CELLS)

    print(f"[INFO] Number of cells detected: {len(cells)}")

    
    # Determine active vs inactive cells
    active_cells = [cell for cell in cells if cell.is_valid]

    print(f"[INFO] Number of active cells: {len(active_cells)}")
    print(f"[INFO] Number of inactive cells: {len(cells) - len(active_cells)}")

    if not EXISTING_INTENSITY_PROFILE or not CELLS_FILE_PATH.exists():
        # Load FITC images (if any) and add timepoints to each cell
        if not PARALLELELIZE:
            get_cells_intensity_profiles(cells, FITC_IMG_PATH, ROI_SCALE, FITC_FILE_PATTERN, PADDING)
        else:
            print("[DEBUG] Running in parallelized mode for intensity profile computation...")
            get_cells_intensity_profiles_parallelized(cells, FITC_IMG_PATH, FITC_FILE_PATTERN, PADDING, ROI_SCALE)

        # Save the updated cells with intensity profiles
        save_pickle_file(cells, CELLS_FILE_PATH)
    else:
        cells = load_existing_cells(CELLS_FILE_PATH, EXISTING_CELLS)


    print("[INFO] Pipeline completed successfully.")
    end = time.time()
    print(f"[INFO] Total time taken: {end - start:.2f} seconds")