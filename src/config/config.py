from pathlib import Path

# ==========================
# PATH CONFIGURATION
# ==========================
DATA_DIR = Path("D:/Mateo/20250326/Data/IS1")  # Directory containing input data
OUTPUT_DIR = Path("D:/Mateo/20250326/Output/IS1")  # Directory for output files

# Input image directories
HOECHST_IMG_PATH = DATA_DIR / "HOECHST"
FITC_IMG_PATH = DATA_DIR / "FITC"

# Output files
ACTIVE_CELLS_FILE_PATH = OUTPUT_DIR / "active_cells.pkl"
CELLS_FILE_PATH = OUTPUT_DIR / "cells.pkl"
NUCLEI_MASK_PATH = OUTPUT_DIR / "nuclei_mask.TIF"
OVERLAY_PATH = OUTPUT_DIR / "overlay.TIF"
TEMP_OVERLAY_PATH = OUTPUT_DIR / "temp_overlay.TIF"

# ==========================
# FILE PATTERNS AND PADDING
# ==========================
FITC_FILE_PATTERN = r"20250326__w3FITC_t(\d+).TIF"
HOECHST_FILE_PATTERN = r"20250326__w2DAPI_t(\d+).TIF"
PADDING = 5  # Filename zero-padding digits

# ==========================
# IMAGE PROCESSING PARAMETERS
# ==========================
ROI_SCALE = 0.75  # Scale for ROI cropping (e.g., 0.75 = 75%)
SMALL_OBJECT_THRESHOLD = 200  # Minimum pixel count for valid cell

# ==========================
# FLAGS
# ==========================
EXISTING_CELLS = True  # Load precomputed cells from file
EXISTING_MASK = True  # Load precomputed mask from file
EXISTING_INTENSITY_PROFILE = True  # Load intensity traces if available
SAVE_OVERLAY = True  # Save segmentation overlay
PARALLELELIZE = True  # Use parallel processing for intensity extraction
