from pathlib import Path

# ==========================
# PATH CONFIGURATION
# ==========================
DATA_DIR = Path("D:/Mateo/20250326/Data/IS1")  # Directory containing input data
OUTPUT_DIR = Path("D:/Mateo/20250326/Output/IS1")  # Directory for output files
HOECHST_IMG_PATH = DATA_DIR / "HOECHST"  # Path to Hoechst images
FITC_IMG_PATH = DATA_DIR / "FITC"  # Path to FITC images
CELLS_FILE_PATH = OUTPUT_DIR / "cells.pkl"  # Path to save/load cells data
NUCLEI_MASK_PATH = OUTPUT_DIR / "nuclei_mask.TIF"  # Path to save/load nuclei mask
OVERLAY_PATH = OUTPUT_DIR / "overlay.TIF"  # Path to save overlay image

# ==========================
# FILE PATTERNS AND PADDING
# ==========================
FITC_FILE_PATTERN = r"20250326__w3FITC_t(\d+).TIF"  # Regex pattern for FITC images
HOECHST_FILE_PATTERN = r"20250326__w2DAPI_t(\d+).TIF"  # Regex pattern for Hoechst images
PADDING = 5  # Number of digits to pad filenames (e.g., t00001)

# ==========================
# IMAGE PROCESSING PARAMETERS
# ==========================
ROI_SCALE = 0.75  # Scale for ROI cropping (e.g., 0.75 means 75% of the original size)
SMALL_OBJECT_THRESHOLD = 20  # Minimum pixel count for a cell to be considered valid

# ==========================
# FLAGS
# ==========================
EXISTING_CELLS = True  # Whether to load existing cells from a file
EXISTING_MASK = True  # Whether to load an existing nuclei mask
SAVE_OVERLAY = True  # Whether to save the overlay image
PARALLELELIZE = True  # Whether to use parallel processing for loading images

