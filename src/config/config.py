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
PADDING = 5  # Filename zero-padding digits

# ==========================
# IMAGE PROCESSING PARAMETERS
# ==========================
ROI_SCALE = 0.75  # Scale for ROI cropping (e.g., 0.75 = 75%)
SMALL_OBJECT_THRESHOLD = 200  # Minimum pixel count for valid cell
GAUSSIAN_SIGMA = 0.5  # Sigma for Gaussian filter
HPF_CUTOFF = 0.001  # High-pass filter cutoff frequency (Hz)
SAMPLING_FREQ = 1.0  # Sampling frequency (Hz)
ORDER = 9  # Filter order
BTYPE = 'highpass'  # Filter type

# ==========================
# FLAGS
# ==========================
EXISTING_CELLS = True  # Load precomputed cells from file
EXISTING_MASK = True  # Load precomputed mask from file
EXISTING_INTENSITY_PROFILE = True  # Load intensity traces if available
SAVE_OVERLAY = True  # Save segmentation overlay
PARALLELELIZE = True  # Use parallel processing for intensity extraction
