from pathlib import Path

# Paths
DATA_DIR = Path("D:/Mateo/20250326/Data/IS1")
OUTPUT_DIR = Path("D:/Mateo/20250326/Output/IS1")

# Padding for filenames
FITC_FILE_PATTERN = r"20250326__w3FITC_t(\d+).TIF"
HOECHST_FILE_PATTERN = r"20250326__w2DAPI_t(\d+).TIF"
PADDING = 5  # Number of digits to pad the numbers to

ROI_SCALE = 0.75  # Scale for ROI cropping (e.g., 0.75 means 75% of the original size)

SMALL_OBJECT_THRESHOLD = 20  # Minimum pixel count for a cell to be considered valid