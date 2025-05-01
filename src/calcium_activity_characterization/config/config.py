# ==========================
# FILE PATTERNS AND PADDING
# ==========================
PADDING = 5  # Filename zero-padding digits

# ==========================
# IMAGE PROCESSING PARAMETERS
# ==========================
ROI_SCALE = 0.75  # Scale for ROI cropping (e.g., 0.75 = 75%)
SMALL_OBJECT_THRESHOLD = 200  # Minimum pixel count for valid cell
SAMPLING_FREQ = 1.0  # Sampling frequency (Hz)
BTYPE = 'highpass'  # Filter type
DETRENDING_MODE = 'butterworth'  # Detrending mode ('wavelet', 'butterworth', 'fir', 'exponentialfit', 'diff', 'savgol', 'movingaverage')


# ==========================
# SIGNAL PROCESSING PARAMETERS
# ==========================

SIGNAL_PROCESSING_PARAMETERS = {
    "wavelet": {
        "wavelet": "db4",           # Name of the wavelet ('db4', 'sym4', etc.)
        "level": None,              # Decomposition level (None = max possible)
        "sigma": 2.0,               # Gaussian smoothing after detrending
        "normalize_method": "deltaf"  # Normalization type
    },

    "fir": {
        "cutoff": 0.001,            # High-pass cutoff (normalized to Nyquist)
        "numtaps": 201,             # Number of filter taps (must be odd)
        "sigma": 2.0,
        "normalize_method": "deltaf"
    },

    "butterworth": {
        "cutoff": 0.003,            # High-pass cutoff frequency (Hz)
        "order": 6,                 # Filter order
        "mode": "ba",              # 'sos' or 'ba'
        "sigma": 6.0,
        "normalize_method": "deltaf"
    },

    "exponentialfit": {
        # No specific parameters needed
        "sigma": 2.0,
        "normalize_method": "deltaf"
    },

    "diff": {
        # No specific parameters needed
        "sigma": 2.0,
        "normalize_method": "deltaf"
    },

    "savgol": {
        "presmoothing_sigma": 6.0,   # Sigma for Gaussian pre-smoothing
        "window_length": 601,        # Must be odd
        "polyorder": 2,              # Polynomial order (degree)
        "sigma": 4.0,
        "normalize_method": "none"
    },

    "movingaverage": {
        "window_size": 101,          # Moving average window size
        "sigma": 2.0,
        "normalize_method": "deltaf"
    }
}




# ==========================
# FLAGS
# ==========================
EXISTING_CELLS = True  # Load precomputed cells from file
EXISTING_MASK = True  # Load precomputed mask from file
EXISTING_PROCESSED_INTENSITY = True  # Load intensity traces if available
EXISTING_RAW_INTENSITY = True  # Load raw intensity traces if available
SAVE_OVERLAY = True  # Save segmentation overlay
PARALLELELIZE = True  # Use parallel processing for intensity extraction


HARDDRIVE_PATH = "D:/Mateo"



# ==========================
# CONFIGURATION ARCOS PARAMETERS
# ==========================

BINDATA_PARAMETERS = {
    "smooth_k": 3,
    "bias_k": 51,
    "peak_threshold": 0.2,
    "binarization_threshold": 0.1,
    "polynomial_degree": 1,
    "bias_method": "runmed",  # can be 'lm', 'runmed', or 'none'
    "n_jobs": -1
}


TRACKING_PARAMETERS = {
    "position_columns": ["x", "y"],       # Columns indicating cell centroid
    "frame_column": "frame",              # Timepoint column
    "id_column": "trackID",               # Unique cell ID
    "binarized_measurement_column": "intensity.bin",  # Binary activation column
    "clid_column": "event_id",            # Event ID output column

    # ARCOS-specific parameters
    "eps": 10.0,               # Spatial proximity threshold
    "min_clustersize": 3,     # Minimum number of cells to form an event
    "allow_merges": True,     # Allow events to merge
    "allow_splits": True,     # Allow events to split
    "stability_threshold": 5, # Minimum duration (frames) for event stability
    "linking_method": "nearest"  # Method to associate objects across frames
}