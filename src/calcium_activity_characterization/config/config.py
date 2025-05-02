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


# ==========================
# SIGNAL PROCESSING PARAMETERS
# ==========================

SIGNAL_PROCESSING = {
    "pipeline": "arcos",  # Options: 'arcos', 'custom'
    "apply": {
        "detrending": True,
        "smoothing": True,
        "normalization": True,
    },
    "detrending_mode": "butterworth",  # Used only if pipeline == 'custom' - 'butterworth', 'wavelet', 'fir', 'exponentialfit', 'diff', 'savgol', 'movingaverage'
    "normalizing_method": "deltaf",  # Used only if pipeline == 'custom' - 'deltaf', 'zscore', 'minmax', 'percentile'
}

# TODO: Normalization parameters dictionary

SIGNAL_PROCESSING_PARAMETERS = {
    "sigma": 2.0,          # Global Gaussian smoothing σ

    "methods": {
        "wavelet": {
            "wavelet": "db4", # Wavelet type: 'db4', 'haar', etc.
            "level": None     # Decomposition level (None for automatic)
        },

        "fir": {
            "cutoff": 0.001,
            "numtaps": 201,
            "sampling_freq": 1.0  # Sampling rate in Hz
        },

        "butterworth": {
            "cutoff": 0.003,
            "order": 6,
            "mode": "ba",   # 'ba' for Butterworth, 'sos' for SOS
            "btype": "highpass",
            "sampling_freq": 1.0  # Sampling rate in Hz
        },

        "exponentialfit": {},

        "diff": {},

        "savgol": {
            "presmoothing_sigma": 6.0,
            "window_length": 601,
            "polyorder": 2
        },

        "movingaverage": {
            "window_size": 101
        }
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