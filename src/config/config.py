# ==========================
# FILE PATTERNS AND PADDING
# ==========================
PADDING = 5  # Filename zero-padding digits

# ==========================
# IMAGE PROCESSING PARAMETERS
# ==========================
ROI_SCALE = 0.75  # Scale for ROI cropping (e.g., 0.75 = 75%)
SMALL_OBJECT_THRESHOLD = 200  # Minimum pixel count for valid cell
GAUSSIAN_SIGMA = 2.5  # Sigma for Gaussian filter
HPF_CUTOFF = 0.001  # High-pass filter cutoff frequency (Hz)
SAMPLING_FREQ = 1.0  # Sampling frequency (Hz)
ORDER = 9  # Filter order
BTYPE = 'highpass'  # Filter type
DETRENDING_MODE = 'savgol'  # Detrending mode ('wavelet', 'butterworth', 'fir', 'exponentialfit', 'diff', 'savgol', 'movingaverage')


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
        "cutoff": 0.001,            # High-pass cutoff frequency (Hz)
        "order": 2,                 # Filter order
        "mode": "sos",              # 'sos' or 'ba'
        "sigma": 2.0,
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
