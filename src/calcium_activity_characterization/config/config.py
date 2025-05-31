# TODO: Normalization parameters dictionary
# TODO: deal with deprecated pixel values

# ==========================
# FLAGS
# ==========================
DEBUGGING = True  # Enable debugging mode
DEBUGGING_FILE_PATH = "D:/Mateo/20250326/Data/IS1"
SAVE_OVERLAY = True  # Save segmentation overlay
ARCOS_TRACKING = False  # Use ARCOS tracking for event detection

PARALLELELIZE = True  # Use parallel processing for raw trace computing
HARDDRIVE_PATH = "D:/Mateo" # Path to the hard drive for file operations


# ==========================
# IMAGE PROCESSING PARAMETERS
# ==========================
ROI_SCALE = 0.75  # Scale for ROI cropping (e.g., 0.75 = 75%)
SMALL_OBJECT_THRESHOLD = 200  # Minimum pixel count for valid cell
BIG_OBJECT_THRESHOLD = 10000  # Maximum pixel count for valid cell
PADDING = 5  # Filename zero-padding digits

# ==========================
# INDIVIDUAL CELLS SIGNAL PROCESSING PARAMETERS
# ==========================


INDIV_SIGNAL_PROCESSING_PARAMETERS = {
    "apply": {
        "detrending": False,
        "smoothing": True,
        "normalization": False,
        "cut_trace": False
    },

    "detrending_mode": "butterworth",  # Used only if pipeline == 'custom' - 'butterworth', 'wavelet', 'fir', 'exponentialfit', 'diff', 'savgol', 'movingaverage'
    "normalizing_method": "deltaf",  # Used only if pipeline == 'custom' - 'deltaf', 'zscore', 'minmax', 'percentile'
    "sigma": 10.0,          # Global Gaussian smoothing σ
    "cut_trace_num_points": 0,  # Number of points to keep after cutting the trace

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
# INDIVIDUAL CELLS PEAK DETECTION PARAMETERS
# ==========================

INDIV_PEAK_DETECTION_PARAMETERS = {
    "method": "skimage",  # only 'skimage' supported for now
    "params": {
        "skimage": {
            "prominence": 3, # Minimum prominence of peaks
            "distance": 20,  # Minimum distance between peaks
            "height": None,
            "threshold": None,
            "width": None,
            "scale_class_quantiles": [0.33, 0.66],
            "relative_height": 0.3 # Relative height for FWHM calculation
        }
    },
    "peak_grouping": {
        "overlap_margin": 0,  # Margin for grouping overlapping peaks
        "verbose": False  # Print grouping information
    }
}


# ==========================
# GLOBAL ACTIVITY TRACE SIGNAL PROCESSING PARAMETERS
# ==========================


GLOBAL_SIGNAL_PROCESSING_PARAMETERS = {
    "apply": {
        "detrending": False,
        "smoothing": True,
        "normalization": False,
        "cut_trace": True
    },

    "detrending_mode": "butterworth",  # Used only if pipeline == 'custom' - 'butterworth', 'wavelet', 'fir', 'exponentialfit', 'diff', 'savgol', 'movingaverage'
    "normalizing_method": "deltaf",  # Used only if pipeline == 'custom' - 'deltaf', 'zscore', 'minmax', 'percentile'
    "sigma": 4.0,          # Global Gaussian smoothing σ
    "cut_trace_num_points": 50,  # Number of points to keep after cutting the trace

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
# GLOBAL ACTIVITY TRACE PEAK DETECTION PARAMETERSych
# ==========================

GLOBAL_PEAK_DETECTION_PARAMETERS = {
    "method": "skimage",  # only 'skimage' supported for now
    "params": {
        "skimage": {
            "prominence": 10, # Minimum prominence of peaks
            "distance": 20,  # Minimum distance between peaks
            "height": None,
            "threshold": None,
            "width": None,
            "scale_class_quantiles": [0.33, 0.66],
            "relative_height": 0.3 # Relative height for FWHM calculation
        }
    },
    "peak_grouping": {
        "overlap_margin": 0,  # Margin for grouping overlapping peaks
        "verbose": False  # Print grouping information
    }
}




# ==========================
# PEAK CLUSTERING PARAMETERS
# ==========================

PEAK_CLUSTERING_PARAMETERS = {
    "method": "fixed",  # Clustering method: 'adaptive', 'fixed'
    "adaptive_window_factor": 0.5,  # Factor to determine time window size for adaptive clustering
    "fixed_window_size": 20,     # how wide the window is around each peak
    "score_weights": {
        "time": 0.7,
        "duration": 0.3
    }
}


# ==========================
# MVGC PARAMETERS
# ==========================


GC_PREPROCESSING = {
    "apply": {
        "detrending": True,
        "smoothing": True,
        "normalization": True,
        "cut_trace": False
    },

    "detrending_mode": "movingaverage",  # Used only if pipeline == 'custom' - 'butterworth', 'wavelet', 'fir', 'exponentialfit', 'diff', 'savgol', 'movingaverage'
    "normalizing_method": "zscore",  # Used only if pipeline == 'custom' - 'deltaf', 'zscore', 'minmax', 'percentile'
    "sigma":2.0,

    "method": {
        "movingaverage": {
            "window_size": 351  # Window size for moving average detrending
        }
    }
}

GC_PARAMETERS = {
    "mode": "pairwise",  # or "multivariate"
    "trace": "binary_trace",  # Name of the trace attribute - "binary_trace", "gc_trace", or "raw_intensity_trace"
    "parameters": {
        "pairwise": {
            "window_size": 150,
            "lag_order": 3,
            "min_cells": 1,
            "threshold_links": False,
            "pvalue_threshold": 0.001
        },
        "multivariate": {
            "window_size": 400,
            "lag_order": 5,
            "min_cells": 3
            # Add MVGC-specific controls here later if needed
        }
    }
}



# ==========================
# CORRELATION PARAMETERS
# ==========================

CORRELATION_PARAMETERS = {
    "parallelize": True,  # Use parallel processing for similarity calculation
    "window_size": 100,  # Window size for similarity calculation
    "step_percent": 0.75,  # Percentage of window size for step size calculation
    "lag_percent": 0.25,   # Percentage of window size for lag calculation
    "method": "cross_correlation",  # Similarity method: 'cross_correlation', 'jaccard', 'pearson', 'spearman'
    "params": {
        "cross_correlation": {
            "mode": "full",  # Mode for cross-correlation: 'full', 'valid', 'same'
            "method": "direct",  # Method for cross-correlation: 'direct', 'fft'
        },
        "jaccard": {
        },
        "pearson": {
        },
        "spearman": {
        }
    }
}

CLUSTERING_PARAMETERS = {
    "method": "agglomerative",  # Clustering method: 'dbscan', 'hdbscan', 'agglomerative', 'affinity_propagation', 'graph_community'
    "min_cluster_size": 3,  # Minimum size of clusters
    "params": {
        "dbscan": {
            "eps": 0.03,  # Maximum distance between two samples for them to be considered as in the same cluster
            "min_samples": 3,  # Number of samples in a neighborhood for a point to be considered as a core point
            "metric": "precomputed"  # Distance metric to use
        },
        "hdbscan": {
            "min_cluster_size": 3,  # Minimum size of clusters
            "min_samples": 3,  # Minimum number of samples in a neighborhood for a point to be considered as a core point
            "metric": "precomputed",  # Distance metric to use
            "clustering_method": "eom",  # Clustering method: 'eom' or 'leaf'
            "probability_threshold": 0.85,  # Probability threshold for cluster assignment
            "cluster_selection_epsilon": 0.5  # Epsilon for cluster selection
        },
        "agglomerative": {
            "n_clusters": None,  # Number of clusters to form - if None, the number of clusters is determined by the distance threshold
            "distance_threshold": 0.5,  # Distance threshold to apply when forming clusters
            "linkage": "complete",  # Linkage criterion: 'ward', 'complete', 'average', 'single'
            "metric": "precomputed",  # Metric used to compute the linkage: 'precomputed', 'euclidean', 'manhattan', etc.
            "auto_threshold": True  # Automatically determine the distance threshold based on the data
        },
        "affinity_propagation": {
            "damping": 0.9,  # Damping factor for affinity propagation
            "max_iter": 200,  # Maximum number of iterations
            "convergence_iter": 15,  # Number of iterations with no change to declare convergence
            "preference": None,  # Preference parameter for affinity propagation
            "affinity": "precomputed"  # Affinity metric: 'euclidean', 'manhattan', 'precomputed', etc.
        },
        "graph_community": {
            "threshold": 0.7,  # Threshold for community detection
        }
    }
}



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
    "eps": 50.0,               # Spatial proximity threshold
    "eps_prev": 150.0,          # Maximum distance for linking previous clusters
    "min_clustersize": 15,     # Minimum number of cells to form an event
    "allow_merges": False,     # Allow events to merge
    "allow_splits": False,     # Allow events to split
    "stability_threshold": 30, # Minimum duration (frames) for event stability
    "linking_method": "nearest",  # Method to associate objects across frames - "nearest", "transportation"
    "clustering_method": "dbscan",  # Clustering method for event detection - "dbscan", "hdbscan"
    "min_samples": 1,   # Minimum number of samples for clustering
    "remove_small_clusters": False,  # Remove small clusters
    "min_size_for_split": 1,  # Minimum size for split detection
    "reg": 1, # Regularization parameter for transportation solver
    "reg_m": 10, # Regularization parameter for transportation solver
    "cost_threshold": 0, # Cost threshold for event association
    "n_prev": 1, # Number of previous frames to consider for event association
    "predictor": False, # Predictor for event association
    "n_jobs": 10, # Number of parallel jobs for event association
    "show_progress": True, # Show progress bar for event association
}