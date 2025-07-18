# ==========================
# FLAGS
# ==========================
DEBUGGING = True  # Enable debugging mode
DEBUGGING_FILE_PATH = "D:/Mateo/20250326/Data/IS1"

HARDDRIVE_PATH = "D:/Mateo" # Path to the hard drive for file operations

# ==========================
# SEGMENTATION PARAMETERS
# ==========================

SEGMENTATION_PARAMETERS = {
    "segmentation_method": "mesmer",  # Segmentation method to use - 'mesmer', 'cellpose', 'watershed'
    
    "mesmer_parameters": {
        "image_mpp": 0.5,  # Microns per pixel for Mesmer segmentation
        "postprocess_kwargs_nuclear": {
            "maxima_threshold": 0.2,  # Threshold for maxima detection
            "maxima_smooth": 2.5,  # Smoothing for maxima detection
            "interior_threshold": 0.05,  # Threshold for interior detection
            "interior_smooth": 1,  # Smoothing for interior detection
            "small_objects_threshold": 25,  # Minimum size for small objects
            "fill_holes_threshold": 15,  # Threshold for filling holes
            "radius": 2  # Radius for morphological operations
        }
    },
    
    "save_overlay": True  # Save segmentation overlay
}


# ==========================
# IMAGE PROCESSING PARAMETERS
# ==========================

HOECHST_IMAGE_PROCESSING_PARAMETERS = {
    "apply": {
        "padding": True,
        "cropping": True,
        "hot_pixel_cleaning": False
    },

    "padding_digits": 5,  # e.g. t00001
    "roi_scale": 0.75,  # 1.0 = no crop

    "hot_pixel_cleaning": {
        "method": "replace",         # "replace" or "clip"
        "use_auto_threshold": True,
        "percentile": 99.9,
        "mad_scale": 20.0,
        "static_threshold": 2000,
        "window_size": 3
    }
}

FITC_IMAGE_PROCESSING_PARAMETERS = {
    "apply": {
        "padding": True,
        "cropping": True,
        "hot_pixel_cleaning": True
    },

    "padding_digits": 5,  # e.g. t00001
    "roi_scale": 0.75,  # 1.0 = no crop

    "hot_pixel_cleaning": {
        "method": "replace",         # "replace" or "clip"
        "use_auto_threshold": False,
        "percentile": 99.9,
        "mad_scale": 20.0,
        "static_threshold": 2000,
        "window_size": 3
    }
}


TRACE_EXTRACTION_PARAMETERS = {
    "parallelize": True,            # Use multiprocessing to speed up trace extraction
    "trace_version_name": "raw"     # The version key used to store in Cell.trace
}


# ==========================
# CELL DETECTION PARAMETERS
# ==========================

CELL_FILTERING_PARAMETERS = {
    "border_margin": 20,  # Margin from image border to exclude cells
    "object_size_thresholds": {
        "min": 500,  # Minimum size in pixels for a cell to be considered valid
        "max": 10000  # Maximum size in pixels for a cell to be considered valid
    }  # Size thresholds for filtering cells
}


# ==========================
# BASELINE SIGNAL PROCESSING PARAMETERS
# ==========================

BASELINE_PEAK_DETECTION_PARAMETERS = {
    "method": "skimage",  # only 'skimage' supported for now
    "params": {
        "skimage": {
            "prominence": 1, # Minimum prominence of peaks
            "distance": 20,  # Minimum distance between peaks
            "height": None,
            "threshold": None,
            "width": None,
            "scale_class_quantiles": [0.33, 0.66],
            "relative_height": 0.97, # Relative height for relative duration calculation
            "full_duration_threshold": 0.95 # Threshold for full duration of peaks
        }
    },
    "peak_grouping": {
        "overlap_margin": 0,  # Margin for grouping overlapping peaks
        "verbose": False  # Print grouping information
    },
    "start_frame": None,  # Starting frame for peak detection (None for no limit)
    "end_frame": None,  # Ending frame for peak detection (None for no limit)
    "filter_overlapping_peaks": False  # Filter overlapping peaks based on prominence
}



# ==========================
# INDIVIDUAL CELLS SIGNAL PROCESSING PARAMETERS
# ==========================
INDIV_SIGNAL_PROCESSING_PARAMETERS = {
    "apply": {
        "detrending": True,
        "normalization": True,
        "smoothing": True,
    },
    "cut_trace_num_points": 100, # Number of points to cut from the start of the trace
    "smoothing_sigma": 2.0, # Sigma for Gaussian smoothing after detrending
    "normalizing_method": "zscore", # Normalization method: 'deltaf', 'zscore', 'minmax', 'percentile'
    "normalization_parameters": {
        "epsilon": 1e-8,
        "min_range": 1e-2, 
        "percentile_baseline": 10, 
    },
    "detrending_mode": "localminima",  # or localminima, polynomial, exponentialfit, butterworth, savgol, robustpoly, fir, wavelet, doublecurvefitting, movingaverage, diff

    "methods": {
        # One entry per method, only the one matching `detrending_mode` is used.
        "movingaverage": {
            "window_size": 201, # Window size for moving average detrending
            "peak_detector_params": BASELINE_PEAK_DETECTION_PARAMETERS
        },
        "polynomial": {
            "degree": 2, # Polynomial degree for detrending
            "peak_detector_params": BASELINE_PEAK_DETECTION_PARAMETERS
        },
        "robustpoly": {
            "degree": 2, # Polynomial degree for robust detrending
            "method": "huber",  # or "ransac"
            "peak_detector_params": BASELINE_PEAK_DETECTION_PARAMETERS
        },
        "exponentialfit": {
            "peak_detector_params": BASELINE_PEAK_DETECTION_PARAMETERS
        },
        "savgol": {
            "window_length": 101, # Window length for Savitzky-Golay filter
            "polyorder": 2, # Polynomial order for Savitzky-Golay filter
            "peak_detector_params": BASELINE_PEAK_DETECTION_PARAMETERS
        },
        "butterworth": {
            "cutoff": 0.003, # Cutoff frequency for Butterworth filter
            "order": 6, # Order of the Butterworth filter
            "sampling_freq": 1.0
        },
        "fir": {
            "cutoff": 0.001, # Cutoff frequency for FIR filter
            "numtaps": 201, # Number of taps for FIR filter
            "sampling_freq": 1.0
        },
        "wavelet": {
            "wavelet": "db4", # Wavelet type for wavelet detrending - 'db4', 'haar', etc.
            "level": 3 # Decomposition level for wavelet transform (None for automatic)
        },
        "doublecurvefitting": { # On-going development, not yet implemented
            "fit_method": "movingaverage",
            "window_size": 121,
            "mask_method": "percentile",  # or "histogram"
            "percentile_bounds": [0, 75],
            "max_iterations": 5,
        },
        "localminima": {
            "verbose": False,
            "minima_detection": {
                "order": 15
            },
            "edge_anchors": {
                "window": 50,
                "delta": 0.03
            },
            "filtering": {
                "shoulder_neighbor_dist": 400,
                "shoulder_window": 100,
                "angle_thresh_deg": 10
            },
            "crossing_correction": {
                "min_dist": 10,
                "max_iterations": 10
            },
            "fitting": {
                "method": "linear"  # only 'linear' supported for now
            },
            "diagnostics": {
                "enabled": False, # Enable diagnostics plots
                "output_dir": "D:/Mateo/20250326/Output/IS1/plot-diagnostics"
            }
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
            "height": 10.0,  # Absolute peak height threshold — a peak must reach at least this value (e.g., 10× noise σ after normalization)
            "threshold": None,  # Minimum vertical drop to each neighbor — avoids detecting small shoulders or micro-peaks
            "distance": 20,  # Minimum number of frames between two peaks — helps prevent double-detection of the same event
            "prominence": None,  # Not used — Prominence-based filtering
            "width": None,  # Optional - If set, filters peaks by their width at `rel_height`
            "relative_height": 0.3,  # Used for measuring peak width and duration — the fraction of peak height where width is evaluated
            "full_duration_threshold": 0.95  # Used to compute full peak duration (e.g., near base) — defines the fraction of max height to use when measuring width
        }
    },
    "peak_grouping": {
        "overlap_margin": 0,  # Margin for grouping overlapping peaks
        "verbose": False  # Print grouping information
    },
    "start_frame": None,  # Starting frame for peak detection (None for no limit)
    "end_frame": None,  # Ending frame for peak detection (None for no limit)
    "filter_overlapping_peaks": True  # Filter overlapping peaks based on prominence
}


# ==========================
# GLOBAL ACTIVITY TRACE SIGNAL PROCESSING PARAMETERS
# ==========================

POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS = {
    "apply": {
        "detrending": False,
        "smoothing": True,
        "normalization": False,
        "cut_trace": False
    },

    "detrending_mode": "butterworth",  # Used only if pipeline == 'custom' - 'butterworth', 'wavelet', 'fir', 'exponentialfit', 'diff', 'savgol', 'movingaverage'
    "normalizing_method": "deltaf",  # Used only if pipeline == 'custom' - 'deltaf', 'zscore', 'minmax', 'percentile'
    "sigma": 5.0,          # Global Gaussian smoothing σ
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
# ACTIVITY TRACE PEAK DETECTION PARAMETERS
# ==========================

ACTIVITY_TRACE_PEAK_DETECTION_PARAMETERS = {
    "method": "skimage",  # only 'skimage' supported for now
    "params": {
        "skimage": {
            "prominence": 10, # Minimum prominence of peaks
            "distance": 20,  # Minimum distance between peaks
            "height": None,
            "threshold": None,
            "width": None,
            "scale_class_quantiles": [0.33, 0.66],
            "relative_height": 0.3, # Relative height for relative duration calculation
            "full_duration_threshold": 0.95 # Threshold for full duration of peaks
        }
    },
    "peak_grouping": {
        "overlap_margin": 0,  # Margin for grouping overlapping peaks
        "verbose": False  # Print grouping information
    },
    "start_frame": 50,  # Starting frame for peak detection (None for no limit)
    "end_frame": None,  # Ending frame for peak detection (None for no limit)
    "filter_overlapping_peaks": False  # Filter overlapping peaks based on prominence
}


# ==========================
# EVENTS DETECTION PARAMETERS
# ==========================

EVENT_EXTRACTION_PARAMETERS = {
    "min_cell_count": 2,  # Minimum number of unique cells required to form an event
    "convex_hull": {
        "min_points": 3,       # Minimum number of points to compute convex hull
        "min_duration": 1      # Minimum time difference to compute propagation speed
    },
    "threshold_ratio": 0.4,  # Minimum ratio of active cells at peak to trigger global event detection
    "radius": 300.0,  # Radius for spatial clustering of events
    "global_max_comm_time": 10,  # Maximum time (in frames) for communication between cells in GLOBAL events
    "seq_max_comm_time": 10,  # Maximum time (in frames) for communication between cells in SEQUENTIAL events
}



# ==========================
# UNUSED / DEPRECATED PARAMETERS (for reference, not currently used)
# ==========================
# This section is for parameters that are not currently used in the pipeline,
# but are kept here for reference or potential future use.


# ==========================
# GLOBAL TRACE PEAK DETECTION PARAMETERS
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
            "relative_height": 0.3, # Relative height for relative duration calculation
            "full_duration_threshold": 0.95 # Threshold for full duration of peaks
        }
    },
    "peak_grouping": {
        "overlap_margin": 0,  # Margin for grouping overlapping peaks
        "verbose": False  # Print grouping information
    },
    "start_frame": 50,  # Starting frame for peak detection (None for no limit)
    "end_frame": None,  # Ending frame for peak detection (None for no limit)
    "filter_overlapping_peaks": False  # Filter overlapping peaks based on prominence
}


# ==========================
# IMPULSE TRACE PEAK DETECTION PARAMETERS
# ==========================

IMPULSE_PEAK_DETECTION_PARAMETERS = {
    "method": "skimage",  # only 'skimage' supported for now
    "params": {
        "skimage": {
            "prominence": 0.5, # Minimum prominence of peaks
            "distance": 5,  # Minimum distance between peaks
            "height": None,
            "threshold": None,
            "width": None,
            "scale_class_quantiles": [0.33, 0.66],
            "relative_height": 0.3, # Relative height for relative duration calculation
            "full_duration_threshold": 0.95 # Threshold for full duration of peaks
        }
    },
    "peak_grouping": {
        "overlap_margin": 0,  # Margin for grouping overlapping peaks
        "verbose": False  # Print grouping information
    },
    "start_frame": 50,  # Starting frame for peak detection (None for no limit)
    "end_frame": None,  # Ending frame for peak detection (None for no limit)
    "filter_overlapping_peaks": False  # Filter overlapping peaks based on prominence
}


# ==========================
# SPATIAL CLUSTERING PARAMETERS
# ==========================

SPATIAL_CLUSTERING_PARAMETERS = {
    "trace": "impulse_trace",  # Name of the trace attribute to use for clustering - "impulse_trace", "activity_trace"
    "apply": {
        "use_indirect_neighbors": False,  # Use indirect neighbors for clustering
        "use_sequential": True  # Use sequential clustering
    },
    "indirect_neighbors_num": 1,  # Consider indirect neighbors up to this number of nodes away
    "seq_max_comm_time": 10, # Maximum time (in frame) for communication between cells
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
    "method": "crosscorrelation",  # Similarity method: 'cross_correlation', 'jaccard', 'pearson', 'spearman'
    "params": {
        "crosscorrelation": {
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

ARCOS_TRACKING = False  # Use ARCOS tracking for event detection

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