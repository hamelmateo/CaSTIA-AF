# config/structures.py
# Master configuration schema using dataclasses + Enums

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
from abc import ABC


# ===========================
# FLAGS
# ===========================
@dataclass
class DebugConfig:
    """
    Configuration for debugging paths and environment settings.

    Attributes:
        debugging (bool): Enable or disable debug mode. Default is True.
        debugging_file_path (str): Path to a dataset or experiment used for debugging. Default is "D:/Mateo/20250326/Data/IS1".
        harddrive_path (str): Root hard drive path for accessing data. Default is "D:/Mateo".
    """
    debugging: bool = True
    debugging_file_path: str = "D:/Mateo/20250326/Data/IS1"
    harddrive_path: str = "D:/Mateo"


# ===========================
# SEGMENTATION PARAMETERS
# ===========================
class SegmentationMethod(str, Enum):
    """
    Enum for available segmentation methods.
    """
    MESMER = "mesmer"
    CELLPOSE = "cellpose" # Not implemented yet
    WATERSHED = "watershed" # Not implemented yet

@dataclass
class SegmentationParams(ABC):
    """
    Base class for segmentation parameters.
    """
    pass

@dataclass
class MesmerParams(SegmentationParams):
    """
    Parameters for MESMER segmentation.

    Attributes:
        image_mpp (float): Microns per pixel. Default is 0.5.
        maxima_threshold (float): Threshold for maxima detection. Default is 0.2.
        maxima_smooth (float): Smoothing for maxima. Default is 2.5.
        interior_threshold (float): Threshold for interior probability. Default is 0.05.
        interior_smooth (float): Smoothing for interior mask. Default is 1.0.
        small_objects_threshold (int): Minimum object size. Default is 25.
        fill_holes_threshold (int): Maximum hole size to fill. Default is 15.
        radius (int): Radius used for boundary refinement. Default is 2.
    """
    image_mpp: float = 0.5
    maxima_threshold: float = 0.2
    maxima_smooth: float = 2.5
    interior_threshold: float = 0.05
    interior_smooth: float = 1.0
    small_objects_threshold: int = 25
    fill_holes_threshold: int = 15
    radius: int = 2

@dataclass
class CellPoseParams(SegmentationParams):
    """
    Parameters for CellPose segmentation.
    (Currently not implemented.)
    """
    pass

@dataclass
class WatershedParams(SegmentationParams):
    """
    Parameters for Watershed segmentation.
    (Currently not implemented.)
    """
    pass

@dataclass
class SegmentationConfig:
    """
    Full segmentation configuration.

    Controls which segmentation method is used and its corresponding parameters.

    Attributes:
        method (SegmentationMethod): Chosen segmentation method. Default is SegmentationMethod.MESMER.
        parameters (SegmentationParams): Parameters associated with the selected method. 
            - If method = MESMER → uses MesmerParams (default)
            - If method = CELLPOSE → uses CellPoseParams (not implemented)
            - If method = WATERSHED → uses WatershedParams (not implemented)
        save_overlay (bool): Whether to save an overlay of the segmentation results. Default is True.

    Notes:
        - SegmentationParams is an abstract base class. Each method has its own subclass:
            * MesmerParams
            * CellPoseParams
            * WatershedParams
    """
    method: SegmentationMethod = SegmentationMethod.MESMER
    params: SegmentationParams = field(default_factory=MesmerParams)
    save_overlay: bool = True


# ===========================
# IMAGE PROCESSING PARAMETERS
# ===========================
class HotPixelMethod(str, Enum):
    """Enum for available hot pixel correction methods."""
    REPLACE = "replace"
    CLIP = "clip"

@dataclass
class HotPixelParameters:
    """
    Parameters for hot pixel correction.

    Attributes:
        method (HotPixelMethod): Strategy for correction. Default is HotPixelMethod.REPLACE.
        use_auto_threshold (bool): Use automatic thresholding. Default is True.
        percentile (float): Percentile used for auto-threshold. Default is 99.9.
        mad_scale (float): Scale factor for MAD. Default is 20.0.
        static_threshold (int): Absolute intensity threshold for static mode. Default is 2000.
        window_size (int): Window size for filtering. Default is 3.
    """
    method: HotPixelMethod = HotPixelMethod.REPLACE
    use_auto_threshold: bool = True
    percentile: float = 99.9
    mad_scale: float = 20.0
    static_threshold: int = 2000
    window_size: int = 3

@dataclass
class ImageProcessingPipeline:
    """
    Toggle steps in the image preprocessing pipeline.

    Attributes:
        padding (bool): Apply zero-padding to images. Default is True.
        cropping (bool): Apply center-cropping to ROI. Default is True.
        hot_pixel_cleaning (bool): Apply hot pixel correction. Default is False.
    """
    padding: bool = True
    cropping: bool = True
    hot_pixel_cleaning: bool = False

@dataclass
class ImageProcessingConfig:
    """
    Configuration for image preprocessing.

    Attributes:
        pipeline (ImageProcessingPipeline): Which steps to apply.
        padding_digits (int): Digits used in file naming (e.g., 5 -> t00001). Default is 5.
        roi_scale (float): Fraction of image to crop (centered). 1.0 = no crop. Default is 0.75.
        hot_pixel_cleaning (HotPixelParameters): Parameters for hot pixel cleanup.
    """
    pipeline: ImageProcessingPipeline = field(default_factory=ImageProcessingPipeline)
    padding_digits: int = 5
    roi_scale: float = 0.75
    hot_pixel_cleaning: HotPixelParameters = field(default_factory=HotPixelParameters)


# ===========================
# TRACE EXTRACTION PARAMETERS
# ===========================
@dataclass
class TraceExtractionConfig:
    """
    Configuration for how cell traces are extracted from image sequences.

    Attributes:
        parallelize (bool): Whether to CPU-parallelize the extraction. Default is True.
        trace_version_name (str): Name of the trace version (e.g., 'raw'). Default is "raw".
    """
    parallelize: bool = True
    trace_version_name: str = "raw"


# ===========================
# CELL DETECTION PARAMETERS
# ===========================
@dataclass
class ObjectSizeThresholds:
    """
    Size constraints for detected objects.

    Attributes:
        min (int): Minimum object area in pixels. Default is 500.
        max (int): Maximum object area in pixels. Default is 10000.
    """
    min: int = 500
    max: int = 10000

@dataclass
class CellFilteringConfig:
    """
    Configuration for filtering segmented cells.

    Attributes:
        border_margin (int): Exclude cells within this many pixels from image border. Default is 20.
        object_size_thresholds (ObjectSizeThresholds): Min/max size thresholds for filtering. Defaults to (500, 10000).
    """
    border_margin: int = 20
    object_size_thresholds: ObjectSizeThresholds = field(default_factory=ObjectSizeThresholds)


# ===========================
# PEAK DETECTION PARAMETERS
# ===========================
class PeakDetectionMethod(str, Enum):
    """
    Enum of supported peak detection strategies.
    """
    SKIMAGE = "skimage"
    CUSTOM = "custom"
    THRESHOLD = "threshold"

@dataclass
class PeakDetectorParams(ABC):
    """
    Abstract base class for peak detection parameters.
    """
    pass

@dataclass
class SkimageParams(PeakDetectorParams):
    """
    Parameters for peak detection using skimage.find_peaks.

    Attributes:
        prominence (float): Minimum prominence. Default is None.
        distance (int): Minimum distance between peaks. Default is 10.
        height (float): Minimum height. Default is 20.
        threshold (float): Threshold for detecting a peak. Default is None.
        width (float): Minimum peak width. Default is None.
        scale_class_quantiles (Tuple[float, float]): Quantiles to assign peak scale class. Default is (0.33, 0.66).
        relative_height (float): Height relative to max for filtering. Default is 0.3.
        full_duration_threshold (float): Duration relative to full trace. Default is 0.95.
    """
    prominence: float = None
    distance: int = 10
    height: float = 20
    threshold: float = None
    width: float = None
    scale_class_quantiles: Tuple[float, float] = (0.33, 0.66)
    relative_height: float = 0.3
    full_duration_threshold: float = 0.95


@dataclass
class CustomParams(PeakDetectorParams):
    """
    Parameters for custom peak detection 
    (not yet implemented).
    """
    pass

@dataclass
class ThresholdParams(PeakDetectorParams):
    """
    Parameters for threshold-based peak detection 
    (not yet implemented).
    """
    pass

@dataclass
class PeakGroupingParams:
    """
    Parameters controlling how closely-timed peaks are grouped.

    Attributes:
        overlap_margin (int): Allowed frame overlap between peaks. Default is 0.
        verbose (bool): Verbosity flag. Default is False.
    """
    overlap_margin: int = 0
    verbose: bool = False

@dataclass
class PeakDetectionConfig:
    """
    Full peak detection configuration.

    Attributes:
        method (PeakDetectionMethod): Detection method to use. Default is SKIMAGE.
        params (PeakDetectorParams): Parameters associated with method. Default is SkimageParams.
            - If method = SKIMAGE → uses SkimageParams (default)
            - If method = CUSTOM → uses CustomParams (not implemented)
            - If method = THRESHOLD → uses ThresholdParams (not implemented)
        peak_grouping (PeakGroupingParams): Parameters for peak grouping. Default values are overlap_margin=0, verbose=False.
        start_frame (int): Optional start frame to restrict detection. Default is None.
        end_frame (int): Optional end frame to restrict detection. Default is None.
        filter_overlapping_peaks (bool): Whether to remove overlapping peaks. Default is True.

    Notes:
        - PeakDetectorParams is an abstract base class. Each method has its own subclass:
            * SkimageParams
            * CustomParams
            * ThresholdParams
    """
    method: PeakDetectionMethod = PeakDetectionMethod.SKIMAGE
    params: PeakDetectorParams = field(default_factory=SkimageParams)
    peak_grouping: PeakGroupingParams = field(default_factory=PeakGroupingParams)
    start_frame: int = None
    end_frame: int = None
    filter_overlapping_peaks: bool = True


# ===========================
# NORMALIZATION METHODS PARAMETERS
# ===========================
class NormalizationMethod(str, Enum):
    """
    Enum of supported normalization methods.
    """
    DELTAF = "deltaf"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    PERCENTILE = "percentile"

@dataclass
class NormalizationParams(ABC):
    """
    Abstract base class for normalization parameter sets.

    Attributes:
        epsilon (float): Small value to avoid division by zero. Default is 1e-8.
    """
    epsilon: float = 1e-8

@dataclass
class ZScoreParams(NormalizationParams):
    """
    Parameters for Z-score normalization.

    Attributes:
        smoothing_sigma (float): Standard deviation for Gaussian smoothing. Default is 2.0.
        residuals_clip_percentile (float): Percentile for clipping residuals. Default is 80.0.
        residuals_min_number (int): Minimum number of residuals for analysis. Default is 20.
    """
    smoothing_sigma: float = 2.0
    residuals_clip_percentile: float = 80.0
    residuals_min_number: int = 20

@dataclass
class PercentileParams(NormalizationParams):
    """
    Parameters for percentile-based normalization.

    Attributes:
        percentile_baseline (float): Percentile used for baseline calculation. Default is 10.
    """
    percentile_baseline: float = 10

@dataclass
class MinMaxParams(NormalizationParams):
    """
    Parameters for min-max normalization.

    Attributes:
        min_range (float): Minimum range value. Default is 1e-2.
    """
    min_range: float = 1e-2

@dataclass
class DeltaFParams(NormalizationParams):
    """
    Parameters for deltaF normalization.

    Attributes:
        percentile_baseline (float): Percentile used for baseline calculation. Default is 10.
        min_range (float): Minimum range value. Default is 1e-2.
    """
    percentile_baseline: float = 10
    min_range: float = 1e-2

# ===========================
# DETRENDING METHODS PARAMETERS
# ===========================
class DetrendingMethod(str, Enum):
    """
    Enum of supported detrending methods.
    """
    LOCALMINIMA = "localminima"
    MOVINGAVERAGE = "movingaverage"
    POLYNOMIAL = "polynomial"
    ROBUSTPOLY = "robustpoly"
    EXPONENTIAL = "exponentialfit"
    SAVGOL = "savgol"
    BUTTERWORTH = "butterworth"
    FIR = "fir"
    WAVELET = "wavelet"
    DOUBLECURVE = "doublecurvefitting"

@dataclass
class DetrendingParams(ABC):
    """
    Abstract base class for detrending parameters.

    Attributes:
        cut_trace_num_points (int): Number of points to cut from the trace. Default is 100.
    """
    cut_trace_num_points: int = 100

# Baseline-based detrending
@dataclass
class BaselineSubtractionDetrendingParams(DetrendingParams):
    """
    Parameters for baseline subtraction detrending.

    Attributes:
        baseline_detection_params (PeakDetectionConfig): Parameters for baseline detection. Default is PeakDetectionConfig().
    """
    baseline_detection_params: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)

@dataclass
class MovingAverageParams(BaselineSubtractionDetrendingParams):
    """
    Parameters for moving average detrending.

    Attributes:
        window_size (int): Size of the moving average window. Default is 201.
    """
    window_size: int = 201

@dataclass
class PolynomialParams(BaselineSubtractionDetrendingParams):
    """
    Parameters for polynomial detrending.

    Attributes:
        degree (int): Degree of the polynomial fit. Default is 2.
    """
    degree: int = 2

@dataclass
class RobustPolyParams(BaselineSubtractionDetrendingParams):
    """
    Parameters for robust polynomial detrending.

    Attributes:
        degree (int): Degree of the polynomial fit. Default is 2.
        method (str): Robust fitting method. Default is "huber".
    """
    degree: int = 2
    method: str = "huber"  # or "ransac"

@dataclass
class SavgolParams:
    """
    Parameters for Savitzky-Golay filter detrending.

    Attributes:
        window_length (int): Length of the filter window. Must be odd. Default is 101.
        polyorder (int): Order of the polynomial used to fit the samples. Default is 2.
    """
    window_length: int = 101
    polyorder: int = 2

# Filter-based detrending
@dataclass
class FilterDetrendingParams(DetrendingParams):
    """
    Parameters for filter-based detrending.

    Attributes:
        sampling_freq (float): Sampling frequency of the signal. Default is 1.0.
    """
    sampling_freq: float = 1.0

@dataclass
class ButterworthParams(FilterDetrendingParams):
    """
    Parameters for Butterworth filter detrending.

    Attributes:
        cutoff (float): Cutoff frequency for the filter. Default is 0.003.
        order (int): Order of the filter. Default is 6.
    """
    cutoff: float = 0.003
    order: int = 6

@dataclass
class FIRParams(FilterDetrendingParams):
    """
    Parameters for FIR filter detrending.

    Attributes:
        cutoff (float): Cutoff frequency for the filter. Default is 0.001.
        numtaps (int): Number of taps in the FIR filter. Default is 201.
    """
    cutoff: float = 0.001
    numtaps: int = 201

@dataclass
class WaveletParams(FilterDetrendingParams):
    """
    Parameters for wavelet detrending.

    Attributes:
        wavelet (str): Name of the wavelet to use. Default is "db4".
        level (int): Level of decomposition. Default is 3.
    """
    wavelet: str = "db4"
    level: int = 3

# Specific taylored detrending
@dataclass
class DoubleCurveFittingParams(DetrendingParams):
    """
    Parameters for double curve fitting detrending.

    Attributes:
        fit_method (str): Fitting method to use. Default is "movingaverage".
        window_size (int): Size of the moving average window. Default is 121.
        mask_method (str): Masking method to use. Default is "percentile".
        percentile_bounds (Tuple[int, int]): Percentile bounds for masking. Default is (0, 75).
        max_iterations (int): Maximum number of iterations for fitting. Default is 5.
    """
    fit_method: str = "movingaverage"
    window_size: int = 121
    mask_method: str = "percentile"
    percentile_bounds: Tuple[int, int] = (0, 75)
    max_iterations: int = 5

@dataclass
class LocalMinimaParams(DetrendingParams):
    """
    Parameters for local minima detrending.

    Attributes:
        verbose (bool): Whether to print debug information. Default is False.
        minima_detection_order (int): Order of the minima detection. Default is 15.
        edge_anchors_window (int): Window size for edge anchors. Default is 50.
        edge_anchors_delta (float): Delta for edge anchors. Default is 0.03.
        filtering_shoulder_neighbor_dist (int): Distance for shoulder neighbor filtering. Default is 400.
        filtering_shoulder_window (int): Window size for shoulder filtering. Default is 100.
        filtering_angle_thresh_deg (int): Angle threshold for filtering shoulders. Default is 10.
        crossing_correction_min_dist (int): Minimum distance for crossing correction. Default is 10.
        crossing_correction_max_iterations (int): Maximum iterations for crossing correction. Default is 10.
        fitting_method (str): Method for fitting the local minima. Default is "linear".
        diagnostics_enabled (bool): Whether to enable diagnostics. Default is False.
        diagnostics_output_dir (str): Directory for diagnostics output. Default is "D:/Mateo/20250326/Output/IS1/plot-diagnostics".
    """
    verbose: bool = False
    minima_detection_order: int = 15

    edge_anchors_window: int = 50
    edge_anchors_delta: float = 0.03

    filtering_shoulder_neighbor_dist: int = 400
    filtering_shoulder_window: int = 100
    filtering_angle_thresh_deg: int = 10

    crossing_correction_min_dist: int = 10
    crossing_correction_max_iterations: int = 10
    
    fitting_method: str = "linear"

    diagnostics_enabled: bool = False
    diagnostics_output_dir: str = "D:/Mateo/20250326/Output/IS1/plot-diagnostics"

# ===========================
# SIGNAL PROCESSING PARAMETERS
# ===========================
@dataclass
class SignalProcessingPipeline:
    """
    Configuration for the signal processing pipeline.

    Attributes:
        detrending (bool): Whether to apply detrending. Default is True.
        normalization (bool): Whether to apply normalization. Default is True.
        smoothing (bool): Whether to apply smoothing. Default is True.
    """
    detrending: bool = True
    normalization: bool = True
    smoothing: bool = True

@dataclass
class NormalizationConfig:
    """
    Configuration for normalization methods.

    Attributes:
        method (NormalizationMethod): Normalization method to use. Default is NormalizationMethod.ZSCORE.
        params (NormalizationParams): Parameters associated with the method.
            - If method = ZSCORE → uses ZScoreParams (default)
            - If method = MINMAX → uses MinMaxParams
            - If method = PERCENTILE → uses PercentileParams
            - If method = DELTAF → uses DeltaFParams
    """
    method: NormalizationMethod = NormalizationMethod.ZSCORE
    params: NormalizationParams = field(default_factory=ZScoreParams)

@dataclass
class DetrendingConfig:
    """
    Configuration for detrending methods.

    Attributes:
        method (DetrendingMethod): Detrending method to use. Default is DetrendingMethod.LOCALMINIMA.
        params (DetrendingParams): Parameters associated with the method.
            - If method = LOCALMINIMA → uses LocalMinimaParams (default)
            - If method = MOVINGAVERAGE → uses MovingAverageParams
            - If method = POLYNOMIAL → uses PolynomialParams
            - If method = ROBUSTPOLY → uses RobustPolyParams
            - If method = EXPONENTIAL → uses ExponentialFitParams
            - If method = SAVGOL → uses SavgolParams
            - If method = BUTTERWORTH → uses ButterworthParams
            - If method = FIR → uses FIRParams
            - If method = WAVELET → uses WaveletParams
            - If method = DOUBLECURVE → uses DoubleCurveFittingParams
    """
    method: DetrendingMethod = DetrendingMethod.LOCALMINIMA
    params: DetrendingParams = field(default_factory=LocalMinimaParams)

@dataclass
class SignalProcessingConfig:
    """
    Configuration for signal processing steps.

    Attributes:
        pipeline (SignalProcessingPipeline): Which steps to apply in the processing pipeline.
        smoothing_sigma (float): Standard deviation for Gaussian smoothing. Default is 3.0.
        normalization (NormalizationConfig): Configuration for normalization methods.
        detrending (DetrendingConfig): Configuration for detrending methods.
    """
    pipeline: SignalProcessingPipeline = field(default_factory=SignalProcessingPipeline)
    smoothing_sigma: float = 3.0
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    detrending: DetrendingConfig = field(default_factory=DetrendingConfig)

# ===========================
# EVENT DETECTION PARAMETERS
# ===========================
@dataclass
class ConvexHullParams:
    """
    Parameters for convex hull-based event detection.

    Attributes:
        min_points (int): Minimum number of points to form a convex hull. Default is 3.
        min_duration (int): Minimum duration (in frames) for an event. Default is 1.
    """
    min_points: int = 3
    min_duration: int = 1

@dataclass
class EventExtractionConfig:
    """
    Configuration for event extraction from activity traces.

    Attributes:
        min_cell_count (int): Minimum number of cells required to form an event. Default is 2.
        threshold_ratio (float): Ratio of the maximum activity trace value to consider an event. Default is 0.4.
        radius (float): Radius for spatial clustering of events. Default is 300.0.
        global_max_comm_time (int): Maximum communication time for global events. Default is 10.
        seq_max_comm_time (int): Maximum communication time for sequential events. Default is 10.
        convex_hull (ConvexHullParams): Parameters for convex hull-based event detection. Default is ConvexHullParams().
    """
    min_cell_count: int = 2
    threshold_ratio: float = 0.4
    radius: float = 300.0
    global_max_comm_time: int = 10
    seq_max_comm_time: int = 10
    convex_hull: ConvexHullParams = field(default_factory=ConvexHullParams)


# ===========================
# GLOBAL CONFIG
# ===========================
@dataclass
class GlobalConfig:
    """
    Master configuration for the calcium activity characterization pipeline.

    Attributes:
        debug (DebugConfig): Debugging configuration.
        segmentation (SegmentationConfig): Segmentation configuration.
        image_processing_hoechst (ImageProcessingConfig): Image processing configuration for Hoechst channel.
        image_processing_fitc (ImageProcessingConfig): Image processing configuration for FITC channel.
        trace_extraction (TraceExtractionConfig): Trace extraction configuration.
        cell_filtering (CellFilteringConfig): Cell filtering configuration.
        cell_trace_processing (SignalProcessingConfig): Signal processing configuration for cell traces.
        cell_trace_peak_detection (PeakDetectionConfig): Peak detection configuration for cell traces.
        activity_trace_processing (SignalProcessingConfig): Signal processing configuration for activity traces.
        activity_trace_peak_detection (PeakDetectionConfig): Peak detection configuration for activity traces.
        event_extraction (EventExtractionConfig): Event extraction configuration.
    """
    debug: DebugConfig
    segmentation: SegmentationConfig
    image_processing_hoechst: ImageProcessingConfig
    image_processing_fitc: ImageProcessingConfig
    trace_extraction: TraceExtractionConfig
    cell_filtering: CellFilteringConfig
    cell_trace_processing: SignalProcessingConfig
    cell_trace_peak_detection: PeakDetectionConfig
    activity_trace_processing: SignalProcessingConfig
    activity_trace_peak_detection: PeakDetectionConfig
    event_extraction: EventExtractionConfig


# ==================================================================
# UNUSED / DEPRECATED PARAMETERS (not used, retained for future use)
# ==================================================================

# ==========================
# SPATIAL CLUSTERING PARAMETERS
# ==========================
@dataclass
class SpatialClusteringParameters:
    trace: str = "impulse_trace"
    use_indirect_neighbors: bool = False
    indirect_neighbors_num: int = 1
    use_sequential: bool = True
    seq_max_comm_time: int = 10

# ==========================
# PEAK CLUSTERING PARAMETERS
# ==========================
@dataclass
class ScoreWeights:
    time: float = 0.7
    duration: float = 0.3

@dataclass
class PeakClusteringParams:
    method: str = "fixed"
    adaptive_window_factor: float = 0.5
    fixed_window_size: int = 20
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

# ==========================
# GRANGER CAUSALITY PARAMETERS
# ==========================
class GrangerCausalityMethod(str, Enum):
    PAIRWISE = "pairwise"
    MULTIVARIATE = "multivariate"

@dataclass
class GrangerCausalityParams(ABC):
    window_size: int = 150
    lag_order: int = 3
    min_cells: int = 1

@dataclass
class PairwiseParams(GrangerCausalityParams):
    threshold_links: bool = False
    pvalue_threshold: float = 0.001

@dataclass
class MultiVariateParams(GrangerCausalityParams):
    pass

@dataclass
class GrangerCausalityConfig:
    method: GrangerCausalityMethod = GrangerCausalityMethod.PAIRWISE
    trace: str = "binary_trace"
    parameters: GrangerCausalityParams = field(default_factory=PairwiseParams)

# ==========================
# CORRELATION PARAMETERS
# ==========================
class CorrelationMethod(str, Enum):
    CROSSCORRELATION = "crosscorrelation"
    JACCARD = "jaccard"
    PEARSON = "pearson"
    SPEARMAN = "spearman"

@dataclass
class CorrelationParams(ABC):
    pass

class CrossCorrelationParams(CorrelationParams):
    mode: str = "full"
    method: str = "direct"

class JaccardParams(CorrelationParams):
    pass

class PearsonParams(CorrelationParams):
    pass

class SpearmanParams(CorrelationParams):
    pass

@dataclass
class CorrelationConfig:
    parallelize: bool = True
    window_size: int = 100
    step_percent: float = 0.75
    lag_percent: float = 0.25
    method: CorrelationMethod = CorrelationMethod.CROSSCORRELATION
    params: CorrelationParams = field(default_factory=CorrelationParams)


# ==========================
# CLUSTERING PARAMETERS
# ==========================
class ClusteringMethod(str, Enum):
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    AGGLOMERATIVE = "agglomerative"
    AFFINITYPROPAGATION = "affinitypropagation"
    GRAPHCOMMUNITY = "graphcommunity"

class AffinityMetric(str, Enum):
    PRECOMPUTED = "precomputed"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

class LinkageType(str, Enum):
    COMPLETE = "complete"
    WARD = "ward"
    AVERAGE = "average"
    SINGLE = "single"

@dataclass
class ClusteringParams(ABC):
    pass

dataclass
class DbscanParams(ClusteringParams):
    eps: float = 0.03
    min_samples: int = 3
    metric: AffinityMetric = AffinityMetric.PRECOMPUTED

@dataclass
class HdbscanParams(ClusteringParams):
    min_cluster_size: int = 3
    min_samples: int = 3
    metric: AffinityMetric = AffinityMetric.PRECOMPUTED
    clustering_method: str = "eom" # or "leaf"
    probability_threshold: float = 0.85
    cluster_selection_epsilon: float = 0.5

@dataclass
class AgglomerativeParams(ClusteringParams):
    #TODO - check is we can put None if we put a type hint
    n_clusters: int = None
    distance_threshold: float = 0.5
    linkage: LinkageType = LinkageType.COMPLETE
    metric: AffinityMetric = AffinityMetric.PRECOMPUTED
    auto_threshold: bool = True

@dataclass
class AffinityPropagationParams(ClusteringParams):
    preference: Optional[float] = None
    damping: float = 0.9
    max_iter: int = 200
    convergence_iter: int = 15
    affinity: AffinityMetric = AffinityMetric.PRECOMPUTED

@dataclass
class GraphCommunityParams(ClusteringParams):
    threshold: float = 0.7

@dataclass
class ClusteringConfig:
    method: ClusteringMethod = ClusteringMethod.AGGLOMERATIVE
    min_cluster_size: int = 3
    params: ClusteringParams = field(default_factory=ClusteringParams)

# ==========================
# ARCOS PARAMETERS
# ==========================
@dataclass
class ArcosBindataParameters:
    smooth_k: int = 3
    bias_k: int = 51
    peak_threshold: float = 0.2
    binarization_threshold: float = 0.1
    polynomial_degree: int = 1
    bias_method: str = "runmed" # can be 'lm', 'runmed', or 'none'
    n_jobs: int = -1

@dataclass
class ArcosTrackingParameters:
    position_columns: List[str] = field(default_factory=lambda: ["x", "y"])
    frame_column: str = "frame"
    id_column: str = "trackID"
    binarized_measurement_column: str = "intensity.bin"
    clid_column: str = "event_id"
    eps: float = 50.0
    eps_prev: float = 150.0
    min_clustersize: int = 15
    allow_merges: bool = False
    allow_splits: bool = False
    stability_threshold: int = 30
    linking_method: str = "nearest"
    clustering_method: str = "dbscan"
    min_samples: int = 1
    remove_small_clusters: bool = False
    min_size_for_split: int = 1
    reg: int = 1
    reg_m: int = 10
    cost_threshold: int = 0
    n_prev: int = 1
    predictor: bool = False
    n_jobs: int = 10
    show_progress: bool = True
