"""
Module that will make the link between arcos lib and our own pipeline.
"""

import pandas as pd
from arcos4py.tools import binData, track_events_dataframe
from calcium_activity_characterization.utilities.loader import save_pickle_file
from calcium_activity_characterization.data.cells import Cell

class ArcosEventDetector:
    """
    Class for processing cells signal and detecting calcium events using the ARCoS library.
    """
    def __init__(self, bindata_params: dict, tracking_params: dict):
        """
        Initialize the ArcosEventDetector.

        Args:
            params (dict): Parameters for the ARCoS algorithm.
        """
        self.bindata_params = bindata_params
        self.tracking_params = tracking_params


    def run(self, active_cells: list[Cell]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute event detection on the raw calcium trace.

        Args:
            active_cells (list[Cell]): List of Cell objects to process.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the events DataFrame and the lineage tracker DataFrame.
        """
        
        df_processed = self._arcos4py_signal_processing_pipeline(active_cells)
        return self._arcos_track_events(df_processed)
    

    def _prepare_arcos_input(self, active_cells) -> pd.DataFrame:
        """
        Prepare the input DataFrame for ARCoS binarization and event tracking.

        Args:
            active_cells (list[Cell]): List of Cell objects to process.
            
        Returns:
            pd.DataFrame: DataFrame in adapted ARCoS format.
        """

        df_list = [cell.get_arcos_dataframe() for cell in active_cells]
        arcos_input_df = pd.concat(df_list, ignore_index=True)
        return arcos_input_df


    def _arcos4py_signal_processing_pipeline(self, active_cells: list[Cell]) -> pd.DataFrame:
        """
        Process the raw calcium traces using ARCoS binarization.

        Args:
            active_cells (list[Cell]): List of Cell objects to process.

        Returns:
            pd.DataFrame: DataFrame containing the binarized data.
        """
        # Create DataFrame from active_cells
        df_raw = self._prepare_arcos_input(active_cells)

        # Instantiate binData with parameters
        binarizer = binData(**self.bindata_params)

        # Run binarization
        df_processed = binarizer.run(
            df_raw,
            group_column="trackID",
            measurement_column="intensity",
            frame_column="frame"
        )

        return df_processed


    def _arcos_track_events(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Track events in the DataFrame using arcos4py's track_events_dataframe function.

        Args:
            df (pd.DataFrame): DataFrame containing the binarized data.

                Must contain the following columns:
                - frame: Timepoint index.
                - trackID: Unique cell identifier.
                - x: X-coordinate of centroid.
                - y: Y-coordinate of centroid.
                - intensity.bin: Binarized intensity trace.
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the events DataFrame and the lineage tracker DataFrame.
        """

        events_df, lineage_tracker = track_events_dataframe(
            df,
            **self.tracking_params
        )

        return events_df, lineage_tracker