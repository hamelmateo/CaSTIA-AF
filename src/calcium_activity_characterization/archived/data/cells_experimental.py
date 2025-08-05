import pandas as pd
from calcium_activity_characterization.logger import logger




def get_arcos_dataframe(self) -> pd.DataFrame:
    """
    Return a DataFrame formatted for arcos4py binarization and event tracking.

    The DataFrame contains the following columns:
        - frame: Timepoint index.
        - trackID: Unique cell identifier.
        - x: X-coordinate of centroid.
        - y: Y-coordinate of centroid.
        - intensity: Raw intensity trace.

    Returns:
        pd.DataFrame: DataFrame with cell information formatted for arcos4py.
    """
    if not self.trace.binary:
        logger.warning(f"Cell {self.label} has no binary trace.")
        return pd.DataFrame()

    data = {
        'frame': range(len(self.trace.binary)),
        'trackID': [self.label] * len(self.trace.binary),
        'x': [self.centroid[1]] * len(self.trace.binary),
        'y': [self.centroid[0]] * len(self.trace.binary),
        'intensity': self.trace.versions["raw"],
        'intensity.bin': self.trace.binary
    }

    return pd.DataFrame(data)