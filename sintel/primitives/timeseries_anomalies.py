import numpy as np
import pandas as pd

from orion.primitives.timeseries_anomalies import _find_sequences
from orion.evaluation.utils import from_list_points_timestamps

def format_anomalies(y_hat, index, interval=21600, anomaly_padding=50):
    """Format binary predictions into anomalous sequences.
    Args:
        y_hat (ndarray):
            Array of predictions.
        index (ndarray):
            Array of indices of the windows.
        threshold (int):
            Space between indices.
        anomaly_padding (int):
            Optional. Number of errors before and after a found anomaly that are added to the
            anomalous sequence. If not given, 50 is used.
    Returns:
        ndarray:
            Array containing start-index, end-index for each anomalous sequence that
            was found.
    """
    gap = interval + 2 * anomaly_padding
    anomalies = from_list_points_timestamps(index[y_hat.astype(bool)], gap=gap)

    return np.asarray(anomalies)
