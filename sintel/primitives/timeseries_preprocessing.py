import numpy as np
import pandas as pd

def rolling_window_sequences_labels(X, index, window_size, target_size=1, step_size=1,
                                    target_column=1, positive_class=1, min_percent=0.01):
    """Create rolling window sequences out of time series data.
    The function creates an array of input sequences and an array of label sequences by rolling
    over the input sequence with a specified window.
    Args:
        X (ndarray):
            N-dimensional sequence to iterate over.
        index (ndarray):
            Array containing the index values of X.
        window_size (int):
            Length of the input sequences.
        target_size (int):
            Length of the target sequences.
        step_size (int):
            Indicating the number of steps to move the window forward each round.
        target_column (int):
            Indicating which column of X is the target.
        positive_class (int or str):
            Indicating which value is considered the positive class in the target column.
        min_percent (float):
            Optional. Indacting the minimum percentage of anomalous values to consider
            the entire window as anomalous.
    Returns:
        ndarray, ndarray, ndarray, ndarray:
            * input sequences.
            * label sequences.
            * first index value of each input sequence.
            * first index value of each label sequence.
    """
    out_X = list()
    out_y = list()
    X_index = list()
    y_index = list()

    target = X[:, target_column]
    X_ = X[:, :target_column]  # remove label

    start = 0
    max_start = len(X) - window_size - target_size + 1
    while start < max_start:
        end = start + window_size

        labels = target[start:end]
        classes, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(classes, counts))
        if class_counts.get(positive_class, 0) / sum(class_counts.values()) > min_percent:
            out_y.append([positive_class])
        else:
            out_y.append([1 - positive_class])

        out_X.append(X_[start:end])
        X_index.append(index[start])
        y_index.append(index[end])
        start = start + step_size

    return np.asarray(out_X), np.asarray(out_y), np.asarray(X_index), np.asarray(y_index)

