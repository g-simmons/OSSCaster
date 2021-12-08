import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from osscaster.constants import DATA_COLUMNS


def reshape_X(seq, n_timesteps: int):
    """
    - Reshape data to shape required by LSTM Model
    NOTE: truncates data to n_timesteps
    """
    N = int(len(seq) / n_timesteps)
    if N <= 0:
        raise ValueError("need more data.")

    nf = seq.shape[1]
    new_seq = np.zeros((N, n_timesteps, nf))
    for i in range(N):
        new_seq[i, :, :] = seq[i : i + n_timesteps]

    return new_seq


def reshape_y(seq, n_timesteps):
    N = int(len(seq) / n_timesteps)
    if N <= 0:
        raise ValueError("need more data.")

    nf = seq.shape[1]
    new_seq = np.zeros((N, nf))
    for i in range(N):
        new_seq[i, :] = seq[i * n_timesteps]

    return new_seq


def prep_features_for_model(features_df: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_original = scaler.fit_transform(features_df[DATA_COLUMNS].values)
    X = reshape_X(X_original, n_timesteps=len(features_df))
    return X
