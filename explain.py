# -*- coding: utf-8 -*-
"""A LIME explanation module for model prediction.

Original code: https://zenodo.org/record/4564072#.YYml957MKck.

Original paper: https://dl.acm.org/doi/10.1145/3468264.3468563

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Both feature importances for classes.
        * Not proceed. Reason: Importance of graduation is equivalent to the
            negative of the importance of retirement.

"""
import math

from lime import lime_tabular
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def reshape_X(seq, n_timesteps):
    # N = len(seq) - n_timesteps - 1
    N = int(len(seq) / n_timesteps)
    if N <= 0:
        raise ValueError('need more data.')

    nf = seq.shape[1]
    new_seq = np.zeros((N, n_timesteps, nf))
    for i in range(N):
        new_seq[i, :, :] = seq[i:i + n_timesteps]

    return new_seq


def reshape_y(seq, n_timesteps):
    # N = len(seq) - n_timesteps - 1
    N = int(len(seq) / n_timesteps)
    if N <= 0:
        raise ValueError('need more data.')

    nf = seq.shape[1]
    new_seq = np.zeros((N, nf))
    for i in range(N):
        new_seq[i, :] = seq[i * n_timesteps]

    return new_seq


def plot_trajectories(trajectories, feature):
    """
    """
    feature_trajectory = [trajectories[t][feature]
                          for t in trajectories.keys()]
    plt.plot(
        list(trajectories.keys()),
        feature_trajectory)
    plt.show()


def explain_consecutive_instances(
        data_points,
        model,
        training_data,
        training_labels=None,
        random_state=None):
    """
    """
    feature_names = [
        'active_devs', 'num_commits', 'num_files', 'num_emails',
        'c_percentage', 'e_percentage', 'inactive_c', 'inactive_e',
        'c_nodes', 'c_edges', 'c_c_coef', 'c_mean_degree', 'c_long_tail',
        'e_nodes', 'e_edges', 'e_c_coef', 'e_mean_degree', 'e_long_tail']

    explainer = lime_tabular.RecurrentTabularExplainer(
        training_data=training_data,
        training_labels=training_labels,
        feature_names=feature_names,
        class_names=['Graduated', 'Retired'],
        discretize_continuous=False,
        random_state=random_state)

    exp = explainer.explain_instance(
        data_points,
        model.predict,
        labels=(0, 1),
        num_features=len(data_points) * len(feature_names),
        num_samples=5000)

    feature_score_pairs = exp.as_list()

    # Split timestamp and feature name and sort by timestamp.
    feature_t_score_triples \
        = [(f.split('-')[0][:-2], f.split('-')[1], s)
            for f, s in feature_score_pairs]

    trajectories = {i: {} for i in range(len(data_points))}
    for f, t, s in feature_t_score_triples:
        trajectories[int(t)][f] = s

    # Sort keys to be consistent to the input.
    for i in range(len(data_points)):
        trajectories[i] = {f: trajectories[i][f] for f in feature_names}

    return trajectories


if __name__ == '__main__':
    N_TIMESTEPS = 8

    df = pd.read_csv(
        f"Sustainability_Analysis/Reformat_data/{N_TIMESTEPS}.csv")
    df.replace('Graduated', '1', inplace=True)
    df.replace('Retired', '0', inplace=True)

    data_columns = [
        'active_devs', 'num_commits', 'num_files', 'num_emails',
        'c_percentage', 'e_percentage', 'inactive_c', 'inactive_e',
        'c_nodes', 'c_edges', 'c_c_coef', 'c_mean_degree', 'c_long_tail',
        'e_nodes', 'e_edges', 'e_c_coef', 'e_mean_degree', 'e_long_tail']
    target_columns = ['status']

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_original = scaler.fit_transform(df[data_columns].values)
    X = reshape_X(X_original, n_timesteps=N_TIMESTEPS)
    y = reshape_y(df[target_columns].values, n_timesteps=N_TIMESTEPS)
    y = to_categorical(y.astype(int))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42)

    model = Sequential()
    model.add(LSTM(64, input_shape=(N_TIMESTEPS, len(data_columns))))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])
    model.fit(
        X_train,
        y_train,
        batch_size=30,
        epochs=100,
        validation_data=(X_test, y_test),
        verbose=1)

    trajectories = explain_consecutive_instances(
        data_points=X_test[0],
        model=model,
        training_data=X_train,
        training_labels=y_train)

    # plot_trajectories(trajectories, 'num_commits')
