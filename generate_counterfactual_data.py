# -*- coding: utf-8 -*-
"""Methods of generating counterfactual data for CRN.

Authors:
    Fangzhou Li: fzli@ucdavis.edu

Todo:
    - Move load data to util.py since it is used in multiple files.
    - Argument for the number of samples generating.
    - Current approach might have issue:
        - Adding 10% might generate larger than 1.0 value, which is not trained
            in the model.
    - Can we group multiple actions together? Pertubing on 1 action seems
        trivial in terms of prediction changes.

"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

ACTIONS = ['i10', 'i30', 'd10', 'd30']
FEATURES = [
    'active_devs', 'num_commits', 'num_files', 'num_emails',
    'c_percentage', 'e_percentage', 'inactive_c', 'inactive_e',
    'c_nodes', 'c_edges', 'c_c_coef', 'c_mean_degree', 'c_long_tail',
    'e_nodes', 'e_edges', 'e_c_coef', 'e_mean_degree', 'e_long_tail']
FEATURES_INTERVENE = ['active_devs', 'num_commits', 'num_files', 'num_emails']
TARGETS = ['status']
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

def load_data(n_timesteps=8):
    df = pd.read_csv(
        f"Sustainability_Analysis/Reformat_data/{n_timesteps}.csv")
    df.replace('Graduated', '1', inplace=True)
    df.replace('Retired', '0', inplace=True)
    X_original = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
        df[FEATURES].values)
    y_original = df[TARGETS].values
    N = int(len(X_original) / n_timesteps)

    X = np.zeros((N, n_timesteps, X_original.shape[1]))
    y = np.zeros((N, y_original.shape[1]))
    for i in range(N):
        X[i, :, :] = X_original[i:i + n_timesteps]
    for i in range(N):
        y[i, :] = y_original[i * n_timesteps]
    y = to_categorical(y.astype(int))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test


def generate_counterfactual_feature_value(feature_value, action):
    """Generates a counterfactual feature value.

    Args:
        feature_value (float): The original feature value.
        action (str): The action taken.

    Returns:
        (float): The counterfactual feature value.

    """
    if action == 'i10':
        feature_value *= 1.1
    elif action == 'i30':
        feature_value *= 1.3
    elif action == 'd10':
        feature_value *= 0.9
    elif action == 'd30':
        feature_value *= 0.7
    else:
        raise ValueError('unknown action.')

    return feature_value

def generate_counterfactual_project(x, model, n_actions):
    """Generates a counterfactual data point.

    Args:
        x (np.array): A project data point, shape=(n_timestamps, n_features).
        model (tf.keras.Model): The trained LSTM simulator.
        n_actions (int): The number of counterfactual timestamps taken.

    Returns:
        (tuple): (counterfactual data, prediction, action).
            counterfactual data (np.array):
                shape=(n_timesteps + n_actions, n_features).
            prediction (np.array): [p_graduated, p_retired].
            action (str): {feature}_{action}.

    """
    results = []

    x_ctf = np.copy(x)
    for i in range(n_actions):
        action = np.random.choice(ACTIONS)
        feature = np.random.choice(FEATURES_INTERVENE)

        xt = np.copy(x_ctf[-1])  # Last time step.
        xt[FEATURES.index(feature)] = generate_counterfactual_feature_value(
            xt[FEATURES.index(feature)], action)
        x_ctf = np.append(x_ctf, [xt], axis=0)
        y_ctf = model.predict(np.expand_dims(x_ctf[-len(x):], axis=0))
        results.append((x_ctf[-1], y_ctf, f"{feature}-{action}"))

        # print(
        #     f"Generating {i + 1}-th counterfactual time point...\n"
        #     f"Feature: {feature}\n"
        #     f"Action : {action}")

    return results


def generate_counterfactual_data(
        model, n_samples=10000, n_timestamps=8, n_actions=8):
    """Generates a counterfactual data.

    Args:
        model (tf.keras.Model): The trained LSTM simulator.
        n_samples (int): The number of generated counterfactual data points.
        n_timestamps (int): The number of timestamps in a project.
        n_actions (int): The number of counterfactual timestamps taken.

    Returns:
        (tuple): (counterfactual data, prediction, action).
            counterfactual data (np.array):
                shape=(n_timesteps + n_actions, n_features).
            prediction (np.array): [p_graduated, p_retired].
            action (str): {feature}_{action}.

    """
    len_x_new = n_timestamps + n_actions
    X_train, X_test, y_train, y_test = load_data(n_timestamps)

    x_df_list = []
    for i in range(n_samples):
        idx = np.random.choice(len(X_train))
        x = X_train[idx]
        y = y_train[idx]

        x_ctf = generate_counterfactual_project(x, model, n_actions)

        x_new = x.tolist() + [item[0] for item in x_ctf]
        x_df = pd.DataFrame(x_new, columns=FEATURES)

        x_df['project'] = [i + 1] * len_x_new
        x_df['month'] = [j + 1 for j in range(len_x_new)]
        x_df['prob_grad'] = [np.nan] * n_timestamps \
            + [item[1][0][0] for item in x_ctf]
        x_df['action'] = [np.nan] * n_timestamps \
            + [item[2] for item in x_ctf]
        x_df['status'] \
            = ['Graduated' if y[0] == 1 else 'Retired'
               for _ in range(n_timestamps)] \
            + ['Graduated' if item[1][0][0] > 0.5 else 'Retired'
               for item in x_ctf]

        x_df_list.append(x_df)

    x_df = pd.concat(x_df_list)
    x_df = x_df[['project', 'month', 'status', 'prob_grad', 'action']
                + FEATURES]

    x_df.to_csv("test.csv", index=False)


if __name__ == '__main__':
    N_TIMESTEPS = 8

    model = load_model(f'models/model_{N_TIMESTEPS}.h5')

    generate_counterfactual_data(
        model, n_samples=5, n_timestamps=N_TIMESTEPS, n_actions=8)
