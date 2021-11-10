# -*- coding: utf-8 -*-
"""A LIME explanation module for model prediction.

Original code: https://zenodo.org/record/4564072#.YYml957MKck.

Original paper: https://dl.acm.org/doi/10.1145/3468264.3468563

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Data file path?

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


def explain_instance_in_time_window(
        data_points,
        model,
        labels=(1,),
        num_features=-1,
        num_samples=5000,
        X=None,
        y=None,
        feature_names=None,
        random_state=None):
    """
    """
    feature_names = [
        'active_devs', 'num_commits', 'num_files', 'num_emails',
        'c_percentage', 'e_percentage', 'inactive_c', 'inactive_e',
        'c_nodes', 'c_edges', 'c_c_coef', 'c_mean_degree', 'c_long_tail',
        'e_nodes', 'e_edges', 'e_c_coef', 'e_mean_degree', 'e_long_tail']

    explainer = lime_tabular.RecurrentTabularExplainer(
        training_data=None,
        training_labels=None,
        feature_names=feature_names,
        class_names=['Graduated', 'Retired'],
        discretize_continuous=False,
        random_state=random_state)

    for data_point in data_points:
        pass

    explainer.explain_instance(
        data_point,
        model.predict,
        labels=labels,
        num_features=num_features,
        num_samples=num_samples)


if __name__ == '__main__':
    # Load a model that is used to test explanation API.

    # Use N_TIMESTEPS as time of lookback for LSTM
    # qutertile_coefs_path = './results/test_qutertile_coefs.csv'
    # aggregated_coefs_path = './results/aggregated_coefs.csv'
    # monthly_coefs_path = './results/monthly_coefs.csv'

    # with open(aggregated_coefs_path, 'w') as f:
    #     f.write('project,feature,weight\n')
    # with open(qutertile_coefs_path, 'w') as f:
    #     f.write('project,feature,quartile,weight\n')
    # with open(monthly_coefs_path, 'w') as f:
    #     f.write('project,feature,month,weight\n')

    # month_list = list(range(1, 31))[::-1]
    # month_list = list(range(12,13))[::-1]
    project_set = set()

    N_TIMESTEPS = 30

    df = pd.read_csv(
        f"Sustainability_Analysis/Reformat_data/30.csv")
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
        X, y, test_size=0.2)

    model = Sequential()
    model.add(LSTM(64, input_shape=(N_TIMESTEPS, len(data_columns))))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    model.fit(X_train, y_train, batch_size=30, epochs=100, verbose=1)
    model.save(f'./models/model_{N_TIMESTEPS}.h5')

    # for N_TIMESTEPS in tqdm(month_list):
    #     df = pd.read_csv(
    #         f"Sustainability_Analysis/Reformat_data/{N_TIMESTEPS}.csv")
    #     df.replace('Graduated', '1', inplace=True)
    #     df.replace('Retired', '0', inplace=True)

    #     data_columns = [
    #         'active_devs', 'num_commits', 'num_files', 'num_emails',
    #         'c_percentage', 'e_percentage', 'inactive_c', 'inactive_e',
    #         'c_nodes', 'c_edges', 'c_c_coef', 'c_mean_degree', 'c_long_tail',
    #         'e_nodes', 'e_edges', 'e_c_coef', 'e_mean_degree', 'e_long_tail']
    #     target_columns = ['status']

    #     scaler = MinMaxScaler(feature_range=(-1, 1))
    #     X_original = scaler.fit_transform(df[data_columns].values)
    #     X = reshape_X(X_original, n_timesteps=N_TIMESTEPS)
    #     y = reshape_y(df[target_columns].values, n_timesteps=N_TIMESTEPS)
    #     y = to_categorical(y.astype(int))
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=0.2)

    #     model = Sequential()
    #     model.add(LSTM(64, input_shape=(N_TIMESTEPS, len(data_columns))))
    #     model.add(Dropout(0.3))
    #     model.add(Dense(2, activation='softmax'))
    #     model.compile(loss='binary_crossentropy', optimizer=Adam())
    #     model.fit(X_train, y_train, batch_size=30, epochs=100, verbose=1)
        # model.save('./models/reduced_model_{}.h5'.format(N_TIMESTEPS))

        # y_pred = np.argmax(model.predict(X_test), axis=1)
        # y_true = np.argmax(y_test, axis=1)
        # # metrics = classification_report(y_true, y_pred, output_dict = True)

        # explainer = lime_tabular.RecurrentTabularExplainer(
        #     X, training_labels=y,
        #     feature_names=data_columns,
        #     discretize_continuous=False,
        #     class_names=['Graduated', 'Retired'])

        # pids = df['project'].unique()
        # pairs = [(pid, x) for pid, x in zip(pids, X) if pid not in project_set]

        # for pid, x in tqdm(pairs):
        #     project_set.add(pid)
        #     # return the all coefs of all timesteps
        #     # len(X[i]) is the number of timesteps
        #     exp = explainer.explain_instance(
        #         x,
        #         model.predict,
        #         num_features=len(data_columns) * len(x),
        #         labels=(1,))
        #     res = exp.as_list()
        #     dic = {}
        #     for feature, weight in res:
        #         # remove the month index
        #         *fs, m = feature.split('_')
        #         m = N_TIMESTEPS + int(m[1:])
        #         # combine and get the feature name
        #         fn = '_'.join(fs)
        #         # convert the time to quartile time
        #         quartile = str(math.ceil(4*m/N_TIMESTEPS))

        #         if fn not in dic:
        #             dic[fn] = {}
        #         if quartile not in dic[fn]:
        #             dic[fn][quartile] = []
        #         dic[fn][quartile].append(weight)

        #         with open(monthly_coefs_path, 'a') as f:
        #             things = [str(pid), fn, str(m), str(weight)]
        #             f.write(','.join(things))
        #             f.write('\n')

        #     with open(qutertile_coefs_path, 'a') as f:
        #         for fn in dic:
        #             for quartile in dic[fn]:
        #                 weight = sum(dic[fn][quartile]) / len(dic[fn][quartile])
        #                 things = [str(pid), fn, quartile, str(weight)]
        #                 f.write(','.join(things))
        #                 f.write('\n')

        #     with open(aggregated_coefs_path, 'a') as f:
        #         for fn in dic:
        #             weights = []
        #             for quartile in dic[fn]:
        #                 weights += dic[fn][quartile]
        #             weight = sum(weights) / len(weights)
        #             things = [str(pid), fn, str(weight)]
        #             f.write(','.join(things))
        #             f.write('\n')

    # print('all done.')