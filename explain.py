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
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()
sns.set(font_scale=0.8)

class SustainabilityExplainer:

    def __init__(
            self,
            dataset_train,
            feature_names,
            class_names,
            random_state=None):
        self.dataset_train = dataset_train
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state

    def explain_by_lime(self, X, model, save_plot=None):
        """
        """
        explainer = lime_tabular.RecurrentTabularExplainer(
                training_data=self.dataset_train[0],
                training_labels=self.dataset_train[1],
                feature_names=self.feature_names,
                class_names=['Graduated', 'Retired'],
                discretize_continuous=False,
                random_state=self.random_state)

        exp = explainer.explain_instance(
            X,
            model.predict,
            labels=(0, 1),
            num_features=len(X) * len(self.feature_names),
            num_samples=5000)

        feature_score_pairs = exp.as_list()

        # Split timestamp and feature name and sort by timestamp.
        feature_t_score_triples \
            = [(f.split('-')[0][:-2], f.split('-')[1], s)
                for f, s in feature_score_pairs]

        lime_values = {i: {} for i in range(len(X))}
        for f, t, s in feature_t_score_triples:
            lime_values[int(t)][f] = s

        # Sort keys to be consistent to the input.
        for i in range(len(X)):
            lime_values[i] = {
                f: lime_values[i][f] for f in self.feature_names}

        if save_plot is not None:
            df_lime_values = pd.DataFrame(lime_values).T
            ax = sns.boxplot(data=df_lime_values)
            ax.tick_params(axis='x', rotation=90)
            plt.savefig(save_plot)

        return lime_values

    def explain_by_shap(self, x, model, save_plot=None):
        explainer = shap.DeepExplainer(model, self.dataset_train[0])

        x = np.expand_dims(x, axis=0)
        shap_values = explainer.shap_values(x)
        shap_values = [np.mean(sv, axis=1) for sv in shap_values]

        if save_plot is not None:
            fig = shap.force_plot(
                base_value=explainer.expected_value[1],
                shap_values=shap_values[1],
                features=data_columns,
                matplotlib=True,
                show=False)
            fig.savefig(save_plot)

        return shap_values


if __name__ == '__main__':

    def reshape_X(seq, n_timesteps):
        N = int(len(seq) / n_timesteps)
        if N <= 0:
            raise ValueError('need more data.')

        nf = seq.shape[1]
        new_seq = np.zeros((N, n_timesteps, nf))
        for i in range(N):
            new_seq[i, :, :] = seq[i:i + n_timesteps]

        return new_seq


    def reshape_y(seq, n_timesteps):
        N = int(len(seq) / n_timesteps)
        if N <= 0:
            raise ValueError('need more data.')

        nf = seq.shape[1]
        new_seq = np.zeros((N, nf))
        for i in range(N):
            new_seq[i, :] = seq[i * n_timesteps]

        return new_seq

    N_TIMESTEPS = 8
    RANDOM_STATE = 42

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
        X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)

    # model = Sequential()
    # model.add(LSTM(64, input_shape=(N_TIMESTEPS, len(data_columns))))
    # model.add(Dropout(0.3))
    # model.add(Dense(2, activation='softmax'))
    # model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=Adam(),
    #     metrics=['accuracy'])
    # model.fit(
    #     X_train,
    #     y_train,
    #     batch_size=30,
    #     epochs=100,
    #     validation_data=(X_test, y_test),
    #     verbose=1)
    # model.save('models/model_' + str(N_TIMESTEPS) + '.h5')

    model = load_model('models/model_' + str(N_TIMESTEPS) + '.h5')
    # trajectories = explain_consecutive_instances(
    #     data_points=X_test[0],
    #     feature_names=data_columns,
    #     model=model,
    #     training_data=X_train,
    #     training_labels=y_train)

    # plot_trajectories(trajectories)

    explainer = SustainabilityExplainer(
        dataset_train=(X_train, y_train),
        feature_names=data_columns,
        class_names=['Graduated', 'Retired'],
        random_state=RANDOM_STATE)

    explainer.explain_by_lime(
        X_test[0], model, save_plot="./test_lime.png")

    explainer.explain_by_shap(
        X_test[0], model, save_plot="./test_shap.png")
