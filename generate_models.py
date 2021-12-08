from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    RANDOM_STATE = 42

    for n_timesteps in range(1, 31):
        df = pd.read_csv(
            f"Sustainability_Analysis/Reformat_data/{n_timesteps}.csv")
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
        X = reshape_X(X_original, n_timesteps=n_timesteps)
        y = reshape_y(df[target_columns].values, n_timesteps=n_timesteps)
        y = to_categorical(y.astype(int))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)

        model = Sequential()
        model.add(LSTM(64, input_shape=(n_timesteps, len(data_columns))))
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
        model.save('models/model_' + str(n_timesteps) + '.h5')
