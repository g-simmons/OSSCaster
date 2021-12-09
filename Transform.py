import numpy as np
import pandas as pd
import os
from os import listdir

data_path = '/home/jiujiu/Downloads/Sustainability_Analysis/Reformat_data/test/'


# df = pd.read_csv(data_path + '{}.csv'.format(5))
# df.replace('Graduated', 1, inplace=True)
# df.replace('Retired', 0, inplace=True)
# df_dict = df.to_dict()
# print(df_dict)

def get_scaling_params(sim):
    real_idx = list(sim.keys())

    # df = pd.DataFrame({k: sim[k] for k in real_idx})
    means = {}
    stds = {}
    seq_lengths = sim['sequence_lengths']
    for k in real_idx:
        if k not in ('treatments', 'sequence_lengths'):
            active_values = []
            for i in range(seq_lengths.shape[0]):
            # for i in range(len(sim[k])):
                active_values += list(sim[k][i])

            means[k] = np.mean(active_values)
            stds[k] = np.std(active_values)

    return pd.Series(means), pd.Series(stds)


def transform(num_month, id_entry):
    # df = pd.read_csv(data_path + '{}.csv'.format(num_month))
    df = pd.read_csv(data_path + '16.csv')
    df.replace('Graduated', 1, inplace=True)
    df.replace('Retired', 0, inplace=True)
    df_dict = df.to_dict()

    # print(df_dict[/'project'].keys())

    num_project = int(len(df_dict['project']) / 24)

    output = {}
    for key in df_dict.keys():
        if key != 'project' and key != 'month':
            if key == 'treatments':
                output[key] = np.empty((num_project, num_month), dtype='|S20')
            else:
                output[key] = np.zeros((num_project, num_month))
    output['sequence_lengths'] = np.full(num_project, int(num_month * 0.5))
    id2num = {}  # convert project id to number
    i = 0
    # convert project id to number
    for j in df_dict['project'].keys():
        if df_dict['project'][j] not in id2num.keys():
            id2num[df_dict['project'][j]] = i
            i += 1
    # reformat data
    for i in df_dict['project'].keys():  # iterate through every entry in dictionary
        if df_dict['month'][i] <= id_entry or df_dict['month'][i] > id_entry + num_month:
            continue
        for key in df_dict.keys():  # every feature
            if key != 'project' and key != 'month':
                if key == 'prob_grad':
                    if pd.isna(df_dict[key][i]):
                        output[key] \
                            [id2num[df_dict['project'][i]]] \
                            [i % num_month] = df_dict['status'][i]  # stuffing
                    else:
                        output[key] \
                            [id2num[df_dict['project'][i]]] \
                            [i % num_month] = df_dict[key][i]
                elif key == 'treatments':
                    if pd.isna(df_dict[key][i]):
                        output[key] \
                            [id2num[df_dict['project'][i]]] \
                            [i % num_month] = 'active_devs-n'  # stuffing: no action
                    else:
                        output[key] \
                            [id2num[df_dict['project'][i]]] \
                            [i % num_month] = df_dict[key][i]
                    # print('original: ', df_dict[key][i], ' now: ',
                    #       output[key][id2num[df_dict['project'][i]]][i % num_month])
                # print(key, df_dict['project'][i], i % num_month)
                else:
                    output[key] \
                        [id2num[df_dict['project'][i]]] \
                        [i % num_month] = df_dict[key][i]
    return output, num_project


def truncate(dict, start, end, indexes):
    output = {}
    for key in dict.keys():
        output[key] = dict[key][start:end]
    for i in range(end - start):
        output[key][i] = dict[key][indexes[start + i]]
    return output

def get_data(model_root, id_entry, num_month):
    dataset, num_project = transform(num_month, id_entry)

    training_index = int(num_project * 0.8)
    validation_index = int(num_project * 0.9)
    test_index = num_project

    pickle_file = os.path.join(model_root, 'new_cancer_sim.p')

    indexes = np.arange(num_project)
    np.random.shuffle(indexes)
    # print(indexes)

    training_data = truncate(dataset, 0, training_index, indexes)
    validation_data = truncate(dataset, training_index, validation_index, indexes)
    test_data = truncate(dataset, validation_index, test_index, indexes)
    scaling_data = get_scaling_params(training_data)

    pickle_map = {  # 'chemo_coeff': chemo_coeff,
        # 'radio_coeff': radio_coeff,
        'num_time_steps': num_month,
        'training_data': training_data,
        'validation_data': validation_data,
        'test_data': test_data,
        'test_data_factuals': test_data,
        'test_data_seq': test_data,
        'scaling_data': scaling_data,
        'window_size': 20}
    return pickle_map


if __name__ == "__main__":
    a, num_project = transform(16, 9)
    print(len(a['prob_grad']))
    print(num_project)
    # b = truncate(a, 100, 200)
    # print(b)
