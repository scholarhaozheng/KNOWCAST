import pickle
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import torch
import pydotplus
import os
from sklearn.preprocessing import OneHotEncoder

from metro_data_convertor.Find_project_root import Find_project_root

# Generating pytorch geographical-temporal signal
# Used in generating trip-generation-based loss function

def PYGT_signal_generation(base_dir, prefix, station_manager_dict_name, graph_sz_conn_no_name,
                           station_manager_dict_file_path, graph_sz_conn_root, train_result_array_OD_or_DO_file_path,
                           normalization_params_file_path, str_prdc_attr, Using_lat_lng_or_index):

    with open(station_manager_dict_file_path, 'rb') as f:
        station_manager_dict = pickle.load(f, errors='ignore')
    with open(graph_sz_conn_root, 'rb') as f:
        graph_sz_conn_no_11 = pickle.load(f, errors='ignore')
    with open(train_result_array_OD_or_DO_file_path, 'rb') as f:
        train_result_array_OD_or_DO = pickle.load(f, errors='ignore')

    signal_dict_filename = os.path.join(base_dir, f'{prefix}_{str_prdc_attr}_signal_dict.pkl')
    Trip_Generation_TYPE=[]
    if str_prdc_attr=="prdc":
        Trip_Generation_TYPE='Trip_Production_In_Station_or_Out_Station'
    elif str_prdc_attr=="attr":
        Trip_Generation_TYPE="Trip_Attraction_In_Station_or_Out_Station"
    else:
        Trip_Generation_TYPE=[]

    train_result_array_Out_Station = train_result_array_OD_or_DO[Trip_Generation_TYPE]
    timestamps = list(train_result_array_Out_Station.keys())
    unix_timestamps = [ts.timestamp() for ts in timestamps]
    df_time = pd.DataFrame({
        'datetime': timestamps,
        'unix_timestamp': unix_timestamps
    })
    df_time['weekday'] = df_time['datetime'].dt.weekday
    df_time['minutes_of_day'] = df_time['datetime'].dt.hour * 60 + df_time['datetime'].dt.minute

    features = []
    targets = []
    additional_feature = []

    latitudes = []
    longitudes = []
    station_indices = []

    for station in station_manager_dict['stations'].values():
        latitudes.append(station["lat_lng"][0])
        longitudes.append(station["lat_lng"][1])
        station_indices.append(station["index"])

    latitudes = np.array(latitudes).reshape(-1, 1)
    longitudes = np.array(longitudes).reshape(-1, 1)
    station_indices = np.array(station_indices).reshape(-1, 1)

    scaler_lat = StandardScaler()
    scaler_lng = StandardScaler()

    latitudes_standardized = scaler_lat.fit_transform(latitudes)
    longitudes_standardized = scaler_lng.fit_transform(longitudes)

    onehot_encoder = OneHotEncoder(sparse=False)
    station_indices_encoded = onehot_encoder.fit_transform(station_indices)

    standardized_lat_lng_index = {
        station_id: (latitudes_standardized[i][0], longitudes_standardized[i][0], station_indices_encoded[i])
        for i, station_id in enumerate(station_manager_dict['stations'].keys())
    }

    for timestamp, value in train_result_array_Out_Station.items():
        num_stations = len(station_manager_dict['stations'])
        ST_F_F = np.zeros((num_stations, 2 + station_indices_encoded.shape[1]))
        ST_T = np.zeros(num_stations)

        df_time_row = df_time[df_time['datetime'] == timestamp]
        if df_time_row.empty:
            continue

        for station_id, station in station_manager_dict['stations'].items():
            ST_F_F[station["index"], 0] = df_time_row['weekday'].values[0]
            ST_F_F[station["index"], 1] = df_time_row['minutes_of_day'].values[0]
            ST_F_F[station["index"], 2:] = standardized_lat_lng_index[station_id][2]

            ST_T[station["index"]] = value[station["index"], 0]

        features.append(ST_F_F)
        targets.append(ST_T)
        additional_feature.append([df_time_row['weekday'].values[0], df_time_row['minutes_of_day'].values[0]])

    features = [np.array(f, dtype=np.float32) for f in features]
    targets = [np.array(t, dtype=np.int64) for t in targets]
    additional_feature = [np.array(af, dtype=np.float32) for af in additional_feature]

    non_onehot_features = [f[:, :2] for f in features]
    all_non_onehot_features = np.concatenate(non_onehot_features, axis=0)

    if(prefix=="train"):
        feature_means = all_non_onehot_features.mean(axis=0)
        feature_stds = all_non_onehot_features.std(axis=0)
        normalized_non_onehot_features = [(f[:, :2] - feature_means) / feature_stds for f in features]

        normalized_features = [np.concatenate((norm_f, f[:, 2:]), axis=1) for norm_f, f in
                               zip(normalized_non_onehot_features, features)]

        normalization_params = {'means': feature_means, 'stds': feature_stds}
        with open(normalization_params_file_path, 'wb') as f:
            pickle.dump(normalization_params, f)
    else:
        with open(normalization_params_file_path, 'rb') as f:
            normalization_params_dict = pickle.load(f, errors='ignore')
        feature_means=normalization_params_dict['means']
        feature_stds=normalization_params_dict['stds']
        normalized_non_onehot_features = [(f[:, :2] - feature_means) / feature_stds for f in features]
        normalized_features = [np.concatenate((norm_f, f[:, 2:]), axis=1) for norm_f, f in
                               zip(normalized_non_onehot_features, features)]

    row, col = np.nonzero(graph_sz_conn_no_11)
    edge_index = np.array([row, col])
    edge_weight = graph_sz_conn_no_11[row, col]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    signal_dict = {
        'features': normalized_features,
        'targets': targets,
        'additional_feature': additional_feature,
        'edge_index': edge_index,
        'edge_weight': edge_weight
    }
    with open(signal_dict_filename, 'wb') as f:
        pickle.dump(signal_dict, f)