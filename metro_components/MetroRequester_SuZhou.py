from metro_components.StationManager import StationManager
from metro_components.LineManager import LineManager
import pandas as pd
import numpy as np
import pickle
from metro_data_convertor.convert_objects_to_dict import convert_objects_to_dict

import math


def haversine_distance(lat_lng_1, lat_lng_2):
    lat1, lng1 = math.radians(lat_lng_1[0]), math.radians(lat_lng_1[1])
    lat2, lng2 = math.radians(lat_lng_2[0]), math.radians(lat_lng_2[1])
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    R = 6371.0

    distance = R * c
    return distance


def request_suzhou_metro_data(excel_path, graph_sz_conn_root, station_index_root, station_manager_dict_root, result_API_root):
    df = pd.read_excel(excel_path, header=None)
    df_API = pd.read_excel(result_API_root, header=0)
    station_manager = StationManager()
    line_manager = LineManager()
    num_columns = len(df.columns)
    station_index = 0
    for col in range(num_columns):
        line_number = df.iloc[0, col]
        for row in range(1, len(df)):
            station_name = df.iloc[row, col]
            if station_name != 0 and station_name != '0':
                ori_station_num = len(list(station_manager.stations.values()))
                matching_row = df_API[df_API["station_name"] == station_name]
                lat_lng = []
                if not matching_row.empty:
                    lat = matching_row['lat'].values[0]
                    lng = matching_row['lng'].values[0]
                    lat_lng = [lat, lng]
                else:
                    print('No matching station name found.')

                station_manager.add_station(station_name, line_number, station_index, lat_lng)
                line_manager.add_line(line_number, station_manager.stations[station_name])
                if ori_station_num < len(list(station_manager.stations.values())):
                    station_index += 1

    for line_key, line_value in LineManager.lines.items():
        sequence_of_the_line = 0
        for station_of_the_line in line_value.stations:
            station_of_the_line.station_sequence_of_the_line[line_key] = sequence_of_the_line
            station_manager.stations[station_of_the_line.name].station_sequence_of_the_line[line_key] = sequence_of_the_line
            sequence_of_the_line += 1

    station_sequence=[]
    adj_mtx = np.zeros((len(list(station_manager.stations.values())), len(list(station_manager.stations.values()))))
    for line_key, line_value in LineManager.lines.items():
        for station_1_of_the_line in line_value.stations:
            for station_2_of_the_line in line_value.stations:
                if (station_1_of_the_line.station_sequence_of_the_line[line_key]-station_2_of_the_line.station_sequence_of_the_line[line_key]==1 or
                    station_2_of_the_line.station_sequence_of_the_line[line_key] - station_1_of_the_line.station_sequence_of_the_line[line_key] == 1):
                    adj_mtx[station_1_of_the_line.index, station_2_of_the_line.index]=1

    line_manager.adj_mx = adj_mtx

    graph_sz_conn_no_11 = open(graph_sz_conn_root, 'wb')
    pickle.dump(adj_mtx, graph_sz_conn_no_11)
    graph_sz_conn_no_11.close()

    station_index_no_11 = open(station_index_root, 'wb')
    pickle.dump(station_manager.station_index, station_index_no_11)
    station_index_no_11.close()

    index_station = {index: station for station, index in station_manager.station_index.items()}
    station_manager.index_station = index_station

    station_distance_matrix = np.zeros((len(list(station_manager.stations.values())), len(list(station_manager.stations.values()))))

    for key, metro_station_1 in station_manager.stations.items():
        for key, metro_station_2 in station_manager.stations.items():
            lat_lng_1 = metro_station_1.lat_lng
            lat_lng_2 = metro_station_2.lat_lng
            distance = haversine_distance(lat_lng_1, lat_lng_2)
            station_distance_matrix[metro_station_1.index][metro_station_2.index] = distance

    station_manager.station_distance_matrix = station_distance_matrix


    station_manager_dict_no_11 = open(station_manager_dict_root, 'wb')
    pickle.dump(station_manager, station_manager_dict_no_11)
    station_manager_dict_no_11.close()

    with open(station_manager_dict_root, 'rb') as f:
        station_manager = pickle.load(f, errors='ignore')

    print(station_manager.__dict__)
    print(StationManager.__dict__)
    StationManager_dict = convert_objects_to_dict(StationManager)
    station_manager_dict = convert_objects_to_dict(station_manager)

    merged_dict = {**{k: v for k, v in StationManager_dict.items() if k != 'index_station'},
                   **{k: v for k, v in station_manager_dict.items() if k != 'index_station'}}

    if 'index_station' in station_manager_dict:
        merged_dict['index_station'] = station_manager_dict['index_station']

    station_manager_dict_no_11 = open(station_manager_dict_root, 'wb')
    pickle.dump(merged_dict, station_manager_dict_no_11)
    station_manager_dict_no_11.close()
    station_manager.print_all_info()
    return station_manager, line_manager


"""project_root = Find_project_root()
excel_path = os.path.join(project_root, 'data', 'suzhou', 'Suzhou_zhandian_no_11.xlsx')
graph_sz_conn_root = os.path.join(project_root, 'data', 'suzhou', 'graph_sz_conn_no_11.pkl')
station_index_root = os.path.join(project_root, 'data', 'suzhou', 'station_index_no_11.pkl')
request_suzhou_metro_data(excel_path, graph_sz_conn_root, station_index_root)"""