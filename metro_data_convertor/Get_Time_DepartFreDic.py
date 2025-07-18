import pickle

import pandas as pd
import os
from metro_components import MetroRequester_SuZhou
from metro_components.Section import Section
from datetime import datetime, timedelta
from metro_data_convertor.Convert_objects_to_dict import Convert_objects_to_dict
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

def get_station_range(line, stations, start_name, end_name):
    start_station = None
    end_station = None
    start_index = None
    end_index = None

    for index, station in enumerate(stations):
        if station.name == start_name:
            start_station = station
            start_index = index
        if station.name == end_name:
            end_station = station
            end_index = index
        if start_station is not None and end_station is not None:
            break
    if start_station.station_sequence_of_the_line[line] > end_station.station_sequence_of_the_line[line]:
        return stations[end_index:start_index + 1]
    elif start_station.station_sequence_of_the_line[line] < end_station.station_sequence_of_the_line[line]:
        return stations[start_index:end_index + 1][::-1]
    return []

def process_operating_stations(file_path, current_time, current_date, line_manager, df_dic):
    StaInOper = {}
    workbook = pd.ExcelFile(file_path)

    subline_list = ["1号线", "2号线", "3号线", "4号线", "4号线支线", "5号线"]
    subline_dic = {"1号线":1, "2号线":2, "3号线":3, "4号线":4, "4号线支线":44, "5号线":5}

    for subline in subline_list:
        sheet_name = f"行车间隔与运营时间调整{subline}"
        if sheet_name in workbook.sheet_names:
            if sheet_name in df_dic:
                df = df_dic[sheet_name]
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df_dic[sheet_name] = df
            time_columns = [col for col in df.columns if ':' in str(col)]
            for col in time_columns[:]:
                try:
                    df[col] = df[col].apply(
                        lambda x: pd.to_datetime(x, format='%H:%M:%S').time() if isinstance(x, str) and ':' in x else x)
                except Exception as e:
                    print(f"Error converting column {col}: {e}")
                    time_columns.remove(col)

            for index, row in df.iterrows():
                row_date = pd.to_datetime(row['StartTime']).date()
                current_date = pd.to_datetime(current_date).date()
                if row_date == current_date:
                    first_depart = row['FirstDepart']
                    last_depart = row['LastDepart']

                    first_depart_time = pd.to_datetime(first_depart, format='%H:%M:%S').time()
                    last_depart_time = pd.to_datetime(last_depart, format='%H:%M:%S').time()
                    current_time_time = pd.to_datetime(current_time, format='%H:%M:%S').time()

                    if first_depart_time <= current_time_time <= last_depart_time:
                        sub_line_stations = line_manager.lines[subline_dic[subline]].stations
                        influ_line = row['InfluLine']
                        direc = row['Direc']
                        influ_sec1 = row['InfluSec1']
                        influ_sec2 = row['InfluSec2']
                        if (pd.isna(influ_sec1)) and (pd.isna(influ_sec2)):
                            influ_sec1 = sub_line_stations[0].name
                            influ_sec2 = sub_line_stations[-1].name
                        """
                        print("Row columns and types:")
                        for col in row.index:
                            print(f"{col}: {type(col)}" "================" f"{row[col]}: {type(row[col])}")
                        """
                        depart_freq = 0
                        for i in range(len(time_columns) - 1):
                            start_time = pd.to_datetime(time_columns[i], format='%H:%M:%S').time()
                            end_time = pd.to_datetime(time_columns[i + 1], format='%H:%M:%S').time()
                            if start_time <= current_time_time < end_time:
                                depart_freq = row[end_time]
                                break

                        stations_range = get_station_range(subline_dic[subline], sub_line_stations, influ_sec1, influ_sec2)
                        for i, station in enumerate(stations_range[:-1]):
                            start_station = station
                            end_station = stations_range[i + 1]
                            section = Section(start_station, end_station, subline_dic[subline])
                            section.depart_freq = depart_freq
                            StaInOper[start_station.name, end_station.name] = section

    return StaInOper,df_dic

def Get_Time_DepartFreDic(suzhou_sub_data_file_path, Time_DepartFreDic_filename, time_interval,
                          excel_path, graph_sz_conn_root, station_index_root,
                          start_date_str, end_date_str, station_manager_dict_root, result_API_root):
    start_date = datetime.strptime(start_date_str, '%Y/%m/%d')
    end_date = datetime.strptime(end_date_str, '%Y/%m/%d')
    # Assuming these functions and classes are already defined and imported
    station_manager, line_manager = MetroRequester_SuZhou.request_suzhou_metro_data(
        excel_path, graph_sz_conn_root, station_index_root, station_manager_dict_root, result_API_root)
    Time_DepartFreDic = {}
    df_dic={}
    current_date = start_date
    while current_date <= end_date:
        current_time = datetime.strptime('00:00:00', '%H:%M:%S')
        while current_time < datetime.strptime('23:59:59', '%H:%M:%S'):
            combined_datetime = datetime.combine(current_date, current_time.time())
            StaInOper, df_dic = process_operating_stations(suzhou_sub_data_file_path, current_time.strftime('%H:%M:%S'),
                                                   current_date.strftime('%Y/%m/%d'), line_manager, df_dic)
            Time_DepartFreDic[combined_datetime] = StaInOper
            """for key in StaInOper:
                print(key[0] + "-" + key[1])"""
            current_time += time_interval
        current_date += timedelta(days=1)

    converted_Time_DepartFreDic = Convert_objects_to_dict(Time_DepartFreDic)

    with open(Time_DepartFreDic_filename, 'wb') as f:
        pickle.dump(converted_Time_DepartFreDic, f)


'''project_root = Find_project_root()
Time_DepartFreDic_filename = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic.pkl')
suzhou_sub_data_file_path = os.path.join(project_root, 'data', 'suzhou', 'suzhou_sub_data.xlsx')
excel_path = os.path.join(project_root, 'data', 'suzhou', 'Suzhou_zhandian_no_11.xlsx')
graph_sz_conn_root = os.path.join(project_root, 'data', 'suzhou', 'graph_sz_conn_no_11.pkl')
station_index_root = os.path.join(project_root, 'data', 'suzhou', 'station_index_no_11.pkl')
time_interval = timedelta(minutes=15)
Get_Time_DepartFreDic(suzhou_sub_data_file_path, Time_DepartFreDic_filename, time_interval,
                      excel_path, graph_sz_conn_root, station_index_root, '2019/1/1', '2019/1/1')'''