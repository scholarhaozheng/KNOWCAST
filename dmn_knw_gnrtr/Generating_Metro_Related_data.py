import pickle

from metro_components import MetroRequester_SuZhou
from metro_components.Route import Route
from metro_components.Path import Path
import heapq
import copy
import os
from metro_data_convertor.Find_project_root import Find_project_root
from datetime import timedelta
from metro_data_convertor.Generating_logit_probabilities import Generating_logit_probabilities
from metro_data_convertor.Get_Time_DepartFreDic import Get_Time_DepartFreDic
from metro_data_convertor.Process_Time_DepartFreDic import Process_Time_DepartFreDic
from metro_data_convertor.Reprocessing_OD_visiting_prob import Reprocessing_OD_visiting_prob

def get_same_lines(from_station, to_station):
    line_numbers = []
    for each_line in from_station.lines:
        if each_line in to_station.lines:
            line_numbers.append(each_line)
    return line_numbers



def dijkstra(v_matrix, start_index, end_index, bias):
    """
    Initialize the distance and path record arrays.

    :param v_matrix: A 2D list representing the graph, where each element contains edge information.
    :param start_index: The index of the starting vertex.
    :param end_index: The index of the target vertex.
    :param bias: An additional cost to add to the distance for each edge.
    :return: A tuple containing the shortest distance and the path from start to end.
    """

    book = [0] * len(v_matrix)
    dis = []
    for i in range(0, len(v_matrix)):
        dis.append((v_matrix[start_index][i].stops, Path().add_path(v_matrix[start_index][i])))
    dis[start_index] = (0,Path())
    while True:
        candidates = [(d, idx) for idx, (d, _) in enumerate(dis) if book[idx] == 0]
        if candidates:
            u = min(candidates)[1]
        else:
            break
        if dis[u][0] == 9999:
            break
        for v in range(len(v_matrix[u])):
            if not book[v]:
                if v_matrix[u][v].stops < 9999:
                    new_distance = dis[u][0] + v_matrix[u][v].stops + bias
                    if new_distance < dis[v][0]:
                        dis[v] = (new_distance, dis[u][1].add_path(v_matrix[u][v]))
                elif v_matrix[u][v].stops == 9999:
                    continue
        book[u] = 1
    return dis[end_index]

def yen_ksp(start_station, terminal_station, k, v_matrix, bias, station_index, line_manager, station_manager):
    original_v_matrix = copy.deepcopy(v_matrix)
    start_index = station_index[start_station]
    terminal_index = station_index[terminal_station]
    distance, first_path = dijkstra(v_matrix, start_index, terminal_index, bias)
    paths = [first_path]
    potential_k_paths = []

    def contains_subsequence(sequence, subsequence):
        """Check if the sequence contains a subsequence that matches the subsequence in order."""
        sub_len = len(subsequence)
        for i in range(len(sequence) - sub_len + 1):
            if sequence[i:i + sub_len] == subsequence:
                return True
        return False

    for i in range(1, k):
        the_latest_added_path=paths[-1]
        for j in range(len(the_latest_added_path.routes) - 1):
            route_belongs_dif_line=the_latest_added_path.routes[j]
            for spur_node in route_belongs_dif_line.stations:
                spur_node_index = station_index[spur_node.name]
                route_list_contains_root_path = the_latest_added_path.routes[:j + 1]
                root_path = Path()
                for route_with_partial_root in route_list_contains_root_path:
                    list_route_of_partial_root = []
                    for station in route_with_partial_root.stations:
                        list_route_of_partial_root.append(station)
                        if station.name == spur_node.name:
                            break
                    if(len(list_route_of_partial_root)>1):
                        route_of_partial_root = Route()
                        route_of_partial_root.construct_route(line_manager, station_manager, list_route_of_partial_root[0],
                                                              list_route_of_partial_root[-1], get_same_lines(list_route_of_partial_root[0],list_route_of_partial_root[-1]))
                        root_path=root_path.add_path(route_of_partial_root)
                spur_nodes_indices_in_stations_and_paths = [
                    (spur_nodes_path_index, spur_nodes_station_index)
                    for spur_nodes_path_index, path in enumerate(paths)
                    for spur_nodes_station_index, station in enumerate(path.station_visit_sequence)
                    if station.name == spur_node.name
                ]
                # Locate all indices of paths in 'paths' that contain 'spur_node' and all indices of routes in 'path.routes'.
                # Then find all arcs in 'path' that pass through the 'spur_node', identify their endpoints,
                # and ensure that all arcs from 'spur_node' to these endpoints are not traversed.
                # (Note that arcs that pass through 'spur_node' and the endpoint but do not start from 'spur_node' and the endpoint should also be blocked.)
                spur_node_to_station_list = []
                for spur_node_to_path_station_index in spur_nodes_indices_in_stations_and_paths:
                    spur_node_to_station=paths[spur_node_to_path_station_index[0]].station_visit_sequence[
                        spur_node_to_path_station_index[1] + 1]
                    spur_node_to_station_list.append(spur_node_to_station)
                    for spur_node_to_station in spur_node_to_station_list:
                        # Find all arcs in 'v_matrix' that contain 'spur_node' to all points in 'spur_node_to_index_list' and set their values to 9999.
                        # First, determine whether the route from 'spur_node' to 'spur_node_to_index_list' is in the forward or reverse direction.
                        # After identifying this, find the corresponding routes and then locate all matching indices.
                        same_line = get_same_lines(spur_node, spur_node_to_station)[0]
                        selected_line = line_manager.lines[same_line]
                        blocked_routes=[]
                        for from_stations in selected_line.stations:
                            for to_stations in selected_line.stations:
                                if spur_node.station_sequence_of_the_line[same_line] > spur_node_to_station.station_sequence_of_the_line[same_line]:  # 反着开
                                    if ((from_stations.station_sequence_of_the_line[same_line]>=spur_node.station_sequence_of_the_line[same_line]) & (to_stations.station_sequence_of_the_line[same_line]<=spur_node_to_station.station_sequence_of_the_line[same_line])):
                                        blocked_routes.append((from_stations,to_stations))
                                elif spur_node.station_sequence_of_the_line[same_line] < spur_node_to_station.station_sequence_of_the_line[same_line]:
                                    if ((from_stations.station_sequence_of_the_line[same_line] <= spur_node.station_sequence_of_the_line[same_line]) & (to_stations.station_sequence_of_the_line[same_line] >= spur_node_to_station.station_sequence_of_the_line[same_line])):
                                        blocked_routes.append((from_stations, to_stations))
                        for blocked_route in blocked_routes:
                            v_matrix[blocked_route[0].index][blocked_route[1].index].stops = 9999
                # Calculate the shortest path from spur_node to terminal.
                distance_plus_bias, spur_path = dijkstra(v_matrix, spur_node_index, terminal_index, bias)
                total_stops = sum(route.stops for route in spur_path.routes)
                if (total_stops < 9999) & (total_stops > 0):
                    new_path=copy.deepcopy(root_path)
                    for route_of_spur_path in spur_path.routes:
                        new_path=new_path.add_path(route_of_spur_path)
                    heapq.heappush(potential_k_paths, (distance_plus_bias, new_path))
                # Restore the original graph.
                v_matrix = copy.deepcopy(original_v_matrix)

        if(potential_k_paths==[]):
            continue
        duplicate_or_loop = True
        while duplicate_or_loop:
            duplicate_or_loop = False
            if (potential_k_paths == []):
                continue
            potential_new_path = heapq.heappop(potential_k_paths)[1]
            for path in paths:
                if path.station_visit_sequence_index == potential_new_path.station_visit_sequence_index:
                    duplicate_or_loop = True
                    break
            if len(set(potential_new_path.station_visit_sequence_index)) < len(
                    potential_new_path.station_visit_sequence_index):
                duplicate_or_loop = True
            if duplicate_or_loop==False:
                paths.append(potential_new_path)
    return paths

def find_routes(exist_path_number, OD_path_dic, section_path_dic, path_section_dic, start_station, terminal_station, v_matrix, bias, station_index, line_manager, station_manager, k):
    top_k_paths = yen_ksp(start_station, terminal_station, k, v_matrix, bias, station_index, line_manager, station_manager)
    print("Top", k, "routes from", start_station, "to", terminal_station)
    paths_info = []
    for path in top_k_paths:
        path_info = {
            "path_number": exist_path_number,
            "from_stop": path.from_stop,
            "to_stop": path.to_stop,
            "start_index": path.start_index,
            "end_index": path.end_index,
            "routes": path.routes,
            "station_visit_sequence_index": path.station_visit_sequence_index,
            "station_visit_sequence": path.station_visit_sequence,
            "number_of_stations": len(path.station_visit_sequence),
            "number_of_transfers": len(path.routes)-1,
        }
        paths_info.append(path_info)
        print("Path number", exist_path_number, ":")
        for each_route in path.routes:
            print_route_info(each_route, line_manager)
            for sta_index in range(len(each_route.stations)-1):
                station_name_pair_keys = (each_route.stations[sta_index].name,each_route.stations[sta_index+1].name)
                station_pair_values = (each_route.stations[sta_index], each_route.stations[sta_index + 1])
                if exist_path_number in path_section_dic:
                    path_section_dic[exist_path_number].append(station_pair_values)
                else:
                    path_section_dic[exist_path_number] = [station_pair_values]
                if station_name_pair_keys in section_path_dic:
                    section_path_dic[station_name_pair_keys].append(path_info)
                else:
                    section_path_dic[station_name_pair_keys] = [path_info]
        exist_path_number += 1
    OD_path_dic[(start_station, terminal_station)] = paths_info
    return exist_path_number, OD_path_dic, section_path_dic, path_section_dic

def print_route_info(each_route, line_manager):
    print("Take line " + str(
        each_route.line_number) + " from " + each_route.from_stop + " to " + each_route.to_stop + " (" + str(
        each_route.stops) + " stops)")
    line_manager.print_stops(each_route.line_number, each_route.from_stop, each_route.to_stop)

def Generating_Metro_Related_data(excel_path, graph_sz_conn_root, station_index_root, station_manager_dict_root,
                                  result_API_root, suzhou_sub_data_file_path, Time_DepartFreDic_filename, time_interval,
                                  Time_DepartFreDic_file_path, Time_DepartFreDic_Array_file_path,
                                  OD_path_visit_prob_array_file_path, train_dict_file_path, OD_path_visit_prob_dic_file_path,
                                  train_filename, start_date_str, end_date_str):
    # station_manager, line_manager = MetroRequester.request_shanghai_metro_data()
    station_manager, line_manager = MetroRequester_SuZhou.request_suzhou_metro_data(excel_path, graph_sz_conn_root, station_index_root, station_manager_dict_root, result_API_root)
    stations_list = list(station_manager.stations.values())
    station_index = station_manager.station_index
    v_matrix = []
    book = []
    dis = []

    # Initialize your StationManager and LineManager here
    # station_manager = StationManager()
    # line_manager = LineManager()
    def get_input_or_default(prompt, default):
        while True:
            user_input = input(prompt)
            if user_input == "":
                return default
            try:
                return int(user_input)
            except ValueError:
                print("Please enter a valid number, or press Enter to use the default value.")

    k = get_input_or_default(
        "Enter the number of routes to query K (default is 3, press Enter to use the default value): ", 3)
    bias = get_input_or_default(
        "What is the time equivalent of one transfer compared to traveling a few stops? bias (default is 3, press Enter to use the default value): ",
        3)
    for i in range(0, len(stations_list)):
        v_matrix.append([])
        for j in range(0, len(stations_list)):
            same_lines = get_same_lines(stations_list[i], stations_list[j])
            new_route = Route()
            new_route.construct_route(line_manager, station_manager, stations_list[i], stations_list[j], same_lines)
            v_matrix[i].append(new_route)

    while True:
        execution_mode = input("Please enter 'hand' for manual input, or 'auto' for automatic traversal: ")
        exist_path_number = 1
        OD_path_dic = {}
        section_path_dic = {}
        path_section_dic = {}

        if execution_mode == '' or execution_mode == 'hand':
            def get_valid_station_input(prompt, station_index):
                while True:
                    station_name = input(prompt)
                    if station_name in station_index:
                        return station_name
                    else:
                        print("The entered station is invalid, please try again.")

            start_station = get_valid_station_input("Please enter the starting station: ", station_index)
            terminal_station = get_valid_station_input("Please enter the destination station: ", station_index)
            print(f"==================== Paths from {start_station} to {terminal_station}. ====================")
            exist_path_number, OD_path_dic, section_path_dic, path_section_dic = (
                find_routes(exist_path_number, OD_path_dic, section_path_dic, path_section_dic, start_station,
                            terminal_station, v_matrix, bias, station_index, line_manager, station_manager, k))
            break
        elif execution_mode == 'auto':
            for i in range(len(stations_list)):
                for j in range(len(stations_list)):
                    if i != j:
                        v_matrix_ori = copy.deepcopy(v_matrix)
                        start_station = stations_list[i].name
                        terminal_station = stations_list[j].name
                        print(f"====================从 {start_station} 到 {terminal_station} 的路径。====================")
            exist_path_number, OD_path_dic, section_path_dic, path_section_dic = (
                find_routes(exist_path_number, OD_path_dic, section_path_dic, path_section_dic, start_station,
                            terminal_station, v_matrix_ori, bias, station_index, line_manager, station_manager, k))
            train_dict = {
                'OD_path_dic': OD_path_dic,
                'section_path_dic': section_path_dic,
                'path_section_dic': path_section_dic,
            }

            with open(train_filename, 'wb') as f:
                pickle.dump(train_dict, f)

            print("Automatic traversal completed.")
            print("Program ended.")
            break
        else:
            print("Invalid input, please enter 'hand' or 'auto'.")
