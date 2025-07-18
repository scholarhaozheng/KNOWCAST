import numpy as np
import pickle
from metro_components import MetroRequester_SuZhou
from metro_components.OD import OD
from metro_components.Section import Section
import os
import sys
from metro_data_convertor.Convert_objects_to_dict import Convert_objects_to_dict

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
components_dir = os.path.join(parent_dir, 'metro_components')
sys.path.append(components_dir)

print(sys.path)

def calculate_logit_probabilities(paths_info):
	utilities = np.array(
		[-path_info["number_of_stations"] - path_info["number_of_transfers"] * 3 for path_info in paths_info])
	exp_utilities = np.exp(utilities)
	probabilities = exp_utilities / np.sum(exp_utilities)

	for i, path_info in enumerate(paths_info):
		path_info["probability"] = probabilities[i]

	return paths_info

def Generating_logit_probabilities(train_dict_file_path, OD_path_visit_prob_dic_file_path,
                                           station_manager_dict_file_path, graph_sz_conn_root, station_manager_dict_root, station_index_root, result_API_root):
	with open(station_manager_dict_file_path, 'rb') as f:
		station_manager_dict = pickle.load(f, errors='ignore')
	with open(train_dict_file_path, 'rb') as f:
		train_dict = pickle.load(f, errors='ignore')
	OD_path_dic=train_dict["OD_path_dic"]
	OD_path_visit_prob_dic={}
	for key, paths_info in OD_path_dic.items():
		start_station_name = key[0]
		terminal_station_name = key[1]
		paths_info = calculate_logit_probabilities(paths_info)
		sec_vst_freq_dic = {}
		pass_sections_list=[]
		for path_info in paths_info:
			probability = path_info["probability"]
			for route in path_info["routes"]:
				for sta_index in range(len(route.stations) - 1):
					start_idx = route.stations[sta_index].index
					end_idx = route.stations[sta_index + 1].index
					pass_arc=(route.stations[sta_index].name,route.stations[sta_index + 1].name)
					pass_section = Section(route.stations[sta_index], route.stations[sta_index + 1], route.line_number)
					pass_section.visit_prob += probability
					pass_sections_list.append(pass_section)
					if pass_arc in sec_vst_freq_dic:
						sec_vst_freq_dic[pass_arc] += probability
					else:
						sec_vst_freq_dic[pass_arc] = probability
					print(station_manager_dict['index_station'][start_idx]+"-"+station_manager_dict['index_station'][end_idx]+str(sec_vst_freq_dic[pass_arc]))
		od_instance = OD(start_station_name, terminal_station_name)
		od_instance.pass_sections_list = pass_sections_list
		od_instance.sec_vst_freq_dic = sec_vst_freq_dic
		OD_path_visit_prob_dic[key] = (od_instance)

	converted_OD_path_visit_prob_dic = Convert_objects_to_dict(OD_path_visit_prob_dic)

	with open(OD_path_visit_prob_dic_file_path, 'wb') as f:
		pickle.dump(converted_OD_path_visit_prob_dic, f)
		
"""project_root = Find_project_root()
train_dict_file_path = os.path.join(project_root, 'data', 'suzhou', 'train_dict.pkl')
OD_path_visit_prob_dic_file_path = os.path.join(project_root, 'data', 'suzhou', 'OD_path_visit_prob_dic.pkl')
Generating_logit_probabilities(train_dict_file_path, OD_path_visit_prob_dic_file_path)
"""