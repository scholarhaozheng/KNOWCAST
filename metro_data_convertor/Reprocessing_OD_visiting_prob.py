import numpy as np
import pickle


def Reprocessing_OD_visiting_prob(OD_path_visit_prob_dic_file_path, OD_path_visit_prob_array_file_path):
    with open(OD_path_visit_prob_dic_file_path, 'rb') as f:
        OD_path_visit_prob_dic = pickle.load(f, errors='ignore')

    OD_path_visit_prob_array = {}

    for key, value in OD_path_visit_prob_dic.items():
        pass_sections_list = value['pass_sections_list']

        for sec_dic in pass_sections_list:
            section_line = sec_dic['section_line']

            if section_line not in OD_path_visit_prob_array:
                OD_path_visit_prob_array[section_line] = np.zeros((154, 1))

            line_num = section_line
            start_station_index = sec_dic['start_station']['index']
            terminal_station_index = sec_dic['terminal_station']['index']
            visit_prob = sec_dic['visit_prob']

            OD_path_visit_prob_array[line_num][start_station_index] += visit_prob
            OD_path_visit_prob_array[line_num][terminal_station_index] += visit_prob

    for line, array in OD_path_visit_prob_array.items():
        print(f"Line: {line}, Array: \n{array}")

    with open(OD_path_visit_prob_array_file_path, 'wb') as f:
        pickle.dump(OD_path_visit_prob_array, f)

"""project_root = Find_project_root()
OD_path_visit_prob_array_file_path = os.path.join(project_root, 'data', 'suzhou', 'OD_path_visit_prob_array.pkl')
OD_path_visit_prob_dic_file_path = os.path.join(project_root, 'data', 'suzhou', 'OD_path_visit_prob_dic.pkl')
Reprocessing_OD_visiting_prob(OD_path_visit_prob_dic_file_path, OD_path_visit_prob_array_file_path)"""