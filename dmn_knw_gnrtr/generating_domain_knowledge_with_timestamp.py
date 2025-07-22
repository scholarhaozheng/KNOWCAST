import pickle
import os
import numpy as np
import torch
from metro_data_convertor.Find_project_root import Find_project_root

def generating_traffic_assignment_knowledge_with_timestamp(base_dir, prefix, repeated_or_not_repeated, seq_len):
    Time_DepartFreDic_Array_file_path = os.path.join(base_dir, f'{prefix}_Time_DepartFreDic_Array.pkl')
    Date_and_time_OD_path_dic_file_path = os.path.join(base_dir, f'Date_and_time_OD_path_cp_factors_dic.pkl')
    Time_DepartFreDic_Matrix_file_path=os.path.join(base_dir, f'{prefix}_Time_DepartFreDic_Matrix.pkl')
    Date_and_time_OD_path_Matrix_file_path=os.path.join(base_dir, f'{prefix}_Date_and_time_OD_path_Matrix.pkl')
    OD_feature_array_file_path=os.path.join(base_dir, f'{prefix}_OD_feature_array.pkl')
    main_data_file_path = os.path.join(base_dir, f'{prefix}.pkl')
    with open(Time_DepartFreDic_Array_file_path, 'rb') as f:
        Time_DepartFreDic_Array = pickle.load(f, errors='ignore')

    if repeated_or_not_repeated=="not_repeated":
        with open(Date_and_time_OD_path_dic_file_path, 'rb') as f:
            Date_and_time_OD_path_cp_factors_dic = pickle.load(f, errors='ignore')
        with open(main_data_file_path, 'rb') as f:
            main_data = pickle.load(f, errors='ignore')
        with open(OD_feature_array_file_path, 'rb') as f:
            OD_feature_array = pickle.load(f, errors='ignore')

        Time_DepartFreDic_Matrix = []
        Date_and_time_OD_path_cp_Matrix = []
        keys_sorted = sorted(main_data["xtime"])
        for idx in range(0, len(keys_sorted)):
            temp_list = [Time_DepartFreDic_Array[keys_sorted[idx - j]] for j in range(seq_len)]
            Date_and_time_OD_path_cp_factors_temp_list = [Date_and_time_OD_path_cp_factors_dic[keys_sorted[idx - j]] for j in range(seq_len)]

            Time_DepartFreDic_Matrix.append(temp_list)
            Date_and_time_OD_path_cp_Matrix.append(Date_and_time_OD_path_cp_factors_temp_list)

        output_file_Time_DepartFreDic_Matrix = open(Time_DepartFreDic_Matrix_file_path, 'wb')
        pickle.dump(Time_DepartFreDic_Matrix, output_file_Time_DepartFreDic_Matrix)

        output_Date_and_time_OD_path_Matrix = open(Date_and_time_OD_path_Matrix_file_path, 'wb')
        pickle.dump(Date_and_time_OD_path_cp_Matrix, output_Date_and_time_OD_path_Matrix)

        repeated_OD_feature_array_ = []

        file2 = open(os.path.join(base_dir, f'{prefix}_repeated_OD_feature_array.pkl'), 'wb')
        pickle.dump(repeated_OD_feature_array_, file2)
    else:
        sparse_tensors_5D = torch.load(os.path.join(base_dir, f'{prefix}_sparse_5d_tensor.pt'))

        with open(OD_feature_array_file_path, 'rb') as f:
            OD_feature_array = pickle.load(f, errors='ignore')

        Time_DepartFre_Array = Time_DepartFreDic_Array[next(iter(Time_DepartFreDic_Array))]

        log_dir = os.path.join(f"{base_dir}{os.path.sep}OD", prefix + '.pkl')
        with open(log_dir, 'rb') as f:
            matrix = pickle.load(f)['Time_DepartFreDic_Matrix']
        T = matrix.shape[0]
        repeated_OD_feature_array = np.repeat(np.expand_dims(OD_feature_array, axis=0), matrix.shape[0] + seq_len, axis=0)
        repeated_5D_OD_path_lst = [sparse_tensors_5D for _ in range(matrix.shape[0] + seq_len)]
        repeated_Time_DepartFre_Array = np.repeat(np.expand_dims(Time_DepartFre_Array, axis=0), matrix.shape[0] + seq_len,
                                                  axis=0)

        repeated_OD_feature_array_ = []
        repeated_5D_OD_path_lst_ = []
        repeated_Time_DepartFre_Array_ = []
        for i in range(0, T):
            features_temp_array = np.array([repeated_OD_feature_array[i + j] for j in range(seq_len)])
            temp_tdf_array = np.array([repeated_Time_DepartFre_Array[i + j] for j in range(seq_len)])
            repeated_OD_feature_array_.append(features_temp_array)
            repeated_Time_DepartFre_Array_.append(temp_tdf_array)

            temp_5D_path_lst = [repeated_5D_OD_path_lst[i + j] for j in range(seq_len)]
            repeated_5D_OD_path_lst_.append(temp_5D_path_lst)

        repeated_OD_feature_array_ = np.array(repeated_OD_feature_array_)
        repeated_Time_DepartFre_Array_ = np.array(repeated_Time_DepartFre_Array_, dtype=np.float16)

        output_file_path_5D = open(os.path.join(base_dir, f'{prefix}_repeated_sparse_5D_tensors.pt'), 'wb')
        torch.save(repeated_5D_OD_path_lst_, output_file_path_5D)

        file2 = open(os.path.join(base_dir, f'{prefix}_repeated_OD_feature_array.pkl'), 'wb')
        pickle.dump(repeated_OD_feature_array_, file2)

        file_tdf = open(os.path.join(base_dir, f'{prefix}_repeated_Time_DepartFre_Array.pkl'), 'wb')
        pickle.dump(repeated_Time_DepartFre_Array_, file_tdf)
