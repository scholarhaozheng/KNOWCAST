import os
from functools import partial
import numpy as np
import pickle

import psutil
import torch
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tensorly.decomposition import parafac
import tensorly as tl

Parallel_or_not = False
Batch_or_not = True
One_by_one_or_not = False

def batch_processing(Time_DepartFreDic, OD_path_dic, OD_feature_array_dic,
                        Date_and_time_OD_path_cp_factors_dic,
                        station_manager_dict, batch_size, OD_feature_array_file_path,
                        Date_and_time_OD_path_cp_factors_dic_file_path, processing_function, device):

    keys = list(Time_DepartFreDic.keys())
    total_batches = (len(keys) + batch_size - 1) // batch_size

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPU！")

    for batch_idx in range(total_batches):
        batch_keys = keys[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_dict = {key: Time_DepartFreDic[key] for key in batch_keys}

        results = []
        with ProcessPoolExecutor(max_workers=min(len(batch_dict), os.cpu_count() - 20)) as executor:
            futures = []
            for i, item in enumerate(batch_dict.items()):
                gpu_id = i % num_gpus  # 轮转分配 GPU
                # 提交任务时传入 gpu_id 参数
                futures.append(executor.submit(
                    processing_function, item, OD_path_dic, station_manager_dict, None, gpu_id
                ))
            for future in tqdm(futures, total=len(futures)):
                results.append(future.result())

        for date_and_time, OD_feature_array, cp_weights, cp_factors in results:
            OD_feature_array_dic[date_and_time] = OD_feature_array
            Date_and_time_OD_path_cp_factors_dic[date_and_time] = (cp_weights, cp_factors)

        with open(f"{OD_feature_array_file_path}", 'wb') as f:
            pickle.dump(OD_feature_array_dic, f)
        with open(f"{Date_and_time_OD_path_cp_factors_dic_file_path}", 'wb') as f:
            pickle.dump(Date_and_time_OD_path_cp_factors_dic, f)
        print(f"Batch {batch_idx + 1}/{total_batches} processed and saved.")
    return OD_feature_array_dic, Date_and_time_OD_path_cp_factors_dic

def processing_Time_DepartFreDic_item(Time_DepartFreDic_item, OD_path_dic, station_manager_dict, device, gpu_id):
    if gpu_id is not None:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
    rank = 5
    station_index = station_manager_dict['station_index']
    index_station = station_manager_dict['index_station']
    num_stations = len(station_index)
    sparse_shape = (num_stations, num_stations, 3, num_stations, num_stations)
    Time_DepartFre_key, Time_DepartFre_value = Time_DepartFreDic_item
    date_and_time = Time_DepartFre_key
    OD_feature_array = np.zeros((num_stations, num_stations, 3, 2))
    all_indices = []
    all_values = []
    if not Time_DepartFre_value:
        shape_1 = (154, int(rank))
        shape_2 = (154, int(rank))
        shape_3 = (3, int(rank))
        shape_4 = (154, int(rank))
        shape_5 = (154, int(rank))
        cp_factors = [
            torch.zeros(*shape_1),
            torch.zeros(*shape_2),
            torch.zeros(*shape_3),
            torch.zeros(*shape_4),
            torch.zeros(*shape_5)
        ]
        cp_weights = torch.ones(rank)

    else:
        for (origin, destination), list_of_paths in OD_path_dic.items():
            origin_idx = station_index[origin]
            destination_idx = station_index[destination]
            path_matrices = [np.zeros((num_stations, num_stations), dtype=np.int8) for _ in range(3)]
            feature_matrices = [np.zeros(2) for _ in range(3)]
            for idx, path_dict in enumerate(list_of_paths[:3]):
                adjacency_matrix = np.zeros((num_stations, num_stations), dtype=np.int8)
                feature_matrix = np.zeros(2)
                feature_matrix[0] = path_dict['number_of_stations']
                feature_matrix[1] = path_dict['number_of_transfers']
                feature_matrices[idx] = feature_matrix
                station_visit_sequence = path_dict['station_visit_sequence']
                for i in range(len(station_visit_sequence) - 1):
                    current_station = station_visit_sequence[i]['index']
                    next_station = station_visit_sequence[i + 1]['index']
                    if (index_station[current_station],
                            index_station[next_station]) not in Time_DepartFre_value:
                        break
                    elif (Time_DepartFre_value[(index_station[current_station], index_station[next_station])][
                              "depart_freq"] == 0):
                        break
                    else:
                        adjacency_matrix[current_station, next_station] = 1

                path_matrices[idx] = adjacency_matrix

            for idx in range(len(list_of_paths), 3):
                path_matrices[idx] = path_matrices[0]
                feature_matrices[idx] = feature_matrices[0]

            OD_feature_array[origin_idx, destination_idx] = np.array(feature_matrices)

            for k in range(3):
                adj_matrix = path_matrices[k]
                indices = np.nonzero(adj_matrix)
                values = adj_matrix[indices]
                indices = np.array(indices)
                indices = torch.tensor(indices, dtype=torch.long)
                values = torch.tensor(values, dtype=torch.int8)
                for i in range(indices.size(1)):
                    all_indices.append(
                        [origin_idx, destination_idx, k, indices[0, i].item(), indices[1, i].item()])
                    all_values.append(values[i].item())

        if not all_indices:
            shape_1 = (154, int(rank))
            shape_2 = (154, int(rank))
            shape_3 = (3, int(rank))
            shape_4 = (154, int(rank))
            shape_5 = (154, int(rank))
            cp_factors = [
                torch.zeros(*shape_1),
                torch.zeros(*shape_2),
                torch.zeros(*shape_3),
                torch.zeros(*shape_4),
                torch.zeros(*shape_5)
            ]
            cp_weights = torch.ones(rank)

        else:
            all_indices = torch.tensor(all_indices).T.to(device)
            all_values = torch.tensor(all_values).to(device)
            all_values.to(torch.int8)
            all_indices.to(torch.int16)
            sparse_tensor = torch.sparse_coo_tensor(all_indices, all_values, sparse_shape).coalesce()
            tl.set_backend('pytorch')
            sparse_tensor = sparse_tensor.to(dtype=torch.float32)
            pid = os.getpid()
            process = psutil.Process(pid)
            mem_info = process.memory_info()
            print(
                f"Process {pid} - CPU: RSS = {mem_info.rss / 1024 ** 2:.2f} MB, VMS = {mem_info.vms / 1024 ** 2:.2f} MB")
            print(
                f"Process {pid} - GPU: Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB, Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            cp_decomp = parafac(sparse_tensor.to_dense(), rank=rank, init='random', n_iter_max=100, tol=1e-4, random_state=42)
            cp_weights, cp_factors = cp_decomp


    return date_and_time, OD_feature_array, cp_weights, cp_factors

def generating_OD_section_pssblty_sparse_array_0209(base_dir, station_manager_dict_name,
                                               Time_DepartFreDic_file_path, OD_path_dic_file_path,
                                               station_manager_dict_file_path, OD_feature_array_file_path,
                                               Date_and_time_OD_path_cp_factors_dic_file_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(OD_feature_array_file_path):
        with open(OD_feature_array_file_path, 'rb') as f:
            OD_feature_array_dic = pickle.load(f)
        print("Loaded existing OD_feature_array_dic.")
    else:
        OD_feature_array_dic = {}

    if os.path.exists(Date_and_time_OD_path_cp_factors_dic_file_path):
        with open(Date_and_time_OD_path_cp_factors_dic_file_path, 'rb') as f:
            Date_and_time_OD_path_cp_factors_dic = pickle.load(f)
        print("Loaded existing Date_and_time_OD_path_cp_factors_dic.")
    else:
        Date_and_time_OD_path_cp_factors_dic = {}

    with open(Time_DepartFreDic_file_path, 'rb') as file:
        Time_DepartFreDic = pickle.load(file)

    with open(OD_path_dic_file_path, 'rb') as f:
        OD_path_dic = pickle.load(f)

    with open(station_manager_dict_file_path, 'rb') as f:
        station_manager_dict = pickle.load(f)

    remaining_Time_DepartFreDic = {k: v for k, v in Time_DepartFreDic.items() if k not in OD_feature_array_dic}
    print(f"Total time nodes: {len(Time_DepartFreDic)}, remaining to process: {len(remaining_Time_DepartFreDic)}")

    if Parallel_or_not:
        with ProcessPoolExecutor(max_workers=60) as executor:
            processing_function_with_args = partial(processing_Time_DepartFreDic_item, OD_path_dic=OD_path_dic,
                                                    station_manager_dict=station_manager_dict, device = device)
            results = list(tqdm(executor.map(processing_function_with_args, Time_DepartFreDic.items()),
                                total=len(Time_DepartFreDic)))

        for date_and_time, OD_feature_array, (cp_weights, cp_factors) in results:
            OD_feature_array_dic[date_and_time] = OD_feature_array
            Date_and_time_OD_path_cp_factors_dic[date_and_time] = (cp_weights, cp_factors)
    if One_by_one_or_not:
        count = 0
        for Time_DepartFreDic_item in tqdm(Time_DepartFreDic.items(), desc="Processing items"):
            count += 1
            Time_DepartFre_key, Time_DepartFre_value = Time_DepartFreDic_item
            date_and_time = Time_DepartFre_key
            date_and_time, OD_feature_array, cp_weights, cp_factors = processing_Time_DepartFreDic_item(Time_DepartFreDic_item, OD_path_dic, station_manager_dict, device)
            OD_feature_array_dic[date_and_time] = OD_feature_array
            Date_and_time_OD_path_cp_factors_dic[date_and_time] = (cp_weights, cp_factors)
            print(f"Processed item {count}: {date_and_time}")

            if count % 100 == 0:
                with open(OD_feature_array_file_path, 'wb') as f:
                    pickle.dump(OD_feature_array_dic, f)
                with open(Date_and_time_OD_path_cp_factors_dic_file_path, 'wb') as f:
                    pickle.dump(Date_and_time_OD_path_cp_factors_dic, f)

    if Batch_or_not:
        batch_processing(
            Time_DepartFreDic = remaining_Time_DepartFreDic,
            batch_size = 10,
            OD_feature_array_file_path = OD_feature_array_file_path,
            Date_and_time_OD_path_cp_factors_dic_file_path = Date_and_time_OD_path_cp_factors_dic_file_path,
            Date_and_time_OD_path_cp_factors_dic = Date_and_time_OD_path_cp_factors_dic,
            processing_function = processing_Time_DepartFreDic_item,
            OD_path_dic = OD_path_dic,
            station_manager_dict = station_manager_dict,
            OD_feature_array_dic=OD_feature_array_dic,
            device=device
        )

    with open(OD_feature_array_file_path, 'wb') as f:
        pickle.dump(OD_feature_array_dic, f)
    with open(Date_and_time_OD_path_cp_factors_dic_file_path, 'wb') as f:
        pickle.dump(Date_and_time_OD_path_cp_factors_dic, f)
