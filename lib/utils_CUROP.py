# part of this code are copied from DCRNN
import logging
import os
import pickle
import sys
import numpy as np
import torch
from torch_geometric.data import Batch, Data
# from sklearn.externals import joblib
import tempfile
import gc
import shutil
import argparse
import yaml
import concurrent.futures
from config import args

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def read_cfg_file(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg

cfg = read_cfg_file(args.config_filename)

four_step_method_included = cfg['domain_knowledge_loaded']['four_step_method']
history_distribution_included = cfg['domain_knowledge_loaded']['history_distribution']
trip_generation_included = cfg['domain_knowledge_types_included']['trip_generation']
trip_distribution_included = cfg['domain_knowledge_types_included']['trip_distribution']
depart_freq_included = cfg['domain_knowledge_types_included']['depart_freq']
traffic_assignment_included = cfg['domain_knowledge_types_included']['traffic_assignment']

class DataLoader(object):
    def __init__(self,
                 data_dict=None,
                 x_od=None,
                 y_od=None,
                 xtime=None,
                 ytime=None,
                 batch_size=None,
                 unfinished=None,
                 history=None,
                 yesterday=None,
                 PINN_od_features=None,
                 PINN_od_additional_features=None,
                 Time_DepartFreDic_Array=None,
                 repeated_sparse_5D_tensors=None,
                 pad_with_last_sample=True,
                 shuffle=False,
                 num_workers=0):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """

        if data_dict is not None:
            x_od = data_dict.get('x_od')
            y_od = data_dict.get('y_od')
            xtime = data_dict.get('xtime')
            ytime = data_dict.get('ytime')
            unfinished = data_dict.get('unfinished')
            history = data_dict.get('history')
            yesterday = data_dict.get('yesterday')
            PINN_od_features = data_dict.get('PINN_od_features')
            PINN_od_additional_features = data_dict.get('PINN_od_additional_features')
            Time_DepartFreDic_Array = data_dict.get('Time_DepartFreDic_Array')
            repeated_sparse_5D_tensors = data_dict.get('repeated_sparse_5D_tensors')

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x_od) % batch_size)) % batch_size
            x_od_padding = np.repeat(x_od[-1:], num_padding, axis=0)
            xtime_padding = np.repeat(xtime[-1:], num_padding, axis=0)
            y_od_padding = np.repeat(y_od[-1:], num_padding, axis=0)
            ytime_padding = np.repeat(ytime[-1:], num_padding, axis=0)

            if trip_generation_included:
                PINN_od_features_padding = np.repeat(PINN_od_features[-1:], num_padding, axis=0)
                PINN_od_additional_features_padding = np.repeat(PINN_od_additional_features[-1:], num_padding, axis=0)
                PINN_od_features = np.concatenate([PINN_od_features, PINN_od_features_padding], axis=0)
                PINN_od_additional_features = np.concatenate(
                    [PINN_od_additional_features, PINN_od_additional_features_padding], axis=0)

            if depart_freq_included:
                Time_DepartFreDic_Array_padding = np.repeat(Time_DepartFreDic_Array[-1:], num_padding, axis=0)
                Time_DepartFreDic_Array = np.concatenate([Time_DepartFreDic_Array, Time_DepartFreDic_Array_padding], axis=0)

            if traffic_assignment_included:
                repeated_sparse_5D_padding = [repeated_sparse_5D_tensors[-1]] * num_padding
                repeated_sparse_5D_tensors = repeated_sparse_5D_tensors + repeated_sparse_5D_padding

            x_od = np.concatenate([x_od, x_od_padding], axis=0)
            xtime = np.concatenate([xtime, xtime_padding], axis=0)
            y_od = np.concatenate([y_od, y_od_padding], axis=0)
            ytime = np.concatenate([ytime, ytime_padding], axis=0)

            if history_distribution_included and history_distribution_included is not None:
                unfi_num_padding = (batch_size - (len(unfinished) % batch_size)) % batch_size
                unfinished_padding = np.repeat(unfinished[-1:], unfi_num_padding, axis=0)
                unfinished = np.concatenate([unfinished, unfinished_padding], axis=0)
                history_num_padding = (batch_size - (len(history) % batch_size)) % batch_size
                history_padding = np.repeat(history[-1:], history_num_padding, axis=0)
                history = np.concatenate([history, history_padding], axis=0)
                yesterday_num_padding = (batch_size - (len(yesterday) % batch_size)) % batch_size
                yesterday_padding = np.repeat(yesterday[-1:], yesterday_num_padding, axis=0)
                yesterday = np.concatenate([yesterday, yesterday_padding], axis=0)

        self.size = len(x_od)
        self.num_batch = int(self.size // self.batch_size)
        self.x_od = x_od
        self.y_od = y_od
        self.xtime = xtime
        self.ytime = ytime

        if history_distribution_included:
            self.unfinished = unfinished
            self.history = history
            self.yesterday = yesterday
        if trip_generation_included:
            self.PINN_od_features = PINN_od_features
            self.PINN_od_additional_features = PINN_od_additional_features
        if depart_freq_included:
            self.Time_DepartFreDic_Array = Time_DepartFreDic_Array
        if traffic_assignment_included:
            self.repeated_sparse_5D_tensors = repeated_sparse_5D_tensors

        self.do_shuffle = shuffle
        if shuffle:
            self.shuffle()#??????

    def __len__(self):
        return self.num_batch

    def shuffle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chunk_size = 80
            permutation = np.random.permutation(self.size)

            # 1. 生成 memmap 文件
            x_mem = self._apply_permutation_in_chunks_memmap(
                self.x_od, permutation, chunk_size, 'x_od.dat', temp_dir
            )
            y_mem = self._apply_permutation_in_chunks_memmap(
                self.y_od, permutation, chunk_size, 'y_od.dat', temp_dir
            )
            xtime_mem = self._apply_permutation_in_chunks_memmap(
                self.xtime, permutation, chunk_size, 'xtime.dat', temp_dir
            )
            ytime_mem = self._apply_permutation_in_chunks_memmap(
                self.ytime, permutation, chunk_size, 'ytime.dat', temp_dir
            )

            if history_distribution_included:
                unfinished_mem = self._apply_permutation_in_chunks_memmap(
                    self.unfinished, permutation, chunk_size, 'unfinished.dat', temp_dir
                )
                history_mem = self._apply_permutation_in_chunks_memmap(
                    self.history, permutation, chunk_size, 'history.dat', temp_dir
                )
                yesterday_mem = self._apply_permutation_in_chunks_memmap(
                    self.yesterday, permutation, chunk_size, 'yesterday.dat', temp_dir
                )
            if trip_generation_included:
                pinn_feat_mem = self._apply_permutation_in_chunks_memmap(
                    self.PINN_od_features, permutation, chunk_size, 'PINN_od_features.dat', temp_dir
                )
                pinn_add_feat_mem = self._apply_permutation_in_chunks_memmap(
                    self.PINN_od_additional_features, permutation, chunk_size, 'PINN_od_additional_features.dat', temp_dir
                )
            if depart_freq_included:
                time_dep_mem = self._apply_permutation_in_chunks_memmap(
                    self.Time_DepartFreDic_Array, permutation, chunk_size, 'Time_DepartFreDic_Array.dat', temp_dir
                )
            if traffic_assignment_included:
                rep_sparse_list = [self.repeated_sparse_5D_tensors[i] for i in permutation]

            self.x_od = np.array(x_mem)
            self.y_od = np.array(y_mem)
            self.xtime = np.array(xtime_mem)
            self.ytime = np.array(ytime_mem)

            if history_distribution_included:
                self.unfinished = np.array(unfinished_mem)
                self.history = np.array(history_mem)
                self.yesterday = np.array(yesterday_mem)
            if trip_generation_included:
                self.PINN_od_features = np.array(pinn_feat_mem)
                self.PINN_od_additional_features = np.array(pinn_add_feat_mem)
            if depart_freq_included:
                self.Time_DepartFreDic_Array = np.array(time_dep_mem)
            if traffic_assignment_included:
                self.repeated_sparse_5D_tensors = rep_sparse_list

        print(f'Shuffle Completed')

    def _apply_permutation_in_chunks_memmap(self, array, permutation, chunk_size, filename, temp_dir):
        shape = array.shape
        dtype = array.dtype
        filename = os.path.join(temp_dir, filename)
        with open(filename, 'wb') as f:
            f.write(b'\x00' * np.product(shape) * np.dtype(dtype).itemsize)
        mmapped_result = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)
        for start_idx in range(0, len(permutation), chunk_size):
            end_idx = min(start_idx + chunk_size, len(permutation))
            mmapped_result[start_idx:end_idx] = array[permutation[start_idx:end_idx]]
        mmapped_result.flush()
        with open(filename, 'r+') as f:
            os.fsync(f.fileno())
        return mmapped_result

    def _get_batch(self, idx):
        start_ind = self.batch_size * idx
        end_ind = min(self.size, self.batch_size * (idx + 1))

        batch_dict = {
            "x_od": self.x_od[start_ind:end_ind, ...],
            "y_od": self.y_od[start_ind:end_ind, ...],
            "xtime": self.xtime[start_ind:end_ind, ...],
            "ytime": self.ytime[start_ind:end_ind, ...],
        }

        if history_distribution_included:
            batch_dict["unfinished"] = self.unfinished[start_ind:end_ind, ...]
            batch_dict["history"] = self.history[start_ind:end_ind, ...]
            batch_dict["yesterday"] = self.yesterday[start_ind:end_ind, ...]

        if trip_generation_included:
            batch_dict["PINN_od_features"] = self.PINN_od_features[start_ind:end_ind, ...]
            batch_dict["PINN_od_additional_features"] = self.PINN_od_additional_features[start_ind:end_ind, ...]

        if depart_freq_included:
            batch_dict["Time_DepartFreDic_Array"] = self.Time_DepartFreDic_Array[start_ind:end_ind, ...]

        if traffic_assignment_included:
            batch_dict["repeated_sparse_5D_tensors"] = self.repeated_sparse_5D_tensors[start_ind:end_ind]

        return batch_dict

    def __iter__(self):
        if self.num_workers > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for batch in executor.map(self._get_batch, range(self.num_batch)):
                    yield batch
        else:
            for idx in range(self.num_batch):
                yield self._get_batch(idx)

    def get_iterator(self):
        return iter(self)

class StandardScaler_Torch:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info('Log directory: %s', log_dir)
    return logger


def load_dataset(dataset_dir,
                    batch_size,
                    test_batch_size=None,
                    scaler_axis=(0,
                                 1,
                                 2,
                                 3),
                    **kwargs):
    data = {}

    for category in ['train', 'val', 'test']:
        cat_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))

        data['x_' + category] = cat_data['finished']
        data['xtime_' + category] = cat_data['xtime']
        data['y_' + category] = cat_data['y']
        data['ytime_' + category] = cat_data['ytime']

        if history_distribution_included:
            data['unfinished_' + category] = cat_data['unfinished']

            history_data = load_pickle(os.path.join(dataset_dir, category + '_history_long.pkl'))
            his_sum = np.sum(history_data['history'], axis=-1)
            history_distribution = np.nan_to_num(np.divide(history_data['history'], np.expand_dims(his_sum, axis=-1)))

            yesterday_data = load_pickle(os.path.join(dataset_dir, category + '_history_short.pkl'))
            yesterday_sum = np.sum(yesterday_data['history'], axis=-1)
            yesterday_distribution = np.nan_to_num(np.divide(yesterday_data['history'], np.expand_dims(yesterday_sum, axis=-1)))

            data['history_' + category] = np.multiply(history_distribution, cat_data['unfinished'])
            data['yesterday_' + category] = np.multiply(yesterday_distribution, cat_data['unfinished'])

        if trip_generation_included:
            PINN_data_od = load_pickle(os.path.join(dataset_dir, category + '_signal_dict_array.pkl'))
        if depart_freq_included:
            Time_DepartFreDic_Array = load_pickle(os.path.join(dataset_dir, category + '_Time_DepartFreDic_Matrix.pkl'))
        if traffic_assignment_included:
            repeated_sparse_5D_tensors = load_pickle(os.path.join(dataset_dir, category + '_Date_and_time_OD_path_Matrix.pkl'))
        if trip_generation_included:
            data['PINN_od_features_' + category] = np.array(PINN_data_od["features"])
            data['PINN_od_additional_features_' + category] = PINN_data_od["additional_feature"]
        if depart_freq_included:
            data['Time_DepartFreDic_Array_' + category] = Time_DepartFreDic_Array
        if traffic_assignment_included:
            data['repeated_sparse_5D_tensors_' + category] = repeated_sparse_5D_tensors

    scaler = StandardScaler(mean=data['x_train'].mean(axis=scaler_axis),
                            std=data['x_train'].std(axis=scaler_axis))

    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])

        if history_distribution_included:
            data['unfinished_' + category] = scaler.transform(data['unfinished_' + category])
            data['history_' + category] = scaler.transform(data['history_' + category])
            data['yesterday_' + category] = scaler.transform(data['yesterday_' + category])

    if depart_freq_included:
        depart_train = np.array(data['Time_DepartFreDic_Array_train'])
        mean_dep = depart_train.mean()
        std_dep = depart_train.std()
        depart_scaler = StandardScaler(mean=mean_dep, std=std_dep)
        for category in ['train', 'val', 'test']:
            arr = data[f'Time_DepartFreDic_Array_{category}']
            data[f'Time_DepartFreDic_Array_{category}'] = depart_scaler.transform(arr)
        data['depart_freq_scaler'] = depart_scaler

    def create_dataloader(phase, data, batch_size, shuffle, num_workers=0):
        return DataLoader(
            data_dict={
                'x_od': data[f'x_{phase}'],
                'y_od': data[f'y_{phase}'],
                'xtime': data[f'xtime_{phase}'],
                'ytime': data[f'ytime_{phase}'],
                'unfinished': data.get(f'unfinished_{phase}'),
                'history': data.get(f'history_{phase}'),
                'yesterday': data.get(f'yesterday_{phase}'),
                'PINN_od_features': data.get(f'PINN_od_features_{phase}'),
                'PINN_od_additional_features': data.get(f'PINN_od_additional_features_{phase}'),
                'Time_DepartFreDic_Array': data.get(f'Time_DepartFreDic_Array_{phase}'),
                'repeated_sparse_5D_tensors': data.get(f'repeated_sparse_5D_tensors_{phase}')
            },
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    data['train_loader'] = create_dataloader('train', data, batch_size=batch_size, shuffle=True, num_workers=48)
    data['val_loader'] = create_dataloader('val', data, batch_size=test_batch_size, shuffle=False, num_workers=48)
    data['test_loader'] = create_dataloader('test', data, batch_size=test_batch_size, shuffle=False, num_workers=48)
    data['scaler'] = scaler

    return data

def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx.astype(np.float32)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def move_to_device(item, device):
    if torch.is_tensor(item):
        return item.to(device)
    elif isinstance(item, (list, tuple)):
        return type(item)(move_to_device(x, device) for x in item)
    else:
        return item

class SimpleBatch(list):
    def to(self, device):
        for ele in self:
            ele.to(device)
        return self

def collate_wrapper(x_od, y_od,
                    num_nodes, edge_index, edge_attr, seq_len, horizon, device,
                    unfinished=None,
                    history=None,
                    yesterday=None,
                    PINN_od_features=None,
                    PINN_od_additional_features=None,
                    # OD_feature_array=None,
                    Time_DepartFreDic_Array=None,
                    repeated_sparse_5D_tensors=None,
                    return_y=True):
    x_od = torch.tensor(x_od, dtype=torch.float, device=device)
    y_od = torch.tensor(y_od, dtype=torch.float, device=device)
    x_od = x_od.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_od_T_first = y_od.transpose(dim0=1, dim1=0)

    if history_distribution_included:
        unfinished = torch.tensor(unfinished, dtype=torch.float, device=device)
        unfinished = unfinished.transpose(dim0=1, dim1=0)
        history = torch.tensor(history, dtype=torch.float, device=device)
        history = history.transpose(dim0=1, dim1=0)
        yesterday = torch.tensor(yesterday, dtype=torch.float, device=device)
        yesterday = yesterday.transpose(dim0=1, dim1=0)

    if trip_generation_included:
        PINN_od_features = torch.tensor(PINN_od_features, dtype=torch.float, device=device)
        PINN_od_additional_features = torch.tensor(PINN_od_additional_features, dtype=torch.float, device=device)
        PINN_od_features_T_first = PINN_od_features.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
        PINN_od_additional_features_T_first = PINN_od_additional_features.transpose(dim0=1, dim1=0)
    if traffic_assignment_included:
        Time_DepartFreDic_Array = torch.tensor(Time_DepartFreDic_Array, dtype=torch.float, device=device)
        Time_DepartFreDic_Array_T_first = Time_DepartFreDic_Array.transpose(dim0=1, dim1=0)

    if traffic_assignment_included:
        transposed_repeated_sparse_5D_tensors = list(map(list, zip(*repeated_sparse_5D_tensors)))


    edge_index = edge_index.clone().detach().to(device)
    edge_attr = edge_attr.clone().detach().to(device)

    T = x_od.size()[0]
    H = y_od_T_first.size()[0]
    N = x_od.size()[1]
    sequences = []
    sequences_y = []
    for t in range(T - seq_len, T):
        cur_batch_x_od = x_od[t]

        if history_distribution_included:
            cur_batch_unfinished = unfinished[t]
            cur_batch_history = history[t]
            cur_batch_yesterday = yesterday[t]
        if trip_generation_included:
            cur_batch_PINN_od_features = PINN_od_features_T_first[t]
            cur_batch_PINN_od_additional_features = PINN_od_additional_features_T_first[t]
        if traffic_assignment_included:
            cur_batch_Time_DepartFreDic_Array = Time_DepartFreDic_Array_T_first[t]
        if traffic_assignment_included:
            cur_repeated_sparse_5D_tensors = transposed_repeated_sparse_5D_tensors[t]

        batch = Batch.from_data_list([
            Data(
                x_od=cur_batch_x_od[i].to(device),
                unfinished=cur_batch_unfinished[i].to(device) if history_distribution_included else None,
                history=cur_batch_history[i].to(device) if history_distribution_included else None,
                yesterday=cur_batch_yesterday[i].to(device) if history_distribution_included else None,
                PINN_od_features=cur_batch_PINN_od_features[i].to(device) if trip_generation_included else None,
                PINN_od_additional_features=cur_batch_PINN_od_additional_features[i].to(
                    device) if trip_generation_included else None,
                # OD_feature_array=cur_batch_OD_feature_array[i].to(device) if four_step_method_included else None,
                Time_DepartFreDic_Array=cur_batch_Time_DepartFreDic_Array[i].to(
                    device) if depart_freq_included else None,
                repeated_sparse_5D_tensors=(
                    tuple(move_to_device(item, device) for item in cur_repeated_sparse_5D_tensors[i])
                ) if traffic_assignment_included else None,
                num_nodes=num_nodes,
                edge_index=edge_index.to(device),
                edge_attr=edge_attr.to(device)
            ) for i in range(N)
        ])

        sequences.append(batch)

    for t in range(H - horizon, H):
        cur_batch_y_od = y_od_T_first[t]

        batch_y = Batch.from_data_list([
            Data(y_od=cur_batch_y_od[i]) for i in range(N)
        ])
        sequences_y.append(batch_y)
    if return_y:
        return SimpleBatch(sequences), SimpleBatch(sequences_y), y_od
    else:
        return SimpleBatch(sequences), SimpleBatch(sequences_y)
