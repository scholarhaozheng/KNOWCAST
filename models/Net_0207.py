from torch import nn
import torch
import torch.nn.functional as F
import random
import math
import sys
import os
import numpy as np
import copy
from metro_data_convertor.Find_project_root import Find_project_root
from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
from dmn_knw_gnrtr.run_PYGT_0917 import RecurrentGCN
from dmn_knw_gnrtr.fit_trip_distribution_model import compute_flow

sys.path.insert(0, os.path.abspath('../..'))

from models.OD_Net_att import ODNet_att


class UtilityLayer(nn.Module):
    def __init__(self, input_dim):
        super(UtilityLayer, self).__init__()
        # Initialize the weights for utility calculation
        self.weights = nn.Parameter(torch.tensor([-0.30, -0.573]))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.weights.device)
        # x is expected to be of shape (origins, destinations, paths, 2)
        utility = torch.matmul(x, self.weights)
        return utility


class LogitLayer(nn.Module):
    def __init__(self):
        super(LogitLayer, self).__init__()

    def forward(self, utility):
        # utility is expected to be of shape (origins, destinations, paths)
        exp_utility = torch.exp(utility)
        probability = exp_utility / exp_utility.sum(dim=-1, keepdim=True)
        return probability


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class ImpedanceLayer(nn.Module):
    def __init__(self, initial_gamma=1.0):
        super(ImpedanceLayer, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(initial_gamma))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, c_ij):
        epsilon = 1e-6
        mask = torch.eye(c_ij.shape[0]).to(self.device)
        c_ij = c_ij + mask * epsilon  # 为对角线项添加 epsilon
        gamma_clamped = torch.clamp(self.gamma, min=0.9, max=1)
        # return torch.exp(-gamma_clamped * torch.log(torch.tensor(c_ij)))
        return torch.pow(c_ij, -gamma_clamped)

# 示例网络结构
class GravityModelNetwork(nn.Module):
    def __init__(self):
        super(GravityModelNetwork, self).__init__()
        self.impedance_layer = ImpedanceLayer(initial_gamma=1.0)

    def forward(self, O_i, D_j, c_ij):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c_ij = torch.from_numpy(c_ij).float().to(self.device)
        O_i = O_i.to(self.device)
        D_j = D_j.to(self.device)
        f_cij = self.impedance_layer(c_ij)
        mask = torch.eye(c_ij.shape[0]).to(self.device)
        f_cij = f_cij * (1 - mask)
        q_ij = O_i * D_j * f_cij / torch.sum(D_j * f_cij, dim=-1, keepdim=True)

        return q_ij


class Net_0207(torch.nn.Module):

    def __init__(self, cfg, logger):
        super(Net_0207, self).__init__()
        self.logger = logger
        self.cfg = cfg

        self.trip_distribution_included = cfg['domain_knowledge_types_included']['trip_distribution']
        self.depart_freq_included = cfg['domain_knowledge_types_included']['depart_freq']
        self.traffic_assignment_included = cfg['domain_knowledge_types_included']['traffic_assignment']

        self.four_step_method = cfg['domain_knowledge_loaded']['four_step_method']
        self.history_distribution = cfg['domain_knowledge_loaded']['history_distribution']

        self.additional_section_feature_dim = cfg['model']['additional_section_feature_dim']
        self.additional_frequency_feature_dim = cfg['model']['additional_frequency_feature_dim']
        self.additional_distribution_dim = cfg['model']['additional_distribution_dim']

        self.num_finished_input_dim = cfg['model']['input_dim']
        self.num_unfinished_input_dim = cfg['model']['input_dim']
        if self.trip_distribution_included:
            self.num_finished_input_dim += self.additional_distribution_dim
            self.num_unfinished_input_dim += self.additional_distribution_dim
        if self.depart_freq_included:
            self.num_finished_input_dim += self.additional_frequency_feature_dim
            self.num_unfinished_input_dim += self.additional_frequency_feature_dim
        if self.traffic_assignment_included:
            self.num_finished_input_dim += self.additional_section_feature_dim
            self.num_unfinished_input_dim += self.additional_section_feature_dim

        self.OD = ODNet_att(cfg, logger)

        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_rnn_layers = cfg['model']['num_rnn_layers']

        self.seq_len = cfg['model']['seq_len']
        self.horizon = cfg['model']['horizon']
        self.head = cfg['model'].get('head', 4)
        self.d_channel = cfg['model'].get('channel', 512)

        self.use_curriculum_learning = self.cfg['model']['use_curriculum_learning']
        self.cl_decay_steps = torch.FloatTensor(data=[self.cfg['model']['cl_decay_steps']])
        self.use_input = cfg['model'].get('use_input', True)

        self.mediate_activation = nn.PReLU(self.num_units)

        self.global_step = 0

        self.batch_size = cfg['data']['batch_size']

        self.utility_layer = UtilityLayer(input_dim=2)
        self.logit_layer = LogitLayer()

        self.autoencoder_od_feature = SimpleAutoencoder(input_dim=self.num_nodes * self.num_nodes * 3,
                                             latent_dim=self.additional_section_feature_dim)

        self.autoencoder_distribution = SimpleAutoencoder(input_dim=self.num_nodes,
                                                        latent_dim=self.additional_distribution_dim)

        self.project_root = Find_project_root()
        station_manager_dict_root = os.path.join(self.project_root, 'data', 'suzhou', 'station_manager_dict_no_11.pkl')
        with open(station_manager_dict_root, 'rb') as f:
            station_manager_dict = pickle.load(f)
        self.station_manager = station_manager_dict

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        try:
            return k / (k + math.exp(step / k))
        except OverflowError:
            return float('inf')

    def encoder_od_do(self, sequences, edge_index, edge_attr=None):
        enc_hiddens_od = [None] * self.num_rnn_layers

        finished_hidden_od = None
        long_his_hidden_od = None
        short_his_hidden_od = None

        for t, batch in enumerate(sequences):
            if self.history_distribution:
                encoder_first_out_od, finished_hidden_od, \
                    long_his_hidden_od, short_his_hidden_od, \
                    enc_first_hidden_od = self.OD.encoder_first_layer(batch,
                                                                      finished_hidden_od,
                                                                      long_his_hidden_od,
                                                                      short_his_hidden_od,
                                                                      edge_index,
                                                                      edge_attr)
            else:
                encoder_first_out_od, finished_hidden_od, \
                    enc_first_hidden_od = self.OD.encoder_first_layer(batch,
                                                                      finished_hidden_od,
                                                                      edge_index,
                                                                      edge_attr)

            enc_hiddens_od[0] = enc_first_hidden_od

            enc_mid_out_od = encoder_first_out_od

            for index in range(self.num_rnn_layers - 1):
                enc_mid_out_od = self.mediate_activation(enc_mid_out_od)
                enc_mid_out_od, enc_mid_hidden_od = self.OD.encoder_second_layer(index,
                                                                                 enc_mid_out_od,
                                                                                 enc_hiddens_od[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)

                enc_hiddens_od[index + 1] = enc_mid_hidden_od

        return enc_hiddens_od

    def scheduled_sampling(self, out, label, GO):
        if self.training and self.use_curriculum_learning:
            c = random.uniform(0, 1)
            T = self.inverse_sigmoid_scheduler_sampling(
                self.global_step,
                self.cl_decay_steps)
            use_truth_sequence = True if c < T else False
        else:
            use_truth_sequence = False

        if use_truth_sequence:
            # Feed the prev label as the next input
            decoder_input = label
        else:
            # detach from history as input
            decoder_input = out.detach().view(-1, self.num_output_dim)
        if not self.use_input:
            decoder_input = GO.detach()

        return decoder_input

    def decoder_od_do(self, sequences, enc_hiddens_od, edge_index, edge_attr=None):
        predictions_od = []

        GO_od = torch.zeros(enc_hiddens_od[0].size()[0],
                            self.num_output_dim,
                            dtype=enc_hiddens_od[0].dtype,
                            device=enc_hiddens_od[0].device)

        dec_input_od = GO_od
        dec_hiddens_od = enc_hiddens_od

        for t in range(self.horizon):
            dec_first_out_od, dec_first_hidden_od = self.OD.decoder_first_layer(dec_input_od,
                                                                                dec_hiddens_od[0],
                                                                                edge_index,
                                                                                edge_attr)

            dec_hiddens_od[0] = dec_first_hidden_od
            dec_mid_out_od = dec_first_out_od

            for index in range(self.num_rnn_layers - 1):
                dec_mid_out_od = self.mediate_activation(dec_mid_out_od)
                dec_mid_out_od, dec_mid_hidden_od = self.OD.decoder_second_layer(index,
                                                                                 dec_mid_out_od,
                                                                                 dec_hiddens_od[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)

                dec_hiddens_od[index + 1] = dec_mid_hidden_od
                dec_mid_out_od = dec_mid_out_od

            dec_mid_out_od = dec_mid_out_od.reshape(-1, self.num_units)

            dec_mid_out_od = self.OD.output_layer(dec_mid_out_od).view(-1, self.num_nodes, self.num_output_dim)

            predictions_od.append(dec_mid_out_od)

            dec_input_od = self.scheduled_sampling(dec_mid_out_od, sequences[t].y_od, GO_od)

        if self.training:
            self.global_step += 1

        return torch.stack(predictions_od).transpose(0, 1)

    def forward(self, sequences, sequences_y):
        extended_sequences = []
        extended_sequences_y = []

        max_shape = None
        for i, data_batch in enumerate(sequences):
            num_graphs = data_batch.ptr.size(0) - 1
            for j in range(num_graphs):
                start_idx = data_batch.ptr[j]
                end_idx = data_batch.ptr[j + 1]
                x_od_sliced = data_batch.x_od[start_idx:end_idx]
                max_shape = x_od_sliced.shape[1]
                if self.depart_freq_included:
                    max_shape = max_shape + self.additional_frequency_feature_dim
                if self.traffic_assignment_included:
                    max_shape = max_shape + self.additional_section_feature_dim
                if self.trip_distribution_included:
                    max_shape = max_shape + self.additional_distribution_dim
                break
            break

        for i, data_batch in enumerate(sequences):
            data_batch_y = sequences_y[i]
            num_graphs = data_batch.ptr.size(0) - 1
            data_batch_new = copy.deepcopy(data_batch)
            data_batch_y_new = copy.deepcopy(data_batch_y)

            if data_batch_new.x_od.shape[1] < max_shape:
                padding_size = max_shape - data_batch_new.x_od.shape[1]
                data_batch_new.x_od = F.pad(data_batch_new.x_od, (0, padding_size), "constant", 0)

            if self.history_distribution:
                if data_batch_new.history.shape[1] < max_shape:
                    padding_size = max_shape - data_batch_new.history.shape[1]
                    data_batch_new.history = F.pad(data_batch_new.history, (0, padding_size), "constant", 0)
                if data_batch_new.yesterday.shape[1] < max_shape:
                    padding_size = max_shape - data_batch_new.yesterday.shape[1]
                    data_batch_new.yesterday = F.pad(data_batch_new.yesterday, (0, padding_size), "constant", 0)

            for j in range(num_graphs):
                start_idx = data_batch.ptr[j]
                end_idx = data_batch.ptr[j + 1]
                edge_index = (data_batch.edge_index[:, (data_batch.edge_index[0] >= start_idx) & (
                        data_batch.edge_index[0] < end_idx)] - start_idx)
                edge_attr = data_batch.edge_attr[
                    (data_batch.edge_index[0] >= start_idx) & (data_batch.edge_index[0] < end_idx)]
                x_od_sliced = data_batch.x_od[start_idx:end_idx]
                if self.history_distribution:
                    unfinished_sliced = data_batch.unfinished[start_idx:end_idx]
                    history_sliced = data_batch.history[start_idx:end_idx]
                    yesterday_sliced = data_batch.yesterday[start_idx:end_idx]
                if self.depart_freq_included:
                    Time_DepartFreDic_Array_sliced = data_batch.Time_DepartFreDic_Array[start_idx:end_idx]
                if self.traffic_assignment_included:
                    cp_factors_sliced = []
                    cp_weight_sliced = []
                    for cp_factors in data_batch.repeated_sparse_5D_tensors[1]:
                        factor_start_idx = int((len(cp_factors) / data_batch.batch_size) * j)
                        factor_end_idx = int((len(cp_factors) / data_batch.batch_size) * (j + 1))
                        cp_factors_sliced.append(cp_factors[factor_start_idx:factor_end_idx])
                    cp_weight_sliced.append(data_batch.repeated_sparse_5D_tensors[0][0:5])

                if self.trip_distribution_included:
                    PINN_od_features_sliced = data_batch.PINN_od_features[start_idx:end_idx]
                    PINN_od_additional_features_sliced = data_batch.PINN_od_additional_features[
                                                         j * num_graphs:(j + 1) * num_graphs]
                    #OD_feature_array_sliced = data_batch.OD_feature_array[start_idx:end_idx]
                    #if isinstance(OD_feature_array_sliced, np.ndarray):
                        #OD_feature_array_sliced = torch.from_numpy(OD_feature_array_sliced).float().to(self.device)

                project_root = Find_project_root()
                base_dir = os.path.join(project_root, f"data{os.path.sep}suzhou")
                with open(os.path.join(base_dir, f"OD_feature_array.pkl"), 'rb') as f:
                    OD_feature_array_sliced = next(iter(pickle.load(f, errors='ignore').values()))

                if self.traffic_assignment_included:
                    # Step 1: Compute utility
                    utility = self.utility_layer(OD_feature_array_sliced)  # shape: (origins, destinations, paths)

                    # Step 2: Compute possibility using Logit
                    OD_path_possibility = self.logit_layer(utility)  # shape: (origins, destinations, paths)

                    # Step 3: Compute OD_section_possibility
                    num_origins, num_destinations, num_paths = OD_path_possibility.size()
                    num_stations = self.num_nodes
                    OD_section_possibility = torch.zeros(num_origins, num_stations, num_stations,
                                                         device=self.device)
                    import time
                    start_time = time.time()
                    simplified = 154215423215421542
                    [A_factors, B_factors, C_factors, D_factors, E_factors] = cp_factors_sliced
                    E_sum = E_factors.sum(dim=0)  # shape: (rank,)
                    combined_factor = cp_weight_sliced[0] * E_sum  # shape: (rank,)

                    if simplified == 1541543154:
                        A_exp = A_factors.view(self.num_nodes, 1, 1, 1, 5)
                        B_exp = B_factors.view(1, self.num_nodes, 1, 1, 5)
                        C_exp = C_factors.view(1, 1, 3, 1, 5)
                        D_exp = D_factors.view(1, 1, 1, self.num_nodes, 5)
                        cf_exp = combined_factor.view(1, 1, 1, 1, 5)
                        T_eff = (A_exp * B_exp * C_exp * D_exp * cf_exp).sum(dim=-1).to(self.device)  # shape: (154, 154, 3, 154)
                    elif simplified == 154215423215421542:
                        T_eff = None
                        for i in range(5):
                            # 将各因子沿 r 维拆分后调整形状，使其能广播为目标形状 (154, 154, 3, 154)
                            a_term = A_factors[:, i].view(154, 1, 1, 1)  # (154,1,1,1)
                            b_term = B_factors[:, i].view(1, 154, 1, 1)  # (1,154,1,1)
                            c_term = C_factors[:, i].view(1, 1, 3, 1)  # (1,1,3,1)
                            d_term = D_factors[:, i].view(1, 1, 1, 154)  # (1,1,1,154)
                            factor = combined_factor[i]  # 标量

                            # 此时各张量广播相乘得到 (154,154,3,154) 形状的结果
                            term = a_term * b_term * c_term * d_term * factor

                            # 逐项累加
                            if T_eff is None:
                                T_eff = term
                            else:
                                T_eff += term

                        T_eff = T_eff.to(self.device)

                    weighted_T_eff = T_eff * OD_path_possibility.unsqueeze(-1)  # keeps shape [O,D,P,S]
                    #OD_section_possibility = torch.mul(cp_factors_sliced, OD_path_possibility_5D)
                    flattened_od_section = weighted_T_eff.view(num_origins, -1)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    # print(f"execution_time: {execution_time} 秒")

                    # Step 4: Process OD_section_possibility through autoencoder_od_feature
                    latent_features, _ = self.autoencoder_od_feature(flattened_od_section)

                distribution_features=[]

                if self.trip_distribution_included:
                    # Step 5: Trip distribution
                    trip_distribution_dic = {}
                    device_cpu = torch.device('cpu')
                    zero_tensor = torch.zeros((self.seq_len, self.num_nodes), device=self.device)
                    nested_list_with_arrays = [zero_tensor[i].cpu().numpy() for i in range(1)]
                    signal_dict = {
                        'features': [PINN_od_features_sliced.cpu().numpy()],
                        'targets': nested_list_with_arrays,
                        'additional_feature': [PINN_od_additional_features_sliced.cpu().numpy()],
                        'edge_index': edge_index,
                        'edge_weight': edge_attr
                    }
                    trip_gnr_signal = StaticGraphTemporalSignal(
                        features=signal_dict["features"],
                        targets=signal_dict["targets"],
                        additional_feature1=signal_dict["additional_feature"],
                        edge_index=signal_dict["edge_index"].clone().cpu(),
                        edge_weight=signal_dict["edge_weight"].clone().cpu()
                    )

                    hyperparams_path = os.path.join(self.project_root, f"data{os.path.sep}suzhou{os.path.sep}",
                                                    f'hyperparameters.pkl')
                    with open(hyperparams_path, 'rb') as f:
                        hyperparameters = pickle.load(f)

                    RGCN_node_features = hyperparameters['RGCN_node_features']
                    RGCN_hidden_units = hyperparameters['RGCN_hidden_units']
                    RGCN_output_dim = hyperparameters['RGCN_output_dim']
                    RGCN_K = hyperparameters['RGCN_K']

                    for str_prdc_attr in ("prdc", "attr"):
                        RecurrentGCN_trip_prdc = RecurrentGCN(node_features=RGCN_node_features,
                                                              hidden_units=RGCN_hidden_units,
                                                              output_dim=RGCN_output_dim,
                                                              K=RGCN_K)
                        RecurrentGCN_model_path = os.path.join(self.project_root, f"data{os.path.sep}suzhou{os.path.sep}",
                                                               f'{str_prdc_attr}_RecurrentGCN_model.pth')
                        RecurrentGCN_trip_prdc.load_state_dict(torch.load(RecurrentGCN_model_path))
                        RecurrentGCN_trip_prdc.eval()
                        RecurrentGCN_trip_prdc.to(self.device)
                        with torch.no_grad():
                            for snap_time, snapshot in enumerate(trip_gnr_signal):
                                snapshot_x = snapshot.x.to(device_cpu)
                                snapshot_edge_index = snapshot.edge_index.to(device_cpu)
                                snapshot_edge_attr = snapshot.edge_attr.to(device_cpu)
                                RecurrentGCN_trip_prdc.to(device_cpu)
                                y_hat = RecurrentGCN_trip_prdc(snapshot_x, snapshot_edge_index, snapshot_edge_attr)
                                trip_distribution_dic[str_prdc_attr] = y_hat

                    def load_from_pkl(filename):
                        with open(filename, 'rb') as file:
                            return pickle.load(file)

                    pkl_filename = os.path.join(self.project_root, f"data{os.path.sep}suzhou", 'trip_generation_trained_params.pkl')
                    loaded_params = load_from_pkl(pkl_filename)
                    q_predicted = compute_flow(trip_distribution_dic["prdc"], trip_distribution_dic["attr"],
                                               self.station_manager['station_distance_matrix'], loaded_params['gamma'], loaded_params['a'], loaded_params['b'])
                    distribution_features, _ = self.autoencoder_distribution(q_predicted.to(self.device))
                x_od_combined = x_od_sliced

                if self.history_distribution:
                    history_combined = history_sliced
                    yesterday_combined = yesterday_sliced
                if self.traffic_assignment_included:
                    x_od_combined = torch.cat((x_od_sliced, latent_features), dim=1)
                    history_combined = torch.cat((history_sliced, latent_features), dim=1)
                    yesterday_combined = torch.cat((yesterday_sliced, latent_features), dim=1)
                if self.depart_freq_included:
                    x_od_combined = torch.cat((x_od_combined, Time_DepartFreDic_Array_sliced), dim=1)
                    history_combined = torch.cat((history_combined, Time_DepartFreDic_Array_sliced), dim=1)
                    yesterday_combined = torch.cat((yesterday_combined, Time_DepartFreDic_Array_sliced), dim=1)
                if self.trip_distribution_included:
                    x_od_combined = torch.cat((x_od_combined, distribution_features), dim=1)
                    history_combined = torch.cat((history_combined, distribution_features), dim=1)
                    yesterday_combined = torch.cat((yesterday_combined, distribution_features), dim=1)

                current_shape = data_batch_new.x_od[start_idx:end_idx].shape[1]
                target_shape = x_od_combined.shape[1]
                if current_shape < target_shape:
                    padding_size = target_shape - current_shape
                    x_od_combined = F.pad(x_od_combined, (0, padding_size), "constant", 0)
                    if self.history_distribution:
                        history_combined = F.pad(history_combined, (0, padding_size), "constant", 0) if \
                            history_combined.shape[1] < target_shape else history_combined[:, :target_shape]
                        yesterday_combined = F.pad(yesterday_combined, (0, padding_size), "constant", 0) if \
                            yesterday_combined.shape[1] < target_shape else yesterday_combined[:, :target_shape]
                data_batch_new.x_od[start_idx:end_idx] = x_od_combined
                if self.history_distribution:
                    data_batch_new.unfinished[start_idx:end_idx]
                    data_batch_new.history[start_idx:end_idx] = history_combined
                    data_batch_new.yesterday[start_idx:end_idx] = yesterday_combined
            extended_sequences.append(data_batch_new)
            extended_sequences_y.append(data_batch_y_new)
        edge_index = sequences[0].edge_index.detach()
        edge_attr = sequences[0].edge_attr.detach()
        enc_hiddens_od = self.encoder_od_do(extended_sequences,
                                            edge_index=edge_index,
                                            edge_attr=edge_attr)
        predictions_od = self.decoder_od_do(extended_sequences_y,
                                            enc_hiddens_od,
                                            edge_index=edge_index,
                                            edge_attr=edge_attr)
        return predictions_od
