# filename: train_save_history.py
# encoding:utf-8
import random
import time
from functools import partial
import cProfile
import pstats
import yaml
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from visualization_tools.plot_graph import plot_step_points_from_excel, plot_metric_pair, plot_training_curves, \
    plot_od_losses, plot_od_time_series, plot_od_time_series_first_step

mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0
from torch_geometric_temporal import StaticGraphTemporalSignal
from config import args
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.nn.init import xavier_uniform_
from lib import utils_CUROP as utils
from lib.utils_CUROP import collate_wrapper
from metro_data_convertor.Find_project_root import Find_project_root
from models.Net_0207 import Net_0207
from dmn_knw_gnrtr.run_PYGT_0917 import RecurrentGCN
import pickle
from train_evaluate_functions import evaluate
from train_evaluate_functions import run_model
from train_evaluate_functions import _get_log_dir
from train_evaluate_functions import StepLR2
from collections import OrderedDict
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
MODE = "train"  #  inference/test: "test"
train_eval_period = 10

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
"""parser = argparse.ArgumentParser()
parser.add_argument('--config_filename',
                    default=None,
                    type=str,
                    help='Configuration filename for restoring the model.')
"""
import pandas as pd
def read_cfg_file(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg

def init_weights(m, Using_GAT_or_RGCN):
    classname = m.__class__.__name__  # 2
    if classname.find('Conv') != -1 and classname.find(Using_GAT_or_RGCN) == -1:
        xavier_uniform_(m.weight.data)
    if type(m) == nn.Linear:
        xavier_uniform_(m.weight.data)
        # xavier_uniform_(m.bias.data)

def compute_epoch_od_errors(y_preds, y_truth):
    abs_err = np.abs(y_preds - y_truth)
    mean_err = abs_err.mean(axis=(0,1))
    return mean_err

def main(args):
    cfg = read_cfg_file(args.config_filename)
    log_dir = _get_log_dir(cfg)

    if MODE == "train":
        all_epochs_dir = os.path.join(log_dir, "all_epochs")
        os.makedirs(all_epochs_dir, exist_ok=True)
        best_epoch = None

    import shutil
    src_cfg = args.config_filename
    dst_cfg = os.path.join(log_dir, os.path.basename(src_cfg))
    shutil.copy(src_cfg, dst_cfg)

    trial_number = cfg['train'].get('trial_number', None)
    if trial_number is not None:
        best_val_mae   = float('inf')
        best_model_path = os.path.join(
            log_dir,
            f"trial_{trial_number}_best.pt"
        )
    else:
        best_val_mae   = float('inf')
        best_model_path = os.path.join(log_dir, "best_full_training_model.pt")

    log_level = cfg.get('log_level', 'INFO')
    logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(cfg)
    batch_size = cfg['data']['batch_size']
    seq_len = cfg['model']['seq_len']
    horizon = cfg['model']['horizon']
    num_nodes = cfg['model']['num_nodes']
    input_dim_m = cfg['model']['input_dim']
    four_step_method_included = cfg['domain_knowledge_loaded']['four_step_method']
    history_distribution_included = cfg['domain_knowledge_loaded']['history_distribution']
    Using_GAT_or_RGCN = cfg['domain_knowledge']['Using_GAT_or_RGCN']

    project_root = Find_project_root()
    hyperparams_path = os.path.join(project_root, f"data{os.path.sep}suzhou{os.path.sep}",
                                    f'hyperparameters.pkl')
    with open(hyperparams_path, 'rb') as f:
        hyperparameters = pickle.load(f)

    RGCN_node_features = hyperparameters['RGCN_node_features']
    RGCN_hidden_units = hyperparameters['RGCN_hidden_units']
    RGCN_output_dim = hyperparameters['RGCN_output_dim']
    RGCN_K = hyperparameters['RGCN_K']

    adj_mx_list = []
    graph_pkl_filename = cfg['data']['graph_pkl_filename']

    trip_generation_included = cfg['domain_knowledge_types_included']['trip_generation']
    trip_distribution_included = cfg['domain_knowledge_types_included']['trip_distribution']
    depart_freq_included = cfg['domain_knowledge_types_included']['depart_freq']
    traffic_assignment_included = cfg['domain_knowledge_types_included']['traffic_assignment']

    if not isinstance(graph_pkl_filename, list):
        graph_pkl_filename = [graph_pkl_filename]

    src = []
    dst = []
    for g in graph_pkl_filename:
        adj_mx = utils.load_graph_data(g)
        for i in range(len(adj_mx)):
            adj_mx[i, i] = 0
        adj_mx_list.append(adj_mx)

    adj_mx = np.stack(adj_mx_list, axis=-1)
    print("adj_mx:", adj_mx.shape)
    if cfg['model'].get('norm', False):
        print('row normalization')
        adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)
    src, dst = adj_mx.sum(axis=-1).nonzero()
    print("src, dst:", src.shape, dst.shape)
    edge_index_np = np.array([src, dst])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_attr = torch.tensor(adj_mx[adj_mx.sum(axis=-1) != 0],
                             dtype=torch.float,
                             device=device)
    print("train, edge:", edge_index.shape, edge_attr.shape)
    output_dim = cfg['model']['output_dim']
    for i in range(adj_mx.shape[-1]):
        logger.info(adj_mx[..., i])

    dataset = utils.load_dataset(**cfg['data'], scaler_axis=(0, 1, 2, 3))
    for k, v in dataset.items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    scaler_od = dataset['scaler']
    scaler_od_torch = utils.StandardScaler_Torch(scaler_od.mean,
                                                 scaler_od.std,
                                                 device=device)
    logger.info('scaler_od.mean:{}, scaler_od.std:{}'.format(scaler_od.mean,
                                                             scaler_od.std))

    model = Net_0207(cfg, logger).to(device)
    model.apply(partial(init_weights, Using_GAT_or_RGCN))

    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['train']['base_lr'],
                           eps=cfg['train']['epsilon'])
    scheduler = StepLR2(optimizer=optimizer,
                        milestones=cfg['train']['steps'],
                        gamma=cfg['train']['lr_decay_ratio'],
                        min_lr=cfg['train']['min_learning_rate'])

    max_grad_norm = cfg['train']['max_grad_norm']
    train_patience = cfg['train']['patience']

    update = {}
    for category in ['od']:
        update['val_steady_count_' + category] = 0
        update['last_val_mae_' + category] = 1e60
        update['last_val_mape_net_' + category] = 1e6

    horizon = cfg['model']['horizon']

    with open(graph_pkl_filename[0], 'rb') as f:
        graph_sz_conn_no_11 = pickle.load(f, errors='ignore')

    row, col = np.nonzero(graph_sz_conn_no_11)
    edge_index = np.array([row, col])
    edge_weight = graph_sz_conn_no_11[row, col]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    epochs_idx = []
    train_losses  = []
    val_loss_list = []
    val_mae_list   = []
    val_mape_list  = []
    val_rmse_list  = []
    train_mae_list = []
    train_mape_list = []
    train_rmse_list = []
    test_loss_list = []
    test_mae_list = []
    test_mape_list = []
    test_rmse_list = []
    epoch_od_errors_list = []
    od_losses = []
    weight_history = []
    for epoch in range(cfg['train']['epochs']):
        total_loss = 0
        epoch_od_loss = 0
        i = 0
        begin_time = time.perf_counter()
        dataset['train_loader'].shuffle()
        train_iterator = dataset['train_loader'].get_iterator()
        model.train()
        print("Total Batch：", len(dataset['train_loader']))
        for idx, batch in enumerate(dataset['train_loader']):
            print(f"{idx + 1} / {len(dataset['train_loader'])} Batch")
            x_od = batch['x_od']
            y_od = batch['y_od']
            xtime = batch['xtime']
            ytime = batch['ytime']
            unfinished = batch.get('unfinished')
            history = batch.get('history')
            yesterday = batch.get('yesterday')
            PINN_od_features = batch.get('PINN_od_features')
            PINN_od_additional_features = batch.get('PINN_od_additional_features')
            Time_DepartFreDic_Array = batch.get('Time_DepartFreDic_Array')
            repeated_sparse_5D_tensors = batch.get('repeated_sparse_5D_tensors')

            optimizer.zero_grad()
            y_od = y_od[:, :horizon, :, :output_dim]
            sequences, sequences_y, y_od = collate_wrapper(
                x_od=x_od, y_od=y_od,
                unfinished=unfinished if history_distribution_included else None,
                history=history if history_distribution_included else None,
                yesterday=yesterday if history_distribution_included else None,
                PINN_od_features=PINN_od_features if four_step_method_included else None,
                PINN_od_additional_features=PINN_od_additional_features if four_step_method_included else None,
                #OD_feature_array=OD_feature_array if four_step_method_included else None,
                Time_DepartFreDic_Array=Time_DepartFreDic_Array if depart_freq_included else None,
                repeated_sparse_5D_tensors=repeated_sparse_5D_tensors if traffic_assignment_included else None,
                num_nodes=num_nodes,
                edge_index=edge_index, edge_attr=edge_attr, device=device,
                seq_len=seq_len, horizon=horizon
            )

            y_od_pred = model(sequences, sequences_y)

            y_od_pred = scaler_od_torch.inverse_transform(y_od_pred)  # *std+mean
            y_od = scaler_od_torch.inverse_transform(y_od)
            loss_od = criterion(y_od_pred, y_od)

            if trip_generation_included:
                cost_PINN = 0
                for i_sub_features in range(batch_size):
                    sub_PINN_od_features = PINN_od_features[i_sub_features]
                    sub_od_additional_features = PINN_od_additional_features[i_sub_features]
                    # PINN Loss
                    zero_tensor = torch.zeros((seq_len, num_nodes), device=device)
                    nested_list_with_arrays = [zero_tensor[i].cpu().numpy() for i in range(seq_len)]
                    sub_PINN_od_features_list_of_arrays = [sub_PINN_od_features[i] for i in range(seq_len)]
                    sub_od_additional_features_list_of_arrays = [sub_od_additional_features[i] for i in range(seq_len)]
                    signal_dict = {
                        'features': sub_PINN_od_features_list_of_arrays,
                        'targets': nested_list_with_arrays,
                        'additional_feature': sub_od_additional_features_list_of_arrays,
                        'edge_index': edge_index,
                        'edge_weight': edge_weight
                    }
                    trip_gnr_signal = StaticGraphTemporalSignal(
                        features=signal_dict["features"],
                        targets=signal_dict["targets"],
                        additional_feature1=signal_dict["additional_feature"],
                        edge_index=signal_dict["edge_index"],
                        edge_weight=signal_dict["edge_weight"]
                    )

                    for str_prdc_attr in ("prdc", "attr"):
                        RecurrentGCN_trip_prdc = RecurrentGCN(node_features=RGCN_node_features,
                                                              hidden_units=RGCN_hidden_units,
                                                              output_dim=RGCN_output_dim,
                                                              K=RGCN_K)

                        RecurrentGCN_model_path = os.path.join(project_root, f"data{os.path.sep}suzhou{os.path.sep}",
                                                               f'{str_prdc_attr}_RecurrentGCN_model.pth')
                        RecurrentGCN_trip_prdc.load_state_dict(torch.load(RecurrentGCN_model_path))
                        RecurrentGCN_trip_prdc.eval()

                        if (str_prdc_attr == 'prdc'):
                            y_od_pred_sum = y_od_pred.sum(dim=-1)
                        else:
                            y_od_pred_sum = y_od_pred.sum(dim=-2)

                        with torch.no_grad():
                            for snap_time, snapshot in enumerate(trip_gnr_signal):
                                y_hat = RecurrentGCN_trip_prdc(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                                y_hat = y_hat.clone().detach().requires_grad_(True).to(device)
                                y_od_pred_sum_ = y_od_pred_sum[i_sub_features][snap_time]
                                cost_PINN = cost_PINN + abs(torch.mean((y_hat - y_od_pred_sum_)))
                pinn_decay_epochs = 10
                if epoch < pinn_decay_epochs:
                    pinn_coef = 1.0 - (epoch / pinn_decay_epochs)
                else:
                    pinn_coef = 0.0
                pinn_weight = cfg['model']['PINN_value'] * pinn_coef
                loss = loss_od + pinn_weight * cost_PINN
            else:
                loss = loss_od

            total_loss += loss.item()
            epoch_od_loss += loss_od.item()
            loss.backward()

            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            w = model.utility_layer.weights.detach().cpu().numpy()
            weight_history.append([float(w[0]), float(w[1])])
            print(f"Epoch {epoch + 1} utility weights = {weight_history[-1]}")

            i += 1

        logger.info(('Epoch:{}, total_loss:{}').format(epoch, total_loss / i))
        avg_train_loss = total_loss / i
        train_losses.append(avg_train_loss)
        avg_od_loss = epoch_od_loss / i
        od_losses.append(avg_od_loss)

        if (epoch + 1) % train_eval_period == 0:
            train_result = evaluate(
                model=model,
                dataset=dataset,
                dataset_type='train',
                num_nodes=num_nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
                device=device,
                seq_Len=seq_len,
                horizon=horizon,
                output_dim=output_dim,
                four_step_method_included=four_step_method_included,
                history_distribution_included=history_distribution_included,
                traffic_assignment_included=traffic_assignment_included,
                depart_freq_included=depart_freq_included,
                logger=logger,
                detail=True,
                cfg=cfg
            )
            train_mae_list.append(train_result['MAE_od'])
            train_mape_list.append(train_result['MAPE_net_od'])
            train_rmse_list.append(train_result['RMSE_od'])
        else:
            train_mae_list.append(None)
            train_mape_list.append(None)
            train_rmse_list.append(None)

        val_result = evaluate(model=model,
                              dataset=dataset,
                              dataset_type='val',
                              num_nodes=num_nodes,
                              edge_index=edge_index,
                              edge_attr=edge_attr,
                              device=device,
                              seq_Len=seq_len,
                              horizon=horizon,
                              output_dim=output_dim,
                              four_step_method_included=four_step_method_included,
                              history_distribution_included=history_distribution_included,
                              traffic_assignment_included=traffic_assignment_included,
                              depart_freq_included=depart_freq_included,
                              logger=logger,
                              detail=True,
                              cfg=cfg)

        val_loss_list.append(val_result['MAE_od'])
        val_mae_list.append(val_result['MAE_od'])
        val_mape_list.append(val_result['MAPE_net_od'])
        val_rmse_list.append(val_result['RMSE_od'])
        epochs_idx.append(epoch + 1)

        time_elapsed = time.perf_counter() - begin_time

        if trial_number is None:
            with torch.no_grad():
                val_iter = dataset['val_loader'].get_iterator()
                y_preds_list = run_model(
                    model, val_iter,
                    num_nodes, edge_index, edge_attr, device,
                    seq_len, horizon, output_dim,
                    four_step_method_included,
                    history_distribution_included,
                    traffic_assignment_included,
                    depart_freq_included
                )
            y_preds = np.concatenate(y_preds_list, axis=0)
            y_preds = scaler_od.inverse_transform(y_preds)
            gt      = dataset['y_val']
            y_truth = scaler_od.inverse_transform(gt)
            min_len = min(y_preds.shape[0], y_truth.shape[0])
            y_preds = y_preds[:min_len]
            y_truth = y_truth[:min_len]
            mean_err = compute_epoch_od_errors(y_preds, y_truth)
            epoch_od_errors_list.append(mean_err)

        val_category = ['od']
        for category in val_category:
            logger.info('{}:'.format(category))
            logger.info(('val_mae:{}, val_mape_net:{}'
                         'r_loss={:.2f},lr={},  time_elapsed:{}').format(
                val_result['MAE_' + category],
                val_result['MAPE_net_' + category],
                0,
                str(scheduler.get_lr()),
                time_elapsed))
            if update['last_val_mae_' + category] > val_result['MAE_' + category]:
                logger.info('val_mae decreased from {} to {}'.format(
                    update['last_val_mae_' + category],
                    val_result['MAE_' + category]))
                update['last_val_mae_' + category] = val_result['MAE_' + category]
                update['val_steady_count_' + category] = 0
            else:
                update['val_steady_count_' + category] += 1

            if update['last_val_mape_net_' + category] > val_result['MAPE_net_' + category]:
                logger.info('val_mape_net decreased from {} to {}'.format(
                    update['last_val_mape_net_' + category],
                    val_result['MAPE_net_' + category]))
                update['last_val_mape_net_' + category] = val_result['MAPE_net_' + category]

        if (epoch + 1) % cfg['train']['test_every_n_epochs'] == 0:
            test_result = evaluate(model=model,
                     dataset=dataset,
                     dataset_type='test',
                     num_nodes=num_nodes,
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     device=device,
                     seq_Len=seq_len,
                     horizon=horizon,
                     output_dim=output_dim,
                     four_step_method_included=four_step_method_included,
                     history_distribution_included=history_distribution_included,
                     traffic_assignment_included=traffic_assignment_included,
                     depart_freq_included=depart_freq_included,
                     logger=logger,
                     cfg=cfg)
            test_loss_list.append(test_result['MAE_od'])
            test_mae_list.append(test_result['MAE_od'])
            test_mape_list.append(test_result['MAPE_net_od'])
            test_rmse_list.append(test_result['RMSE_od'])
        else:
            test_loss_list.append(None)
            test_mae_list.append(None)
            test_mape_list.append(None)
            test_rmse_list.append(None)

        if MODE == "train":
            epoch_model_path = os.path.join(all_epochs_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), epoch_model_path)

        current_val_mae = val_result['MAE_od']
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"[Trial {trial_number}] new best MAE={current_val_mae:.4f}, "
                f"saved model to {best_model_path}"
            )
            if MODE == "train":
                best_epoch = epoch + 1

        if train_patience <= update['val_steady_count_od']:
            logger.info('early stopping.')
            break
        scheduler.step()

    df = pd.DataFrame(weight_history,
                      columns=['w_station_count', 'w_transfer_count'])
    df.index += 1
    df.index.name = 'epoch'
    excel_path = os.path.join(log_dir, 'utility_weights.xlsx')
    df.to_excel(excel_path)
    print(f"Saved utility weights table to {excel_path}")

    plot_step_points_from_excel(
        filepath=excel_path,
        columns=['w_station_count', 'w_transfer_count'],
        step=15,
        output_pdf=os.path.join(excel_path,'utility_weights.pdf')
    )

    plot_training_curves(
        train_losses,
        val_mae_list,
        val_mape_list,
        val_rmse_list,
        save_dir=log_dir
    )

    no_pinn_dir = os.path.join(log_dir, 'no_pinn_loss')
    plot_training_curves(
        od_losses,
        val_mae_list,
        val_mape_list,
        val_rmse_list,
        save_dir = no_pinn_dir)

    x_epochs = epochs_idx

    plot_metric_pair(x_epochs, train_losses, val_loss_list, 'Loss', log_dir)
    plot_metric_pair(x_epochs, train_mae_list, val_mae_list, 'MAE', log_dir)
    plot_metric_pair(x_epochs, train_mape_list, val_mape_list, 'MAPE', log_dir)
    plot_metric_pair(x_epochs, train_rmse_list, val_rmse_list, 'RMSE', log_dir)

    one_OD_pair = True
    if one_OD_pair:
        if trial_number is None:
            errors = np.stack(epoch_od_errors_list, axis=0)  # shape (E, n_nodes, n_nodes)
            initial = errors[0]
            final = errors[-1]
            ratio = abs((initial - final) / (initial + 1e-8))
            n_nodes = ratio.shape[0]
            y_true_all = scaler_od.inverse_transform(dataset['y_val'])
            all_pairs = [(i, j) for i in range(n_nodes) for j in range(input_dim_m - 1)]
            valid_pairs = [
                (i, j) for (i, j) in all_pairs
                if not np.all(y_true_all[..., i, j] == 0)
            ]
            all_scores = [ratio[i, j] for (i, j) in valid_pairs]
            k = 5
            topk_idx = np.argsort(all_scores)[::-1][:k]
            topk_pairs = [all_pairs[idx] for idx in topk_idx]
            plot_od_losses(
                epoch_od_errors_list,
                topk_pairs,
                save_path=os.path.join(log_dir, f'top{k}_od_losses.pdf')
            )

            y_pred_all = np.concatenate(y_preds_list, axis=0)[:min_len]
            y_true_all = scaler_od.inverse_transform(dataset['y_val'])
            y_pred_all = scaler_od.inverse_transform(y_pred_all)
            min_len = min(y_pred_all.shape[0], dataset['y_val'].shape[0])

            plot_od_time_series(
                y_true_all,
                y_pred_all,
                topk_pairs,
                save_dir=os.path.join(log_dir, f'od_series')
            )

            plot_od_time_series_first_step(
                y_true_all,
                y_pred_all,
                topk_pairs,
                save_dir=os.path.join(log_dir, f'od_series')
            )

    metrics_df = pd.DataFrame(OrderedDict([
        ('Epoch', x_epochs),
        ('Train_Loss', train_losses),
        ('Val_Loss', val_loss_list),
        ('Train_MAE', train_mae_list),
        ('Val_MAE', val_mae_list),
        ('Train_MAPE', train_mape_list),
        ('Val_MAPE', val_mape_list),
        ('Train_RMSE', train_rmse_list),
        ('Val_RMSE', val_rmse_list),
    ]))
    excel_path = os.path.join(log_dir, 'training_validation_metrics.xlsx')
    metrics_df.to_excel(excel_path, index=False)
    print(f"Saved metrics Excel to: {excel_path}")

    if MODE == "train" and best_epoch is not None:
        best_epoch_path = os.path.join(log_dir, "best_epoch.txt")
        with open(best_epoch_path, "w") as f:
            f.write(str(best_epoch))
        print(f"Saved best epoch ({best_epoch}) to: {best_epoch_path}")

    best_val_mae = update['last_val_mae_od']
    return best_val_mae

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main(args)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(100)