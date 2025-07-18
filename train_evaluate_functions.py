# filename: train_evaluate_functions.py
from lib.utils_CUROP import collate_wrapper
import torch
import numpy as np
from lib import metrics
import time
import os
from torch.optim.lr_scheduler import MultiStepLR

def run_model(model, data_iterator, num_nodes,
              edge_index, edge_attr, device, seq_len, horizon, output_dim,
              four_step_method_included,
              history_distribution_included,
              traffic_assignment_included,
              depart_freq_included):
    model.eval()
    y_od_pred_list = []
    for _, batch in enumerate(data_iterator):
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
        y_od = y_od[..., :output_dim]
        sequences, sequences_y, y_od = collate_wrapper(
            x_od=x_od, y_od=y_od,
            num_nodes=num_nodes,
            edge_index=edge_index, edge_attr=edge_attr, device=device, seq_len=seq_len, horizon=horizon,
            unfinished=unfinished if history_distribution_included else None,
            history=history if history_distribution_included else None,
            yesterday=yesterday if history_distribution_included else None,
            PINN_od_features=PINN_od_features if four_step_method_included else None,
            PINN_od_additional_features=PINN_od_additional_features if four_step_method_included else None,
            #OD_feature_array=OD_feature_array if four_step_method_included else None,
            Time_DepartFreDic_Array=Time_DepartFreDic_Array if depart_freq_included else None,
            repeated_sparse_5D_tensors=repeated_sparse_5D_tensors if traffic_assignment_included else None
        )
        # (T, N, num_nodes, num_out_channels)
        with torch.no_grad():
            y_od_pred = model(sequences, sequences_y)
            if y_od_pred is not None:
                y_od_pred_list.append(y_od_pred.detach().cpu().numpy())

    return y_od_pred_list


def evaluate(model,
            dataset,
            dataset_type,
            num_nodes,
            edge_index,
            edge_attr,
            device,
            seq_Len,
            horizon,
            output_dim,
            four_step_method_included,
            history_distribution_included,
            traffic_assignment_included,
            depart_freq_included,
            logger,
            detail=True,
            cfg=None,
            format_result=False):
    if detail:
        logger.info('Evaluation_{}_Begin:'.format(dataset_type))

    y_od_preds = run_model(
    model,
    data_iterator=dataset['{}_loader'.format(dataset_type)].get_iterator(),
    num_nodes=num_nodes,
    edge_index=edge_index,
    edge_attr=edge_attr,
    device=device,
    seq_len=seq_Len,
    horizon=horizon,
    output_dim=output_dim,
    four_step_method_included=four_step_method_included,
    history_distribution_included=history_distribution_included,
    traffic_assignment_included=traffic_assignment_included,
    depart_freq_included=depart_freq_included)

    evaluate_category = []
    if len(y_od_preds) > 0:
        evaluate_category.append("od")
    results = {}
    for category in evaluate_category:
        if category == 'od':
            y_preds = y_od_preds
            scaler = dataset['scaler']
            gt = dataset['y_{}'.format(dataset_type)]

        y_preds = np.concatenate(y_preds, axis=0)  # concat in batch_size dim.
        mae_list = []
        mape_net_list = []
        rmse_list = []
        mae_sum = 0

        mape_net_sum = 0
        rmse_sum = 0
        # horizon = dataset['y_{}'.format(dataset_type)].shape[1]
        logger.info("{}:".format(category))
        horizon = cfg['model']['horizon']
        for horizon_i in range(horizon):
            y_truth = scaler.inverse_transform(
                gt[:, horizon_i, :, :output_dim])

            y_pred = scaler.inverse_transform(
                y_preds[:y_truth.shape[0], horizon_i, :, :output_dim])
            y_pred[y_pred < 0] = 0
            mae = metrics.masked_mae_np(y_pred, y_truth)
            mape_net = metrics.masked_mape_np(y_pred, y_truth)
            rmse = metrics.masked_rmse_np(y_pred, y_truth)
            mae_sum += mae
            mape_net_sum += mape_net
            rmse_sum += rmse
            mae_list.append(mae)

            mape_net_list.append(mape_net)
            rmse_list.append(rmse)

            msg = "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE_net: {:.4f}"
            if detail:
                logger.info(msg.format(horizon_i + 1, mae, rmse, mape_net))
        results['MAE_' + category] = mae_sum / horizon
        results['RMSE_' + category] = rmse_sum / horizon
        results['MAPE_net_' + category] = mape_net_sum / horizon
    if detail:
        logger.info('Evaluation_{}_End:'.format(dataset_type))
    if format_result:
        for i in range(len(mae_list)):
            print('{:.2f}'.format(mae_list[i]))
            print('{:.2f}'.format(rmse_list[i]))
            print('{:.2f}%'.format(mape_net_list[i] * 100))
            print()
    else:
        # return mae_sum / horizon, rmse_sum / horizon, mape_sta_sum / horizon, mape_pair_sum / horizon, mape_net_sum/ horizon, mape_distribution_sum / horizon
        return results

def _get_log_dir(kwargs):
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        batch_size = kwargs['data'].get('batch_size')
        learning_rate = kwargs['train'].get('base_lr')
        num_rnn_layers = kwargs['model'].get('num_rnn_layers')
        rnn_units = kwargs['model'].get('rnn_units')
        structure = '-'.join(['%d' % rnn_units for _ in range(num_rnn_layers)])

        run_id = 'CUROP_%s_lr%g_bs%d_%s/' % (
            structure,
            learning_rate,
            batch_size,
            time.strftime('%m%d%H%M%S'))
        base_dir = kwargs.get('base_dir')
        log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):

        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate
