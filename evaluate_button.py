# encoding:utf-8
import os
import random
import yaml
import numpy as np
import torch
from config import args
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.init import xavier_uniform_
from lib import utils_CUROP as utils
from models.Net_0207 import Net_0207
from train_evaluate_functions import evaluate
from train_evaluate_functions import _get_log_dir
from train_evaluate_functions import run_model
import optuna

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
"""parser = argparse.ArgumentParser()
parser.add_argument('--config_filename',
                    default=None,
                    type=str,
                    help='Configuration filename for restoring the model.')
args = argparse.Namespace(config_filename='data/config/eval_sz_dim35_units96_h4c512.yaml')"""

def read_cfg_file(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg

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

def init_weights(m):
    classname = m.__class__.__name__  # 2
    if classname.find('Conv') != -1 and classname.find('RGCN') == -1:
        xavier_uniform_(m.weight.data)
    if type(m) == nn.Linear:
        xavier_uniform_(m.weight.data)
        #xavier_uniform_(m.bias.data)

def toDevice(datalist, device):
    for i in range(len(datalist)):
        datalist[i] = datalist[i].to(device)
    return datalist

def main(args):
    # 加载配置文件
    cfg = read_cfg_file(args.config_filename)

    # 初始化日志
    log_dir = _get_log_dir(cfg)
    log_level = cfg.get('log_level', 'INFO')
    logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)

    # 选择设备
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    #  all edge_index in same dataset is same
    # edge_index = adjacency_to_edge_index(adj_mx)  # alreay added self-loop

    logger.info(cfg)
    batch_size = cfg['data']['batch_size'] #
    seq_len = cfg['model']['seq_len']
    horizon = cfg['model']['horizon']
    num_nodes = cfg['model']['num_nodes']
    four_step_method_included = cfg['domain_knowledge_loaded']['four_step_method']
    history_distribution_included = cfg['domain_knowledge_loaded']['history_distribution']
    traffic_assignment_included = cfg['domain_knowledge_types_included']['traffic_assignment']
    depart_freq_included = cfg['domain_knowledge_types_included']['depart_freq']
    Using_GAT_or_RGCN = cfg['domain_knowledge']['Using_GAT_or_RGCN']
    # edge_index = utils.load_pickle(cfg['data']['edge_index_pkl_filename'])

    adj_mx_list = []
    graph_pkl_filename = cfg['data']['graph_pkl_filename']

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
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
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

    model = Net_0207(cfg, logger).to(device)
    state = torch.load(cfg['model']['save_path'])
    model.load_state_dict(state, strict=False)
    evaluate(model=model,
             dataset=dataset,
             dataset_type='test',
             edge_index=edge_index,
             edge_attr=edge_attr,
             device=device,
             seq_Len=seq_len,
             horizon=horizon,
             output_dim=output_dim,
             num_nodes=num_nodes,
             four_step_method_included=four_step_method_included,
             history_distribution_included=history_distribution_included,
             traffic_assignment_included=traffic_assignment_included,
             depart_freq_included=depart_freq_included,
             logger=logger,
             cfg=cfg)

    test_iter = dataset['test_loader'].get_iterator()
    y_preds_list = run_model(
        model, test_iter,
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        device=device,
        seq_len=seq_len,
        horizon=horizon,
        output_dim=output_dim,
        four_step_method_included=four_step_method_included,
        history_distribution_included=history_distribution_included,
        traffic_assignment_included=traffic_assignment_included,
        depart_freq_included=depart_freq_included
    )
    y_pred = np.concatenate(y_preds_list, axis=0)
    scaler = dataset['scaler']
    y_pred = scaler.inverse_transform(y_pred)
    y_true = scaler.inverse_transform(dataset['y_test'])
    min_len = min(y_pred.shape[0], y_true.shape[0])
    y_pred = y_pred[:min_len]
    y_true = y_true[:min_len]
    np.save(os.path.join(log_dir, 'test_pred.npy'), y_pred)
    np.save(os.path.join(log_dir, 'test_true.npy'), y_true)
    print("save"+os.path.join(log_dir, 'test_pred.npy'))

if __name__ == "__main__":
    main(args)
