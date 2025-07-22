from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
import os
from RecurrentGCN_for_trip_generation import RecurrentGCN


def test_PYGT(base_dir, prefix, str_prdc_attr):
    dir_path = os.path.join(base_dir, f'test_dataset.pkl')
    with open(dir_path, 'rb') as f:
        test_dataset = pickle.load(f, errors='ignore')

    hyperparams_path = os.path.join(base_dir, f'hyperparameters.pkl')
    with open(hyperparams_path, 'rb') as f:
        hyperparameters = pickle.load(f)

    RGCN_node_features = hyperparameters['RGCN_node_features']
    RGCN_hidden_units = hyperparameters['RGCN_hidden_units']
    RGCN_output_dim = hyperparameters['RGCN_output_dim']
    RGCN_K = hyperparameters['RGCN_K']

    RecurrentGCN_model = RecurrentGCN(node_features=RGCN_node_features, hidden_units=RGCN_hidden_units, output_dim=RGCN_output_dim,
                                      K=RGCN_K)

    model_path = os.path.join(base_dir, f"{str_prdc_attr}_RecurrentGCN_model.pth")
    RecurrentGCN_model.load_state_dict(torch.load(model_path))

    RecurrentGCN_model.eval()
    cost_PINN = 0
    with torch.no_grad():
        for snap_time, snapshot in enumerate(test_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            non_zero_mask = snapshot.y != 0
            mape = torch.mean(
                torch.abs((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) / snapshot.y[non_zero_mask])) * 100
            cost_PINN = cost_PINN + mape
        cost_PINN = cost_PINN / (snap_time + 1)
        cost_PINN = cost_PINN.item()

    print("MSE: {:.4f}".format(cost_PINN))

