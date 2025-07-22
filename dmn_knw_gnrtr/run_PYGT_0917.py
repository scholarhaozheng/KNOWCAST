from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
import os
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from tqdm import tqdm
from RecurrentGCN_for_trip_generation import RecurrentGCN

def run_PYGT(base_dir, prefix, str_prdc_attr, RGCN_node_features, RGCN_hidden_units, RGCN_output_dim, RGCN_K, lr, epoch_num, train_ratio):
    dir_path = os.path.join(base_dir, f'{prefix}_{str_prdc_attr}_signal_dict.pkl')
    with open(dir_path, 'rb') as f:
        signal_dict = pickle.load(f, errors='ignore')

    signal = StaticGraphTemporalSignal(
        features=signal_dict["features"],
        targets=signal_dict["targets"],
        additional_feature1=signal_dict["additional_feature"],
        edge_index=signal_dict["edge_index"],
        edge_weight=signal_dict["edge_weight"]
    )

    from torch_geometric_temporal.signal import temporal_signal_split
    train_dataset, test_dataset = temporal_signal_split(signal, train_ratio=train_ratio)

    with open(os.path.join(base_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)

    RecurrentGCN_model = RecurrentGCN(node_features=RGCN_node_features, hidden_units=RGCN_hidden_units,
                                      output_dim=RGCN_output_dim, K=RGCN_K)

    optimizer = torch.optim.Adam(RecurrentGCN_model.parameters(), lr=lr)

    RecurrentGCN_model.train()
    for epoch in tqdm(range(epoch_num)):
        cost_PINN = 0
        total_mape = 0
        for snap_time, snapshot in enumerate(train_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost_PINN = cost_PINN + torch.mean((y_hat - snapshot.y) ** 2)
            non_zero_mask = snapshot.y != 0
            mape = torch.mean(
                torch.abs((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) / snapshot.y[non_zero_mask])) * 100
            total_mape += mape

        cost_PINN = cost_PINN / (snap_time + 1)
        cost_PINN.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_mape = total_mape / (snap_time + 1)

        print(f"Epoch {epoch + 1}/{epoch_num}, MSE: {cost_PINN.item():.4f}, MAPE: {avg_mape.item():.4f}%")

    model_save_path = os.path.join(base_dir, f'{str_prdc_attr}_RecurrentGCN_model.pth')
    torch.save(RecurrentGCN_model.state_dict(), model_save_path)

    hyperparameters = {
        "RGCN_node_features": RGCN_node_features,
        "RGCN_hidden_units": RGCN_hidden_units,
        "RGCN_output_dim": RGCN_output_dim,
        "RGCN_K": RGCN_K
    }

    hyperparams_save_path = os.path.join(base_dir, f'hyperparameters.pkl')
    with open(hyperparams_save_path, 'wb') as f:
        pickle.dump(hyperparameters, f)
