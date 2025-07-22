import numpy as np
from scipy.optimize import minimize
import os
from metro_data_convertor.Find_project_root import Find_project_root
import pickle
import torch
from torch.optim import Adam

def impedance_function(C, gamma):
    C = torch.tensor(C, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    return torch.pow(C + 1e-6, -gamma)

def compute_flow(O, D, C, gamma, a, b):
    f_c = impedance_function(C, gamma)

    if isinstance(a, torch.Tensor) == False:
        a = torch.tensor(a, dtype=torch.float32)
    if isinstance(b, torch.Tensor) == False:
        b = torch.tensor(b, dtype=torch.float32)
    if isinstance(O, torch.Tensor) == False:
        O = torch.tensor(O, dtype=torch.float32)
    if isinstance(D, torch.Tensor) == False:
        D = torch.tensor(D, dtype=torch.float32)
    if isinstance(f_c, torch.Tensor) == False:
        f_c = torch.tensor(f_c, dtype=torch.float32)

    q_v = torch.ger(a, b) * O[:, None].flatten() * D[None, :].flatten() * f_c
    q_v.fill_diagonal_(0)
    return q_v

def objective_function(params, O_data, D_data, C_data, q_obs_data, time_steps):
    gamma = params[0]
    a = params[1:len(O_data[0]) + 1]
    b = params[len(O_data[0]) + 1:]

    total_mse = 0
    for t in range(time_steps):
        O = O_data[t]
        D = D_data[t]
        C = C_data
        q_obs = q_obs_data[t]

        q_v = compute_flow(O, D, C, gamma, a, b)

        mse = torch.mean((q_v - torch.tensor(q_obs, dtype=q_v.dtype, device=q_v.device)) ** 2)

        total_mse += mse

    return total_mse


def fit_trip_distribution_model(O_data, D_data, C_data, q_obs_data, time_steps, initial_gamma, lr_gener, maxiter):
    """
    Parameters:
    O_data: list of torch.Tensor, departure volume data stored by time points.
    D_data: list of torch.Tensor, arrival volume data stored by time points.
    C_data: torch.Tensor, transportation impedance matrix, assuming the impedance matrix is the same at each time point.
    q_obs_data: list of torch.Tensor, observed flow data stored by time points.
    time_steps: int, number of time steps.
    initial_gamma: float, initial gamma parameter, default value is 0.5.
    lr_gener: float, learning rate.
    maxiter: int, maximum number of iterations, default value is 1000.

    Returns:
    optimal_gamma: float, optimal gamma parameter.
    a_fitted: torch.Tensor, optimal a parameter.
    b_fitted: torch.Tensor, optimal b parameter.
    q_predicted_list: list of torch.Tensor, predicted flow matrices for each time point.
    """

    initial_a = torch.ones(len(O_data[0]), requires_grad=True)
    initial_b = torch.ones(len(D_data[0]), requires_grad=True)
    gamma = torch.tensor([initial_gamma], requires_grad=True)

    optimizer = Adam([gamma, initial_a, initial_b], lr=lr_gener)

    for i in range(maxiter):
        optimizer.zero_grad()

        loss = objective_function(torch.cat([gamma, initial_a, initial_b]), O_data, D_data, C_data, q_obs_data,
                                  time_steps)

        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    optimal_gamma = gamma.item()
    a_fitted = initial_a.detach()
    b_fitted = initial_b.detach()

    q_predicted_list = []
    for t in range(time_steps):
        O_new = O_data[t]
        D_new = D_data[t]
        q_predicted = compute_flow(O_new, D_new, C_data, optimal_gamma, a_fitted, b_fitted)
        q_predicted_list.append(q_predicted)

        print(f'Optimal gamma parameter: {optimal_gamma}')
        print(f'Optimal a parameter: {a_fitted}')
        print(f'Optimal b parameter: {b_fitted}')

    return optimal_gamma, a_fitted, b_fitted, q_predicted_list
