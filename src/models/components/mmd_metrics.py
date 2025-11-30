import math
from typing import Union

import numpy as np
import ot as pot
import torch
from scipy.optimize import linear_sum_assignment

from src.models.components.mmd import mix_rbf_mmd2
from src.models.components.optimal_transport import wasserstein


def energy_distances(pred, true, prefix=""):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    energy_w2 = math.sqrt(pot.emd2_1d(true, pred))
    energy_w1 = pot.emd2_1d(true, pred, metric="euclidean")
    mean_dist = np.abs(pred.mean() - true.mean())
    cropped_pred = np.clip(pred, -1000, 1000)
    cropped_true = np.clip(true, -1000, 1000)
    cropped_energy_w2 = math.sqrt(pot.emd2_1d(cropped_true, cropped_pred))
    cropped_energy_w1 = pot.emd2_1d(cropped_true, cropped_pred, metric="euclidean")
    return_dict = {
        f"{prefix}/energy_w2": energy_w2,
        f"{prefix}/energy_w1": energy_w1,
        f"{prefix}/mean_dist": mean_dist,
        f"{prefix}/cropped_energy_w2": cropped_energy_w2,
        f"{prefix}/cropped_energy_w1": cropped_energy_w1,
    }
    return return_dict


def compute_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def compute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
        "Eq-EMD2",
    ]
    a = pred
    b = true
    w1 = wasserstein(a, b, power=1)
    w2 = wasserstein(a, b, power=2)

    mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
    mean_dists = compute_distances(torch.mean(a, dim=0), torch.mean(b, dim=0))
    median_dists = compute_distances(torch.median(a, dim=0)[0], torch.median(b, dim=0)[0])
    dists = [w1, w2, mmd_rbf, *mean_dists, *median_dists]
    return NAMES, dists


def compute_distribution_distances_with_prefix(pred, true, prefix):
    pred = pred.cpu()
    true = true.cpu()
    names, dists = compute_distribution_distances(pred, true)
    names = [f"{prefix}/{name}" for name in names]
    metrics = dict(zip(names, dists))
    return metrics