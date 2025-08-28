import numpy as np
from pykalman import KalmanFilter


def compute_nll(kf: KalmanFilter, traj: np.ndarray):
    return -kf.loglikelihood(traj)


def compute_mse(kf_est: np.ndarray, true_pos: np.ndarray):
    err = kf_est - true_pos
    mse_axes = np.mean(err ** 2)
    mse_all = float(np.mean(err ** 2))
    return {"mse": mse_all, "per_axis": mse_axes}
