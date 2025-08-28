from typing import List

import numpy as np
from pykalman import KalmanFilter


class CVKalmanFilter:
    def __init__(self,
                 dim: int,
                 dt: float,
                 F: np.ndarray = None,
                 H: np.ndarray = None,
                 Q: np.ndarray = None,
                 R: np.ndarray = None,
                 x0: np.ndarray = None,
                 P0: np.ndarray = None,
                 transition_offsets: np.ndarray = None,
                 process_noise: bool = False):
        self.dim = dim
        F_constant, H_constant, Q_constant = _constant_velocity_matrices(dim, dt, process_noise)
        F = F if F is not None else F_constant
        H = H if H is not None else H_constant
        Q = Q if Q is not None else Q_constant

        self.kf = KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_offsets=transition_offsets,
            transition_covariance=Q,  # Process noise
            observation_covariance=R,
            initial_state_mean=x0,
            initial_state_covariance=P0,
        )

    def _project_R_to_diag(self, rmin=1e-5, rmax=10) -> np.ndarray:
        R = self.kf.observation_covariance
        R = 0.5 * (R + R.T)
        d = np.clip(np.diag(R), rmin, rmax)
        self.kf.observation_covariance = np.diag(d)

    def _project_Q_to_vel_random_walk(self,
                                      tie_axes: bool = False,
                                      q_bounds=(1e-8, 1e+3)) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (Q_projected, qv_per_axis).
        """
        Q = self.kf.transition_covariance
        Q = 0.5 * (Q + Q.T)
        n = 2 * self.dim
        Qp = np.zeros((n, n), float)
        # read per-axis v-variance candidates
        qv_list = []
        for a in range(self.dim):
            i = 2 * a
            qv_list.append(max(0.0, float(Q[i + 1, i + 1])))
        if tie_axes:
            qv = float(np.clip(np.median(qv_list), *q_bounds))
            for a in range(self.dim):
                i = 2 * a
                Qp[i + 1, i + 1] = qv
            self.kf.transition_covariance = Qp
            return Qp, np.array([qv] * self.dim)
        else:
            qv_arr = np.clip(np.array(qv_list), q_bounds[0], q_bounds[1])
            for a, qv in enumerate(qv_arr):
                i = 2 * a
                Qp[i + 1, i + 1] = float(qv)
            self.kf.transition_covariance = Qp
            return Qp, qv_arr

    def run_cv(self, measurements: np.ndarray,
               em_iters: int = 0,
               estimate_v: bool = False,
               vars_to_estimate: List[str] = None):
        """
        measurements: [T, dim] positions only.
        em_iters=0 -> just filter/smooth with given R.
        em_iters>0 -> first estimate R via EM (only observation_covariance), then filter/smooth.

        Returns dict with filtered/smoothed states + learned R (if EM used).
        """
        if em_iters > 0:
            for i in range(em_iters):
                # kf = kf.em(measurements, n_iter=em_iters, em_vars=['observation_covariance'])
                if estimate_v:
                    vx0_hat = (measurements[1, 0] - measurements[0, 0]) / 0.04
                    vy0_hat = (measurements[1, 1] - measurements[0, 1]) / 0.04
                else:
                    vx0_hat = 0
                    vy0_hat = 0
                if self.dim == 3:
                    x0 = np.array([measurements[0, 0], 0, measurements[0, 1], 0, measurements[0, 2], 0])
                else:
                    x0 = np.array([measurements[0, 0], 0, measurements[0, 1], 0])
                self.kf.initial_state_mean = x0
                self.kf = self.kf.em(measurements, n_iter=1, em_vars=vars_to_estimate)

                #These methods already update the kf matrices
                self._project_R_to_diag()
                self._project_Q_to_vel_random_walk()
        xf, Pf = self.kf.filter(measurements)
        xs, Ps = self.kf.smooth(measurements)
        dim = measurements.shape[1]
        out = {
            "filtered_x": xf, "filtered_P": Pf,
            "smoothed_x": xs, "smoothed_P": Ps,
            "pos_filt": xf[:, :2 * dim:2],  # take every (pos,vel) pair's pos -> indices 0,2,(4)
            "vel_filt": xf[:, 1:2 * dim:2],  # corresponding velocities -> indices 1,3,(5)
            "pos_smooth": xs[:, :2 * dim:2],
            "vel_smooth": xs[:, 1:2 * dim:2],
        }
        if em_iters > 0:
            out["R_learned"] = self.kf.observation_covariance
            out["F_learned"] = self.kf.transition_matrices
        return out


def _constant_velocity_matrices(dim: int, dt: float, process_noise: bool = False):
    """
    Constant-velocity model (positions only observed).
    State per axis: [pos, vel]; overall state: [x, vx, y, vy, (z, vz)]
    Returns F, H, Q (Q=0 here).
    """
    assert dim in (2, 3)
    if dim == 2:
        F = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=float)
        H = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 0, 1, 0],  # measure y
        ], dtype=float)
        if not process_noise:
            Q = np.zeros((4, 4), dtype=float)  # no process noise
        else:
            Q = np.eye(4, 4) * 1e1  # for estimating the process noise later
    else:  # dim == 3
        F = np.array([
            [1, dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        H = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 0, 1, 0, 0, 0],  # y
            [0, 0, 0, 0, 1, 0],  # z
        ], dtype=float)
        if not process_noise:
            Q = np.zeros((6, 6), dtype=float)  # no process noise
        else:
            Q = np.eye(6, 6) * 1e1  # for estimating the process noise later

    return F, H, Q
