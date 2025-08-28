from typing import List

import numpy as np
from pykalman import KalmanFilter


class CAKalmanFilter:
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
        self.dt = dt
        F_constant, H_constant, Q_constant = constant_acceleration_matrices(dim, dt)
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

    def _project_Q_to_white_jerk(self, q_bounds=(1e-6, 1e+1), tie_axes=True):
        Q = 0.5 * (self.kf.transition_covariance + self.kf.transition_covariance.T)
        n = 3 * self.dim
        dt = self.dt  # store dt in the class
        B = np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                      [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                      [dt ** 3 / 6, dt ** 2 / 2, dt]], float)
        bb = B.reshape(-1);
        denom = float(bb @ bb)

        Qp = np.zeros((n, n), float);
        qs = []
        for a in range(self.dim):
            i = 3 * a
            Qa = Q[i:i + 3, i:i + 3]
            q_hat = float((Qa.reshape(-1) @ bb) / denom)  # LS fit onto B
            q_hat = float(np.clip(q_hat, *q_bounds))
            qs.append(q_hat)
            Qp[i:i + 3, i:i + 3] = q_hat * B

        if tie_axes:
            q_med = float(np.clip(np.median(qs), *q_bounds))
            for a in range(self.dim):
                Qp[3 * a:3 * a + 3, 3 * a:3 * a + 3] = q_med * B
        self.kf.transition_covariance = Qp

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
                n = 3 * self.dim
                x0 = np.zeros(n)
                for a in range(self.dim):
                    x0[3 * a + 0] = float(measurements[0, a])  # pos
                    x0[3 * a + 1] = 0.0  # vel unknown
                    x0[3 * a + 2] = 0.0  # acc unknown
                self.kf.initial_state_mean = x0
                self.kf = self.kf.em(measurements, n_iter=1, em_vars=vars_to_estimate)

                # These methods already update the kf matrices
                self._project_R_to_diag()
                self._project_Q_to_white_jerk()
        xf, Pf = self.kf.filter(measurements)
        xs, Ps = self.kf.smooth(measurements)
        dim = measurements.shape[1]
        out = {
            "filtered_x": xf, "filtered_P": Pf,
            "smoothed_x": xs, "smoothed_P": Ps,
            "pos_filt": xf[:, :3 * dim:2],  # take every (pos,vel) pair's pos -> indices 0,2,(4)
            "vel_filt": xf[:, 1:3 * dim:2],  # corresponding velocities -> indices 1,3,(5)
            "pos_smooth": xs[:, :3 * dim:2],
            "vel_smooth": xs[:, 1:3 * dim:2],
        }
        if em_iters > 0:
            out["R_learned"] = self.kf.observation_covariance
            out["F_learned"] = self.kf.transition_matrices
        return out

def constant_acceleration_matrices(dim: int, dt: float):
    assert dim in (2, 3)
    # per-axis F3 and H3
    F3 = np.array([[1, dt, 0.5 * dt * dt],
                   [0, 1, dt],
                   [0, 0, 1]], float)
    H3 = np.array([[1, 0, 0]], float)  # observe position only

    q = 1e1
    # White-jerk spectral density q per axis → Q block:
    Q_axis = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                           [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                           [dt ** 3 / 6, dt ** 2 / 2, dt]])

    # assemble block-diagonal F and stacked H
    Z = np.zeros((3, 3))
    if dim == 2:
        F = np.block([[F3, Z],
                      [Z, F3]])
        H = np.block([[H3, np.zeros((1, 3))],
                      [np.zeros((1, 3)), H3]])
        Q = np.block([[Q_axis, Z],
                     [Z, Q_axis]])
    else:
        F = np.block([[F3, Z, Z],
                      [Z, F3, Z],
                      [Z, Z, F3]])
        H = np.block([[H3, np.zeros((1, 6))],
                      [np.zeros((1, 3)), H3, np.zeros((1, 3))],
                      [np.zeros((1, 6)), H3]])
        Q = np.block([[Q_axis, Z, Z],
                     [Z, Q_axis, Z],
                      [Z, Z, Q_axis]])
    return F, H, Q
