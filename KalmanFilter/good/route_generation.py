import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)



def generate_cv_trajectory(T, dt, dim=3, v=None, x0=None, noise_std=None, number_of_trajectories=1, seed=None):
    """
    Generate a constant-velocity trajectory with additive Gaussian noise.

    Args:
        T (int): number of time steps.
        dt (float): time step (constant).
        dim (int): 2 or 3. (Output is always shaped [T, 3]; z is zero if dim=2.)
        v (array-like | None): velocity per axis (len=dim). Default zeros.
        x0 (array-like | None): initial position per axis (len=dim). Default zeros.
        noise_std (float | array-like | None): per-axis std of additive Gaussian noise.
            If scalar, applied to all dims. Default zeros (no noise).
        seed (int | None): RNG seed for reproducibility.

    Returns:
        traj_noisy (np.ndarray): shape (T, 3) noisy positions.
    """
    assert dim in (2, 3), "dim must be 1, 2 or 3"
    rng = np.random.default_rng(seed)
    #Setting the speed
    v = np.random.uniform(low=-4, high=4, size=(dim)) if v is None else np.asarray(v, dtype=float)

    #Setting initial state
    x0 = np.random.uniform(low=0, high=4, size=(dim)) if x0 is None else np.asarray(x0, dtype=float)

    #Generate noise std vector
    if noise_std is None:
        noise_std = np.zeros(dim, dtype=float)
    else:
        noise_std = np.asarray(noise_std, dtype=float)
        if noise_std.ndim == 0:
            noise_std = np.full(dim, float(noise_std))

    trajectories = []
    for i in range(number_of_trajectories):
        #represents the equation: x = x0 + v*t (at time t)
        #Using broadcastin here
        t = np.arange(T, dtype=float) * dt
        t = t.reshape(T, 1)
        clean = x0.reshape(1, -1) + t * v.reshape(1, -1)

        # additive Gaussian noise per time & axis
        noise = rng.normal(loc=0.0, scale=noise_std[None, :], size=(T, dim))
        noisy = clean + noise
        trajectories.append((noisy, clean))

    return trajectories

import numpy as np

def generate_cv_with_micro_velocity_changes(
    T: int,
    dt: float,
    dim: int = 2,
    v: np.ndarray | None = None,         # initial velocity per axis
    x0: np.ndarray | None = None,        # initial position per axis
    vel_change_std: float | np.ndarray = 0.0, # per-axis std of *acceleration* (units: pos/s^2)
    noise_std: float | np.ndarray = 0.0, # per-axis std of *measurement* noise (units: pos)
    number_of_trajectories: int = 1,
    seed: int | None = None,
):
    """
    Generate nearly-constant-velocity trajectories with small random accelerations.
    State evolution per axis a at each step t (t=1..T-1):
        a_t ~ N(0, accel_std^2)
        v_t   = v_{t-1} + a_t * dt
        x_t   = x_{t-1} + v_{t-1} * dt + 0.5 * a_t * dt^2

    Args:
        T, dt, dim: length, timestep, and dimensionality (2 or 3).
        v:   initial velocity vector (len=dim). If None, sampled U[-4,4]^dim.
        x0:  initial position vector (len=dim). If None, sampled U[0,4]^dim.
        vel_change_std:  per-axis std of *acceleration* noise. Scalar or array (len=dim).
        noise_std:  per-axis std of additive *measurement* noise. Scalar or array.
        number_of_trajectories: how many (noisy, clean) pairs to generate.
        seed: RNG seed.

    Returns:
        list of (noisy, clean), each np.ndarray with shape (T, dim).
        - clean: true positions including micro-acceleration (no measurement noise)
        - noisy: clean + measurement noise
    """
    assert dim in (2, 3), "dim must be 2 or 3"
    rng = np.random.default_rng(seed)

    # Broadcast helpers
    def as_vec(x, name):
        if isinstance(x, (int, float)):
            return np.full(dim, float(x))
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return np.full(dim, float(x))
        assert x.shape == (dim,), f"{name} must be length {dim}"
        return x

    vel_change_std = as_vec(vel_change_std, "accel_std")
    noise_std = as_vec(noise_std, "noise_std")

    trajectories = []
    for _ in range(number_of_trajectories):
        v0 = (rng.uniform(-4, 4, size=dim) if v is None else as_vec(v, "v"))
        x0 = (rng.uniform(0, 4, size=dim) if x0 is None else as_vec(x0, "x0"))

        clean = np.empty((T, dim), dtype=float)
        vel = np.empty((T, dim), dtype=float)

        clean[0] = x0
        vel[0]   = v0

        # step-by-step evolution with white acceleration
        dt2 = dt * dt
        for t in range(1, T):
            a_t = rng.normal(0.0, vel_change_std, size=dim)  # acceleration at step t
            vel[t] = vel[t - 1] + a_t
            clean[t] = clean[t - 1] + vel[t - 1] * dt

        # add measurement noise on positions
        meas_noise = rng.normal(0.0, noise_std, size=(T, dim))
        noisy = clean + meas_noise

        trajectories.append((noisy, clean))

    return trajectories


def generate_ca_trajectory(T, dt, dim=3, a=None, v=None, x_start=None, measurement_noise_std=None, seed=None,
                           number_of_trajectories:int = 1):
    """
    Generate a constant-velocity trajectory with additive Gaussian noise.
    The noise is added only AFTER the trajectory is calculated
    Args:
        T (int): number of time steps.
        dt (float): time step (constant).
        dim (int): 2 or 3. (Output is always shaped [T, 3]; z is zero if dim=2.)
        v (array-like | None): velocity per axis (len=dim). Default zeros.
        x0 (array-like | None): initial position per axis (len=dim). Default zeros.
        measurement_noise_std (float | array-like | None): per-axis std of additive Gaussian noise.
            If scalar, applied to all dims. Default zeros (no noise).
        seed (int | None): RNG seed for reproducibility.

    Returns:
        traj_noisy (np.ndarray): shape (T, 3) noisy positions.
    """
    def as_vec(x, name):
        if isinstance(x, (int, float)):
            return np.full(dim, float(x))
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return np.full(dim, float(x))
        assert x.shape == (dim,), f"{name} must be length {dim}"
        return x

    assert dim in (2, 3), "dim must be 1, 2 or 3"
    rng = np.random.default_rng(seed)
    trajectories = []
    for _ in range(number_of_trajectories):
        v0 = (rng.uniform(-4, 4, size=dim) if v is None else as_vec(v, "v"))
        x0 = (rng.uniform(0, 4, size=dim) if x_start is None else as_vec(x_start, "x_start"))
        a = (rng.uniform(-2, 2, size=dim) if a is None else as_vec(x0, "x0"))

        clean = np.empty((T, dim), dtype=float)
        vel = np.empty((T, dim), dtype=float)

        clean[0] = x0
        vel[0] = v0

        #Setting the acceleration
        a = np.zeros(dim, dtype=float) if a is None else np.asarray(a, dtype=float)

        #Setting the speed
        v0 = np.zeros(dim, dtype=float) if v0 is None else np.asarray(v0, dtype=float)

        #Setting initial state
        x0 = np.zeros(dim, dtype=float) if x0 is None else np.asarray(x0, dtype=float)

        #Generate noise std vector
        if measurement_noise_std is None:
            measurement_noise_std = np.zeros(dim, dtype=float)
        else:
            measurement_noise_std = np.asarray(measurement_noise_std, dtype=float)
            if measurement_noise_std.ndim == 0:
                measurement_noise_std = np.full(dim, float(measurement_noise_std))

        #represents the equation: x = x0 + v0*t + 0.5*a*t^2 (at time t)
        #Using broadcastin here
        t = np.arange(T, dtype=float) * dt
        t = t.reshape(T, 1)
        x0 = x0.reshape(1, -1)
        v0 = v0.reshape(1, -1)
        a = a.reshape(1, -1)
        clean = x0 + v0*t + 0.5 * a * t**2

        # additive Gaussian noise per time & axis
        noise = rng.normal(loc=0.0, scale=measurement_noise_std[None, :], size=(T, dim))
        noisy = clean + noise
        trajectories.append((noisy, clean))

    return trajectories
