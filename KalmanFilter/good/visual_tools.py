import numpy as np
from matplotlib import pyplot as plt

def plot_trajectory_with_kf(
    traj,
    show_clean=None,      # (T,d) optional
    kf_filt=None,         # (T,d) positions OR (T,2d) states [pos,vel] per axis
    kf_smooth=None,       # same as above
    title="Trajectory",
    equal_xy=True,
    legend_loc="best",
):
    """
    Plot trajectory (T,2) or (T,3). KF overlays are optional.

    If kf_* is full state (T,2d), the function auto-selects position columns [0,2,(4)].
    No dummy dimensions are added anywhere.

    Args:
        traj       : np.ndarray (T,2) or (T,3) noisy positions.
        show_clean : np.ndarray (T,d) clean positions (optional).
        kf_filt    : np.ndarray (T,d) or (T,2d) filtered (optional).
        kf_smooth  : np.ndarray (T,d) or (T,2d) smoothed (optional).
        title      : str
        equal_xy   : bool. Keep equal aspect for XY (and cubic box in 3D if supported).
        legend_loc : str
    """
    traj = np.asarray(traj, dtype=float)
    assert traj.ndim == 2 and traj.shape[1] in (2, 3), "traj must be (T,2) or (T,3)"
    T, d = traj.shape

    def _coerce_positions(arr, name):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.shape == (T, d):
            return arr
        # Accept full state (T, 2d): [pos0, vel0, pos1, vel1, ...]
        if arr.ndim == 2 and arr.shape[0] == T and arr.shape[1] == 2*d:
            return arr[:, 0::2]
        raise ValueError(f"{name} must be (T,{d}) positions or (T,{2*d}) full states; got {arr.shape}")

    show_clean = _coerce_positions(show_clean, "show_clean") if show_clean is not None else None
    kf_filt    = _coerce_positions(kf_filt, "kf_filt")       if kf_filt    is not None else None
    kf_smooth  = _coerce_positions(kf_smooth, "kf_smooth")   if kf_smooth  is not None else None

    # Decide 3D only from traj (match your original behavior)
    has_z = (d == 3) and (np.ptp(traj[:, 2]) > 1e-12)

    if has_z:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="noisy", lw=1.8)
        if show_clean is not None:
            ax.plot(show_clean[:, 0], show_clean[:, 1], show_clean[:, 2],
                    "--", label="clean", lw=1.5, alpha=0.8)
        if kf_filt is not None:
            ax.plot(kf_filt[:, 0], kf_filt[:, 1], kf_filt[:, 2],
                    "-..", label="KF filtered", lw=1.5)
        if kf_smooth is not None:
            ax.plot(kf_smooth[:, 0], kf_smooth[:, 1], kf_smooth[:, 2],
                    ":", label="KF smoothed", lw=2.0)

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(title)
        if equal_xy:
            try:
                ax.set_box_aspect((1, 1, 1))  # mpl >= 3.3
            except Exception:
                pass
        ax.legend(loc=legend_loc)
        plt.tight_layout()
        plt.show()
        return

    # 2D layout: XY path + components over time
    fig, (ax_xy, ax_t) = plt.subplots(1, 2, figsize=(12, 5))

    # XY path
    ax_xy.plot(traj[:, 0], traj[:, 1], label="noisy", lw=1.8)
    if show_clean is not None:
        ax_xy.plot(show_clean[:, 0], show_clean[:, 1], "--", label="clean", lw=1.5, alpha=0.8)
    if kf_filt is not None:
        ax_xy.plot(kf_filt[:, 0], kf_filt[:, 1], "-.", label="KF filtered", lw=1.5)
    if kf_smooth is not None:
        ax_xy.plot(kf_smooth[:, 0], kf_smooth[:, 1], ":", label="KF smoothed", lw=2.0)

    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_title("XY path")
    if equal_xy:
        ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True); ax_xy.legend(loc=legend_loc)

    # Components vs time
    t = np.arange(T)
    labels = ["x", "y"] if d == 2 else ["x", "y", "z"]

    def _plot_series(ax, series, style, prefix, **kw):
        for i, name in enumerate(labels):
            ax.plot(t, series[:, i], style, label=f"{prefix} {name}", **kw)

    _plot_series(ax_t, traj, "-", "noisy")
    if show_clean is not None:
        _plot_series(ax_t, show_clean, "--", "clean", alpha=0.8)
    if kf_filt is not None:
        _plot_series(ax_t, kf_filt, "-.", "KF filtered")
    if kf_smooth is not None:
        _plot_series(ax_t, kf_smooth, ":", "KF smoothed", lw=2.0)

    ax_t.set_xlabel("t (steps)"); ax_t.set_ylabel("position"); ax_t.set_title("Components")
    ax_t.grid(True); ax_t.legend(loc=legend_loc, ncols=2 if d == 3 else 1)

    plt.tight_layout()
    plt.show()

