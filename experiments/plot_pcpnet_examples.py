import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
from benchmark_utilities import eval_mmd_reg, eval_icp_cpu, eval_gicp
from benchmark_utilities import eval_filterreg, eval_gmmreg, eval_svr


def plot_point_clouds(X, Y, save_path):
    """Plot two point clouds and save the resulting figure.

    The point clouds are displayed in a three-dimensional orthographic view.
    The resulting figure is saved to the location specified by `save_path`.

    Args:
        X: Matrix of points for the first point cloud.
        Y: Matrix of points for the second point cloud.
        save_path: Path at which to save the resulting figure.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=0.2)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=0.2)
    ax.view_init(elev=0, azim=0)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    P = np.vstack([X, Y])
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    center = (mins + maxs) / 2
    radius = 0.65 * (maxs - mins).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()


def plot_pcpnet_examples(data_path, sample_index, save_file_prefix):
    """Plot registration results for a sample from a processed PCPNet dataset.

    This function loads a source and target point cloud, runs several rigid
    registration methods, and saves a plot of the alignment produced by each
    method. It also saves a plot of the point clouds before registration. All
    plots are saved as PNG files in "results/figures".

    Args:
        data_path: Path to an HDF5 file containing the processed PCPNet dataset.
        sample_index: Index of the sample to load and register.
        save_file_prefix: Prefix to use for the saved figure filenames.
    """
    sample_string = f"sample_{sample_index:06d}"
    with h5py.File(data_path, "r") as data_file:
        X = np.array(data_file[f"{sample_string}/X"], dtype=np.float32)
        Y = np.array(data_file[f"{sample_string}/Y"], dtype=np.float32)
        R = np.array(data_file[f"{sample_string}/R"], dtype=np.float32)
        t = np.array(data_file[f"{sample_string}/t"], dtype=np.float32)

    save_path = f"results/figures/{save_file_prefix}_source_target.png"
    plot_point_clouds(X, Y, save_path)

    save_path = f"results/figures/{save_file_prefix}_icp_pt2pt.png"
    pred_R, pred_t = eval_icp_cpu(X, Y)
    plot_point_clouds(X @ pred_R.T + pred_t, Y, save_path)

    save_path = f"results/figures/{save_file_prefix}_icp_pt2pl.png"
    pred_R, pred_t = eval_icp_cpu(X, Y, use_point_to_plane=True)
    plot_point_clouds(X @ pred_R.T + pred_t, Y, save_path)

    save_path = f"results/figures/{save_file_prefix}_gicp.png"
    pred_R, pred_t = eval_gicp(X, Y)
    plot_point_clouds(X @ pred_R.T + pred_t, Y, save_path)

    save_path = f"results/figures/{save_file_prefix}_filterreg.png"
    pred_R, pred_t = eval_filterreg(X, Y)
    plot_point_clouds(X @ pred_R.T + pred_t, Y, save_path)

    save_path = f"results/figures/{save_file_prefix}_gmmreg.png"
    pred_R, pred_t = eval_gmmreg(X, Y)
    plot_point_clouds(X @ pred_R.T + pred_t, Y, save_path)

    save_path = f"results/figures/{save_file_prefix}_svr.png"
    pred_R, pred_t = eval_svr(X, Y)
    plot_point_clouds(X @ pred_R.T + pred_t, Y, save_path)

    save_path = f"results/figures/{save_file_prefix}_mmd.png"
    key = jax.random.key(0)
    W = jax.random.laplace(key, (32, 3)) / 0.75
    pred_R, pred_t = eval_mmd_reg([W], X, Y)
    plot_point_clouds(X @ pred_R.T + pred_t, Y, save_path)


def main():
    """Generate registration figures for selected processed PCPNet datasets.

    This function selects examples from the gradient, high-noise, and outlier
    PCPNet datasets. For each example, it plots the point clouds before
    registration and the alignment produced by each registration method. All
    figures are saved as PNG files in "results/figures".
    """
    data_path = "datasets/processed/pcpnet_gradient.hdf5"
    sample_index = 8
    save_file_prefix = "gradient"
    plot_pcpnet_examples(data_path, sample_index, save_file_prefix)

    data_path = "datasets/processed/pcpnet_high_noise.hdf5"
    sample_index = 3
    save_file_prefix = "noise"
    plot_pcpnet_examples(data_path, sample_index, save_file_prefix)

    data_path = "datasets/processed/pcpnet_outliers_020.hdf5"
    sample_index = 6
    save_file_prefix = "outliers"
    plot_pcpnet_examples(data_path, sample_index, save_file_prefix)


if __name__ == "__main__":
    main()
