import json
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "high-vis"])
plt.rcParams.update(
    {
        "figure.figsize": (3.25, 2.8),
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True,
    }
)

all_alg_keys_cpu = [
    "MMD-Reg-Gaussian-32-0.75",
    "MMD-Reg-Laplace-32-0.75",
    "ICP-Point-To-Point-CPU",
    "ICP-Point-To-Plane-CPU",
    "GICP",
    "FilterReg",
    "GMMReg",
    "SVR",
    "CPD",
]

all_alg_keys_gpu = [
    "MMD-Reg-Gaussian-32-0.75",
    "MMD-Reg-Laplace-32-0.75",
    "ICP-Point-To-Point-GPU",
    "ICP-Point-To-Plane-GPU",
]

all_alg_names = [
    "MMD-Reg G (Ours)",
    "MMD-Reg L (Ours)",
    "ICP Pt2Pt",
    "ICP Pt2Pl",
    "GICP",
    "FilterReg",
    "GMMReg",
    "SVR",
    "CPD",
]

all_colors = [
    "#000000",
    "#e6091c",
    "#8936df",
    "#8b4513",
    "#26eb47",
    "#1f77b4",
    "#ff7f0e",
    "#ff00f2",
    "#25d7fd",
]

all_linestyles = [
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
]

all_markers = [
    None,
    None,
    None,
    None,
    ".",
    ".",
    ".",
    ".",
    "x",
]


def plot_pcpnet_cpu():
    """Plot CPU benchmark results for the processed PCPNet datasets.

    This function loads the CPU benchmark results and plots the average
    runtime, rotation error, and translation error against the number of
    points. The plots are saved as PDF files in "results/figures".
    """
    alg_keys = all_alg_keys_cpu
    alg_names = all_alg_names
    colors = all_colors
    linestyles = all_linestyles
    markers = all_markers

    with open("results/pcpnet_cpu.json") as f:
        data = json.load(f)

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    possible_number_of_points = [1, 2, 4, 8, 10, 20, 30, 40, 50]  # thousands

    for k, n, c, l, m in zip(alg_keys, alg_names, colors, linestyles, markers):
        run_times = []
        ro_errors = []
        tr_errors = []
        num_point = []
        for p in possible_number_of_points:
            data_key = f"datasets/processed/pcpnet_time_{p:02d}k.hdf5"
            if k in data[data_key]:
                run_time = data[data_key][k]["Average Run Time"]
                ro_error = data[data_key][k]["Average Rotation Error"]
                tr_error = data[data_key][k]["Average Translation Error"]
                run_times.append(run_time)
                ro_errors.append(ro_error)
                tr_errors.append(tr_error)
                num_point.append(p)
        ax0.plot(num_point, run_times, label=n, c=c, ls=l, marker=m)
        ax1.plot(num_point, ro_errors, label=n, c=c, ls=l, marker=m)
        ax2.plot(num_point, tr_errors, label=n, c=c, ls=l, marker=m)

    ax0.legend()
    xlabel = "Number of Points (Thousands)"
    ylabel = "Average CPU Runtime (Seconds)"
    save_path = "results/figures/plot_pcpnet_cpu_time.pdf"
    ax0.set(xlabel=xlabel, ylabel=ylabel)
    fig0.savefig(save_path, bbox_inches="tight", pad_inches=0.01)

    ylabel = "Average Rotation Error (Degrees)"
    save_path = "results/figures/plot_pcpnet_cpu_ro_error.pdf"
    ax1.set(xlabel=xlabel, ylabel=ylabel)
    fig1.savefig(save_path, bbox_inches="tight", pad_inches=0.01)

    ylabel = "Average Translation Error (-)"
    save_path = "results/figures/plot_pcpnet_cpu_tr_error.pdf"
    ax2.set(xlabel=xlabel, ylabel=ylabel)
    fig2.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_pcpnet_gpu():
    """Plot GPU benchmark results for the processed PCPNet datasets.

    This function loads the GPU benchmark results and plots the average
    runtime against the number of points for the selected registration
    methods. The plot is saved as a PDF file in "results/figures".
    """
    alg_keys = all_alg_keys_gpu
    alg_names = all_alg_names[:4]
    colors = all_colors[:4]
    linestyles = all_linestyles[:4]
    markers = all_markers[:4]

    with open("results/pcpnet_gpu.json") as f:
        data = json.load(f)

    fig0, ax0 = plt.subplots()

    number_of_points = [1, 10, 20, 30, 40, 50]  # thousands

    for k, n, c, l, m in zip(alg_keys, alg_names, colors, linestyles, markers):
        run_times = []
        for p in number_of_points:
            data_key = f"datasets/processed/pcpnet_time_{p:02d}k.hdf5"
            run_time = data[data_key][k]["Average Run Time"]
            run_times.append(run_time)
        ax0.plot(number_of_points, run_times, label=n, c=c, ls=l, marker=m)

    xlabel = "Number of Points (Thousands)"
    ylabel = "Average GPU Runtime (Seconds)"
    save_path = "results/figures/plot_pcpnet_gpu_time.pdf"
    ax0.set(xlabel=xlabel, ylabel=ylabel)
    fig0.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_pcpnet_outliers():
    """Plot registration errors for processed PCPNet datasets with outliers.

    This function loads the CPU benchmark results and plots the average
    rotation and translation errors against the percentage of outliers. The
    plots and their shared legend are saved as PDF files in "results/figures".
    """
    alg_keys = all_alg_keys_cpu[:8]
    alg_names = all_alg_names[:8]
    colors = all_colors[:8]
    linestyles = all_linestyles[:8]
    markers = all_markers[:8]

    with open("results/pcpnet_cpu.json") as f:
        data = json.load(f)

    fig0, ax0 = plt.subplots(figsize=(0.5, 0.5))
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    outliers = [0, 1, 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for k, n, c, l, m in zip(alg_keys, alg_names, colors, linestyles, markers):
        ro_errors = []
        tr_errors = []
        for j in outliers:
            data_key = f"datasets/processed/pcpnet_outliers_{j:03d}.hdf5"
            ro_error = data[data_key][k]["Average Rotation Error"]
            tr_error = data[data_key][k]["Average Translation Error"]
            ro_errors.append(ro_error)
            tr_errors.append(tr_error)
        ax1.plot(outliers, ro_errors, label=n, c=c, ls=l, marker=m)
        ax2.plot(outliers, tr_errors, label=n, c=c, ls=l, marker=m)

    handles, labels = ax1.get_legend_handles_labels()
    save_path = "results/figures/plot_outliers_legend.pdf"
    ax0.clear()
    ax0.axis("off")
    ax0.legend(handles, labels, loc="center", frameon=False, ncol=4)
    fig0.savefig(save_path, bbox_inches="tight", pad_inches=0.01)

    xlabel = "Outliers (Percentage)"
    ylabel = "Average Rotation Error (Degrees)"
    save_path = "results/figures/plot_outliers_ro_error.pdf"
    ax1.set(xlabel=xlabel, ylabel=ylabel)
    fig1.savefig(save_path, bbox_inches="tight", pad_inches=0.01)

    ylabel = "Average Translation Error (-)"
    save_path = "results/figures/plot_outliers_tr_error.pdf"
    ax2.set(xlabel=xlabel, ylabel=ylabel)
    fig2.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def main():
    """Generate and save the processed PCPNet benchmark figures.

    This function generates the CPU runtime and error plots, the GPU runtime
    plot, and the outlier error plots. All figures are saved as PDF files in
    "results/figures".
    """
    plot_pcpnet_cpu()
    plot_pcpnet_gpu()
    plot_pcpnet_outliers()


if __name__ == "__main__":
    main()
