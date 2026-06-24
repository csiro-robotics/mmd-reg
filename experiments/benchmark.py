import argparse
import jax
import jax.numpy as jnp
import json
import numpy as np
import os
import time
from benchmark_utilities import load_data_to_list, eval_mmd_reg, eval_icp_cpu
from benchmark_utilities import eval_icp_gpu, eval_multi_scale_icp_gpu
from benchmark_utilities import eval_gicp, eval_cpd, eval_filterreg
from benchmark_utilities import eval_gmmreg, eval_svr
from benchmark_utilities import get_rotation_error, get_translation_error
from tqdm import trange


def get_args():
    """Parse command-line arguments for the registration benchmark.

    This function defines input and output paths, available algorithms, and
    key method parameters. It also validates that mmd_reg_Ds and mmd_reg_ls have
    matching lengths.

    Returns:
        An argparse.Namespace containing the parsed arguments.
    """
    algs = [
        "MMD-Reg",
        "ICP-Point-To-Point-CPU",  # Parameters set for PCPNet dataset.
        "ICP-Point-To-Point-GPU",  # Parameters set for PCPNet dataset.
        "ICP-Point-To-Plane-CPU",  # Parameters set for PCPNet dataset.
        "ICP-Point-To-Plane-GPU",  # Parameters set for PCPNet dataset.
        "GICP",  # Parameters set for PCPNet dataset.
        "CPD",  # Parameters set for PCPNet dataset.
        "FilterReg",  # Parameters set for PCPNet dataset.
        "GMMReg",  # Parameters set for PCPNet dataset.
        "SVR",  # Parameters set for PCPNet dataset.
        "Multi-Scale-ICP-Point-To-Point-GPU",  # Parameters set for Real Outdoor
        "Multi-Scale-ICP-Point-To-Plane-GPU",  # Parameters set for Real Outdoor
    ]
    data_path = os.path.join("Datasets", "pcpnet_test_05k.hdf5")
    save_path = os.path.join("Results", "benchmark.json")
    Ds = [128]
    ls = [1.0]
    dist = "Laplace"
    dists = ["Gaussian", "Laplace"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="SVR", choices=algs)
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--save_path", type=str, default=save_path)
    parser.add_argument("--mmd_reg_Ds", nargs="+", type=int, default=Ds)
    parser.add_argument("--mmd_reg_ls", nargs="+", type=float, default=ls)
    parser.add_argument("--mmd_reg_dist", type=str, default=dist, choices=dists)
    args = parser.parse_args()
    if len(args.mmd_reg_Ds) != len(args.mmd_reg_ls):
        raise ValueError("len(args.mmd_reg_Ds) != len(args.mmd_reg_ls)")
    return args


def main():
    """Run the benchmark for the selected registration algorithm.

    This function loads the dataset, runs the chosen algorithm on each sample,
    computes rotation and translation errors, measures run time, and saves
    aggregate results to a JSON file keyed by dataset path.
    """
    args = get_args()
    data_list = load_data_to_list(args.data_path)
    num_samples = len(data_list)
    all_errors_R = np.zeros(num_samples, dtype=np.float32)
    all_errors_t = np.zeros(num_samples, dtype=np.float32)
    all_run_time = np.zeros(num_samples, dtype=np.float32)

    if args.algorithm == "MMD-Reg":
        Ws = []
        Ds = args.mmd_reg_Ds
        ls = args.mmd_reg_ls
        keys = jax.random.split(jax.random.key(0), num=len(Ds))
        dist = args.mmd_reg_dist
        dist_fn = jax.random.laplace if dist == "Laplace" else jax.random.normal
        for key, D, l in zip(keys, Ds, ls):
            W = dist_fn(key, (D, 3)) / l
            W = W.block_until_ready()
            Ws.append(W)

    for sample_index in trange(num_samples, mininterval=10.0):
        X, Y, R, t = data_list[sample_index]
        start_time = time.time()
        if args.algorithm == "MMD-Reg":
            X, Y = jnp.asarray(X), jnp.asarray(Y)
            pred_R, pred_t = eval_mmd_reg(Ws, X, Y)
        elif args.algorithm == "ICP-Point-To-Point-CPU":
            pred_R, pred_t = eval_icp_cpu(X, Y)
        elif args.algorithm == "ICP-Point-To-Point-GPU":
            pred_R, pred_t = eval_icp_gpu(X, Y)
        elif args.algorithm == "ICP-Point-To-Plane-CPU":
            pred_R, pred_t = eval_icp_cpu(X, Y, use_point_to_plane=True)
        elif args.algorithm == "ICP-Point-To-Plane-GPU":
            pred_R, pred_t = eval_icp_gpu(X, Y, use_point_to_plane=True)
        elif args.algorithm == "GICP":
            pred_R, pred_t = eval_gicp(X, Y)
        elif args.algorithm == "CPD":
            pred_R, pred_t = eval_cpd(X, Y)
        elif args.algorithm == "FilterReg":
            pred_R, pred_t = eval_filterreg(X, Y)
        elif args.algorithm == "GMMReg":
            pred_R, pred_t = eval_gmmreg(X, Y)
        elif args.algorithm == "SVR":
            pred_R, pred_t = eval_svr(X, Y)
        elif args.algorithm == "Multi-Scale-ICP-Point-To-Point-GPU":
            pred_R, pred_t = eval_multi_scale_icp_gpu(X, Y)
        else:
            pred_R, pred_t = eval_multi_scale_icp_gpu(
                X, Y, use_point_to_plane=True
            )
        run_time = time.time() - start_time
        all_errors_R[sample_index] = get_rotation_error(pred_R, R)
        all_errors_t[sample_index] = get_translation_error(pred_t, t)
        all_run_time[sample_index] = run_time

    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as file:
            save_dict = json.load(file)
    else:
        save_dict = {}

    if args.data_path not in save_dict:
        save_dict[args.data_path] = {}

    if args.algorithm == "MMD-Reg":
        algorithm_key = [
            args.algorithm,
            args.mmd_reg_dist,
            "-".join(map(str, args.mmd_reg_Ds)),
            "-".join(map(str, args.mmd_reg_ls)),
        ]
        algorithm_key = "-".join(algorithm_key)  # E.g., MMD-Reg-Laplace-128-1.0
    else:
        algorithm_key = args.algorithm

    algorithm_dict = {
        "Average Rotation Error": np.mean(all_errors_R).item(),
        "Average Translation Error": np.mean(all_errors_t).item(),
        "Average Run Time": np.mean(all_run_time).item(),
    }
    save_dict[args.data_path][algorithm_key] = algorithm_dict

    with open(args.save_path, "w") as file:
        json.dump(save_dict, file, indent=4)

    print(algorithm_dict)


if __name__ == "__main__":
    main()
