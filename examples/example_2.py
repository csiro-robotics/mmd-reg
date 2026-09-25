import jax
from mmd_reg import mmd_reg
from example_utilities import draw_point_clouds, get_point_clouds_mesh
from example_utilities import get_rotation_error, get_translation_error

jax.config.update("jax_default_matmul_precision", "highest")


def main():
    """Run MMD-Reg on a pair of point clouds and visualize the result."""
    # Get source point cloud X, target point cloud Y, and true transformation.
    X, Y, true_R, true_t = get_point_clouds_mesh()

    # Visualize point clouds. The view is interactive.
    draw_point_clouds(X, Y)

    # Get predicted registration solution from MMD-Reg.
    # Take care if timing MMD-Reg because JAX uses asynchronous dispatch.
    # MMD-Reg is faster after JIT compilation for the same input shapes.
    mmd_reg_jit = jax.jit(mmd_reg)
    Ds = [256, 256, 256, 256]
    ls = [1.0, 0.5, 0.25, 0.125]
    Ws = []
    key = jax.random.key(0)
    for D, l in zip(Ds, ls):
        key, W_key = jax.random.split(key, num=2)
        W = jax.random.laplace(W_key, (D, 3)) / l  # Can also use normal.
        Ws.append(W)
    pred_R, pred_t = mmd_reg_jit(Ws, X, Y)

    # Visualize predicted registration. The view is interactive.
    draw_point_clouds(X @ pred_R.T + pred_t, Y)

    # Compute and print registration errors.
    rotation_error = get_rotation_error(pred_R, true_R)
    translation_error = get_translation_error(pred_t, true_t)
    print(f"Rotation error (degrees): {float(rotation_error):.6f}")
    print(f"Translation error (-): {float(translation_error):.6f}")


if __name__ == "__main__":
    main()
