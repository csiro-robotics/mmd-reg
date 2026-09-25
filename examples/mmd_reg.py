import jax
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt


def skew(x):
    """Compute the skew-symmetric cross-product matrix of vector `x`.

    Args:
        x: Three-dimensional vector.

    Returns:
        The skew-symmetric cross-product matrix of vector `x`.
    """
    K = jnp.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])
    return K


def expm_skew(x):
    """Compute the rotation matrix from the matrix exponential of `skew(x)`.

    Args:
        x: Three-dimensional vector.

    Returns:
        The rotation matrix from the matrix exponential of `skew(x)`.
    """
    return jax.scipy.linalg.expm(skew(x))


def transform(params, X):
    """Transform the points in matrix `X` using `params`.

    Args:
        params: Six-dimensional vector that is the concatenation of a rotation
            vector and a translation vector.
        X: Matrix of points for a point cloud.

    Returns:
        The matrix of transformed points.
    """
    rotation_vector = params[:3]
    translation_vector = params[3:]
    return jnp.matmul(X, expm_skew(rotation_vector).T) + translation_vector


def random_feature_map(x, W):
    """Compute the random feature map of vector `x`.

    Args:
        x: Three-dimensional vector.
        W: Matrix of random weights.

    Returns:
        The random feature map of vector `x`.
    """
    Wx = jnp.matmul(W, x).reshape((-1, 1))
    v = jnp.concatenate([jnp.sin(Wx), jnp.cos(Wx)], axis=1).reshape((-1,))
    D = jnp.size(W, axis=0)
    return jnp.sqrt(1.0 / D) * v


def approx_maximum_mean_discrepancy_residual(W, X, Y):
    """Compute the residual values of an approximated maximum mean discrepancy.

    Args:
        W: Matrix of random weights for the random feature map.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The residual values of an approximated maximum mean discrepancy.
    """
    vmap_rfm = jax.vmap(random_feature_map, (0, None), 0)
    X_rfm = vmap_rfm(X, W)  # (N, 2 * D)
    Y_rfm = vmap_rfm(Y, W)  # (M, 2 * D)
    return jnp.mean(X_rfm, axis=0) - jnp.mean(Y_rfm, axis=0)


def inner_objective_residual(inner_params, W, X, Y):
    """Compute the residual values of the inner objective function.

    Args:
        inner_params: Six-dimensional vector that is the concatenation of a
            rotation vector and a translation vector.
        W: Matrix of random weights for the random feature map.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The residual values of the inner objective function.
    """
    X = transform(inner_params, X)
    return approx_maximum_mean_discrepancy_residual(W, X, Y)


def inner_objective_solution(init_inner_params, W, X, Y, **kwargs):
    """Run the Levenberg Marquardt solver on the inner objective function.

    Args:
        init_inner_params: For the LM solver, the initial six-dimensional
            vector that is the concatenation of a rotation vector and a
            translation vector.
        W: Matrix of random weights for the random feature map.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        **kwargs: Additional keyword arguments passed to the LM solver.

    Returns:
        Six-dimensional vector returned by the LM solver.
    """
    inner_solver = LevenbergMarquardt(inner_objective_residual, **kwargs)
    inner_params = inner_solver.run(init_inner_params, W, X, Y).params
    return inner_params


def mmd_reg(Ws, X, Y):
    """Run MMD-Reg to predict the rigid transform from `X` to `Y`.

    Args:
        Ws: List of matrices of random weights for the random feature maps.
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.

    Returns:
        The predicted rotation matrix and translation vector.
    """
    inner_params = jnp.zeros((6,))
    for W in Ws:
        inner_params = inner_objective_solution(
            inner_params,
            W,
            X,
            Y,
            tol=1e-5,
            maxiter=100,
            solver="cholesky",
            materialize_jac=True,
            damping_parameter=1.0,
            verbose=False,
        )
    pred_r = inner_params[:3]
    pred_t = inner_params[3:]
    pred_R = expm_skew(pred_r)
    return pred_R, pred_t
