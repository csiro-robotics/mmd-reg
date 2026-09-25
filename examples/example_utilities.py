import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
from jax.scipy.spatial.transform import Rotation


def draw_point_clouds(X, Y, point_size=1):
    """Draw the point clouds in matrices `X` and `Y`.

    Args:
        X: Matrix of points for the source point cloud.
        Y: Matrix of points for the target point cloud.
        point_size: Size of the points in the visualization.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    pcd_X = o3d.geometry.PointCloud()
    pcd_Y = o3d.geometry.PointCloud()
    pcd_X.points = o3d.utility.Vector3dVector(X)
    pcd_Y.points = o3d.utility.Vector3dVector(Y)
    pcd_X.paint_uniform_color((1.0, 0.0, 0.0))
    pcd_Y.paint_uniform_color((0.0, 0.0, 0.0))
    o3d.visualization.draw(
        [pcd_X, pcd_Y], point_size=point_size, show_skybox=False, raw_mode=True
    )


def get_point_clouds_mesh(num_points=100000, degrees=45.0, translation=0.5):
    """Generate a source point cloud and a target point cloud from a mesh.

    Args:
        num_points: Number of points in each point cloud.
        degrees: Magnitude of the ground truth rotation in degrees.
        translation: Magnitude of the ground truth translation vector.

    Returns:
        The matrices of points for the source and target point clouds, and
        the ground truth rotation matrix and translation vector.
    """
    # Generate ground truth rotation matrix and translation vector.
    key = jax.random.key(0)
    key, key_r, key_t = jax.random.split(key, num=3)
    true_r = jax.random.normal(key_r, (3,))
    true_r = true_r / jnp.linalg.norm(true_r)
    true_r = true_r * jnp.deg2rad(degrees)  # True rotation vector.
    true_R = Rotation.from_rotvec(true_r).as_matrix()  # True rotation matrix.
    true_t = jax.random.normal(key_t, (3,))
    true_t = true_t / jnp.linalg.norm(true_t)
    true_t = true_t * translation  # True translation vector.

    # Generate source point cloud and target point cloud.
    mesh = o3d.io.read_triangle_mesh(o3d.data.DamagedHelmetModel().path)
    XY = mesh.sample_points_uniformly(2 * num_points).points
    XY = jnp.array(XY)
    XY = XY / jnp.max(jnp.linalg.norm(XY, axis=1))
    X = XY[:num_points, :]
    Y = XY[num_points:, :]
    Y = Y @ true_R.T + true_t
    return X, Y, true_R, true_t


def get_point_clouds_real(num_points=100000, voxel_size=0.15):
    """Load a real source point cloud and a real target point cloud.

    Args:
        num_points: Number of points randomly sampled from each
            voxel-downsampled point cloud.
        voxel_size: Voxel size used to downsample the point clouds.

    Returns:
        The matrices of points for the source and target point clouds.
    """
    np.random.seed(0)
    X = np.fromfile("real_source.bin", dtype=np.float32).reshape(-1, 6)[:, :3]
    Y = np.fromfile("real_target.bin", dtype=np.float32).reshape(-1, 6)[:, :3]
    pcd_X = o3d.geometry.PointCloud()
    pcd_Y = o3d.geometry.PointCloud()
    pcd_X.points = o3d.utility.Vector3dVector(X)
    pcd_Y.points = o3d.utility.Vector3dVector(Y)
    pcd_X = pcd_X.voxel_down_sample(voxel_size=voxel_size)
    pcd_Y = pcd_Y.voxel_down_sample(voxel_size=voxel_size)
    X = np.asarray(pcd_X.points)
    Y = np.asarray(pcd_Y.points)
    assert len(X) >= num_points
    assert len(Y) >= num_points
    X = np.random.permutation(X)[:num_points]
    Y = np.random.permutation(Y)[:num_points]
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)
    return X, Y


def get_rotation_error(pred_rotation_matrix, true_rotation_matrix):
    """Compute the angular error in degrees between two rotation matrices.

    Args:
        pred_rotation_matrix: Predicted rotation matrix.
        true_rotation_matrix: Ground truth rotation matrix.

    Returns:
        The angular error in degrees.
    """
    relative_rotation_matrix = pred_rotation_matrix @ true_rotation_matrix.T
    trace = jnp.trace(relative_rotation_matrix)
    error = jnp.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    error = jnp.arccos(error)
    error = jnp.rad2deg(error)
    return error


def get_translation_error(pred_translation_vector, true_translation_vector):
    """Compute the L2 (Euclidean) distance between two translation vectors.

    Args:
        pred_translation_vector: Predicted translation vector.
        true_translation_vector: Ground truth translation vector.

    Returns:
        The L2 (Euclidean) distance.
    """
    translation_difference = pred_translation_vector - true_translation_vector
    error = jnp.linalg.norm(translation_difference)
    return error
