import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as SciPyRotation
from torch.utils.data import Dataset
from torchvision.transforms import v2


class PointCloudDataset(Dataset):
    """Point cloud dataset.

    Args:
        data_path: The path to a HDF5 file containing the processed data.
        transform: A function that takes a sample and transforms it.
    """

    def __init__(self, data_path, transform=None):
        self.data_dict = self._load_data_to_dict(data_path)
        self.transform = transform

    def _load_data_to_dict(self, data_path):
        """Load a HDF5 file into a Python dictionary to improve performance.

        Args:
            data_path: The path to a HDF5 file containing the processed data.

        Returns:
            A dictionary.
        """
        with h5py.File(data_path, "r") as data_file:
            num_samples = len(data_file)
        data_dict = {}
        for sample_index in range(num_samples):
            sample_string = f"sample_{sample_index:06d}"
            with h5py.File(data_path, "r") as data_file:
                X = np.array(data_file[f"{sample_string}/X"], dtype=np.float32)
                Y = np.array(data_file[f"{sample_string}/Y"], dtype=np.float32)
                R = np.array(data_file[f"{sample_string}/R"], dtype=np.float32)
                t = np.array(data_file[f"{sample_string}/t"], dtype=np.float32)
            data_dict[sample_index] = {"X": X, "Y": Y, "R": R, "t": t}
        return data_dict

    def __len__(self):
        """Return the number of samples."""
        return len(self.data_dict)

    def __getitem__(self, index):
        """Return a sample with index ``index``.

        Args:
            index: The index of the sample.

        Returns:
            A tuple (X, Y, R, t) where X is the matrix of points for the source
            point cloud, Y is the matrix of points for the target point cloud,
            R is the true rotation matrix and t is the true translation vector.
        """
        X = self.data_dict[index]["X"]  # Has shape (M, 3)
        Y = self.data_dict[index]["Y"]  # Has shape (N, 3)
        R = self.data_dict[index]["R"]  # Has shape (3, 3)
        t = self.data_dict[index]["t"]  # Has shape (3,)
        if self.transform is not None:
            X, Y, R, t = self.transform(X, Y, R, t)
        return X, Y, R, t


class CenterSource:
    """Center the source point cloud."""

    def __call__(self, X, Y, R, t):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y = R @ (x - c) + (t + R @ c)``.
        c = np.mean(X, axis=0)
        X = X - c
        t = t + np.matmul(R, c)
        return X, Y, R, t


class CenterTarget:
    """Center the target point cloud."""

    def __call__(self, X, Y, R, t):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y - c = R @ x + (t - c)``.
        c = np.mean(Y, axis=0)
        Y = Y - c
        t = t - c
        return X, Y, R, t


def randomly_choose_indices(indices, num_choices):
    """Randomly choose a number of indices, keeping duplicates to a minimum.

    Args:
        indices: A NumPy array of indices to choose from.
        num_choices: The number of indices to randomly choose.

    Returns:
        A NumPy array of indices.
    """
    ceil = int(np.ceil(num_choices / len(indices)))
    permutations = [np.random.permutation(indices) for i in range(ceil)]
    permutations = np.concatenate(permutations)
    choice = permutations[:num_choices]
    np.random.shuffle(choice)
    return choice


class RandomSubsample:
    """Randomly subsample both point clouds independently."""

    def __init__(self, number_of_points=1024):
        self.number_of_points = number_of_points

    def __call__(self, X, Y, R, t):
        X_indices = np.arange(len(X))
        Y_indices = np.arange(len(Y))
        X_choice = randomly_choose_indices(X_indices, self.number_of_points)
        Y_choice = randomly_choose_indices(Y_indices, self.number_of_points)
        X = X[X_choice]
        Y = Y[Y_choice]
        return X, Y, R, t


class RandomCrop:
    """Randomly crop two fully overlapping point clouds."""

    def __init__(self, number_of_points_left=717):
        self.number_of_points_left = number_of_points_left

    def __call__(self, X, Y, R, t):
        X = X @ R.T + t
        # Generate normal vectors for two random hyperplanes.
        normal_vector_1 = np.random.randn(3)
        normal_vector_1 = normal_vector_1 / np.linalg.norm(normal_vector_1)
        normal_vector_2 = np.random.randn(3)
        normal_vector_2 = normal_vector_2 / np.linalg.norm(normal_vector_2)
        # Crop X based on the first hyperplane.
        X_dot_products = np.linalg.matmul(X, normal_vector_1)
        X_left = np.argsort(X_dot_products)[: self.number_of_points_left]
        X = X[X_left]
        # Crop Y based on the second hyperplane.
        Y_dot_products = np.linalg.matmul(Y, normal_vector_2)
        Y_left = np.argsort(Y_dot_products)[: self.number_of_points_left]
        Y = Y[Y_left]
        X = (X - t) @ R
        return X, Y, R, t


class RandomJitter:
    """Add random noise to both point clouds independently."""

    def __init__(self, scale=0.01, min_clip=-0.05, max_clip=0.05):
        self.scale = scale
        self.min_clip = min_clip
        self.max_clip = max_clip

    def __call__(self, X, Y, R, t):
        X_noise = np.random.normal(scale=self.scale, size=X.shape)
        X_noise = np.clip(X_noise, self.min_clip, self.max_clip)
        X = X + X_noise
        Y_noise = np.random.normal(scale=self.scale, size=X.shape)
        Y_noise = np.clip(Y_noise, self.min_clip, self.max_clip)
        Y = Y + Y_noise
        return X, Y, R, t


def get_random_rotation_matrix(min_degrees, max_degrees):
    """
    Return a rotation matrix using randomly chosen Euler angles.

    Args:
        min_degrees: The minimum value for each Euler angle in degrees.
        max_degrees: The maximum value for each Euler angle in degrees.

    Returns:
        A rotation matrix.
    """
    euler_angles = np.random.uniform(min_degrees, max_degrees, size=3)
    # Same as using anglez, angley, anglex = np.deg2rad(euler_angles) in DCP.
    R = SciPyRotation.from_euler("zyx", euler_angles, degrees=True).as_matrix()
    return R


class RandomRotateSource:
    """Randomly rotate the source point cloud."""

    def __init__(self, min_degrees=0.0, max_degrees=45.0):
        self.min_degrees = min_degrees
        self.max_degrees = max_degrees

    def __call__(self, X, Y, R, t):
        # If ``x, y`` are three-dimensional vectors,
        # ``R, Q`` are rotation matrices and ``y = R @ x + t``,
        # then ``y = (R @ Q.T) @ (Q @ x) + t``.
        Q = get_random_rotation_matrix(self.min_degrees, self.max_degrees)
        X = np.matmul(X, Q.T)
        R = np.matmul(R, Q.T)
        return X, Y, R, t


class RandomRotateTarget:
    """Randomly rotate the target point cloud."""

    def __init__(self, min_degrees=0.0, max_degrees=45.0):
        self.min_degrees = min_degrees
        self.max_degrees = max_degrees

    def __call__(self, X, Y, R, t):
        # If ``x, y`` are three-dimensional vectors,
        # ``R, Q`` are rotation matrices and ``y = R @ x + t``,
        # then ``Q @ y = (Q @ R) @ x + (Q @ t)``.
        Q = get_random_rotation_matrix(self.min_degrees, self.max_degrees)
        Y = np.matmul(Y, Q.T)
        R = np.matmul(Q, R)
        t = np.matmul(Q, t)
        return X, Y, R, t


def get_random_translation_vector(lower_bound, upper_bound):
    """
    Return a random translation vector, where each axis is uniformly sampled.

    Args:
        lower_bound: The minimum value for each axis of the translation vector.
        upper_bound: The maximum value for each axis of the translation vector.

    Returns:
        A translation vector.
    """
    t = np.random.uniform(lower_bound, upper_bound, 3)
    return t


class RandomTranslateSource:
    """Randomly translate the source point cloud."""

    def __init__(self, lower_bound=-0.5, upper_bound=0.5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, X, Y, R, t):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y = R @ (x + c) + (t - R @ c)``.
        c = get_random_translation_vector(self.lower_bound, self.upper_bound)
        X = X + c
        t = t - np.matmul(R, c)
        return X, Y, R, t


class RandomTranslateTarget:
    """Randomly translate the target point cloud."""

    def __init__(self, lower_bound=-0.5, upper_bound=0.5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, X, Y, R, t):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y + c = R @ x + (t + c)``.
        c = get_random_translation_vector(self.lower_bound, self.upper_bound)
        Y = Y + c
        t = t + c
        return X, Y, R, t


class RandomShuffle:
    """Randomly reorder the points in both point clouds independently."""

    def __call__(self, X, Y, R, t):
        X_permutation = np.random.permutation(len(X))
        Y_permutation = np.random.permutation(len(Y))
        X = X[X_permutation]
        Y = Y[Y_permutation]
        return X, Y, R, t


class RandomSwap:
    """Randomly swap the source point cloud with the target point cloud."""

    def __call__(self, X, Y, R, t):
        # If ``x, y`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``x = (R.T) @ y + (- R.T @ t)``.
        if np.random.random() > 0.5:
            X, Y = Y, X
            t = -1.0 * np.matmul(R.T, t)
            R = R.T
        return X, Y, R, t


def draw_point_clouds(X, Y):
    pcd_X = o3d.geometry.PointCloud()
    pcd_X.points = o3d.utility.Vector3dVector(X)
    pcd_X.paint_uniform_color((1.0, 0.0, 0.0))
    pcd_Y = o3d.geometry.PointCloud()
    pcd_Y.points = o3d.utility.Vector3dVector(Y)
    o3d.visualization.draw(
        [pcd_X, pcd_Y], point_size=4, show_skybox=False, raw_mode=True
    )


if __name__ == "__main__":
    train_data_path = "Datasets/Processed/modelnet40_clean_train.hdf5"
    val_data_path = "Datasets/Processed/modelnet40_clean_val.hdf5"
    test_data_path = "Datasets/Processed/modelnet40_clean_test.hdf5"
    train_tr = v2.Compose([RandomRotateSource(), RandomTranslateSource()])
    train_ds = PointCloudDataset(train_data_path, transform=train_tr)
    val_ds = PointCloudDataset(val_data_path, transform=None)
    test_ds = PointCloudDataset(test_data_path, transform=None)
    X, Y, R, t = test_ds[0]
    draw_point_clouds(X, Y)  # Draw source point cloud and target point cloud
    draw_point_clouds(X @ R.T + t, Y)  # Draw aligned point clouds
