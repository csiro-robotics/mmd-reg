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
                X = np.array(data_file[f"{sample_string}/X"])
                Y = np.array(data_file[f"{sample_string}/Y"])
                R = np.array(data_file[f"{sample_string}/R"])
                t = np.array(data_file[f"{sample_string}/t"])
                Xo = np.array(data_file[f"{sample_string}/Xo"])
                Yo = np.array(data_file[f"{sample_string}/Yo"])
            data_dict[sample_index] = {
                "X": X.astype(np.float32),
                "Y": Y.astype(np.float32),
                "R": R.astype(np.float32),
                "t": t.astype(np.float32),
                "Xo": Xo.astype(np.bool),
                "Yo": Yo.astype(np.bool),
            }
        return data_dict

    def __len__(self):
        """Return the number of samples."""
        return len(self.data_dict)

    def __getitem__(self, index):
        """Return a sample with index ``index``.

        Args:
            index: The index of the sample.

        Returns:
            A tuple (X, Y, R, t, Xo, Yo) where X is the matrix of points for
            the source point cloud, Y is the matrix of points for the target
            point cloud, R is the true rotation matrix, t is the true
            translation vector, Xo is the overlap mask for X and Yo is the
            overlap mask for Y.
        """
        X = self.data_dict[index]["X"]  # Has shape (M, 3)
        Y = self.data_dict[index]["Y"]  # Has shape (N, 3)
        R = self.data_dict[index]["R"]  # Has shape (3, 3)
        t = self.data_dict[index]["t"]  # Has shape (3,)
        Xo = self.data_dict[index]["Xo"]  # Has shape (M,)
        Yo = self.data_dict[index]["Yo"]  # Has shape (N,)
        if self.transform is not None:
            X, Y, R, t, Xo, Yo = self.transform(X, Y, R, t, Xo, Yo)
        return X, Y, R, t, Xo, Yo


class CenterSource:
    """Center the source point cloud."""

    def __call__(self, X, Y, R, t, Xo, Yo):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y = R @ (x - c) + (t + R @ c)``.
        c = np.mean(X, axis=0)
        X = X - c
        t = t + np.matmul(R, c)
        return X, Y, R, t, Xo, Yo


class CenterTarget:
    """Center the target point cloud."""

    def __call__(self, X, Y, R, t, Xo, Yo):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y - c = R @ x + (t - c)``.
        c = np.mean(Y, axis=0)
        Y = Y - c
        t = t - c
        return X, Y, R, t, Xo, Yo


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

    def __call__(self, X, Y, R, t, Xo, Yo):
        X_indices = np.arange(len(X))
        Y_indices = np.arange(len(Y))
        X_choice = randomly_choose_indices(X_indices, self.number_of_points)
        Y_choice = randomly_choose_indices(Y_indices, self.number_of_points)
        X = X[X_choice]
        Y = Y[Y_choice]
        Xo = Xo[X_choice]
        Yo = Yo[Y_choice]
        return X, Y, R, t, Xo, Yo


class RandomSubsampleAndOverlap:
    """Randomly subsample both point clouds independently to get new overlap."""

    def __init__(self, number_of_points=1024):
        self.number_of_points = number_of_points

    def _sample_indices(self, overlap_mask):
        p = np.random.uniform(0.1, 0.9)
        k_true = int(self.number_of_points * p)
        k_false = self.number_of_points - k_true
        true_indices = np.nonzero(overlap_mask)[0]
        false_indices = np.nonzero(~overlap_mask)[0]
        selected_true = randomly_choose_indices(true_indices, k_true)
        selected_false = randomly_choose_indices(false_indices, k_false)
        selected_indices = np.concatenate([selected_true, selected_false])
        np.random.shuffle(selected_indices)
        return selected_indices

    def __call__(self, X, Y, R, t, Xo, Yo):
        assert np.any(Xo) and np.any(Yo), "Must be partially overlapping."
        assert ~np.all(Xo) and ~np.all(Yo), "Must not be fully overlapping."
        X_selected_indices = self._sample_indices(Xo)
        Y_selected_indices = self._sample_indices(Yo)
        X = X[X_selected_indices]
        Y = Y[Y_selected_indices]
        Xo = Xo[X_selected_indices]
        Yo = Yo[Y_selected_indices]
        return X, Y, R, t, Xo, Yo


class RandomCrop:
    """Randomly crop two fully overlapping point clouds."""

    def __init__(self, number_of_points_left=717):
        self.number_of_points_left = number_of_points_left

    def __call__(self, X, Y, R, t, Xo, Yo):
        assert np.all(Xo) and np.all(Yo), "Must be fully overlapping."
        X = X @ R.T + t
        # Generate normal vectors for two random hyperplanes.
        normal_vector_1 = np.random.randn(3)
        normal_vector_1 = normal_vector_1 / np.linalg.norm(normal_vector_1)
        normal_vector_2 = np.random.randn(3)
        normal_vector_2 = normal_vector_2 / np.linalg.norm(normal_vector_2)
        # Compute the points in X that lie below both hyperplanes, these points
        # are in the overlapping region. Then, based on the first hyperplane,
        # crop X and its overlap mask.
        X_dot_products_1 = np.linalg.matmul(X, normal_vector_1)
        X_dot_products_2 = np.linalg.matmul(X, normal_vector_2)
        X_left_1 = np.argsort(X_dot_products_1)[: self.number_of_points_left]
        X_left_2 = np.argsort(X_dot_products_2)[: self.number_of_points_left]
        Xo_new = np.zeros(len(X))
        Xo_new[np.intersect1d(X_left_1, X_left_2)] = 1.0
        X = X[X_left_1]
        Xo_new = Xo_new[X_left_1].astype(np.bool)
        # Compute the points in Y that lie below both hyperplanes, these points
        # are in the overlapping region. Then, based on the second hyperplane,
        # crop Y and its overlap mask.
        Y_dot_products_1 = np.linalg.matmul(Y, normal_vector_1)
        Y_dot_products_2 = np.linalg.matmul(Y, normal_vector_2)
        Y_left_1 = np.argsort(Y_dot_products_1)[: self.number_of_points_left]
        Y_left_2 = np.argsort(Y_dot_products_2)[: self.number_of_points_left]
        Yo_new = np.zeros(len(Y))
        Yo_new[np.intersect1d(Y_left_1, Y_left_2)] = 1.0
        Y = Y[Y_left_2]
        Yo_new = Yo_new[Y_left_2].astype(np.bool)
        X = (X - t) @ R
        return X, Y, R, t, Xo_new, Yo_new


class RandomJitter:
    """Add random noise to both point clouds independently."""

    def __init__(self, scale=0.01, min_clip=-0.05, max_clip=0.05):
        self.scale = scale
        self.min_clip = min_clip
        self.max_clip = max_clip

    def __call__(self, X, Y, R, t, Xo, Yo):
        X_noise = np.random.normal(scale=self.scale, size=X.shape)
        X_noise = np.clip(X_noise, self.min_clip, self.max_clip)
        X = X + X_noise
        Y_noise = np.random.normal(scale=self.scale, size=X.shape)
        Y_noise = np.clip(Y_noise, self.min_clip, self.max_clip)
        Y = Y + Y_noise
        return X, Y, R, t, Xo, Yo


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

    def __call__(self, X, Y, R, t, Xo, Yo):
        # If ``x, y`` are three-dimensional vectors,
        # ``R, Q`` are rotation matrices and ``y = R @ x + t``,
        # then ``y = (R @ Q.T) @ (Q @ x) + t``.
        Q = get_random_rotation_matrix(self.min_degrees, self.max_degrees)
        X = np.matmul(X, Q.T)
        R = np.matmul(R, Q.T)
        return X, Y, R, t, Xo, Yo


class RandomRotateTarget:
    """Randomly rotate the target point cloud."""

    def __init__(self, min_degrees=0.0, max_degrees=45.0):
        self.min_degrees = min_degrees
        self.max_degrees = max_degrees

    def __call__(self, X, Y, R, t, Xo, Yo):
        # If ``x, y`` are three-dimensional vectors,
        # ``R, Q`` are rotation matrices and ``y = R @ x + t``,
        # then ``Q @ y = (Q @ R) @ x + (Q @ t)``.
        Q = get_random_rotation_matrix(self.min_degrees, self.max_degrees)
        Y = np.matmul(Y, Q.T)
        R = np.matmul(Q, R)
        t = np.matmul(Q, t)
        return X, Y, R, t, Xo, Yo


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

    def __call__(self, X, Y, R, t, Xo, Yo):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y = R @ (x + c) + (t - R @ c)``.
        c = get_random_translation_vector(self.lower_bound, self.upper_bound)
        X = X + c
        t = t - np.matmul(R, c)
        return X, Y, R, t, Xo, Yo


class RandomTranslateTarget:
    """Randomly translate the target point cloud."""

    def __init__(self, lower_bound=-0.5, upper_bound=0.5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, X, Y, R, t, Xo, Yo):
        # If ``x, y, c`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``y + c = R @ x + (t + c)``.
        c = get_random_translation_vector(self.lower_bound, self.upper_bound)
        Y = Y + c
        t = t + c
        return X, Y, R, t, Xo, Yo


class RandomShuffle:
    """Randomly reorder the points in both point clouds independently."""

    def __call__(self, X, Y, R, t, Xo, Yo):
        X_permutation = np.random.permutation(len(X))
        Y_permutation = np.random.permutation(len(Y))
        X = X[X_permutation]
        Y = Y[Y_permutation]
        Xo = Xo[X_permutation]
        Yo = Yo[Y_permutation]
        return X, Y, R, t, Xo, Yo


class RandomSwap:
    """Randomly swap the source point cloud with the target point cloud."""

    def __call__(self, X, Y, R, t, Xo, Yo):
        # If ``x, y`` are three-dimensional vectors,
        # ``R`` is a rotation matrix and ``y = R @ x + t``,
        # then ``x = (R.T) @ y + (- R.T @ t)``.
        if np.random.random() > 0.5:
            X, Y = Y, X
            Xo, Yo = Yo, Xo
            t = -1.0 * np.matmul(R.T, t)
            R = R.T
        return X, Y, R, t, Xo, Yo


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
    train_data_path = "Datasets/Processed/modelnet40_partial_train.hdf5"
    val_data_path = "Datasets/Processed/modelnet40_partial_val.hdf5"
    test_data_path = "Datasets/Processed/modelnet40_partial_test.hdf5"
    train_tr = v2.Compose(
        [RandomCrop(), RandomRotateSource(), RandomTranslateSource()]
    )
    train_ds = PointCloudDataset(train_data_path, transform=train_tr)
    val_ds = PointCloudDataset(val_data_path, transform=None)
    test_ds = PointCloudDataset(test_data_path, transform=None)
    X, Y, R, t, Xo, Yo = test_ds[0]
    draw_point_clouds(X, Y)  # Draw source point cloud and target point cloud
    draw_point_clouds(X @ R.T + t, Y)  # Draw aligned point clouds
    draw_point_clouds(X[Xo] @ R.T + t, Y[Yo])  # Draw aligned overlapping points
