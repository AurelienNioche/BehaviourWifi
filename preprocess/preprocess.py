import os.path

import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import quaternion_apply, axis_angle_to_quaternion
import numpy as np
import pandas as pd
import glob

from utils.utils import path_to_alphanumeric


class JointDataset(Dataset):
    def __init__(self, x, joint_names):
        super().__init__()
        self.joint_names = joint_names
        self.furthest_distance = None
        self.centroid = None
        self.reference_idx = None
        self.x = self._transform(self._scale(x))

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx]

    def _scale(self, x):
        # Center the points and reduce to unit-sphere
        x = x.reshape(-1, 3)
        self.centroid = torch.mean(x, axis=0)
        x -= self.centroid
        self.furthest_distance = torch.max(torch.sqrt(torch.sum(x ** 2, axis=-1)))
        x /= self.furthest_distance
        # Put back into the original shape
        x = x.reshape(-1, 32, 3)
        return x

    def _unscale(self, x):
        if self.furthest_distance is None:
            raise ValueError("Dataset has not been scaled yet")
        x *= self.furthest_distance
        x += self.centroid
        return x

    def _transform(self, x):
        """
            Compute the transformation matrix that transforms the reference points into the target points.
            references = torch.from_numpy(np.random.randint(0, 10, size=((n_obs, 3)))).float()
            targets = torch.from_numpy(np.random.randint(0, 10, size=((n_obs, n_joints, 3)))).float()
        """

        # We use the pelvis as the reference
        pelvis_idx = list(self.joint_names).index("PELVIS")
        self.reference_idx = pelvis_idx

        # Get the references
        references = x[:, self.reference_idx]

        # Split the tensor into two parts, before and after the pelvis_idx
        first_part = x[:, :self.reference_idx]
        second_part = x[:, self.reference_idx+1:]

        # Stack the two parts together
        targets = torch.cat((first_part, second_part), dim=1)

        # Get the number of joints
        n_joints = targets.size(1)

        # Normalize references and targets
        normalized_ref = references / torch.norm(references, dim=-1, keepdim=True)
        normalized_tgt = targets / torch.norm(targets, dim=-1, keepdim=True)

        # Calculate the axis of rotation
        axis = torch.cross(normalized_ref.unsqueeze(1), normalized_tgt)
        axis = axis / torch.norm(axis, dim=-1, keepdim=True)

        # The angle of rotation is the arccosine of the dot product of the normalized vectors
        angle_of_rotation = torch.acos((normalized_ref.unsqueeze(1) * normalized_tgt).sum(-1))

        # The axis-angle representation is the axis of rotation scaled by the angle of rotation
        axis_angle = axis * angle_of_rotation.unsqueeze(-1)

        # Convert the axis-angle representation to a quaternion
        rotation = axis_angle_to_quaternion(axis_angle)

        # Apply the rotation to references
        rotated_ref = quaternion_apply(rotation, references.unsqueeze(1).expand(-1, n_joints, -1))

        # Compute the translation vector
        translation = targets - rotated_ref

        transformations = torch.cat((rotation, translation), dim=-1)
        flatten_transformations = transformations.reshape(transformations.size(0), -1)

        data = torch.cat((references, flatten_transformations), dim=-1)

        return data

    def _untransform(self, x):

        n_joints = len(self.joint_names)

        references = x[:, :3]
        references = references.unsqueeze(1)

        transformation = x[:, 3:]
        transformation = transformation.reshape(-1, n_joints-1, 7)

        rotation = transformation[:, :, :4]
        translation = transformation[:, :, 4:]

        # Apply the rotation
        rotated = quaternion_apply(rotation, references.expand(-1, n_joints-1, -1))

        # Apply the translation
        transformed = rotated + translation

        # Put back things together
        data = torch.cat((transformed[:, :self.reference_idx, :],
                          references,
                          transformed[:, self.reference_idx:, :]), dim=1)

        return data

    def untransform_and_unscale(self, x=None):
        if x is None:
            x = self.x.clone()
        x = self._untransform(x)
        x = self._unscale(x)
        return x


def make_joint_dataset(device: torch.device, data_folder: str, bkp_folder: str = "data/bkp"):

    os.makedirs(bkp_folder, exist_ok=True)

    bkp_name = path_to_alphanumeric(data_folder)

    bkp_file = f"{bkp_folder}/{bkp_name}.pt"

    if os.path.exists(bkp_file):
        return torch.load(bkp_file)

    files = []
    runs = glob.glob(f"{data_folder}/run_*")
    for r in runs:
        labels = glob.glob(f"{r}/processed/labelled/camera/joints/sit_stand/front/*")
        for lb in labels:
            files += glob.glob(f"{lb}/*")

    print("number of files:", len(files))
    joint_names = None
    xs = []
    for f in files:
        # "data/unlabelled/camera/joints/front_sit_stand.csv"
        with open(f, 'r') as f_:
            df = pd.read_csv(
                f_, names=["frame_id", "timestamp", "joint_name", "x", "y", "z"])
            if not len(df):
                continue

        # Get the unique joint names and frame IDs
        joint_names = np.sort(df["joint_name"].unique())
        frame_ids = df["frame_id"].unique()

        n_joint = len(joint_names)

        for f_id in frame_ids:
            if len(df[df.frame_id == f_id]) != n_joint:
                df = df[df.frame_id != f_id]
                frame_ids = frame_ids[frame_ids != f_id]

        # Create a multi-index using 'frame_id' and 'joint_names'
        df.set_index(['frame_id', 'joint_name'], inplace=True)

        # Sort the index to ensure the data is in the correct order
        df.sort_index(inplace=True)

        # Convert the DataFrame to a NumPy array and reshape it
        x = df[['x', 'y', 'z']].values.reshape((frame_ids.size, joint_names.size, 3))

        # Add to the list
        xs.append(x)

    # Concatenate the list of NumPy arrays
    x = np.concatenate(xs)

    # Convert the NumPy array to a PyTorch tensor
    x = torch.from_numpy(x).float().to(device)

    # Create the dataset
    dataset = JointDataset(joint_names=joint_names, x=x)
    torch.save(dataset, bkp_file)


# def make_joint_dataset(device: torch.device):
#
#     with open("data/unlabelled/camera/joints/front_sit_stand.csv") as f:
#         data = pd.read_csv(f, header=0)
#         data = data.rename(columns={"x-axis": "x", "y-axis": "y", "z-axis": "z", "joint_names": "joint_name"})
#         # print(data.head())
#
#     # Get the unique joint names and frame IDs
#     joint_names = np.sort(data["joint_name"].unique())
#     frame_ids = data["frame_id"].unique()
#
#     # Create a multi-index using 'frame_id' and 'joint_names'
#     data.set_index(['frame_id', 'joint_name'], inplace=True)
#
#     # Sort the index to ensure the data is in the correct order
#     data.sort_index(inplace=True)
#
#     # Convert the DataFrame to a NumPy array and reshape it
#     x = data[['x', 'y', 'z']].values.reshape((frame_ids.size, joint_names.size, 3))
#
#     # Remove the first X frames and the last X frames
#     x = x[500:-500]
#
#     # Convert the NumPy array to a PyTorch tensor
#     x = torch.from_numpy(x).float().to(device)
#
#     # print("initial shape", x.size())
#
#     dataset = JointDataset(joint_names=joint_names, x=x)
#     return dataset
