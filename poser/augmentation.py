import numpy as np
import torch

from poser.geometry import convert_matrix_to_quat, convert_quat_to_matrix
from utils.constants import Constants


class RandomRotation:
    def __init__(self):
        self.min_angle, self.max_angle = (-np.pi, np.pi)

    def __call__(self, locations, rotations):
        batch_size = locations.shape[0]
        device = locations.device

        # Generate random angles per sample
        thetas = torch.rand(batch_size, device=device) * (self.max_angle - self.min_angle) + self.min_angle
        cos_theta = torch.cos(thetas)
        sin_theta = torch.sin(thetas)

        # Construct batched rotation matrices (B, 3, 3)
        rot_matrices = torch.zeros(batch_size, 3, 3, device=device)
        rot_matrices[:, 0, 0] = cos_theta
        rot_matrices[:, 0, 1] = sin_theta
        rot_matrices[:, 1, 0] = -sin_theta
        rot_matrices[:, 1, 1] = cos_theta
        rot_matrices[:, 2, 2] = 1

        # Apply rotation to locations
        locations = torch.bmm(locations, rot_matrices.transpose(1, 2))

        # Apply to root rotations (index 0)
        root_mats = convert_quat_to_matrix(rotations[:, 0])
        new_root_mats = torch.bmm(rot_matrices, root_mats)
        rotations[:, 0] = convert_matrix_to_quat(new_root_mats)

        return locations, rotations


class MirrorSkeleton:
    def __init__(self):
        self.prob = 0.5
        self.bone_pairs_idx = [
            (Constants.BONE_IDX[left], Constants.BONE_IDX[right])
            for left, right in Constants.BONE_PAIRS
        ]

    def __call__(self, locations, rotations):
        device = locations.device
        batch_size = locations.shape[0]
        mask = torch.rand(batch_size, device=device) < self.prob
        swap_indices = torch.where(mask)[0]  # Shape: [n]

        if len(swap_indices) == 0:
            return locations, rotations

        # Reshape indices for broadcasting
        swap_indices_expanded = swap_indices.view(-1, 1)  # Shape: [n, 1]

        for l, r in self.bone_pairs_idx:
            # Create bone indices tensor and broadcast
            bone_indices = torch.tensor([l, r], device=device).view(1, -1)  # Shape: [1, 2]

            # Swap locations
            locations[swap_indices_expanded, bone_indices] = locations[swap_indices_expanded, bone_indices.flip(1)].clone()

            # Swap rotations
            rotations[swap_indices_expanded, bone_indices] = rotations[swap_indices_expanded, bone_indices.flip(1)].clone()

        # Mirror X-axis for swapped samples
        locations[swap_indices, :, 0] *= -1

        # Mirror rotations using matrix transformation
        M = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device)
        swap_rots = rotations[swap_indices].view(-1, 4)  # Shape: [n * num_bones, 4]
        swap_mats = convert_quat_to_matrix(swap_rots)  # Shape: [n * num_bones, 3, 3]
        mirrored_mats = M @ swap_mats @ M  # Mirror operation
        mirrored_quats = convert_matrix_to_quat(mirrored_mats)  # Shape: [n * num_bones, 4]
        rotations[swap_indices] = mirrored_quats.view(-1, rotations.shape[1], 4)  # Reshape back

        return locations, rotations
