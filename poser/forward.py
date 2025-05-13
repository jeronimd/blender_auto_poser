import torch
import torch.nn as nn

from utils.constants import Constants
from utils.path_manager import PathManager


class ForwardKinematicsLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_manager = PathManager()

        # Load bone connections as indices {child: parent}
        self.bone_connections = {
            Constants.BONE_IDX[child]: Constants.BONE_IDX[parent] if parent is not None else -1
            for child, parent in Constants.BONE_CONNECTIONS.items()
        }

        # Load bone offsets
        self.offsets = torch.tensor([
            Constants.DEFAULT_BONE_OFFSETS_LOCAL_LOCATION[joint] for joint in Constants.BONE_IDX
        ], dtype=torch.float32)

    def forward(self, local_rots, root_pos=None):
        # Sanity checks
        assert not torch.isnan(local_rots).any(), "NaNs in ForwardKinematicsLayer forward local_rots"

        if root_pos is None:
            batch_size = local_rots.shape[0]
            root_pos = torch.zeros(batch_size, 3, device=local_rots.device)

        assert not torch.isnan(root_pos).any(), "NaNs in ForwardKinematicsLayer forward root_pos"

        batch_size, num_joints = local_rots.shape[:2]
        device = local_rots.device

        # Initialize results list - store tensors of shape [B, 1, 3, 3] and [B, 1, 3]
        all_global_rots = [local_rots[:, 0:1]]  # Start with root rotation
        all_global_pos = [root_pos.unsqueeze(1)]  # Start with root position (add joint dim)

        for j in range(1, num_joints):
            parent = self.bone_connections.get(j, -1)
            if parent == -1:
                raise RuntimeError(f"Joint {j} has no valid parent")

            parent_rot = all_global_rots[parent][:, 0]  # Get B,3,3 from list element B,1,3,3
            parent_pos = all_global_pos[parent][:, 0]  # Get B,3 from list element B,1,3

            local_rot = local_rots[:, j]
            offset = self.offsets[j].to(device)

            new_global_rot_j = torch.matmul(parent_rot, local_rot)
            transformed_offset = torch.einsum('bij,j->bi', parent_rot, offset)
            new_global_pos_j = parent_pos + transformed_offset

            all_global_rots.append(new_global_rot_j.unsqueeze(1))
            all_global_pos.append(new_global_pos_j.unsqueeze(1))

        global_pos = torch.cat(all_global_pos, dim=1)
        global_rots = torch.cat(all_global_rots, dim=1)

        assert not torch.isnan(global_pos).any(), "NaNs in ForwardKinematicsLayer forward global_pos"
        assert not torch.isnan(global_rots).any(), "NaNs in ForwardKinematicsLayer forward global_rots"

        return global_pos, global_rots
