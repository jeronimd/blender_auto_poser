import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
import pytorch_lightning as pl
import torch

from poser.forward import ForwardKinematicsLayer
from poser.geometry import (compute_geodesic_distance_from_two_matrices,
                            convert_euler_to_matrix, convert_matrix_to_ortho,
                            convert_ortho_to_matrix, convert_quat_to_matrix,
                            normalize_vector)
from poser.network import WeightedProtoRes
from utils.constants import Constants


class OptionalLookAtModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.nb_joints = len(Constants.BONE_IDX)

        # Hyper Parameters
        self.rot_loss_scale = 1.0
        self.fk_loss_scale = 100.0
        self.pos_loss_scale = 100.0
        self.lookat_loss_scale = 1.0
        self.true_lookat_loss_scale = 1.0

        self.loss_scales_learnable = False

        self.pos_effector_probability = 1.0
        self.rot_effector_probability = 1.0
        self.lookat_effector_probability = 1.0  # The lookat effector makes the model non-deterministic, not sure why, but its probably the normalize_vector

        self.min_effectors_count = 3
        self.max_effectors_count = 16
        self.min_pos_effectors = 3

        self.max_effector_noise_scale = 0.1
        self.effector_noise_exp = 13.0
        self.max_effector_weight = 1000.0

        self.lookat_distance_std = 5.0

        self.root_idx = Constants.BONE_IDX['pelvis']
        effector_indices = [Constants.BONE_IDX[name] for name in [
            'pelvis', 'spine001', 'spine002', 'spine003', 'neck', 'head',
            'shoulder_l', 'upperarm_l', 'forearm_l', 'hand_l',
            'shoulder_r', 'upperarm_r', 'forearm_r', 'hand_r',
            'upperleg_l', 'lowerleg_l', 'foot_l', 'toe_l',
            'upperleg_r', 'lowerleg_r', 'foot_r', 'toe_r'
        ]]

        self.fk_layer = ForwardKinematicsLayer()
        self.net = WeightedProtoRes(self.nb_joints)

        self.rot_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.rot_loss_scale)), requires_grad=self.loss_scales_learnable)
        self.fk_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.fk_loss_scale)), requires_grad=self.loss_scales_learnable)
        self.pos_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.pos_loss_scale)), requires_grad=self.loss_scales_learnable)
        self.lookat_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.lookat_loss_scale)), requires_grad=self.loss_scales_learnable)
        self.true_lookat_loss_scale = torch.nn.Parameter(torch.tensor(-0.5 * np.log(self.true_lookat_loss_scale)), requires_grad=self.loss_scales_learnable)

        self.pos_multinomial = torch.zeros((1, self.nb_joints))
        self.rot_multinomial = torch.zeros((1, self.nb_joints))
        self.lookat_multinomial = torch.zeros((1, self.nb_joints))
        self.pos_multinomial[:, effector_indices] = 1
        self.rot_multinomial[:, effector_indices] = 1
        self.lookat_multinomial[:, effector_indices] = 1

        self.type_multinomial = torch.tensor([
            self.pos_effector_probability,
            self.rot_effector_probability,
            self.lookat_effector_probability
        ], dtype=torch.float)

    @torch.no_grad()
    def get_data_from_batch(self, batch):
        input_data = {}
        device = batch["joint_positions"].device
        batch_size = batch["joint_positions"].shape[0]

        # ======================
        # FORWARD KINEMATICS
        # ======================
        joint_rotations = batch["joint_rotations"]
        joint_rotations_mat = convert_quat_to_matrix(joint_rotations.view(-1, 4)).view(batch_size, -1, 3, 3)
        _, joint_world_rotations_mat = self.fk_layer.forward(joint_rotations_mat)

        # ======================
        # EFFECTOR TYPE SAMPLING
        # ======================
        # note: we must have at least one positional effector (for translation invariance)
        num_random_effectors = torch.randint(low=self.min_effectors_count, high=self.max_effectors_count + 1, size=(1,)).numpy()[0]
        random_effector_types = torch.multinomial(input=self.type_multinomial, num_samples=num_random_effectors, replacement=True)
        num_pos_effectors = (random_effector_types == 0).sum()
        num_rot_effectors = (random_effector_types == 1).sum()
        num_lookat_effectors = (random_effector_types == 2).sum()
        # Forces a minimum number of position effector without shifting the expected number of position effectors
        # There are probably better ways to do this... This can also cause the actual number of effectors to be higher than expected
        num_pos_effectors = max(self.min_pos_effectors, num_pos_effectors)
        # # if look-at is not generalized, we cannot draw more look-at effector than the number of possible look-at joints
        # if not self.hparams.generalized_lookat:
        #     num_lookat_effectors = min((self.lookat_multinomial != 0).sum(), num_lookat_effectors)

        # ====================
        # POSITIONAL EFFECTORS
        # ====================
        pos_effector_ids = torch.multinomial(input=self.pos_multinomial.repeat(batch_size, 1), num_samples=num_pos_effectors, replacement=False).to(device)
        pos_effector_tolerances = self.max_effector_noise_scale * torch.pow(torch.rand(size=pos_effector_ids.shape).to(device), self.effector_noise_exp)
        pos_effector_weights = torch.ones(size=pos_effector_ids.shape).to(device)  # blending weights are always set to 1 during training
        pos_effectors_in = torch.gather(batch["joint_positions"], dim=1, index=pos_effector_ids.unsqueeze(2).repeat(1, 1, 3))
        pos_noise = pos_effector_tolerances.unsqueeze(2) * torch.randn((batch_size, num_pos_effectors, 3)).type_as(pos_effectors_in)
        pos_effectors_in = pos_effectors_in + pos_noise

        input_data["position_data"] = pos_effectors_in
        input_data["position_weight"] = pos_effector_weights
        input_data["position_tolerance"] = pos_effector_tolerances
        input_data["position_id"] = pos_effector_ids

        # ====================
        # ROTATIONAL EFFECTORS
        # ====================
        if num_rot_effectors > 0:
            joint_world_rotations_ortho6d = convert_matrix_to_ortho(joint_world_rotations_mat.view(-1, 3, 3)).view(batch_size, -1, 6)
            rot_effector_ids = torch.multinomial(input=self.rot_multinomial.repeat(batch_size, 1), num_samples=num_rot_effectors, replacement=False).to(device)
            rot_effectors_in = torch.gather(joint_world_rotations_ortho6d, dim=1, index=rot_effector_ids.unsqueeze(2).repeat(1, 1, 6))
            rot_effector_weight = torch.ones(size=rot_effector_ids.shape).to(device)  # blending weights are always set to 1 during training
            rot_effector_tolerances = self.max_effector_noise_scale * torch.pow(torch.rand(size=rot_effector_ids.shape).to(device), self.effector_noise_exp)
            rot_noise = rot_effector_tolerances.unsqueeze(2) * torch.randn((batch_size, num_rot_effectors, 3)).type_as(rot_effectors_in)
            rot_noise = convert_euler_to_matrix(rot_noise.view(-1, 3))  # TODO: pick std that makes more sense for angles
            rot_effectors_in_mat = convert_ortho_to_matrix(rot_effectors_in.view(-1, 6))
            rot_effectors_in_mat = torch.matmul(rot_noise, rot_effectors_in_mat)
            rot_effectors_in = convert_matrix_to_ortho(rot_effectors_in_mat).view(batch_size, num_rot_effectors, 6)

            input_data["rotation_data"] = rot_effectors_in
            input_data["rotation_weight"] = rot_effector_weight
            input_data["rotation_tolerance"] = rot_effector_tolerances
            input_data["rotation_id"] = rot_effector_ids
        else:
            input_data["rotation_data"] = torch.zeros((batch_size, 0, 6)).type_as(joint_rotations)
            input_data["rotation_weight"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["rotation_tolerance"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["rotation_id"] = torch.zeros((batch_size, 0), dtype=torch.int64).to(device)

        # =================
        # LOOK-AT EFFECTORS
        # =================
        if num_lookat_effectors > 0:
            # Note: we set replacement to True for the generalized look-at as we expect user to be able to provide
            #       multiple look-at constraints on the same joint, for instance for simulating an Aim constraint
            lookat_effector_ids = torch.multinomial(input=self.lookat_multinomial.repeat(batch_size, 1), num_samples=num_lookat_effectors, replacement=True).to(device)
            lookat_effector_weights = torch.ones(size=lookat_effector_ids.shape).to(device)  # blending weights are always set to 1 during training
            lookat_positions = torch.gather(batch["joint_positions"], dim=1, index=lookat_effector_ids.unsqueeze(2).repeat(1, 1, 3))
            lookat_effector_tolerances = torch.zeros(size=lookat_effector_ids.shape).to(device)  # TODO: self.hparams.max_effector_noise_scale * torch.pow(torch.rand(size=lookat_effector_ids.shape).to(device), self.hparams.effector_noise_exp)
            lookat_rotations_mat = torch.gather(joint_world_rotations_mat, dim=1, index=lookat_effector_ids.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3))
            # if self.hparams.generalized_lookat:
            local_lookat_directions = torch.randn((batch_size, num_lookat_effectors, 3)).type_as(lookat_positions)
            # else:
            #     local_lookat_directions = torch.zeros((batch_size, num_lookat_effectors, 3)).type_as(lookat_positions)
            #     local_lookat_directions[:, :, 2] = 1.0
            local_lookat_directions = normalize_vector(local_lookat_directions.view(-1, 3)).view(batch_size, num_lookat_effectors, 3)
            lookat_directions = torch.matmul(lookat_rotations_mat.view(-1, 3, 3), local_lookat_directions.view(-1, 3).unsqueeze(2)).squeeze(1).view(batch_size, num_lookat_effectors, 3)
            lookat_distance = 1e-3 + self.lookat_distance_std * torch.abs(torch.randn((batch_size, num_lookat_effectors, 1)).type_as(lookat_directions))
            lookat_positions = lookat_positions + lookat_distance * lookat_directions
            # if self.hparams.add_effector_noise:
            #     # lookat_noise = lookat_effector_tolerances.unsqueeze(2) * torch.randn((batch_size, num_lookat_effectors, 3)).type_as(lookat_positions)
            #     lookat_positions = lookat_positions  # + lookat_noise
            lookat_effectors_in = torch.cat([lookat_positions, local_lookat_directions], dim=2)

            input_data["lookat_data"] = lookat_effectors_in
            input_data["lookat_weight"] = lookat_effector_weights
            input_data["lookat_tolerance"] = lookat_effector_tolerances
            input_data["lookat_id"] = lookat_effector_ids
        else:
            input_data["lookat_data"] = torch.zeros((batch_size, 0, 6)).type_as(joint_rotations)
            input_data["lookat_weight"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["lookat_tolerance"] = torch.zeros((batch_size, 0)).type_as(joint_rotations)
            input_data["lookat_id"] = torch.zeros((batch_size, 0), dtype=torch.int64).to(device)

        target_data = {
            "joint_positions": batch["joint_positions"],
            "root_joint_position": batch["joint_positions"][:, self.root_idx, :],
            "joint_rotations": joint_rotations,
            "joint_rotations_mat": joint_rotations_mat,
            "joint_world_rotations_mat": joint_world_rotations_mat
        }

        return input_data, target_data

    def pack_data(self, input_data):
        effector_data = []  # effector + tolerance
        effector_ids = []
        effector_types = []
        effector_weight = []

        # POSITIONS
        pos_effectors_in = input_data["position_data"]
        pos_effectors_in = torch.cat([pos_effectors_in, torch.zeros((pos_effectors_in.shape[0], pos_effectors_in.shape[1], 3)).type_as(pos_effectors_in), input_data["position_tolerance"].unsqueeze(2)], dim=2)  # padding with zeros
        effector_data.append(pos_effectors_in)
        effector_ids.append(input_data["position_id"])
        effector_types.append(torch.zeros_like(input_data["position_id"]))
        effector_weight.append(input_data["position_weight"])

        # ROTATIONS
        effector_data.append(torch.cat([input_data["rotation_data"], input_data["rotation_tolerance"].unsqueeze(2)], dim=2))
        effector_ids.append(input_data["rotation_id"])
        effector_types.append(torch.ones_like(input_data["rotation_id"]))
        effector_weight.append(input_data["rotation_weight"])

        # LOOK-AT
        effector_data.append(torch.cat([input_data["lookat_data"], input_data["lookat_tolerance"].unsqueeze(2)], dim=2))
        effector_ids.append(input_data["lookat_id"])
        effector_types.append(2 * torch.ones_like(input_data["lookat_id"]))
        effector_weight.append(input_data["lookat_weight"])

        return {
            "effectors": torch.cat(effector_data, dim=1),
            "effector_type": torch.cat(effector_types, dim=1),
            "effector_id": torch.cat(effector_ids, dim=1),
            "effector_weight": torch.cat(effector_weight, dim=1)
        }

    def make_translation_invariant(self, input_data):
        # re-reference with WEIGHTED centroid of positional effectors
        # IMPORTANT NOTE 1: we create a new data structure and tensors to avoid side-effects of modifying input data
        # IMPORTANT NOTE 2: centroid is weighted so that effectors with null blending weights don't impact computations in any way
        referenced_input_data = input_data.copy()
        pos_weights = referenced_input_data["position_weight"].unsqueeze(2)
        pos_weights_sum = pos_weights.sum(dim=1, keepdim=True)
        reference_pos = (referenced_input_data["position_data"] * pos_weights).sum(dim=1, keepdim=True) / pos_weights_sum
        referenced_input_data["position_data"] = referenced_input_data["position_data"] - reference_pos
        referenced_input_data["lookat_data"] = torch.cat([referenced_input_data["lookat_data"][:, :, 0:3] - reference_pos, referenced_input_data["lookat_data"][:, :, 3:6]], dim=2)

        return referenced_input_data, reference_pos

    def forward(self, input_data):
        referenced_input_data, reference_pos = self.make_translation_invariant(input_data)

        packed_data = self.pack_data(referenced_input_data)  # Concatenate all

        effectors_in = packed_data["effectors"]
        effector_ids = packed_data["effector_id"]
        effector_types = packed_data["effector_type"]
        effector_weights = packed_data["effector_weight"]

        out_positions, out_rotations = self.net(effectors_in, effector_weights, effector_ids, effector_types)

        joint_positions = out_positions.view(-1, self.nb_joints, 3) + reference_pos
        joint_rotations = out_rotations.view(-1, self.nb_joints, 6)

        return {
            "joint_positions": joint_positions,
            "joint_rotations": joint_rotations,
            "root_joint_position": joint_positions[:, self.root_idx, :]
        }

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        self.log_losses(losses, "train/")
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        self.log_losses(losses, prefix="validation/")

    def shared_step(self, batch):
        input_data, target_data = self.get_data_from_batch(batch)
        batch_size = input_data["position_data"].shape[0]

        in_position_data = input_data["position_data"]
        in_position_ids = input_data["position_id"]
        in_position_tolerance = input_data["position_tolerance"]

        in_rotation_data = input_data["rotation_data"]
        in_rotation_ids = input_data["rotation_id"]
        in_rotation_tolerance = input_data["rotation_tolerance"]

        in_lookat_data = input_data["lookat_data"]
        in_lookat_ids = input_data["lookat_id"]
        in_lookat_tolerance = input_data["lookat_tolerance"]

        target_joint_positions = target_data["joint_positions"]
        target_root_joint_positions = target_data["root_joint_position"]
        target_joint_rotations_mat = target_data["joint_rotations_mat"]
        target_joint_rotations_fk = target_data["joint_world_rotations_mat"]

        # ==================
        # FORWARD PASS
        # ==================
        predicted = self.forward(input_data)
        predicted_root_joint_position = predicted["root_joint_position"]
        predicted_joint_positions = predicted["joint_positions"]
        predicted_joint_rotations = predicted["joint_rotations"]

        predicted_joint_rotations_mat = convert_ortho_to_matrix(predicted_joint_rotations.view(-1, 6)).view(-1, self.nb_joints, 3, 3)  # compute rotation matrices
        predicted_joint_positions_fk, predicted_joint_rotations_fk = self.fk_layer.forward(predicted_joint_rotations_mat, predicted_root_joint_position)

        # ==================
        # POSITION EFFECTORS
        # ==================
        pos_effector_w = self.compute_weights_from_std(in_position_tolerance, self.max_effector_weight)
        joint_positions_weights = torch.ones(predicted_joint_positions_fk.shape[0:2]).type_as(in_position_tolerance)  # to mimick the old behaviour, we apply a 1 weight to all joints by default
        joint_positions_weights.scatter_(dim=1, index=in_position_ids.view(batch_size, -1), src=pos_effector_w.view(batch_size, -1))

        # ==================
        # ROTATION EFFECTORS
        # ==================
        rot_effector_w = self.compute_weights_from_std(in_rotation_tolerance, self.max_effector_weight)
        joint_rotations_weights = torch.zeros(predicted_joint_rotations_fk.shape[0:2]).type_as(in_rotation_tolerance)  # to mimick the old behaviour, we apply a 0 weight to all joints by default
        joint_rotations_weights.scatter_(dim=1, index=in_rotation_ids.view(batch_size, -1), src=rot_effector_w.view(batch_size, -1))

        # =================
        # LOOK-AT EFFECTORS
        # =================
        # TODO: look-at margin
        if in_lookat_data.shape[1] > 0:
            lookat_target = in_lookat_data[:, :, 0:3]
            local_lookat_directions = in_lookat_data[:, :, 3:6]
            lookat_rotations_mat = torch.gather(predicted_joint_rotations_fk, dim=1, index=in_lookat_ids.unsqueeze(2).unsqueeze(3).repeat(1, 1, 3, 3))
            predicted_lookat_directions = torch.matmul(lookat_rotations_mat.view(-1, 3, 3), local_lookat_directions.view(-1, 3).unsqueeze(2)).squeeze(1).view_as(local_lookat_directions)
            predicted_lookat_positions = torch.gather(predicted_joint_positions_fk, dim=1, index=in_lookat_ids.unsqueeze(2).repeat(1, 1, 3))
            predicted_target_directions = lookat_target - predicted_lookat_positions
            predicted_target_directions = normalize_vector(predicted_target_directions.view(-1, 3)).view_as(predicted_target_directions)
            true_lookat_loss = self.angular_loss(predicted_lookat_directions.view(-1, 3), predicted_target_directions.view(-1, 3))
        else:
            true_lookat_loss = torch.zeros((1)).type_as(in_lookat_data)

        # =================
        # LOSSES
        # =================
        fk_loss = self.weighted_mse(predicted_joint_positions_fk.view(-1, 3), target_joint_positions.view(-1, 3), joint_positions_weights.view(-1))
        pos_loss = self.weighted_mse(predicted_joint_positions.view(-1, 3), target_joint_positions.view(-1, 3), joint_positions_weights.view(-1))
        rot_loss = self.geodesic_loss(predicted_joint_rotations_mat.view(-1, 3, 3), target_joint_rotations_mat.view(-1, 3, 3))
        lookat_loss = self.weighted_geodesic_loss(predicted_joint_rotations_fk.view(-1, 3, 3), target_joint_rotations_fk.view(-1, 3, 3), joint_rotations_weights.view(-1))

        # This is loss weighting as per https://arxiv.org/pdf/1705.07115.pdf
        total_loss = 0.0
        rot_loss_scale_exp = torch.exp(-2 * self.rot_loss_scale)
        total_loss = total_loss + rot_loss_scale_exp * rot_loss + self.rot_loss_scale
        fk_loss_scale_exp = torch.exp(-2 * self.fk_loss_scale)
        total_loss = total_loss + fk_loss_scale_exp * fk_loss + self.fk_loss_scale
        pos_loss_scale_exp = torch.exp(-2 * self.pos_loss_scale)
        total_loss = total_loss + pos_loss_scale_exp * pos_loss + self.pos_loss_scale
        lookat_loss_scale_exp = torch.exp(-2 * self.lookat_loss_scale)
        total_loss = total_loss + lookat_loss_scale_exp * lookat_loss + self.lookat_loss_scale
        true_lookat_loss_scale_exp = torch.exp(-2 * self.true_lookat_loss_scale)
        total_loss = total_loss + true_lookat_loss_scale_exp * true_lookat_loss + self.true_lookat_loss_scale

        return {
            "total": total_loss,
            "fk": fk_loss,
            "position": pos_loss,
            "rotation": rot_loss,
            "lookat": lookat_loss,
            "true_lookat": true_lookat_loss,
            "scale_rot": rot_loss_scale_exp,
            "scale_fk": fk_loss_scale_exp,
            "scale_pos": pos_loss_scale_exp,
            "scale_lookat": lookat_loss_scale_exp,
            "scale_true_lookat": true_lookat_loss_scale_exp,
            "rotation_max": predicted_joint_rotations.max(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    def on_train_epoch_start(self) -> None:
        try:
            dataset = self.trainer.train_dataloader.dataset
            dataset.set_epoch(self.current_epoch)
        except Exception:
            pass
        return super().on_train_epoch_start()

    def log_losses(self, losses, prefix: str = ""):
        for k, v in losses.items():
            if v is not None:
                self.log(prefix + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                         sync_dist=True)

    @staticmethod
    def compute_weights_from_std(std: torch.Tensor, max_weight: float, std_at_max: float = 1e-3) -> torch.Tensor:
        m = max_weight * std_at_max
        return m / std.clamp(min=std_at_max)

    @staticmethod
    def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        ret = ((pred - target) ** 2).sum(dim=1) / pred.shape[1]
        ret = ret * weights
        return torch.sum(ret) / (weights.sum() + eps)

    @staticmethod
    def geodesic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        theta = compute_geodesic_distance_from_two_matrices(pred, target)
        error = theta.mean()
        return error

    @staticmethod
    def weighted_geodesic_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        theta = compute_geodesic_distance_from_two_matrices(target, pred)
        error = torch.sum(theta * weights) / (weights.sum() + eps)
        return error

    @staticmethod
    def angular_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        loss = (target * pred).sum(dim=1)
        loss = torch.acos(loss.clamp(-1.0 + eps, 1.0 - eps))
        loss = loss.mean()
        return loss
