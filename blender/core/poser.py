from pathlib import Path

import torch
from mathutils import Quaternion, Vector

from ...poser.geometry import convert_ortho_to_quat, convert_quat_to_ortho
from ...poser.model import OptionalLookAtModel
from ...utils.constants import Constants
from ...utils.logger import Logger
from ...utils.path_manager import PathManager
from ..core.keyframes import keyframes_manager


class PoserManager:
    def __init__(self):
        self.path_manager = PathManager()
        self.logger = Logger.setup(__name__, "DEBUG")
        self.device = "cpu"

        self.current_model = None
        self.is_model_active = False

        self.models_list = {}

    def get_models(self):
        models_dir = Path(self.path_manager.models_path)
        self.models_list = {}

        for model_path in models_dir.rglob("*.ckpt"):
            stem = model_path.stem
            key = stem
            suffix = 1
            # Ensure the key is unique by appending a suffix if needed
            while key in self.models_list:
                key = f"{stem} ({suffix})"
                suffix += 1
            # Store the resolved path to avoid symlink issues
            self.models_list[key] = str(model_path.resolve())

        return list(self.models_list.keys())

    def load_model(self, model_name):
        model_path = self.models_list.get(model_name)
        self.logger.debug(f"Loading model: {model_name} from {model_path}")
        try:
            print("\033[91m", "If it crashes try to remove only Python38.dll from the folder.", "\033[0m", sep="")  # TODO: remove before packing
            self.current_model = OptionalLookAtModel.load_from_checkpoint(model_path)
            self.current_model.to(self.device)
            self.current_model.eval()
            self.is_model_active = True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.current_model = None
            self.is_model_active = False

    def unload_model(self):
        self.logger.debug("Unloading model")
        self.current_model = None
        self.is_model_active = False

    def predict(self, armature):
        if not self.is_model_active:
            return

        auto_poser = armature.data.auto_poser

        # Make sure that there is at least one location effector
        found_location_effector = False
        for effector in auto_poser.location_effectors:
            if effector.enabled and effector.bone != "NONE":
                found_location_effector = True
                break
        if not found_location_effector:
            return

        input_data = self._format_input(auto_poser)
        with torch.no_grad():
            predicted_pose = self.current_model.forward(input_data)

        self._apply_output(armature, auto_poser, predicted_pose)

    def _format_input(self, auto_poser):
        pos_data_list, pos_weight_list, pos_tolerance_list, pos_id_list = [], [], [], []
        rot_data_list, rot_weight_list, rot_tolerance_list, rot_id_list = [], [], [], []
        lookat_data_list, lookat_weight_list, lookat_tolerance_list, lookat_id_list = [], [], [], []

        # --- Location Effectors ---
        for effector in auto_poser.location_effectors:
            if effector.enabled and effector.bone != "NONE":
                bone_data = getattr(auto_poser.bones, effector.bone)
                bone_index = Constants.BONE_IDX[effector.bone]
                if bone_data.empty_key and bone_index not in pos_id_list:

                    offset_loc = Vector(bone_data.offset_global_location)
                    location = bone_data.empty_key.location - offset_loc

                    pos_data_list.append(list(location))
                    pos_id_list.append(bone_index)
                    pos_tolerance_list.append(effector.tolerance)
                    pos_weight_list.append(1.0)

        # --- Rotation Effectors ---
        for effector in auto_poser.rotation_effectors:
            if effector.enabled and effector.bone != "NONE":
                bone_data = getattr(auto_poser.bones, effector.bone)
                bone_index = Constants.BONE_IDX[effector.bone]
                if bone_data.empty_key and bone_index not in rot_id_list:

                    rotation = bone_data.empty_key.rotation_quaternion

                    ortho6d = convert_quat_to_ortho(torch.tensor([rotation.w, rotation.x, rotation.y, rotation.z], dtype=torch.float32, device=self.device))
                    rot_data_list.append(list(ortho6d))
                    rot_id_list.append(bone_index)
                    rot_tolerance_list.append(effector.tolerance)
                    rot_weight_list.append(1.0)

        # --- LookAt Effectors ---
        for effector in auto_poser.lookat_effectors:
            if effector.enabled and effector.target and effector.bone != "NONE":
                bone_data = getattr(auto_poser.bones, effector.bone)
                bone_index = Constants.BONE_IDX[effector.bone]
                if effector.target:  # and bone_index not in lookat_id_list:  # lookat should be able to have multiple targets
                    world_target_pos = effector.target.location
                    local_direction = [0.0, -1.0, 0.0]  # Fixed local direction
                    lookat_entry = [world_target_pos.x, world_target_pos.y, world_target_pos.z] + local_direction
                    lookat_data_list.append(lookat_entry)
                    lookat_id_list.append(bone_index)
                    lookat_tolerance_list.append(effector.tolerance)
                    lookat_weight_list.append(1.0)

        # --- Tensors ---
        # Data tensors (Shape: 1, N, Dim)
        pos_data_tensor = torch.tensor(pos_data_list, dtype=torch.float32, device=self.device).reshape(len(pos_data_list), 3).unsqueeze(0)
        rot_data_tensor = torch.tensor(rot_data_list, dtype=torch.float32, device=self.device).reshape(len(rot_data_list), 6).unsqueeze(0)
        lookat_data_tensor = torch.tensor(lookat_data_list, dtype=torch.float32, device=self.device).reshape(len(lookat_data_list), 6).unsqueeze(0)

        # Scalar feature tensors (Shape: 1, N)
        pos_weight_tensor = torch.tensor(pos_weight_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_tolerance_tensor = torch.tensor(pos_tolerance_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_id_tensor = torch.tensor(pos_id_list, dtype=torch.int64, device=self.device).unsqueeze(0)

        rot_weight_tensor = torch.tensor(rot_weight_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        rot_tolerance_tensor = torch.tensor(rot_tolerance_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        rot_id_tensor = torch.tensor(rot_id_list, dtype=torch.int64, device=self.device).unsqueeze(0)

        lookat_weight_tensor = torch.tensor(lookat_weight_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        lookat_tolerance_tensor = torch.tensor(lookat_tolerance_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        lookat_id_tensor = torch.tensor(lookat_id_list, dtype=torch.int64, device=self.device).unsqueeze(0)

        # --- Assemble Dictionary ---
        input_data = {
            "position_data": pos_data_tensor, "position_weight": pos_weight_tensor, "position_tolerance": pos_tolerance_tensor, "position_id": pos_id_tensor,
            "rotation_data": rot_data_tensor, "rotation_weight": rot_weight_tensor, "rotation_tolerance": rot_tolerance_tensor, "rotation_id": rot_id_tensor,
            "lookat_data": lookat_data_tensor, "lookat_weight": lookat_weight_tensor, "lookat_tolerance": lookat_tolerance_tensor, "lookat_id": lookat_id_tensor,
        }

        return input_data

    def _apply_output(self, armature, auto_poser, predicted_pose):
        predicted_root_position = predicted_pose["root_joint_position"]
        predicted_rotations = predicted_pose["joint_rotations"]

        positions = predicted_root_position.squeeze(0).numpy()
        rotations = convert_ortho_to_quat(predicted_rotations).squeeze(0).numpy()

        for bone_name, bone_idx in Constants.BONE_IDX.items():
            bone_data = getattr(auto_poser.bones, bone_name)

            if bone_data.bone_mapping_is_valid:
                bone_pose = armature.pose.bones.get(bone_data.mapped_name)

                if bone_name == Constants.ROOT_BONE:
                    world_position = Vector(positions)
                    local_position = bone_pose.bone.matrix_local.inverted() @ world_position

                    offset_loc = Vector(bone_data.offset_global_location)
                    bone_pose.location = local_position + offset_loc

                rotation = Quaternion((rotations[bone_idx]))
                bone_pose.rotation_mode = 'QUATERNION'

                rest_orientation = bone_pose.bone.matrix_local.to_quaternion()
                rotation = rest_orientation.inverted() @ rotation @ rest_orientation

                bone_pose.rotation_quaternion = rotation

        if auto_poser.insert_keyframes_toggle:
            keyframes_manager.insert_keyframes(armature)


poser_manager = PoserManager()
