import bpy
from mathutils import Quaternion, Vector

from ...utils.constants import Constants
from ...utils.logger import Logger
from ...utils.path_manager import PathManager


class OffsetsManager:
    def __init__(self):
        self.logger = Logger.setup(__name__, "DEBUG")
        self.path_manager = PathManager()

    def calculate_offsets(self, armature):
        auto_poser = armature.data.auto_poser

        # Center the armature and set it to rest pose
        current_armature_location = armature.location.copy()
        current_armature_rotation = armature.rotation_quaternion.copy()
        current_pose_position = armature.data.pose_position

        armature.location = (0, 0, 0)
        armature.rotation_quaternion = (1, 0, 0, 0)
        armature.data.pose_position = 'REST'
        bpy.context.view_layer.update()

        # Calculate the offsets for each bone
        for bone_name in Constants.BONE_IDX:
            bone_data = getattr(auto_poser.bones, bone_name)

            if bone_data.bone_mapping_is_valid:
                bone_pose = armature.pose.bones.get(bone_data.mapped_name)

                if bone_pose.parent:
                    local_locations_offset = bone_pose.bone.head_local - bone_pose.bone.parent.head_local
                    local_rotations_offset = bone_pose.bone.parent.matrix_local.to_quaternion().inverted() @ bone_pose.bone.matrix_local.to_quaternion()
                    bone_data.offset_local_location = list(local_locations_offset)
                    bone_data.offset_local_rotation = list(local_rotations_offset)
                else:
                    bone_data.offset_local_location = [0.0, 0.0, 0.0]
                    bone_data.offset_local_rotation = [1.0, 0.0, 0.0, 0.0]

                current_location = bone_pose.head
                current_rotation = bone_pose.bone.matrix_local.to_quaternion()

                default_location = Vector(Constants.DEFAULT_BONE_OFFSETS_GLOBAL_LOCATION[bone_name])
                default_rotation = Quaternion(Constants.DEFAULT_BONE_OFFSETS_GLOBAL_ROTATION[bone_name])

                bone_data.offset_global_location = current_location - default_location
                bone_data.offset_global_rotation = default_rotation.inverted() @ current_rotation
            else:
                bone_data.offset_local_location = Constants.DEFAULT_BONE_OFFSETS_LOCAL_LOCATION[bone_name].copy()
                bone_data.offset_local_rotation = Constants.DEFAULT_BONE_OFFSETS_LOCAL_ROTATION[bone_name].copy()
                bone_data.offset_global_location = Constants.DEFAULT_BONE_OFFSETS_GLOBAL_LOCATION[bone_name].copy()
                bone_data.offset_global_rotation = Constants.DEFAULT_BONE_OFFSETS_GLOBAL_ROTATION[bone_name].copy()

        # Reset the armature back to its original state
        armature.location = current_armature_location
        armature.rotation_quaternion = current_armature_rotation
        armature.data.pose_position = current_pose_position
        bpy.context.view_layer.update()

    def print_default_offsets(self, armature):
        auto_poser = armature.data.auto_poser

        # Center the armature and set it to rest pose
        current_armature_location = armature.location.copy()
        current_armature_rotation = armature.rotation_quaternion.copy()
        current_pose_position = armature.data.pose_position

        armature.location = (0, 0, 0)
        armature.rotation_quaternion = (1, 0, 0, 0)
        armature.data.pose_position = 'REST'
        bpy.context.view_layer.update()

        # Default offsets
        default_bone_offsets_local_location = {}
        default_bone_offsets_local_rotation = {}
        default_bone_offsets_global_location = {}
        default_bone_offsets_global_rotation = {}
        default_bone_offsets_global_rotation_matrix = {}

        for bone_name in Constants.BONE_IDX:
            bone_data = getattr(auto_poser.bones, bone_name)
            if bone_data.bone_mapping_is_valid:
                bone_pose = armature.pose.bones.get(bone_data.mapped_name)

                if bone_pose.parent:
                    local_locations_offset = bone_pose.bone.head_local - bone_pose.bone.parent.head_local
                    local_rotations_offset = bone_pose.bone.parent.matrix_local.to_quaternion().inverted() @ bone_pose.bone.matrix_local.to_quaternion()
                    default_bone_offsets_local_location[bone_name] = list(local_locations_offset)
                    default_bone_offsets_local_rotation[bone_name] = list(local_rotations_offset)
                else:
                    default_bone_offsets_local_location[bone_name] = [0.0, 0.0, 0.0]
                    default_bone_offsets_local_rotation[bone_name] = [1.0, 0.0, 0.0, 0.0]

                default_bone_offsets_global_location[bone_name] = list(bone_pose.bone.head_local)  # == bone_pose.head
                default_bone_offsets_global_rotation[bone_name] = list(bone_pose.bone.matrix_local.to_quaternion())  # == bone_pose.matrix.to_quaternion()
                # default_bone_offsets_global_rotation[bone_name] = list(bone_pose.matrix_basis.to_quaternion()) # Current rotation locally
                # default_bone_offsets_global_rotation[bone_name] = list(bone_pose.bone.matrix.to_quaternion())  # dont know about this one, but it might be useful
                default_bone_offsets_global_rotation_matrix[bone_name] = list(bone_pose.bone.matrix_local.to_3x3())


        # Print the default offsets
        print("Armature:", armature.name)
        print("Default Offsets:")
        print("    DEFAULT_BONE_OFFSETS_LOCAL_LOCATION = {")
        for bone_name, offset in default_bone_offsets_local_location.items():
            print(f'        "{bone_name}": {offset},')
        print("    }\n")
        print("    DEFAULT_BONE_OFFSETS_LOCAL_ROTATION = {")
        for bone_name, offset in default_bone_offsets_local_rotation.items():
            print(f'        "{bone_name}": {offset},')
        print("    }\n")
        print("    DEFAULT_BONE_OFFSETS_GLOBAL_LOCATION = {")
        for bone_name, offset in default_bone_offsets_global_location.items():
            print(f'        "{bone_name}": {offset},')
        print("    }\n")
        print("    DEFAULT_BONE_OFFSETS_GLOBAL_ROTATION = {")
        for bone_name, offset in default_bone_offsets_global_rotation.items():
            print(f'        "{bone_name}": {offset},')
        print("    }\n")
        print("    DEFAULT_BONE_OFFSETS_GLOBAL_ROTATION_MATRIX = {")
        for bone_name, offset in default_bone_offsets_global_rotation_matrix.items():
            print(f'        "{bone_name}": {[list(item) for item in offset]},')
        print("    }")


        # Reset the armature back to its original state
        armature.location = current_armature_location
        armature.rotation_quaternion = current_armature_rotation
        armature.data.pose_position = current_pose_position
        bpy.context.view_layer.update()


offsets_manager = OffsetsManager()
