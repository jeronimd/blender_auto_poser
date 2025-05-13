import bpy

from ...utils.constants import Constants
from ...utils.logger import Logger


class KeyframesManager:
    def __init__(self):
        self.logger = Logger.setup(__name__, "DEBUG")

    def insert_keyframes(self, armature):
        auto_poser = armature.data.auto_poser

        original_active_object = bpy.context.view_layer.objects.active
        original_mode = armature.mode
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        bpy.ops.pose.select_all(action='DESELECT')

        for bone_name in Constants.BONE_IDX:
            bone_data = getattr(auto_poser.bones, bone_name)

            if bone_data.bone_mapping_is_valid:
                pose_bone = armature.pose.bones.get(bone_data.mapped_name)
                pose_bone.bone.select = True

        bpy.ops.pose.visual_transform_apply()

        for bone_name in Constants.BONE_IDX:
            bone_data = getattr(auto_poser.bones, bone_name)

            if bone_data.bone_mapping_is_valid:
                pose_bone = armature.pose.bones.get(bone_data.mapped_name)

                pose_bone.keyframe_insert(data_path="location")
                pose_bone.keyframe_insert(data_path="rotation_quaternion" if pose_bone.rotation_mode == 'QUATERNION' else "rotation_euler")
                # pose_bone.keyframe_insert(data_path="scale")

        bpy.ops.object.mode_set(mode=original_mode)
        bpy.context.view_layer.objects.active = original_active_object


keyframes_manager = KeyframesManager()
