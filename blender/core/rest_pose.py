import json
import math

import bpy

from ...utils.constants import Constants
from ...utils.logger import Logger
from ...utils.path_manager import PathManager


class RestPoseManager:
    def __init__(self):
        self.logger = Logger.setup(__name__, "DEBUG")
        self.path_manager = PathManager()
        self.rest_poses_file_path = self.path_manager.rest_poses_file
        self.rest_poses = self._load_rest_poses()

    def _load_rest_poses(self):
        with open(self.rest_poses_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def change_rest_pose(self, armature):
        auto_poser = armature.data.auto_poser

        original_active_object = bpy.context.view_layer.objects.active
        original_mode = original_active_object.mode

        self._pose_armature(armature, auto_poser)
        self._apply_rest_pose(armature)

        # Restore original mode and active object
        bpy.context.view_layer.objects.active = original_active_object
        bpy.ops.object.mode_set(mode=original_mode)

    def _pose_armature(self, armature, auto_poser):
        self.rest_poses = self._load_rest_poses()

        rest_pose_actions = self.rest_poses.get(auto_poser.rest_pose_name, {})
        if not rest_pose_actions:
            return

        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        # Clear existing pose first
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        bpy.ops.pose.select_all(action='DESELECT')
        # Apply each bone action instructions
        for bone_name in Constants.BONE_IDX:
            bone_data = getattr(auto_poser.bones, bone_name)
            if bone_data.bone_mapping_is_valid:
                bone_actions = rest_pose_actions.get(bone_name, {})
                if not bone_actions:
                    continue

                pose_bone = armature.pose.bones.get(bone_data.mapped_name)
                pose_bone.bone.select = True
                original_rotation_mode = pose_bone.rotation_mode

                pose_bone.rotation_mode = bone_actions["rotation_mode"]
                bone_rotations = bone_actions["rotations"]
                for rotation in bone_rotations:
                    space = rotation["space"]  # "GLOBAL" or "LOCAL"
                    axis = rotation["axis"]
                    degrees = rotation["degrees"]

                    angle = -math.radians(degrees)
                    bpy.ops.transform.rotate(value=angle, orient_axis=axis, orient_type=space)

                pose_bone.rotation_mode = original_rotation_mode
                pose_bone.bone.select = False

    def _apply_rest_pose(self, armature):
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

        # Find all armature modifiers in the armature
        armature_modifiers = []
        for child in armature.children:
            if child.type != 'MESH':
                continue
            for mod_index, modifier in enumerate(child.modifiers):
                if modifier.type == 'ARMATURE' and modifier.object == armature:
                    mod_data = {
                        'mesh': child,
                        'modifier_name': modifier.name,
                        'stack_index': mod_index,
                        'use_vertex_groups': modifier.use_vertex_groups,
                        'use_bone_envelopes': modifier.use_bone_envelopes,
                        'use_deform_preserve_volume': modifier.use_deform_preserve_volume,
                        'use_multi_modifier': modifier.use_multi_modifier,
                        'vertex_group': modifier.vertex_group,
                        'invert_vertex_group': modifier.invert_vertex_group,
                    }
                    armature_modifiers.append(mod_data)

        # Apply the modifiers
        for mesh_data in armature_modifiers:
            mesh = mesh_data['mesh']
            bpy.context.view_layer.objects.active = mesh
            bpy.ops.object.modifier_apply(modifier=mesh_data['modifier_name'])

        # Apply the pose as rest pose
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.armature_apply(selected=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create new armature modifiers
        for mesh_data in armature_modifiers:
            mesh = mesh_data['mesh']
            bpy.context.view_layer.objects.active = mesh

            # Create new armature modifier
            new_modifier = mesh.modifiers.new(name=mesh_data['modifier_name'], type='ARMATURE')
            new_modifier.object = armature
            new_modifier.use_vertex_groups = mesh_data['use_vertex_groups']
            new_modifier.use_bone_envelopes = mesh_data['use_bone_envelopes']
            new_modifier.use_deform_preserve_volume = mesh_data['use_deform_preserve_volume']
            new_modifier.use_multi_modifier = mesh_data['use_multi_modifier']
            new_modifier.vertex_group = mesh_data['vertex_group']
            new_modifier.invert_vertex_group = mesh_data['invert_vertex_group']
            bpy.ops.object.modifier_move_to_index(modifier=new_modifier.name, index=mesh_data['stack_index'])


rest_pose_manager = RestPoseManager()
