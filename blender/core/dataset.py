import contextlib
import io
from pathlib import Path

import bpy
import msgpack

from ...utils.constants import Constants
from ...utils.logger import Logger
from ...utils.path_manager import PathManager


class DatasetManager:
    def __init__(self):
        self.logger = Logger.setup(__name__, "DEBUG")
        self.path_manager = PathManager()

    def clear_scene(self):
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        for collection in bpy.data.collections:
            bpy.data.collections.remove(collection, do_unlink=True)

        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block, do_unlink=True)
        for block in bpy.data.materials:
            bpy.data.materials.remove(block, do_unlink=True)
        for block in bpy.data.textures:
            bpy.data.textures.remove(block, do_unlink=True)
        for block in bpy.data.images:
            bpy.data.images.remove(block, do_unlink=True)
        for block in bpy.data.armatures:
            bpy.data.armatures.remove(block, do_unlink=True)

    def process_animations(self):
        self.logger.info("Processing animations")
        self.clear_scene()

        animations_list = self._get_animations_list()
        dataset = {}
        for animation_path in animations_list:
            dataset.update(self._process_animation(animation_path))
            if animation_path != animations_list[-1]:
                self.clear_scene()  # Don't clear scene after processing the last animation

        self._save_dataset(dataset)

    def _process_animation(self, animation_path):
        self.logger.debug(f"Processing animation: {self._get_animation_name(animation_path)}")
        self._load_animation(animation_path)

        armature = self._get_armature()
        armature = self._fix_armature(armature)

        anim_name = self._get_animation_name(animation_path)
        animation_timeline = self._get_animation_timeline(armature)

        anim_dataset = {anim_name: {}}
        for frame in range(*animation_timeline):
            bpy.context.scene.frame_set(frame)
            anim_dataset[anim_name][f"frame_{frame}"] = self._get_pose(armature)
        return anim_dataset

    def _get_pose(self, armature):
        bone_data = {}
        for default_name, mapped_name in Constants.BONE_MAPPINGS.items():
            bone = armature.pose.bones.get(mapped_name)

            # Compute the bone's local rotation in global space coordinate system.
            local_quat = bone.matrix_basis.to_quaternion()
            rest_orientation = bone.bone.matrix_local.to_quaternion()
            world_rotation_aligned = rest_orientation @ local_quat @ rest_orientation.inverted()

            bone_data[default_name] = {
                "location": list(bone.head),
                "rotation": list(world_rotation_aligned)
            }
        return bone_data

    def _get_animations_list(self):
        path = Path(self.path_manager.animations_dir)
        return list(path.glob('**/*.fbx'))

    def _get_animation_name(self, file_path):
        parts = file_path.parts[-2:]  # Get folder and file name
        return f"{parts[0]}/{parts[1][:-4]}"

    def _load_animation(self, file_path):
        with contextlib.redirect_stdout(io.StringIO()):
            bpy.ops.import_scene.fbx(filepath=str(file_path))

    def _get_animation_timeline(self, armature):
        keyframes = {int(k.co[0]) for fc in armature.animation_data.action.fcurves for k in fc.keyframe_points}
        frames = sorted(list(keyframes))
        return int(frames[0]), int(frames[-1]) + 1

    def _get_armature(self):
        bpy.ops.object.select_all(action='DESELECT')
        armature = next((obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE'), None)
        armature.select_set(True)
        return armature

    def _save_dataset(self, dataset):
        self.logger.warning("Saving Dataset")
        try:
            output_path = Path(self.path_manager.parsed_animations_file)
            with open(output_path, 'wb') as f:
                msgpack.pack(dataset, f)
            self.logger.info("Dataset saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {e}")

    def _fix_armature(self, armature):
        # apply armature scale
        if armature.scale != (1, 1, 1):
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            for fcurve in armature.animation_data.action.fcurves:
                if Constants.BONE_MAPPINGS[Constants.ROOT_BONE] in fcurve.data_path:
                    for keyframe in fcurve.keyframe_points:
                        if 'location' in fcurve.data_path:
                            keyframe.co[1] *= 0.01

        # apply armature rotation
        if armature.rotation_euler != (0, 0, 0):
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        return armature


dataset_manager = DatasetManager()
