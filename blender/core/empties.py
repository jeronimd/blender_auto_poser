import bpy

from ...utils.constants import Constants


class EmptiesManager:
    def __init__(self):
        self.prefix = "AP"
        self.last_empty_state = {"location": None, "rotation": None}
        self.update_threshold = {"location": 0.01, "rotation": 0.1}  # Minimum distance/angle to update empty

    def handle_empty_update(self, empty):
        current_pos = empty.location
        current_rot = empty.rotation_quaternion

        if self.last_empty_state["location"] is None:
            self.last_empty_state["location"] = current_pos.copy()
            self.last_empty_state["rotation"] = current_rot.copy()
            return False

        position_diff = (current_pos - self.last_empty_state["location"]).length
        rotation_diff = current_rot.rotation_difference(self.last_empty_state["rotation"]).angle

        if position_diff > self.update_threshold["location"] or rotation_diff > self.update_threshold["rotation"]:
            self.last_empty_state["location"] = current_pos.copy()
            self.last_empty_state["rotation"] = current_rot.copy()
            return True

        return False

    def create_empties(self, armature):
        auto_poser = armature.data.auto_poser

        main_collection = self._create_collection("Auto Poser Empties")
        armature_collection = self._create_collection(f"{self.prefix}_{armature.name}_empties", main_collection)
        key_collection = self._create_collection(f"{self.prefix}_{armature.name}_key_empties", armature_collection)

        for bone in Constants.BONE_IDX:
            bone_data = getattr(auto_poser.bones, bone)

            if bone_data.is_active(['location', 'rotation']):
                if bone_data.empty_key is None:
                    bone_data.empty_key = self._create_empty(
                        f"{self.prefix}_{armature.name}_{bone}_key",
                        armature,
                        bone_data,
                        key_collection,
                        size=0.1 if bone in Constants.BONES_BODY else 0.01,
                    )
            elif bone_data.empty_key:
                bpy.data.objects.remove(bone_data.empty_key, do_unlink=True)
                bone_data.empty_key = None
                bpy.context.view_layer.objects.active = armature

        for effector in auto_poser.lookat_effectors:
            bone_name = getattr(effector, "bone")
            if bone_name != "NONE":
                if effector.target is None:
                    effector.target = self._create_empty(
                        f"{self.prefix}_{armature.name}_{effector.bone}_target",
                        armature,
                        getattr(auto_poser.bones, bone_name),
                        key_collection,
                        'PLAIN_AXES',
                        size=0.15 if effector.bone in Constants.BONES_BODY else 0.015,
                    )

    def delete_empties(self, armature):
        bpy.context.view_layer.objects.active = armature

        for bone in Constants.BONE_IDX:
            bone_data = getattr(armature.data.auto_poser.bones, bone)
            if bone_data.empty_key:
                bpy.data.objects.remove(bone_data.empty_key, do_unlink=True)
                bone_data.empty_key = None

        for effector in armature.data.auto_poser.lookat_effectors:
            if effector.target:
                bpy.data.objects.remove(effector.target, do_unlink=True)
                effector.target = None

        self._cleanup_collections(armature)

    def _create_empty(self, name, armature, bone_data, collection, display_type='SPHERE',
                      size=0.1, show_in_front=True, is_selectable=True):
        auto_poser = armature.data.auto_poser

        empty = bpy.data.objects.new(name=name, object_data=None)
        empty.empty_display_type = display_type
        empty.empty_display_size = size * auto_poser.empties_scale
        empty.hide_select = not is_selectable
        empty.show_in_front = show_in_front
        empty.rotation_mode = 'QUATERNION'

        if bone_data.bone_mapping_is_valid:
            bone_pose = armature.pose.bones.get(bone_data.mapped_name)
            local_quat = bone_pose.matrix_basis.to_quaternion()
            rest_orientation = bone_pose.bone.matrix_local.to_quaternion()
            rotation = rest_orientation @ local_quat @ rest_orientation.inverted()

            empty.rotation_quaternion = rotation
            empty.location = bone_pose.head

        collection.objects.link(empty)
        empty.parent = armature
        return empty

    def _create_collection(self, name, parent=None):
        collection = bpy.data.collections.get(name)
        if not collection:
            collection = bpy.data.collections.new(name)
            if parent:
                parent.children.link(collection)
            else:
                bpy.context.scene.collection.children.link(collection)
        return collection

    def _cleanup_collections(self, armature):
        main_collection = bpy.data.collections.get("Auto Poser Empties")
        if not main_collection:
            return

        armature_collection = bpy.data.collections.get(f"{self.prefix}_{armature.name}_empties")
        if armature_collection:
            for child_collection in armature_collection.children:
                for obj in child_collection.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(child_collection)
            bpy.data.collections.remove(armature_collection)
            if len(main_collection.children) == 0:
                bpy.data.collections.remove(main_collection)

    def move_empties(self, armature):
        auto_poser = armature.data.auto_poser

        if not auto_poser.update_empties:
            return

        for bone in Constants.BONE_IDX:
            bone_data = getattr(auto_poser.bones, bone)

            if bone_data.is_active(['location', 'rotation']):
                if bone_data.empty_key:
                    bone_pose = armature.pose.bones.get(bone_data.mapped_name)
                    local_quat = bone_pose.matrix_basis.to_quaternion()
                    rest_orientation = bone_pose.bone.matrix_local.to_quaternion()
                    rotation = rest_orientation @ local_quat @ rest_orientation.inverted()

                    bone_data.empty_key.rotation_quaternion = rotation
                    bone_data.empty_key.location = bone_pose.head


empties_manager = EmptiesManager()
