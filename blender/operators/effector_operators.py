import bpy

from ...utils.helper import get_armature_from_object


class AddLocationEffector(bpy.types.Operator):
    bl_idname = "auto_poser.add_location_effector"
    bl_label = "Add Location Effector"
    bone_name: bpy.props.StringProperty()

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser
        auto_poser.location_effectors.add()
        bpy.ops.auto_poser.create_empties()
        return {'FINISHED'}


class RemoveLocationEffector(bpy.types.Operator):
    bl_idname = "auto_poser.remove_location_effector"
    bl_label = "Remove Location Effector"
    bone_name: bpy.props.StringProperty()
    index: bpy.props.IntProperty()

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser
        auto_poser.location_effectors.remove(self.index)
        bpy.ops.auto_poser.create_empties()
        return {'FINISHED'}


class AddRotationEffector(bpy.types.Operator):
    bl_idname = "auto_poser.add_rotation_effector"
    bl_label = "Add Rotation Effector"
    bone_name: bpy.props.StringProperty()

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser
        auto_poser.rotation_effectors.add()
        bpy.ops.auto_poser.create_empties()
        return {'FINISHED'}


class RemoveRotationEffector(bpy.types.Operator):
    bl_idname = "auto_poser.remove_rotation_effector"
    bl_label = "Remove Rotation Effector"
    bone_name: bpy.props.StringProperty()
    index: bpy.props.IntProperty()

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser
        auto_poser.rotation_effectors.remove(self.index)
        bpy.ops.auto_poser.create_empties()
        return {'FINISHED'}


class AddLookAtEffector(bpy.types.Operator):
    bl_idname = "auto_poser.add_lookat_effector"
    bl_label = "Add Look-at Effector"
    bone_name: bpy.props.StringProperty()

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser
        auto_poser.lookat_effectors.add()
        bpy.ops.auto_poser.create_empties()
        return {'FINISHED'}


class RemoveLookAtEffector(bpy.types.Operator):
    bl_idname = "auto_poser.remove_lookat_effector"
    bl_label = "Remove Look-at Effector"
    bone_name: bpy.props.StringProperty()
    index: bpy.props.IntProperty()

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser

        lookat_effector = auto_poser.lookat_effectors[self.index]
        bpy.data.objects.remove(lookat_effector.target, do_unlink=True)
        auto_poser.lookat_effectors.remove(self.index)
        bpy.ops.auto_poser.create_empties()
        return {'FINISHED'}


class CreateSimpleLocationEffectors(bpy.types.Operator):
    bl_idname = "auto_poser.create_simple_location_effectors"
    bl_label = "Create Simple Location Effectors"
    bone_name: bpy.props.StringProperty()
    index: bpy.props.IntProperty()

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser
        auto_poser.location_effectors.clear()
        for bone_name in ['pelvis', 'head', 'hand_l', 'hand_r', 'foot_l', 'foot_r']:
            effector = auto_poser.location_effectors.add()
            effector.bone = bone_name

        return {'FINISHED'}


classes = [
    AddLocationEffector,
    RemoveLocationEffector,
    AddRotationEffector,
    RemoveRotationEffector,
    AddLookAtEffector,
    RemoveLookAtEffector,
    CreateSimpleLocationEffectors
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
