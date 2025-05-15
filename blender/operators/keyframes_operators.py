import bpy

from ...utils.helper import get_armature_from_object
from ..core.keyframes import keyframes_manager


class InsertKeyframes(bpy.types.Operator):
    bl_idname = "auto_poser.insert_keyframes"
    bl_label = "Insert Keyframes"

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        keyframes_manager.insert_keyframes(armature)
        return {'FINISHED'}


classes = [
    InsertKeyframes,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
