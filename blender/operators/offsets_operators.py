import bpy

from ...utils.helper import get_armature_from_object
from ..core.offsets import offsets_manager


class CalculateOffsets(bpy.types.Operator):
    bl_idname = "auto_poser.calculate_offsets"
    bl_label = "Calculate Offsets"

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        offsets_manager.calculate_offsets(armature)
        self.report({'INFO'}, "Offsets calculated successfully")
        return {'FINISHED'}


class PrintDefaultOffsets(bpy.types.Operator):
    bl_idname = "auto_poser.print_default_offsets"
    bl_label = "Print Default Offsets"

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        offsets_manager.print_default_offsets(armature)
        return {'FINISHED'}


classes = [
    CalculateOffsets,
    PrintDefaultOffsets
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
