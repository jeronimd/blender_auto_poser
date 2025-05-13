import bpy

from ..core.empties import empties_manager


class CreateEmptiesOperator(bpy.types.Operator):
    bl_idname = "auto_poser.create_empties"
    bl_label = "Create Empties"
    bl_description = "Create key and predicted empties for selected bones"

    def execute(self, context):
        armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        empties_manager.create_empties(armature)
        return {'FINISHED'}


class DeleteEmptiesOperator(bpy.types.Operator):
    bl_idname = "auto_poser.delete_empties"
    bl_label = "Delete Empties"
    bl_description = "Delete all empties for the selected armature"

    def execute(self, context):
        armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        empties_manager.delete_empties(armature)
        return {'FINISHED'}


classes = [
    CreateEmptiesOperator,
    DeleteEmptiesOperator
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
