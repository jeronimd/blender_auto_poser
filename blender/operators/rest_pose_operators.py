import bpy

from ...utils.helper import get_armature_from_object
from ..core.rest_pose import rest_pose_manager


class ChangeRestPose(bpy.types.Operator):
    bl_idname = "auto_poser.change_rest_pose"
    bl_label = "Change Rest Pose"
    bl_description = "This is a hacky way of doing it so don't expect it much from it."

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        rest_pose_manager.change_rest_pose(armature)
        return {'FINISHED'}


classes = [
    ChangeRestPose,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
