import bpy

from ..core.dataset import dataset_manager


class ParseAnimations(bpy.types.Operator):
    bl_idname = "auto_poser.parse_animations"
    bl_label = "parse_animations"

    def execute(self, context):
        if context.scene.objects:
            self.report({"ERROR"}, "Scene is not empty")
            return {"CANCELLED"}

        dataset_manager.process_animations()
        return {"FINISHED"}


class ParseAnimationClearScene(bpy.types.Operator):
    bl_idname = "auto_poser.parse_animation_clear_scene"
    bl_label = "parse_animation_clear_scene"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event, title="Confirm Clear Scene")

    def execute(self, context):
        dataset_manager.clear_scene()
        return {"FINISHED"}




classes = [
    ParseAnimations,
    ParseAnimationClearScene,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
