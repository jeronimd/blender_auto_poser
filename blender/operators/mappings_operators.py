import bpy

from ..core.mappings import mappings_manager
from ...utils.helper import get_armature_from_object


class AddMapping(bpy.types.Operator):
    bl_idname = "auto_poser.add_mapping"
    bl_label = "Add Mapping"

    mapping_name: bpy.props.StringProperty(name="Mapping Name", default="New Mapping")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "mapping_name")

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        if mappings_manager.create_mapping(armature, self.mapping_name):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            self.report({'INFO'}, f"Mapping '{self.mapping_name}' saved")
            return {'FINISHED'}

        return {'CANCELLED'}


class RemoveMapping(bpy.types.Operator):
    bl_idname = "auto_poser.remove_mapping"
    bl_label = "Remove Mapping"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.auto_poser_stored_armature
        if not armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        auto_poser = armature.data.auto_poser

        if not auto_poser.mapping_name:
            self.report({'ERROR'}, "No mapping selected")
            return {'CANCELLED'}

        if mappings_manager.remove_mapping(auto_poser.mapping_name):
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            self.report({'INFO'}, "Mapping removed")
            return {'FINISHED'}

        return {'CANCELLED'}


classes = [
    AddMapping,
    RemoveMapping,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
