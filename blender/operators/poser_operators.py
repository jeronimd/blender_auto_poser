import bpy

from ...utils.helper import format_items, get_armature_from_object
from ..core.empties import empties_manager
from ..core.poser import poser_manager


class ActivateModel(bpy.types.Operator):
    bl_idname = "auto_poser.poser_activate_model"
    bl_label = "poser_activate_model"

    @classmethod
    def poll(cls, context):
        return not poser_manager.is_model_active or auto_poser_on_scene_update not in bpy.app.handlers.depsgraph_update_post

    def execute(self, context):
        poser_manager.load_model(context.scene.auto_poser_model)

        if auto_poser_on_scene_update not in bpy.app.handlers.depsgraph_update_post and poser_manager.is_model_active:
            bpy.app.handlers.depsgraph_update_post.append(auto_poser_on_scene_update)

        if auto_poser_on_timeline_update not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(auto_poser_on_timeline_update)
        return {"FINISHED"}


class DeactivateModel(bpy.types.Operator):
    bl_idname = "auto_poser.poser_deactivate_model"
    bl_label = "poser_deactivate_model"

    @classmethod
    def poll(cls, context):
        return poser_manager.is_model_active or auto_poser_on_scene_update in bpy.app.handlers.depsgraph_update_post

    def execute(self, context):
        poser_manager.unload_model()

        # Remove all handlers containing 'auto_poser_on_scene_update'
        for handler in list(bpy.app.handlers.depsgraph_update_post):
            if 'auto_poser_on_scene_update' in str(handler):
                bpy.app.handlers.depsgraph_update_post.remove(handler)

        # Remove all handlers containing 'auto_poser_on_timeline_update'
        for handler in list(bpy.app.handlers.frame_change_post):
            if 'auto_poser_on_timeline_update' in str(handler):
                bpy.app.handlers.frame_change_post.remove(handler)
        return {"FINISHED"}


classes = [
    ActivateModel,
    DeactivateModel,
]


def auto_poser_on_scene_update(*args):
    active_object = bpy.context.active_object

    if not active_object:
        return

    # Only accept empties to make it more efficient
    if active_object.type != 'EMPTY':
        return

    # Make sure the empty is from AutoPoser, this way the user can still use the empties for their armature
    if not active_object.name.startswith(f"{empties_manager.prefix}_"):
        return

    # armature = get_armature_from_object(active_object)
    armature = bpy.context.scene.auto_poser_stored_armature
    if not armature:
        return

    if empties_manager.handle_empty_update(active_object):
        poser_manager.predict(armature)


def auto_poser_on_timeline_update(*args):
    active_object = bpy.context.active_object

    if not active_object:
        return

    # armature = get_armature_from_object(active_object)
    armature = bpy.context.scene.auto_poser_stored_armature
    if not armature:
        return

    empties_manager.move_empties(armature)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    def get_formatted_models_items(self, context):
        return format_items(list(poser_manager.get_models()), add_empty=False)

    def load_model_wrapper(self, context):
        if poser_manager.is_model_active:
            poser_manager.load_model(self.auto_poser_model)

    bpy.types.Scene.auto_poser_model = bpy.props.EnumProperty(name="Models", items=get_formatted_models_items, update=load_model_wrapper)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.auto_poser_model

    poser_manager.unload_model()

    # Remove all handlers containing 'auto_poser_on_scene_update'
    for handler in list(bpy.app.handlers.depsgraph_update_post):
        if 'auto_poser_on_scene_update' in str(handler):
            bpy.app.handlers.depsgraph_update_post.remove(handler)

    # Remove all handlers containing 'auto_poser_on_timeline_update'
    for handler in list(bpy.app.handlers.frame_change_post):
        if 'auto_poser_on_timeline_update' in str(handler):
            bpy.app.handlers.frame_change_post.remove(handler)
