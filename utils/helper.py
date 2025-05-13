import bpy

from ..utils.constants import Constants


def format_items(items, description="", add_empty=True):
    list_items = []
    if not items:
        return list_items
    if add_empty:  # Add empty option at the start
        list_items = [("NONE", "", "")]
    list_items.extend([(item, item, "") for item in items])
    # list_items.extend([(item.upper(), item, "") for item in items])
    return list_items


def get_armature_from_object(obj):
    if not obj:
        return None

    if obj.type == 'ARMATURE':
        if hasattr(obj.data, 'auto_poser'):
            return obj
        else:
            return None

    if obj.parent and obj.parent.type == 'ARMATURE':
        if hasattr(obj.parent.data, 'auto_poser'):
            return obj.parent
        else:
            return None

    return None
