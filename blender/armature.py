import bpy

from ..blender.core.mappings import mappings_manager
from ..blender.core.rest_pose import rest_pose_manager
from ..utils.constants import Constants
from ..utils.helper import format_items, get_armature_from_object

_msgbus_owner = object()


class BoneData(bpy.types.PropertyGroup):
    def _update_bone_mapping_is_valid(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.objects.get(self.id_data.name)
        self.bone_mapping_is_valid = self.mapped_name in armature.pose.bones

    mapped_name: bpy.props.StringProperty(name="Mapped Name", default="", update=_update_bone_mapping_is_valid)
    bone_mapping_is_valid: bpy.props.BoolProperty(name="Mapping is Valid", default=False)

    empty_key: bpy.props.PointerProperty(name="Empty Key", type=bpy.types.Object)

    offset_local_location: bpy.props.FloatVectorProperty(name="Offset Local Location", size=3)
    offset_local_rotation: bpy.props.FloatVectorProperty(name="Offset Local Rotation", size=4)

    offset_global_location: bpy.props.FloatVectorProperty(name="Offset Location", size=3)
    offset_global_rotation: bpy.props.FloatVectorProperty(name="Offset Rotation", size=4)

    def is_active(self, effector_types=None):
        if effector_types is None:
            effector_types = ['location', 'rotation', 'lookat']

        # Get the armature (parent) property group
        armature = self.id_data.auto_poser

        # Map effector type names to collections
        effector_map = {
            'location': armature.location_effectors,
            'rotation': armature.rotation_effectors,
            'lookat': armature.lookat_effectors
        }

        # Get the bone name by finding which property this BoneData belongs to
        for prop_name in armature.bones.__annotations__:
            if getattr(armature.bones, prop_name) == self:
                bone_name = prop_name
                break
        else:
            return False

        # Check only the specified effector types
        for eff_type in effector_types:
            if eff_type not in effector_map:
                continue
            collection = effector_map[eff_type]
            for effector in collection:
                if effector.bone == bone_name:  # and effector.enabled:
                    return True

        return False


class BoneProperties(bpy.types.PropertyGroup):
    # Core body
    pelvis: bpy.props.PointerProperty(type=BoneData)
    spine001: bpy.props.PointerProperty(type=BoneData)
    spine002: bpy.props.PointerProperty(type=BoneData)
    spine003: bpy.props.PointerProperty(type=BoneData)
    neck: bpy.props.PointerProperty(type=BoneData)
    head: bpy.props.PointerProperty(type=BoneData)
    head_end: bpy.props.PointerProperty(type=BoneData)

    # Left arm chain
    shoulder_l: bpy.props.PointerProperty(type=BoneData)
    upperarm_l: bpy.props.PointerProperty(type=BoneData)
    forearm_l: bpy.props.PointerProperty(type=BoneData)
    hand_l: bpy.props.PointerProperty(type=BoneData)

    # Left hand fingers
    index01_l: bpy.props.PointerProperty(type=BoneData)
    index02_l: bpy.props.PointerProperty(type=BoneData)
    index03_l: bpy.props.PointerProperty(type=BoneData)
    index_end_l: bpy.props.PointerProperty(type=BoneData)

    middle01_l: bpy.props.PointerProperty(type=BoneData)
    middle02_l: bpy.props.PointerProperty(type=BoneData)
    middle03_l: bpy.props.PointerProperty(type=BoneData)
    middle_end_l: bpy.props.PointerProperty(type=BoneData)

    ring01_l: bpy.props.PointerProperty(type=BoneData)
    ring02_l: bpy.props.PointerProperty(type=BoneData)
    ring03_l: bpy.props.PointerProperty(type=BoneData)
    ring_end_l: bpy.props.PointerProperty(type=BoneData)

    pinky01_l: bpy.props.PointerProperty(type=BoneData)
    pinky02_l: bpy.props.PointerProperty(type=BoneData)
    pinky03_l: bpy.props.PointerProperty(type=BoneData)
    pinky_end_l: bpy.props.PointerProperty(type=BoneData)

    thumb01_l: bpy.props.PointerProperty(type=BoneData)
    thumb02_l: bpy.props.PointerProperty(type=BoneData)
    thumb03_l: bpy.props.PointerProperty(type=BoneData)
    thumb_end_l: bpy.props.PointerProperty(type=BoneData)

    # Right arm chain
    shoulder_r: bpy.props.PointerProperty(type=BoneData)
    upperarm_r: bpy.props.PointerProperty(type=BoneData)
    forearm_r: bpy.props.PointerProperty(type=BoneData)
    hand_r: bpy.props.PointerProperty(type=BoneData)

    # Right hand fingers
    index01_r: bpy.props.PointerProperty(type=BoneData)
    index02_r: bpy.props.PointerProperty(type=BoneData)
    index03_r: bpy.props.PointerProperty(type=BoneData)
    index_end_r: bpy.props.PointerProperty(type=BoneData)

    middle01_r: bpy.props.PointerProperty(type=BoneData)
    middle02_r: bpy.props.PointerProperty(type=BoneData)
    middle03_r: bpy.props.PointerProperty(type=BoneData)
    middle_end_r: bpy.props.PointerProperty(type=BoneData)

    ring01_r: bpy.props.PointerProperty(type=BoneData)
    ring02_r: bpy.props.PointerProperty(type=BoneData)
    ring03_r: bpy.props.PointerProperty(type=BoneData)
    ring_end_r: bpy.props.PointerProperty(type=BoneData)

    pinky01_r: bpy.props.PointerProperty(type=BoneData)
    pinky02_r: bpy.props.PointerProperty(type=BoneData)
    pinky03_r: bpy.props.PointerProperty(type=BoneData)
    pinky_end_r: bpy.props.PointerProperty(type=BoneData)

    thumb01_r: bpy.props.PointerProperty(type=BoneData)
    thumb02_r: bpy.props.PointerProperty(type=BoneData)
    thumb03_r: bpy.props.PointerProperty(type=BoneData)
    thumb_end_r: bpy.props.PointerProperty(type=BoneData)

    # Left leg chain
    upperleg_l: bpy.props.PointerProperty(type=BoneData)
    lowerleg_l: bpy.props.PointerProperty(type=BoneData)
    foot_l: bpy.props.PointerProperty(type=BoneData)
    toe_l: bpy.props.PointerProperty(type=BoneData)
    toe_end_l: bpy.props.PointerProperty(type=BoneData)

    # Right leg chain
    upperleg_r: bpy.props.PointerProperty(type=BoneData)
    lowerleg_r: bpy.props.PointerProperty(type=BoneData)
    foot_r: bpy.props.PointerProperty(type=BoneData)
    toe_r: bpy.props.PointerProperty(type=BoneData)
    toe_end_r: bpy.props.PointerProperty(type=BoneData)


class EffectorItem(bpy.types.PropertyGroup):
    def _update_empties(self, context):
        # armature = get_armature_from_object(context.active_object)
        armature = context.scene.objects.get(self.id_data.name)
        if not armature:
            return

        bpy.ops.auto_poser.create_empties()

    ordered_bones = [
        'pelvis', 'spine001', 'spine002', 'spine003', 'neck', 'head',
        'shoulder_l', 'shoulder_r', 'upperarm_l', 'upperarm_r', 'forearm_l', 'forearm_r', 'hand_l', 'hand_r',
        'upperleg_l', 'upperleg_r', 'lowerleg_l', 'lowerleg_r', 'foot_l', 'foot_r', 'toe_l', 'toe_r',
    ]

    enabled: bpy.props.BoolProperty(name="Enabled", default=True)  # , update=_update_empties)
    bone: bpy.props.EnumProperty(name="Bone", items=format_items(ordered_bones), update=_update_empties)
    target: bpy.props.PointerProperty(name="Target", update=_update_empties, type=bpy.types.Object, poll=lambda self, obj: obj.type == 'EMPTY')
    tolerance: bpy.props.FloatProperty(name="Tolerance", default=0.0)


class ArmatureProperties(bpy.types.PropertyGroup):
    bones: bpy.props.PointerProperty(type=BoneProperties)

    location_effectors: bpy.props.CollectionProperty(type=EffectorItem)
    rotation_effectors: bpy.props.CollectionProperty(type=EffectorItem)
    lookat_effectors: bpy.props.CollectionProperty(type=EffectorItem)

    def _get_formatted_mapping_items(self, context):
        return format_items(list(mappings_manager.mappings))  # , add_empty=False)

    def _load_mappings_wrapper(self, context):
        armature = get_armature_from_object(context.active_object)
        # armature = context.scene.objects.get(self.id_data.name)
        mappings_manager.load_mapping(armature, self.mapping_name)

    mapping_name: bpy.props.EnumProperty(name="Mapping", items=_get_formatted_mapping_items, update=_load_mappings_wrapper)
    empties_scale: bpy.props.FloatProperty(name="Empties Scale", default=1.0, min=0.0)
    insert_keyframes_toggle: bpy.props.BoolProperty(name="Insert Keyframes", default=False)

    def _get_formatted_rest_pose_items(self, context):
        return format_items(list(rest_pose_manager.rest_poses))  # , add_empty=False)

    rest_pose_name: bpy.props.EnumProperty(name="Rest Pose", items=_get_formatted_rest_pose_items)
    update_empties: bpy.props.BoolProperty(name="Update Empties", default=True)


classes = [
    BoneData,
    BoneProperties,
    EffectorItem,
    ArmatureProperties,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Armature.auto_poser = bpy.props.PointerProperty(type=ArmatureProperties)

    bpy.types.Scene.auto_poser_stored_armature = bpy.props.PointerProperty(type=bpy.types.Object)
    bpy.types.Scene.auto_poser_stored_armature_name = bpy.props.StringProperty(name="Stored Armature Name", default="")

    def auto_poser_on_selection_change(*args):  # TODO: I need to reloadAddons every time I open blender, I think this is a debug problem and the packed version wont have it.
        active_object = bpy.context.active_object
        current_scene = bpy.context.scene
        armature = get_armature_from_object(active_object)
        if armature:
            current_scene.auto_poser_stored_armature = armature
            current_scene.auto_poser_stored_armature_name = armature.name

    # subscribe_to = bpy.types.LayerObjects, "active"
    # bpy.msgbus.subscribe_rna(key=subscribe_to, owner=_msgbus_owner, args=(), notify=auto_poser_on_selection_change)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Armature.auto_poser

    del bpy.types.Scene.auto_poser_stored_armature
    del bpy.types.Scene.auto_poser_stored_armature_name

    # bpy.msgbus.clear_by_owner(_msgbus_owner)
