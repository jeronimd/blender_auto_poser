import bpy

from ...utils.helper import get_armature_from_object
from ..dependencies import dependencies


class MainPanel(bpy.types.Panel):
    """Main panel for the Auto Poser addon"""
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Auto Poser"
    bl_idname = "AUTO_POSER_PT_MainPanel"
    bl_label = "Auto Poser"
    bl_order = 0

    def draw(self, context):
        layout = self.layout

        if not dependencies.check():
            self._draw_dependency_warning(layout)
            return

        self._draw_navigation_buttons(layout)

        # armature = get_armature_from_object(context.active_object)
        armature = context.scene.objects.get(context.scene.auto_poser_stored_armature_name)

        panel_index = context.scene.auto_poser_chosen_panel
        if not armature and panel_index in [0, 1]:
            layout.label(text="Select an armature", icon='INFO')
            return

        # Draw appropriate panel based on selection
        panel_draws = {
            0: lambda: self._draw_controls(context, armature),
            1: lambda: self._draw_mapping(context, armature),
            2: lambda: self._draw_settings(context, armature)
        }

        if panel_index in panel_draws:
            panel_draws[panel_index]()

    def _draw_dependency_warning(self, layout):
        box = layout.box()
        col = box.column()
        col.operator("auto_poser.install_dependencies", text="Install Dependencies (May take a while.)")
        col.label(text="(Restart Blender after installation.)", icon="ERROR")

    def _draw_navigation_buttons(self, layout):
        row = layout.row(align=True)
        row.scale_y = 1.25
        row.operator("auto_poser.show_points_panel", text="Empties", icon='EMPTY_DATA')
        row.operator("auto_poser.show_mapping_panel", text="Mapping", icon='BONE_DATA')
        row.operator("auto_poser.show_settings_panel", text="Settings", icon='PREFERENCES')

    def _draw_controls(self, context, armature):
        layout = self.layout
        auto_poser = armature.data.auto_poser

        self._draw_header(layout, armature)
        self._draw_keyframes(auto_poser)

        box = layout.box()

        self._draw_empties(layout, box)

        box.separator(type='LINE')

        self._draw_effectors(layout, auto_poser, box)

    def _draw_mapping(self, context, armature):
        layout = self.layout
        auto_poser = armature.data.auto_poser

        self._draw_header(layout, armature)
        self._draw_mapping_controls(layout, auto_poser, armature)

        # Draw bone groups
        body = ["pelvis", "spine001", "spine002", "spine003", "neck", "head", "head_end"]
        left_arm = ["shoulder_l", "upperarm_l", "forearm_l", "hand_l"]
        left_index = ["index01_l", "index02_l", "index03_l", "index_end_l"]
        left_middle = ["middle01_l", "middle02_l", "middle03_l", "middle_end_l"]
        left_ring = ["ring01_l", "ring02_l", "ring03_l", "ring_end_l"]
        left_pinky = ["pinky01_l", "pinky02_l", "pinky03_l", "pinky_end_l"]
        left_thumb = ["thumb01_l", "thumb02_l", "thumb03_l", "thumb_end_l"]
        right_arm = ["shoulder_r", "upperarm_r", "forearm_r", "hand_r"]
        right_index = ["index01_r", "index02_r", "index03_r", "index_end_r"]
        right_middle = ["middle01_r", "middle02_r", "middle03_r", "middle_end_r"]
        right_ring = ["ring01_r", "ring02_r", "ring03_r", "ring_end_r"]
        right_pinky = ["pinky01_r", "pinky02_r", "pinky03_r", "pinky_end_r"]
        right_thumb = ["thumb01_r", "thumb02_r", "thumb03_r", "thumb_end_r"]
        left_leg = ["upperleg_l", "lowerleg_l", "foot_l", "toe_l", "toe_end_l"]
        right_leg = ["upperleg_r", "lowerleg_r", "foot_r", "toe_r", "toe_end_r"]

        self._draw_bone_group(body, "Body", auto_poser, armature)
        self._draw_bone_group(left_arm, "Left Arm", auto_poser, armature)
        self._draw_bone_group(right_arm, "Right Arm", auto_poser, armature)
        self._draw_bone_group(left_leg, "Left Leg", auto_poser, armature)
        self._draw_bone_group(right_leg, "Right Leg", auto_poser, armature)
        self._draw_bone_group(left_index + left_middle + left_ring + left_pinky + left_thumb, "Left Hand", auto_poser, armature)
        self._draw_bone_group(right_index + right_middle + right_ring + right_pinky + right_thumb, "Right Hand", auto_poser, armature)

    def _draw_settings(self, context, armature):
        self._draw_header(self.layout, armature)

        self._draw_poser(context)
        self._draw_settings_empties(context, armature)
        self._draw_rest_pose(context, armature)
        self._draw_settings_dataset(context)
        self._draw_settings_offsets(context)
        # self._draw_settings_constraints(context)

    def _draw_settings_dataset(self, context):
        box = self.layout.box()
        box.label(text="Dataset")

        split = box.split(factor=0.75, align=False)
        split.operator("auto_poser.parse_animations", text="Parse Animations", icon='PRESET')
        split.operator("auto_poser.parse_animation_clear_scene", text="Clear", icon='SCENE_DATA')

    def _draw_settings_offsets(self, context):
        box = self.layout.box()
        box.label(text="Offsets")

        # box.operator("auto_poser.create_default_offsets")
        # box.operator("auto_poser.calculate_offsets")
        box.operator("auto_poser.print_default_offsets")

    def _draw_settings_empties(self, context, armature):
        if not armature:
            return
        auto_poser = armature.data.auto_poser

        box = self.layout.box()
        box.label(text="Empties")

        # split = box.split(factor=0.6, align=False)
        box.prop(auto_poser, "empties_scale")
        # split.prop(auto_poser, "empties_pred_show")

        # box.prop(auto_poser, "empties_scale")
        # box.prop(auto_poser, "empties_pred_show")

        # row = box.row(align=False)
        # # row.alignment = "LEFT"
        # row.prop(auto_poser, "empties_scale")
        # row.alignment = "RIGHT"
        # row.prop(auto_poser, "empties_pred_show")

        box.prop(auto_poser, "update_empties", text="Empties Follow Bones (Laggy)", icon='LINKED')

    def _draw_mapping_controls(self, layout, auto_poser, armature):
        row = layout.row(align=True)
        row.prop(auto_poser, "mapping_name")
        row.operator("auto_poser.add_mapping", text="", icon='IMPORT')
        row.operator("auto_poser.remove_mapping", text="", icon='TRASH')

        row = layout.row()
        # row.operator("auto_poser.load_mapping", text="Load")

        # row.alignment = "LEFT"
        row.operator("auto_poser.calculate_offsets", text="Calculate Offsets", icon='LOOP_FORWARDS')

    def _draw_bone_group(self, bones, title, auto_poser, armature):
        box = self.layout.box()
        box.label(text=title)
        column = box.column(align=True)

        for bone in bones:
            row = column.row(align=True)
            split = row.split(factor=0.3)
            split.label(text=f"{bone}:")
            bone_data = getattr(auto_poser.bones, bone)
            split.prop_search(bone_data, "mapped_name", armature.data, "bones", text="")

    def _draw_empties(self, layout, box):
        # box = layout.box()

        split = box.split(factor=0.85, align=False)
        split.operator("auto_poser.create_empties", text="Create Empties")
        split.operator("auto_poser.delete_empties", text="", icon='TRASH')

    def _draw_effectors(self, layout, auto_poser, box):
        # box = layout.box()
        # box = box.box()

        # Draw each effector type
        self._draw_effector_group(box, "Location", auto_poser.location_effectors,
                                  "auto_poser.add_location_effector",
                                  "auto_poser.remove_location_effector",
                                  "auto_poser.create_simple_location_effectors",
                                  force_first=True)

        self._draw_effector_group(box, "Rotation", auto_poser.rotation_effectors,
                                  "auto_poser.add_rotation_effector",
                                  "auto_poser.remove_rotation_effector")

        self._draw_effector_group(box, "Look-at", auto_poser.lookat_effectors,
                                  "auto_poser.add_lookat_effector",
                                  "auto_poser.remove_lookat_effector",
                                  has_target=True, is_lookat=True)

    def _draw_effector_group(self, layout, title, effectors, add_operator, remove_operator, create_simple=None, has_target=False, force_first=False, is_lookat=False):
        row = layout.row()
        row.label(text=title)
        row.alignment = "RIGHT"
        if create_simple:
            row.operator(create_simple, text="", icon='MOD_ARMATURE')
        row.operator(add_operator, text="", icon='ADD')

        for idx, item in enumerate(effectors):
            row = layout.row()
            row.prop(item, "enabled", text="", icon='MESH_UVSPHERE' if not is_lookat else 'EMPTY_AXIS')
            row.prop(item, "bone", text="")

            # if has_target:
            #     row.prop(item, "target", text="")

            # row.prop(item, "tolerance", text="")

            # if force_first and idx == 0:
            #     row.label(text="", icon='REMOVE')
            #     continue

            remove_op = row.operator(remove_operator, text="", icon='REMOVE')
            remove_op.index = idx

        layout.separator()

    def _draw_settings_constraints(self, context):
        box = self.layout.box()
        box.label(text="Constraints")
        box.operator("auto_poser.create_constraints")
        box.operator("auto_poser.remove_constraints")

    def _draw_header(self, layout, armature):
        if not armature:
            return
        box = layout.box()
        box.label(text=f"Armature: {armature.name}")

    def _draw_poser(self, context):
        box = self.layout.box()
        box.label(text="Poser")

        row = box.row()
        row.prop(context.scene, "auto_poser_model", text="Select Model")

        split = box.split(factor=0.5, align=False)

        split.operator("auto_poser.poser_activate_model", text="Start", icon='PLAY')
        split.operator("auto_poser.poser_deactivate_model", text="Stop", icon='CANCEL')

    def _draw_keyframes(self, auto_poser):
        box = self.layout.box()

        split = box.split(factor=0.75, align=False)
        split.operator("auto_poser.insert_keyframes")
        split.prop(auto_poser, "insert_keyframes_toggle", text="Auto")

        # row = box.row(align=False)
        # # row.alignment = "LEFT"
        # row.operator("auto_poser.insert_keyframes")
        # row.alignment = "RIGHT"
        # row.prop(auto_poser, "insert_keyframes_toggle", text="Auto")

    def _draw_rest_pose(self, context, armature):
        if not armature:
            return
        auto_poser = armature.data.auto_poser

        box = self.layout.box()
        box.label(text="Rest Pose")

        box.prop(auto_poser, "rest_pose_name", text="Select Rest Pose")
        box.operator("auto_poser.change_rest_pose", text="Change Rest Pose to T-Pose", icon='ARMATURE_DATA')


class ShowPointsPanel(bpy.types.Operator):
    bl_idname = "auto_poser.show_points_panel"
    bl_label = "Show Points Panel"

    def execute(self, context):
        context.scene.auto_poser_chosen_panel = 0
        return {'FINISHED'}


class ShowMappingPanel(bpy.types.Operator):
    bl_idname = "auto_poser.show_mapping_panel"
    bl_label = "Show Mapping Panel"

    def execute(self, context):
        context.scene.auto_poser_chosen_panel = 1
        return {'FINISHED'}


class ShowSettingsPanel(bpy.types.Operator):
    bl_idname = "auto_poser.show_settings_panel"
    bl_label = "Show Settings Panel"

    def execute(self, context):
        context.scene.auto_poser_chosen_panel = 2
        return {'FINISHED'}


class InstallDependencies(bpy.types.Operator):
    bl_idname = "auto_poser.install_dependencies"
    bl_label = "Install Dependencies"

    def execute(self, context):
        dependencies.install_dependencies()
        return {"FINISHED"}


classes = [
    MainPanel,
    ShowPointsPanel,
    ShowMappingPanel,
    ShowSettingsPanel,
    InstallDependencies
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.auto_poser_chosen_panel = bpy.props.IntProperty(default=0)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.auto_poser_chosen_panel
