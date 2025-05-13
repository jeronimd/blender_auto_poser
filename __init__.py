# Blender Auto Poser
# A Blender addon that uses AI to predict poses from keypoints
# Based on ProtoRes: Proto-Residual Network for Pose Authoring via Learned Inverse Kinematics
# https://arxiv.org/abs/2106.01981
# Original paper by Boris N. Oreshkin, Florent Bocquelet, Félix G. Harvey, Bay Raitt, Dominic Laflamme
# ProtoRes is licensed for non-commercial academic research purposes only


from .blender.dependencies import dependencies
from .blender.panels import main_panel

bl_info = {
    "name": "Auto Poser",
    "author": "David Jerónimo Rodrigues",
    "version": (0, 1, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > Auto Poser",
    "description": "AI-powered pose prediction from keypoints based on ProtoRes",
    "warning": "ProtoRes is licensed for non-commercial academic research purposes only",
    "doc_url": "https://arxiv.org/abs/2106.01981",
    "category": "Animation",
}

if dependencies.check():
    from .blender import armature
    from .blender.operators import (
        dataset_operators,
        effector_operators,
        empties_operators,
        mappings_operators,
        offsets_operators,
        keyframes_operators,
        poser_operators,
        rest_pose_operators,
    )

def register():
    main_panel.register()
    if dependencies.check():
        armature.register()
        dataset_operators.register()
        offsets_operators.register()
        mappings_operators.register()
        empties_operators.register()
        poser_operators.register()
        effector_operators.register()
        keyframes_operators.register()
        rest_pose_operators.register()

def unregister():
    main_panel.unregister()
    if dependencies.check():
        armature.unregister()
        dataset_operators.unregister()
        offsets_operators.unregister()
        mappings_operators.unregister()
        empties_operators.unregister()
        poser_operators.unregister()
        effector_operators.unregister()
        keyframes_operators.unregister()
        rest_pose_operators.unregister()

if __name__ == "__main__":
    register()
