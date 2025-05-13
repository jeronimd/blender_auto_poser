import os


class PathManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.data_dir = os.path.join(self.base_dir, "data")
        self.animations_dir = os.path.join(self.data_dir, "animations")
        self.parsed_animations_file = os.path.join(self.data_dir, "parsed_animations.msgpack")

        self.models_path = os.path.join(self.base_dir, "poser", "models")

        self.remaps_file = os.path.join(self.base_dir, "blender", "configs", "remaps.json")
        self.rest_poses_file = os.path.join(self.base_dir, "blender", "configs", "rest_poses.json")
