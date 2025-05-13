import json

from ...utils.constants import Constants
from ...utils.logger import Logger
from ...utils.path_manager import PathManager


class MappingsManager:
    def __init__(self):
        self.logger = Logger.setup(__name__, "DEBUG")
        self.path_manager = PathManager()
        self.remaps_file_path = self.path_manager.remaps_file
        self.mappings = self._load_mappings()

    def _load_mappings(self):
        with open(self.remaps_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _write_mappings(self):
        with open(self.remaps_file_path, 'w', encoding='utf-8') as file:
            json.dump(self.mappings, file, indent=2)

    def create_mapping(self, armature, mapping_name):
        auto_poser = armature.data.auto_poser

        # Get remapped names for each bone
        mapping_data = {}
        for bone in Constants.BONE_IDX:
            mapped_name = getattr(auto_poser.bones, bone).mapped_name
            if mapped_name:
                mapping_data[bone] = mapped_name

        # Save mapping
        self.mappings[mapping_name] = mapping_data
        self._write_mappings()

        self.logger.info(f"Mapping '{mapping_name}' saved successfully")
        return True

    def remove_mapping(self, name):
        if name not in self.mappings:
            self.logger.warning(f"Mapping '{name}' not found")
            return False

        del self.mappings[name]
        self._write_mappings()

        self.logger.info(f"Mapping '{name}' removed successfully")
        return True

    def load_mapping(self, armature, mapping_name):
        auto_poser = armature.data.auto_poser
        mapping_data = self.mappings[mapping_name]

        if not mapping_data:
            self.logger.error(f"Mapping '{mapping_name}' not found")
            return False

        for bone in Constants.BONE_IDX:
            mapped_bone = mapping_data.get(bone, "")
            bone_obj = getattr(auto_poser.bones, bone)
            bone_obj.mapped_name = mapped_bone

        self.logger.info(f"Applied mapping '{mapping_name}' to armature")
        return True


mappings_manager = MappingsManager()
