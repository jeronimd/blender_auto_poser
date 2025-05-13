import importlib.util
import os
import site
import subprocess
import sys

from ..utils.logger import Logger


class Dependencies:
    def __init__(self):
        self.logger = Logger.setup(__name__, "WARNING")

        self.dependencies = ["msgpack", "torch", "pytorch_lightning"]
        self.packages_path = os.path.dirname(site.getusersitepackages())
        self.site_packages_path = os.path.join(self.packages_path, "site-packages")
        self.scripts_path = os.path.join(self.packages_path, "Scripts")

        self.dependencies_installed = False

        self.logger.debug(f"packages_path: {self.packages_path}")
        self.register_paths()
        self.update_dependencies()

    def install_dependencies(self):
        self.logger.info("Installing dependencies")
        try:
            for dependency in self.dependencies:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
            self.logger.info("Dependencies installed")
        except subprocess.CalledProcessError:
            self.logger.error("Failed to install dependencies")

    def register_paths(self):
        self.logger.debug("Registering paths")
        if self.site_packages_path not in sys.path:
            sys.path.insert(0, self.site_packages_path)
        if self.scripts_path not in sys.path:
            sys.path.insert(0, self.scripts_path)

    def update_dependencies(self):
        dependencies_installed = True
        for dependency in self.dependencies:
            if importlib.util.find_spec(dependency, self.site_packages_path) is None:
                dependencies_installed = False
                break

        if dependencies_installed:
            self.logger.debug("Dependencies are installed")
            self.dependencies_installed = False
            # self.dependencies_installed = True
        else:
            self.logger.warning("Dependencies are not installed")
            self.dependencies_installed = False

    def check(self):
        return self.dependencies_installed


dependencies = Dependencies()
