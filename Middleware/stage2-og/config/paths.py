import os
from pathlib import Path

class PathManager:
    @staticmethod
    def get_base_path():
        return Path(__file__).parent.parent

    @staticmethod
    def get_data_path(data_type="raw"):
        base = PathManager.get_base_path()
        path_mapping = {
            "raw": base / "data" / "raw",
            "processed": base / "data" / "processed",
            "split": base / "data" / "split",
        }
        return path_mapping.get(data_type, base / "data")

    @staticmethod
    def get_checkpoint_path():
        return PathManager.get_base_path() / "models" / "checkpoints"

    @staticmethod
    def get_log_path():
        return PathManager.get_base_path() / "logs"

    @staticmethod
    def ensure_dirs_exist():
        directory_paths = [
            PathManager.get_data_path("raw"),
            PathManager.get_data_path("processed"),
            PathManager.get_data_path("split"),
            PathManager.get_checkpoint_path(),
            PathManager.get_log_path(),
        ]
        for directory_path in directory_paths:
            directory_path.mkdir(parents=True, exist_ok=True)
