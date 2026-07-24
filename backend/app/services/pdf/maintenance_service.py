"""
Maintenance Service Layer.

Provides operations to prune temporary *.tmp files, clean empty directories,
and clear orphan statistics in storage paths.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class MaintenanceService:
    """
    Coordinates cleanup routines.
    """

    def __init__(self, target_dir: Path | None = None):
        """
        Initialize the service.
        """
        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

    def cleanup(self) -> Dict[str, Any]:
        """
        Deletes temporary *.tmp files, empty directories, and orphan statistics.
        Does not touch valid document files.
        """
        logger.info("Starting maintenance cleanup inside uploads directory '%s'...", self.target_dir)

        removed_temp_files = 0
        removed_empty_dirs = 0
        removed_orphan_files = 0

        # Ensure directory exists before walking
        if not self.target_dir.exists():
            return {
                "success": True,
                "removed_temp_files": 0,
                "removed_empty_directories": 0,
                "message": "Uploads directory does not exist. No actions performed."
            }

        # 1. Walk directory and delete *.tmp files and orphan metadata
        for root, dirs, files in os.walk(str(self.target_dir), topdown=False):
            root_path = Path(root)

            for file in files:
                file_path = root_path / file
                
                # Check for temp swap files
                if file.endswith(".tmp"):
                    try:
                        file_path.unlink()
                        removed_temp_files += 1
                        logger.info("Cleaned temporary file: %s", file)
                    except Exception as err:
                        logger.warning("Failed to delete temp file %s: %s", file_path, str(err))

                # Check for orphan files directly in the root of uploads/ folder
                elif root_path == self.target_dir and file not in ["system_statistics.json"]:
                    try:
                        file_path.unlink()
                        removed_orphan_files += 1
                        logger.info("Cleaned orphan root file: %s", file)
                    except Exception as err:
                        logger.warning("Failed to delete orphan root file %s: %s", file_path, str(err))

            # 2. Delete empty directories (except the uploads root directory itself)
            if root_path != self.target_dir:
                try:
                    if not any(root_path.iterdir()):
                        root_path.rmdir()
                        removed_empty_dirs += 1
                        logger.info("Removed empty directory: %s", root_path.name)
                except Exception as err:
                    logger.warning("Failed to remove directory %s: %s", root_path, str(err))

        msg = (
            f"Maintenance cleanup successfully completed. "
            f"Removed {removed_temp_files} temp files, "
            f"{removed_orphan_files} orphan files, "
            f"and {removed_empty_dirs} empty folders."
        )
        logger.info(msg)

        return {
            "success": True,
            "removed_temp_files": removed_temp_files + removed_orphan_files,
            "removed_empty_directories": removed_empty_dirs,
            "message": msg
        }
