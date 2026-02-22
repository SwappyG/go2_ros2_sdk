# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
Camera configuration loader for Go2 robot.
Loads camera calibration data from YAML files for different resolutions.
"""

import yaml
import logging
import re
from pydantic import BaseModel
from pathlib import Path
import go2_robot_sdk

_pkg_root = Path(go2_robot_sdk.__file__).parent.parent


def _get_package_share_directory() -> Path:
    """Package root / share dir - works for pure Python (Poetry) or colcon install."""
    try:
        from ament_index_python.packages import PackageNotFoundError  # pyright: ignore[reportMissingImports]
        from ament_index_python import get_package_share_directory  # pyright: ignore[reportMissingImports]
        try:
            return Path(get_package_share_directory('go2_robot_sdk'))
        except PackageNotFoundError:
            return _pkg_root
    except ImportError:
        return _pkg_root


logger = logging.getLogger(__name__)

class GO2CameraInfoMatrix(BaseModel):
    rows: int
    cols: int
    data: list[float]  # flattened, row by row

class GO2CameraInfo(BaseModel):
    image_width: int
    image_height: int
    camera_name: str
    camera_matrix: GO2CameraInfoMatrix
    distortion_model: str
    distortion_coefficients: GO2CameraInfoMatrix
    rectification_matrix: GO2CameraInfoMatrix
    projection_matrix: GO2CameraInfoMatrix


class CameraConfigLoader:
    """Loader for camera calibration configurations"""
    
    def __init__(self, package_name: str = 'go2_robot_sdk'):
        self.package_name = package_name
        self._camera_info_cache: dict[int, GO2CameraInfo] | None = None
    
    def get_supported_resolutions(self) -> list[int]:
        """Get list of supported camera resolutions"""
        calibration_dir = _get_package_share_directory() / "calibration"
        
        files = calibration_dir.glob("front_camera_*.yaml")
        
        resolutions = []
        for file_path in files:
            filename = file_path.name
            numbers = re.findall(r"\d+", filename)
            if numbers:
                resolutions.append(int(numbers[0]))
        
        return sorted(resolutions)
    
    def load_camera_info_for_resolution(self, height: int) -> GO2CameraInfo | None:
        """
        Load camera info for specific resolution.
        
        Args:
            height: Image height (resolution identifier)
            
        Returns:
            CameraInfo message or None if loading fails
        """
        calibration_dir = _get_package_share_directory() / "calibration"
        yaml_file = calibration_dir / f"front_camera_{height}.yaml"
            
        if not yaml_file.exists():
            logger.warning(f"Camera calibration file not found: {yaml_file}")
            return None
        
        logger.info(f"Loading camera info from file: {yaml_file}")
        
        with Path.open(yaml_file) as file_handle:
            camera_data = yaml.safe_load(file_handle)
            
        try:
            # Create and populate CameraInfo message
            return GO2CameraInfo(
                camera_name=camera_data["camera_name"],
                image_width=camera_data["image_width"],
                image_height=camera_data["image_height"],
                camera_matrix=GO2CameraInfoMatrix(rows=3, cols=3, data=camera_data["camera_matrix"]["data"]),
                distortion_coefficients=GO2CameraInfoMatrix(rows=1, cols=5, data=camera_data["distortion_coefficients"]["data"]),
                rectification_matrix=GO2CameraInfoMatrix(rows=3, cols=3, data=camera_data["rectification_matrix"]["data"]),
                projection_matrix=GO2CameraInfoMatrix(rows=3, cols=4, data=camera_data["projection_matrix"]["data"]),
                distortion_model=camera_data["distortion_model"],
            )
            
        except (KeyError, ValueError):
            logger.exception(f"Failed to load camera info for height {height}")
            return None
    
    def load_all_camera_info(self) -> dict[int, GO2CameraInfo]:
        """
        Load camera info for all supported resolutions.
        
        Returns:
            Dictionary mapping resolution height to CameraInfo messages
        """
        if self._camera_info_cache is not None:
            return self._camera_info_cache
        
        supported_heights = self.get_supported_resolutions()
        logger.info(f"Loading camera info for heights: {supported_heights}")
        
        camera_info_dict = {}
        
        for height in supported_heights:
            camera_info = self.load_camera_info_for_resolution(height)
            if camera_info is not None:
                camera_info_dict[height] = camera_info
        
        self._camera_info_cache = camera_info_dict
        return camera_info_dict
    
    def get_camera_info(self, height: int) -> GO2CameraInfo | None:
        """
        Get camera info for specific height with caching.
        
        Args:
            height: Image height
            
        Returns:
            CameraInfo message or None if not available
        """
        if self._camera_info_cache is None:
            self._camera_info_cache = self.load_all_camera_info()
        
        return self._camera_info_cache.get(height)


# Global loader instance for backward compatibility
_camera_loader: CameraConfigLoader | None = None


def get_camera_loader() -> CameraConfigLoader:
    """Get singleton camera config loader instance"""
    global _camera_loader  # noqa: PLW0603
    if _camera_loader is None:
        _camera_loader = CameraConfigLoader()
    return _camera_loader


def load_camera_info() -> dict[int, GO2CameraInfo]:
    """
    Load camera info for all supported resolutions.
    Backward compatibility function.
    
    Returns:
        Dictionary mapping resolution height to CameraInfo messages
    """
    loader = get_camera_loader()
    return loader.load_all_camera_info() 