# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class RobotState:
    """Robot state information"""
    mode: int
    progress: float
    gait_type: int
    position: list[float]
    body_height: float
    velocity: list[float]
    range_obstacle: list[float]
    foot_force: list[float]
    foot_position_body: list[float]
    foot_speed_body: list[float]


@dataclass
class IMUData:
    """IMU sensor data"""
    quaternion: list[float]
    accelerometer: list[float]
    gyroscope: list[float]
    rpy: list[float]
    temperature: float


@dataclass
class OdometryData:
    """Odometry data"""
    position: dict[str, float]  # x, y, z
    orientation: dict[str, float]  # x, y, z, w (quaternion)


@dataclass
class JointData:
    """Joint data"""
    motor_state: list[dict[str, float]]  # q, dq, ddq, tau


@dataclass
class LidarData:
    """LiDAR sensor data"""
    positions: npt.NDArray[np.uint8]
    uvs: npt.NDArray[np.uint8]
    resolution: float
    origin: list[float]
    stamp: float
    width: list[int] | None = None
    src_size: int | None = None
    compressed_data: bytes | None = None


@dataclass
class CameraData:
    """Camera data"""
    image: npt.NDArray[np.uint8]
    height: int
    width: int
    encoding: str = "bgr8"


@dataclass
class RobotData:
    """Aggregated robot data container"""
    robot_id: str
    timestamp: float
    robot_state: RobotState | None = None
    imu_data: IMUData | None = None
    odometry_data: OdometryData | None = None
    joint_data: JointData | None = None
    lidar_data: LidarData | None = None
    camera_data: CameraData | None = None
    raw_message: str | bytes | None = None
    