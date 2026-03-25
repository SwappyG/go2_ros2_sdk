# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Literal, Sequence, TypeAlias
from pydantic import BaseModel


class Stamp(BaseModel):
    sec: float
    nanosec: float


class BmsState(BaseModel):
    bq_ntc: tuple[int, int]
    current: float
    cycle: int
    mcu_ntc: tuple[int, int]
    soc: int
    version_high: int
    version_low: int
    

class MotorState(BaseModel):
    q: float
    lost: int
    reserve: tuple[int, int]
    temperature: int | float



class LowState(BaseModel):
    bms_state: BmsState
    foot_force: tuple[float, float, float, float]  # in GO2 Pro, these are random numbers
    imu_state: dict[Literal['rpy'], tuple[float, float, float]]
    motor_state: list[MotorState]
    power_v: float
    temperature_ntcl: float


class MultipleState(BaseModel):
    brightness: int
    obstacle_avoid_switch: bool
    uwb_switch: bool
    volume: int
    
class RobotOdomHeader(BaseModel):
    frame_id: str
    stamp: Stamp


class RobotOdomPosition(BaseModel):
    x: float
    y: float
    z: float


class RobotOdomOrientation(BaseModel):
    x: float
    y: float
    z: float
    w: float


class RobotOdomPose(BaseModel):
    position: RobotOdomPosition
    orientation: RobotOdomOrientation


class RobotOdom(BaseModel):
    header: RobotOdomHeader
    pose: RobotOdomPose


class SportModeImuState(BaseModel):
    accelerometer: tuple[float, float, float]
    gyroscope: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    rpy: tuple[float, float, float]
    temperature: float


class SportModeState(BaseModel):
    body_height: float
    error_code: float
    foot_force: tuple[float, float, float, float]
    foot_position_body: Sequence[float]
    foot_raise_height: float
    foot_speed_body: Sequence[float]
    gait_type: int
    imu_state: SportModeImuState
    mode: int
    progress: int
    position: tuple[float, float, float]
    range_obstacle: tuple[float, float, float, float]
    stamp: Stamp
    velocity: tuple[float, float, float]
    yaw_speed: float


@dataclass
class LidarData:
    """LiDAR sensor data"""
    uvs: npt.NDArray[np.uint8]
    resolution: float
    origin: list[float]
    stamp: float
    positions: npt.NDArray[np.uint8] | None = None
    points: npt.NDArray[np.uint8] | None = None
    width: list[int] | None = None
    src_size: int | None = None
    compressed_data: bytes | None = None

RobotData: TypeAlias = RobotOdom | LowState | MultipleState | SportModeState | LidarData

@dataclass
class RobotDataWithRawMessage:
    robot_id: str
    robot_data: RobotData | None
    raw_message: str | bytes
    timestamp: float