# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Literal, Sequence, TypeAlias
from pydantic import BaseModel, Field
import math

class Stamp(BaseModel):
    sec: float
    nanosec: float


class BmsState(BaseModel):
    bq_ntc: tuple[int, int] = (-1, 1)
    current: float = float('-inf')
    cycle: int = -1
    mcu_ntc: tuple[int, int] = (-1, -1)
    soc: int = -1
    version_high: int = -1
    version_low: int = -1
    

class MotorState(BaseModel):
    q: float = float('-inf')
    lost: int = -1
    reserve: tuple[int, int] = (-1, -1)
    temperature: int | float = -1



class LowState(BaseModel):
    bms_state: BmsState = Field(default_factory=BmsState)

    # in GO2 Pro, these are random numbers
    foot_force: tuple[float, float, float, float] = (0, 0, 0, 0)

    imu_state: dict[Literal['rpy'], tuple[float, float, float]] = Field(default_factory=lambda: {'rpy': (0, 0, 0)})
    motor_state: list[MotorState] = Field(default_factory=lambda: [MotorState() for _ in range(12)])
    power_v: float = float('-inf')
    temperature_ntcl: float = float('-inf')


class MultipleState(BaseModel):
    brightness: int = -1
    obstacle_avoid_switch: bool | None = None
    uwb_switch: bool | None = None
    volume: int = -1
    
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
    temperature: float = float('-inf')


class SportModeState(BaseModel):
    body_height: float = float('-inf')
    error_code: float = float('-inf')
    foot_force: tuple[float, float, float, float] = (0, 0, 0, 0)
    foot_position_body: Sequence[float] = Field(default_factory=tuple)
    foot_raise_height: float = float('-inf')
    foot_speed_body: Sequence[float] = Field(default_factory=tuple)
    gait_type: int = -1
    imu_state: SportModeImuState
    mode: int = -2**32
    progress: int = -2**32
    position: tuple[float, float, float] = (0, 0, 0)
    range_obstacle: tuple[float, float, float, float] = (0, 0, 0, 0)
    stamp: Stamp = Field(default_factory=lambda: Stamp(sec=0, nanosec=0))
    velocity: tuple[float, float, float] = (0, 0, 0)
    yaw_speed: float = float('-inf')


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

@dataclass
class LidarMetadata:
    """LiDAR sensor data"""
    num_uvs: int
    resolution: float
    origin: tuple[float, float, float]
    stamp: float
    num_positions: int | None = None
    num_points: int | None = None
    width: list[int] | None = None
    src_size: int | None = None
    num_bytes: int | None = None

RobotData: TypeAlias = RobotOdom | LowState | MultipleState | SportModeState | LidarData

@dataclass
class RobotDataWithRawMessage:
    robot_id: str
    robot_data: RobotData | None
    raw_message: str | bytes
    timestamp: float