# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
from typing import Dict, Any, cast
import json

from go2_robot_sdk.domain.entities.robot_data import (
    RobotData, RobotState, IMUData, OdometryData, JointData, LidarData
)
from go2_robot_sdk.domain.constants.webrtc_topics import RTC_TOPIC

logger = logging.getLogger(__name__)

def parse_datachannel_message(raw_message: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads(raw_message))
    except json.JSONDecodeError as e:
        logger.warning("Failed to decode JSON message")
        raise ValueError("Failed to decode JSON message") from e
    except Exception as e:
        logger.warning(f"go2 datachannel message had unexcepted form. {raw_message=}")
        raise ValueError("Failed to decode JSON message") from e
    
def process_webrtc_message(msg: dict[str, Any], robot_id: str) -> RobotData | None:
    """Process WebRTC message"""
    if msg['type'] != 'msg':
        logger.info(f"msg received on datachannel is not type 'msg': {msg=}")
        return None

    topic = msg['topic']        
    robot_data = RobotData(robot_id=robot_id, timestamp=0.0)
    if topic == RTC_TOPIC["ULIDAR_ARRAY"]:
        robot_data.lidar_data = parse_lidar_data(msg)

    elif topic == RTC_TOPIC["ROBOTODOM"]:
        robot_data.odometry_data = parse_odometry_data(msg['data'])

    elif topic == RTC_TOPIC["LF_SPORT_MOD_STATE"]:
        ret = parse_sport_mode_state(msg['data'])
        if ret is not None:
            robot_data.robot_state, robot_data.imu_data = ret 

    elif topic == RTC_TOPIC["LOW_STATE"]:
        robot_data.joint_data = parse_low_state(msg['data'])

    else:
        return None
    
    return robot_data

def parse_lidar_data(message: dict[str, Any]) -> LidarData | None:
    """Process lidar data"""

    try:
        decoded_data = message['decoded_data']
        if decoded_data is None:
            logger.warning(f"failed to decode lidar message from go2 datachannel")
            return None
        
        data = message['data']
        
        return LidarData(
            positions=decoded_data["positions"],
            uvs=decoded_data.get("uvs"),
            resolution=data.get("resolution", 0.0),
            origin=list(data.get("origin", [0.0, 0.0, 0.0])),
            stamp=data.get("stamp", 0.0),
            width=data.get("width"),
            src_size=data.get("src_size"),
            compressed_data=message.get("compressed_data")
        )
    except Exception as e:
        logger.warning(f"Error processing lidar data: {e=}")
        return None

def parse_odometry_data(data: Dict[str, Any]) -> OdometryData | None:
    """Process odometry data"""
    try:
        pose_data = data['pose']
        position = pose_data['position']
        orientation = pose_data['orientation']

        # Data validation
        pos_vals = [position['x'], position['y'], position['z']]
        rot_vals = [orientation['x'], orientation['y'], orientation['z'], orientation['w']]

        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos_vals + rot_vals):
            logger.warning("Invalid odometry data - skipping")
            raise ValueError("Invalid odometry data - skipping")

        return OdometryData(
            position=position,
            orientation=orientation
        )
    except Exception as e:
        logger.warning(f"gailed to parse odometry data. {e=}. {data=}")
        return None

def parse_sport_mode_state(data: Dict[str, Any]) -> tuple[RobotState, IMUData] | None:
    """Process sport mode state"""
    try:
        robot_state = RobotState(
            mode=data["mode"],
            progress=data["progress"],
            gait_type=data["gait_type"],
            position=_validated_float_list(data["position"]),
            body_height=_validated_float(data["body_height"]),
            velocity=data["velocity"],
            range_obstacle=_validated_float_list(data["range_obstacle"]),
            foot_force=data["foot_force"],
            foot_position_body=_validated_float_list(data["foot_position_body"]),
            foot_speed_body=_validated_float_list(data["foot_speed_body"])
        )
        
        imu_dict = data['imu_state']
        imu_data = IMUData(
            quaternion=_validated_float_list(imu_dict["quaternion"]),
            accelerometer=_validated_float_list(imu_dict["accelerometer"]),
            gyroscope=_validated_float_list(imu_dict["gyroscope"]),
            rpy=_validated_float_list(imu_dict["rpy"]),
            temperature=imu_dict["temperature"]
        )

        return robot_state, imu_data
    except Exception as e:
        logger.warning(f"gailed to parse sport mode data. {type(e).__name__}:{str(e)}")
        return None

def parse_low_state(low_state_data: Dict[str, Any]) -> JointData | None:
    """Process low state data"""
    try:
        return JointData(
            motor_state=low_state_data['motor_state']
        )
    except Exception as e:
        logger.error(f"Error processing low state: {type(e).__name__}:{str(e)}")

def _validated_float_list(data: list[Any]) -> list[float]:
    """Validate a list of float values"""
    if all(isinstance(x, (int, float)) and math.isfinite(x) for x in data):
        return data
    raise ValueError(f"list was not all floats or finite. {data=}")

def _validated_float(value: Any) -> float:
    """Validate a float value"""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    raise ValueError(f"value isn't int or float, {value=}")