# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
from typing import Dict, Any

from go2_robot_sdk.domain.entities.robot_data import RobotData, RobotState, IMUData, OdometryData, JointData, LidarData
from go2_robot_sdk.domain.interfaces.robot_data_publisher import IRobotDataPublisher
from go2_robot_sdk.domain.constants.webrtc_topics import RTC_TOPIC

logger = logging.getLogger(__name__)


class RobotDataService:
    """Service for processing and validating robot data"""

    def __init__(self, publisher: IRobotDataPublisher):
        self.publisher = publisher

    def process_webrtc_message(self, robot_data: RobotData) -> None:
        """Process WebRTC message"""
        try:
            # topic = msg.get('topic')
            # robot_data = RobotData(robot_id=msg.robot_id, timestamp=0.0)

            if robot_data.lidar_data is not None:
                # self._process_lidar_data(msg, robot_data)
                self.publisher.publish_lidar_data(robot_data)
                self.publisher.publish_voxel_data(robot_data)

            elif robot_data.odometry_data is not None:
                # self._process_odometry_data(msg, robot_data)
                self.publisher.publish_odometry(robot_data)

            elif robot_data.robot_state is not None:
                # self._process_sport_mode_state(msg, robot_data)
                self.publisher.publish_robot_state(robot_data)

            elif robot_data.joint_data is not None:
                # self._process_low_state(msg, robot_data)
                self.publisher.publish_joint_state(robot_data)

        except Exception as e:
            logger.error(f"Error processing WebRTC message: {e}")

    def _process_lidar_data(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process lidar data"""
        try:
            decoded_data = msg.get("decoded_data", {})
            data = msg.get("data", {})
            
            robot_data.lidar_data = LidarData(
                positions=decoded_data.get("positions"),
                uvs=decoded_data.get("uvs"),
                resolution=data.get("resolution", 0.0),
                origin=data.get("origin", [0.0, 0.0, 0.0]),
                stamp=data.get("stamp", 0.0),
                width=data.get("width", []),
                src_size=data.get("src_size", 0.0),
                compressed_data=msg.get("compressed_data")
            )
        except Exception as e:
            logger.error(f"Error processing lidar data: {e}")

    def _process_odometry_data(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process odometry data"""
        try:
            pose_data = msg['data']['pose']
            position = pose_data['position']
            orientation = pose_data['orientation']

            # Data validation
            pos_vals = [position['x'], position['y'], position['z']]
            rot_vals = [orientation['x'], orientation['y'], orientation['z'], orientation['w']]

            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos_vals + rot_vals):
                logger.warning("Invalid odometry data - skipping")
                return

            robot_data.odometry_data = OdometryData(
                position=position,
                orientation=orientation
            )
        except Exception as e:
            logger.error(f"Error processing odometry data: {e}")

    def _process_sport_mode_state(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process sport mode state"""
        try:
            data = msg["data"]

            # Data validation
            if not self._validate_float_list(data.get("position", [])):
                return
            if not self._validate_float_list(data.get("range_obstacle", [])):
                return
            if not self._validate_float_list(data.get("foot_position_body", [])):
                return
            if not self._validate_float_list(data.get("foot_speed_body", [])):
                return
            if not self._validate_float(data.get("body_height")):
                return

            robot_data.robot_state = RobotState(
                mode=data["mode"],
                progress=data["progress"],
                gait_type=data["gait_type"],
                position=data["position"],
                body_height=data["body_height"],
                velocity=data["velocity"],
                range_obstacle=data["range_obstacle"],
                foot_force=data["foot_force"],
                foot_position_body=data["foot_position_body"],
                foot_speed_body=data["foot_speed_body"]
            )

            # Process IMU data
            imu_data = data["imu_state"]
            if (self._validate_float_list(imu_data.get("quaternion", [])) and
                self._validate_float_list(imu_data.get("accelerometer", [])) and
                self._validate_float_list(imu_data.get("gyroscope", [])) and
                self._validate_float_list(imu_data.get("rpy", []))):
                
                robot_data.imu_data = IMUData(
                    quaternion=imu_data["quaternion"],
                    accelerometer=imu_data["accelerometer"],
                    gyroscope=imu_data["gyroscope"],
                    rpy=imu_data["rpy"],
                    temperature=imu_data["temperature"]
                )

        except Exception as e:
            logger.error(f"Error processing sport mode state: {e}")

    def _process_low_state(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process low state data"""
        try:
            low_state_data = msg['data']
            robot_data.joint_data = JointData(
                motor_state=low_state_data['motor_state']
            )
        except Exception as e:
            logger.error(f"Error processing low state: {e}")

    def _validate_float_list(self, data: list[Any]) -> bool:
        """Validate a list of float values"""
        return all(isinstance(x, (int, float)) and math.isfinite(x) for x in data)

    def _validate_float(self, value: Any) -> bool:
        """Validate a float value"""
        return isinstance(value, (int, float)) and math.isfinite(value) 