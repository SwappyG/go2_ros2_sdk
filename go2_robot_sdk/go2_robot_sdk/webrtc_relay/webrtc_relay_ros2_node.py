# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
ROS2 node for webrtc_relay: publishes lidar data as sensor_msgs/PointCloud2
and odometry as nav_msgs/Odometry. Instantiated and owned by WebRTCRelayClient.
"""

import logging
import typing as t

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from go2_robot_sdk.infrastructure.sensors.lidar_decoder import update_meshes_for_cloud2

try:
    from sensor_msgs_py import point_cloud2 as point_cloud2_module
except ImportError:
    point_cloud2_module = None

logger = logging.getLogger(__name__)

LIDAR_TOPIC = "/go2/sensor_msgs/PointCloud2"
LIDAR_FRAME_ID = "lidar"
ODOM_TOPIC = "/go2/nav_msgs/Odometry"

# Lidar mount offset in base_link (meters). Adjust for your GO2 lidar position.
LIDAR_OFFSET_X = 0.0
LIDAR_OFFSET_Y = 0.0
LIDAR_OFFSET_Z = 0.0


class WebRTCRelayROS2Node(Node):
    """
    ROS2 node that publishes lidar point clouds from webrtc_relay lidar frames.
    Created and spun by WebRTCRelayClient when enable_ros2_publish is True.
    """

    def __init__(self) -> None:
        super().__init__("go2_lidar_relay")
        if point_cloud2_module is None:
            raise RuntimeError("sensor_msgs_py is required for PointCloud2 publishing")

        # Use RELIABLE so subscribers with default QoS (e.g. RViz) receive messages.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(
            PointCloud2,
            LIDAR_TOPIC,
            qos,
        )
        self._odom_pub = self.create_publisher(
            Odometry,
            ODOM_TOPIC,
            qos,
        )
        self._tf_broadcaster = TransformBroadcaster(self, qos=QoSProfile(depth=10))
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_lidar_transform()
        self.get_logger().info(f"Publishing PointCloud2 on {LIDAR_TOPIC} (frame_id={LIDAR_FRAME_ID})")
        self.get_logger().info(f"Publishing Odometry on {ODOM_TOPIC} (frame_id=odom)")

    def _publish_static_lidar_transform(self) -> None:
        """Publish static base_link -> lidar transform once at startup."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = LIDAR_FRAME_ID
        t.transform.translation.x = LIDAR_OFFSET_X
        t.transform.translation.y = LIDAR_OFFSET_Y
        t.transform.translation.z = LIDAR_OFFSET_Z
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(t)

    def publish_odometry(self, position: dict[str, float], orientation: dict[str, float]) -> None:
        """
        Publish nav_msgs/Odometry from position and orientation dicts.
        position: {"x": float, "y": float, "z": float}
        orientation: {"x": float, "y": float, "z": float, "w": float} (quaternion)
        """
        try:
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_link"

            pos_x = float(position.get("x", 0.0))
            pos_y = float(position.get("y", 0.0))
            pos_z = float(position.get("z", 0.0)) + 0.07  # Match main SDK z offset

            odom_msg.pose.pose.position.x = pos_x
            odom_msg.pose.pose.position.y = pos_y
            odom_msg.pose.pose.position.z = pos_z

            ori_x = float(orientation.get("x", 0.0))
            ori_y = float(orientation.get("y", 0.0))
            ori_z = float(orientation.get("z", 0.0))
            ori_w = float(orientation.get("w", 1.0))

            odom_msg.pose.pose.orientation.x = ori_x
            odom_msg.pose.pose.orientation.y = ori_y
            odom_msg.pose.pose.orientation.z = ori_z
            odom_msg.pose.pose.orientation.w = ori_w

            self._odom_pub.publish(odom_msg)

            # Publish odom -> base_link TF
            odom_trans = TransformStamped()
            odom_trans.header.stamp = odom_msg.header.stamp
            odom_trans.header.frame_id = "odom"
            odom_trans.child_frame_id = "base_link"
            odom_trans.transform.translation.x = pos_x
            odom_trans.transform.translation.y = pos_y
            odom_trans.transform.translation.z = pos_z
            odom_trans.transform.rotation.x = ori_x
            odom_trans.transform.rotation.y = ori_y
            odom_trans.transform.rotation.z = ori_z
            odom_trans.transform.rotation.w = ori_w
            self._tf_broadcaster.sendTransform(odom_trans)
        except Exception as e:
            logger.warning("Failed to publish odometry: %s", e)

    def publish_lidar(self, lidar_frame: dict[str, t.Any]) -> None:
        """
        Build a PointCloud2 from a webrtc_relay lidar_frame and publish it.
        lidar_frame must contain 'decoded_data' (positions, uvs) and 'data' (resolution, origin).
        """
        try:
            dec = lidar_frame.get("decoded_data")
            meta = lidar_frame.get("data")
            if not dec or not meta:
                logger.debug("lidar_frame missing decoded_data or data, skipping publish")
                return

            positions = dec.get("positions")
            uvs = dec.get("uvs")
            resolution = float(meta.get("resolution", 0.01))
            origin_list = meta.get("origin", [0.0, 0.0, 0.0])
            if len(origin_list) != 3:
                logger.warning(f"invalid origin in lidar_frame: {origin_list}")
                return
            origin = (float(origin_list[0]), float(origin_list[1]), float(origin_list[2]))

            if positions is None:
                logger.debug("lidar_frame decoded_data missing positions, skipping publish")
                return
            if uvs is None:
                logger.debug("lidar_frame decoded_data missing uvs, skipping publish")
                return

            points = update_meshes_for_cloud2(
                positions,
                uvs,
                resolution,
                origin,
                0.0,
            )
            if points is None or len(points) == 0:
                return

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = LIDAR_FRAME_ID

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            msg = point_cloud2_module.create_cloud(header, fields, points)
            self._pub.publish(msg)
        except Exception as e:
            logger.warning("Failed to publish lidar PointCloud2: %s", e)
