# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
ROS2 node for webrtc_relay: publishes lidar data as sensor_msgs/PointCloud2
on /go2/sensor_msgs/PointCloud2. Instantiated and owned by WebRTCRelayClient.
"""

import logging
import typing as t

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from go2_robot_sdk.infrastructure.sensors.lidar_decoder import update_meshes_for_cloud2

try:
    from sensor_msgs_py import point_cloud2 as point_cloud2_module
except ImportError:
    point_cloud2_module = None

logger = logging.getLogger(__name__)

LIDAR_TOPIC = "/go2/sensor_msgs/PointCloud2"
LIDAR_FRAME_ID = "lidar"


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
        self.get_logger().info(f"Publishing PointCloud2 on {LIDAR_TOPIC} (frame_id={LIDAR_FRAME_ID})")

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
