# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
from typing import Any, cast
import json

import go2_robot_sdk.domain.entities.robot_data as rd 
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


def process_webrtc_message(
    msg: dict[str, Any], robot_id: str
) -> rd.RobotOdom | rd.LowState | rd.MultipleState | rd.SportModeState | rd.LidarData | None:
    """Process WebRTC message"""
    if msg['type'] not in ['msg', 'res']:
        logger.info(f"msg received on datachannel is not type 'msg': {msg=}")
        return None

    topic = msg['topic']        
    if topic == RTC_TOPIC["ULIDAR_ARRAY"]:
        return parse_lidar_data(msg)

    elif topic == RTC_TOPIC["ROBOTODOM"]:
        return rd.RobotOdom.model_validate(msg['data'])
        
    elif topic == RTC_TOPIC["LF_SPORT_MOD_STATE"]:
        return rd.SportModeState.model_validate(msg['data'])

    elif topic == RTC_TOPIC["LOW_STATE"]:
        return rd.LowState.model_validate(msg['data'])

    elif topic == RTC_TOPIC["MULTIPLE_STATE"]:
        if isinstance(msg['data'], str):
            return rd.MultipleState.model_validate_json(msg['data'])
        else:
            return rd.MultipleState.model_validate(msg['data'])

    else:
        return None


def parse_lidar_data(message: dict[str, Any]) -> rd.LidarData | None:
    """Process lidar data"""

    try:
        decoded_data = message['decoded_data']
        if decoded_data is None:
            return None
        
        data = message['data']
        
        return rd.LidarData(
            positions=decoded_data.get("positions", None),
            points=decoded_data.get("points", None),
            uvs=decoded_data.get("uvs"),
            resolution=data.get("resolution", 0.0),
            origin=list(data.get("origin", [0.0, 0.0, 0.0])),
            stamp=data.get("stamp", 0.0),
            width=data.get("width"),
            src_size=data.get("src_size"),
            compressed_data=message.get("compressed_data"),
        )
    
    # TODO (swapnil): catch a better exception here
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Error processing lidar data: {e=}")
        return None
