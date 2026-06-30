# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
Command generation utilities for Go2 robot.
Contains functions to create properly formatted WebRTC commands.
"""

import datetime
import json
import random
from typing import Any

# Topic constants for different command types
SPORT_MODE_TOPIC = "rt/api/sport/request"
OBSTACLE_AVOIDANCE_TOPIC = "rt/api/obstacles_avoid/request"
WIRELESS_CONTROLLER_TOPIC = "rt/wirelesscontroller"

# Max virtual joystick deflection (matches Unitree obstacles_avoid example).
_WIRELESS_JOY_MAX = 0.9


def generate_id() -> int:
    """Generate a unique command ID based on timestamp and random number"""
    timestamp_part = int(datetime.datetime.now().astimezone().timestamp() * 1000 % 2147483648)
    random_part = random.randint(0, 999)  # noqa: S311
    return timestamp_part + random_part


def create_command_structure(
        api_id: int, 
        parameter: str | dict[str, Any], 
        topic: str = SPORT_MODE_TOPIC,
        command_id: int | None = None,
) -> dict[str, Any]:
    """
    Create a standardized command structure for WebRTC communication.
    
    Args:
        api_id: API command identifier
        parameter: Command parameters (string or dict)
        topic: WebRTC topic for the command
        command_id: Optional specific command ID
        
    Returns:
        Dictionary containing the formatted command structure
    """
    final_id = generate_id() if command_id is None or command_id == 0 else command_id
    
    # Convert parameter to JSON string if it's a dict
    param_str = json.dumps(parameter) if isinstance(parameter, dict) else str(parameter)

    return {
        "type": "msg",
        "topic": topic,
        "data": {
            "header": {
                "identity": {
                    "id": final_id,
                    "api_id": api_id,
                },
            },
            "parameter": param_str,
        },
    }


def gen_command(
        cmd: int,
        parameters: str | dict[str, Any] | None = None,
        topic: str | None = None,
        command_id: int | None = None,
) -> str:
    """
    Generate a general robot command.
    
    Args:
        cmd: Command ID from ROBOT_CMD constants
        parameters: Optional command parameters
        topic: Optional topic override
        command_id: Optional specific command ID
        
    Returns:
        JSON string of the formatted command
    """
    parameter = parameters if parameters is not None else ""
    command = create_command_structure(
        api_id=cmd,
        parameter=parameter,
        topic=topic or SPORT_MODE_TOPIC,
        command_id=command_id,
    )
    return json.dumps(command)

def get_sport_command(
    cmd: int, parameters: str | dict[str, Any] | None = None, command_id: int | None = None
) -> str:
    """Convenience function to generate a sport mode command."""
    return gen_command(
        cmd=cmd,
        parameters=parameters,
        topic=SPORT_MODE_TOPIC,
        command_id=command_id,
    )

def gen_mov_command(
        x: float,
        y: float, 
        z: float,
        obstacle_avoidance: bool = False,
) -> str:
    """
    Generate a movement command for the robot.
    
    Args:
        x: Forward/backward velocity
        y: Left/right velocity  
        z: Rotation velocity (yaw for obstacle avoidance)
        obstacle_avoidance: Whether to use obstacle avoidance mode
        
    Returns:
        JSON string of the formatted movement command
    """
    if obstacle_avoidance:
        # Obstacle avoidance uses different parameter format
        parameters = {"x": x, "y": y, "yaw": z, "mode": 0}
        command = create_command_structure(
            api_id=1003,  # Obstacle avoidance move command
            parameter=parameters,
            topic=OBSTACLE_AVOIDANCE_TOPIC,
        )
    else:
        # Standard sport mode movement
        parameters = {"x": x, "y": y, "z": z}
        command = create_command_structure(
            api_id=1008,  # Sport mode move command
            parameter=parameters,
            topic=SPORT_MODE_TOPIC,
        )

    return json.dumps(command)


def gen_wireless_controller_command(
    forward: float,
    strafe: float,
    rotation: float,
    *,
    ry: float = 0.0,
    keys: int = 0,
) -> str:
    """
    Virtual joystick command for Go2 firmware obstacle avoidance.

    When OA is enabled, the robot expects ``rt/wirelesscontroller`` input (not sport
    api 1008). The firmware applies obstacle filtering to these stick values.
    """
    lx = max(-_WIRELESS_JOY_MAX, min(_WIRELESS_JOY_MAX, -strafe * _WIRELESS_JOY_MAX))
    ly = max(-_WIRELESS_JOY_MAX, min(_WIRELESS_JOY_MAX, forward * _WIRELESS_JOY_MAX))
    rx = max(-_WIRELESS_JOY_MAX, min(_WIRELESS_JOY_MAX, rotation * _WIRELESS_JOY_MAX))
    return json.dumps(
        {
            "type": "msg",
            "topic": WIRELESS_CONTROLLER_TOPIC,
            "data": {"lx": lx, "ly": ly, "rx": rx, "ry": ry, "keys": keys},
        }
    ) 