"""
Configuration settings for the GO2 Robot GUI Client.
All key bindings, velocity settings, and ramp settings are configurable here.
"""
from PySide6.QtCore import Qt
from typing import Dict

class GuiConfig:
    """Configuration class for GUI client settings."""
    
    # Key bindings configuration
    # Maps action names to Qt key codes
    KEY_BINDINGS: Dict[str, Qt.Key] = {
        "forward": Qt.Key.Key_W,
        "backward": Qt.Key.Key_S,
        "rotate_left": Qt.Key.Key_A,
        "rotate_right": Qt.Key.Key_D,
        "strafe_left": Qt.Key.Key_Q,
        "strafe_right": Qt.Key.Key_E,
        "stop": Qt.Key.Key_Space,
        "quit": Qt.Key.Key_P,
    }
    
    # Initial velocity settings (0.0 to 1.0, where 1.0 is maximum)
    # Separate velocities for linear movement (forward/backward/strafe) and rotation
    INITIAL_LINEAR_VELOCITY: float = 0.25
    INITIAL_ROTATION_VELOCITY: float = 0.50
    
    # Velocity ramp time in milliseconds
    # This is the time it takes to ramp from 0 to full velocity
    VELOCITY_RAMP_TIME_MS: int = 1000
    
    # Movement update interval in milliseconds (how often to send commands)
    MOVEMENT_UPDATE_INTERVAL_MS: int = 50  # 20 Hz
    
    @classmethod
    def get_key_for_action(cls, action: str) -> Qt.Key:
        """Get the Qt key code for a given action."""
        return cls.KEY_BINDINGS.get(action)
    
    @classmethod
    def get_action_for_key(cls, key: Qt.Key) -> str | None:
        """Get the action name for a given Qt key code."""
        for action, key_code in cls.KEY_BINDINGS.items():
            if key_code == key:
                return action
        return None

