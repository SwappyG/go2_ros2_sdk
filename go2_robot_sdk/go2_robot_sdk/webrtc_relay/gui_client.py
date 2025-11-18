"""
PyQt5 GUI client for GO2 robot control with integrated video, lidar, and odometry display.
"""
import sys
import asyncio
import argparse
import logging
import typing as t
import numpy as np
import numpy.typing as npt

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QGridLayout, QGroupBox, QLabel, QLineEdit, QCheckBox,  # pyright: ignore[reportUnusedImport]
    QSplitter, QDoubleSpinBox, QSpinBox
)
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QKeyEvent, QFocusEvent

from aiortc import MediaStreamTrack  # type: ignore

from go2_robot_sdk.domain.entities.robot_config import RobotConfig
from go2_robot_sdk.domain.entities.robot_data import RobotData
from go2_robot_sdk.webrtc_relay.webrtc_relay_client import WebRTCRelayClient
from go2_robot_sdk.webrtc_relay.gui_widgets import (
    VideoWidget, LidarWidget, OdometryWidget, StatusWidget
)
from go2_robot_sdk.webrtc_relay.gui_configurations import GuiConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RobotControlSignals(QObject):
    """Qt signals for thread-safe updates from async callbacks."""
    video_frame_ready = Signal(np.ndarray)
    lidar_update = Signal(np.ndarray, int, float, tuple)
    odometry_update = Signal(dict, dict)
    connection_status = Signal(bool)
    status_message = Signal(str)


class ControlPanel(QWidget):
    """Control panel with movement buttons and commands."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Movement controls
        movement_group = QGroupBox("Movement Controls")
        movement_layout = QGridLayout()
        
        # Create movement buttons (using configurable key bindings)
        self.btn_forward = QPushButton("↑ Forward (W)")
        self.btn_backward = QPushButton("↓ Backward (S)")
        self.btn_strafe_left = QPushButton("← Strafe Left (Q)")
        self.btn_strafe_right = QPushButton("→ Strafe Right (E)")
        self.btn_rotate_left = QPushButton("⟲ Rotate Left (A)")
        self.btn_rotate_right = QPushButton("⟳ Rotate Right (D)")
        self.btn_stop = QPushButton("⏹ Stop (Space)")
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 5px;
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #777;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """
        
        for btn in [self.btn_forward, self.btn_backward, self.btn_strafe_left, 
                   self.btn_strafe_right, self.btn_rotate_left, self.btn_rotate_right, self.btn_stop]:
            btn.setStyleSheet(button_style)
            btn.setMinimumHeight(35)
            btn.setMaximumHeight(45)
        
        self.btn_stop.setStyleSheet(button_style + """
            QPushButton { background-color: #8b0000; }
            QPushButton:hover { background-color: #a00000; }
        """)
        
        # Layout movement buttons in a grid
        movement_layout.addWidget(self.btn_forward, 0, 1)
        movement_layout.addWidget(self.btn_strafe_left, 1, 0)
        movement_layout.addWidget(self.btn_stop, 1, 1)
        movement_layout.addWidget(self.btn_strafe_right, 1, 2)
        movement_layout.addWidget(self.btn_backward, 2, 1)
        movement_layout.addWidget(self.btn_rotate_left, 3, 0)
        movement_layout.addWidget(self.btn_rotate_right, 3, 2)
        
        movement_group.setLayout(movement_layout)
        layout.addWidget(movement_group)
        
        # Posture controls
        posture_group = QGroupBox("Posture Controls")
        posture_layout = QVBoxLayout()
        
        self.btn_stand_up = QPushButton("Stand Up")
        self.btn_recovery_stand = QPushButton("Recovery Stand")
        self.btn_sit = QPushButton("Sit Down")
        self.btn_lie_down = QPushButton("Lie Down")
        
        for btn in [self.btn_stand_up, self.btn_recovery_stand, self.btn_sit, self.btn_lie_down]:
            btn.setStyleSheet(button_style)
            btn.setMinimumHeight(30)
            btn.setMaximumHeight(40)
            posture_layout.addWidget(btn)
        
        posture_group.setLayout(posture_layout)
        layout.addWidget(posture_group)
        
        # Fun commands
        fun_group = QGroupBox("Fun Commands")
        fun_layout = QVBoxLayout()
        
        self.btn_balance_stand = QPushButton("⚖️ Balance Stand")
        
        self.btn_balance_stand.setStyleSheet(button_style)
        self.btn_balance_stand.setMinimumHeight(30)
        self.btn_balance_stand.setMaximumHeight(40)
        fun_layout.addWidget(self.btn_balance_stand)
        
        fun_group.setLayout(fun_layout)
        layout.addWidget(fun_group)
        
        # Velocity and Settings
        avoid_group = QGroupBox("Settings")
        avoid_layout = QVBoxLayout()
        
        self.chk_obstacle_avoid = QCheckBox("Enable Obstacle Avoidance")
        self.chk_obstacle_avoid.setStyleSheet("QCheckBox { color: white; padding: 5px; }")
        avoid_layout.addWidget(self.chk_obstacle_avoid)
        
        # Linear velocity control (forward/backward/strafe)
        linear_velocity_layout = QHBoxLayout()
        linear_velocity_layout.addWidget(QLabel("Linear Velocity:"))
        self.linear_velocity_spinbox = QDoubleSpinBox()
        self.linear_velocity_spinbox.setRange(0.0, 1.0)
        self.linear_velocity_spinbox.setSingleStep(0.1)
        self.linear_velocity_spinbox.setValue(GuiConfig.INITIAL_LINEAR_VELOCITY)
        self.linear_velocity_spinbox.setDecimals(2)
        self.linear_velocity_spinbox.setStyleSheet("QDoubleSpinBox { color: white; background-color: #3a3a3a; padding: 5px; }")
        linear_velocity_layout.addWidget(self.linear_velocity_spinbox)
        avoid_layout.addLayout(linear_velocity_layout)
        
        # Rotation velocity control
        rotation_velocity_layout = QHBoxLayout()
        rotation_velocity_layout.addWidget(QLabel("Rotation Velocity:"))
        self.rotation_velocity_spinbox = QDoubleSpinBox()
        self.rotation_velocity_spinbox.setRange(0.0, 1.0)
        self.rotation_velocity_spinbox.setSingleStep(0.1)
        self.rotation_velocity_spinbox.setValue(GuiConfig.INITIAL_ROTATION_VELOCITY)
        self.rotation_velocity_spinbox.setDecimals(2)
        self.rotation_velocity_spinbox.setStyleSheet("QDoubleSpinBox { color: white; background-color: #3a3a3a; padding: 5px; }")
        rotation_velocity_layout.addWidget(self.rotation_velocity_spinbox)
        avoid_layout.addLayout(rotation_velocity_layout)
        
        # Velocity ramp control
        ramp_layout = QHBoxLayout()
        ramp_layout.addWidget(QLabel("Ramp Time (ms):"))
        self.ramp_spinbox = QSpinBox()
        self.ramp_spinbox.setRange(0, 5000)
        self.ramp_spinbox.setSingleStep(100)
        self.ramp_spinbox.setValue(GuiConfig.VELOCITY_RAMP_TIME_MS)
        self.ramp_spinbox.setSuffix(" ms")
        self.ramp_spinbox.setStyleSheet("QSpinBox { color: white; background-color: #3a3a3a; padding: 5px; }")
        ramp_layout.addWidget(self.ramp_spinbox)
        avoid_layout.addLayout(ramp_layout)
        
        avoid_group.setLayout(avoid_layout)
        layout.addWidget(avoid_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Apply dark theme to group boxes
        group_style = """
            QGroupBox {
                color: white;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """
        movement_group.setStyleSheet(group_style)
        posture_group.setStyleSheet(group_style)
        fun_group.setStyleSheet(group_style)
        avoid_group.setStyleSheet(group_style)


class GO2GuiClient(QMainWindow):
    """Main GUI window for GO2 robot control."""
    
    def __init__(self, relay_url: str):
        super().__init__()
        self.relay_url = relay_url
        # self.robot_config = robot_config
        # self.client: WebRTCRelayClient | None = None
        self.signals = RobotControlSignals()
        self.video_track: MediaStreamTrack | None = None
        self.video_task: asyncio.Task[None] | None = None
        self.invoker = GuiInvoker.make_invoker_on_gui_thread()
        
        # Movement state
        self.is_moving = False
        self.current_velocities = {"forward": 0.0, "strafe": 0.0, "rotation": 0.0}
        self.target_velocities = {"forward": 0.0, "strafe": 0.0, "rotation": 0.0}
        
        # Velocity ramping state
        self.velocity_ramp_time_ms = GuiConfig.VELOCITY_RAMP_TIME_MS
        self.ramp_start_times = {"forward": None, "strafe": None, "rotation": None}
        self.ramp_start_velocities = {"forward": 0.0, "strafe": 0.0, "rotation": 0.0}
        
        # Track pressed keys to ensure stop on release
        self.pressed_keys = set()
        
        self.init_ui()
        self.connect_signals()
        
        # Timer for continuous movement updates
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self.send_movement_command)
        self.move_timer.setInterval(GuiConfig.MOVEMENT_UPDATE_INTERVAL_MS)
        
        # Timer for velocity ramping
        self.ramp_timer = QTimer()
        self.ramp_timer.timeout.connect(self._update_velocity_ramp)
        self.ramp_timer.setInterval(20)  # 50 Hz for smooth ramping
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Buffer for latest lidar frame and low-rate update timer (0.2 Hz)
        self._latest_lidar_frame: t.Optional[dict[str, t.Any]] = None
        self._lidar_update_timer = QTimer()
        self._lidar_update_timer.timeout.connect(self._process_latest_lidar_frame)
        self._lidar_update_timer.setInterval(5000)  # 5000 ms = 0.2 Hz
        self._lidar_update_timer.start()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("GO2 Robot Control Center")
        self.setGeometry(100, 100, 1320, 900)
        self.setMinimumSize(800, 600)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: white;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Status bar at top
        self.status_widget = StatusWidget()
        main_layout.addWidget(self.status_widget)
        
        # Content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Video and Lidar
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        self.video_widget = VideoWidget()
        # Use VTK for embedded 3D visualization (fallback to Open3D button if VTK not available)
        self.lidar_widget = LidarWidget(use_voxel_viewer=False, use_vtk=True)
        
        left_layout.addWidget(self.video_widget, stretch=2)
        left_layout.addWidget(self.lidar_widget, stretch=1)
        left_panel.setLayout(left_layout)
        
        # Right panel: Controls and Odometry
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        self.control_panel = ControlPanel()
        self.odometry_widget = OdometryWidget()
        
        right_layout.addWidget(self.control_panel, stretch=2)
        right_layout.addWidget(self.odometry_widget, stretch=1)
        right_panel.setLayout(right_layout)
        
        # Add panels to splitter
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(content_splitter)
        
        # Connection controls at bottom
        connection_layout = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_connect.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #34ce57; }
        """)
        self.btn_disconnect.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #e04555; }
        """)
        self.btn_disconnect.setEnabled(False)
        
        connection_layout.addWidget(self.btn_connect)
        connection_layout.addWidget(self.btn_disconnect)
        main_layout.addLayout(connection_layout)
        
        central_widget.setLayout(main_layout)

    def add_client(self, client: WebRTCRelayClient):
        """Add WebRTCRelayClient to the GUI client."""
        self.client = client
    
    def connect_signals(self):
        """Connect Qt signals to slots."""
        # Control buttons (use appropriate velocity multiplier from spinbox)
        self.control_panel.btn_forward.pressed.connect(
            lambda: self.set_velocity("forward", self.control_panel.linear_velocity_spinbox.value()))
        self.control_panel.btn_forward.released.connect(lambda: self.set_velocity("forward", 0.0))
        
        self.control_panel.btn_backward.pressed.connect(
            lambda: self.set_velocity("forward", -self.control_panel.linear_velocity_spinbox.value()))
        self.control_panel.btn_backward.released.connect(lambda: self.set_velocity("forward", 0.0))
        
        self.control_panel.btn_strafe_left.pressed.connect(
            lambda: self.set_velocity("strafe", self.control_panel.linear_velocity_spinbox.value()))
        self.control_panel.btn_strafe_left.released.connect(lambda: self.set_velocity("strafe", 0.0))
        
        self.control_panel.btn_strafe_right.pressed.connect(
            lambda: self.set_velocity("strafe", -self.control_panel.linear_velocity_spinbox.value()))
        self.control_panel.btn_strafe_right.released.connect(lambda: self.set_velocity("strafe", 0.0))
        
        self.control_panel.btn_rotate_left.pressed.connect(
            lambda: self.set_velocity("rotation", self.control_panel.rotation_velocity_spinbox.value()))
        self.control_panel.btn_rotate_left.released.connect(lambda: self.set_velocity("rotation", 0.0))
        
        self.control_panel.btn_rotate_right.pressed.connect(
            lambda: self.set_velocity("rotation", -self.control_panel.rotation_velocity_spinbox.value()))
        self.control_panel.btn_rotate_right.released.connect(lambda: self.set_velocity("rotation", 0.0))
        
        self.control_panel.btn_stop.clicked.connect(self.stop_movement)
        
        # Posture buttons
        self.control_panel.btn_stand_up.clicked.connect(self.on_stand_up)
        self.control_panel.btn_recovery_stand.clicked.connect(self.on_recovery_stand)
        self.control_panel.btn_sit.clicked.connect(self.on_sit)
        self.control_panel.btn_lie_down.clicked.connect(self.on_lie_down)
        
        # Fun commands
        self.control_panel.btn_balance_stand.clicked.connect(self.on_balance_stand)
        
        # Obstacle avoidance
        self.control_panel.chk_obstacle_avoid.stateChanged.connect(self.on_obstacle_avoid_changed)
        
        # Connection buttons
        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_disconnect.clicked.connect(self.on_disconnect)
        
        # Data update signals
        self.signals.video_frame_ready.connect(self.video_widget.update_frame)
        self.signals.lidar_update.connect(self.on_lidar_update)
        self.signals.odometry_update.connect(self.on_odometry_update)
        self.signals.connection_status.connect(self.status_widget.set_connected)
        self.signals.status_message.connect(self.status_widget.set_info)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard input using configurable key bindings."""
        if event.isAutoRepeat():
            return
        
        key = event.key()
        self.pressed_keys.add(key)
        
        # Use configurable key bindings
        action = GuiConfig.get_action_for_key(key)
        
        if action == "forward":
            velocity_multiplier = self.control_panel.linear_velocity_spinbox.value()
            self.set_target_velocity("forward", velocity_multiplier)
        elif action == "backward":
            velocity_multiplier = self.control_panel.linear_velocity_spinbox.value()
            self.set_target_velocity("forward", -velocity_multiplier)
        elif action == "strafe_left":
            velocity_multiplier = self.control_panel.linear_velocity_spinbox.value()
            self.set_target_velocity("strafe", velocity_multiplier)
        elif action == "strafe_right":
            velocity_multiplier = self.control_panel.linear_velocity_spinbox.value()
            self.set_target_velocity("strafe", -velocity_multiplier)
        elif action == "rotate_left":
            velocity_multiplier = self.control_panel.rotation_velocity_spinbox.value()
            self.set_target_velocity("rotation", velocity_multiplier)
        elif action == "rotate_right":
            velocity_multiplier = self.control_panel.rotation_velocity_spinbox.value()
            self.set_target_velocity("rotation", -velocity_multiplier)
        elif action == "stop":
            self.stop_movement()
        elif action == "quit":
            self.close()
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle keyboard release - stop movement when keys are released."""
        if event.isAutoRepeat():
            return
        
        key = event.key()
        self.pressed_keys.discard(key)
        
        # Use configurable key bindings
        action = GuiConfig.get_action_for_key(key)
        
        if action == "forward" or action == "backward":
            # Check if other forward/backward key is still pressed
            forward_key = GuiConfig.get_key_for_action("forward")
            backward_key = GuiConfig.get_key_for_action("backward")
            if forward_key not in self.pressed_keys and backward_key not in self.pressed_keys:
                self.set_target_velocity("forward", 0.0)
        elif action == "strafe_left" or action == "strafe_right":
            # Check if other strafe key is still pressed
            strafe_left_key = GuiConfig.get_key_for_action("strafe_left")
            strafe_right_key = GuiConfig.get_key_for_action("strafe_right")
            if strafe_left_key not in self.pressed_keys and strafe_right_key not in self.pressed_keys:
                self.set_target_velocity("strafe", 0.0)
        elif action == "rotate_left" or action == "rotate_right":
            # Check if other rotate key is still pressed
            rotate_left_key = GuiConfig.get_key_for_action("rotate_left")
            rotate_right_key = GuiConfig.get_key_for_action("rotate_right")
            if rotate_left_key not in self.pressed_keys and rotate_right_key not in self.pressed_keys:
                self.set_target_velocity("rotation", 0.0)
    
    def set_target_velocity(self, direction: str, value: float):
        """Set target velocity for a specific direction (with ramping)."""
        self.target_velocities[direction] = value
        
        # Start ramping if velocity changed
        if value != self.current_velocities[direction]:
            self.ramp_start_velocities[direction] = self.current_velocities[direction]
            self.ramp_start_times[direction] = QtCore.QDateTime.currentMSecsSinceEpoch()
            
            # Start ramp timer if not already running
            if not self.ramp_timer.isActive():
                self.ramp_timer.start()
        
        # Start/update movement timer if any target velocity is non-zero
        if any(v != 0.0 for v in self.target_velocities.values()):
            if not self.move_timer.isActive():
                self.move_timer.start()
        else:
            self.move_timer.stop()
    
    def _update_velocity_ramp(self):
        """Update current velocities based on ramping logic."""
        current_time = QtCore.QDateTime.currentMSecsSinceEpoch()
        ramp_time_ms = self.control_panel.ramp_spinbox.value()
        
        all_ramped = True
        for direction in ["forward", "strafe", "rotation"]:
            target = self.target_velocities[direction]
            current = self.current_velocities[direction]
            
            if abs(target - current) < 0.001:
                # Already at target
                self.current_velocities[direction] = target
                continue
            
            if ramp_time_ms <= 0:
                # No ramping, set immediately
                self.current_velocities[direction] = target
                continue
            
            # Calculate ramp progress
            start_time = self.ramp_start_times.get(direction)
            if start_time is None:
                start_time = current_time
                self.ramp_start_times[direction] = start_time
                self.ramp_start_velocities[direction] = current
            
            elapsed = current_time - start_time
            progress = min(elapsed / ramp_time_ms, 1.0)
            
            # Interpolate between start and target
            start_vel = self.ramp_start_velocities[direction]
            self.current_velocities[direction] = start_vel + (target - start_vel) * progress
            
            if abs(target - self.current_velocities[direction]) >= 0.001:
                all_ramped = False
        
        # Stop ramp timer if all velocities are ramped
        if all_ramped:
            self.ramp_timer.stop()
        
        # Send command with current (ramped) velocities
        self._send_move_command()
    
    def set_velocity(self, direction: str, value: float):
        """Set velocity directly (used by button controls, bypasses ramping for immediate response)."""
        self.set_target_velocity(direction, value)
    
    def stop_movement(self):
        """Stop all movement immediately using STOPMOVE command."""
        self.target_velocities = {"forward": 0.0, "strafe": 0.0, "rotation": 0.0}
        self.current_velocities = {"forward": 0.0, "strafe": 0.0, "rotation": 0.0}
        self.pressed_keys.clear()
        self.move_timer.stop()
        self.ramp_timer.stop()
        
        # Send STOPMOVE command instead of move command with zeros
        if self.client:
            try:
                loop = getattr(self.client, "_loop", None)
                if loop is None:
                    logger.warning("Client event loop not available; cannot send stop_move command")
                    return

                # Schedule the client's async stop_move() coroutine on the client's event loop
                asyncio.run_coroutine_threadsafe(
                    self.client.stop_move(),
                    loop
                )
            except Exception as e:
                logger.warning(f"Failed to send stop_move command: {e}")
    
    def focusOutEvent(self, event: QFocusEvent):
        """Handle window focus loss - stop all movement for safety."""
        self.stop_movement()
        super().focusOutEvent(event)
    
    def send_movement_command(self):
        """Send current movement command (called by timer)."""
        self._send_move_command()
    
    def _send_move_command(self):
        """Async helper to send move command."""
        if self.client:
            try:
                # print("Sending move command")
                # loop = asyncio.get_event_loop()

                # # pass
                # loop.call_soon_threadsafe(
                #     self.client.move,
                #     self.current_velocities["forward"],
                #     self.current_velocities["strafe"],
                #     self.current_velocities["rotation"]
                # )
                logger.debug("Sending move command")
                # Use the event loop that the client is running on (set in client_main)
                loop = getattr(self.client, "_loop", None)
                if loop is None:
                    logger.warning("Client event loop not available; cannot send move command")
                    return

                # Schedule the client's async move() coroutine on the client's event loop
                asyncio.run_coroutine_threadsafe(
                    self.client.move(
                        self.current_velocities["forward"],
                        self.current_velocities["strafe"],
                        self.current_velocities["rotation"]
                    ),
                    loop
                )
            except Exception as e:
                logger.warning(f"Failed to send move command: {e}")

    def on_connect(self):
        pass
    #     """Connect to the robot."""
    #     try:
    #         self.signals.status_message.emit("Connecting...")
    #         self.btn_connect.setEnabled(False)
            
    #         self.client = WebRTCRelayClient(
    #             relay_url=self.relay_url,
    #             robot_config=self.robot_config,
    #             on_robot_data=self.handle_robot_data,
    #             on_video_track=self.handle_video_track,
    #             on_lidar_frame=self.handle_lidar_frame
    #         )
            
    #         await self.client.start(connect_go2=True)
            
    #         self.signals.connection_status.emit(True)
    #         self.signals.status_message.emit("Connected to robot")
    #         self.btn_disconnect.setEnabled(True)
            
    #     except Exception as e:
    #         logger.error(f"Connection failed: {e}")
    #         self.signals.status_message.emit(f"Connection failed: {e}")
    #         self.btn_connect.setEnabled(True)
    #         QMessageBox.critical(self, "Connection Error", f"Failed to connect: {e}")
    
    def on_disconnect(self):
        """Disconnect from the robot."""
        try:
            if self.video_task:
                self.video_task.cancel()
                self.video_task = None
            
            # if self.client:
            #     await self.client.shutdown()
            #     self.client = None
            
            self.signals.connection_status.emit(False)
            self.signals.status_message.emit("Disconnected")
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
    
    def on_stand_up(self):
        """Command robot to stand up."""
        pass
        # if self.client:
        #     await self.client.stand_up()
    
    def on_recovery_stand(self):
        """Command robot to recovery stand."""
        # if self.client:
        #     await self.client.recovery_stand()
    
    def on_sit(self):
        """Command robot to sit."""
        # if self.client:
        #     await self.client.sit_on_hind_legs()
    
    def on_lie_down(self):
        """Command robot to lie down."""
        # if self.client:
        #     await self.client.lie_down_on_belly()
    
    def on_balance_stand(self):
        """Command robot to perform balance stand."""
        if self.client:
            try:
                loop = getattr(self.client, "_loop", None)
                if loop is None:
                    logger.warning("Client event loop not available; cannot send balance_stand command")
                    return
                
                asyncio.run_coroutine_threadsafe(
                    self.client.balance_stand(),
                    loop
                )
            except Exception as e:
                logger.warning(f"Failed to send balance_stand command: {e}")
    
    def on_obstacle_avoid_changed(self, state: int):
        """Handle obstacle avoidance toggle."""
        # if self.client:
        #     enabled = state == Qt.CheckState.Checked
        #     await self.client.change_obstacle_avoid_state(enabled)
    
    def handle_robot_data(self, robot_data: RobotData):
        """Handle robot data updates."""
        try:
            if robot_data and robot_data.odometry_data:
                self.signals.odometry_update.emit(
                    robot_data.odometry_data.position,
                    robot_data.odometry_data.orientation
                )
        except Exception as e:
            logger.warning(f"Failed to handle robot data: {e}")
    
    def handle_video_track(self, frame):
        """Handle new video track."""
        # logger.info(f"Received video track")

        try:
            img = frame.to_ndarray(format="bgr24")  # pyright: ignore[reportAttributeAccessIssue]
            self.signals.video_frame_ready.emit(img)
        except asyncio.CancelledError:
            logger.info("Video stream processing cancelled")
        except Exception as e:
            logger.warning(f"Video stream error: {e}")
    
    def handle_lidar_frame(self, lidar_frame: dict[str, t.Any]):
        """Handle lidar data updates."""
        try:
            dec = lidar_frame["decoded_data"]
            meta = lidar_frame["data"]
            
            positions = dec["positions"]
            face_count = int(dec["face_count"])
            resolution = float(meta["resolution"])
            origin = tuple(meta["origin"])
            
            self.signals.lidar_update.emit(positions, face_count, resolution, origin)
        except Exception as e:
            logger.warning(f"Failed to handle lidar frame: {e}")
    
    def on_lidar_update(self, positions: npt.NDArray[np.uint8], face_count: int, resolution: float, origin: tuple[float, float, float]):
        """Update lidar widget."""
        self.lidar_widget.update_lidar_data(positions, face_count, resolution, origin)
    
    def update_latest_lidar_frame(self, lidar_frame: dict[str, t.Any]) -> None:
        """Store the most recent lidar frame (called on GUI thread)."""
        self._latest_lidar_frame = lidar_frame

    def _process_latest_lidar_frame(self) -> None:
        """Called by QTimer at 0.2 Hz to process the latest lidar frame if any."""
        if self._latest_lidar_frame is None:
            return
        try:
            # reuse existing handler to decode & emit signals
            self.handle_lidar_frame(self._latest_lidar_frame)
        except Exception as e:
            logger.warning(f"Failed to process latest lidar frame: {e}")
        finally:
            # clear buffer so we only process new incoming frames next tick
            self._latest_lidar_frame = None

    def on_odometry_update(self, position: dict[str, float], orientation: dict[str, float]):
        """Update odometry widget and lidar robot pose."""
        self.odometry_widget.update_odometry(position, orientation)
        self.lidar_widget.update_robot_pose(position, orientation)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop movement timer
        if self.move_timer.isActive():
            self.move_timer.stop()
        
        # Cancel video task
        if self.video_task and not self.video_task.done():
            self.video_task.cancel()
        
        # Close VoxelMapViewer if active
        if hasattr(self.lidar_widget, '_viewer_started') and self.lidar_widget._viewer_started:  # pyright: ignore[reportPrivateUsage]
            try:
                self.lidar_widget.stop_voxel_viewer()
            except Exception as e:
                logger.warning(f"Error closing VoxelMapViewer: {e}")
        
        # Disconnect client synchronously
        # if self.client:
        #     loop = asyncio.get_event_loop()
        #     if loop.is_running():
        #         loop.create_task(self._cleanup_and_close())
        #         event.ignore()  # Delay close until cleanup is done
        #         QTimer.singleShot(500, self.close)  # Force close after 500ms
        #     else:
        #         event.accept()
        # else:
        #     event.accept()
    
    # def _cleanup_and_close(self):
    #     """Async cleanup before close."""
    #     try:
    #         if self.client:
    #             await self.client.shutdown()
    #             self.client = None
    #     except Exception as e:
    #         logger.warning(f"Error during cleanup: {e}")

class GuiInvoker(QtCore.QObject):
    """Helper to run functions that originate from threads back onto main GUI thread"""

    call = QtCore.Signal(object)  # emits a Python callable

    @QtCore.Slot(object)  # type: ignore[reportCallIssue]
    def _run(self, fn: t.Callable[[], None]) -> None:
        fn()

    @classmethod
    def make_invoker_on_gui_thread(cls) -> 'GuiInvoker':
        """factory function for making this class"""
        app = QtWidgets.QApplication.instance()
        if app is None:
            raise RuntimeError("No QApplication running")
        inv = GuiInvoker()
        inv.moveToThread(app.thread())  # ensure GUI-thread affinity
        inv.call.connect(  # type: ignore[reportAttributeAccessIssue]
            inv._run, QtCore.Qt.ConnectionType.QueuedConnection
        )
        return inv
    
async def client_main(api, config, client, on_robot_data, on_video_track, on_lidar_frame):
    """Main client logic."""
    try:
        client._loop = asyncio.get_running_loop()
        await client.start(connect_go2=True)
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        await client.shutdown()
        logger.error(f"Error in client main: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PyQt5 GUI client for GO2 robot")
    parser.add_argument("--api", default="http://localhost:8000", help="WebRTC relay server URL")
    parser.add_argument("--robot-ip", default="192.168.12.1", help="GO2 robot IP address")
    parser.add_argument("--token", default="", help="Robot authentication token")
    args = parser.parse_args()
    

    # Create Qt application with async event loop
    app = QApplication(sys.argv)
    window = GO2GuiClient(relay_url=args.api)
    invoker = GuiInvoker.make_invoker_on_gui_thread()

    async def on_robot_data(robot_data: RobotData):
        def _helper():
            window.handle_robot_data(robot_data)
        invoker.call.emit(_helper)

    # Create robot config
    config = RobotConfig(
        robot_ip_list=[args.robot_ip],
        token=args.token,
        conn_type="webrtc",
        enable_video=True,
        decode_lidar=True,
        publish_raw_voxel=True,
        obstacle_avoidance=True,
        conn_mode='single'
    )

    video_task = None
    async def on_video_track(track: MediaStreamTrack):
        nonlocal video_task
        def _helper(frame):
            window.handle_video_track(frame)

        async def _video_task_runner():
            while True:
                frame = await track.recv()
                invoker.call.emit(lambda: _helper(frame))
    
        video_task = asyncio.create_task(_video_task_runner())

    # lidar_task = None
    async def on_lidar_frame(lidar_frame: dict[str, t.Any]):
        # nonlocal lidar_task
        # def _helper():
        #     window.update_latest_lidar_frame(lidar_frame)

        # async def _lidar_task_runner():
        #     while True:
        #         invoker.call.emit(_helper)

        # lidar_task = asyncio.create_task(_lidar_task_runner())
        def _helper():
            window.update_latest_lidar_frame(lidar_frame)
        invoker.call.emit(_helper)
        # pass

    client = WebRTCRelayClient(
        relay_url=args.api,
        robot_config=config,
        on_robot_data=on_robot_data,
        on_video_track=on_video_track,
        on_lidar_frame=on_lidar_frame,
    )

    window.add_client(client)

    def client_thread():
        try:
            asyncio.run(
                client_main(
                    api=args.api, 
                    config=config, 
                    client=client,
                    on_robot_data=on_robot_data, 
                    on_video_track=on_video_track, 
                    on_lidar_frame=on_lidar_frame
                )
            )
        except Exception as e:
            logger.error(f"Connection failed: {e}")

    import threading
    client_thread_handle = threading.Thread(target=client_thread)
    client_thread_handle.start()
        

    # Enable Ctrl+C handling
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # loop = QEventLoop(app)
    # asyncio.set_event_loop(loop)
    
    # Create and show main window
    window.show()
    
    # Run event loop
    try:
        sys.exit(app.exec_())
        # with loop:
            # loop.run_forever()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    finally:
        if video_task is not None:
            video_task.cancel()
        # Ensure cleanup
        # if window.client:
        #     try:
        #         loop.run_until_complete(window.client.shutdown())
        #     except:
        #         pass
        # loop.close()


if __name__ == "__main__":
    main()
