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
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QThread
from PySide6.QtGui import QKeyEvent, QFocusEvent

from aiortc import MediaStreamTrack  # type: ignore

from go2_robot_sdk.domain.entities.robot_config import RobotConfig
from go2_robot_sdk.domain.entities.robot_data import RobotData
from go2_robot_sdk.webrtc_relay.webrtc_relay_client import WebRTCRelayClient
from go2_robot_sdk.webrtc_relay.gui_widgets import (
    VideoWidget, LidarWidget, OdometryWidget, StatusWidget
)
from go2_robot_sdk.webrtc_relay.gui_configurations import GuiConfig
from go2_robot_sdk.webrtc_relay.keyboard_command_handler import KeyboardCommandHandler, QtInputAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RobotControlSignals(QObject):
    """Qt signals for thread-safe updates from async callbacks."""
    video_frame_ready = Signal(np.ndarray)
    lidar_update = Signal(np.ndarray, int, float, tuple)  # positions, face_count, resolution, origin
    lidar_mesh_update = Signal(object)  # Open3D TriangleMesh
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
        
        # Obstacle avoidance buttons
        obstacle_avoid_layout = QHBoxLayout()
        obstacle_avoid_layout.addWidget(QLabel("Obstacle Avoidance:"))
        self.btn_enable_obstacle_avoid = QPushButton("Enable")
        self.btn_enable_obstacle_avoid.setStyleSheet(button_style + """
            QPushButton { background-color: #28a745; }
            QPushButton:hover { background-color: #34ce57; }
        """)
        self.btn_enable_obstacle_avoid.setMinimumHeight(30)
        self.btn_disable_obstacle_avoid = QPushButton("Disable")
        self.btn_disable_obstacle_avoid.setStyleSheet(button_style + """
            QPushButton { background-color: #dc3545; }
            QPushButton:hover { background-color: #e4606d; }
        """)
        self.btn_disable_obstacle_avoid.setMinimumHeight(30)
        obstacle_avoid_layout.addWidget(self.btn_enable_obstacle_avoid)
        obstacle_avoid_layout.addWidget(self.btn_disable_obstacle_avoid)
        avoid_layout.addLayout(obstacle_avoid_layout)
        
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
        
        # Keyboard command handler (will be initialized in add_client)
        self.keyboard_handler: KeyboardCommandHandler | None = None
        self.keyboard_adapter: QtInputAdapter | None = None
        
        self.init_ui()
        self.connect_signals()
        
        # Timer for continuous movement updates (will be connected to handler)
        self.move_timer = QTimer()
        self.move_timer.setInterval(GuiConfig.MOVEMENT_UPDATE_INTERVAL_MS)
        
        # Timer for velocity ramping (will be connected to handler)
        self.ramp_timer = QTimer()
        self.ramp_timer.setInterval(20)  # 50 Hz for smooth ramping
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Buffer for latest lidar frame and low-rate update timer (0.2 Hz)
        self._latest_lidar_frame: t.Optional[dict[str, t.Any]] = None
        self._lidar_update_timer = QTimer()
        self._lidar_update_timer.timeout.connect(self._process_latest_lidar_frame)
        self._lidar_update_timer.setInterval(5000)  # 5000 ms = 0.2 Hz
        self._lidar_update_timer.start()
        
        # Lidar processing thread (not asyncio)
        self.lidar_processing_thread = LidarProcessingThread(self)
        self.lidar_processing_thread.processed_data_ready.connect(self.on_lidar_data_processed)
        self.lidar_processing_thread.processed_mesh_ready.connect(self.on_lidar_mesh_processed)
        self.lidar_processing_thread.start()
    
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
        self.lidar_widget = LidarWidget()
        
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
        
        # Initialize keyboard command handler
        self.keyboard_handler = KeyboardCommandHandler(client)
        self.keyboard_adapter = self.keyboard_handler.create_qt_adapter()
        
        # Setup Qt timers with handler
        self.keyboard_handler.setup_qt_timers(self.ramp_timer, self.move_timer)
    
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
        
        # Obstacle avoidance buttons
        self.control_panel.btn_enable_obstacle_avoid.clicked.connect(self.on_enable_obstacle_avoid)
        self.control_panel.btn_disable_obstacle_avoid.clicked.connect(self.on_disable_obstacle_avoid)
        
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
        """Handle keyboard input using keyboard command handler."""
        if event.isAutoRepeat():
            return
        
        if self.keyboard_adapter:
            self.keyboard_adapter.handle_key_press(event.key())
        
        # Handle quit action separately (close window)
        if self.keyboard_handler:
            action = self.keyboard_handler._get_action_for_qt_key(event.key())
            if action == "quit":
                self.close()
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle keyboard release using keyboard command handler."""
        if event.isAutoRepeat():
            return
        
        if self.keyboard_adapter:
            self.keyboard_adapter.handle_key_release(event.key())
    
    def set_velocity(self, direction: str, value: float):
        """Set velocity directly (used by button controls)."""
        if self.keyboard_handler:
            self.keyboard_handler.set_target_velocity(direction, value)
    
    def stop_movement(self):
        """Stop all movement immediately using STOPMOVE command."""
        if self.keyboard_handler:
            self.keyboard_handler.stop_movement()
    
    def focusOutEvent(self, event: QFocusEvent):
        """Handle window focus loss - stop all movement for safety."""
        self.stop_movement()
        super().focusOutEvent(event)

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
    
    def on_enable_obstacle_avoid(self):
        """Enable obstacle avoidance."""
        if self.client:
            try:
                loop = getattr(self.client, "_loop", None)
                if loop is None:
                    logger.warning("Client event loop not available; cannot enable obstacle avoidance")
                    return
                
                asyncio.run_coroutine_threadsafe(
                    self.client.change_obstacle_avoid_state(True),
                    loop
                )
                logger.info("Enabled obstacle avoidance")
            except Exception as e:
                logger.warning(f"Failed to enable obstacle avoidance: {e}")
    
    def on_disable_obstacle_avoid(self):
        """Disable obstacle avoidance."""
        if self.client:
            try:
                loop = getattr(self.client, "_loop", None)
                if loop is None:
                    logger.warning("Client event loop not available; cannot disable obstacle avoidance")
                    return
                
                asyncio.run_coroutine_threadsafe(
                    self.client.change_obstacle_avoid_state(False),
                    loop
                )
                logger.info("Disabled obstacle avoidance")
            except Exception as e:
                logger.warning(f"Failed to disable obstacle avoidance: {e}")
    
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
        """Handle lidar data updates - queue to processing thread."""
        # Queue frame to processing thread (non-blocking)
        self.lidar_processing_thread.queue_lidar_frame(lidar_frame)
    
    def on_lidar_data_processed(self, positions: npt.NDArray[np.uint8], face_count: int, resolution: float, origin: tuple[float, float, float]):
        """Update lidar widget with processed voxel data (called from processing thread via signal)."""
        self.lidar_widget.update_lidar_data(positions, face_count, resolution, origin)
    
    def on_lidar_mesh_processed(self, mesh):
        """Update lidar widget with processed Open3D mesh (called from processing thread via signal)."""
        self.lidar_widget.update_lidar_mesh(mesh)
    
    def on_lidar_update(self, positions: npt.NDArray[np.uint8], face_count: int, resolution: float, origin: tuple[float, float, float]):
        """Update lidar widget (legacy signal handler - delegates to processed data handler)."""
        self.on_lidar_data_processed(positions, face_count, resolution, origin)
    
    def update_latest_lidar_frame(self, lidar_frame: dict[str, t.Any]) -> None:
        """Store the most recent lidar frame (called on GUI thread)."""
        self._latest_lidar_frame = lidar_frame

    def _process_latest_lidar_frame(self) -> None:
        """Called by QTimer at 0.2 Hz to process the latest lidar frame if any."""
        if self._latest_lidar_frame is None:
            return
        try:
            # Queue to processing thread instead of processing directly
            self.lidar_processing_thread.queue_lidar_frame(self._latest_lidar_frame)
        except Exception as e:
            logger.warning(f"Failed to queue lidar frame for processing: {e}")
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
        
        # Stop lidar processing thread
        if hasattr(self, 'lidar_processing_thread'):
            self.lidar_processing_thread.stop()
        
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

class LidarProcessingThread(QThread):
    """Thread for processing lidar data in background (not asyncio)."""
    
    # Signals emitted from this thread
    processed_data_ready = Signal(np.ndarray, int, float, tuple)  # positions, face_count, resolution, origin
    processed_mesh_ready = Signal(object)  # Open3D TriangleMesh
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lidar_frame_queue: list[dict[str, t.Any] | None] = [None]  # Single-item queue
        self._running = True
        
    def queue_lidar_frame(self, lidar_frame: dict[str, t.Any]):
        """Queue a lidar frame for processing (called from GUI thread)."""
        self._lidar_frame_queue[0] = lidar_frame
    
    def stop(self):
        """Stop the processing thread."""
        self._running = False
        self.wait()
    
    def run(self):
        """Process lidar frames in background thread."""
        try:
            import open3d as o3d
            from go2_robot_sdk.webrtc_relay.voxel_map_viewer import (
                _positions_u8_to_world_points, _triangles_from_faces
            )
        except ImportError as e:
            logger.error(f"Failed to import required modules for lidar processing: {e}")
            return
        
        while self._running:
            # Check for new lidar frame
            lidar_frame = self._lidar_frame_queue[0]
            if lidar_frame is None:
                self.msleep(10)  # Sleep 10ms if no data
                continue
            
            # Clear queue immediately
            self._lidar_frame_queue[0] = None
            
            try:
                # Extract data from frame
                dec = lidar_frame.get("decoded_data", {})
                meta = lidar_frame.get("data", {})
                
                positions = dec.get("positions")
                face_count = int(dec.get("face_count", 0))
                resolution = float(meta.get("resolution", 0.01))
                origin = tuple(meta.get("origin", (0.0, 0.0, 0.0)))
                
                if positions is None or face_count == 0:
                    continue
                
                # Option 1: Emit raw voxel data (for direct VTK rendering)
                self.processed_data_ready.emit(positions, face_count, resolution, origin)
                
                # Option 2: Generate Open3D mesh and emit
                try:
                    # Convert positions to world points
                    world_points = _positions_u8_to_world_points(
                        positions, resolution, origin
                    )
                    
                    # Generate triangles from faces
                    triangles = _triangles_from_faces(face_count, flip_winding=False)
                    
                    # Create Open3D mesh
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(world_points)
                    mesh.triangles = o3d.utility.Vector3iVector(triangles)
                    
                    # Compute normals for better visualization
                    mesh.compute_vertex_normals()
                    
                    # Emit mesh
                    self.processed_mesh_ready.emit(mesh)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate Open3D mesh: {e}")
                    # Continue even if mesh generation fails
                
            except Exception as e:
                logger.warning(f"Failed to process lidar frame in thread: {e}")
            
            # Small sleep to prevent busy-waiting
            self.msleep(5)


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
