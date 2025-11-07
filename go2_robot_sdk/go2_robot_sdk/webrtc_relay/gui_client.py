"""
PyQt5 GUI client for GO2 robot control with integrated video, lidar, and odometry display.
"""
import sys
import asyncio
import argparse
import logging
import typing as t
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QGridLayout, QGroupBox, QLabel, QLineEdit, QCheckBox,
    QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QKeyEvent

from qasync import QEventLoop, asyncSlot

from aiortc import MediaStreamTrack  # type: ignore

from go2_robot_sdk.domain.entities.robot_config import RobotConfig
from go2_robot_sdk.domain.entities.robot_data import RobotData
from go2_robot_sdk.webrtc_relay.webrtc_relay_client import WebRTCRelayClient
from go2_robot_sdk.webrtc_relay.gui_widgets import (
    VideoWidget, LidarWidget, OdometryWidget, StatusWidget
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class RobotControlSignals(QObject):
    """Qt signals for thread-safe updates from async callbacks."""
    video_frame_ready = pyqtSignal(np.ndarray)
    lidar_update = pyqtSignal(np.ndarray, int, float, tuple)
    odometry_update = pyqtSignal(dict, dict)
    connection_status = pyqtSignal(bool)
    status_message = pyqtSignal(str)


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
        
        # Create movement buttons
        self.btn_forward = QPushButton("↑ Forward (W)")
        self.btn_backward = QPushButton("↓ Backward (S)")
        self.btn_left = QPushButton("← Left (A)")
        self.btn_right = QPushButton("→ Right (D)")
        self.btn_rotate_left = QPushButton("⟲ Rotate Left (Z)")
        self.btn_rotate_right = QPushButton("⟳ Rotate Right (C)")
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
        
        for btn in [self.btn_forward, self.btn_backward, self.btn_left, 
                   self.btn_right, self.btn_rotate_left, self.btn_rotate_right, self.btn_stop]:
            btn.setStyleSheet(button_style)
            btn.setMinimumHeight(35)
            btn.setMaximumHeight(45)
        
        self.btn_stop.setStyleSheet(button_style + """
            QPushButton { background-color: #8b0000; }
            QPushButton:hover { background-color: #a00000; }
        """)
        
        # Layout movement buttons in a grid
        movement_layout.addWidget(self.btn_forward, 0, 1)
        movement_layout.addWidget(self.btn_left, 1, 0)
        movement_layout.addWidget(self.btn_stop, 1, 1)
        movement_layout.addWidget(self.btn_right, 1, 2)
        movement_layout.addWidget(self.btn_backward, 2, 1)
        movement_layout.addWidget(self.btn_rotate_left, 3, 0)
        movement_layout.addWidget(self.btn_rotate_right, 3, 2)
        
        movement_group.setLayout(movement_layout)
        layout.addWidget(movement_group)
        
        # Posture controls
        posture_group = QGroupBox("Posture Controls")
        posture_layout = QVBoxLayout()
        
        self.btn_stand_up = QPushButton("Stand Up")
        self.btn_sit = QPushButton("Sit Down")
        self.btn_lie_down = QPushButton("Lie Down")
        
        for btn in [self.btn_stand_up, self.btn_sit, self.btn_lie_down]:
            btn.setStyleSheet(button_style)
            btn.setMinimumHeight(30)
            btn.setMaximumHeight(40)
            posture_layout.addWidget(btn)
        
        posture_group.setLayout(posture_layout)
        layout.addWidget(posture_group)
        
        # Obstacle avoidance
        avoid_group = QGroupBox("Settings")
        avoid_layout = QVBoxLayout()
        
        self.chk_obstacle_avoid = QCheckBox("Enable Obstacle Avoidance")
        self.chk_obstacle_avoid.setStyleSheet("QCheckBox { color: white; padding: 5px; }")
        avoid_layout.addWidget(self.chk_obstacle_avoid)
        
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
        avoid_group.setStyleSheet(group_style)


class GO2GuiClient(QMainWindow):
    """Main GUI window for GO2 robot control."""
    
    def __init__(self, relay_url: str, robot_config: RobotConfig):
        super().__init__()
        self.relay_url = relay_url
        self.robot_config = robot_config
        self.client: WebRTCRelayClient | None = None
        self.signals = RobotControlSignals()
        self.video_track: MediaStreamTrack | None = None
        self.video_task: asyncio.Task | None = None
        
        # Movement state
        self.is_moving = False
        self.current_velocities = {"forward": 0.0, "strafe": 0.0, "rotation": 0.0}
        
        self.init_ui()
        self.connect_signals()
        
        # Timer for continuous movement updates
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self.send_movement_command)
        self.move_timer.setInterval(100)  # 10 Hz
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("GO2 Robot Control Center")
        self.setGeometry(100, 100, 1320, 800)
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
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Video and Lidar
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        self.video_widget = VideoWidget()
        # Use VTK for embedded 3D visualization (fallback to Open3D button if VTK not available)
        self.lidar_widget = LidarWidget(use_voxel_viewer=True, use_vtk=True)
        
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
    
    def connect_signals(self):
        """Connect Qt signals to slots."""
        # Control buttons
        self.control_panel.btn_forward.pressed.connect(lambda: self.set_velocity("forward", 0.5))
        self.control_panel.btn_forward.released.connect(lambda: self.set_velocity("forward", 0.0))
        
        self.control_panel.btn_backward.pressed.connect(lambda: self.set_velocity("forward", -0.5))
        self.control_panel.btn_backward.released.connect(lambda: self.set_velocity("forward", 0.0))
        
        self.control_panel.btn_left.pressed.connect(lambda: self.set_velocity("strafe", 0.5))
        self.control_panel.btn_left.released.connect(lambda: self.set_velocity("strafe", 0.0))
        
        self.control_panel.btn_right.pressed.connect(lambda: self.set_velocity("strafe", -0.5))
        self.control_panel.btn_right.released.connect(lambda: self.set_velocity("strafe", 0.0))
        
        self.control_panel.btn_rotate_left.pressed.connect(lambda: self.set_velocity("rotation", 0.5))
        self.control_panel.btn_rotate_left.released.connect(lambda: self.set_velocity("rotation", 0.0))
        
        self.control_panel.btn_rotate_right.pressed.connect(lambda: self.set_velocity("rotation", -0.5))
        self.control_panel.btn_rotate_right.released.connect(lambda: self.set_velocity("rotation", 0.0))
        
        self.control_panel.btn_stop.clicked.connect(self.stop_movement)
        
        # Posture buttons
        self.control_panel.btn_stand_up.clicked.connect(self.on_stand_up)
        self.control_panel.btn_sit.clicked.connect(self.on_sit)
        self.control_panel.btn_lie_down.clicked.connect(self.on_lie_down)
        
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
        """Handle keyboard input."""
        key = event.key()
        
        if key == Qt.Key_W:
            self.set_velocity("forward", 0.5)
        elif key == Qt.Key_S:
            self.set_velocity("forward", -0.5)
        elif key == Qt.Key_A:
            self.set_velocity("strafe", 0.5)
        elif key == Qt.Key_D:
            self.set_velocity("strafe", -0.5)
        elif key == Qt.Key_Z:
            self.set_velocity("rotation", 0.5)
        elif key == Qt.Key_C:
            self.set_velocity("rotation", -0.5)
        elif key == Qt.Key_Space:
            self.stop_movement()
    
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle keyboard release."""
        if event.isAutoRepeat():
            return
        
        key = event.key()
        
        if key in (Qt.Key_W, Qt.Key_S):
            self.set_velocity("forward", 0.0)
        elif key in (Qt.Key_A, Qt.Key_D):
            self.set_velocity("strafe", 0.0)
        elif key in (Qt.Key_Z, Qt.Key_C):
            self.set_velocity("rotation", 0.0)
    
    def set_velocity(self, direction: str, value: float):
        """Set velocity for a specific direction."""
        self.current_velocities[direction] = value
        
        # Send command immediately
        asyncio.create_task(self._send_move_command())
        
        # Start/update movement timer if any velocity is non-zero
        if any(v != 0.0 for v in self.current_velocities.values()):
            if not self.move_timer.isActive():
                self.move_timer.start()
        else:
            self.move_timer.stop()
    
    def stop_movement(self):
        """Stop all movement."""
        self.current_velocities = {"forward": 0.0, "strafe": 0.0, "rotation": 0.0}
        self.move_timer.stop()
        asyncio.create_task(self._send_move_command())
    
    def send_movement_command(self):
        """Send current movement command (called by timer)."""
        asyncio.create_task(self._send_move_command())
    
    async def _send_move_command(self):
        """Async helper to send move command."""
        if self.client:
            try:
                await self.client.move(
                    forward_velocity=self.current_velocities["forward"],
                    strafe_velocity=self.current_velocities["strafe"],
                    rotation_velocity=self.current_velocities["rotation"]
                )
            except Exception as e:
                logger.warning(f"Failed to send move command: {e}")
    
    @asyncSlot()
    async def on_connect(self):
        """Connect to the robot."""
        try:
            self.signals.status_message.emit("Connecting...")
            self.btn_connect.setEnabled(False)
            
            self.client = WebRTCRelayClient(
                relay_url=self.relay_url,
                robot_config=self.robot_config,
                on_robot_data=self.handle_robot_data,
                on_video_track=self.handle_video_track,
                on_lidar_frame=self.handle_lidar_frame
            )
            
            await self.client.start(connect_go2=True)
            
            self.signals.connection_status.emit(True)
            self.signals.status_message.emit("Connected to robot")
            self.btn_disconnect.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.signals.status_message.emit(f"Connection failed: {e}")
            self.btn_connect.setEnabled(True)
            QMessageBox.critical(self, "Connection Error", f"Failed to connect: {e}")
    
    @asyncSlot()
    async def on_disconnect(self):
        """Disconnect from the robot."""
        try:
            if self.video_task:
                self.video_task.cancel()
                self.video_task = None
            
            if self.client:
                await self.client.shutdown()
                self.client = None
            
            self.signals.connection_status.emit(False)
            self.signals.status_message.emit("Disconnected")
            self.btn_connect.setEnabled(True)
            self.btn_disconnect.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
    
    @asyncSlot()
    async def on_stand_up(self):
        """Command robot to stand up."""
        if self.client:
            await self.client.stand_up()
    
    @asyncSlot()
    async def on_sit(self):
        """Command robot to sit."""
        if self.client:
            await self.client.sit_on_hind_legs()
    
    @asyncSlot()
    async def on_lie_down(self):
        """Command robot to lie down."""
        if self.client:
            await self.client.lie_down_on_belly()
    
    @asyncSlot()
    async def on_obstacle_avoid_changed(self, state: int):
        """Handle obstacle avoidance toggle."""
        if self.client:
            enabled = state == Qt.Checked
            await self.client.change_obstacle_avoid_state(enabled)
    
    async def handle_robot_data(self, robot_data: RobotData):
        """Handle robot data updates."""
        try:
            if robot_data and robot_data.odometry_data:
                self.signals.odometry_update.emit(
                    robot_data.odometry_data.position,
                    robot_data.odometry_data.orientation
                )
        except Exception as e:
            logger.warning(f"Failed to handle robot data: {e}")
    
    async def handle_video_track(self, track: MediaStreamTrack):
        """Handle new video track."""
        logger.info(f"Received video track: {track}")
        
        # Cancel previous video task if any
        if self.video_task:
            self.video_task.cancel()
        
        self.video_track = track
        self.video_task = asyncio.create_task(self._process_video_stream())
    
    async def _process_video_stream(self):
        """Process video frames from the track."""
        try:
            while self.video_track:
                frame = await self.video_track.recv()
                img = frame.to_ndarray(format="bgr24")  # pyright: ignore[reportAttributeAccessIssue]
                self.signals.video_frame_ready.emit(img)
        except asyncio.CancelledError:
            logger.info("Video stream processing cancelled")
        except Exception as e:
            logger.warning(f"Video stream error: {e}")
    
    async def handle_lidar_frame(self, lidar_frame: dict[str, t.Any]):
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
    
    def on_lidar_update(self, positions: np.ndarray, face_count: int, resolution: float, origin: tuple):
        """Update lidar widget."""
        self.lidar_widget.update_lidar_data(positions, face_count, resolution, origin)
    
    def on_odometry_update(self, position: dict, orientation: dict):
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
        if hasattr(self.lidar_widget, '_viewer_started') and self.lidar_widget._viewer_started:
            try:
                self.lidar_widget.stop_voxel_viewer()
            except Exception as e:
                logger.warning(f"Error closing VoxelMapViewer: {e}")
        
        # Disconnect client synchronously
        if self.client:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._cleanup_and_close())
                event.ignore()  # Delay close until cleanup is done
                QTimer.singleShot(500, self.close)  # Force close after 500ms
            else:
                event.accept()
        else:
            event.accept()
    
    async def _cleanup_and_close(self):
        """Async cleanup before close."""
        try:
            if self.client:
                await self.client.shutdown()
                self.client = None
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PyQt5 GUI client for GO2 robot")
    parser.add_argument("--api", default="http://localhost:8000", help="WebRTC relay server URL")
    parser.add_argument("--robot-ip", default="192.168.12.1", help="GO2 robot IP address")
    parser.add_argument("--token", default="", help="Robot authentication token")
    args = parser.parse_args()
    
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
    
    # Create Qt application with async event loop
    app = QApplication(sys.argv)
    
    # Enable Ctrl+C handling
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Create and show main window
    window = GO2GuiClient(relay_url=args.api, robot_config=config)
    window.show()
    
    # Run event loop
    try:
        with loop:
            loop.run_forever()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    finally:
        # Ensure cleanup
        if window.client:
            try:
                loop.run_until_complete(window.client.shutdown())
            except:
                pass
        loop.close()


if __name__ == "__main__":
    main()
