"""
PyQt5 widgets for displaying robot data in the GUI.
"""
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGridLayout, QGroupBox, QPushButton
from PySide6.QtCore import Qt, Signal, QTimer  # pyright: ignore[reportUnusedImport]
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont  # pyright: ignore[reportUnusedImport]
import numpy as np
import numpy.typing as npt
import cv2
import typing as t  # pyright: ignore[reportUnusedImport]
import logging

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtkmodules.vtkRenderingOpenGL2  # pyright: ignore[reportUnusedImport]
    from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray  # pyright: ignore[reportUnusedImport]
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.vtkRenderingCore import (
        vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow,  # pyright: ignore[reportUnusedImport]
        vtkProperty, vtkCamera  # pyright: ignore[reportUnusedImport]
    )
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter  # pyright: ignore[reportUnusedImport]
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersSources import vtkCubeSource
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]
    print("Warning: VTK not available. Install with: pip install vtk")

from go2_robot_sdk.webrtc_relay.voxel_map_viewer import VoxelMapViewer

logger = logging.getLogger(__name__)


if VTK_AVAILABLE:
    # TODO: this should be moved to a different file so type hinting is cleaner, but not urgent
    class CustomInteractorStyle(vtkInteractorStyleTrackballCamera):
        """Custom VTK interactor style for pan and rotate controls.
        
        - Left click + drag: Pan the view
        - Ctrl + left click + drag: Rotate around Z-axis (vertical axis)
        - Mouse wheel: Zoom in/out
        - Right click + drag: Rotate (alternative)
        """
        
        def __init__(self, parent=None):
            super().__init__()
            self.AddObserver("LeftButtonPressEvent", self.left_button_press)
            self.AddObserver("LeftButtonReleaseEvent", self.left_button_release)
            self.AddObserver("MouseMoveEvent", self.mouse_move)
            self.is_ctrl_pressed = False
            self.last_mouse_x = 0
            self.last_mouse_y = 0
        
        def left_button_press(self, obj, event):
            """Handle left button press."""
            # Check if Ctrl key is pressed
            interactor = self.GetInteractor()
            self.is_ctrl_pressed = interactor.GetControlKey()
            
            # Store mouse position
            self.last_mouse_x, self.last_mouse_y = interactor.GetEventPosition()
            
            if self.is_ctrl_pressed:
                # Ctrl + click = rotate around Z-axis
                self.StartRotate()
            else:
                # Click alone = pan
                self.StartPan()
        
        def left_button_release(self, obj, event):
            """Handle left button release."""
            if self.is_ctrl_pressed:
                self.EndRotate()
            else:
                self.EndPan()
            self.is_ctrl_pressed = False
        
        def mouse_move(self, obj, event):
            """Handle mouse move."""
            interactor = self.GetInteractor()
            
            if self.is_ctrl_pressed and self.GetState() == 2:  # VTKIS_ROTATE
                # Get current mouse position
                x, y = interactor.GetEventPosition()
                
                # Calculate horizontal movement (for Z-axis rotation)
                dx = x - self.last_mouse_x
                
                # Get camera and focal point
                camera = self.GetCurrentRenderer().GetActiveCamera()
                focal_point = camera.GetFocalPoint()
                
                # Rotate around Z-axis at focal point
                # Positive dx = rotate counter-clockwise, negative = clockwise
                angle = dx * 0.5  # Sensitivity factor
                
                # Get camera position relative to focal point
                cam_pos = camera.GetPosition()
                rel_x = cam_pos[0] - focal_point[0]
                rel_y = cam_pos[1] - focal_point[1]
                rel_z = cam_pos[2] - focal_point[2]
                
                # Rotate around Z-axis
                import math
                angle_rad = math.radians(angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                
                new_x = rel_x * cos_a - rel_y * sin_a
                new_y = rel_x * sin_a + rel_y * cos_a
                
                # Set new camera position
                camera.SetPosition(
                    focal_point[0] + new_x,
                    focal_point[1] + new_y,
                    focal_point[2] + rel_z
                )
                
                # Update view up vector to maintain Z as up
                camera.SetViewUp(0, 0, 1)
                
                # Store current position for next move
                self.last_mouse_x = x
                self.last_mouse_y = y
                
                # Trigger render
                interactor.Render()
            else:
                # Normal mouse move behavior
                self.OnMouseMove()
                # Update last position
                self.last_mouse_x, self.last_mouse_y = interactor.GetEventPosition()



class VideoWidget(QWidget):
    """Widget for displaying video stream from the robot."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("Waiting for video...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(100, 100)
        self.label.setStyleSheet("QLabel { background-color: #1e1e1e; color: white; }")
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        logger.info(f"Label size: {self.label.size()}")
    
    def update_frame(self, frame: npt.NDArray[np.uint8]):
        """Update the video display with a new frame."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # # Scale to fit the label while maintaining aspect ratio
            # scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            #     self.label.size(), 
            #     Qt.AspectRatioMode.KeepAspectRatio, 
            #     Qt.TransformationMode.SmoothTransformation
            # )

            # Get label size
            label_size = self.label.size()
            label_width = label_size.width()
            label_height = label_size.height()

            # Skip if label has invalid size
            if label_width <= 0 or label_height <= 0:
                return

            # Calculate scale factor based on height to preserve full image height
            height_scale = label_height / h
            scaled_width = int(w * height_scale)
            scaled_height = label_height

            # Scale image to match calculated dimensions (full height, proportional width)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                scaled_width,
                scaled_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,  # We've already calculated correct aspect ratio
                Qt.TransformationMode.SmoothTransformation
            )

            # Create a black background pixmap matching the label size
            final_pixmap = QPixmap(label_width, label_height)
            final_pixmap.fill(QColor(0, 0, 0))  # Black background

            # Calculate centered position (horizontal centering, vertical is already full height)
            scaled_width_actual = scaled_pixmap.width()
            scaled_height_actual = scaled_pixmap.height()
            x_offset = (label_width - scaled_width_actual) // 2
            y_offset = (label_height - scaled_height_actual) // 2

            # Draw the scaled image centered on the black background
            painter = QPainter(final_pixmap)
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            painter.end()

            self.label.setPixmap(final_pixmap)
        except Exception as e:
            logger.warning(f"Failed to update video frame: {e}")


class LidarWidget(QWidget):
    """Widget for displaying lidar data visualization with 3D mesh rendering."""
    
    def __init__(self, parent=None, use_voxel_viewer=False, use_vtk=False):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("QWidget { background-color: #2d2d2d; }")
        
        self.use_voxel_viewer = use_voxel_viewer
        self.use_vtk = use_vtk and VTK_AVAILABLE
        self.voxel_viewer = None
        self._viewer_started = False
        
        self.positions = None
        self.face_count = 0
        self.resolution = 0.0
        self.origin = (0.0, 0.0, 0.0)
        
        # Robot pose
        self.robot_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.robot_orient = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        
        # Camera/view parameters for 2D rendering
        self.view_rotation = [0.0, 0.0]  # [azimuth, elevation]
        self.view_distance = 10.0
        self.view_center = np.array([0.0, 0.0, 0.0])
        self.last_mouse_pos = None
        
        layout = QVBoxLayout()
        self.info_label = QLabel("Lidar: Waiting for data... | Controls: Drag=Pan, Ctrl+Drag=Rotate Z-axis, Wheel=Zoom")
        self.info_label.setStyleSheet("QLabel { color: white; padding: 5px; background-color: #1e1e1e; }")
        layout.addWidget(self.info_label)
        
        if use_voxel_viewer and not self.use_vtk:
            # Button to toggle 3D viewer (Open3D)
            self.btn_toggle_3d = QPushButton("Open 3D Viewer (Open3D)")
            self.btn_toggle_3d.setStyleSheet("""
                QPushButton {
                    background-color: #4a4a4a;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #5a5a5a; }
            """)
            self.btn_toggle_3d.clicked.connect(self.toggle_3d_viewer)
            layout.addWidget(self.btn_toggle_3d)
            
            # Info label for 3D viewer
            self.viewer_info_label = QLabel("Click button above to open interactive 3D lidar viewer")
            self.viewer_info_label.setStyleSheet("QLabel { color: #aaa; padding: 10px; }")
            self.viewer_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.viewer_info_label.setWordWrap(True)
            layout.addWidget(self.viewer_info_label)
        
        # VTK widget or Canvas for visualization
        if self.use_vtk:
            self._init_vtk_widget()
            layout.addWidget(self.vtk_widget)
        else:
            # Canvas for 2D/3D visualization
            self.canvas_label = QLabel()
            self.canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.canvas_label.setStyleSheet("QLabel { background-color: #1a1a1a; }")
            self.canvas_label.setMinimumSize(400, 400)
            self.canvas_label.setMouseTracking(True)
            self.canvas_label.mousePressEvent = self.mouse_press_event
            self.canvas_label.mouseMoveEvent = self.mouse_move_event
            self.canvas_label.wheelEvent = self.wheel_event
            layout.addWidget(self.canvas_label)
        
        self.setLayout(layout)
    
    def _init_vtk_widget(self):
        """Initialize VTK visualization widget with professional settings."""
        if not VTK_AVAILABLE:
            logger.error("VTK not available")
            return
        
        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setMinimumSize(400, 400)
        
        # Create renderer
        self.vtk_renderer = vtkRenderer()
        
        # Dark gradient background (professional look)
        self.vtk_renderer.SetBackground(0.1, 0.1, 0.15)  # Dark blue-gray
        self.vtk_renderer.SetBackground2(0.05, 0.05, 0.1)  # Even darker at bottom
        self.vtk_renderer.SetGradientBackground(True)
        
        # Create render window with high quality settings
        self.vtk_render_window = self.vtk_widget.GetRenderWindow()
        self.vtk_render_window.AddRenderer(self.vtk_renderer)
        self.vtk_render_window.SetMultiSamples(8)  # Anti-aliasing
        self.vtk_render_window.LineSmoothingOn()
        self.vtk_render_window.PointSmoothingOn()
        self.vtk_render_window.PolygonSmoothingOn()
        
        # Enable high-quality rendering
        self.vtk_renderer.SetUseFXAA(True)  # Fast approximate anti-aliasing
        
        # Create mesh actor
        self.vtk_mesh_mapper = vtkPolyDataMapper()
        self.vtk_mesh_actor = vtkActor()
        self.vtk_mesh_actor.SetMapper(self.vtk_mesh_mapper)
        
        # Enhanced mesh appearance - semi-transparent with edges
        mesh_property = self.vtk_mesh_actor.GetProperty()

        # mesh_property.SetRepresentationToSurface()  # Solid surface
        # mesh_property.SetRepresentationToWireframe()  # Wire mesh
        mesh_property.SetRepresentationToPoints()  # Point cloud

        mesh_property.SetColor(0.6, 0.7, 0.9)  # Light blue
        mesh_property.SetOpacity(0.85)  # Slightly transparent
        mesh_property.SetAmbient(0.3)  # Ambient light component
        mesh_property.SetDiffuse(0.6)  # Diffuse light component
        mesh_property.SetSpecular(0.5)  # Specular highlights
        mesh_property.SetSpecularPower(30)  # Shininess
        mesh_property.EdgeVisibilityOn()  # Show edges
        mesh_property.SetEdgeColor(0.2, 0.2, 0.3)  # Dark blue-gray edges
        mesh_property.SetLineWidth(0.5)  # Thin edges
        
        self.vtk_renderer.AddActor(self.vtk_mesh_actor)
        
        # Create robot cube actor
        self.vtk_robot_source = vtkCubeSource()
        self.vtk_robot_source.SetXLength(0.4)
        self.vtk_robot_source.SetYLength(0.25)
        self.vtk_robot_source.SetZLength(0.15)
        
        self.vtk_robot_mapper = vtkPolyDataMapper()
        self.vtk_robot_mapper.SetInputConnection(self.vtk_robot_source.GetOutputPort())
        
        self.vtk_robot_actor = vtkActor()
        self.vtk_robot_actor.SetMapper(self.vtk_robot_mapper)
        
        # Enhanced robot appearance - glowing red
        robot_property = self.vtk_robot_actor.GetProperty()
        robot_property.SetColor(1.0, 0.2, 0.2)  # Bright red
        robot_property.SetOpacity(0.7)  # Semi-transparent
        robot_property.SetAmbient(0.6)  # More ambient = more glow
        robot_property.SetDiffuse(0.7)
        robot_property.SetSpecular(0.9)  # High specular for metallic look
        robot_property.SetSpecularPower(60)  # Very shiny
        
        self.vtk_renderer.AddActor(self.vtk_robot_actor)
        
        # Add coordinate axes for reference
        try:
            from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
            axes = vtkAxesActor()
            axes.SetTotalLength(1.0, 1.0, 1.0)  # 1 meter axes
            axes.SetShaftTypeToCylinder()
            axes.SetCylinderRadius(0.02)
            axes.SetConeRadius(0.05)
            axes.AxisLabelsOn()
            self.vtk_renderer.AddActor(axes)
        except ImportError:
            logger.warning("vtkAxesActor not available - skipping coordinate axes")
        
        # Setup camera - view from above and behind
        camera = self.vtk_renderer.GetActiveCamera()
        camera.SetPosition(8.0, -8.0, 6.0)  # Behind and above
        camera.SetFocalPoint(0.0, 0.0, 0.0)  # Look at origin
        camera.SetViewUp(0.0, 0.0, 1.0)  # Z-axis is up
        camera.SetViewAngle(45)  # Field of view
        camera.SetClippingRange(0.1, 100.0)  # Near and far clipping planes
        
        # Set up custom interactor style for pan and rotate
        style = CustomInteractorStyle()
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        
        # Initialize interactor
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
    
    def toggle_3d_viewer(self):
        """Toggle the 3D VoxelMapViewer."""
        if not self._viewer_started:
            self.start_voxel_viewer()
        else:
            self.stop_voxel_viewer()
    
    def start_voxel_viewer(self):
        """Start the VoxelMapViewer."""
        if not self._viewer_started:
            try:
                self.voxel_viewer = VoxelMapViewer(
                    window_name="GO2 Lidar 3D View",
                    flip_winding=False,
                    compute_normals_every=15,
                    robot_box_size=(0.4, 0.25, 0.15),
                    robot_color=(0.9, 0.2, 0.2)
                )
                self.voxel_viewer.start()
                self._viewer_started = True
                self.btn_toggle_3d.setText("Close 3D Viewer")
                self.viewer_info_label.setText("3D viewer is open - rotate/zoom with mouse in the Open3D window")
                logger.info("VoxelMapViewer started")
                
                # Send current data if available
                if self.positions is not None:
                    self.voxel_viewer.submit_u8(self.positions, self.face_count, self.resolution, self.origin)
                    self.voxel_viewer.submit_robot_pose(self.robot_pos, self.robot_orient)
            except Exception as e:
                logger.error(f"Failed to start VoxelMapViewer: {e}")
                self.viewer_info_label.setText(f"Error: {e}")
    
    def stop_voxel_viewer(self):
        """Stop the VoxelMapViewer."""
        if self._viewer_started and self.voxel_viewer:
            try:
                self.voxel_viewer.close()
                self.voxel_viewer = None
                self._viewer_started = False
                self.btn_toggle_3d.setText("Open 3D Viewer")
                self.viewer_info_label.setText("Click button above to open interactive 3D lidar viewer")
                logger.info("VoxelMapViewer stopped")
            except Exception as e:
                logger.error(f"Failed to stop VoxelMapViewer: {e}")
    
    def mouse_press_event(self, event):
        """Handle mouse press for view rotation."""
        self.last_mouse_pos = (event.x(), event.y())
    
    def mouse_move_event(self, event):
        """Handle mouse move for view rotation."""
        if self.last_mouse_pos and (event.buttons() & Qt.LeftButton):
            dx = event.x() - self.last_mouse_pos[0]
            dy = event.y() - self.last_mouse_pos[1]
            
            self.view_rotation[0] += dx * 0.5  # azimuth
            self.view_rotation[1] = np.clip(self.view_rotation[1] + dy * 0.5, -89, 89)  # elevation
            
            self.last_mouse_pos = (event.x(), event.y())
            self.update_visualization()
    
    def wheel_event(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        self.view_distance *= 0.9 if delta > 0 else 1.1
        self.view_distance = np.clip(self.view_distance, 1.0, 50.0)
        self.update_visualization()
    
    def update_lidar_data(self, positions: np.ndarray, face_count: int, resolution: float, origin: tuple):
        """Update lidar data and render visualization."""
        self.positions = positions
        self.face_count = face_count
        self.resolution = resolution
        self.origin = origin
        
        # Update view center to robot position on first data
        if self.view_center[0] == 0.0 and self.view_center[1] == 0.0 and self.view_center[2] == 0.0:
            self.view_center = np.array([self.robot_pos["x"], self.robot_pos["y"], self.robot_pos["z"]])
        
        # Update info label
        self.info_label.setText(
            f"Lidar: {face_count} faces, res={resolution:.3f}m, origin=({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})"
        )
        
        # Update VoxelMapViewer if active
        if self._viewer_started and self.voxel_viewer:
            try:
                self.voxel_viewer.submit_u8(positions, face_count, resolution, origin)
            except Exception as e:
                logger.warning(f"Failed to submit lidar data to VoxelMapViewer: {e}")
        
        # Update VTK visualization if enabled
        if self.use_vtk:
            self._update_vtk_mesh()
        else:
            self.update_visualization()
    
    def _update_vtk_mesh(self):
        """Update VTK mesh with current lidar data."""
        if not VTK_AVAILABLE or self.positions is None or self.face_count == 0:
            return
        
        try:
            # Decode positions to world coordinates
            pos = self.positions.reshape(-1, 3).astype(np.float32)
            pts = pos * self.resolution
            origin_arr = np.array(self.origin, dtype=np.float32)
            world_pts = pts + origin_arr
            
            # Create VTK points
            vtk_points = vtkPoints()
            for pt in world_pts:
                vtk_points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
            
            # Create triangles from faces
            triangles = vtkCellArray()
            for i in range(self.face_count):
                base_idx = i * 4
                if base_idx + 3 < len(world_pts):
                    # Create two triangles from quad
                    triangles.InsertNextCell(3)
                    triangles.InsertCellPoint(base_idx + 0)
                    triangles.InsertCellPoint(base_idx + 1)
                    triangles.InsertCellPoint(base_idx + 2)
                    
                    triangles.InsertNextCell(3)
                    triangles.InsertCellPoint(base_idx + 2)
                    triangles.InsertCellPoint(base_idx + 1)
                    triangles.InsertCellPoint(base_idx + 3)
            
            # Create polydata
            polydata = vtkPolyData()
            polydata.SetPoints(vtk_points)
            polydata.SetPolys(triangles)
            
            # Update mapper
            self.vtk_mesh_mapper.SetInputData(polydata)
            self.vtk_mesh_mapper.Update()
            
            # Render
            self.vtk_render_window.Render()
            
        except Exception as e:
            logger.warning(f"Failed to update VTK mesh: {e}")
    
    def _quat_to_vtk_transform(self, position: dict, orientation: dict) -> vtkTransform:
        """Convert quaternion to VTK transform."""
        import math
        
        # Extract quaternion components
        qx, qy, qz, qw = orientation["x"], orientation["y"], orientation["z"], orientation["w"]
        
        # Convert quaternion to rotation matrix
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        
        transform = vtkTransform()
        
        # Set rotation matrix
        matrix = [
            1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy), position["x"],
                2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx), position["y"],
                2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy), position["z"],
                           0,                0,                0,            1
        ]
        
        transform.SetMatrix(matrix)
        return transform
    
    def update_robot_pose(self, position: dict, orientation: dict):
        """Update robot pose."""
        self.robot_pos = position
        self.robot_orient = orientation
        
        # Update VoxelMapViewer if active
        if self._viewer_started and self.voxel_viewer:
            try:
                self.voxel_viewer.submit_robot_pose(position, orientation)
            except Exception as e:
                logger.warning(f"Failed to submit robot pose to VoxelMapViewer: {e}")
        
        # Update VTK robot position
        if self.use_vtk and VTK_AVAILABLE:
            try:
                transform = self._quat_to_vtk_transform(position, orientation)
                self.vtk_robot_actor.SetUserTransform(transform)
                self.vtk_render_window.Render()
            except Exception as e:
                logger.warning(f"Failed to update VTK robot pose: {e}")
        else:
            self.update_visualization()
    
    def update_visualization(self):
        """Render the 3D visualization to the canvas."""
        if self.positions is None or self.face_count == 0:
            return
        
        try:
            # Decode positions to world coordinates
            pos = self.positions.reshape(-1, 3).astype(np.float32)
            pts = pos * self.resolution
            origin_arr = np.array(self.origin, dtype=np.float32)
            world_pts = pts + origin_arr
            
            # Create image
            width = self.canvas_label.width()
            height = self.canvas_label.height()
            if width < 10 or height < 10:
                return
            
            img = np.ones((height, width, 3), dtype=np.uint8) * 20
            
            # Calculate camera position using spherical coordinates
            azimuth_rad = np.radians(self.view_rotation[0])
            elevation_rad = np.radians(self.view_rotation[1])
            
            # Camera orbits around view center
            cam_x = self.view_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            cam_y = self.view_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            cam_z = self.view_distance * np.sin(elevation_rad)
            
            cam_pos = self.view_center + np.array([cam_x, cam_y, cam_z])
            
            # Build camera coordinate system
            forward = self.view_center - cam_pos
            forward = forward / (np.linalg.norm(forward) + 1e-6)
            
            right = np.cross(forward, np.array([0, 0, 1]))
            right_norm = np.linalg.norm(right)
            if right_norm > 1e-6:
                right = right / right_norm
            else:
                right = np.array([1, 0, 0])
            
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-6)
            
            # Projection parameters
            focal_length = min(width, height) * 0.6
            
            # Draw points
            point_count = 0
            for pt in world_pts[::2]:  # Sample every 2nd point
                # Transform to camera space
                rel_pt = pt - cam_pos
                cam_x = np.dot(rel_pt, right)
                cam_y = np.dot(rel_pt, forward)
                cam_z = np.dot(rel_pt, up)
                
                # Project if in front of camera
                if cam_y > 0.1:
                    x_proj = int(width / 2 + (cam_x / cam_y) * focal_length)
                    y_proj = int(height / 2 - (cam_z / cam_y) * focal_length)
                    
                    if 0 <= x_proj < width and 0 <= y_proj < height:
                        # Color by height
                        color_val = int(128 + 127 * np.clip(cam_z / 2.0, -1, 1))
                        cv2.circle(img, (x_proj, y_proj), 1, (color_val, color_val, 255), -1)
                        point_count += 1
            
            # Draw robot position
            robot_pt = np.array([self.robot_pos["x"], self.robot_pos["y"], self.robot_pos["z"]])
            rel_robot = robot_pt - cam_pos
            robot_cam_x = np.dot(rel_robot, right)
            robot_cam_y = np.dot(rel_robot, forward)
            robot_cam_z = np.dot(rel_robot, up)
            
            if robot_cam_y > 0.1:
                x_proj = int(width / 2 + (robot_cam_x / robot_cam_y) * focal_length)
                y_proj = int(height / 2 - (robot_cam_z / robot_cam_y) * focal_length)
                if 0 <= x_proj < width and 0 <= y_proj < height:
                    cv2.circle(img, (x_proj, y_proj), 8, (50, 50, 255), -1)
                    cv2.circle(img, (x_proj, y_proj), 10, (100, 100, 255), 2)
            
            # Add info text
            cv2.putText(img, "Drag: Rotate | Wheel: Zoom", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(img, f"Points: {point_count}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Convert to Qt image
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            self.canvas_label.setPixmap(pixmap)
            
        except Exception as e:
            logger.warning(f"Failed to render lidar visualization: {e}")
    
    def closeEvent(self, event):
        """Cleanup when widget is closed."""
        if self._viewer_started and self.voxel_viewer:
            try:
                self.voxel_viewer.close()
            except Exception as e:
                logger.warning(f"Error closing VoxelMapViewer: {e}")
        
        if self.use_vtk and VTK_AVAILABLE:
            try:
                self.vtk_widget.Finalize()
            except Exception as e:
                logger.warning(f"Error finalizing VTK widget: {e}")
        
        super().closeEvent(event)


class OdometryWidget(QWidget):
    """Widget for displaying odometry information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        
        # Create group box
        group = QGroupBox("Odometry Data")
        group.setStyleSheet("""
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
            QLabel {
                color: #ddd;
                padding: 2px;
            }
        """)
        
        layout = QGridLayout()
        
        # Position labels
        self.pos_x_label = QLabel("X: 0.000")
        self.pos_y_label = QLabel("Y: 0.000")
        self.pos_z_label = QLabel("Z: 0.000")
        
        # Orientation labels
        self.orient_x_label = QLabel("Qx: 0.000")
        self.orient_y_label = QLabel("Qy: 0.000")
        self.orient_z_label = QLabel("Qz: 0.000")
        self.orient_w_label = QLabel("Qw: 1.000")
        
        # Add to layout
        layout.addWidget(QLabel("<b>Position:</b>"), 0, 0, 1, 2)
        layout.addWidget(self.pos_x_label, 1, 0)
        layout.addWidget(self.pos_y_label, 1, 1)
        layout.addWidget(self.pos_z_label, 2, 0)
        
        layout.addWidget(QLabel("<b>Orientation:</b>"), 3, 0, 1, 2)
        layout.addWidget(self.orient_x_label, 4, 0)
        layout.addWidget(self.orient_y_label, 4, 1)
        layout.addWidget(self.orient_z_label, 5, 0)
        layout.addWidget(self.orient_w_label, 5, 1)
        
        group.setLayout(layout)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(group)
        main_layout.addStretch()
        self.setLayout(main_layout)
    
    def update_odometry(self, position: dict, orientation: dict):
        """Update odometry display."""
        try:
            self.pos_x_label.setText(f"X: {position['x']:.3f}")
            self.pos_y_label.setText(f"Y: {position['y']:.3f}")
            self.pos_z_label.setText(f"Z: {position['z']:.3f}")
            
            self.orient_x_label.setText(f"Qx: {orientation['x']:.3f}")
            self.orient_y_label.setText(f"Qy: {orientation['y']:.3f}")
            self.orient_z_label.setText(f"Qz: {orientation['z']:.3f}")
            self.orient_w_label.setText(f"Qw: {orientation['w']:.3f}")
        except Exception as e:
            logger.warning(f"Failed to update odometry display: {e}")


class StatusWidget(QWidget):
    """Widget for displaying connection and system status."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.connection_label = QLabel("Status: Disconnected")
        self.connection_label.setStyleSheet("""
            QLabel {
                color: #ff5555;
                background-color: #2d2d2d;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("QLabel { color: #aaa; padding: 4px; }")
        
        layout = QVBoxLayout()
        layout.addWidget(self.connection_label)
        layout.addWidget(self.info_label)
        self.setLayout(layout)
    
    def set_connected(self, connected: bool):
        """Update connection status."""
        if connected:
            self.connection_label.setText("Status: Connected")
            self.connection_label.setStyleSheet("""
                QLabel {
                    color: #50fa7b;
                    background-color: #2d2d2d;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
        else:
            self.connection_label.setText("Status: Disconnected")
            self.connection_label.setStyleSheet("""
                QLabel {
                    color: #ff5555;
                    background-color: #2d2d2d;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
    
    def set_info(self, text: str):
        """Update info text."""
        self.info_label.setText(text)
