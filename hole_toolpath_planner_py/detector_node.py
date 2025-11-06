from __future__ import annotations

from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Quaternion, Vector3
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from hole_toolpath_planner.msg import Hole, HoleArray
from hole_toolpath_planner.srv import DetectHoles

from .detection import (
    CylinderDetection,
    deduplicate,
    detect_cylindrical_holes,
    load_mesh,
    orthonormal_frame,
    quaternion_from_axes,
)


class HoleDetectorPyNode(Node):
    def __init__(self) -> None:
        super().__init__("hole_toolpath_planner_py")

        self.declare_parameter("sampling.points", 200000)
        self.declare_parameter("sampling.seed", 42)

        self.declare_parameter("cylinder.min_diameter", 0.002)
        self.declare_parameter("cylinder.max_diameter", 0.050)
        self.declare_parameter("cylinder.distance_threshold", 0.0005)
        self.declare_parameter("cylinder.ransac_n", 3)
        self.declare_parameter("cylinder.max_iterations", 2000)
        self.declare_parameter("cylinder.min_inliers", 200)

        self.declare_parameter("surface_circle.into_hint", [0.0, 0.0, 1.0])
        self.declare_parameter("detection.dedupe_center_tol", 0.0005)
        self.declare_parameter("detection.dedupe_diameter_tol", 0.0005)
        self.declare_parameter("rviz.marker_ns", "holes")
        self.declare_parameter("rviz.sphere_scale", 0.003)
        self.declare_parameter("rviz.axis_length", 0.020)
        self.declare_parameter("logging.frame_id", "world")

        self._hole_pub = self.create_publisher(HoleArray, "holes", 10)
        self._marker_pub = self.create_publisher(MarkerArray, "hole_markers", 10)

        self.create_service(DetectHoles, "detect_holes", self._on_detect_request)

        self.get_logger().info("hole_toolpath_planner_py ready")

    def _on_detect_request(
        self,
        request: DetectHoles.Request,
        response: DetectHoles.Response,
    ) -> DetectHoles.Response:
        try:
            mesh = load_mesh(request.mesh_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to load mesh '{request.mesh_path}': {exc}")
            response.holes = HoleArray()
            response.holes.header.frame_id = self.get_parameter("logging.frame_id").value
            response.holes.header.stamp = self.get_clock().now().to_msg()
            return response

        hint = np.array(self.get_parameter("surface_circle.into_hint").value, dtype=float)
        hint = hint / (np.linalg.norm(hint) or 1.0)

        request_min_diameter = max(0.0, float(request.min_diameter))
        request_max_diameter = max(0.0, float(request.max_diameter))
        if 0.0 < request_max_diameter < request_min_diameter:
            self.get_logger().warn(
                f"Request max_diameter ({request_max_diameter:.4f} m) is less than "
                f"min_diameter ({request_min_diameter:.4f} m); clamping to the minimum."
            )
            request_max_diameter = request_min_diameter

        param_min_diameter = float(self.get_parameter("cylinder.min_diameter").value)
        param_max_diameter = float(self.get_parameter("cylinder.max_diameter").value)

        min_diameter = max(param_min_diameter, request_min_diameter)
        if request_max_diameter > 0.0 and param_max_diameter > 0.0:
            max_diameter = min(param_max_diameter, request_max_diameter)
        elif request_max_diameter > 0.0:
            max_diameter = request_max_diameter
        else:
            max_diameter = param_max_diameter

        sampling_points = max(1, int(self.get_parameter("sampling.points").value))
        sampling_seed = int(self.get_parameter("sampling.seed").value)

        detections = detect_cylindrical_holes(
            mesh=mesh,
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            min_length=max(0.0, float(request.min_length)),
            sampling_points=sampling_points,
            sampling_seed=sampling_seed,
            distance_threshold=float(self.get_parameter("cylinder.distance_threshold").value),
            ransac_n=max(3, int(self.get_parameter("cylinder.ransac_n").value)),
            max_iterations=max(10, int(self.get_parameter("cylinder.max_iterations").value)),
            min_inliers=max(1, int(self.get_parameter("cylinder.min_inliers").value)),
            into_hint=hint,
        )

        deduped = deduplicate(
            detections,
            diameter_tol=float(self.get_parameter("detection.dedupe_diameter_tol").value),
            center_tol=float(self.get_parameter("detection.dedupe_center_tol").value),
        )

        summary_lines: list[str] | None = None
        if deduped:
            summary_lines = [
                f"Detected {len(deduped)} hole(s) in mesh '{request.mesh_path}':"
            ]
            for idx, det in enumerate(deduped):
                summary_lines.append(
                    f"  [{idx:02d}] center=({det.center[0]:.4f}, {det.center[1]:.4f}, {det.center[2]:.4f}) m, "
                    f"diameter={det.diameter * 1000.0:.3f} mm, length={det.length * 1000.0:.3f} mm"
                )

        stamp = self.get_clock().now().to_msg()
        frame_id = self.get_parameter("logging.frame_id").value
        array = self._assemble_response(deduped, stamp, frame_id)
        response.holes = array

        if array.holes:
            self._hole_pub.publish(array)
            markers = self._make_markers(array, stamp)
            self._marker_pub.publish(markers)
            if summary_lines is not None:
                self.get_logger().info("\n".join(summary_lines))
        else:
            self.get_logger().warn(f"No holes detected in mesh '{request.mesh_path}'")

        return response

    def _assemble_response(
        self,
        detections: List[CylinderDetection],
        stamp,
        frame_id: str,
    ) -> HoleArray:
        array = HoleArray()
        array.header.stamp = stamp
        array.header.frame_id = frame_id

        for idx, det in enumerate(detections):
            hole = Hole()
            hole.header.stamp = stamp
            hole.header.frame_id = frame_id
            hole.id = idx
            hole.kind = Hole.CYLINDER
            hole.diameter = float(det.diameter)
            hole.length = float(det.length)

            x_axis, y_axis, z_axis = orthonormal_frame(det.axis)
            quat = quaternion_from_axes(x_axis, y_axis, z_axis)

            hole.pose.position = Point(
                x=float(det.center[0]),
                y=float(det.center[1]),
                z=float(det.center[2]),
            )
            hole.pose.orientation = Quaternion(
                x=float(quat[0]),
                y=float(quat[1]),
                z=float(quat[2]),
                w=float(quat[3]),
            )
            hole.axis = Vector3(
                x=float(z_axis[0]),
                y=float(z_axis[1]),
                z=float(z_axis[2]),
            )

            array.holes.append(hole)

        return array

    def _make_markers(self, holes: HoleArray, stamp) -> MarkerArray:
        marker_ns = self.get_parameter("rviz.marker_ns").value
        sphere_scale = float(self.get_parameter("rviz.sphere_scale").value)
        axis_length = float(self.get_parameter("rviz.axis_length").value)

        markers = MarkerArray()
        marker_id = 0

        for hole in holes.holes:
            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = holes.header.frame_id
            sphere.ns = marker_ns
            sphere.id = marker_id
            marker_id += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = hole.pose
            sphere.scale.x = sphere.scale.y = sphere.scale.z = sphere_scale
            sphere.color.r = 0.1
            sphere.color.g = 0.7
            sphere.color.b = 0.3
            sphere.color.a = 0.8
            markers.markers.append(sphere)

            axis_marker = Marker()
            axis_marker.header.stamp = stamp
            axis_marker.header.frame_id = holes.header.frame_id
            axis_marker.ns = f"{marker_ns}_axis"
            axis_marker.id = marker_id
            marker_id += 1
            axis_marker.type = Marker.LINE_STRIP
            axis_marker.action = Marker.ADD
            axis_marker.scale.x = sphere_scale * 0.2
            axis_marker.color.r = 0.2
            axis_marker.color.g = 0.4
            axis_marker.color.b = 0.9
            axis_marker.color.a = 0.9

            origin = np.array(
                [hole.pose.position.x, hole.pose.position.y, hole.pose.position.z],
                dtype=float,
            )
            axis = np.array([hole.axis.x, hole.axis.y, hole.axis.z], dtype=float)
            axis = axis / (np.linalg.norm(axis) or 1.0)

            p0 = origin - axis * axis_length
            p1 = origin + axis * axis_length
            axis_marker.points = [
                Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            ]
            markers.markers.append(axis_marker)

            text = Marker()
            text.header.stamp = stamp
            text.header.frame_id = holes.header.frame_id
            text.ns = f"{marker_ns}_label"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = hole.pose.position.x
            text.pose.position.y = hole.pose.position.y
            text.pose.position.z = hole.pose.position.z + sphere_scale * 1.5
            text.scale.z = sphere_scale
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.9
            text.text = f"id={hole.id} ⌀={hole.diameter * 1000.0:.1f} mm"
            markers.markers.append(text)

        return markers


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HoleDetectorPyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
