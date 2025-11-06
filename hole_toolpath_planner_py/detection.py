from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import open3d as o3d


@dataclass
class CylinderDetection:
    center: np.ndarray  # entry point of the hole (top face)
    axis: np.ndarray    # unit vector pointing along the hole
    diameter: float
    length: float


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Mesh '{path}' is empty or could not be parsed.")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def quaternion_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    rot = np.column_stack((x, y, z))
    qw = math.sqrt(max(0.0, 1.0 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2.0
    denom = 4.0 * qw if abs(qw) > 1e-9 else 1.0
    qx = (rot[2, 1] - rot[1, 2]) / denom
    qy = (rot[0, 2] - rot[2, 0]) / denom
    qz = (rot[1, 0] - rot[0, 1]) / denom
    return np.array([qx, qy, qz, qw])


def orthonormal_frame(z_axis: np.ndarray, seed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = z_axis / (np.linalg.norm(z_axis) or 1.0)
    if seed is None or np.linalg.norm(seed) < 1e-6:
        seed = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = seed - np.dot(seed, z) * z
    if np.linalg.norm(x) < 1e-6:
        seed = np.array([0.0, 1.0, 0.0])
        x = seed - np.dot(seed, z) * z
    x /= np.linalg.norm(x) or 1.0
    y = np.cross(z, x)
    y /= np.linalg.norm(y) or 1.0
    x = np.cross(y, z)
    x /= np.linalg.norm(x) or 1.0
    return x, y, z


def deduplicate(detections: Sequence[CylinderDetection], diameter_tol: float, center_tol: float) -> List[CylinderDetection]:
    result: List[CylinderDetection] = []
    for det in detections:
        keep = True
        for existing in result:
            if np.linalg.norm(det.center - existing.center) > center_tol:
                continue
            if abs(det.diameter - existing.diameter) > diameter_tol:
                continue
            keep = False
            break
        if keep:
            result.append(det)
    return result


def detect_cylindrical_holes(
    mesh: o3d.geometry.TriangleMesh,
    *,
    min_diameter: float,
    max_diameter: float,
    min_length: float,
    sampling_points: int,
    sampling_seed: int,
    distance_threshold: float,
    ransac_n: int,
    max_iterations: int,
    min_inliers: int,
    into_hint: np.ndarray,
) -> List[CylinderDetection]:
    if sampling_seed >= 0:
        # Open3D >= 0.15 exposes utility.random.seed; older releases do not.
        random_backend = getattr(o3d.utility, "random", None)
        seed_fn = getattr(random_backend, "seed", None) if random_backend else None
        if callable(seed_fn):
            seed_fn(sampling_seed)
        random.seed(sampling_seed)
        np.random.seed(sampling_seed)

    hint = into_hint / (np.linalg.norm(into_hint) or 1.0)

    if _has_segment_cylinder():
        return _detect_cylinders_with_open3d(
            mesh=mesh,
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            min_length=min_length,
            sampling_points=sampling_points,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            max_iterations=max_iterations,
            min_inliers=min_inliers,
            hint=hint,
        )

    return _detect_cylinders_with_mesh_clustering(
        mesh=mesh,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
        min_length=min_length,
        distance_threshold=distance_threshold,
        min_inliers=min_inliers,
        hint=hint,
    )


def _has_segment_cylinder() -> bool:
    return hasattr(o3d.geometry.PointCloud, "segment_cylinder")


def _detect_cylinders_with_open3d(
    *,
    mesh: o3d.geometry.TriangleMesh,
    min_diameter: float,
    max_diameter: float,
    min_length: float,
    sampling_points: int,
    distance_threshold: float,
    ransac_n: int,
    max_iterations: int,
    min_inliers: int,
    hint: np.ndarray,
) -> List[CylinderDetection]:
    num_points = max(sampling_points, len(mesh.vertices))
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points, use_triangle_normal=True)
    detections: List[CylinderDetection] = []

    while True:
        points = np.asarray(point_cloud.points)
        if points.size == 0 or len(points) < max(min_inliers, ransac_n):
            break

        try:
            model, inliers = point_cloud.segment_cylinder(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=max_iterations,
            )
        except RuntimeError:
            break

        if len(inliers) < min_inliers:
            break

        axis_point = np.array(model[:3], dtype=float)
        axis_dir = np.array(model[3:6], dtype=float)
        radius = float(model[6])
        diameter = 2.0 * radius

        if max_diameter > 0.0 and diameter > max_diameter:
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            continue
        if diameter < min_diameter:
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            continue

        norm = np.linalg.norm(axis_dir)
        if norm < 1e-9:
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            continue
        axis_dir /= norm
        if axis_dir.dot(hint) < 0.0:
            axis_dir = -axis_dir

        inlier_points = points[np.asarray(inliers)]
        if inlier_points.size == 0:
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            continue

        t_values = (inlier_points - axis_point) @ axis_dir
        t_min = float(np.min(t_values))
        t_max = float(np.max(t_values))
        length = t_max - t_min
        if not np.isfinite(length) or length < min_length:
            point_cloud = point_cloud.select_by_index(inliers, invert=True)
            continue

        origin = axis_point + t_max * axis_dir

        detections.append(
            CylinderDetection(
                center=origin,
                axis=axis_dir,
                diameter=diameter,
                length=length,
            )
        )

        point_cloud = point_cloud.select_by_index(inliers, invert=True)

    return detections


def _detect_cylinders_with_mesh_clustering(
    *,
    mesh: o3d.geometry.TriangleMesh,
    min_diameter: float,
    max_diameter: float,
    min_length: float,
    distance_threshold: float,
    min_inliers: int,
    hint: np.ndarray,
) -> List[CylinderDetection]:
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)
    axis = hint / (np.linalg.norm(hint) or 1.0)

    cluster_ids = np.asarray(mesh.cluster_connected_triangles()[0])
    if cluster_ids.size == 0:
        return []

    min_cluster_tris = max(8, min_inliers // 15)

    detections: List[CylinderDetection] = []

    for cluster_id in np.unique(cluster_ids):
        tri_indices = np.where(cluster_ids == cluster_id)[0]
        if tri_indices.size < 6:
            continue

        normals = triangle_normals[tri_indices]
        alignment = np.abs(normals @ axis)
        if float(np.mean(alignment)) > 0.6:
            continue

        cyl_mask = alignment < 0.35
        if np.count_nonzero(cyl_mask) < min_cluster_tris:
            continue

        filtered_triangles = triangles[tri_indices[cyl_mask]]
        vertex_indices = np.unique(filtered_triangles.reshape(-1))
        cluster_points = vertices[vertex_indices]
        if cluster_points.shape[0] < 6:
            continue

        detection = _fit_cylinder_from_points(
            cluster_points=cluster_points,
            axis_hint=axis,
            min_length=min_length,
            min_diameter=min_diameter,
            max_diameter=max_diameter,
        )
        if detection is None:
            continue

        detections.append(detection)

    return detections


def _fit_cylinder_from_points(
    *,
    cluster_points: np.ndarray,
    axis_hint: np.ndarray,
    min_length: float,
    min_diameter: float,
    max_diameter: float,
) -> CylinderDetection | None:
    centroid = cluster_points.mean(axis=0)
    centered = cluster_points - centroid

    if not np.isfinite(centered).all():
        return None

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    axis_candidates = vh
    alignment = np.abs(axis_candidates @ axis_hint)
    axis_dir = axis_candidates[np.argmax(alignment)]
    axis_dir /= np.linalg.norm(axis_dir) or 1.0

    if axis_dir.dot(axis_hint) < 0.0:
        axis_dir = -axis_dir

    projections = centered @ axis_dir
    t_min = float(np.min(projections))
    t_max = float(np.max(projections))
    length = t_max - t_min
    if not np.isfinite(length) or length < min_length:
        return None

    axis_points = centroid + np.outer(projections, axis_dir)
    radii = np.linalg.norm(cluster_points - axis_points, axis=1)
    if radii.size == 0:
        return None

    radius = float(np.median(radii))
    if not np.isfinite(radius) or radius <= 0.0:
        return None

    diameter = 2.0 * radius
    if diameter < min_diameter:
        return None
    if max_diameter > 0.0 and diameter > max_diameter:
        return None

    origin = centroid + t_max * axis_dir

    return CylinderDetection(
        center=origin,
        axis=axis_dir,
        diameter=diameter,
        length=length,
    )
