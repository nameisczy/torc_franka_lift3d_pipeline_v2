import pickle
from typing import Iterable, List, Literal, TypedDict

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import grey_dilation, binary_dilation
from skimage.morphology import disk, square
from skimage.draw import disk as draw_disk
from line_profiler import LineProfiler
from rich.console import Console
import cv2 as cv

import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

con = Console(highlight=False)


class DependencyEdge(TypedDict):
    source: int
    target: int
    relation: Literal["behind", "below"]


def print_mask(mask: np.ndarray, original: np.ndarray = None) -> None:
    # Print boolean mask as 0s and 1s
    # Use the con object and print 1s as blue, and 0s as grey
    con.print(f"Mask shape/type: {mask.shape} ({mask.dtype})")
    if original is not None:
        con.print(f"Original shape/type: {original.shape} ({original.dtype})")
    mask_str = ""
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            cell = mask[y, x]
            text = ""
            if cell and original is not None and original[y, x]:
                text = "[bold red]X[/bold red]"
            elif cell:
                text = "[bold cyan]#[/bold cyan]"
            else:
                text = "[grey]_[/grey]"
            mask_str += text
            if x < mask.shape[1] - 1:
                mask_str += " "
        mask_str += "\n"
    con.print(mask_str)


def get_behind_below_dependencies(
    mask: np.ndarray,
    depth_image: np.ndarray,
    cam_intr: np.ndarray,
    obj_ids: Iterable[int] = range(32),
) -> List[DependencyEdge]:
    """
    Given an array of image masks and a depth image, returns a list of mask pairs
    which are touching in the image where each pair is ordered based on which mask
    is behind or below another in the depth image with below taking precedence.

    Args:
        masks: Bit mask each representing an object/region
        depth_image: 2D numpy array representing depth values (lower values = closer)

    Returns:
        List of Dict ('source': mask_idx_i, 'target': mask_idx_j, 'relation': 'behind'|'below')
    """
    from perception.perception_fast import PerceptionInterface
    import open3d as o3d

    dependencies: List[DependencyEdge] = []

    # Convert bitmask to boolean masks for each object/region
    masks = [(mask & (1 << i)).astype(bool) for i in obj_ids]
    masks = {i: mask for i, mask in zip(obj_ids, masks) if np.any(mask)}

    pcd_cache1 = {}
    pcd_cache2 = {}

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    dilated_d = {
        i: cv.dilate(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)
        for i, mask in masks.items()
    }

    # Check each pair of masks
    for i, mask_i in masks.items():
        for j, mask_j in masks.items():
            # Avoid duplicate pairs and self-comparison
            if i >= j:
                continue

            # Find boundary pixels for each mask using dilation
            # Dilate mask and subtract original to get boundary
            dilated_i = dilated_d[i]
            dilated_j = dilated_d[j]

            # Check if boundaries overlap (masks are touching)
            touching_region = dilated_i & dilated_j

            if np.any(touching_region):
                # Masks are touching, now determine depth ordering

                # Get average depth for each mask in regions near the boundary
                # Expand touching region slightly to get more depth samples
                # expanded_touching = binary_dilation(touching_region, disk(2))
                expanded_touching = touching_region

                # Get depth values for mask pixels near the touching boundary
                mask_i_near_boundary = mask_i & expanded_touching
                mask_j_near_boundary = mask_j & expanded_touching

                if np.any(mask_i_near_boundary) and np.any(mask_j_near_boundary):
                    depths_i = depth_image[mask_i_near_boundary]
                    depths_j = depth_image[mask_j_near_boundary]

                    if len(depths_i) > 0 and len(depths_j) > 0:
                        # Check which is below first
                        pts_all = []
                        pts_mask = []
                        pts_vis = []
                        pcds = []
                        for ij, mask, mask_bnd in zip(
                            [i, j],
                            [mask_i, mask_j],
                            [mask_i_near_boundary, mask_j_near_boundary],
                        ):
                            if ij in pcd_cache1:
                                pcd, pmask, pts_bnd, inliers = pcd_cache1[ij]
                                pts_all.append(np.array(pcd.points))
                                pts_mask.append(pmask)
                                pts_vis.append(pts_bnd)
                                pcds.append(inliers)
                                continue
                            depth = np.array(depth_image)
                            depth[~mask] = 0
                            depth[mask_bnd] = 0
                            pcd_int = PerceptionInterface.create_pcd(depth, cam_intr)
                            pts_int = np.array(pcd_int.points)

                            depth = np.array(depth_image)
                            depth[~mask_bnd] = 0
                            pcd_bnd = PerceptionInterface.create_pcd(depth, cam_intr)
                            pts_bnd = np.array(pcd_bnd.points)

                            pts_all.append(np.vstack((pts_int, pts_bnd)))
                            pmask = np.zeros(len(pts_all[-1]), dtype=bool)
                            pmask[len(pts_int) :] = True
                            pts_mask.append(pmask)
                            pts_vis.append(pts_bnd)

                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pts_all[-1])
                            inliers, cinds = pcd.remove_statistical_outlier(
                                nb_neighbors=5, std_ratio=5.0
                            )
                            # display_inlier_outlier(pcd, cinds)
                            pcds.append(inliers)

                            pcd_cache1[ij] = (pcd, pmask, pts_bnd, inliers)

                        min_dist = min(pcds[0].compute_point_cloud_distance(pcds[1]))
                        # print("Min distance:", min_dist)
                        below = False
                        if min_dist < 6:
                            normals = []
                            for ij, mask, mask_bnd in zip(
                                [i, j],
                                [mask_i, mask_j],
                                [mask_i_near_boundary, mask_j_near_boundary],
                            ):
                                if ij in pcd_cache2:
                                    normal = pcd_cache2[ij]
                                    normals.append(normal)
                                    continue

                                pcd, _pmask, _pts_bnd, _inliers = pcd_cache1[ij]
                                pcd.estimate_normals(
                                    search_param=o3d.geometry.KDTreeSearchParamKNN(
                                        knn=30
                                    )
                                )
                                try:
                                    pcd.orient_normals_consistent_tangent_plane(k=15)
                                except:
                                    pass
                                pcd.normalize_normals()
                                normals.append(np.array(pcd.normals))

                                pcd_cache2[ij] = normals[-1]

                            down = [0, 1, 0]
                            aligned_i = np.dot(normals[0][pts_mask[0]], down) < -0.71
                            num_down_i = np.sum(aligned_i)
                            r_i = num_down_i / len(aligned_i)
                            aligned_j = np.dot(normals[1][pts_mask[1]], down) < -0.71
                            num_down_j = np.sum(aligned_j)
                            r_j = num_down_j / len(aligned_j)

                            # print("i: ", num_down_i, " / ", len(aligned_i), "   ", r_i)
                            # print("j: ", num_down_j, " / ", len(aligned_j), "   ", r_j)
                            # if r_i > 0.5 > r_j:
                            if num_down_i > 50 > num_down_j:
                                pts_aligned_i = pts_all[0][pts_mask[0]][aligned_i]
                                pts_masked_j = pts_all[1][pts_mask[1]]
                                ktree = KDTree(pts_masked_j)
                                dists, inds = ktree.query(pts_aligned_i, k=1)
                                closest_points = pts_masked_j[inds]
                                # tm.Scene([tm.points.PointCloud(pts_aligned_i,[255,0,0]), tm.points.PointCloud(closest_points,[0,0,255])]).show()
                                avg_z_i_aligned = -np.mean(pts_aligned_i, axis=0)[1]
                                avg_z_j = -np.mean(closest_points, axis=0)[1]
                                if avg_z_i_aligned < avg_z_j:  # add small tolerance
                                    dependencies.append(
                                        {"source": i, "target": j, "relation": "below"}
                                    )
                                    below = True
                            # elif r_j > 0.5 > r_i:
                            elif num_down_j > 50 > num_down_i:
                                pts_aligned_j = pts_all[1][pts_mask[1]][aligned_j]
                                pts_masked_i = pts_all[0][pts_mask[0]]
                                ktree = KDTree(pts_masked_i)
                                dists, inds = ktree.query(pts_aligned_j, k=1)
                                closest_points = pts_masked_i[inds]
                                # tm.Scene([tm.points.PointCloud(pts_aligned_j,[255,0,0]), tm.points.PointCloud(closest_points,[0,0,255])]).show()
                                avg_z_j_aligned = -np.mean(pts_aligned_j, axis=0)[1]
                                avg_z_i = -np.mean(closest_points, axis=0)[1]
                                if avg_z_j_aligned < avg_z_i:  # add small tolerance
                                    dependencies.append(
                                        {"source": j, "target": i, "relation": "below"}
                                    )
                                    below = True
                        if not below:
                            # neither are below so check which is behind
                            avg_depth_i = np.mean(depths_i)
                            avg_depth_j = np.mean(depths_j)

                            # Order by depth: behind mask (higher depth) first
                            if avg_depth_i > avg_depth_j:  # mask_i is behind mask_j
                                dependencies.append(
                                    {"source": i, "target": j, "relation": "behind"}
                                )
                            elif avg_depth_j > avg_depth_i:  # mask_j is behind mask_i
                                dependencies.append(
                                    {"source": j, "target": i, "relation": "behind"}
                                )
                            else:
                                # continue  # equal depth, skip
                                pass

                        # print(dependencies[-1])
                        # visualize normals for debugging
                        # com = np.mean(np.vstack(pts_all), axis=0)
                        # tmesh_i = vis_normals(pts_all[0], normals[0], 0.5)
                        # tmesh_j = vis_normals(pts_all[1], normals[1], 0.5)
                        # tmesh_down = vis_normals([com], [down], 10)
                        # pts_vis_i = tm.points.PointCloud(pts_vis[0],[255,0,0])
                        # pts_vis_j = tm.points.PointCloud(pts_vis[1],[0,0,255])
                        # pts_vis_k = tm.PointCloud([com],[0,255,0])
                        # tm.Scene([tmesh_i, tmesh_j, tmesh_down, pts_vis_i, pts_vis_j, pts_vis_k]).show()
    return dependencies


def dilation(
    mask: np.ndarray, kernel_size: int, debug_print: bool = False
) -> np.ndarray:
    if debug_print:
        con.print("Original mask:")
        print_mask(mask)
    for i in range(20):
        # Disk binary dilation
        kernel = disk(kernel_size // 2)
        dilated_mask = binary_dilation(mask, structure=kernel)
        if debug_print and i == 0:
            con.print("Disk binary dilation:")
            print_mask(dilated_mask, original=mask)

        # Square binary dilation
        kernel = square(kernel_size)
        dilated_mask = binary_dilation(mask, structure=kernel)
        if debug_print and i == 0:
            con.print("Square binary dilation:")
            print_mask(dilated_mask, original=mask)

        # Disk grey dilation
        kernel = disk(kernel_size // 2)
        dilated_mask = grey_dilation(mask, footprint=kernel)
        if debug_print and i == 0:
            con.print("Disk grey dilation:")
            print_mask(dilated_mask, original=mask)

        # Square grey dilation
        dilated_mask = grey_dilation(mask, size=(kernel_size, kernel_size))
        if debug_print and i == 0:
            con.print("Square grey dilation:")
            print_mask(dilated_mask, original=mask)

        # OpenCV ellipse dilation
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(
            bool
        )
        if debug_print and i == 0:
            con.print("OpenCV ellipse dilation:")
            print_mask(dilated_mask, original=mask)

        # OpenCV rectangle dilation
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
        dilated_mask = cv.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(
            bool
        )
        if debug_print and i == 0:
            con.print("OpenCV rectangle dilation:")
            print_mask(dilated_mask, original=mask)

        # OpenCV cross dilation
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
        dilated_mask = cv.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(
            bool
        )
        if debug_print and i == 0:
            con.print("OpenCV cross dilation:")
            print_mask(dilated_mask, original=mask)
    return dilated_mask


def make_circle_mask(size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    rr, cc = draw_disk((size // 2, size // 2), size // 4, shape=mask.shape)
    mask[rr, cc] = 1
    return mask


def test_get_behind_below_dependencies():
    params = pickle.load(open(f"{SCRIPT_DIR}/params.pkl", "rb"))
    print("params: ")
    # pretty print params
    for key, value in params.items():
        print(
            f"{key}: {type(value)} {value.shape if isinstance(value, np.ndarray) else value}"
        )

    np.set_printoptions(threshold=np.inf)

    mask = params["mask"]
    depth_image = params["depth_image"]
    cam_intr = params["cam_intr"]
    obj_ids = params.get("obj_ids", range(32))

    profiler = LineProfiler()
    result = profiler(get_behind_below_dependencies)(
        mask, depth_image, cam_intr, obj_ids=obj_ids
    )

    profiler.dump_stats(f"{SCRIPT_DIR}/get_behind_below_dependencies_results.lprof")


def test_dilation():
    mask = make_circle_mask(size=1920)

    profiler = LineProfiler()
    dilated_mask = profiler(dilation)(mask, kernel_size=5, debug_print=False)
    profiler.dump_stats(f"{SCRIPT_DIR}/dilation_results.lprof")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run debug tests")
    parser.add_argument(
        "test",
        nargs="?",
        default="all",
        choices=["all", "dilation", "dependencies"],
        help="Which test to run",
    )
    args = parser.parse_args()

    if args.test in ["all", "dependencies"]:
        test_get_behind_below_dependencies()
    if args.test in ["all", "dilation"]:
        test_dilation()
