"""
Implement the graph representing relationships between objects or regions
"""

import numpy as np
import networkx as nx
from typing import Iterable, List, Literal, TypedDict
from netgraph import InteractiveGraph
from matplotlib.pyplot import show, figure
import trimesh as tm
import open3d as o3d
from utils.visual_utils import from_color_map
from scipy.spatial import KDTree
from skimage.morphology import binary_dilation, disk
from perception.perception_fast import PerceptionInterface
from utils.visual_utils import vis_normals
import pickle


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    # outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    tm.Scene(
        [
            tm.points.PointCloud(np.array(inlier_cloud.points), [200, 200, 200]),
            tm.points.PointCloud(np.array(outlier_cloud.points), [255, 0, 0]),
        ]
    ).show()


def get_behind_candidates(mask, depth_image, obj_ids=range(32)):
    """
    Given an array of image masks and a depth image, returns a list of mask pairs
    which are touching in the image where each pair is ordered based on which mask
    is behind another in the depth image.

    Args:
        masks: Bit mask each representing an object/region
        depth_image: 2D numpy array representing depth values (lower values = closer)

    Returns:
        List of tuples (mask_idx_behind, mask_idx_front) where:
        - mask_idx_behind: index of the mask that is further away (higher depth)
        - mask_idx_front: index of the mask that is closer (lower depth)
        - Only includes pairs where masks are actually touching
    """

    touching_pairs = []

    # Convert bitmask to boolean masks for each object/region
    masks = [(mask & (1 << i)).astype(bool) for i in obj_ids]
    masks = {i: mask for i, mask in zip(obj_ids, masks) if np.any(mask)}

    # Check each pair of masks
    for i, mask_i in masks.items():
        for j, mask_j in masks.items():
            # Avoid duplicate pairs and self-comparison
            if i >= j:
                continue

            # Find boundary pixels for each mask using dilation
            # Dilate mask and subtract original to get boundary
            kernel = disk(1)  # 3x3 structuring element
            dilated_i = binary_dilation(mask_i, kernel)
            dilated_j = binary_dilation(mask_j, kernel)

            # Check if boundaries overlap (masks are touching)
            touching_region = dilated_i & dilated_j

            if np.any(touching_region):
                # Masks are touching, now determine depth ordering

                # Get average depth for each mask in regions near the boundary
                # Expand touching region slightly to get more depth samples
                expanded_touching = binary_dilation(touching_region, disk(2))

                # Get depth values for mask pixels near the touching boundary
                mask_i_near_boundary = mask_i & expanded_touching
                mask_j_near_boundary = mask_j & expanded_touching

                if np.any(mask_i_near_boundary) and np.any(mask_j_near_boundary):
                    depths_i = depth_image[mask_i_near_boundary]
                    depths_j = depth_image[mask_j_near_boundary]

                    if len(depths_i) > 0 and len(depths_j) > 0:
                        avg_depth_i = np.mean(depths_i)
                        avg_depth_j = np.mean(depths_j)

                        # Order by depth: behind mask (higher depth) first
                        if avg_depth_i > avg_depth_j:  # mask_i is behind mask_j
                            touching_pairs.append((i, j))
                        elif avg_depth_j > avg_depth_i:  # mask_j is behind mask_i
                            touching_pairs.append((j, i))

    return touching_pairs


class DependencyEdge(TypedDict):
    source: int
    target: int
    relation: Literal["behind", "below"]


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

    dependencies: List[DependencyEdge] = []

    # Convert bitmask to boolean masks for each object/region
    masks = [(mask & (1 << i)).astype(bool) for i in obj_ids]
    masks = {i: mask for i, mask in zip(obj_ids, masks) if np.any(mask)}

    pcd_cache1 = {}
    pcd_cache2 = {}
    kernel = disk(4)  # 3x3 structuring element
    dilated_d = {i: binary_dilation(mask, kernel) for i, mask in masks.items()}
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
                                    # pcd.orient_normals_consistent_tangent_plane(k=15)
                                    pcd.orient_normals_toward_camera_location(np.array([0,0,0]))
                                    pcd.normals *= -1
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


def vis_depends(vA, oA, vB, oB, eps=1e-3, behind_axis=0, up_axis=2):
    """
    Check if axis-aligned box of trimesh object A is touching axis-aligned box
    of trimesh obejct B, and simultaneously whether A's center of mass is behind
    *or* below B.

    Definitions (customizable via axes):
      - Touching: AABB intervals overlap or just meet on all 3 axes.
      - Behind:
      - Below:

    Args:
        vA, vB: trimesh.Trimesh objects corresponding to visual points.
        oA, oB: trimesh.Trimesh objects corresponding to occlusion points.
        eps: small tolerance for robust comparisons.
        behind_axis: axis index defining "behind" (default 0 for X).
        up_axis: axis index defining "up" (default 2 for Z).

    Returns:
        result: bool  # touching and (behind or below)
    """

    behind = False
    try:
        vAoB = vA.bounding_box.intersection(oB.convex_hull)
        # print("vAoB volume", vAoB.volume)
        if vAoB.volume > 0:
            # behind = vA.bounds[0][behind_axis] + eps > vB.centroid[behind_axis]
            # behind &= vA.centroid[behind_axis] + eps > vB.bounds[1][behind_axis]
            above = vB.bounds[1][up_axis] - eps < vA.centroid[up_axis]
            above &= vB.centroid[up_axis] - eps < vA.bounds[0][up_axis]
            behind = vA.bounds[0][behind_axis] - eps > vB.centroid[behind_axis]
            behind &= not above
    except Exception as e:
        print(e)

    below = False
    try:
        vAvB = vA.bounding_box.intersection(vB.bounding_box)
        # print("vAvB volume", vAvB.volume)
        if vAvB.volume > 0:
            below = vA.bounds[1][up_axis] - eps < vB.centroid[up_axis]
            below &= vA.centroid[up_axis] - eps < vB.bounds[0][up_axis]
            # vA_vB = vA.bounding_box.difference(vB.bounding_box)
            # vB_vA = vB.bounding_box.difference(vA.bounding_box)
            # vA_vB.visual.face_colors = [255, 0, 0]
            # vAvB.visual.face_colors = [0, 255, 0]
            # vB_vA.visual.face_colors = [0, 0, 255]
            # tm.Scene([vAvB, vA_vB, vB_vA]).show()
    except Exception as e:
        print(e)

    return behind, below


class DepGraph:

    def __init__(self, obj_deps, pickable):
        self.target_id = 0
        self.hidden_id = -1
        # set graph attributes
        self.dict_graph = obj_deps
        # create networkx graph
        self.nx_graph = nx.from_dict_of_dicts(obj_deps, create_using=nx.DiGraph)
        # add pickability info
        for k, v in pickable.items():
            self.nx_graph.nodes[k]["g"] = v

    @classmethod
    def from_grasps(
        self,
        grasp_collisions,  # dict of sets {grasp_index: set(blocking_obj_ids)}
        obj_ids,  # list of object ids in order corresponding to grasps indices
        pickable,  # dict {obj_id: number of grasps} for pickable objects
    ):
        obj_deps = {}
        for i in range(len(obj_ids)):
            oid = obj_ids[i]
            if oid not in obj_deps:
                obj_deps[oid] = {}
            for did in grasp_collisions[i]:
                if oid == did:
                    continue  # no self edges
                if did not in obj_deps[oid]:
                    obj_deps[oid][did] = {"g": 0, "d": "grasps blocked by"}
                obj_deps[oid][did]["g"] += 1

        # create DepGraph instance
        instance = DepGraph(obj_deps, pickable)

        # equally weigh occlusion info
        for i in instance.nx_graph.nodes:
            instance.nx_graph.nodes[i]["v"] = 0

        return instance

    @classmethod
    def from_geometry(
        self,
        grasp_collisions,  # dict of sets {grasp_index: set(blocking_obj_ids)}
        obj_ids,  # list of object ids in order corresponding to grasps indices
        pickable,  # dict {obj_id: number of grasps} for pickable objects
        vis_pts,  # (N, 3) array of visible points
        all_pts,  # (N, 3) array of visible + occluded points
        vis_mask,  # (N,) array of bitmask indicating which object each point belongs to
        all_mask,  # same as previous but for all points
    ):
        ## phase 1: grasp edges ##
        obj_deps = {}
        for i in range(len(obj_ids)):
            oid = obj_ids[i]
            if oid not in obj_deps:
                obj_deps[oid] = {}
            for did in grasp_collisions[i]:
                if oid == did:
                    continue  # no self edges
                if did not in obj_deps[oid]:
                    obj_deps[oid][did] = {"g": 0, "d": "grasps blocked by"}
                obj_deps[oid][did]["g"] += 1

        ## phase 2: occlusion edges ##
        occ = {}
        vis = {}
        for i in range(32):
            omask = (all_mask & (1 << i)).astype(bool)
            vmask = (vis_mask & (1 << i)).astype(bool)
            # print(i, np.sum(vmask))
            if np.count_nonzero(vmask) == 0:
                continue
            occ[i] = tm.points.PointCloud(all_pts[omask])
            vis[i] = tm.points.PointCloud(vis_pts[vmask])

        for i in vis.keys():
            if i not in obj_deps:
                obj_deps[i] = {}
            for j in vis.keys():
                if i == j:
                    continue  # no self edges
                behind, below = vis_depends(vis[i], occ[i], vis[j], occ[j])
                if below:
                    obj_deps[i][j] = {"d": "below"}
                elif behind and j not in obj_deps[i]:
                    obj_deps[i][j] = {"d": "behind"}

        # debug print
        # print(json.dumps(pickable, indent=4))
        # print(json.dumps(obj_deps, indent=4))

        # create DepGraph instance
        instance = DepGraph(obj_deps, pickable)

        # add occlusion volume info
        for i in occ.keys():
            try:
                instance.nx_graph.nodes[i]["v"] = occ[i].convex_hull.volume
            except:
                instance.nx_graph.nodes[i]["v"] = 0

        return instance

    @classmethod
    def from_edges(
        self,
        grasp_collisions,  # dict of sets {grasp_index: set(blocking_obj_ids)}
        obj_ids,  # list of object ids in order corresponding to grasps indices
        pickable,  # dict {obj_id: number of grasps} for pickable objects
        edge_dict,  # dict {'source': int, 'target': int, 'relation': str}
        all_pts,  # (N, 3) array of visible + occluded points
        all_mask,  # same as previous but for all points
    ):
        ## phase 1: grasp edges ##
        obj_deps = {}
        for i in range(len(obj_ids)):
            oid = obj_ids[i]
            if oid not in obj_deps:
                obj_deps[oid] = {}
            for did in grasp_collisions[i]:
                if oid == did:
                    continue  # no self edges
                if did not in obj_deps[oid]:
                    obj_deps[oid][did] = {"g": 0, "d": "grasps blocked by"}
                obj_deps[oid][did]["g"] += 1

        ## phase 2: occlusion edges ##
        for edge in edge_dict:
            src = edge["source"]
            tgt = edge["target"]
            rel = edge["relation"]
            if src not in obj_deps:
                obj_deps[src] = {}
            if rel == "below":
                obj_deps[src][tgt] = {"d": "below"}
            elif rel == "behind" and tgt not in obj_deps[src]:
                obj_deps[src][tgt] = {"d": "behind"}

        # debug print
        # print(json.dumps(pickable, indent=4))
        # print(json.dumps(obj_deps, indent=4))

        # create DepGraph instance
        instance = DepGraph(obj_deps, pickable)

        # add occlusion volume info
        occ = {}
        for i in range(32):
            omask = (all_mask & (1 << i)).astype(bool)
            if np.count_nonzero(omask) == 0:
                continue
            occ[i] = tm.points.PointCloud(all_pts[omask])
        for i in occ.keys():
            if i in instance.nx_graph.nodes:
                try:
                    instance.nx_graph.nodes[i]["v"] = occ[i].convex_hull.volume
                except:
                    instance.nx_graph.nodes[i]["v"] = 0

        return instance

    def keep_only(self, node_set):
        to_remove = (set(self.dict_graph.keys()) | set(self.nx_graph.nodes)) - set(
            node_set
        )
        for node_id in to_remove:
            if node_id in self.dict_graph:
                del self.dict_graph[node_id]
                for k in self.dict_graph:
                    if node_id in self.dict_graph[k]:
                        del self.dict_graph[k][node_id]
            if node_id in self.nx_graph.nodes:
                self.nx_graph.remove_node(node_id)

    def get_graspable(self):
        graspable = set()
        for n, d in self.nx_graph.nodes(data=True):
            if "g" in d:
                graspable.add(n)
        return graspable

    def normalize_edges(self):
        edges = self.nx_graph.edges
        nodes = self.nx_graph.nodes
        # normalize edge weights to be probabilities

        total = {}
        has_below_edge = set()
        for u, v, d in list(edges(data=True)):
            # remove behind edge if v is 'below' u
            if d["d"] == "behind" and (v, u) in edges:
                if edges[v, u]["d"] == "below":
                    self.nx_graph.remove_edge(u, v)

            # keep track of which nodes have 'below' edges
            if d["d"] == "below":
                has_below_edge.add(u)

            # remove edges from graspable nodes
            if "g" in nodes[u]:
                # unless its a 'below' edge
                if d["d"] != "below":
                    self.nx_graph.remove_edge(u, v)
                    continue

            # only consider edges with grasp blocking dependency
            if d["d"] != "grasps blocked by":
                # equally weight non-grasp edges
                # degree = self.nx_graph.out_degree(u)
                edges[u, v]["w"] = 1
                continue

            # accumulate total # of grasps blocked by other objects
            total[u] = total.get(u, 0) + d["g"]

        for u, v, d in list(edges(data=True)):
            # remove edge if u has a 'below' edge
            if u in has_below_edge:
                if d["d"] != "below":
                    self.nx_graph.remove_edge(u, v)
                continue

            # remove behind edge if u has grasps
            if u in total and d["d"] == "behind":
                self.nx_graph.remove_edge(u, v)
                continue

            # update weights for grasp blocking edges
            if d["d"] == "grasps blocked by":
                edges[u, v]["w"] = total[u] / d["g"]

        for u, v, d in list(edges(data=True)):
            if d["d"] != "grasps blocked by":
                # equally weight non-grasp edges
                degree = self.nx_graph.out_degree(u)
                edges[u, v]["w"] = degree

    def add_hidden_edges(self):
        # make target hidden by weights
        if self.target_id not in self.nx_graph.nodes:
            # remove previous node/edges if any
            if self.hidden_id in self.nx_graph.nodes:
                self.nx_graph.remove_node(self.hidden_id)
            self.nx_graph.add_node(self.hidden_id)

            total = 0
            for v, weight in self.nx_graph.nodes(data="v"):
                # only make edges to source nodes
                # if self.nx_graph.in_degree(v) > 0:
                #     continue
                if weight is None or weight == 0:
                    continue
                total += weight
                self.nx_graph.add_edge(
                    self.hidden_id,
                    v,
                    d="hidden by",
                    w=weight,
                )

            # normalize weights and represent as reciprocal of probability
            for v in self.nx_graph.nodes:
                edge = (self.hidden_id, v)
                if edge not in self.nx_graph.edges:
                    continue
                w = self.nx_graph.edges[edge]["w"]
                self.nx_graph.edges[edge]["w"] = total / w

    def sinks(self):
        """
        return a list of sinks and probabilities for sampling them.
        curv adjusts the shape of the probability distribution of a sinks.
        the 'probability' of a sink means how likely it is to be the optimal choice.
        (this probability is estimated as the sum of probabilities of each simple path to the target, where
        the probability of each path is the product of the belief of each edge)
        """
        sinks = []
        probs = []
        for v in self.nx_graph.nodes:
            # only look at pickable nodes
            if "g" not in self.nx_graph.nodes[v]:
                continue
            # only look at sinks
            if self.nx_graph.out_degree(v) > 0:
                continue
            # return target if its a sink
            if v == self.target_id:
                return [self.target_id], [1]

            sum_of_prod = 0
            target = (
                self.target_id
                if self.target_id in self.nx_graph.nodes
                else self.hidden_id
            )
            for paths in nx.all_simple_edge_paths(self.nx_graph, target, v):
                iprod = 1
                for edge in paths:
                    iprod *= self.nx_graph.edges[edge]["w"]
                sum_of_prod += 1 / iprod

            if sum_of_prod > 0:
                sinks.append(v)
                probs.append(sum_of_prod)

        print("Probs", probs, sum(probs))
        # probs = curv(np.array(probs))  # curve distribution
        # probs = probs / sum(probs)  # re-normalize
        return sinks, probs

    def draw(self, to_show=True, block=True, fname=None, axis=None):
        self.draw_graph(self.nx_graph, to_show, block, fname, axis)

    @staticmethod
    def draw_graph(graph, to_show=True, block=True, fname=None, axis=None):
        nodes = graph.nodes
        edges = graph.edges

        elabel = (
            lambda d: (
                f"{round(100/d.get('w'), 2)}%"
                if "w" in d and d["d"] in ("grasps blocked by", "hidden by")
                else f"{d.get('g', '')}"
            )
            + f" {d['d']}"
        )

        # create new figure to save to file
        if fname is not None:
            f = figure()

        interactive = InteractiveGraph(
            graph,
            node_layout="dot",
            node_color={
                k: from_color_map(k, 32).tolist() if k >= 0 else [1, 0, 0]
                for k in nodes
            },
            node_size={k: 6 if "g" in v else 4 for k, v in nodes(data=True)},
            node_label_fontdict={"size": 14},
            node_labels={
                k: f"{k}" if v is None else f"{k}\n({v})" for k, v in nodes(data="g")
            },
            arrows=True,
            edge_layout="curved",
            edge_labels={(i, j): elabel(d) for i, j, d in edges(data=True)},
            edge_alpha={(i, j): 0.5 if "g" in nodes[i] else 1 for i, j in edges},
            ax=axis,
        )

        if fname is not None:
            f.savefig(fname)
        if to_show:
            show(block=block)

    def __str__(self):
        return self.gen_graphml(self.nx_graph)

    @staticmethod
    def gen_graphml(graph):
        result = ""
        for line in nx.generate_graphml(graph):
            result += line + "\n"
        result = result.replace('attr.name="w"', 'attr.name="belief"')
        result = result.replace('attr.name="d"', 'attr.name="relation"')
        result = result.replace('attr.name="v"', 'attr.name="occlusion volume"')
        result = result.replace(
            'for="node" attr.name="g"',
            'for="node" attr.name="# grasps"',
        )
        result = result.replace(
            'for="edge" attr.name="g"', 'for="edge" attr.name="# blocked grasps"'
        )
        result = result.replace('"-1"', '"0"')
        return result

    def describe(self):
        result = ""
        for v, g in self.nx_graph.nodes(data="g", default=0):
            # result += f"Object {v} has {g} grasps available.\n"
            if g > 0:
                result += f"Object {v} is currently graspable.\n"
        for u, v, d in self.nx_graph.edges(data=True):
            if d["d"] == "grasps blocked by":
                # result += f"{d['g']} grasps of object {u} are blocked by object {v}.\n"
                result += f"object {u} is blocked by object {v}.\n"
            elif d["d"] == "hidden by":
                result += f"Object {u} may be hidden by object {v}.\n"
            elif d["d"] == "behind":
                result += f"Object {u} is behind object {v}.\n"
            elif d["d"] == "below":
                result += f"Object {u} is below object {v}.\n"
            elif d["d"] == "both":
                result += f"Object {u} is behind and below object {v}.\n"
        return result
