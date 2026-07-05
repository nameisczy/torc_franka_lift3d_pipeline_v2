from scipy.spatial import KDTree
import numpy as np
import time

points = np.random.rand(5000, 3)

print(points)

t0 = time.time()

tree = KDTree(points)

unexplored = set(range(len(points)))
explored = set()

frontier = np.array([0])

while frontier is not None and len(frontier) != 0:

    close_pts = tree.query_ball_point(points[frontier], 0.01)
    # print(close_pts)

    close_pts_flattened = [pt for pts in close_pts for pt in pts]

    explored.update(frontier)
    unexplored.difference_update(frontier)
    
    frontier = list(set(close_pts_flattened) - explored)

t1 = time.time()

print(explored)

print("total time:", t1-t0)